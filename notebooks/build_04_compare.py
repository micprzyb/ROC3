"""Builds notebooks/04_comparison_and_selection.ipynb."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

md, code = new_markdown_cell, new_code_cell
cells = []

cells += [md(r"""# 4 — Comparing elasticity models, and choosing one

We now have seven tuned models and no labels. This notebook answers the two questions that
remain, both of which can be settled from observed data:

1. **Is the heterogeneity real?** A model that outputs a different elasticity for every
   product has made a strong claim. There is a way to check it that does not assume the
   answer.
2. **What is it worth?** Turn each model into the pricing decision it implies and estimate
   what that decision earns — always measured against the best *uniform* price move,
   because a model that cannot beat one flat number has not earned its complexity.

The key idea throughout: **you cannot check one row's elasticity, but you can check a
group's**, because a group contains price variation. Everything below is a way of
exploiting that."""),

code("""import sys, pathlib, time, warnings
sys.path.insert(0, str(pathlib.Path.cwd().parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from elasticity_lab.bench import real_frame, synth_frame, frame_summary
from elasticity_lab.models import MODEL_REGISTRY, build_model
from elasticity_lab.splits import RollingOriginSplit
from elasticity_lab.tuning import CausalScorer, tune
from elasticity_lab.evaluation import (predictive_metrics, gates, blp_calibration,
                                       policy_value, oracle_metrics, evaluate_model,
                                       compare_models, elasticity_summary)

pd.set_option("display.width", 210); pd.set_option("display.max_columns", 60)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})
INK, ACCENT, WARN, GOOD = "#3a3a3a", "#1f6feb", "#c0392b", "#1a7f5a"

N_PRODUCTS = 700
syn = synth_frame(N_PRODUCTS, confounding_price=0.6, confounding_demand=0.8)
real = real_frame(N_PRODUCTS)
sp = RollingOriginSplit(n_splits=4, horizon=8, min_train_weeks=30)
syn_scorer, real_scorer = CausalScorer(syn, sp, seed=0), CausalScorer(real, sp, seed=0)
print(frame_summary(real).to_string())"""),

md(r"""## 4.1 GATES — grouped average treatment effects

Sort the validation rows by the model's predicted elasticity, cut into five bins, and then
**re-estimate each bin's elasticity from the data**, orthogonally:

$$\hat\varepsilon_g = -\frac{\langle \tilde t, \tilde y\rangle_g}{\langle \tilde t, \tilde t\rangle_g}$$

where $\tilde y, \tilde t$ are the residualised outcome and price from §3.2. The predicted
column is what the model says; the estimated column is what the data say.

**If the model's ranking is real, the estimated column rises across bins.** If the model's
"heterogeneity" is noise, the estimated column is flat no matter how much the predicted
column varies — and a flat estimated column next to a steep predicted one is the single
most common way an elasticity model is wrong while looking fine.

This is the continuous-treatment analogue of Chernozhukov, Demirer, Duflo & Fernández-Val
(2018). The residuals must come from nuisance models that never saw the validation rows,
or the bins inherit the nuisance model's overfitting; `CausalScorer` guarantees that."""),

code("""fold = len(syn_scorer.folds_) - 1
tr, va = syn_scorer.folds_[fold]
res = syn_scorer.residuals(fold)

het = {"r_learner": build_model("r_learner", n_folds=3),
       "lgbm_demand": build_model("lgbm_demand"),
       "shrunk_per_product": build_model("shrunk_per_product")}
tables = {}
for name, m in het.items():
    m.fit(syn.iloc[tr])
    e = np.asarray(m.elasticity(syn.iloc[va]), float)
    g = gates(res["ry"], res["rt"], e, n_groups=5)
    # the synthetic panel lets us add the column the real panel can never have
    edges = np.quantile(e, np.linspace(0, 1, 6)); edges[0], edges[-1] = -np.inf, np.inf
    b = np.clip(np.searchsorted(edges, e, side="right") - 1, 0, 4)
    truth = syn.iloc[va].true_elasticity.to_numpy(float)
    g["eps_TRUE"] = [truth[b == k].mean() for k in range(5)]
    tables[name] = g
    print(f"\\n=== {name} ===  monotone: {g.attrs['monotone']}   "
          f"spread {g.attrs['spread']:+.3f} (t = "
          f"{g.attrs['spread']/g.attrs['spread_se']:+.2f})")
    print(g[["group", "n", "eps_predicted", "eps_estimated", "se", "eps_TRUE"]]
          .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))"""),

md("""The `eps_TRUE` column exists only because this is the synthetic panel — it is the
audit of the audit. Where `eps_estimated` tracks `eps_TRUE`, the GATES procedure is
working, and that is the evidence that lets us rely on it in §4.6 where `eps_TRUE` does
not exist."""),

code("""fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.2), sharey=True)
for a, (name, g) in zip(ax, tables.items()):
    a.errorbar(g.group, g.eps_estimated, yerr=1.96*g.se, fmt="o-", color=ACCENT,
               lw=2, ms=6, capsize=3, label="estimated from data")
    a.plot(g.group, g.eps_predicted, "s--", color=WARN, lw=1.6, ms=5,
           label="predicted by model")
    a.plot(g.group, g.eps_TRUE, "^:", color=GOOD, lw=1.6, ms=6, label="TRUE (oracle)")
    a.set_xlabel("quintile of predicted elasticity")
    a.set_title(name, loc="left")
ax[0].set_ylabel("elasticity"); ax[0].legend(frameon=False, fontsize=8)
fig.suptitle("GATES: does the ranking survive contact with the data?", x=.01, ha="left")
fig.tight_layout()"""),

md(r"""## 4.2 BLP — is the *scale* of the heterogeneity right?

GATES checks the ordering. The Best-Linear-Predictor test checks the magnitude. Regress

$$\tilde y \;=\; \beta_0\,\tilde t \;+\; \beta_1\,\tilde t\,\bigl(\hat\varepsilon(x) - \overline{\hat\varepsilon}\bigr) \;+\; \text{error}$$

* $-\beta_0$ estimates the average elasticity;
* $\beta_1 \approx -1$ means the predicted spread is correctly scaled (negative because
  $\theta = -\varepsilon$);
* $\beta_1 \approx 0$ means the predictions carry no usable heterogeneity;
* $|\beta_1| < 1$ means the model **over-states** how much elasticities differ — the usual
  outcome for a flexible learner, which reads noise as signal.

The $t$-statistic on $\beta_1$ is a formal test that the model detects any real
heterogeneity at all."""),

code("""rows = []
for name, m in het.items():
    e = np.asarray(m.elasticity(syn.iloc[va]), float)
    b = blp_calibration(res["ry"], res["rt"], e)
    o = oracle_metrics(e, syn.iloc[va].true_elasticity.to_numpy(float))
    rows.append({"model": name, **b, "oracle_corr": o["oracle_corr"],
                 "oracle_rmse": o["oracle_rmse"]})
blp = pd.DataFrame(rows)
blp[["model", "blp_avg_elasticity", "beta1", "beta1_se", "beta1_t", "implied_spread",
     "heterogeneity_detected", "oracle_corr", "oracle_rmse"]] \\
   .style.format({c: "{:+.4f}" for c in
                  ["blp_avg_elasticity", "beta1", "beta1_se", "beta1_t", "implied_spread",
                   "oracle_corr",
                   "oracle_rmse"]}).hide(axis="index")"""),

md("""On the synthetic panel we can check the check: `beta1_t` is computed from observed data
only, `oracle_corr` from the truth. Where they agree, the BLP test is doing its job — which
is the evidence that lets us trust it on the real panel, where `oracle_corr` does not
exist."""),

md(r"""## 4.3 Decision value — what is the model worth?

Log revenue is $\log p + \log q$, so nudging log price by $d$ changes log revenue by
$d\,(1 - \varepsilon)$: raise the price where demand is **inelastic** ($\varepsilon < 1$),
cut it where it is **elastic**. The model's implied policy is therefore

$$d(x) = +\delta \ \text{ if } \hat\varepsilon(x) < 1, \qquad -\delta \ \text{ otherwise.}$$

Evaluating it needs the *true* elasticity, which we do not have — so it is estimated
group-wise, exactly as in GATES: bin by $\hat\varepsilon$, identify each bin's elasticity
from its own price variation, and average.

**The reference point is the whole point.** `value_best_uniform` is the best single
$\pm\delta$ applied to everyone, including doing nothing. `lift_over_uniform` is the
difference — and it is routinely zero or negative. Reporting it is what stops a model from
being adopted on the strength of a good-looking score."""),

code("""rows = []
for name, m in het.items():
    e = np.asarray(m.elasticity(syn.iloc[va]), float)
    rows.append({"model": name, **policy_value(res["ry"], res["rt"], e, delta=0.05)})
# a constant-elasticity model, for contrast: it prices everything the same way
m_const = build_model("dml_partialling", n_folds=3).fit(syn.iloc[tr])
e_const = np.asarray(m_const.elasticity(syn.iloc[va]), float)
rows.append({"model": "dml_partialling (constant)",
             **policy_value(res["ry"], res["rt"], e_const, delta=0.05)})
pv = pd.DataFrame(rows)
pv.style.format({c: "{:+.5f}" for c in pv.columns if c != "model"}).hide(axis="index")"""),

code("""fig, ax = plt.subplots(figsize=(7.4, 3.2))
xs = np.arange(len(pv)); w = .36
ax.bar(xs - w/2, pv.value_model, w, label="model's targeted policy", color=ACCENT)
ax.bar(xs + w/2, pv.value_best_uniform, w, label="best uniform move", color=INK, alpha=.6)
ax.axhline(0, color=INK, lw=1)
ax.set_xticks(xs); ax.set_xticklabels(pv.model, rotation=18, ha="right")
ax.set_ylabel("estimated change in mean log revenue\\nfrom a 5% price move")
ax.set_title("Targeting only pays if it beats the flat move", loc="left")
ax.legend(frameon=False)
fig.tight_layout()"""),

md("""## 4.4 Everything at once, on the synthetic panel

`evaluate_model` runs the full battery on the last fold: predictive metrics, R-loss and its
two normalisations, BLP, GATES, policy value, and — because this is the synthetic panel —
the oracle comparison the others are approximating."""),

code("""models = {n: build_model(n) for n in MODEL_REGISTRY}
allm = compare_models(models, syn, syn_scorer, fold=-1, truth_col="true_elasticity")
cols = ["name", "avg_elasticity", "blp_avg_elasticity", "eps_sd", "rmse",
        "rloss_skill", "rloss_vs_constant",
        "beta1", "beta1_t", "implied_spread", "gates_monotone", "gates_spread",
        "lift_over_uniform",
        "oracle_rmse", "oracle_bias", "oracle_corr"]
allm[cols].sort_values("oracle_rmse").style.format(
    {c: "{:+.4f}" for c in cols if c not in ("name", "gates_monotone")}
).hide(axis="index")"""),

md(r"""### Which observable metric would have picked the right model?

This is the payoff of having a synthetic panel at all. Rank the models by each *observable*
metric and see which ranking matches the ranking by oracle error. Whichever observable
metric wins here is the one to trust on the real panel."""),

code("""obs = ["rmse", "rloss_skill", "rloss_vs_constant", "beta1_t", "lift_over_uniform"]
sign = {"rmse": +1, "rloss_skill": -1, "rloss_vs_constant": -1, "beta1_t": +1,
        "lift_over_uniform": -1}   # +1 = lower is better, matching oracle_rmse
d = allm.dropna(subset=["oracle_rmse"] + obs)
sel = pd.DataFrame([{"observable metric": c,
                     "rho with oracle_rmse":
                         sign[c] * d[[c, "oracle_rmse"]].corr("spearman").iloc[0, 1]}
                    for c in obs]).sort_values("rho with oracle_rmse", ascending=False)
sel.style.format({"rho with oracle_rmse": "{:+.3f}"}).hide(axis="index")"""),

md("""Positive means the metric ranks models the way the truth does. Predictive RMSE sitting
low or negative in this table is the same finding as notebook 3, now at the level of *model
choice* rather than hyperparameter choice."""),

md("""## 4.5 A specification check: does the winner depend on the DGP?

Every conclusion so far comes from one synthetic world. A benchmark that only works in the
world it was designed for is worthless, so we vary the world:

* **exogenous prices** — no confounding at all, where the naive estimator should be fine;
* **heavy confounding** — where orthogonalisation should pay most;
* **Poisson outcome** — counts rather than lognormal, which should suit the Poisson GLM and
  penalise the log-scale models.

If the ranking is stable across these, it is a property of the methods. If it flips, that
is worth knowing too — and it is the honest version of "no method is universally best"
(Curth & van der Schaar, 2021)."""),

code("""from elasticity_lab.simulate import SyntheticConfig

worlds = {
    "exogenous price":  SyntheticConfig(confounding_price=0.0, confounding_demand=0.0),
    "mild confounding": SyntheticConfig(confounding_price=0.35, confounding_demand=0.8),
    "heavy confounding": SyntheticConfig(confounding_price=1.0, confounding_demand=1.2),
    "poisson outcome":  SyntheticConfig(confounding_price=0.35, confounding_demand=0.8,
                                        outcome="poisson"),
}
rows = []
for wname, cfg in worlds.items():
    w = synth_frame(400, config=cfg)
    for mname in MODEL_REGISTRY:
        m = build_model(mname).fit(w)
        e = np.asarray(m.elasticity(w), float)
        t = w.true_elasticity.to_numpy(float)
        rows.append({"world": wname, "model": mname,
                     "bias": float(e.mean() - t.mean()),
                     "oracle_rmse": float(np.sqrt(np.mean((e - t) ** 2)))})
robust = pd.DataFrame(rows)
robust.pivot(index="model", columns="world", values="bias") \\
      .style.format("{:+.3f}").background_gradient(cmap="RdYlGn_r", vmin=-1.5, vmax=1.5)"""),

code("""fig, ax = plt.subplots(figsize=(8.4, 3.4))
piv = robust.pivot(index="model", columns="world", values="oracle_rmse")
piv = piv.loc[piv.mean(axis=1).sort_values().index]
xs = np.arange(len(piv))
for i, w in enumerate(piv.columns):
    ax.bar(xs + (i - 1.5) * 0.2, piv[w], 0.2, label=w)
ax.set_xticks(xs); ax.set_xticklabels(piv.index, rotation=20, ha="right")
ax.set_ylabel("RMSE vs true elasticity"); ax.legend(frameon=False, fontsize=8, ncol=2)
ax.set_title("Is the ranking a property of the method or of the world?", loc="left")
fig.tight_layout()"""),

md("""## 4.6 The real panel: tune, evaluate, choose

Now the same battery on data with no answer key, using the R-loss to tune (the conclusion
of notebook 3) and the grouped diagnostics to judge."""),

code("""N_TRIALS = 12
tuned, chosen = {}, {}
for name in MODEL_REGISTRY:
    t0 = time.time()
    r = tune(name, real, objective="rloss", n_trials=N_TRIALS, scorer=real_scorer, seed=0)
    tuned[name] = r["best_params"]
    chosen[name] = build_model(name, **r["best_params"])
    print(f"{name:20s} rloss={r['best_metrics']['rloss']:.4f}  ({time.time()-t0:.0f}s)")

final = compare_models(chosen, real, real_scorer, fold=-1)
fcols = ["name", "avg_elasticity", "blp_avg_elasticity", "eps_sd", "rmse", "r2",
         "rloss_skill",
         "rloss_vs_constant", "beta1", "beta1_t", "implied_spread",
         "heterogeneity_detected",
         "gates_monotone", "gates_spread", "value_model", "value_best_uniform",
         "lift_over_uniform"]
final[fcols].sort_values("rloss_vs_constant", ascending=False).style.format(
    {c: "{:+.4f}" for c in fcols
     if c not in ("name", "heterogeneity_detected", "gates_monotone")}
).hide(axis="index")"""),

code("""best_het = final.sort_values("rloss_vs_constant", ascending=False).iloc[0]["name"]
m = chosen[best_het]
r = evaluate_model(m, real, real_scorer, fold=-1)
g = r["_gates_table"]
print(f"best by rloss_vs_constant: {best_het}")
print(g[["group", "n", "eps_predicted", "eps_estimated", "se"]]
      .to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

fig, ax = plt.subplots(1, 2, figsize=(9.4, 3.2))
ax[0].errorbar(g.group, g.eps_estimated, yerr=1.96*g.se, fmt="o-", color=ACCENT,
               lw=2, ms=6, capsize=3, label="estimated from data")
ax[0].plot(g.group, g.eps_predicted, "s--", color=WARN, lw=1.6, ms=5, label="predicted")
ax[0].axhline(1.0, color=INK, ls=":", lw=1.2)
ax[0].set_xlabel("quintile of predicted elasticity"); ax[0].set_ylabel("elasticity")
ax[0].set_title(f"GATES on the real panel — {best_het}", loc="left")
ax[0].legend(frameon=False, fontsize=8)
ax[1].hist(np.clip(r["_elasticity"], -2, 6), bins=60, color=ACCENT, alpha=.85)
ax[1].axvline(1.0, color=WARN, lw=2, label="unit elasticity")
ax[1].axvline(0.0, color=INK, lw=1.2, ls=":", label="0 = demand rises with price")
ax[1].set_xlabel("predicted elasticity"); ax[1].legend(frameon=False, fontsize=8)
ax[1].set_title("Distribution of predictions", loc="left")
fig.tight_layout()"""),

md("""The dotted line at 1.0 in the left panel is the decision boundary: products estimated
below it are inelastic, and raising their price raises revenue. Where the *estimated*
(blue) series crosses it is the actionable finding; where only the *predicted* (red) series
crosses it, the model is claiming something the data do not support."""),

md("""## 4.7 A selection rule

Putting the pieces together, in the order they should be applied:

1. **Tune on the R-loss**, never on predictive RMSE (notebook 3).
2. **Check `rloss_vs_constant` on held-out data.** ≤ 0 means the heterogeneity does not
   beat one flat number — use the constant-elasticity model, which will be better estimated
   and easier to defend.
3. **Check the BLP `beta1_t`.** Not significantly negative means no detectable
   heterogeneity, whatever the spread of the predictions looks like.
4. **Check GATES monotonicity.** A non-monotone estimated column means the ranking is
   unreliable even if the average is fine.
5. **Check `lift_over_uniform`.** If targeting does not beat the best flat move, the
   complexity is not paying for itself.
6. **Report nested-CV numbers**, and sanity-check `pct_negative` — a model producing many
   upward-sloping demand curves is telling you its predictions are noise.

Steps 2–5 are all computable without ever knowing a single true elasticity. That is the
point of the whole exercise."""),

code("""verdict = final.copy()
verdict["passes_2_beats_constant"] = verdict.rloss_vs_constant > 0
verdict["passes_3_blp"] = verdict.beta1_t < -1.96
verdict["passes_4_gates"] = verdict.gates_monotone.fillna(False)
verdict["passes_5_policy"] = verdict.lift_over_uniform > 0
pf = [c for c in verdict.columns if c.startswith("passes_")]
verdict["checks_passed"] = verdict[pf].sum(axis=1)
verdict[["name"] + pf + ["checks_passed", "avg_elasticity"]] \\
    .sort_values("checks_passed", ascending=False).style.hide(axis="index")"""),

md("""A constant-elasticity model failing checks 3–5 is **not** a mark against it: it never
claimed heterogeneity, so tests of heterogeneity do not apply. The checks decide between
models that *do* make the claim, and tell you when to fall back to one that does not.

## 4.8 Summary

* Heterogeneity is checkable without labels, at the level of **groups**: GATES for the
  ranking, BLP for the scale, policy value for the money.
* The reference for every heterogeneous claim is the **best constant** — fitted on the same
  held-out data, so beating it means something.
* Which model wins depends on the world (§4.5). Report the range, not one number.
* On the real panel, the honest deliverable is: an average elasticity with a confidence
  interval, a statement of whether per-product variation is detectable, and — if it is —
  what acting on it is estimated to be worth.""")]

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
out = "notebooks/04_comparison_and_selection.ipynb"
nbf.write(nb, out)
print("wrote", out, f"({len(cells)} cells)")
