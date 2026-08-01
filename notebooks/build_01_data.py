"""Builds notebooks/01_data_and_the_identification_problem.ipynb."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell

md, code = new_markdown_cell, new_code_cell
cells = []

cells += [md(r"""# 1 — The data, and why elasticity is hard

**What this notebook is for.** Before fitting anything we have to establish three things,
because each one silently ruins an elasticity study if you skip it:

1. **Is there price variation at all?**  An elasticity is a *derivative with respect to
   price*.  If a product's price never moved, no method — however clever — can recover its
   elasticity, and any number a model reports for it is an artefact of the prior.
2. **Is the price we measured actually a price?**  The obvious construction,
   `revenue / quantity`, is contaminated in a way that manufactures the very relationship
   we are trying to measure.  We quantify the damage.
3. **Why can't we just regress log quantity on log price?**  We do it, get a number, and
   then decompose exactly where that number comes from — including a channel that has
   nothing to do with confounding and that most write-ups never mention.

**The dataset.** UCI *Online Retail II* (Chen, 2019): 1,067,371 transaction lines from a
UK online giftware wholesaler, December 2009 to December 2011. Public, no registration,
and unusually rich in price variation because the seller repriced constantly.

<https://archive.ics.uci.edu/dataset/502/online+retail+ii>"""),

md("""## 1.0 Setup

Everything below is built from the `elasticity_lab` package that sits beside this
notebook, so the notebook stays readable and the logic stays testable."""),

code("""import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from elasticity_lab.data import (get_panel, load_transactions, clean_transactions,
                                 panel_fingerprint, division_bias_diagnostic,
                                 PANEL_SCHEMA)
from elasticity_lab.features import add_features, CONTROL_FEATURES, feature_glossary
from elasticity_lab.splits import (RollingOriginSplit, PurgedRollingSplit,
                                   GroupedProductSplit, NaiveKFold, split_report)

pd.set_option("display.width", 160)
pd.set_option("display.max_columns", 40)
plt.rcParams.update({"figure.dpi": 110, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False,
                     "axes.spines.right": False})
INK, ACCENT, WARN = "#3a3a3a", "#1f6feb", "#c0392b\""""),

md("""## 1.1 From transactions to a product-week panel

Raw lines are cleaned (returns, credit notes, non-product stock codes such as `POST` and
`BANK CHARGES`, non-positive prices) and aggregated to product × ISO-week. Products active
in fewer than 20 distinct weeks are dropped: they cannot support a within-product
estimate, and keeping them would let the pooled models average over rows that carry no
information."""),

code("""panel = get_panel(min_weeks=20)
print(f"panel: {panel.shape[0]:,} product-weeks | "
      f"{panel.stock_code.nunique():,} products | {panel.week.nunique()} weeks")
print(f"fingerprint: {panel_fingerprint(panel)}   "
      "(assert this if you re-run and want to be sure it is the same data)")
pd.DataFrame({"column": PANEL_SCHEMA.keys(), "meaning": PANEL_SCHEMA.values()})"""),

code("""panel.head(8)"""),

md("""## 1.2 Is there enough price variation?

The quantity that matters is the **within-product standard deviation of log price**. A
product with `sd = 0` contributes nothing. A rough rule: the standard error of a
product's own elasticity scales like `1 / (sd_log_price · sqrt(n_weeks))`, so both axes
of the plot below are directly proportional to how much that product can tell us."""),

code("""wp = (panel.assign(lp=np.log(panel.price_modal))
            .groupby("stock_code")
            .agg(sd_log_price=("lp", "std"), n_weeks=("lp", "size")))

print(wp.sd_log_price.describe(percentiles=[.1, .25, .5, .75, .9]).round(4).to_string())
print(f"\\nproducts with sd(log price) > 0.10 : "
      f"{(wp.sd_log_price > 0.10).mean():.1%}")
print(f"products with sd(log price) < 0.02 : "
      f"{(wp.sd_log_price < 0.02).mean():.1%}  <- these are uninformative")

fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.2))
ax[0].hist(wp.sd_log_price.clip(0, 0.8), bins=60, color=ACCENT, alpha=.85)
ax[0].axvline(0.10, color=WARN, lw=2, ls="--")
ax[0].set_xlabel("within-product sd of log price"); ax[0].set_ylabel("products")
ax[0].set_title("Price variation is plentiful", loc="left")
ax[1].scatter(wp.n_weeks, wp.sd_log_price.clip(0, 0.8), s=4, alpha=.15, color=INK)
ax[1].set_xlabel("weeks active"); ax[1].set_ylabel("sd of log price")
ax[1].set_title("Information per product = both axes", loc="left")
fig.tight_layout()"""),

md(r"""## 1.3 The measurement trap: never use revenue ÷ quantity as the price

It is the natural choice and it is wrong. In this data larger order lines get a discount,
so within a single product-week

$$\frac{\partial \log(\text{line price})}{\partial \log(\text{line quantity})} < 0 .$$

A week that happens to contain one big order therefore shows a *low* average unit value and
a *high* quantity — a downward-sloping "demand curve" produced entirely by arithmetic. This
is **division bias** (Borjas 1980; Deaton 1988 on unit values). It does not cancel out with
more data; it is a bias, not noise.

The fix is a price measure that is not a function of the week's quantity. We use the
**modal posted line price**."""),

code("""tx = clean_transactions(load_transactions())
diag = division_bias_diagnostic(tx)
print(f"within product-week:  d log(price) / d log(quantity) = {diag['slope']:+.4f}"
      f"   (t = {diag['t_stat']:,.0f},  n = {diag['n']:,})")
print("negative slope => bigger orders are discounted => unit value moves WITH quantity")"""),

md("""### How much does it change the answer?

We fit the same two-way fixed-effects specification twice, changing only the price column."""),

code("""def fe_elasticity(price_col):
    d = add_features(panel, price_col=price_col)
    y, x = d.log_qty.to_numpy(float), d.log_price.to_numpy(float)
    f = pd.DataFrame({"y": y, "x": x, "p": d.stock_code.values, "w": d.week.values})
    for _ in range(12):                      # iterative two-way demeaning
        for col in ("y", "x"):
            f[col] -= f.groupby("p", observed=True)[col].transform("mean")
            f[col] -= f.groupby("w", observed=True)[col].transform("mean")
    m = sm.OLS(f.y.values, f.x.values[:, None]).fit()
    return -float(m.params[0]), float(m.bse[0])

rows = []
for col in ("price_modal", "price_median", "price_unitvalue"):
    e, se = fe_elasticity(col)
    rows.append({"price measure": col, "FE elasticity": round(e, 4), "se": round(se, 4)})
t = pd.DataFrame(rows)
t["overstatement vs modal"] = (t["FE elasticity"] / t["FE elasticity"].iloc[0] - 1)
t.style.format({"overstatement vs modal": "{:+.1%}"})"""),

md("""The unit-value column reports demand as substantially more price-sensitive than it is.
A pricing team acting on it would cut prices it should have held. Note that the *standard
errors are tiny* in both cases — this is a bias no amount of data fixes, and no confidence
interval warns you about."""),

md(r"""## 1.4 Why the obvious regression fails

Now the central problem. Fit the textbook specification

$$\log q_{it} \;=\; a \;-\; \varepsilon \,\log p_{it} \;+\; u_{it}$$

and read off $\varepsilon$."""),

code("""df = add_features(panel)                       # modal price, leakage-safe features
print(f"modelling frame: {df.shape[0]:,} rows x {df.shape[1]} cols, "
      f"{df.isna().sum().sum()} NaNs")

X = sm.add_constant(df.log_price.to_numpy(float))
naive = sm.OLS(df.log_qty.to_numpy(float), X).fit()
print(f"\\nnaive pooled elasticity : {-naive.params[1]:+.4f} "
      f"(se {naive.bse[1]:.4f})")
fe, fe_se = fe_elasticity("price_modal")
print(f"two-way fixed effects    : {fe:+.4f} (se {fe_se:.4f})")"""),

md("""Two defensible-looking specifications, two different answers. Which is right?

**On real data this question has no answer**, because the true elasticity is not recorded
anywhere. That is not a gap in the dataset; it is the structure of the problem. Every
customer is observed at one price, so the counterfactual quantity at another price is
never in the data.

So we build a world where we *do* know the answer."""),

md(r"""## 1.5 A semi-synthetic panel where the truth is known

We keep the real prices, the real calendar and the real covariates, and replace only the
quantities with a simulated demand system:

$$\log q_{it} \;=\; 3.0 \;+\; \alpha_i \;+\; \gamma_t \;+\; g(X_{it}) \;-\; \varepsilon_i \log p_{it} \;+\; \lambda\, u_{it} \;+\; \text{noise}$$

with the price itself nudged by the same hidden factor,
$\log p_{it} = \log p^{\text{real}}_{it} + \theta\, u_{it} + \text{noise}$.

* $\varepsilon_i$ — **the answer**: a per-product elasticity, drawn so that it correlates
  with observable product characteristics (expensive, slow-moving products are less elastic).
* $u_{it} = h(X_{it})$ — the confounder. It moves price *and* quantity, so a naive
  regression attributes $\theta\lambda$ of its effect to price.
* Crucially $u$ is a **nonlinear function of observed controls**. It is therefore
  removable in principle but not by a linear model — which is exactly what makes the
  hyperparameter search in notebook 3 a real test rather than a formality."""),

code("""from elasticity_lab.simulate import (SyntheticConfig, make_synthetic_panel,
                                      naive_bias_decomposition)

cfg = SyntheticConfig(confounding_price=0.6, confounding_demand=0.8, seed=0)
# make_synthetic_panel returns a ready feature frame: it engineers the controls from the
# real panel, overwrites price and quantity, then repairs the price-derived columns.
syn = make_synthetic_panel(panel, cfg)

print(f"true mean elasticity : {syn.true_elasticity.mean():+.4f}")
print(f"true sd  elasticity  : {syn.true_elasticity.std():+.4f}")
print(f"true range           : [{syn.true_elasticity.min():.2f}, "
      f"{syn.true_elasticity.max():.2f}]")"""),

md("""### Where the naive estimate's error actually comes from

This decomposition is an **exact identity**, not an approximation — the channels sum to the
naive estimate to within floating point. Each row is the amount of the naive slope that is
attributable to one component of the data-generating process."""),

code("""dec = naive_bias_decomposition(syn)
print("identity holds exactly:", dec.attrs["exact"])
dec.style.format({"contribution": "{:+.4f}"}).hide(axis="index")"""),

md("""Read the table from the top. The true mean elasticity is what we want; everything between
it and the last two rows is error. The confounder is the largest channel here, which is
what you would expect — we set it that way. The row worth stopping on is the last one.

*Elasticity heterogeneity* is a source of error that has **nothing to do with
confounding**. A pooled regression does not estimate the average elasticity; it estimates

$$\\frac{\\mathrm{Cov}(\\log p,\\; \\varepsilon \\log p)}{\\mathrm{Var}(\\log p)},$$

a **variance-weighted** average in which products whose price moved a lot dominate. If
those products differ systematically in elasticity from the rest — and here they do, by
construction and in reality — the pooled number is wrong even with zero confounding.

The next cell settles it: set the confounding to exactly zero and re-run. If the
heterogeneity channel were a by-product of confounding it would vanish with it."""),

code("""clean_cfg = SyntheticConfig(confounding_price=0.0, confounding_demand=0.0, seed=0)
clean = make_synthetic_panel(panel, clean_cfg)
d2 = naive_bias_decomposition(clean)
print("identity holds exactly:", d2.attrs["exact"])
d2.rename(columns={"contribution": "contribution (no confounding)"}) \\
  .style.format({"contribution (no confounding)": "{:+.4f}"}).hide(axis="index")"""),

md("""The confounder channel is now exactly `-0.0000`, and the heterogeneity channel is
**larger than before**, not smaller. It is a separate defect, and controlling for
confounders — however well — does not touch it.

**The lesson for model design.** Two independent things must be fixed, and they need
different tools:

| problem | what fixes it |
|---|---|
| confounding — price correlates with unobserved demand | controls, fixed effects, orthogonalisation (Double ML) |
| heterogeneity weighting — the average is variance-weighted | estimating `eps(x)` per unit and averaging deliberately |

A model that only fixes the first still reports the wrong average. This is why the zoo in
notebook 2 contains both constant-effect and heterogeneous-effect estimators."""),

md("""## 1.6 Features, and the two rules that keep them honest

Every engineered feature obeys two rules. Both exist to protect the estimate, not the
R-squared.

**Rule 1 — nothing looks forward.** Every lag and rolling statistic is computed with
`shift(1)` *inside the product*.
**Rule 2 — nothing is a function of the current week's quantity.** Same trap as the unit
value: a feature built from this week's quantity leaks the outcome into the predictors.

The columns split into **controls** (legitimate confounder proxies, fed to nuisance models)
and **price features** (functions of the treatment, which must never enter a nuisance
model that is supposed to be price-free)."""),

code("""pd.DataFrame({"feature": feature_glossary.keys(),
              "meaning": feature_glossary.values()})"""),

code("""print(f"{len(CONTROL_FEATURES)} control features:")
print("  " + ", ".join(CONTROL_FEATURES))"""),

md("""## 1.7 Cross-validation: the default is wrong here, twice over

`KFold(shuffle=True)` leaks in two independent ways on a panel:

* **through time** — a random fold puts week 40 in training and week 39 in validation, so
  the model interpolates a series it has already seen either side of;
* **through product identity** — the same SKU appears in both halves, so a flexible model
  can memorise its level.

Neither leak is visible in the score; both make it better. We measure them."""),

code("""designs = {
    "NaiveKFold (wrong)":      NaiveKFold(n_splits=5),
    "RollingOrigin":           RollingOriginSplit(n_splits=5, horizon=8),
    "PurgedRolling (gap=12)":  PurgedRollingSplit(n_splits=5, horizon=8, gap=12),
    "GroupedProduct":          GroupedProductSplit(n_splits=5),
}
split_report(df, designs)"""),

md("""`weeks_shared_train_valid` and `pct_valid_products_also_in_train` are the two leak
channels. The naive design shares both; rolling origin closes the time channel; grouping
closes the product channel; nothing closes both without discarding most of the data, so we
pick per question:

* **rolling origin** is the default for hyperparameter search — it matches deployment,
  where you always predict forward;
* **purged rolling** adds a 12-week embargo so a validation row's rolling window cannot
  overlap training;
* **grouped by product** answers the different question "does this transfer to a SKU I have
  never seen?"."""),

md("""### What the leak is worth, in score points

Concretely: fit the same gradient-boosted model under the naive design and under rolling
origin, and compare.

**A performance trap, since we are about to hit it.** `n_jobs=-1` is the wrong default for
LightGBM on data this size. On the 24-core host used here a fit that takes **0.09 s** at
`n_jobs=8` takes **30.7 s** at `n_jobs=-1` — a 340x slowdown, because the trees are small
and the threads spend their time synchronising rather than working. The first version of
this cell used `-1` and did not finish within an hour. `elasticity_lab.models.N_THREADS`
fixes the count at `min(8, cores/2)`; override with `ELASTICITY_LAB_THREADS`."""),

code("""import lightgbm as lgb
from elasticity_lab.features import PRICE_FEATURES, OUTCOME
from elasticity_lab.models import N_THREADS      # NOT n_jobs=-1 -- see the note below

cols = list(CONTROL_FEATURES) + list(PRICE_FEATURES)
def cv_rmse(splitter):
    out = []
    for tr, va in splitter.split(df):
        m = lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=63,
                              verbosity=-1, n_jobs=N_THREADS, random_state=0)
        m.fit(df.iloc[tr][cols], df.iloc[tr][OUTCOME])
        p = m.predict(df.iloc[va][cols])
        out.append(float(np.sqrt(np.mean((p - df.iloc[va][OUTCOME]) ** 2))))
    return float(np.mean(out))

r_naive = cv_rmse(NaiveKFold(n_splits=5))
r_roll  = cv_rmse(RollingOriginSplit(n_splits=5, horizon=8))
print(f"RMSE, naive shuffled K-fold : {r_naive:.4f}   <- flattering, and false")
print(f"RMSE, rolling origin        : {r_roll:.4f}")
print(f"the leak is worth           : {100*(r_roll-r_naive)/r_naive:+.1f}% of RMSE")"""),

md("""## 1.8 What we established

1. The panel has real, plentiful price variation — **81%** of products move log price by
   more than 0.10, and only 10% are effectively fixed-price — so the question is answerable.
2. `revenue / quantity` inflates the fixed-effects elasticity from **1.571 to 2.004**
   (+27.6%). Use a posted price.
3. The naive pooled regression is wrong for **two separate reasons**. Confounding is the
   one everybody controls for; heterogeneity weighting is the one that **survives at
   exactly zero confounding**, so no amount of controlling fixes it.
4. Shuffled K-fold understates RMSE by **half** here (0.78 vs 1.56 — the honest number is
   100% larger). Every score from here on uses rolling origin or a purged variant.
5. Because the truth is unknowable on the real panel, methods get **graded on the
   semi-synthetic twin** and only then applied to the real one.

**Next:** `02_model_zoo.ipynb` — seven estimators, what each assumes, and what each
recovers.""")]

nb = new_notebook(cells=cells, metadata={
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"}})
out = "notebooks/01_data_and_the_identification_problem.ipynb"
nbf.write(nb, out)
print("wrote", out, f"({len(cells)} cells)")
