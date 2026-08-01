"""
Re-analyse the trial histories written by 11_hpo_experiment.py.

Separated from 11_ because 11_ costs ~2 hours and this costs a second: every trial of every
search recorded *all* metrics, so the interesting questions can be asked afterwards without
refitting anything.  That design is the reason this file can exist.
"""
import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path("experiments/out")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
fmt = lambda v: f"{v:+.4f}"                                                 # noqa: E731

hist = pd.read_parquet(OUT / "11_trial_histories.parquet")
comp = pd.read_csv(OUT / "11_objective_comparison.csv")
hist = hist[hist.get("failed", 0) == 0].dropna(subset=["oracle", "rloss", "rmse"])
print(f"{len(hist):,} completed trials across {hist.model.nunique()} models "
      f"and {hist.tuned_on.nunique()} objectives\n")

#: Models whose elasticity comes from an orthogonal (Neyman-orthogonal) score.  The split
#: matters: see the conclusion below.
ORTHOGONAL = {"dml_partialling", "r_learner"}

# ===================================================================== 1. does it matter?
print("=" * 100)
print("1.  DOES THE TUNING OBJECTIVE CHANGE THE ANSWER?")
print("=" * 100)
piv = comp.pivot(index="model", columns="tuned_on", values=["oracle", "oracle_bias", "rmse"])
piv.columns = [f"{a}[{b}]" for a, b in piv.columns]
piv["oracle_gain"] = piv["oracle[rmse]"] - piv["oracle[rloss]"]
piv["rmse_cost"] = piv["rmse[rloss]"] - piv["rmse[rmse]"]
piv["rel_gain_%"] = 100 * piv["oracle_gain"] / piv["oracle[rmse]"]
print(piv.sort_values("oracle_gain", ascending=False)
      .to_string(float_format=fmt))
print(f"\n  better with R-loss : {int((piv.oracle_gain > 0).sum())} / {len(piv)} models")
print(f"  median gain        : {piv.oracle_gain.median():+.4f} "
      f"({piv['rel_gain_%'].median():+.1f}% of the oracle error)")
print(f"  median RMSE given up: {piv.rmse_cost.median():+.4f}")
print(f"  largest gain       : {piv.oracle_gain.idxmax()} "
      f"{piv.oracle_gain.max():+.4f} ({piv['rel_gain_%'].max():+.0f}%)")
print(f"  only loss          : {piv.oracle_gain.idxmin()} {piv.oracle_gain.min():+.4f}")

# ============================================================ 2. the criterion that counts
print("\n" + "=" * 100)
print("2.  DOES THE OBJECTIVE *RANK* CONFIGURATIONS THE WAY THE TRUTH DOES?")
print("=" * 100)
print("This, not the level, is what a hyperparameter search needs.  Spearman rho between")
print("each observable objective and the oracle error, over all trials of that model.\n")
rows = []
for name, g in hist.groupby("model"):
    if len(g) < 8 or g.oracle.std() < 1e-9:
        continue
    rows.append({"model": name, "orthogonal": name in ORTHOGONAL, "trials": len(g),
                 "oracle_spread": float(g.oracle.max() - g.oracle.min()),
                 "rho(rloss)": g[["rloss", "oracle"]].corr("spearman").iloc[0, 1],
                 "rho(rmse)": g[["rmse", "oracle"]].corr("spearman").iloc[0, 1]})
rk = pd.DataFrame(rows).sort_values("rho(rloss)", ascending=False)
print(rk.to_string(index=False, float_format=fmt))
print(f"\n  median rho(R-loss, oracle)         : {rk['rho(rloss)'].median():+.3f}")
print(f"  median rho(predictive RMSE, oracle): {rk['rho(rmse)'].median():+.3f}")
print(f"  models where predictive RMSE ANTI-ranks (rho < 0): "
      f"{int((rk['rho(rmse)'] < 0).sum())} / {len(rk)}")

# ============================================ 3. the exception, and why it is not a defect
print("\n" + "=" * 100)
print("3.  THE EXCEPTION: ALREADY-ORTHOGONAL MODELS")
print("=" * 100)
a = rk[~rk.orthogonal]
b = rk[rk.orthogonal]
print(f"  non-orthogonal models  (n={len(a)}): median rho(R-loss) "
      f"{a['rho(rloss)'].median():+.3f}, median oracle spread "
      f"{a.oracle_spread.median():.3f}")
print(f"  orthogonal models      (n={len(b)}): median rho(R-loss) "
      f"{b['rho(rloss)'].median():+.3f}, median oracle spread "
      f"{b.oracle_spread.median():.3f}")
print("""
Read this carefully, because it is the one place the recommendation has a caveat.

`dml_partialling` is the single model where the R-loss ranks configurations *backwards*
(rho = -0.97).  The mechanism is not mysterious.  A Double ML estimate is Neyman-orthogonal,
so its elasticity barely moves with its own nuisance hyperparameters — and the R-loss is
scored with a FIXED nuisance model shared by all candidates.  Whatever residual confounding
that fixed scoring model leaves behind defines the theta it prefers, and that theta is
slightly biased.  Candidates closer to the truth therefore score marginally worse.

So the R-loss is only as good as the nuisance model used to score it, and for an estimator
that is already orthogonal there is little left for it to fix.

What makes the mistake cheap is not orthogonality in general but how much is at stake in
this particular search.  `oracle_spread` above is the range of elasticity error across all
configurations tried.  For `dml_partialling` it is **0.121** — the narrowest in the table,
so ranking it backwards costs almost nothing (-0.0475).  For `lgbm_demand` it is **4.19**,
the widest by a factor of four, and there the R-loss gains +1.61.  That asymmetry, not the
sign of any single correlation, is the argument: the R-loss helps most exactly where the
choice matters most, and its one failure is on the model where the choice matters least.

(Note that `r_learner` is also orthogonal and the R-loss ranks it +0.74, so this is not a
property of orthogonal estimators as a class.  Its second-stage model of eps(x) has real
hyperparameters that do move the answer -- spread 1.07 -- and the R-loss selects among them
correctly.)

`poisson_glm` is a different non-result: rho is near zero for the R-loss because its
elasticity hardly depends on its hyperparameters at all (spread 0.218 over 47 trials), so
there is almost nothing for any objective to select between.""")

# ========================================================== 4. the practical decision rule
print("=" * 100)
print("4.  WHAT TO DO")
print("=" * 100)
worst_rmse = rk.loc[rk["rho(rmse)"].idxmin()]
print(f"""
  Tune on the R-loss.  It ranks correctly for {int((rk['rho(rloss)'] > 0.7).sum())} of
  {len(rk)} models, at a median cost of {piv.rmse_cost.median():+.3f} in predictive RMSE
  -- which is the right trade, because predictive RMSE is not the deliverable.

  Do NOT tune on predictive RMSE.  It anti-ranks for {int((rk['rho(rmse)'] < 0).sum())} of
  {len(rk)} models; worst is {worst_rmse.model} at rho = {worst_rmse['rho(rmse)']:+.3f},
  where a better-predicting configuration is reliably a worse elasticity estimate.

  The gain concentrates where the estimator is not already orthogonal.  For an S-learner it
  is the difference between a usable model and an unusable one.""")

summary = rk.merge(piv[["oracle_gain", "rmse_cost", "rel_gain_%"]], left_on="model",
                   right_index=True, how="left")
summary.to_csv(OUT / "14_hpo_analysis.csv", index=False)
print(f"\nwrote {OUT / '14_hpo_analysis.csv'}")

# --------------------------------------------------------------------------------------
# The winning hyperparameters, in a form you can actually paste into build_model().
#
# 11_objective_comparison.csv was written by a version of tune() that recovered
# best_params from a DataFrame row.  A row has one dtype, so every integer came back as a
# float, and LightGBM rejects `n_estimators=450.0` outright.  tune() now keeps each trial's
# dict verbatim; this repairs the already-written artefact rather than re-running a
# two-hour search for a type.
# --------------------------------------------------------------------------------------
import json                                                                 # noqa: E402

from elasticity_lab.tuning import SEARCH_SPACES, _DummyTrial                # noqa: E402


class _IntSpy(_DummyTrial):
    """Records which parameters the search space declares as integers."""

    def __init__(self):
        self.ints: set[str] = set()

    def suggest_int(self, name, lo, hi, **kw):
        self.ints.add(name)
        return lo


fixed = {}
for _, row in comp.iterrows():
    spy = _IntSpy()
    SEARCH_SPACES[row.model](spy)
    params = json.loads(row.best_params)
    params = {k: (int(round(float(v))) if k in spy.ints else v)
              for k, v in params.items()}
    params = {k: (None if isinstance(v, float) and np.isnan(v) else v)
              for k, v in params.items()}
    fixed[f"{row.model}:{row.tuned_on}"] = params

with open(OUT / "14_best_params.json", "w") as f:
    json.dump(fixed, f, indent=2)
print(f"wrote {OUT / '14_best_params.json'} ({len(fixed)} configurations, ints repaired)")
print("\nthe R-loss winner for the S-learner — the configuration behind the +62% gain:")
print(json.dumps(fixed["lgbm_demand:rloss"], indent=2))
print("\n...and the RMSE winner it beats:")
print(json.dumps(fixed["lgbm_demand:rmse"], indent=2))
