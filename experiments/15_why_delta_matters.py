"""
Why the R-loss and RMSE searches land on such different S-learners.

The two winners from experiments/11 differ most in one place:

    tuned on R-loss : delta = 0.184, 400 trees,  38 leaves
    tuned on RMSE   : delta = 0.015, 1150 trees, 72 leaves

`delta` is the step of the finite difference that turns a fitted demand surface into an
elasticity, and it turns out to be the single most consequential hyperparameter in the
S-learner -- more consequential than anything about the fit itself.

A boosted tree is **piecewise constant in log price**.  Differencing it over a step smaller
than the typical distance to a price split divides an almost-always-zero numerator by an
almost-zero denominator, which is a recipe for enormous per-row variance.

The obvious guess -- that this *attenuates* the elasticity toward zero -- is what this
script was written to confirm, and the measurement says it is wrong.  Across a 60x range of
delta the **bias barely moves** (-0.434 to -0.455) while the **RMSE against the truth falls
by a factor of 5.7** (4.24 to 0.74).  The damage is variance, not bias.  The second guess,
that deeper trees make it worse, is also wrong: more leaves means more price splits nearby,
so the dead zone *shrinks* with depth (16.0% at 8 leaves, 1.8% at 255).

Both are recorded here rather than quietly deleted, because the corrected version is the
more useful statement: predictive RMSE is indifferent to the variance of a derivative it
never computes, and the R-loss is not.
"""
import json
import pathlib

import numpy as np
import pandas as pd

from elasticity_lab.bench import synth_frame
from elasticity_lab.models import LightGBMDemand, perturb_price
from elasticity_lab.features import CONTROL_FEATURES, PRICE_FEATURES

OUT = pathlib.Path("experiments/out")
pd.set_option("display.width", 200)

syn = synth_frame(800, confounding_price=0.6, confounding_demand=0.8)
truth = syn.true_elasticity.to_numpy(float)
print(f"{len(syn):,} rows, true mean elasticity {truth.mean():+.4f}\n", flush=True)

best = json.loads((OUT / "14_best_params.json").read_text())
cols = list(CONTROL_FEATURES) + list(PRICE_FEATURES)


def dead_zone(model, df, delta):
    """Share of rows whose two-sided difference is exactly zero — i.e. same leaf."""
    up = model.m_.predict(perturb_price(df, +delta)[cols].to_numpy(float))
    dn = model.m_.predict(perturb_price(df, -delta)[cols].to_numpy(float))
    d = up - dn
    return float(np.mean(np.abs(d) < 1e-12)), -(d) / (2 * delta)


# ------------------------------------------- 1. the two tuned winners, side by side
print("=" * 96)
print("1.  THE TWO WINNERS")
print("=" * 96)
rows = []
for tag in ("rloss", "rmse"):
    p = dict(best[f"lgbm_demand:{tag}"])
    m = LightGBMDemand(**p).fit(syn)
    zero, e = dead_zone(m, syn, p["delta"])
    rows.append({"tuned_on": tag, "delta": p["delta"], "n_estimators": p["n_estimators"],
                 "num_leaves": p["num_leaves"], "dead_zone_%": 100 * zero,
                 "mean_eps": e.mean(), "true_mean": truth.mean(),
                 "bias": e.mean() - truth.mean(),
                 "oracle_rmse": float(np.sqrt(np.mean((e - truth) ** 2)))})
t1 = pd.DataFrame(rows)
print(t1.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

# --------------------------------- 2. hold the fitted model fixed, vary only the step
print("\n" + "=" * 96)
print("2.  ONE FITTED MODEL, VARYING ONLY THE FINITE-DIFFERENCE STEP")
print("=" * 96)
print("Nothing about the fit changes here.  Only the ruler used to measure its slope.\n")
base = dict(best["lgbm_demand:rmse"])
fitted = LightGBMDemand(**base).fit(syn)
rows = []
for d in (0.005, 0.01, 0.015, 0.03, 0.05, 0.10, 0.18, 0.30):
    zero, e = dead_zone(fitted, syn, d)
    rows.append({"delta": d, "dead_zone_%": 100 * zero, "mean_eps": e.mean(),
                 "bias": e.mean() - truth.mean(),
                 "oracle_rmse": float(np.sqrt(np.mean((e - truth) ** 2)))})
t2 = pd.DataFrame(rows)
print(t2.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))
lo, hi = t2.iloc[0], t2.iloc[-1]
print(f"\n  at delta={lo.delta}: {lo['dead_zone_%']:.1f}% of rows get an elasticity of "
      f"exactly zero, bias {lo.bias:+.3f}")
print(f"  at delta={hi.delta}: {hi['dead_zone_%']:.1f}% dead, bias {hi.bias:+.3f}")

# --------------------------------------- 3. does tree depth make the dead zone worse?
print("\n" + "=" * 96)
print("3.  DOES TREE DEPTH DRIVE THE DEAD ZONE?  (it does -- in the opposite direction)")
print("=" * 96)
print("More leaves means more splits on price, so a small step is MORE likely to cross one.")
print("The dead zone therefore shrinks with depth.  Bias is mildly U-shaped and is not")
print("what delta is controlling.\n")
rows = []
for leaves in (8, 16, 32, 64, 128, 255):
    p = dict(base); p["num_leaves"] = leaves
    m = LightGBMDemand(**p).fit(syn)
    for d in (0.015, 0.18):
        zero, e = dead_zone(m, syn, d)
        rows.append({"num_leaves": leaves, "delta": d, "dead_zone_%": 100 * zero,
                     "bias": e.mean() - truth.mean()})
t3 = pd.DataFrame(rows)
print(t3.pivot(index="num_leaves", columns="delta",
               values=["dead_zone_%", "bias"]).to_string(float_format=lambda v: f"{v:+.3f}"))

for t, f in [(t1, "15_delta_winners.csv"), (t2, "15_delta_sweep.csv"),
             (t3, "15_delta_by_depth.csv")]:
    t.to_csv(OUT / f, index=False)

print(f"""
{"=" * 96}
CONCLUSION
{"=" * 96}
`delta` is not a nuisance knob, it is part of the estimator -- and section 2 is the proof,
because nothing about the fitted surface changes down that column.  Only the ruler changes,
and the elasticity error moves by a factor of 5.7.

What delta controls is VARIANCE, not bias.  This is worth stating precisely because the
intuitive story is the other one.  A boosted tree is piecewise constant in log price, so a
step smaller than the typical distance to a price split divides an almost-always-zero
numerator by an almost-zero denominator.  The result is not a systematic pull toward zero --
the mean elasticity is flat to within 0.02 across the whole sweep -- it is per-row noise
large enough to make the model useless while leaving its average intact.

Two consequences.

  * A model can have a perfectly respectable *average* elasticity and per-row predictions
    that are worthless.  Reporting only the mean hides this completely; `oracle_rmse` and
    the BLP calibration of notebook 4 do not.

  * Predictive RMSE cannot see any of it.  The fitted surface is identical for every delta
    in section 2, so predictive loss is *exactly the same* while the elasticity error moves
    from 4.24 to 0.74.  A search minimising it is choosing delta at random.  The R-loss can
    see it, because a high-variance elasticity fits the residualised outcome badly, and so
    it selects a step large enough to measure with.

That is the mechanism behind the headline number in section 4.4 of the write-up: the same
model class, the same data, a 62% difference in elasticity error.
""")
print(f"wrote {OUT}/15_delta_*.csv")
