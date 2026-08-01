"""
Prediction P4, and the first look at the price-level classifier.

P4 is a *theorem*, so it is the cheapest possible check that the plumbing is right:

  * the gauge map ``p_k -> c_k p_k`` (renormalised) leaves VUS bit-for-bit unchanged;
  * the same map shifts each arc elasticity by exactly
    ``-(log c_{k+1} - log c_k) / (bar_r_{k+1} - bar_r_k)``.

Together those two facts say VUS carries **no information about the elasticity level** —
the whole answer to "how does VUS relate to elasticity quality" (PLAN §4.1).  If either
half fails numerically, it is a bug in this package, not a finding.
"""
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

import roc3                                                             # noqa: E402
from elasticity_lab.bench import synth_frame, real_frame                # noqa: E402
from elasticity_lab.pricelevels import (PriceLevelClassifier, PriceLevels,  # noqa: E402
                                        gauge_transform, isotonic_decreasing,
                                        vus_report)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)
fmt = lambda v: f"{v:+.6f}"                                             # noqa: E731
rng = np.random.default_rng(0)
FAILURES = []


def check(label, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'}  {label}   {detail}", flush=True)
    if not ok:
        FAILURES.append(label)


# ============================================================ 0. isotonic projection
print("=" * 96)
print("0.  PAVA: the projection onto the decreasing cone")
print("=" * 96)
v = np.array([[3.0, 1.0, 2.0], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
p = isotonic_decreasing(v)
print(pd.DataFrame({"input": [list(x) for x in v], "projected": [list(x) for x in p]})
      .to_string(index=False))
check("already-decreasing rows are untouched", np.allclose(p[2], v[2]))
check("increasing row collapses to its mean", np.allclose(p[1], 2.0))
check("output is decreasing everywhere", bool(np.all(np.diff(p, axis=1) <= 1e-12)))
# the L2-projection property: it is the CLOSEST decreasing vector
for row in range(3):
    cand = p[row] + rng.normal(0, 0.3, 3)
    cand = np.sort(cand)[::-1]                       # any decreasing vector
    d_proj = np.sum((v[row] - p[row]) ** 2)
    d_cand = np.sum((v[row] - cand) ** 2)
    check(f"row {row}: projection is at least as close as a random decreasing vector",
          d_proj <= d_cand + 1e-9, f"{d_proj:.4f} <= {d_cand:.4f}")

# ============================================================ 1. levels
print("\n" + "=" * 96)
print("1.  PRICE LEVELS")
print("=" * 96)
syn = synth_frame(700, confounding_price=0.6, confounding_demand=0.8)
for K in (3, 4, 5):
    lv = PriceLevels(K=K).fit(syn)
    print(f"\nK = {K}")
    print(lv.describe().to_string(index=False, float_format=lambda x: f"{x:+.4f}"))
    check(f"K={K}: level prices strictly increasing", bool(np.all(np.diff(lv.bar_r_) > 0)))
    check(f"K={K}: levels roughly balanced",
          lv.counts_.min() > 0.5 * lv.counts_.mean(), f"counts {lv.counts_}")

# ============================================================ 2. THE GAUGE THEOREM (P4)
print("\n" + "=" * 96)
print("2.  P4 -- THE GAUGE THEOREM")
print("=" * 96)
m = PriceLevelClassifier(K=3, n_estimators=200).fit(syn)
lv = m.levels_.transform(syn)
proba = m.posterior(syn)
bar_r = m.levels_.bar_r_

vus0 = roc3.vus_forced_choice(lv, proba)
surf0 = roc3.roc_surface(lv, proba, resolution=160).vus
log_s0 = m.demand_curve(syn)
arcs0 = -np.diff(log_s0, axis=1) / np.diff(bar_r)[None, :]

print(f"baseline:  VUS(3AFC) = {vus0:.10f}   VUS(geometric) = {surf0:.10f}")
print(f"           mean arc elasticities = {arcs0.mean(0)}")

rows = []
for trial in range(5):
    c = np.exp(rng.normal(0, 0.8, 3))
    pg = gauge_transform(proba, c)
    vg = roc3.vus_forced_choice(lv, pg)
    sg = roc3.roc_surface(lv, pg, resolution=160).vus

    # the same gauge applied to the demand curve
    log_sg = log_s0 + np.log(c)[None, :]
    arcs_g = -np.diff(log_sg, axis=1) / np.diff(bar_r)[None, :]
    predicted_shift = -np.diff(np.log(c)) / np.diff(bar_r)
    actual_shift = arcs_g.mean(0) - arcs0.mean(0)

    rows.append({"trial": trial, "c": np.round(c, 3).tolist(),
                 "dVUS_3afc": vg - vus0, "dVUS_geom": sg - surf0,
                 "predicted_arc_shift": np.round(predicted_shift, 6).tolist(),
                 "actual_arc_shift": np.round(actual_shift, 6).tolist(),
                 "shift_error": float(np.max(np.abs(predicted_shift - actual_shift)))})
g = pd.DataFrame(rows)
print("\n" + g.to_string(index=False))

# The 3AFC estimator is a rank statistic and the invariance is EXACT.  The geometric one
# integrates over a finite grid of weight vectors, so a gauge moves which grid points map
# to which rules and the answer wobbles at the grid's own O(1/R) accuracy.  That is the
# documented behaviour of the estimator (README "Caveats"), not a violation -- so the two
# get different tolerances, and the difference is itself worth seeing.
check("VUS (3AFC) is invariant under the gauge -- EXACTLY",
      g.dVUS_3afc.abs().max() == 0.0, f"max |dVUS| = {g.dVUS_3afc.abs().max():.2e}")
check("VUS (geometric) is invariant to within its grid resolution",
      g.dVUS_geom.abs().max() < 1e-4, f"max |dVUS| = {g.dVUS_geom.abs().max():.2e}")
check("arc elasticities shift by exactly the predicted amount",
      g.shift_error.max() < 1e-10, f"max error = {g.shift_error.max():.2e}")

span = float(np.abs(g.actual_arc_shift.apply(lambda a: np.max(np.abs(a))).max()))
print(f"""
  The gauge moved the arc elasticities by up to {span:.3f} while moving VUS by
  {g.dVUS_3afc.abs().max():.1e}.  That is the point: the transformation VUS cannot see is
  exactly the one that sets the answer.  VUS constrains the ranking; the LEVEL is fixed
  entirely by calibration -- which, in a randomised price test, is what randomisation
  buys you for free.""")

# ============================================================ 3. the model, three ways
print("=" * 96)
print("3.  THE CLASSIFIER, UNCONSTRAINED vs CONSTRAINED")
print("=" * 96)
truth = syn.true_elasticity.to_numpy(float)
rows = []
for constraint in ("none", "isotonic", "softplus"):
    for use_pi in (True, False) if constraint == "none" else (True,):
        mm = PriceLevelClassifier(K=3, constraint=constraint, use_propensity=use_pi,
                                  n_estimators=200).fit(syn)
        e = np.asarray(mm.elasticity(syn), float)
        d = mm.diagnostics(syn)
        rows.append({"constraint": constraint, "use_pi": use_pi,
                     "mean_eps": e.mean(), "sd_eps": e.std(),
                     "bias": e.mean() - truth.mean(),
                     "oracle_rmse": float(np.sqrt(np.mean((e - truth) ** 2))),
                     "corr": float(np.corrcoef(e, truth)[0, 1]) if e.std() > 1e-9 else np.nan,
                     "pct_eps_negative": float(np.mean(e < 0)),
                     "pct_arc_negative": d["pct_arc_negative"],
                     "min_pi_p5": d["min_propensity_p5"],
                     "ess_frac": d["kish_ess_frac"]})
t = pd.DataFrame(rows)
print(f"true mean elasticity = {truth.mean():+.4f}, true sd = {truth.std():.4f}\n")
print(t.to_string(index=False, float_format=lambda v: f"{v:+.4f}"))

check("the constrained models produce no negative arc elasticities",
      float(t[t.constraint != "none"].pct_arc_negative.max()) == 0.0)

# ============================================================ 4. VUS in context
print("\n" + "=" * 96)
print("4.  VUS, AND WHY THE RAW NUMBER IS NOT INTERPRETABLE")
print("=" * 96)
for constraint in ("none", "softplus"):
    mm = PriceLevelClassifier(K=3, constraint=constraint, n_estimators=200).fit(syn)
    rep = vus_report(mm, syn)
    print(f"\n--- constraint = {constraint} ---")
    for k, v in rep.items():
        print(f"    {k:22s} {v}")

print("\n" + "=" * 96)
if FAILURES:
    print(f"{len(FAILURES)} FAILURES: {FAILURES}")
    raise SystemExit(1)
print("all checks passed")
