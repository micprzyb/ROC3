#!/usr/bin/env python
"""
Validation suite for the 3-class ROC surface / VUS.

Every claim in ``docs/PLAN.md`` §6 is checked numerically here.  Run:

    .venv/bin/python experiments/01_validation.py

Results are appended (by hand) to ``docs/IDEAS_LOG.md``.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from roc3 import (CHANCE_VUS, hypervolume3d, ordinal_roc_surface, roc_surface,
                  vus_forced_choice, vus_ordinal)
from roc3.core import (convex_hull_vus, hypervolume3d_mc, monotone_envelope,
                       _pareto_mask)
from roc3.datasets import (make_adversarial, make_gaussian_3class, make_ordinal_3class,
                           make_perfect, make_uninformative)
from roc3.inference import vus_3afc_inference
from roc3.vus import _exact_fast_vus, forced_choice_profile, hum

FAILURES: list[str] = []


def check(name, cond, detail=""):
    tag = "PASS" if cond else "FAIL"
    if not cond:
        FAILURES.append(f"{name}: {detail}")
    print(f"  [{tag}] {name}{('   ' + detail) if detail else ''}")
    return cond


def section(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ======================================================================================
section("1. Hypervolume kernel — hand-checkable cases")
# ======================================================================================
cases = [
    ("single corner (1,1,1)", np.array([[1.0, 1.0, 1.0]]), 1.0),
    ("two boxes (incl-excl)", np.array([[1, .5, .8], [.5, 1, .6]]), 0.55),
    ("three axis corners", np.eye(3), 0.0),
    ("half cube", np.array([[1.0, 1.0, 0.5]]), 0.5),
]
for nm, pts, want in cases:
    got = hypervolume3d(pts)
    check(f"HV {nm}", abs(got - want) < 1e-12, f"got {got:.12f}, want {want}")

# A dense sample of the chance plane must give 1/6 in the limit.
t = np.linspace(0, 1, 401)
U, V = np.meshgrid(t, t, indexing="ij")
m = (U + V) <= 1
plane = np.stack([U[m], V[m], 1 - U[m] - V[m]], axis=1)
hv = hypervolume3d(plane)
check("HV of the chance plane -> 1/6", abs(hv - CHANCE_VUS) < 2e-3,
      f"got {hv:.5f}, want {CHANCE_VUS:.5f} (staircase, converges from below)")
check("convex-hull VUS of the three corners == 1/6",
      abs(convex_hull_vus(np.eye(3)) - CHANCE_VUS) < 1e-9,
      f"got {convex_hull_vus(np.eye(3)):.6f}")

rng = np.random.default_rng(7)
for trial in range(3):
    pts = rng.random((300, 3))
    exact = hypervolume3d(pts)
    mc, se = hypervolume3d_mc(pts, 200_000, seed=trial)
    check(f"HV exact vs Monte Carlo #{trial}", abs(exact - mc) < 4 * se + 1e-9,
          f"exact {exact:.5f}  mc {mc:.5f} +- {se:.5f}")

# Pareto mask must not change the hypervolume.
pts = rng.random((500, 3))
check("Pareto filtering leaves HV unchanged",
      abs(hypervolume3d(pts) - hypervolume3d(pts[_pareto_mask(pts)])) < 1e-12)


# ======================================================================================
section("2. Extreme models — perfect / uninformative / adversarial")
# ======================================================================================
for nm, (y, P), want, tol in [
    ("perfect", make_perfect(200), 1.0, 1e-9),
    ("uninformative", make_uninformative(400), CHANCE_VUS, 0.03),
    ("adversarial (cyclic)", make_adversarial(200), 0.0, 1e-9),
]:
    s = roc_surface(y, P, resolution=200)
    fc = vus_forced_choice(y, P)
    check(f"{nm}: geometric VUS", abs(s.vus - want) < tol, f"{s.vus:.5f} (want {want})")
    check(f"{nm}: 3AFC VUS", abs(fc - want) < tol, f"{fc:.5f} (want {want})")


# ======================================================================================
section("3. Binary reduction — the 2-class down-set area IS the AUC")
# ======================================================================================
def binary_downset_area(y, score):
    """Area of the down-set of the achievable (specificity, sensitivity) points.

    This is the 2-class instance of exactly the construction used for the 3-class VUS:
    sweep the whole 1-parameter rule family, collect operating points, take the volume
    of the region they dominate.
    """
    y = np.asarray(y)
    score = np.asarray(score, dtype=float)
    u = np.unique(score)
    cuts = np.concatenate(([u[0] - 1.0], (u[:-1] + u[1:]) / 2.0, [u[-1] + 1.0]))
    pos, neg = score[y == 1], score[y == 0]
    se = (pos[None, :] > cuts[:, None]).mean(axis=1)
    sp = (neg[None, :] <= cuts[:, None]).mean(axis=1)
    pts = np.stack([sp, se, np.ones_like(se)], axis=1)      # lift to 3-D, z == 1
    return hypervolume3d(pts)


from sklearn.metrics import roc_auc_score  # noqa: E402

rng = np.random.default_rng(11)
worst = 0.0
for trial in range(8):
    n = int(rng.integers(60, 400))
    yb = rng.integers(0, 2, n)
    if yb.sum() in (0, n):
        continue
    sb = rng.normal(yb * rng.uniform(0, 2.0), 1.0)
    a1, a2 = binary_downset_area(yb, sb), roc_auc_score(yb, sb)
    worst = max(worst, abs(a1 - a2))
check("down-set area == sklearn roc_auc_score (8 random problems)", worst < 1e-12,
      f"max |diff| = {worst:.2e}")

# ... and the rank-based estimator reduces to the AUC too (HUM at k = 2).
yb = rng.integers(0, 2, 4000)
sb = rng.normal(yb * 0.9, 1.0)
h2 = hum(yb, np.stack([-sb, sb], axis=1), score_type="log", n_samples=2_000_000)["hum"]
auc = roc_auc_score(yb, sb)
check("HUM(k=2) == AUC", abs(h2 - auc) < 2e-3, f"HUM {h2:.5f}  AUC {auc:.5f}")


# ======================================================================================
section("4. Monotonicity in signal strength")
# ======================================================================================
prev_g = prev_f = -1.0
mono = True
print(f"  {'sep':>5}  {'VUS_geom':>9}  {'VUS_3AFC':>9}  {'VUS_adj':>8}  {'M':>6}")
from roc3.metrics import hand_till_m  # noqa: E402

for sep in [0.1, 0.3, 0.6, 1.0, 1.5, 2.0, 2.5, 3.0]:
    X, y, P = make_gaussian_3class(400, sep, seed=2)
    s = roc_surface(y, P, resolution=200)
    f = vus_forced_choice(y, P)
    print(f"  {sep:5.2f}  {s.vus:9.4f}  {f:9.4f}  {s.vus_adjusted:8.4f}  "
          f"{hand_till_m(y, P):6.4f}")
    mono &= (s.vus > prev_g - 1e-9) and (f > prev_f - 1e-9)
    prev_g, prev_f = s.vus, f
check("both VUS estimators increase monotonically with separation", mono)


# ======================================================================================
section("5. Grid convergence of the geometric VUS")
# ======================================================================================
X, y, P = make_gaussian_3class(300, 1.0, seed=1)
target = vus_forced_choice(y, P)
print(f"  reference 3AFC VUS = {target:.5f}")
print(f"  {'R':>5}  {'rules':>8}  {'VUS':>8}  {'hull':>8}  {'gap to 3AFC':>12}  {'s':>6}")
vals = []
for R in [20, 40, 80, 160, 320, 640]:
    t0 = time.time()
    s = roc_surface(y, P, resolution=R)
    dt = time.time() - t0
    vals.append(s.vus)
    print(f"  {R:5d}  {len(s.S):8d}  {s.vus:8.5f}  {s.vus_convex_hull:8.5f}  "
          f"{s.vus - target:+12.5f}  {dt:6.2f}")
check("VUS increases monotonically with grid resolution",
      all(vals[i] <= vals[i + 1] + 1e-12 for i in range(len(vals) - 1)))
# Richardson-style extrapolation of the O(1/R) tail.
d = vals[-1] - vals[-2]
check("extrapolated limit within 0.005 of the 3AFC value",
      abs(vals[-1] + d - target) < 0.005,
      f"extrapolated {vals[-1] + d:.5f}  vs 3AFC {target:.5f}")


# ======================================================================================
section("6. Do the two VUS definitions agree?  (Scurfield/Mossman equivalence)")
# ======================================================================================
print("  ideal observer (true posteriors) — the case the theorem covers")
rows = []
for seed in range(6):
    sep = 0.4 + 0.4 * seed
    X, y, P = make_gaussian_3class(250, sep, seed=100 + seed)
    s = roc_surface(y, P, resolution=400)
    f = vus_forced_choice(y, P)
    rows.append((sep, s.vus, f, s.vus_convex_hull))
    print(f"    sep {sep:4.2f}   geom {s.vus:.4f}   3AFC {f:.4f}   "
          f"diff {s.vus - f:+.4f}   hull {s.vus_convex_hull:.4f}")
diffs = np.array([abs(r[1] - r[2]) for r in rows])
check("ideal observer: |geom - 3AFC| < 0.01 at R=400", diffs.max() < 0.01,
      f"max |diff| = {diffs.max():.4f}")

print("\n  deliberately mis-calibrated / non-ideal scorers")
rng = np.random.default_rng(3)
bad = []
for seed in range(6):
    X, y, P = make_gaussian_3class(250, 1.2, seed=200 + seed)
    # temperature + per-class distortion + a random monotone warp of one column
    Q = P ** rng.uniform(0.4, 2.5)
    Q *= rng.uniform(0.3, 3.0, size=3)
    Q[:, 1] = Q[:, 1] ** rng.uniform(0.5, 2.0)
    Q /= Q.sum(axis=1, keepdims=True)
    s = roc_surface(y, Q, resolution=400)
    f = vus_forced_choice(y, Q)
    bad.append((s.vus, f))
    print(f"    seed {seed}    geom {s.vus:.4f}   3AFC {f:.4f}   diff {s.vus - f:+.4f}")
bd = np.array([abs(a - b) for a, b in bad])
print(f"    max |diff| over non-ideal scorers = {bd.max():.4f}")


# ======================================================================================
section("7. Invariance of the 3AFC VUS")
# ======================================================================================
X, y, P = make_gaussian_3class(200, 1.3, seed=42)
base = vus_forced_choice(y, P)
Pt = P ** 0.37
Pt /= Pt.sum(axis=1, keepdims=True)
check("invariant to temperature scaling p -> p^alpha",
      abs(vus_forced_choice(y, Pt) - base) < 1e-9,
      f"{vus_forced_choice(y, Pt):.6f} vs {base:.6f}")
Pc = P * np.array([5.0, 0.2, 1.7])
Pc /= Pc.sum(axis=1, keepdims=True)
check("invariant to class re-weighting p_k -> c_k p_k",
      abs(vus_forced_choice(y, Pc) - base) < 1e-9,
      f"{vus_forced_choice(y, Pc):.6f} vs {base:.6f}")
sg = roc_surface(y, P, resolution=200).vus
sgc = roc_surface(y, Pc, resolution=200).vus
check("geometric VUS invariant to class re-weighting", abs(sg - sgc) < 1e-9,
      f"{sg:.6f} vs {sgc:.6f}")


# ======================================================================================
section("8. 3AFC estimator internals: exact vs exact_fast vs Monte Carlo")
# ======================================================================================
X, y, P = make_gaussian_3class(90, 1.1, seed=17)
r_exact = forced_choice_profile(y, P, method="exact")
r_fast = forced_choice_profile(y, P, method="exact_fast")
r_mc = forced_choice_profile(y, P, method="mc", n_samples=2_000_000)
check("exact == exact_fast (no ties present)",
      abs(r_exact.vus - r_fast.vus) < 1e-12,
      f"{r_exact.vus:.8f} vs {r_fast.vus:.8f}")
check("Monte Carlo within 4 SE of exact",
      abs(r_exact.vus - r_mc.vus) < 4 * r_mc.mc_stderr,
      f"{r_mc.vus:.5f} +- {r_mc.mc_stderr:.5f} vs exact {r_exact.vus:.5f}")
check("the six permutation probabilities sum to 1",
      abs(r_exact.profile.sum() - 1.0) < 1e-9, f"{r_exact.profile.sum():.10f}")


# ======================================================================================
section("9. Envelope integration == hypervolume")
# ======================================================================================
X, y, P = make_gaussian_3class(300, 1.4, seed=5)
s = roc_surface(y, P, resolution=300)
for R in (101, 201, 401, 801):
    _, _, Z = monotone_envelope(s.S, resolution=R)
    print(f"  envelope R={R:4d}   mean(Z) = {Z.mean():.5f}   exact HV = {s.vus:.5f}   "
          f"diff {Z.mean() - s.vus:+.5f}")
_, _, Z = monotone_envelope(s.S, resolution=801)
check("gridded envelope integrates to the exact hypervolume",
      abs(Z.mean() - s.vus) < 2e-3, f"{Z.mean():.5f} vs {s.vus:.5f}")


# ======================================================================================
section("10. Ordinal mode (Mossman / Nakas)")
# ======================================================================================
y, sm = make_ordinal_3class(200, separation=1.3, seed=8)


def brute_p_ordered(y, s):
    a, b, c = s[y == 0], s[y == 1], s[y == 2]
    lt = (a[:, None, None] < b[None, :, None]) & (b[None, :, None] < c[None, None, :])
    return lt.mean()


vo = vus_ordinal(y, sm)
vb = brute_p_ordered(y, sm)
check("vus_ordinal == brute-force P(s1<s2<s3)", abs(vo - vb) < 1e-12,
      f"{vo:.8f} vs {vb:.8f}")
osurf = ordinal_roc_surface(y, sm, resolution=400)
print(f"  geometric VUS of the two-cut-point family = {osurf.vus:.4f}   "
      f"rank VUS = {vo:.4f}")
check("ordinal geometric VUS close to the rank VUS", abs(osurf.vus - vo) < 0.02,
      f"{osurf.vus:.4f} vs {vo:.4f}")

y0, s0 = make_ordinal_3class(300, separation=0.0, seed=9)
check("uninformative ordinal marker -> 1/6", abs(vus_ordinal(y0, s0) - CHANCE_VUS) < 0.03,
      f"{vus_ordinal(y0, s0):.4f}")
check("direction='decreasing' mirrors the result",
      abs(vus_ordinal(y, -sm, direction="decreasing") - vo) < 1e-12)


# ======================================================================================
section("11. Inference")
# ======================================================================================
X, y, P = make_gaussian_3class(150, 1.2, seed=21)
inf = vus_3afc_inference(y, P)
print(f"  VUS {inf['vus']:.4f}  se {inf['stderr']:.4f}  "
      f"95% CI ({inf['ci'][0]:.4f}, {inf['ci'][1]:.4f})  p={inf['p_value']:.2e}")
check("closed-form point estimate == exact_fast", abs(inf["vus"] - _exact_fast_vus(
    *[np.log(np.clip(P[y == k] / P[y == k].sum(1, keepdims=True), 1e-12, None))
      for k in range(3)])) < 1e-12)

# Calibration of the closed-form SE against a stratified bootstrap.
from roc3.inference import bootstrap_vus  # noqa: E402

bs = bootstrap_vus(y, P, kind="3afc", n_boot=200, seed=1,
                   fc_kwargs={"method": "mc", "n_samples": 60_000})
print(f"  bootstrap se {bs['3afc']['stderr']:.4f}   closed-form se {inf['stderr']:.4f}")
check("closed-form SE agrees with the bootstrap SE within 25%",
      abs(bs["3afc"]["stderr"] - inf["stderr"]) < 0.25 * inf["stderr"],
      f"{bs['3afc']['stderr']:.4f} vs {inf['stderr']:.4f}")

# Null behaviour of the permutation test.
from roc3.inference import permutation_test_vus  # noqa: E402

yn, Pn = make_uninformative(150, seed=4)
pt = permutation_test_vus(yn, Pn, kind="3afc", n_perm=200, seed=2)
print(f"  null model: observed {pt['observed']:.4f}, null mean {pt['null_mean']:.4f}, "
      f"p = {pt['p_value']:.3f}")
check("permutation test does not reject on a null model", pt["p_value"] > 0.05,
      f"p = {pt['p_value']:.3f}")
check("permutation null centres on 1/6", abs(pt["null_mean"] - CHANCE_VUS) < 0.02,
      f"{pt['null_mean']:.4f}")


# ======================================================================================
section("12. Threshold selection behaves")
# ======================================================================================
from roc3.datasets import make_asymmetric_clinical  # noqa: E402

X, y, P, names = make_asymmetric_clinical(seed=0)
surf = roc_surface(names[y], P, classes=names, resolution=200)
for crit in ("youden", "balanced_accuracy", "closest_to_perfection", "maximin",
             "accuracy", "max_volume"):
    op = surf.select(crit)
    print(f"  {crit:>22}  Se = {np.round(op.sensitivities, 3)}   "
          f"w = {np.round(op.weights, 3)}")
check("youden and balanced_accuracy pick the same rule",
      surf.select("youden").index == surf.select("balanced_accuracy").index)
op_c = surf.select("balanced_accuracy", constraints={"malignant": 0.95})
check("constraint is respected", op_c.sensitivities[2] >= 0.95,
      f"Se[malignant] = {op_c.sensitivities[2]:.4f}")
check("constrained optimum is no better than the unconstrained one",
      op_c.sensitivities.mean() <= surf.select("balanced_accuracy").sensitivities.mean()
      + 1e-12)
try:
    surf.select("youden", constraints={"malignant": 0.999, "benign": 0.999})
    check("infeasible constraints raise", False, "no exception")
except ValueError as e:
    check("infeasible constraints raise a helpful error", "not achievable" in str(e))

# The confusion matrix stored on the grid must match a fresh argmax evaluation.
op = surf.select("youden")
check("stored confusion == recomputed confusion at the selected w",
      np.array_equal(op.confusion, surf.confusion_at(op.weights)),
      f"\n{op.confusion}\n{surf.confusion_at(op.weights)}")


# ======================================================================================
section("SUMMARY")
# ======================================================================================
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("All checks passed.")
