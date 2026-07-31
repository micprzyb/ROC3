#!/usr/bin/env python
"""
The logical constraint  0 < p1 < p2 < p3 < 1  :  propositions and tests.

Each numbered block states a proposition and then tries to break it.  See
``docs/CONSTRAINED.md`` for the derivations.

    .venv/bin/python experiments/04_constrained.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "figures"

from roc3 import CHANCE_VUS, roc_surface, vus_forced_choice  # noqa: E402
from roc3.constrained import (ORDER_CHAMBER_VERTICES, achievable_diagonals,  # noqa: E402
                              check_order_constraint, constrained_report,
                              constrained_world, format_constrained_report,
                              gauge_normalize, normalized_vus, order_chamber_sample,
                              pairwise_auc_ceiling, plot_ceiling, scalar_ceilings,
                              to_ordered_gauge, vus_ceiling)
from roc3.core import convex_hull_vus  # noqa: E402
from roc3.datasets import (make_asymmetric_clinical, make_gaussian_3class,  # noqa: E402
                           make_perfect, make_uninformative)

FAILURES: list[str] = []


def check(name, cond, detail=""):
    if not cond:
        FAILURES.append(f"{name}: {detail}")
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}{('   ' + detail) if detail else ''}")
    return cond


def section(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# ======================================================================================
section("P1 · READING B — the output constraint is pure GAUGE")
# ======================================================================================
print("""  Claim: for ANY model there is a single vector of class multipliers c such that
  p_k -> c_k p_k satisfies p1 < p2 < p3 for every row, and the ROC surface and both VUS
  estimators are bit-for-bit unchanged.  If true, the constraint tells us nothing about
  the model's ranking quality.""")

for name, (X, y, P) in [
    ("gaussian sep=1.2", make_gaussian_3class(300, 1.2, seed=1)),
    ("gaussian sep=2.5", make_gaussian_3class(300, 2.5, seed=2)),
    ("gaussian sep=0.4", make_gaussian_3class(300, 0.4, seed=3)),
]:
    Q, c = to_ordered_gauge(P)
    oc = check_order_constraint(Q)
    s0, s1 = roc_surface(y, P, resolution=200), roc_surface(y, Q, resolution=200)
    f0, f1 = vus_forced_choice(y, P), vus_forced_choice(y, Q)
    same_set = np.array_equal(np.unique(s0.S, axis=0), np.unique(s1.S, axis=0))
    print(f"\n  {name}   log-multipliers log c = {np.round(np.log(np.clip(c, 1e-300, None)), 2)}")
    check(f"{name}: constraint holds after regauging",
          oc["fraction_satisfied"] == 1.0,
          f"{oc['fraction_satisfied']:.1%}, min log margin {oc['min_log_margin']:+.3f}")
    check(f"{name}: 3AFC VUS unchanged", abs(f0 - f1) < 1e-12,
          f"{f0:.10f} vs {f1:.10f}")
    check(f"{name}: geometric VUS unchanged", abs(s0.vus - s1.vus) < 1e-12,
          f"{s0.vus:.10f} vs {s1.vus:.10f}")
    check(f"{name}: identical set of operating points", same_set)

# The converse: a constrained model can still be a perfect ranker.
y, P = make_perfect(200, confidence=0.999, seed=4)
Q, _ = to_ordered_gauge(P)
oc = check_order_constraint(Q)
check("a CONSTRAINED model can still have VUS = 1",
      abs(vus_forced_choice(y, Q) - 1.0) < 1e-12 and oc["satisfied"],
      f"VUS {vus_forced_choice(y, Q):.6f}, constraint satisfied {oc['satisfied']}, "
      f"smallest p1 = {Q[:, 0].min():.3e}")

# ...but the gauge costs representable range: the shift needed is the spread of the
# model's log-odds, so a sharp enough model cannot be re-gauged on the probability scale.
print("\n  dynamic range consumed by the ordered gauge:")
for sep in (1.2, 2.5, 6.0, 20.0):
    X, y, P = make_gaussian_3class(200, sep, seed=4)
    L, _ = to_ordered_gauge(P, return_log=True)
    fits = L.min() >= np.log(np.finfo(float).tiny)
    print(f"    sep {sep:5.1f}   min log p = {L.min():8.1f}   "
          f"representable as float64 probabilities: {fits}")

X, y, P = make_gaussian_3class(200, 20.0, seed=4)
try:
    to_ordered_gauge(P)
    check("gauge guard fires when float64 cannot carry the range", False,
          "no exception raised")
except ValueError as e:
    check("gauge guard fires when float64 cannot carry the range", "float64" in str(e))
Llog, _ = to_ordered_gauge(P, return_log=True)
check("...and on the log scale the identity still holds exactly",
      abs(vus_forced_choice(y, Llog, score_type="log")
          - vus_forced_choice(y, P)) < 1e-12,
      f"{vus_forced_choice(y, Llog, score_type='log'):.10f} vs "
      f"{vus_forced_choice(y, P):.10f}")


# ======================================================================================
section("P2 · what the output constraint DOES break: argmax and the weight simplex")
# ======================================================================================
X, y, P, names = make_asymmetric_clinical(seed=0)
Q, c = to_ordered_gauge(P)
surf_q = roc_surface(y, Q, classes=np.arange(3), resolution=200)
pred = np.argmax(Q, axis=1)
prev = np.bincount(y, minlength=3) / len(y)
print(f"  constrained model: argmax predicts class {np.unique(pred)} for every sample")
check("argmax collapses onto the last class", len(np.unique(pred)) == 1
      and np.unique(pred)[0] == 2)
check("argmax accuracy equals the last class's prevalence",
      abs((pred == y).mean() - prev[2]) < 1e-12,
      f"{(pred == y).mean():.4f} vs prevalence {prev[2]:.4f}")
check("...while the surface is perfectly healthy",
      abs(surf_q.vus - roc_surface(y, P, classes=np.arange(3), resolution=200).vus) < 1e-12,
      f"VUS {surf_q.vus:.4f}")

op_raw = surf_q.select("youden")
G, ref = gauge_normalize(Q, reference="geometric_mean")
surf_g = roc_surface(y, G, classes=np.arange(3), resolution=200)
op_g = surf_g.select("youden")
print(f"\n  optimal weights, raw constrained scores : {np.round(op_raw.weights, 5)}"
      f"   (max/min = {op_raw.weights.max() / op_raw.weights.min():.0f})")
print(f"  optimal weights, gauge-normalised       : {np.round(op_g.weights, 5)}"
      f"   (max/min = {op_g.weights.max() / op_g.weights.min():.1f})")
check("gauge normalisation changes no number", abs(surf_g.vus - surf_q.vus) < 1e-12,
      f"VUS {surf_q.vus:.10f} vs {surf_g.vus:.10f}")
check("...but pulls the optimum back towards the middle of the simplex",
      op_g.weights.max() / op_g.weights.min()
      < op_raw.weights.max() / op_raw.weights.min())
check("...and the operating point is literally the same",
      np.allclose(op_raw.sensitivities, op_g.sensitivities),
      f"{np.round(op_raw.sensitivities, 4)} vs {np.round(op_g.sensitivities, 4)}")


# ======================================================================================
section("P3 · chance level is unaffected by the output constraint")
# ======================================================================================
yu, Pu = make_uninformative(400, seed=7)
Qu, _ = to_ordered_gauge(Pu)
v = vus_forced_choice(yu, Qu)
check("uninformative constrained model still scores ~1/6", abs(v - CHANCE_VUS) < 0.03,
      f"{v:.4f} vs {CHANCE_VUS:.4f}")


# ======================================================================================
section("P4 · READING A — the constraint on the WORLD cuts ROC space")
# ======================================================================================
print("""  Claim: if the TRUE posterior is ordered then for any decision regions R_m,
      pi_i C[i,m] <= pi_j C[j,m]   for i < j,
  because integrating pi_i f_i < pi_j f_j over R_m gives it directly.  Nine linear
  inequalities that no classifier of any kind can escape.""")

for pi in [np.array([1 / 6, 1 / 3, 1 / 2]),
           np.array([0.2, 0.3, 0.5]),
           np.array([0.30, 0.33, 0.37]),
           np.array([0.02, 0.08, 0.90])]:
    V = achievable_diagonals(pi)
    ceil = vus_ceiling(pi)
    sc = scalar_ceilings(pi)
    pw = pairwise_auc_ceiling(pi)
    reach = np.any(np.all(V >= 1 - 1e-9, axis=1))
    print(f"\n  priors {np.round(pi, 3)}")
    print(f"    VUS ceiling          {ceil:.4f}   (vs 1.000 unconstrained)")
    print(f"    max accuracy         {sc['accuracy']:.4f}   "
          f"(LP agrees: {sc['accuracy_lp']:.4f})")
    print(f"    max balanced acc     {sc['balanced_accuracy']:.4f}")
    print(f"    max worst-class Se   {sc['min_sensitivity']:.4f}")
    print(f"    pairwise AUC caps    " + ", ".join(f"{k}:{v:.3f}" for k, v in pw.items()))
    check(f"priors {np.round(pi, 2)}: perfect corner unreachable", not reach)
    check(f"priors {np.round(pi, 2)}: ceiling strictly below 1", ceil < 1.0 - 1e-6,
          f"{ceil:.6f}")
    check(f"priors {np.round(pi, 2)}: ceiling above chance", ceil > CHANCE_VUS)
    check(f"priors {np.round(pi, 2)}: max accuracy == prevalence of class 3",
          abs(sc["accuracy"] - sc["accuracy_lp"]) < 1e-6,
          f"closed form {sc['accuracy']:.6f}, LP {sc['accuracy_lp']:.6f}")


# ======================================================================================
section("P5 · two independent derivations of the ceiling must agree")
# ======================================================================================
print("""  (a) exact: project the confusion polytope, take the down-set volume of the hull.
  (b) LP: solve  max S3 s.t. S1>=u, S2>=v  on a grid and integrate.
  (c) in the BINARY case the same construction must reproduce the closed form
      1 - pi_1/(2 pi_2) that a bounded-likelihood-ratio argument gives.""")

for pi in [np.array([1 / 6, 1 / 3, 1 / 2]), np.array([0.2, 0.3, 0.5])]:
    a = vus_ceiling(pi, method="exact")
    b = vus_ceiling(pi, method="lp", resolution=121)
    check(f"priors {np.round(pi, 2)}: exact vs LP ceiling", abs(a - b) < 3e-3,
          f"exact {a:.5f}   LP {b:.5f}")


def binary_ceiling(pi0, pi1):
    """Down-set area of the achievable (S0, S1) region for two ordered classes."""
    import itertools as it
    # free vars (a0, a1) with C[k,0] = a_k, C[k,1] = 1 - a_k
    rows, rhs = [], []
    for k in range(2):
        r = np.zeros(2); r[k] = -1.0; rows.append(r); rhs.append(0.0)
        r = np.zeros(2); r[k] = 1.0; rows.append(r); rhs.append(1.0)
    rows.append(np.array([pi0, -pi1])); rhs.append(0.0)                 # m = 0
    rows.append(np.array([-pi0, pi1])); rhs.append(float(pi1 - pi0))    # m = 1
    A, b = np.array(rows), np.array(rhs)
    V = []
    for idx in it.combinations(range(len(A)), 2):
        M = A[list(idx)]
        if abs(np.linalg.det(M)) < 1e-12:
            continue
        x = np.linalg.solve(M, b[list(idx)])
        if np.all(A @ x <= b + 1e-9):
            V.append(x)
    V = np.unique(np.round(np.array(V), 9), axis=0)
    S = np.stack([V[:, 0], 1.0 - V[:, 1], np.ones(len(V))], axis=1)     # (S0, S1, 1)
    return convex_hull_vus(S)


print()
for pi0, pi1 in [(0.2, 0.8), (0.4, 0.6), (0.1, 0.9), (0.49, 0.51)]:
    got = binary_ceiling(pi0, pi1)
    want = 1.0 - pi0 / (2.0 * pi1)
    check(f"binary pi=({pi0},{pi1}): polytope ceiling == 1 - pi0/(2 pi1)",
          abs(got - want) < 1e-9, f"{got:.8f} vs {want:.8f}")


# ======================================================================================
section("P6 · is the ceiling attainable?  build the best constrained world we can")
# ======================================================================================
print("""  Construction: sample x, assign it an ordered posterior p(x) inside the order
  chamber T = {p1<p2<p3}, then draw y ~ Categorical(p(x)).  The true posterior is p by
  construction, so this is a genuine Reading-A world and the ideal observer is p itself.
  The extreme points of T are (1/3,1/3,1/3), (0,1/2,1/2) and (0,0,1) -- so even the most
  informative allowed posterior leaves irreducible label noise on classes 1 and 2.""")


print(f"\n  {'atoms':>7} {'conc':>6} {'pi (realised)':>24} {'VUS':>8} {'ceiling':>9} "
      f"{'captured':>9}")
best, best_cfg = 0.0, None
configs = [(3, None), (12, 0.5), (40, 0.15), (40, 0.4), (120, 0.15), (400, 0.12),
           (1200, 0.12), ("lattice", None)]
for n_atoms, conc in configs:
    y, P, atoms, w = constrained_world(n=60_000, n_atoms=n_atoms,
                                       concentration=conc or 0.3, seed=11)
    oc = check_order_constraint(P)
    pi = np.bincount(y, minlength=3) / len(y)
    s = roc_surface(y, P, resolution=200)
    ceil = vus_ceiling(pi)
    cap = (s.vus - CHANCE_VUS) / (ceil - CHANCE_VUS)
    if cap > best:
        best, best_cfg = cap, (n_atoms, conc)
    print(f"  {str(n_atoms):>7} {str(conc):>6} {str(np.round(pi, 3)):>24} {s.vus:8.4f} "
          f"{ceil:9.4f} {cap:9.1%}")
    check(f"n_atoms={n_atoms}: true posterior is ordered", oc["satisfied"],
          f"{oc['fraction_satisfied']:.1%}, min margin {oc['min_log_margin']:+.2e}")
    check(f"n_atoms={n_atoms}: VUS respects the ceiling", s.vus <= ceil + 1e-6,
          f"VUS {s.vus:.4f} vs ceiling {ceil:.4f}")
print(f"\n  best fraction of the attainable volume captured: {best:.1%}  "
      f"(config {best_cfg})")
print("""  The ceiling is a bound over ALL classifiers on ANY world with these priors.
  A single world only exposes a 2-parameter family of Bayes rules, so it cannot sweep
  the whole polytope frontier -- the gap below is expected, not a bug.  See
  docs/CONSTRAINED.md section 6.""")

# Adversarial search: can ANY constrained world beat the ceiling?
print("\n  random search over constrained worlds — trying to break the ceiling:")
rng = np.random.default_rng(0)
worst_slack, n_tried = np.inf, 0
for trial in range(80):
    k = int(rng.integers(3, 40))
    lam = rng.dirichlet(np.full(3, float(rng.uniform(0.1, 2.0))), size=k)
    atoms = order_chamber_sample(lam)
    y, P, _, _ = constrained_world(n=30_000, atoms=atoms,
                                   weights=rng.dirichlet(np.ones(k)), seed=trial)
    if len(np.unique(y)) < 3:
        continue
    pi = np.bincount(y, minlength=3) / len(y)
    if not (pi[0] < pi[1] < pi[2]):
        continue
    n_tried += 1
    s = roc_surface(y, P, resolution=120)
    worst_slack = min(worst_slack, vus_ceiling(pi) - s.vus)
check(f"no constrained world in {n_tried} random trials exceeded its ceiling",
      worst_slack > -1e-6, f"tightest slack observed {worst_slack:+.5f}")


# ======================================================================================
section("P7 · the practical payoff — re-normalising the index")
# ======================================================================================
y, P, _, _ = constrained_world(n=40_000, seed=5, n_atoms=120, concentration=0.15)
pi = np.bincount(y, minlength=3) / len(y)
rep = constrained_report(y, P, classes=np.arange(3), resolution=200)
print(format_constrained_report(rep))
check("normalised VUS lands in [0, 1]", 0.0 <= rep["VUS_normalized"] <= 1.0,
      f"{rep['VUS_normalized']:.4f}")
check("normalised VUS is larger than the raw VUS (the scale was too pessimistic)",
      rep["VUS_normalized"] > rep["VUS"],
      f"raw {rep['VUS']:.4f} -> normalised {rep['VUS_normalized']:.4f}")

fig = plot_ceiling(rep["surface"], pi, path=FIG / "08_constrained_ceiling.png",
                   title="Logical constraint 0 < p₁ < p₂ < p₃ < 1 — the reachable part "
                         "of ROC space")
print(f"\nwrote {FIG / '08_constrained_ceiling.png'}")


# ======================================================================================
section("P8 · how the ceiling depends on the priors")
# ======================================================================================
print("  The constraint implies pi1 < pi2 < pi3.  The more ordered the priors, the more")
print("  room the constraint leaves; near-equal priors are nearly fatal.\n")
print(f"  {'priors':>26} {'VUS ceiling':>12} {'max bal.acc':>12} {'max acc':>9}")
rows = []
for pi in [np.array([0.32, 0.33, 0.35]), np.array([0.25, 0.33, 0.42]),
           np.array([1 / 6, 1 / 3, 1 / 2]), np.array([0.10, 0.25, 0.65]),
           np.array([0.05, 0.20, 0.75]), np.array([0.02, 0.08, 0.90]),
           np.array([0.005, 0.045, 0.95])]:
    ceil = vus_ceiling(pi)
    sc = scalar_ceilings(pi)
    rows.append((pi, ceil))
    print(f"  {str(np.round(pi, 3)):>26} {ceil:12.4f} {sc['balanced_accuracy']:12.4f} "
          f"{sc['accuracy']:9.4f}")
ceils = [c for _, c in rows]
check("ceiling increases as the priors become more strongly ordered",
      all(ceils[i] <= ceils[i + 1] + 1e-9 for i in range(len(ceils) - 1)),
      f"{np.round(ceils, 4)}")
check("near-equal priors force the ceiling towards chance", ceils[0] < 0.35,
      f"{ceils[0]:.4f} for priors {np.round(rows[0][0], 3)}")


# ======================================================================================
section("SUMMARY")
# ======================================================================================
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print("  -", f)
    sys.exit(1)
print("All checks passed.")
