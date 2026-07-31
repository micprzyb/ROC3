#!/usr/bin/env python
"""
Every number quoted in ``docs/CONSTRAINED.md``, recomputed from scratch.

Uses one three-atom constrained world throughout, small enough that the master lemma and
the confusion polytope can be checked by hand.

    .venv/bin/python experiments/06_constrained_numbers.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roc3 import CHANCE_VUS, binary_auc, roc_surface, vus_forced_choice  # noqa: E402
from roc3.constrained import (achievable_diagonals, ceiling_surface,  # noqa: E402
                              check_order_constraint, confusion_polytope,
                              gauge_normalize, normalized_vus, pairwise_auc_ceiling,
                              scalar_ceilings, to_ordered_gauge, vus_ceiling)
from roc3.datasets import make_gaussian_3class, make_perfect  # noqa: E402

np.set_printoptions(suppress=True, precision=4, linewidth=130)


def head(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# --------------------------------------------------------------------------------------
# The three-atom constrained world used throughout the note
# --------------------------------------------------------------------------------------
ATOM = ["A", "B", "C"]
G = np.array([0.50, 0.30, 0.20])                       # P(x = atom m)
POST = np.array([[0.20, 0.30, 0.50],                   # P(Y=k | x = A)
                 [0.10, 0.30, 0.60],                   # P(Y=k | x = B)
                 [0.05, 0.15, 0.80]])                  # P(Y=k | x = C)
PI = G @ POST                                          # class prevalences

head("§1  A world whose TRUE POSTERIOR is ordered")
print(f"{'atom':>5} {'P(x=atom)':>10} {'p1':>7} {'p2':>7} {'p3':>7}   ordered?")
for m in range(3):
    print(f"{ATOM[m]:>5} {G[m]:10.2f} {POST[m,0]:7.2f} {POST[m,1]:7.2f} {POST[m,2]:7.2f}"
          f"   {POST[m,0] < POST[m,1] < POST[m,2]}")
print(f"\nclass prevalences  pi_k = sum_m P(x=m) p_k(m) = {np.round(PI, 4)}"
      f"   (sums to {PI.sum():.4f})")
print(f"the constraint forces pi1 < pi2 < pi3:  {PI[0] < PI[1] < PI[2]}")

print("\nthe JOINT table  J[k,m] = P(Y=k AND x=atom m) = P(x=m) * p_k(m):")
J = G[None, :] * POST.T                                # J[k, m]
print(f"{'':>10}" + "".join(f"{('x='+a):>10}" for a in ATOM) + f"{'row sum':>10}")
for k in range(3):
    print(f"  class {k+1:<3}" + "".join(f"{J[k,m]:10.3f}" for m in range(3))
          + f"{J[k].sum():10.3f}")
print("\n  -> every COLUMN increases down the class index, because the posterior does.")
print("     That single fact is the master lemma.")

print("\nclass-conditional distributions  f_k(m) = J[k,m] / pi_k:")
F = J / PI[:, None]
for k in range(3):
    print(f"  f_{k+1} = {np.round(F[k], 4)}   (sums to {F[k].sum():.4f})")

head("§2  The master lemma, checked on every rule")
print("A 'rule' assigns each atom to a predicted class.  With 3 atoms there are 3^3 = 27")
print("deterministic rules.  C[k,m] = P(predict m | Y=k) = sum over atoms sent to m.\n")


def confusion(assign):
    """assign[m] = predicted class (0-based) for atom m."""
    C = np.zeros((3, 3))
    for m, pred in enumerate(assign):
        C[:, pred] += F[:, m]
    return C


for assign, name in [((0, 1, 2), "A->1, B->2, C->3"),
                     ((2, 2, 2), "always predict 3"),
                     ((0, 0, 2), "A->1, B->1, C->3")]:
    C = confusion(assign)
    S = np.diag(C)
    print(f"rule: {name}")
    print(f"  confusion C[k,m] (rows=true, cols=predicted):\n{np.round(C, 4)}")
    print(f"  operating point (S1,S2,S3) = {np.round(S, 4)}")
    worst = min(PI[j] * C[j, m] - PI[i] * C[i, m]
                for i, j in itertools.combinations(range(3), 2) for m in range(3))
    print(f"  lemma  pi_i C[i,m] <= pi_j C[j,m]  for all i<j, all m:  "
          f"holds, tightest slack {worst:+.4f}")
    print(f"  accuracy = sum_k pi_k S_k = {float(PI @ S):.4f}\n")

print(f"max accuracy over ALL rules and all classifiers = pi_3 = {PI[2]:.4f}")
best = max(float(PI @ np.diag(confusion(a)))
           for a in itertools.product(range(3), repeat=3))
print(f"best of the 27 deterministic rules             = {best:.4f}   "
      f"(the constant rule attains it)")

head("§3  Why the perfect corner is unreachable")
print(f"S1 = S2 = 1 would need C[1,1] = 1 and C[2,2] = 1, hence C[2,1] = 0.")
print(f"The lemma with i=1, j=2, m=1 then says  pi_1 * 1 <= pi_2 * 0,")
print(f"i.e. {PI[0]:.4f} <= 0.  False.  So (1,1,1) is not achievable by anything.\n")
print("The three cutting planes, with these priors:")
for (i, j), lab in zip([(0, 1), (1, 2), (0, 2)],
                       ["pi1*S1 + pi2*S2 <= pi2", "pi2*S2 + pi3*S3 <= pi3",
                        "pi1*S1 + pi3*S3 <= pi3"]):
    print(f"  {lab:<28} ->  {PI[i]:.4f}*S{i+1} + {PI[j]:.4f}*S{j+1} <= {PI[j]:.4f}"
          f"   at (1,1,1): {PI[i] + PI[j]:.4f} <= {PI[j]:.4f} is "
          f"{PI[i] + PI[j] <= PI[j]}")

head("§4  Pairwise AUC ceilings, and a construction that attains one")
print("The constraint pi_i f_i <= pi_j f_j means the likelihood ratio is bounded:")
print("     L = f_i / f_j  <=  R := pi_j / pi_i .")
print("A bounded likelihood ratio caps the AUC at 1 - 1/(2R) = 1 - pi_i/(2 pi_j).\n")
caps = pairwise_auc_ceiling(PI)
for (i, j), v in caps.items():
    R = PI[j] / PI[i]
    print(f"  classes {i+1} vs {j+1}:  R = {PI[j]:.4f}/{PI[i]:.4f} = {R:.3f}"
          f"   ->  AUC <= 1 - 1/(2*{R:.3f}) = {v:.4f}")

print("\nIs the bound tight?  Build the extremal case explicitly for R = 4.")
print("  Two atoms u, v.  Class j sits at u with prob 1-1/R and at v with prob 1/R;")
print("  class i sits entirely at v.  Then L(u) = 0 and L(v) = R exactly.")
R = 4.0
n_j, n_i = 400, 100
scores_j = np.r_[np.zeros(int(n_j * (1 - 1 / R))), np.full(int(n_j / R), R)]
scores_i = np.full(n_i, R)
got = binary_auc(scores_i, scores_j)
print(f"  class j scores: {int(n_j*(1-1/R))} at 0 and {int(n_j/R)} at {R:.0f}")
print(f"  class i scores: {n_i} at {R:.0f}")
print(f"  AUC (ties get 1/2) = {got:.6f}      bound 1 - 1/(2R) = {1 - 1/(2*R):.6f}"
      f"      attained: {abs(got - (1 - 1/(2*R))) < 1e-12}")

head("§5  The confusion polytope, concretely")
A_ub, b_ub, pi_used = confusion_polytope(PI)
print("Free variables x = (a0,b0, a1,b1, a2,b2) with C[k,0]=a_k, C[k,1]=b_k,")
print("C[k,2] = 1 - a_k - b_k.  So 6 numbers describe a whole confusion matrix.")
print(f"Facets (inequalities): {A_ub.shape[0]}  =  9 simplex constraints "
      f"(a_k>=0, b_k>=0, a_k+b_k<=1)  +  9 lemma constraints")
V = achievable_diagonals(PI)
print(f"\nThe polytope has vertices; projecting each onto its diagonal (S1,S2,S3)")
print(f"and discarding duplicates leaves {len(V)} corner operating points.")
print("The ones on the Pareto frontier (nothing beats them on all three axes):")
from roc3.core import _pareto_mask  # noqa: E402
front = V[_pareto_mask(V)]
for s in front[np.lexsort((-front[:, 2], -front[:, 1], -front[:, 0]))]:
    print(f"    ({s[0]:.4f}, {s[1]:.4f}, {s[2]:.4f})")

ceil_exact = vus_ceiling(PI, method="exact")
ceil_lp = vus_ceiling(PI, method="lp", resolution=121)
print(f"\nVUS ceiling, exact (vertex enumeration + hull volume) = {ceil_exact:.5f}")
print(f"VUS ceiling, LP grid (independent method)             = {ceil_lp:.5f}")
sc = scalar_ceilings(PI)
print(f"\nno classifier on this world can exceed:")
print(f"  accuracy               {sc['accuracy']:.4f}   (closed form pi_3; "
      f"LP agrees: {sc['accuracy_lp']:.4f})")
print(f"  balanced accuracy      {sc['balanced_accuracy']:.4f}")
print(f"  worst-class Se         {sc['min_sensitivity']:.4f}")
print(f"  VUS                    {ceil_exact:.4f}   (chance {CHANCE_VUS:.4f}, "
      f"unconstrained max 1.0000)")

u, v, Z = ceiling_surface(PI, resolution=41)
print(f"\nthe ceiling as a surface Zmax(u,v) = best S3 given S1>=u, S2>=v:")
print(f"    {'':>8}" + "".join(f"{f'v={vv:.2f}':>9}" for vv in v[::10]))
for iu in range(0, len(u), 10):
    print(f"    u={u[iu]:.2f}" + "".join(f"{Z[iu, iv]:9.3f}" for iv in range(0, len(v), 10)))
print(f"    mean height = {Z.mean():.4f} ~ the ceiling above")

head("§6  Reading B: the gauge, and what it does and does not change")
X, y, P = make_gaussian_3class(300, 1.2, seed=1)
Q, c = to_ordered_gauge(P)
oc = check_order_constraint(Q)
s0 = roc_surface(y, P, resolution=200)
s1 = roc_surface(y, Q, resolution=200)
print(f"original model: constraint satisfied on "
      f"{check_order_constraint(P)['fraction_satisfied']:.1%} of rows")
print(f"after p_k -> c_k p_k with log c = "
      f"{np.round(np.log(np.clip(c, 1e-300, None)), 3)}:")
print(f"  constraint satisfied on {oc['fraction_satisfied']:.1%} of rows")
print(f"  geometric VUS   {s0.vus:.10f}  ->  {s1.vus:.10f}")
print(f"  3AFC VUS        {vus_forced_choice(y, P):.10f}  ->  "
      f"{vus_forced_choice(y, Q):.10f}")
print(f"  identical set of operating points: "
      f"{np.array_equal(np.unique(s0.S, axis=0), np.unique(s1.S, axis=0))}")

print("\nBut the PREDICTIONS at a fixed w do change -- the same rule now needs a")
print("different w.  Take the plain-argmax rule of the original model:")
op0 = s0.sensitivities_at(np.ones(3))
print(f"  original, w = (1, 1, 1)          -> Se = {np.round(op0, 4)}")
print(f"  regauged, w = (1, 1, 1)          -> Se = "
      f"{np.round(s1.sensitivities_at(np.ones(3)), 4)}   (a different rule!)")
w_equiv = 1.0 / np.clip(c, 1e-300, None)
w_equiv /= w_equiv.sum()
print(f"  regauged, w = 1/c = {np.round(w_equiv, 6)} -> Se = "
      f"{np.round(s1.sensitivities_at(w_equiv), 4)}   (the SAME rule again)")

head("§7  'It always predicts class 3, so surely it is useless'")
yp, Pp = make_perfect(200, confidence=0.999, seed=4)
Qp, _ = to_ordered_gauge(Pp)
pred = np.argmax(Qp, axis=1)
prev = np.bincount(yp, minlength=3) / len(yp)
sp = roc_surface(yp, Qp, resolution=200)
print(f"constrained model, argmax prediction for every sample: "
      f"{np.unique(pred)}  (only class 3)")
print(f"  accuracy of that default rule = {(pred == yp).mean():.4f} "
      f"= prevalence of class 3 = {prev[2]:.4f}")
print(f"  per-class sensitivity at w=(1,1,1) = "
      f"{np.round(sp.sensitivities_at(np.ones(3)), 4)}")
print(f"  ...and yet   VUS = {sp.vus:.6f}   3AFC = {vus_forced_choice(yp, Qp):.6f}")
print(f"  smallest p1 in the model output: {Qp[:, 0].min():.3e} (still ordered: "
      f"{check_order_constraint(Qp)['satisfied']})")
print("\nThe argmax point is ONE point on the surface; VUS is the whole surface.")
op = sp.select("youden")
print(f"  best rule on the surface: w = {np.round(op.weights, 6)} -> "
      f"Se = {np.round(op.sensitivities, 4)}")

head("§8  Which reading am I in?  The diagnostic")
for label, yy, PP in [("balanced priors + ordered outputs", y, Q),
                      ("the constrained world of section 1", None, None)]:
    if yy is None:
        rng = np.random.default_rng(0)
        idx = rng.choice(3, size=60_000, p=G)
        PP = POST[idx]
        u2 = rng.random(len(PP))[:, None]
        yy = (u2 > np.cumsum(PP, axis=1)).sum(axis=1)
    pri = np.bincount(yy, minlength=3) / len(yy)
    print(f"  {label}")
    print(f"     observed prevalences {np.round(pri, 4)}   ordered? "
          f"{pri[0] < pri[1] < pri[2]}")
    print(f"     -> Reading {'A is possible' if pri[0] < pri[1] < pri[2] else 'A is IMPOSSIBLE; must be B (gauge)'}")

head("§7 of the note  A caveat: coarse models starve the staircase estimator")
rng = np.random.default_rng(0)
idx = rng.choice(3, size=60_000, p=G)
Pw = POST[idx]
uu = rng.random(len(Pw))[:, None]
yw = (uu > np.cumsum(Pw, axis=1)).sum(axis=1)
pri = np.bincount(yw, minlength=3) / len(yw)
sw = roc_surface(yw, Pw, resolution=200)
cw = vus_ceiling(pri)
print(f"sampled {len(yw)} cases from the 3-atom world; realised priors "
      f"{np.round(pri, 4)}  (population {np.round(PI, 4)})")
print(f"\n  the ideal observer here IS the true posterior, so this is the best")
print(f"  any model could do on this world -- and yet:\n")
print(f"    geometric VUS (staircase)    {sw.vus:.4f}   <- BELOW chance "
      f"{CHANCE_VUS:.4f}")
print(f"    convex-hull VUS (randomised) {sw.vus_convex_hull:.4f}")
print(f"    3AFC VUS (rank-based)        {vus_forced_choice(yw, Pw):.4f}")
print(f"    ceiling from the priors      {cw:.4f}")
pts = np.unique(sw.S, axis=0)
print(f"\n  why: only {len(pts)} distinct operating points are reachable, and all but one")
print("  have a zero coordinate, so their boxes are flat and contribute no volume:")
for p in pts[np.lexsort((-pts[:, 2], -pts[:, 1], -pts[:, 0]))]:
    star = "   <- the only one with volume" if p.prod() > 0 else ""
    print(f"     ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f})   box {p.prod():.4f}{star}")
print("\n  the chance level of 1/6 assumes a CONTINUUM of operating points filling the")
print("  chance plane; three atoms cannot get near it.  It converges away with richness:")
from roc3.constrained import constrained_world  # noqa: E402
print(f"\n  {'atoms':>6} {'op. points':>11} {'VUS':>8} {'hull':>8} {'3AFC':>8} "
      f"{'ceiling':>9}")
for na in (3, 12, 60, 300):
    yy, PP, _, _ = constrained_world(n=60_000, n_atoms=na, concentration=0.35, seed=11)
    pr = np.bincount(yy, minlength=3) / len(yy)
    ss = roc_surface(yy, PP, resolution=200)
    print(f"  {na:>6} {len(np.unique(ss.S, axis=0)):>11} {ss.vus:8.3f} "
          f"{ss.vus_convex_hull:8.3f} {vus_forced_choice(yy, PP):8.3f} "
          f"{vus_ceiling(pr):9.3f}")
print("\n  Practical rule: for a model with few distinct score vectors, read VUS_3AFC or")
print("  vus_convex_hull, not the staircase.  A large gap between them IS the diagnosis.")

head("§5 of the note  Renormalising, on a rich constrained world")
yy, PP, _, _ = constrained_world(n=40_000, n_atoms=120, concentration=0.15, seed=5)
pr = np.bincount(yy, minlength=3) / len(yy)
ss = roc_surface(yy, PP, resolution=200)
cc = vus_ceiling(pr)
print(f"  realised priors {np.round(pr, 4)}")
print(f"  VUS                                     {ss.vus:.4f}")
print(f"  ceiling                                 {cc:.4f}")
print(f"  on the usual [1/6, 1] scale             "
      f"{(ss.vus - CHANCE_VUS) / (1 - CHANCE_VUS):.4f}")
print(f"  renormalised onto [chance, ceiling]     "
      f"{normalized_vus(ss.vus, pr, ceiling=cc):.4f}")
