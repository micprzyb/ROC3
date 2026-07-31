#!/usr/bin/env python
"""
Every number quoted in ``docs/TUTORIAL.md``, recomputed from scratch.

The tutorial uses one nine-sample toy dataset throughout so that each quantity can be
checked by hand.  Run this to regenerate (and verify) all of it:

    .venv/bin/python experiments/05_tutorial_numbers.py
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from roc3 import (binary_auc, forced_choice_profile, hypervolume3d,  # noqa: E402
                  roc_surface, vus_forced_choice)
from roc3.core import monotone_envelope  # noqa: E402

np.set_printoptions(suppress=True, precision=4, linewidth=120)


def head(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# --------------------------------------------------------------------------------------
# The toy dataset used everywhere in the tutorial
# --------------------------------------------------------------------------------------
NAMES = [f"x{i}" for i in range(1, 10)]
Y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 3])              # true class of each sample
P = np.array([
    [0.70, 0.20, 0.10],    # x1  true 1
    [0.55, 0.30, 0.15],    # x2  true 1
    [0.30, 0.45, 0.25],    # x3  true 1   <- argmax says 2 (wrong)
    [0.20, 0.60, 0.20],    # x4  true 2
    [0.35, 0.40, 0.25],    # x5  true 2
    [0.15, 0.35, 0.50],    # x6  true 2   <- argmax says 3 (wrong)
    [0.10, 0.20, 0.70],    # x7  true 3
    [0.25, 0.25, 0.50],    # x8  true 3
    [0.30, 0.40, 0.30],    # x9  true 3   <- argmax says 2 (wrong)
])

head("§1  The toy dataset")
print(f"{'sample':>7} {'true':>5} {'p1':>7} {'p2':>7} {'p3':>7}   argmax")
for i, nm in enumerate(NAMES):
    am = int(np.argmax(P[i])) + 1
    tag = "" if am == Y[i] else "   <- wrong"
    print(f"{nm:>7} {Y[i]:>5} {P[i, 0]:7.2f} {P[i, 1]:7.2f} {P[i, 2]:7.2f}   {am}{tag}")
print(f"\nrow sums: {P.sum(axis=1)}")
print(f"class counts n1,n2,n3 = {np.bincount(Y, minlength=4)[1:]}")
acc = (np.argmax(P, axis=1) + 1 == Y).mean()
print(f"plain-argmax accuracy = {int(acc * 9)}/9 = {acc:.4f}")

head("§2  log scores, and the three pairwise log-ratios")
L = np.log(P)
D12, D13, D23 = L[:, 0] - L[:, 1], L[:, 0] - L[:, 2], L[:, 1] - L[:, 2]
print(f"{'sample':>7} {'log p1':>8} {'log p2':>8} {'log p3':>8}"
      f" {'D12':>8} {'D13':>8} {'D23':>8}")
for i, nm in enumerate(NAMES):
    print(f"{nm:>7} {L[i, 0]:8.4f} {L[i, 1]:8.4f} {L[i, 2]:8.4f}"
          f" {D12[i]:8.4f} {D13[i]:8.4f} {D23[i]:8.4f}")
print("\ncheck D13 = D12 + D23:", np.allclose(D13, D12 + D23))

head("§3  The weighted-argmax rule, applied by hand")
for w in [np.array([1.0, 1.0, 1.0]),
          np.array([2.0, 1.0, 1.0]),
          np.array([1.0, 1.0, 2.0]),
          np.array([5.0, 1.0, 0.5])]:
    wp = P * w
    pred = np.argmax(wp, axis=1) + 1
    print(f"\nw = {w}")
    print(f"{'sample':>7} {'w1*p1':>8} {'w2*p2':>8} {'w3*p3':>8}  pred  true")
    for i, nm in enumerate(NAMES):
        print(f"{nm:>7} {wp[i, 0]:8.3f} {wp[i, 1]:8.3f} {wp[i, 2]:8.3f}"
              f" {pred[i]:5d} {Y[i]:5d}")
    cm = np.zeros((3, 3), dtype=int)
    np.add.at(cm, (Y - 1, pred - 1), 1)
    S = np.diag(cm) / cm.sum(axis=1)
    print(f"  confusion (rows=true, cols=predicted):\n{cm}")
    print(f"  operating point (S1,S2,S3) = {np.round(S, 4)}"
          f"    sum = {S.sum():.4f}")

head("§4  Binary warm-up: AUC two ways, by hand")
yb = np.array([0, 0, 1, 1])
sb = np.array([0.10, 0.60, 0.40, 0.80])
print(f"labels {yb}, scores {sb}")
pos, neg = sb[yb == 1], sb[yb == 0]
pairs = [(p, n, p > n) for p in pos for n in neg]
print("\nall positive/negative pairs, and who scores higher:")
for p, n, win in pairs:
    print(f"  positive {p:.2f}  vs  negative {n:.2f}   positive wins: {win}")
frac = np.mean([w for _, _, w in pairs])
print(f"  fraction of pairs the positive wins = {frac}")
print(f"  binary_auc()                        = {binary_auc(pos, neg)}")

cuts = np.concatenate(([-np.inf], np.sort(np.unique(sb)), [np.inf]))
print("\nthreshold sweep -> operating points (Se0 = specificity, Se1 = sensitivity):")
pts = []
for t in cuts:
    pred = (sb > t).astype(int)
    se1 = (pred[yb == 1] == 1).mean()
    se0 = (pred[yb == 0] == 0).mean()
    pts.append((se0, se1))
    print(f"  t = {t:6.2f}   predict 1 if score > t   Se0 = {se0:.2f}   Se1 = {se1:.2f}")
pts3 = np.array([[a, b, 1.0] for a, b in pts])
print(f"\n  area of the down-set of those points = {hypervolume3d(pts3):.4f}")
print(f"  ... which equals the AUC above        = {binary_auc(pos, neg):.4f}")

head("§5  The 3-class surface for the toy dataset: every distinct operating point")
surf = roc_surface(Y, P, classes=[1, 2, 3], resolution=200)
uniq, idx = np.unique(surf.S, axis=0, return_index=True)
order = np.lexsort((-uniq[:, 2], -uniq[:, 1], -uniq[:, 0]))
print(f"the rule grid contains {len(surf.S)} weight vectors, which produce "
      f"{len(uniq)} distinct operating points:\n")
print(f"{'S1':>6} {'S2':>6} {'S3':>6}   {'sum':>6}   example w (normalised)")
for r in order:
    s = uniq[r]
    w = surf.W[idx[r]]
    print(f"{s[0]:6.3f} {s[1]:6.3f} {s[2]:6.3f}   {s.sum():6.3f}   "
          f"({w[0]:.3f}, {w[1]:.3f}, {w[2]:.3f})")

head("§6  VUS as a volume, computed by hand")
pareto = uniq[surf.pareto_mask()[idx][order]] if False else None
from roc3.core import _pareto_mask  # noqa: E402
pm = _pareto_mask(uniq)
front = uniq[pm]
print("Pareto-optimal operating points (nothing else beats them on all three axes):")
for s in front[np.lexsort((-front[:, 2], -front[:, 1], -front[:, 0]))]:
    print(f"  ({s[0]:.3f}, {s[1]:.3f}, {s[2]:.3f})   box volume "
          f"{s[0] * s[1] * s[2]:.4f}")
print(f"\nsum of the individual box volumes (over-counts overlaps): "
      f"{sum(s.prod() for s in front):.4f}")
print(f"volume of their UNION (the down-set)  = VUS = {surf.vus:.4f}")
print(f"chance-corrected VUS_adj              = {surf.vus_adjusted:.4f}")

_, _, Z = monotone_envelope(surf.S, resolution=601)
print(f"cross-check by integrating the envelope Z(u,v): mean(Z) = {Z.mean():.4f}")

head("§7  The 3-alternative forced choice, fully worked")
by_class = {k: [i for i in range(9) if Y[i] == k] for k in (1, 2, 3)}
print("A 'trio' is one sample drawn from each class.  There are "
      f"{len(by_class[1])} x {len(by_class[2])} x {len(by_class[3])} = "
      f"{len(by_class[1]) * len(by_class[2]) * len(by_class[3])} of them here.\n")
perms = list(itertools.permutations([1, 2, 3]))

for trio in [(0, 3, 6), (2, 4, 8)]:                    # (x1,x4,x7) and (x3,x5,x9)
    a, b, c = trio
    print("-" * 74)
    print(f"TRIO: {NAMES[a]} (truly class 1), {NAMES[b]} (truly class 2), "
          f"{NAMES[c]} (truly class 3)")
    print("  We are told these three are one of each class, but NOT which is which.")
    print("  We must hand out the three labels, one each.  Six ways to do it.\n")
    print(f"  log-score table  log p_k(x)  (this is the 'key' we sort on):")
    print(f"    {'':>6} {'k=1':>9} {'k=2':>9} {'k=3':>9}")
    for nm_i in trio:
        print(f"    {NAMES[nm_i]:>6} {L[nm_i, 0]:9.4f} {L[nm_i, 1]:9.4f} "
              f"{L[nm_i, 2]:9.4f}")
    print()
    scores = []
    for sg in perms:
        s = L[a, sg[0] - 1] + L[b, sg[1] - 1] + L[c, sg[2] - 1]
        scores.append(s)
    best = int(np.argmax(scores))
    for t, (sg, s) in enumerate(zip(perms, scores)):
        tag = "   <-- WINNER" if t == best else ""
        truth = "  (this is the truth)" if sg == (1, 2, 3) else ""
        print(f"    give {NAMES[a]}->{sg[0]}, {NAMES[b]}->{sg[1]}, {NAMES[c]}->{sg[2]}"
              f"   total log-score {s:8.4f}"
              f"   product {np.exp(s):.6f}{tag}{truth}")
    print(f"\n  => the trio is sorted {'CORRECTLY' if perms[best] == (1, 2, 3) else 'WRONGLY'}"
          f"; the winning assignment is {perms[best]}")

print("-" * 74)
n_ok, n_tot = 0, 0
detail = []
for a in by_class[1]:
    for b in by_class[2]:
        for c in by_class[3]:
            sc = [L[a, sg[0] - 1] + L[b, sg[1] - 1] + L[c, sg[2] - 1] for sg in perms]
            win = perms[int(np.argmax(sc))]
            n_ok += win == (1, 2, 3)
            n_tot += 1
            detail.append((NAMES[a], NAMES[b], NAMES[c], win))
print(f"\nOver all {n_tot} trios the correct assignment wins {n_ok} times.")
print(f"VUS_3AFC = {n_ok}/{n_tot} = {n_ok / n_tot:.4f}")
print(f"library vus_forced_choice()         = {vus_forced_choice(Y, P, classes=[1,2,3]):.4f}")
print(f"geometric VUS from the surface      = {surf.vus:.4f}")

print("\nthe six assignments, and how often each wins (the '3AFC profile'):")
fc = forced_choice_profile(Y, P, classes=[1, 2, 3], method="exact")
for t in np.argsort(-fc.profile):
    sg = perms[t]
    tag = "   <- correct" if sg == (1, 2, 3) else ""
    print(f"  (class1 case, class2 case, class3 case) -> labels {sg}"
          f"   wins {fc.profile[t]:.4f} of the time{tag}")
print(f"  the six probabilities sum to {fc.profile.sum():.6f}")

print("\ntrios the model gets WRONG:")
for a, b, c, win in detail:
    if win != (1, 2, 3):
        print(f"  ({a}, {b}, {c})  ->  labelled {win}")

head("§8  The five algebraic conditions, checked against the brute force")
A, B, C = L[:, 0] - L[:, 1], L[:, 0] - L[:, 2], L[:, 1] - L[:, 2]
print(f"{'sample':>7} {'A=logp1-logp2':>15} {'B=logp1-logp3':>15} {'C=logp2-logp3':>15}")
for i, nm in enumerate(NAMES):
    print(f"{nm:>7} {A[i]:15.4f} {B[i]:15.4f} {C[i]:15.4f}")
ok = 0
for a in by_class[1]:
    for b in by_class[2]:
        for c in by_class[3]:
            cond = (A[a] > A[b], C[b] > C[c], B[a] > B[c],
                    A[a] + C[b] > B[c], B[a] > A[b] + C[c])
            ok += all(cond)
print(f"\ntrios satisfying all five conditions: {ok}/{n_tot} = {ok / n_tot:.4f}")
print(f"brute-force answer from §7:            {n_ok}/{n_tot} = {n_ok / n_tot:.4f}")
print("agree:", ok == n_ok)

head("§9  Choosing an operating point")
for crit in ("youden", "maximin", "closest_to_perfection", "accuracy", "max_volume"):
    op = surf.select(crit)
    print(f"{crit:>22}  S = {np.round(op.sensitivities, 3)}   "
          f"w = {np.round(op.weights, 3)}")
print()
op = surf.select("maximin", constraints={3: 0.99})
print(op.describe())

head("§10  A worked partial VUS and a constrained query")
print(f"partial VUS over the box [0.5,1]^3 (normalised) = "
      f"{surf.partial_vus((0.5, 0.5, 0.5)):.4f}")
print(f"VUS with randomised rules allowed (convex hull) = {surf.vus_convex_hull:.4f}")
print(f"head-room from re-thresholding alone            = "
      f"{surf.vus_convex_hull - surf.vus:+.4f}")
