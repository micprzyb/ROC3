#!/usr/bin/env python
"""
Threshold-selection case study: using the surface to actually decide something.

The whole point of a ROC curve is that you can put your finger on it and read off a
threshold.  This script does that for the 3-class surface, end to end:

  Q1  What does the default (plain argmax) rule do?
  Q2  I must not miss malignant cases.  Is Se >= 0.95 achievable, and at what price?
  Q3  Plot the *price of a guarantee*: how the other two classes degrade as the floor rises.
  Q4  I have a cost matrix instead of a hard floor -- what does it choose?
  Q5  How uncertain is the operating point I picked?
  Q6  Model A vs model B: is the difference in VUS real?

    .venv/bin/python experiments/03_threshold_case_study.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FIG = ROOT / "figures"

from roc3 import achievable, roc_surface, vus_forced_choice  # noqa: E402
from roc3.datasets import make_asymmetric_clinical  # noqa: E402
from roc3.inference import bootstrap_operating_point, vus_3afc_inference  # noqa: E402
from roc3.plots import THEMES, _style_axes  # noqa: E402


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


X, y_idx, P, names = make_asymmetric_clinical(n=(600, 400, 200), seed=0)
y = names[y_idx]
surf = roc_surface(y, P, classes=names, resolution=300)
print(f"model: 3-class clinical triage, n = {list(surf.counts)}  "
      f"({', '.join(names)})")
print(f"VUS = {surf.vus:.4f}   chance-corrected = {surf.vus_adjusted:+.4f}   "
      f"(chance 0.167, perfect 1.000)")

# --------------------------------------------------------------------------------------
banner("Q1 · the default rule")
# --------------------------------------------------------------------------------------
cm = surf.confusion_at(np.ones(3))
se = np.diag(cm) / cm.sum(axis=1)
print(f"plain argmax  ->  Se = " + ",  ".join(f"{n} {v:.3f}" for n, v in zip(names, se)))
print(f"              accuracy {np.trace(cm) / cm.sum():.3f}   "
      f"balanced accuracy {se.mean():.3f}")
print("Note the rare, dangerous class is the worst served: argmax quietly optimises")
print("accuracy, which is dominated by the common class.")

# --------------------------------------------------------------------------------------
banner("Q2 · hard requirement: Se[malignant] >= 0.95")
# --------------------------------------------------------------------------------------
req = {"malignant": 0.95}
print(f"achievable at all?  {achievable(surf, req)}")
op = surf.select("balanced_accuracy", constraints=req)
print(op.describe())
best_unconstrained = surf.select("balanced_accuracy")
print(f"price of the guarantee: balanced accuracy "
      f"{best_unconstrained.sensitivities.mean():.4f} -> "
      f"{op.sensitivities.mean():.4f}  "
      f"({op.sensitivities.mean() - best_unconstrained.sensitivities.mean():+.4f})")

# --------------------------------------------------------------------------------------
banner("Q3 · the price of a guarantee, as a curve")
# --------------------------------------------------------------------------------------
floors = np.round(np.arange(0.60, 0.995, 0.025), 3)
rows = []
for f in floors:
    if not achievable(surf, {"malignant": f}):
        break
    o = surf.select("balanced_accuracy", constraints={"malignant": f})
    rows.append((f, *o.sensitivities, o.sensitivities.mean(), *o.weights))
rows = np.array(rows)
print(f"  {'floor':>6}  {'Se benign':>10}  {'Se indet':>9}  {'Se malig':>9}"
      f"  {'bal.acc':>8}   weights (relative)")
for r in rows:
    w = r[5:8] / r[5:8].max()
    print(f"  {r[0]:6.3f}  {r[1]:10.3f}  {r[2]:9.3f}  {r[3]:9.3f}  {r[4]:8.3f}"
          f"   {w[0]:.2f} / {w[1]:.2f} / {w[2]:.2f}")

th = THEMES["light"]
fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8), facecolor=th["page"])
ax = axes[0]
for k, nm in enumerate(names):
    ax.plot(rows[:, 0], rows[:, 1 + k], color=th["series"][k], linewidth=2.0,
            label=f"Se[{nm}]")
ax.plot(rows[:, 0], rows[:, 4], color=th["muted"], linewidth=1.6, linestyle="-",
        label="balanced accuracy")
_style_axes(ax, th)
ax.set_xlabel("required floor on Se[malignant]", fontsize=9)
ax.set_ylabel("achieved value", fontsize=9)
ax.set_ylim(0, 1)
ax.set_title("The price of a guarantee\nbest rule subject to a floor on the rare class",
             fontsize=11, color=th["ink"], pad=8)
leg = ax.legend(loc="lower left", fontsize=8.5, frameon=True,
                facecolor=th["surface"], edgecolor=th["grid"])
for t in leg.get_texts():
    t.set_color(th["ink2"])
# direct-label the endpoint of each series (relief for the sub-3:1 aqua)
for k, nm in enumerate(names):
    ax.annotate(f"{rows[-1, 1 + k]:.2f}", (rows[-1, 0], rows[-1, 1 + k]),
                textcoords="offset points", xytext=(4, 0), fontsize=8,
                color=th["ink2"], va="center")

ax = axes[1]
rel = rows[:, 5:8] / rows[:, 5:8].max(axis=1, keepdims=True)
for k, nm in enumerate(names):
    ax.plot(rows[:, 0], rel[:, k], color=th["series"][k], linewidth=2.0,
            label=f"$w$[{nm}]")
_style_axes(ax, th)
ax.set_xlabel("required floor on Se[malignant]", fontsize=9)
ax.set_ylabel("relative weight (max = 1)", fontsize=9)
ax.set_yscale("log")
ax.set_title("...and the rule that delivers it\nclass weights to apply before the argmax",
             fontsize=11, color=th["ink"], pad=8)
leg = ax.legend(loc="lower left", fontsize=8.5, frameon=True,
                facecolor=th["surface"], edgecolor=th["grid"])
for t in leg.get_texts():
    t.set_color(th["ink2"])
fig.tight_layout()
fig.savefig(FIG / "07_price_of_a_guarantee.png", dpi=150, facecolor=th["page"],
            bbox_inches="tight")
print(f"\nwrote {FIG / '07_price_of_a_guarantee.png'}")

# --------------------------------------------------------------------------------------
banner("Q4 · a cost matrix instead of a floor")
# --------------------------------------------------------------------------------------
costs = np.array([[0.0, 1.0, 2.0],
                  [2.0, 0.0, 3.0],
                  [20.0, 8.0, 0.0]])
print("cost[true, predicted] =\n", costs)
op_cost = surf.select("expected_cost", costs=costs)
print(op_cost.describe())
exp_cost = -op_cost.objective
base_cost = -np.einsum("k,ki,ki->", surf.counts / surf.counts.sum(),
                       surf.confusion_at(np.ones(3))
                       / surf.counts[:, None], -costs)
print(f"expected cost per case: plain argmax {base_cost:.4f}  ->  "
      f"optimised {exp_cost:.4f}   ({100 * (base_cost - exp_cost) / base_cost:+.1f}%)")
print("Note this criterion is the only one that can see *which* wrong class was chosen;")
print("the surface itself only shows the diagonal, so we read the off-diagonals from the")
print("stored confusion matrices.")

# --------------------------------------------------------------------------------------
banner("Q5 · how uncertain is the chosen operating point?")
# --------------------------------------------------------------------------------------
bci = bootstrap_operating_point(y, P, op.weights, classes=names, n_boot=2000, seed=0)
for k, nm in enumerate(names):
    print(f"  Se[{nm:<14}] = {bci['sensitivities'][k]:.3f}   "
          f"95% CI ({bci['ci_lower'][k]:.3f}, {bci['ci_upper'][k]:.3f})")
print("The rule is held fixed here, so this is the uncertainty of the operating point")
print("you would deploy -- it does not include the optimism from having *chosen* the")
print("weights on this same sample.")

inf = vus_3afc_inference(y, P, classes=names)
print(f"\n  VUS (3AFC) = {inf['vus']:.4f}   se {inf['stderr']:.4f}   "
      f"95% CI ({inf['ci'][0]:.4f}, {inf['ci'][1]:.4f})")

# --------------------------------------------------------------------------------------
banner("Q6 · model A vs model B — is the VUS difference real?")
# --------------------------------------------------------------------------------------
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

cv = StratifiedKFold(5, shuffle=True, random_state=0)
PA = cross_val_predict(LinearDiscriminantAnalysis(), X, y_idx, cv=cv,
                       method="predict_proba")
PB = cross_val_predict(RandomForestClassifier(n_estimators=300, min_samples_leaf=8,
                                              random_state=0),
                       X, y_idx, cv=cv, method="predict_proba")
vA, vB = vus_forced_choice(y_idx, PA), vus_forced_choice(y_idx, PB)
print(f"  LDA           VUS = {vA:.4f}")
print(f"  RandomForest  VUS = {vB:.4f}")

rng = np.random.default_rng(0)
idx_by_class = [np.flatnonzero(y_idx == k) for k in range(3)]
d = np.empty(400)
for b in range(len(d)):
    take = np.concatenate([rng.choice(ix, len(ix), replace=True) for ix in idx_by_class])
    d[b] = (vus_forced_choice(y_idx[take], PB[take], method="mc", n_samples=40_000)
            - vus_forced_choice(y_idx[take], PA[take], method="mc", n_samples=40_000))
lo, hi = np.quantile(d, [0.025, 0.975])
print(f"  paired stratified bootstrap of the difference (B - A), 400 replicates:")
print(f"    mean {d.mean():+.4f}   95% CI ({lo:+.4f}, {hi:+.4f})   "
      f"{'-> difference is significant' if lo * hi > 0 else '-> not significant'}")
print("\nPairing matters: both models are scored on the same resampled cases, so the")
print("common sampling noise cancels and the interval is much tighter than two")
print("independent intervals would suggest.")
