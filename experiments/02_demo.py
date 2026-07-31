#!/usr/bin/env python
"""
End-to-end demos: generate every figure in ``figures/``.

    .venv/bin/python experiments/02_demo.py
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
FIG.mkdir(exist_ok=True)

from roc3 import roc_surface  # noqa: E402
from roc3.datasets import (make_asymmetric_clinical, make_gaussian_3class,  # noqa: E402
                           make_ordinal_3class, wine_probabilities)
from roc3.plots import dashboard, interactive_surface, ordinal_dashboard  # noqa: E402


def banner(t):
    print("\n" + "=" * 78 + f"\n{t}\n" + "=" * 78)


# --------------------------------------------------------------------------------------
banner("DEMO 1 · asymmetric clinical problem (the main example)")
# --------------------------------------------------------------------------------------
X, y, P, names = make_asymmetric_clinical(n=(600, 400, 200), seed=0)
y = names[y]

fig, out = dashboard(
    y, P, classes=names, criterion="youden", resolution=240, lattice=110,
    title="3-class ROC surface — synthetic clinical triage "
          "(benign / indeterminate / malignant)",
    path=FIG / "01_clinical_youden.png",
)
print(out["operating_point"].describe())

# The same model, but the deployment requirement is "never miss a malignant".
fig2, out2 = dashboard(
    y, P, classes=names, criterion="balanced_accuracy",
    constraints={"malignant": 0.95}, resolution=240, lattice=110,
    title="Same model, hard requirement: Se[malignant] ≥ 0.95",
    path=FIG / "02_clinical_constrained.png",
)
print(out2["operating_point"].describe())

# Cost-driven: missing a malignant costs 20x a false alarm.
costs = np.array([[0.0, 1.0, 2.0],       # true benign        -> predicted ...
                  [2.0, 0.0, 3.0],       # true indeterminate
                  [20.0, 8.0, 0.0]])     # true malignant
fig3, out3 = dashboard(
    y, P, classes=names, criterion="expected_cost", costs=costs,
    resolution=240, lattice=110,
    title="Same model, cost matrix (missing a malignant costs 20×)",
    path=FIG / "03_clinical_cost.png",
)
print(out3["operating_point"].describe())

surf = out["surface"]
interactive_surface(surf, path=str(FIG / "01_clinical_interactive.html"),
                    title="Clinical triage — 3-class ROC surface")
print(f"wrote {FIG / '01_clinical_interactive.html'}")

# --------------------------------------------------------------------------------------
banner("DEMO 2 · real data: sklearn 'wine', out-of-fold logistic-regression scores")
# --------------------------------------------------------------------------------------
yw, Pw, wnames = wine_probabilities(seed=0)
fig4, out4 = dashboard(
    wnames[yw], Pw, classes=wnames, criterion="maximin", resolution=240, lattice=110,
    title="3-class ROC surface — UCI wine, out-of-fold logistic regression on two "
          "features (alcohol, alcalinity of ash)",
    path=FIG / "04_wine.png",
)
print(out4["operating_point"].describe())

# --------------------------------------------------------------------------------------
banner("DEMO 3 · ordered classes, single marker (Mossman / Nakas mode)")
# --------------------------------------------------------------------------------------
yo, so = make_ordinal_3class(n_per_class=300, separation=1.35, seed=3)
stage = np.array(["stage I", "stage II", "stage III"])
fig5, out5 = ordinal_dashboard(
    stage[yo], so, classes=stage, criterion="youden", resolution=320,
    title="Ordered 3-class ROC surface — one biomarker, two cut-offs",
    path=FIG / "05_ordinal.png",
)
print(out5["operating_point"].describe())

# --------------------------------------------------------------------------------------
banner("DEMO 4 · what the surface looks like as the model improves")
# --------------------------------------------------------------------------------------
import matplotlib.pyplot as plt  # noqa: E402

from roc3.plots import THEMES, plot_surface_3d  # noqa: E402

th = THEMES["light"]
seps = [0.4, 0.9, 1.5, 2.4]
fig6 = plt.figure(figsize=(19, 5.2), facecolor=th["page"])
for i, sep in enumerate(seps):
    _, ys, Ps = make_gaussian_3class(400, sep, seed=1)
    s = roc_surface(ys, Ps, resolution=200)
    ax = fig6.add_subplot(1, len(seps), i + 1, projection="3d")
    plot_surface_3d(ax, s, th, resolution=61)
    ax.set_facecolor(th["page"])
    ax.set_title(f"separation {sep:.1f}\nVUS {s.vus:.3f}   "
                 f"chance-corrected {s.vus_adjusted:+.3f}",
                 fontsize=10, color=th["ink"], pad=10)
fig6.suptitle("The surface inflates from the chance plane to the perfect corner "
              "as the model improves",
              fontsize=14, color=th["ink"], x=0.02, ha="left", y=0.99)
fig6.savefig(FIG / "06_signal_sweep.png", dpi=140, facecolor=th["page"],
             bbox_inches="tight")
print(f"wrote {FIG / '06_signal_sweep.png'}")

print("\nAll figures written to", FIG)
for p in sorted(FIG.glob("*")):
    if not p.name.startswith("_"):
        print(f"  {p.name}  ({p.stat().st_size / 1024:.0f} kB)")
