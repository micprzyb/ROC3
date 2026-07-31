"""
roc3.ordinal
============

The **ordered** three-class ROC surface (Mossman 1999; Nakas & Yiannoutsos 2004).

When the three classes are naturally ordered -- mild < moderate < severe, stage I < II <
III, reject < review < approve -- and a *single* continuous marker ``s(x)`` drives the
decision, the natural rule family is two cut-points

    y_hat = 1  if  s < t1
          = 2  if  t1 <= s < t2
          = 3  if  s >= t2                       (t1 <= t2)

and the operating point is again the triple of true class fractions

    S1 = P(s < t1 | y = 1),   S2 = P(t1 <= s < t2 | y = 2),   S3 = P(s >= t2 | y = 3).

This is a genuinely different (and, for ordinal problems, more directly actionable)
parameterisation than the weighted-argmax family in :mod:`roc3.core`: the two free
parameters are *literal cut-off values on the marker*, so reading a point off the surface
hands you two numbers you can put straight into production.

The rank-based volume for this family has the classical closed form

    VUS = P( s(X1) < s(X2) < s(X3) )

with 1/6 for an uninformative marker and 1 for a perfect one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .core import CHANCE_VUS, hypervolume3d, monotone_envelope
from .thresholds import CRITERIA, objective_values

__all__ = ["OrdinalSurface", "ordinal_roc_surface", "vus_ordinal", "OrdinalOperatingPoint"]


def vus_ordinal(y, marker, *, classes=None, direction="increasing") -> float:
    """``P(s1 < s2 < s3)`` with the usual 1/2 credit for pairwise ties.

    ``O(n log n)``: for every class-2 value ``v`` count the class-1 values below and the
    class-3 values above, then average the product.
    """
    y = np.asarray(y)
    s = np.asarray(marker, dtype=float)
    classes = np.unique(y) if classes is None else np.asarray(classes)
    if len(classes) != 3:
        raise ValueError("vus_ordinal needs exactly 3 ordered classes")
    if direction not in ("increasing", "decreasing"):
        raise ValueError("direction must be 'increasing' or 'decreasing'")
    if direction == "decreasing":
        s = -s

    s1 = np.sort(s[y == classes[0]])
    s2 = s[y == classes[1]]
    s3 = np.sort(s[y == classes[2]])
    n1, n2, n3 = len(s1), len(s2), len(s3)
    if min(n1, n2, n3) == 0:
        raise ValueError("every class needs at least one observation")

    lo = np.searchsorted(s1, s2, side="left")                     # strictly below
    lo_t = np.searchsorted(s1, s2, side="right") - lo             # ties
    hi = n3 - np.searchsorted(s3, s2, side="right")               # strictly above
    hi_t = np.searchsorted(s3, s2, side="right") - np.searchsorted(s3, s2, side="left")

    left = lo + 0.5 * lo_t
    right = hi + 0.5 * hi_t
    return float(np.mean(left * right) / (n1 * n3))


@dataclass
class OrdinalOperatingPoint:
    index: int
    thresholds: tuple
    sensitivities: np.ndarray
    confusion: np.ndarray
    criterion: str
    objective: float
    classes: np.ndarray
    constraints: dict
    direction: str = "increasing"

    @property
    def rates(self) -> np.ndarray:
        return self.confusion / self.confusion.sum(axis=1, keepdims=True)

    def describe(self, digits: int = 4) -> str:
        from .metrics import confusion_report

        cls = list(self.classes)
        rep = confusion_report(self.confusion, cls)
        t1, t2 = self.thresholds
        if self.direction == "increasing":
            rule = (f"{cls[0]} if s < {t1:.6g};  {cls[1]} if {t1:.6g} <= s < {t2:.6g};  "
                    f"{cls[2]} if s >= {t2:.6g}")
        else:
            rule = (f"{cls[0]} if s > {t2:.6g};  {cls[1]} if {t1:.6g} < s <= {t2:.6g};  "
                    f"{cls[2]} if s <= {t1:.6g}")
        lines = ["-" * 66,
                 f"ORDINAL OPERATING POINT  ({self.criterion}"
                 + (f", constraints={self.constraints}" if self.constraints else "") + ")",
                 "-" * 66,
                 f"Decision rule:  {rule}",
                 ""]
        lines.append("Confusion (rows = true class):")
        lines.append("            " + "".join(f"{('->' + str(c)):>10}" for c in cls) + "     Se")
        for k, c in enumerate(cls):
            row = "".join(f"{self.confusion[k, i]:>10d}" for i in range(3))
            lines.append(f"  true {str(c):<6}{row}   {rep['sensitivity'][k]:.{digits}f}")
        lines.append("")
        lines.append(f"  accuracy {rep['accuracy']:.{digits}f}"
                     f"   balanced accuracy {rep['balanced_accuracy']:.{digits}f}"
                     f"   macro-F1 {rep['macro_f1']:.{digits}f}")
        lines.append(f"  objective ({self.criterion}) = {self.objective:.{digits}f}")
        lines.append("-" * 66)
        return "\n".join(lines)


@dataclass
class OrdinalSurface:
    """Two-cut-point ROC surface for ordered classes."""

    T1: np.ndarray                 # (G,) lower cut-point
    T2: np.ndarray                 # (G,) upper cut-point, >= T1
    S: np.ndarray                  # (G, 3) true class fractions
    CM: np.ndarray                 # (G, 3, 3) raw confusion counts
    classes: np.ndarray
    counts: np.ndarray
    marker: np.ndarray = field(repr=False, default=None)
    y_idx: np.ndarray = field(repr=False, default=None)
    direction: str = "increasing"

    @property
    def rates(self) -> np.ndarray:
        return self.CM / self.counts[None, :, None]

    @property
    def vus(self) -> float:
        """Geometric VUS (down-set hypervolume) of this rule family."""
        return hypervolume3d(self.S)

    @property
    def vus_adjusted(self) -> float:
        return (self.vus - CHANCE_VUS) / (1.0 - CHANCE_VUS)

    def envelope(self, resolution: int = 201):
        return monotone_envelope(self.S, resolution=resolution)

    def select(self, criterion="youden", *, constraints=None, costs=None, priors=None):
        if criterion not in CRITERIA:
            raise ValueError(f"criterion must be one of {CRITERIA}")
        mask = np.ones(len(self.S), dtype=bool)
        for key, floor in (constraints or {}).items():
            k = int(np.flatnonzero(self.classes == key)[0]) if key in self.classes else int(key)
            mask &= self.S[:, k] >= float(floor)
        if not mask.any():
            raise ValueError(f"constraints {constraints} are not achievable")
        obj = objective_values(self.S, self.rates, self.counts, criterion,
                               priors=priors, costs=costs)
        scored = np.where(mask, obj, -np.inf)
        best = float(scored.max())
        cand = np.flatnonzero(scored >= best - 1e-12)
        idx = int(cand[np.argmax(self.S[cand].min(axis=1))])
        return OrdinalOperatingPoint(
            index=idx,
            thresholds=(float(self.T1[idx]), float(self.T2[idx])),
            sensitivities=self.S[idx].copy(),
            confusion=self.CM[idx].copy(),
            criterion=criterion,
            objective=float(obj[idx]),
            classes=self.classes,
            constraints=dict(constraints or {}),
            direction=self.direction,
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"OrdinalSurface(n_rules={len(self.S)}, classes={list(self.classes)}, "
                f"VUS={self.vus:.4f})")


def ordinal_roc_surface(
    y,
    marker,
    *,
    classes=None,
    direction="increasing",
    resolution: int = 200,
    max_cells: int = 4_000_000,
) -> OrdinalSurface:
    """Build the two-cut-point ROC surface for an ordered 3-class problem.

    Parameters
    ----------
    y : array-like
        Labels; ``classes`` (default ``np.unique(y)``) must be given in *increasing*
        severity order.
    marker : array-like
        The continuous score.  ``direction='increasing'`` means larger values indicate
        later classes.
    resolution : int
        Number of candidate cut-points per axis.  The rule grid is the upper triangle
        ``t1 <= t2``, so roughly ``resolution**2 / 2`` rules.
    """
    y = np.asarray(y)
    s = np.asarray(marker, dtype=float).ravel()
    if len(y) != len(s):
        raise ValueError("y and marker must have the same length")
    classes = np.unique(y) if classes is None else np.asarray(classes)
    if len(classes) != 3:
        raise ValueError("ordinal_roc_surface needs exactly 3 ordered classes")
    if direction == "decreasing":
        s = -s
    elif direction != "increasing":
        raise ValueError("direction must be 'increasing' or 'decreasing'")

    lookup = {c: i for i, c in enumerate(classes)}
    y_idx = np.array([lookup[v] for v in y], dtype=np.intp)
    counts = np.bincount(y_idx, minlength=3)
    if np.any(counts == 0):
        raise ValueError("every class needs at least one observation")

    u = np.unique(s)
    if u.size > resolution:
        u = np.unique(np.quantile(u, np.linspace(0.0, 1.0, resolution)))
    span = max(np.ptp(u), 1.0) if u.size else 1.0
    cuts = np.concatenate((
        [u[0] - 0.05 * span],
        (u[:-1] + u[1:]) / 2.0 if u.size > 1 else np.empty(0),
        [u[-1] + 0.05 * span],
    ))

    i1, i2 = np.triu_indices(len(cuts), k=0)      # t1 <= t2
    T1, T2 = cuts[i1], cuts[i2]
    G = len(T1)

    by_class = [s[y_idx == k] for k in range(3)]
    CM = np.empty((G, 3, 3), dtype=np.int64)
    chunk = max(1, int(max_cells // max(1, int(counts.max()))))
    for lo in range(0, G, chunk):
        hi = min(G, lo + chunk)
        t1 = T1[lo:hi, None]
        t2 = T2[lo:hi, None]
        for k, sk in enumerate(by_class):
            below = (sk[None, :] < t1).sum(1)
            above = (sk[None, :] >= t2).sum(1)
            CM[lo:hi, k, 0] = below
            CM[lo:hi, k, 2] = above
            CM[lo:hi, k, 1] = counts[k] - below - above
    S = np.einsum("gkk->gk", CM) / counts[None, :]

    if direction == "decreasing":       # report cut-points on the user's original scale
        T1, T2 = -T2, -T1

    return OrdinalSurface(
        T1=T1, T2=T2, S=S, CM=CM, classes=classes, counts=counts,
        marker=s, y_idx=y_idx, direction=direction,
    )
