"""
roc3.thresholds
===============

Turn a point of the ROC surface into a **deployable rule**.

Every rule in the family is ``y_hat(x) = argmax_k w_k * p_k(x)``, so "choosing a
threshold" means choosing the weight vector ``w`` on the 2-simplex.  For two classes this
collapses to the familiar single cut-off ``p_1 > w_0 / (w_0 + w_1)``.

Criteria available (``criterion=``):

``youden``                Generalised Youden index, ``S1 + S2 + S3 - 1``  (Nakas et al.)
``balanced_accuracy``     ``mean_k S_k`` -- *the same argmax as* ``youden`` (affine map);
                          both names are provided and the equivalence is documented, not hidden.
``closest_to_perfection`` minimise the Euclidean distance from ``(S1,S2,S3)`` to ``(1,1,1)``
``maximin``               maximise ``min_k S_k`` (worst-class guarantee)
``accuracy``              maximise the prior-weighted accuracy ``sum_k pi_k S_k``
``expected_cost``         minimise ``sum_k pi_k sum_i cost[k, i] P(pred=i | y=k)``
                          -- the only criterion that can see *off-diagonal* structure
``max_volume``            the operating point that maximises ``S1*S2*S3`` (the largest box
                          under the surface; "MV" in the ClusROC package)

Any criterion can be combined with per-class sensitivity floors via ``constraints``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "OperatingPoint",
    "select_operating_point",
    "achievable",
    "objective_values",
    "CRITERIA",
]

CRITERIA = (
    "youden",
    "balanced_accuracy",
    "closest_to_perfection",
    "maximin",
    "accuracy",
    "expected_cost",
    "max_volume",
)
_CRITERIA = CRITERIA

#: Human-readable labels, used by the plots.
CRITERION_LABELS = {
    "youden": "generalised Youden  $S_1+S_2+S_3-1$",
    "balanced_accuracy": "balanced accuracy  $\\overline{S}$",
    "closest_to_perfection": "closeness to $(1,1,1)$",
    "maximin": "worst-class sensitivity  $\\min_k S_k$",
    "accuracy": "accuracy  $\\sum_k \\pi_k S_k$",
    "expected_cost": "$-$expected cost",
    "max_volume": "box volume  $S_1 S_2 S_3$",
}


def objective_values(S, rates, counts, criterion, *, priors=None, costs=None):
    """Score every rule under ``criterion`` (higher is always better).

    Shared by the weighted-argmax surface and the ordinal two-cut-point surface.
    """
    S = np.asarray(S, dtype=float)
    counts = np.asarray(counts, dtype=float)
    if priors is None:
        pri = counts / counts.sum()
    else:
        pri = np.asarray(priors, dtype=float)
        pri = pri / pri.sum()

    if criterion == "youden":
        return S.sum(axis=1) - 1.0
    if criterion == "balanced_accuracy":
        return S.mean(axis=1)
    if criterion == "closest_to_perfection":
        return -np.linalg.norm(1.0 - S, axis=1)
    if criterion == "maximin":
        return S.min(axis=1)
    if criterion == "accuracy":
        return S @ pri
    if criterion == "max_volume":
        return S.prod(axis=1)
    if criterion == "expected_cost":
        if costs is None:
            raise ValueError("criterion='expected_cost' requires costs=(3,3) array")
        Cm = np.asarray(costs, dtype=float)
        if Cm.shape != (3, 3):
            raise ValueError("costs must be a 3x3 array, costs[true, predicted]")
        if rates is None:
            raise ValueError("expected_cost needs the full confusion rates")
        return -np.einsum("k,gki,ki->g", pri, np.asarray(rates, dtype=float), Cm)
    raise ValueError(f"criterion must be one of {CRITERIA}, got {criterion!r}")


@dataclass
class OperatingPoint:
    """A single, fully specified decision rule and what it does."""

    index: int
    weights: np.ndarray            # (3,) simplex weights defining argmax_k w_k p_k(x)
    log_ratios: tuple              # (a, b) = (log(w2/w1), log(w3/w1))
    sensitivities: np.ndarray      # (3,) true class fractions
    confusion: np.ndarray          # (3, 3) raw counts, rows = true class
    criterion: str
    objective: float
    classes: np.ndarray
    constraints: dict

    @property
    def rates(self) -> np.ndarray:
        return self.confusion / self.confusion.sum(axis=1, keepdims=True)

    def describe(self, digits: int = 4) -> str:
        """The 'operating point card' -- everything needed to deploy and audit the rule."""
        from .metrics import confusion_report

        cls = list(self.classes)
        rep = confusion_report(self.confusion, cls)
        w = self.weights / self.weights.max()
        lines = []
        lines.append("-" * 66)
        lines.append(f"OPERATING POINT  ({self.criterion}"
                     + (f", constraints={self.constraints}" if self.constraints else "")
                     + ")")
        lines.append("-" * 66)
        lines.append("Decision rule:  predict argmax_k  w_k * p_k(x)")
        lines.append("      weights w = " + ", ".join(
            f"{c}:{v:.{digits}f}" for c, v in zip(cls, self.weights)))
        lines.append("   (relative)   = " + ", ".join(
            f"{c}:{v:.3f}" for c, v in zip(cls, w)))
        lines.append(f"   log-ratios   a=log(w[{cls[1]}]/w[{cls[0]}])={self.log_ratios[0]:+.3f}"
                     f"   b=log(w[{cls[2]}]/w[{cls[0]}])={self.log_ratios[1]:+.3f}")
        lines.append("")
        head = "            " + "".join(f"{('->' + str(c)):>10}" for c in cls) + "     Se"
        lines.append("Confusion (rows = true class):")
        lines.append(head)
        for k, c in enumerate(cls):
            row = "".join(f"{self.confusion[k, i]:>10d}" for i in range(3))
            lines.append(f"  true {str(c):<6}{row}   {rep['sensitivity'][k]:.{digits}f}")
        lines.append("")
        lines.append(f"  accuracy {rep['accuracy']:.{digits}f}"
                     f"   balanced accuracy {rep['balanced_accuracy']:.{digits}f}"
                     f"   macro-F1 {rep['macro_f1']:.{digits}f}")
        lines.append(f"  sensitivity {np.round(rep['sensitivity'], digits)}"
                     f"   precision {np.round(rep['precision'], digits)}")
        lines.append(f"  objective ({self.criterion}) = {self.objective:.{digits}f}")
        lines.append("-" * 66)
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (f"OperatingPoint(criterion={self.criterion!r}, "
                f"Se={np.round(self.sensitivities, 4)}, "
                f"w={np.round(self.weights, 4)})")


def _resolve_class(surface, key) -> int:
    if isinstance(key, (int, np.integer)) and 0 <= int(key) < 3:
        arr = np.flatnonzero(surface.classes == key)
        # Prefer an exact label match; fall back to positional index.
        return int(arr[0]) if arr.size else int(key)
    arr = np.flatnonzero(surface.classes == key)
    if arr.size == 0:
        raise KeyError(f"class {key!r} not found in {list(surface.classes)}")
    return int(arr[0])


def _constraint_mask(surface, constraints) -> np.ndarray:
    mask = np.ones(len(surface.S), dtype=bool)
    for key, floor in (constraints or {}).items():
        mask &= surface.S[:, _resolve_class(surface, key)] >= float(floor)
    return mask


def achievable(surface, constraints) -> bool:
    """Is there any rule in the family meeting all the sensitivity floors?"""
    return bool(_constraint_mask(surface, constraints).any())


def select_operating_point(
    surface,
    criterion: str = "youden",
    *,
    constraints: dict | None = None,
    costs=None,
    priors=None,
) -> OperatingPoint:
    """Choose the best rule on the surface under ``criterion`` and optional floors.

    Parameters
    ----------
    surface : OperatingSurface
    criterion : str
        One of the criteria listed in the module docstring.
    constraints : dict, optional
        ``{class_label: minimum_sensitivity}``.  Raises if infeasible, and the message
        reports the best simultaneously achievable value for each constrained class.
    costs : (3, 3) array, optional
        Required for ``criterion='expected_cost'``.  ``costs[k, i]`` is the cost of
        predicting ``i`` when the truth is ``k``.
    priors : (3,) array, optional
        Class prevalences for ``accuracy``/``expected_cost``.  Defaults to the empirical
        prevalence in the evaluation sample -- override this when deployment prevalence
        differs from the test set.
    """
    if criterion not in _CRITERIA:
        raise ValueError(f"criterion must be one of {_CRITERIA}, got {criterion!r}")
    S = surface.S
    mask = _constraint_mask(surface, constraints)
    if not mask.any():
        best = {k: float(surface.S[:, _resolve_class(surface, k)].max())
                for k in (constraints or {})}
        raise ValueError(
            f"constraints {constraints} are not achievable by any rule in the family. "
            f"Best single-class values (each attainable only on its own): {best}. "
            "Relax a floor, or improve the model."
        )

    obj = objective_values(
        S, surface.rates, surface.counts, criterion, priors=priors, costs=costs
    )
    scored = np.where(mask, obj, -np.inf)
    # Tie-break towards the most balanced rule so the choice is deterministic and sane.
    best = float(scored.max())
    cand = np.flatnonzero(scored >= best - 1e-12)
    idx = int(cand[np.argmax(S[cand].min(axis=1))])

    return OperatingPoint(
        index=idx,
        weights=surface.W[idx].copy(),
        log_ratios=(float(surface.A[idx]), float(surface.B[idx])),
        sensitivities=S[idx].copy(),
        confusion=surface.CM[idx].copy(),
        criterion=criterion,
        objective=float(obj[idx]),
        classes=surface.classes,
        constraints=dict(constraints or {}),
    )
