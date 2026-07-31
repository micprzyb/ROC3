"""
roc3.metrics
============

Familiar scalar comparators (Hand-Till M, one-vs-rest AUC, accuracy...) plus the
one-call :func:`summary` that puts every figure of merit side by side.

These are *context*, not the answer: none of them is a volume and none of them yields a
surface.  They are here so the VUS can be read against numbers people already trust.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

from .core import CHANCE_VUS, prepare_scores, roc_surface

__all__ = [
    "binary_auc",
    "hand_till_m",
    "ovr_auc",
    "ovo_auc_matrix",
    "roc_curve_points",
    "confusion_report",
    "summary",
]


def binary_auc(pos_scores, neg_scores) -> float:
    """Mann-Whitney AUC with the standard 1/2 credit for ties."""
    pos = np.asarray(pos_scores, dtype=float)
    neg = np.asarray(neg_scores, dtype=float)
    n1, n0 = len(pos), len(neg)
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def ovo_auc_matrix(y_idx, scores) -> np.ndarray:
    """``A[i, j] = A(i|j)``: AUC of column ``i`` separating class ``i`` from class ``j``."""
    C = scores.shape[1]
    A = np.full((C, C), np.nan)
    for i in range(C):
        for j in range(C):
            if i == j:
                continue
            A[i, j] = binary_auc(scores[y_idx == i, i], scores[y_idx == j, i])
    return A


def hand_till_m(y, scores, *, classes=None, score_type="proba") -> float:
    """Hand & Till (2001) M: mean over class pairs of ``(A(i|j) + A(j|i)) / 2``.

    Perfect 1, chance 1/2.  Prior-insensitive but blind to genuinely 3-way confusions,
    and it is not the volume of anything -- see ``docs/PLAN.md`` §A2.
    """
    y_idx, logp, classes, _ = prepare_scores(y, scores, classes=classes, score_type=score_type)
    A = ovo_auc_matrix(y_idx, logp)
    C = len(classes)
    vals = [0.5 * (A[i, j] + A[j, i]) for i in range(C) for j in range(i + 1, C)]
    return float(np.mean(vals))


def ovr_auc(y, scores, *, classes=None, score_type="proba", average="macro"):
    """One-vs-rest AUC per class (and the macro average)."""
    y_idx, logp, classes, _ = prepare_scores(y, scores, classes=classes, score_type=score_type)
    per = np.array(
        [binary_auc(logp[y_idx == k, k], logp[y_idx != k, k]) for k in range(len(classes))]
    )
    if average == "macro":
        return float(per.mean()), per
    return per


def roc_curve_points(pos_scores, neg_scores):
    """``(fpr, tpr)`` of the empirical ROC curve, including both endpoints."""
    pos = np.asarray(pos_scores, dtype=float)
    neg = np.asarray(neg_scores, dtype=float)
    s = np.concatenate([pos, neg])
    lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(-s, kind="mergesort")
    s, lab = s[order], lab[order]
    tp = np.cumsum(lab)
    fp = np.cumsum(1 - lab)
    keep = np.r_[np.diff(s) != 0, True]      # one point per distinct threshold
    tpr = np.r_[0.0, tp[keep] / max(len(pos), 1)]
    fpr = np.r_[0.0, fp[keep] / max(len(neg), 1)]
    return fpr, tpr


def confusion_report(cm, classes=None) -> dict:
    """Per-class and overall metrics from a raw confusion matrix (rows = true class)."""
    cm = np.asarray(cm, dtype=float)
    C = cm.shape[0]
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)
    with np.errstate(divide="ignore", invalid="ignore"):
        sens = np.where(support > 0, tp / support, np.nan)
        prec = np.where(predicted > 0, tp / predicted, np.nan)
        f1 = np.where((sens + prec) > 0, 2 * sens * prec / (sens + prec), 0.0)
    total = cm.sum()
    return {
        "classes": list(classes) if classes is not None else list(range(C)),
        "confusion": cm.astype(int),
        "support": support.astype(int),
        "sensitivity": sens,
        "precision": prec,
        "f1": f1,
        "accuracy": float(tp.sum() / total) if total else float("nan"),
        "balanced_accuracy": float(np.nanmean(sens)),
        "macro_f1": float(np.nanmean(f1)),
    }


def summary(
    y,
    scores,
    *,
    classes=None,
    score_type="proba",
    resolution: int = 160,
    fc_method: str = "auto",
    surface=None,
) -> dict:
    """Every figure of merit for a 3-class model, in one dict.

    Includes both VUS estimators, the chance-corrected versions, the convex-hull
    head-room, the plain-``argmax`` operating point, and the familiar comparators.
    """
    from .vus import forced_choice_profile

    y_idx, logp, classes, counts = prepare_scores(
        y, scores, classes=classes, score_type=score_type
    )
    if len(classes) != 3:
        raise ValueError("summary() is for 3 classes; use roc3.vus.hum for other counts")

    surf = surface if surface is not None else roc_surface(
        y_idx, logp, classes=np.arange(3), score_type="log", resolution=resolution
    )
    fc = forced_choice_profile(y_idx, logp, classes=np.arange(3), score_type="log",
                               method=fc_method)

    vus = surf.vus
    argmax_cm = surf.confusion_at(np.ones(3))
    rep = confusion_report(argmax_cm, classes)
    macro_ovr, per_ovr = ovr_auc(y_idx, logp, classes=np.arange(3), score_type="log")

    return {
        "classes": list(classes),
        "counts": counts.astype(int),
        # --- the headline pair -------------------------------------------------
        "VUS": vus,
        "VUS_adjusted": (vus - CHANCE_VUS) / (1 - CHANCE_VUS),
        "VUS_3AFC": fc.vus,
        "VUS_3AFC_adjusted": fc.vus_adjusted,
        "VUS_convex_hull": surf.vus_convex_hull,
        "chance_VUS": CHANCE_VUS,
        "afc_profile": fc.profile,
        "afc_method": fc.method,
        "afc_stderr": fc.mc_stderr,
        # --- comparators --------------------------------------------------------
        "hand_till_M": hand_till_m(y_idx, logp, classes=np.arange(3), score_type="log"),
        "ovr_macro_auc": macro_ovr,
        "ovr_auc_per_class": per_ovr,
        # --- the default operating point (plain argmax) -------------------------
        "argmax_accuracy": rep["accuracy"],
        "argmax_balanced_accuracy": rep["balanced_accuracy"],
        "argmax_macro_f1": rep["macro_f1"],
        "argmax_sensitivity": rep["sensitivity"],
        "argmax_confusion": rep["confusion"],
        "n_rules": len(surf.S),
    }


def format_summary(s: dict) -> str:
    """Pretty-print :func:`summary` output."""
    cls = [str(c) for c in s["classes"]]
    L = []
    L.append("=" * 68)
    L.append("3-CLASS ROC SURFACE SUMMARY")
    L.append("=" * 68)
    L.append("classes : " + ",  ".join(f"{c} (n={n})"
                                       for c, n in zip(cls, s["counts"])))
    L.append("")
    L.append("Volume under the ROC surface        (perfect 1.000, chance 0.167)")
    L.append(f"  VUS   (geometric, down-set)     : {s['VUS']:.4f}")
    L.append(f"  VUS   (3AFC rank statistic)     : {s['VUS_3AFC']:.4f}"
             + (f"  +/- {s['afc_stderr']:.4f} (MC)" if s.get("afc_stderr") else ""))
    L.append(f"  VUS   (convex hull / randomised): {s['VUS_convex_hull']:.4f}"
             f"   head-room {s['VUS_convex_hull'] - s['VUS']:+.4f}")
    L.append("")
    L.append("Chance-corrected                    (perfect 1.000, chance 0.000)")
    L.append(f"  VUS_adj (geometric)             : {s['VUS_adjusted']:.4f}")
    L.append(f"  VUS_adj (3AFC)                  : {s['VUS_3AFC_adjusted']:.4f}")
    L.append("")
    L.append("Comparators                         (perfect 1.000, chance 0.500)")
    L.append(f"  Hand-Till M (mean pairwise AUC) : {s['hand_till_M']:.4f}")
    L.append(f"  macro one-vs-rest AUC           : {s['ovr_macro_auc']:.4f}"
             f"   per class {np.round(s['ovr_auc_per_class'], 4)}")
    L.append("")
    L.append("Default operating point (plain argmax, w = (1,1,1))")
    L.append(f"  accuracy {s['argmax_accuracy']:.4f}   balanced accuracy "
             f"{s['argmax_balanced_accuracy']:.4f}   macro-F1 {s['argmax_macro_f1']:.4f}")
    L.append(f"  per-class sensitivity           : {np.round(s['argmax_sensitivity'], 4)}")
    L.append("")
    prof = s["afc_profile"]
    if np.all(np.isfinite(prof)):
        from .vus import PERMUTATIONS
        L.append("3AFC confusion profile (which mis-assignment of a trio wins)")
        for t in np.argsort(-prof):
            p = PERMUTATIONS[t]
            tag = "   <- correct" if t == 0 else ""
            L.append(f"  ({cls[p[0]]}, {cls[p[1]]}, {cls[p[2]]}) : {prof[t]:.4f}{tag}")
    L.append("=" * 68)
    return "\n".join(L)
