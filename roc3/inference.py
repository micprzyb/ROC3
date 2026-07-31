"""
roc3.inference
==============

Uncertainty for the VUS and for a chosen operating point.

Three tools:

1. :func:`vus_3afc_inference` -- **closed-form** standard error for the rank-based VUS.
   ``VUS_3AFC`` is a three-sample U-statistic of degree (1,1,1), so by the Hoeffding
   decomposition ``Var ~= z1/n1 + z2/n2 + z3/n3`` where ``zk`` is the variance of the
   conditional expectation given one argument.  The three sets of conditional means are
   obtained *exactly* as a by-product of the triple sweep, so no resampling is needed.
   This is the Nakas & Yiannoutsos (2004) variance, computed for the general
   (non-ordinal) score-vector case.

2. :func:`bootstrap_vus` -- **stratified bootstrap** (resample within class).  Works for
   either VUS flavour and needs no theory; the fallback when assumptions are doubtful.

3. :func:`permutation_test_vus` -- exact-in-the-limit test of ``H0: VUS = 1/6`` by
   permuting the labels.

:func:`bootstrap_operating_point` gives a confidence region for the true class fraction
triplet ``(S1, S2, S3)`` at a *fixed* rule -- the quantity Bantis et al. (2022) argue you
should report next to the VUS.
"""

from __future__ import annotations

import numpy as np

from .core import CHANCE_VUS, prepare_scores, roc_surface
from .vus import forced_choice_profile

__all__ = [
    "vus_3afc_inference",
    "bootstrap_vus",
    "permutation_test_vus",
    "bootstrap_operating_point",
]


def _fast_margins(L0, L1, L2):
    """Exact VUS_3AFC plus the three sets of conditional means (Hoeffding projections).

    Uses the five-condition characterisation, so ties are counted as losses; with
    continuous scores that is a measure-zero event.  Returns ``(vus, m0, m1, m2)``.
    """
    A0, B0 = L0[:, 0] - L0[:, 1], L0[:, 0] - L0[:, 2]
    A1, C1 = L1[:, 0] - L1[:, 1], L1[:, 1] - L1[:, 2]
    B2, C2 = L2[:, 0] - L2[:, 2], L2[:, 1] - L2[:, 2]
    n0, n1, n2 = len(L0), len(L1), len(L2)

    m0 = np.zeros(n0)
    m1 = np.zeros(n1)
    m2 = np.zeros(n2)
    total = 0
    for i in range(n0):
        mj = A0[i] > A1
        mk = B0[i] > B2
        if not mj.any() or not mk.any():
            continue
        jj = np.flatnonzero(mj)
        kk = np.flatnonzero(mk)
        Aj, Cj = A1[jj], C1[jj]
        Bk, Ck = B2[kk], C2[kk]
        ok = Cj[:, None] > Ck[None, :]
        ok &= (Cj[:, None] + A0[i]) > Bk[None, :]
        ok &= B0[i] > (Aj[:, None] + Ck[None, :])
        c = int(ok.sum())
        total += c
        m0[i] = c
        m1[jj] += ok.sum(axis=1)
        m2[kk] += ok.sum(axis=0)

    vus = total / (n0 * n1 * n2)
    return vus, m0 / (n1 * n2), m1 / (n0 * n2), m2 / (n0 * n1)


def vus_3afc_inference(y, scores, *, classes=None, score_type="proba", alpha=0.05):
    """Exact VUS_3AFC with a closed-form (U-statistic) standard error and Wald CI.

    Returns a dict with ``vus``, ``stderr``, ``ci``, ``z``, ``p_value`` (two-sided test of
    ``H0: VUS = 1/6``), and the three Hoeffding variance components.

    Cost is ``O(n1*n2*n3)`` in the worst case but is heavily pruned in practice.
    """
    from scipy.stats import norm

    y_idx, logp, classes, counts = prepare_scores(
        y, scores, classes=classes, score_type=score_type
    )
    if len(classes) != 3:
        raise ValueError("vus_3afc_inference requires exactly 3 classes")
    L = [logp[y_idx == k] for k in range(3)]
    vus, m0, m1, m2 = _fast_margins(*L)
    n0, n1, n2 = counts

    z1 = float(np.var(m0, ddof=1)) if n0 > 1 else 0.0
    z2 = float(np.var(m1, ddof=1)) if n1 > 1 else 0.0
    z3 = float(np.var(m2, ddof=1)) if n2 > 1 else 0.0
    var = z1 / n0 + z2 / n1 + z3 / n2
    se = float(np.sqrt(max(var, 0.0)))

    q = norm.ppf(1 - alpha / 2)
    ci = (max(0.0, vus - q * se), min(1.0, vus + q * se))
    zstat = (vus - CHANCE_VUS) / se if se > 0 else np.inf
    return {
        "vus": float(vus),
        "vus_adjusted": (vus - CHANCE_VUS) / (1 - CHANCE_VUS),
        "stderr": se,
        "ci": ci,
        "alpha": alpha,
        "components": {"zeta_1": z1, "zeta_2": z2, "zeta_3": z3},
        "z": float(zstat),
        "p_value": float(2 * norm.sf(abs(zstat))) if np.isfinite(zstat) else 0.0,
    }


def bootstrap_vus(
    y,
    scores,
    *,
    classes=None,
    score_type="proba",
    kind="geometric",
    n_boot=200,
    alpha=0.05,
    resolution=80,
    fc_kwargs=None,
    seed=0,
    progress=False,
):
    """Stratified bootstrap CI for the VUS.

    Parameters
    ----------
    kind : {'geometric', '3afc', 'both'}
        Which estimator to resample.  ``'geometric'`` rebuilds the surface each replicate
        (use a smaller ``resolution`` than for the final figure -- the bias from a coarser
        grid is common to all replicates and cancels in the interval width).
    n_boot : int
        Number of replicates.
    """
    y_idx, logp, classes, counts = prepare_scores(
        y, scores, classes=classes, score_type=score_type
    )
    rng = np.random.default_rng(seed)
    idx_by_class = [np.flatnonzero(y_idx == k) for k in range(len(classes))]
    kinds = ("geometric", "3afc") if kind == "both" else (kind,)
    out = {k: np.empty(n_boot) for k in kinds}
    fc_kwargs = dict(fc_kwargs or {})

    for b in range(n_boot):
        take = np.concatenate([rng.choice(ix, size=len(ix), replace=True)
                               for ix in idx_by_class])
        yb, pb = y_idx[take], logp[take]
        if "geometric" in out:
            out["geometric"][b] = roc_surface(
                yb, pb, classes=np.arange(len(classes)), score_type="log",
                resolution=resolution,
            ).vus
        if "3afc" in out:
            out["3afc"][b] = forced_choice_profile(
                yb, pb, classes=np.arange(len(classes)), score_type="log", **fc_kwargs
            ).vus
        if progress and (b + 1) % max(1, n_boot // 10) == 0:
            print(f"    bootstrap {b + 1}/{n_boot}", flush=True)

    res = {}
    for k, v in out.items():
        lo, hi = np.quantile(v, [alpha / 2, 1 - alpha / 2])
        res[k] = {
            "mean": float(v.mean()),
            "stderr": float(v.std(ddof=1)),
            "ci": (float(lo), float(hi)),
            "replicates": v,
        }
    return res


def permutation_test_vus(
    y,
    scores,
    *,
    classes=None,
    score_type="proba",
    kind="3afc",
    n_perm=500,
    resolution=60,
    seed=0,
):
    """Test ``H0: the scores carry no information`` (VUS = 1/6) by permuting labels.

    Returns the observed VUS, the null distribution, and a one-sided p-value
    ``(1 + #{null >= observed}) / (1 + n_perm)``.
    """
    y_idx, logp, classes, counts = prepare_scores(
        y, scores, classes=classes, score_type=score_type
    )
    rng = np.random.default_rng(seed)

    def stat(yy):
        if kind == "geometric":
            return roc_surface(yy, logp, classes=np.arange(len(classes)),
                               score_type="log", resolution=resolution).vus
        return forced_choice_profile(yy, logp, classes=np.arange(len(classes)),
                                     score_type="log").vus

    observed = stat(y_idx)
    null = np.empty(n_perm)
    perm = y_idx.copy()
    for b in range(n_perm):
        rng.shuffle(perm)
        null[b] = stat(perm)
    p = (1 + int((null >= observed).sum())) / (1 + n_perm)
    return {
        "observed": float(observed),
        "null_mean": float(null.mean()),
        "null_sd": float(null.std(ddof=1)),
        "p_value": float(p),
        "null": null,
        "kind": kind,
    }


def bootstrap_operating_point(
    y,
    scores,
    w,
    *,
    classes=None,
    score_type="proba",
    n_boot=1000,
    alpha=0.05,
    seed=0,
):
    """Confidence intervals for ``(S1, S2, S3)`` at a *fixed* rule ``argmax_k w_k p_k``.

    The rule is held fixed, so this is the honest uncertainty of the operating point you
    would actually deploy -- it does not include the extra optimism from having *chosen*
    ``w`` on the same data.  For that, wrap the whole selection in
    :func:`bootstrap_vus`-style resampling.
    """
    y_idx, logp, classes, counts = prepare_scores(
        y, scores, classes=classes, score_type=score_type
    )
    w = np.asarray(w, dtype=float)
    pred = np.argmax(logp + np.log(np.clip(w / w.sum(), 1e-300, None)), axis=1)
    correct = (pred == y_idx)

    rng = np.random.default_rng(seed)
    C = len(classes)
    reps = np.empty((n_boot, C))
    per_class = [correct[y_idx == k] for k in range(C)]
    for b in range(n_boot):
        for k in range(C):
            v = per_class[k]
            reps[b, k] = v[rng.integers(0, len(v), len(v))].mean()
    lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2], axis=0)
    point = np.array([v.mean() for v in per_class])
    return {
        "sensitivities": point,
        "ci_lower": lo,
        "ci_upper": hi,
        "replicates": reps,
        "alpha": alpha,
    }
