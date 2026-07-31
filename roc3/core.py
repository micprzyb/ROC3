"""
roc3.core
=========

The 3-class ROC **surface** (Mossman 1999 / Scurfield 1996) and its **volume** (VUS).

Design (see ``docs/PLAN.md`` §3):

*Rule family.*  Every point of the surface is a deployable classifier from the
Equal-Error-Utility Bayes family

    y_hat(x; w) = argmax_k  w_k * p_k(x),      w in the 2-simplex.

For 2 classes this is exactly the ordinary threshold sweep, so the whole construction
degenerates to the familiar ROC curve / AUC.

*Axes.*  The operating point is the triple of **true class fractions** (per-class
sensitivities)  S_k(w) = P(y_hat = k | y = k).  Sweeping w over the simplex traces a 2-D
surface inside the unit cube.

*Volume.*  VUS is the volume of the **down-set** of the achievable operating points, i.e.
the hypervolume indicator with reference point at the origin.  Perfect model -> 1,
uninformative model -> 1/6, adversarial -> 0.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

__all__ = [
    "prepare_scores",
    "roc_surface",
    "OperatingSurface",
    "hypervolume3d",
    "convex_hull_vus",
    "monotone_envelope",
    "simplex_lattice",
    "lattice_surface",
    "CHANCE_VUS",
]

#: Volume under the surface of an uninformative 3-class model:  1/3! .
CHANCE_VUS = 1.0 / 6.0

#: Floor applied to probabilities before taking logs.  Only differences of log scores
#: enter any rule in the family, so this floor is what bounds the representable dynamic
#: range of a score matrix.  It used to be 1e-12, which silently flattened confident
#: models -- and, worse, destroyed the class-rescaling gauge invariance that
#: :mod:`roc3.constrained` relies on, because re-gauging pushes one column far down.
#: 1e-300 keeps every float64-representable probability intact while still giving a
#: finite log (about -690).
_EPS = 1e-300


# --------------------------------------------------------------------------------------
# input handling
# --------------------------------------------------------------------------------------
def prepare_scores(y, scores, *, classes=None, score_type="proba"):
    """Validate inputs and return ``(y_idx, log_scores, classes, counts)``.

    Parameters
    ----------
    y : array-like, shape (n,)
        True labels (any hashable values).
    scores : array-like, shape (n, C)
        Column ``j`` is the score for ``classes[j]``.
    classes : sequence, optional
        Class order for the columns of ``scores``.  Defaults to ``np.unique(y)``.
    score_type : {'proba', 'log'}
        ``'proba'``: non-negative scores, rows renormalised to sum to 1.
        ``'log'``: scores are already on a log / logit scale and used as-is.

    Notes
    -----
    Only *differences* of the log-scores enter any rule in the family, so an arbitrary
    per-row additive constant (equivalently, per-row multiplicative constant on the
    probability scale) is irrelevant.  That is why un-normalised logits are acceptable.
    """
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError(f"scores must be 2-D (n, C); got shape {scores.shape}")
    if len(y) != len(scores):
        raise ValueError(f"len(y)={len(y)} != len(scores)={len(scores)}")

    classes = np.unique(y) if classes is None else np.asarray(classes)
    if len(classes) != scores.shape[1]:
        raise ValueError(
            f"scores has {scores.shape[1]} columns but {len(classes)} classes were given"
        )

    lookup = {c: i for i, c in enumerate(classes)}
    missing = set(np.unique(y)) - set(lookup)
    if missing:
        raise ValueError(f"labels {sorted(missing)} are not in classes={list(classes)}")
    y_idx = np.array([lookup[v] for v in y], dtype=np.intp)

    if score_type == "proba":
        if np.any(scores < 0):
            raise ValueError(
                "negative values in `scores` with score_type='proba'; "
                "pass score_type='log' if these are logits"
            )
        row = scores.sum(axis=1, keepdims=True)
        if np.any(row <= 0):
            raise ValueError("every row of `scores` must have a positive sum")
        log_scores = np.log(np.clip(scores / row, _EPS, None))
    elif score_type == "log":
        log_scores = scores.copy()
        if not np.all(np.isfinite(log_scores)):
            finite = log_scores[np.isfinite(log_scores)]
            floor = (finite.min() - 40.0) if finite.size else -40.0
            log_scores = np.nan_to_num(log_scores, neginf=floor, posinf=floor + 80.0)
    else:
        raise ValueError("score_type must be 'proba' or 'log'")

    counts = np.bincount(y_idx, minlength=len(classes))
    if np.any(counts == 0):
        empty = [classes[i] for i in np.flatnonzero(counts == 0)]
        raise ValueError(f"no samples for class(es) {empty}; cannot form a ROC surface")
    return y_idx, log_scores, classes, counts


# --------------------------------------------------------------------------------------
# grid over the 2-parameter rule family
# --------------------------------------------------------------------------------------
def _sweep_grid(values: np.ndarray, resolution: int, pad: float) -> np.ndarray:
    """Grid of cut points that separates the distinct entries of ``values``.

    Points are placed strictly *between* consecutive distinct data values (plus one
    outside each end), so strict inequalities are unambiguous and every combinatorially
    distinct rule along this axis is reachable.
    """
    u = np.unique(values[np.isfinite(values)])
    if u.size == 0:
        return np.array([-pad, pad])
    if u.size > resolution:
        u = np.unique(np.quantile(u, np.linspace(0.0, 1.0, resolution)))
    mid = (u[:-1] + u[1:]) / 2.0 if u.size > 1 else np.empty(0)
    return np.concatenate(([u[0] - pad], mid, [u[-1] + pad]))


@dataclass
class OperatingSurface:
    """The 3-class ROC surface: a 2-parameter family of deployable rules.

    Attributes
    ----------
    A, B : ndarray, shape (G,)
        Log weight ratios ``a = log(w2/w1)``, ``b = log(w3/w1)`` for each grid rule.
    W : ndarray, shape (G, 3)
        The same rules as normalised weights on the simplex.
    S : ndarray, shape (G, 3)
        Operating points: per-class sensitivities (true class fractions).
    classes : ndarray, shape (3,)
        Class labels, in the column order of the score matrix.
    counts : ndarray, shape (3,)
        Number of samples per class.
    """

    A: np.ndarray
    B: np.ndarray
    W: np.ndarray
    S: np.ndarray
    classes: np.ndarray
    counts: np.ndarray
    y_idx: np.ndarray = field(repr=False)
    log_scores: np.ndarray = field(repr=False)
    grid_shape: tuple = ()
    #: (G, 3, 3) raw confusion counts per rule; ``CM[g, true, predicted]``.
    CM: np.ndarray = field(default=None, repr=False)

    @property
    def rates(self) -> np.ndarray:
        """(G, 3, 3) row-normalised confusion matrices; ``rates[g, k, i] = P(pred=i|y=k)``."""
        return self.CM / self.counts[None, :, None]

    # ---- volumes -----------------------------------------------------------------
    @property
    def vus(self) -> float:
        """Volume under the surface (down-set hypervolume).  Perfect 1, chance 1/6."""
        return hypervolume3d(self.S)

    @property
    def vus_adjusted(self) -> float:
        """Chance-corrected VUS: ``(VUS - 1/6) / (1 - 1/6)``.  Chance 0, perfect 1."""
        return (self.vus - CHANCE_VUS) / (1.0 - CHANCE_VUS)

    def partial_vus(self, floors=(0.0, 0.0, 0.0), normalize=True) -> float:
        """VUS restricted to the box ``[floors_k, 1]^3`` (analogue of partial AUC).

        With ``normalize=True`` the result is divided by the box volume, so a model that
        can meet every requirement inside the box scores 1.
        """
        floors = np.asarray(floors, dtype=float)
        if np.any(floors < 0) or np.any(floors >= 1):
            raise ValueError("floors must lie in [0, 1)")
        keep = np.all(self.S >= floors, axis=1)
        if not keep.any():
            return 0.0
        shifted = (self.S[keep] - floors) / (1.0 - floors)
        vol = hypervolume3d(shifted)
        return vol if normalize else vol * float(np.prod(1.0 - floors))

    @property
    def vus_convex_hull(self) -> float:
        """VUS achievable if *randomised* rules are allowed (analogue of AUC-under-hull).

        Randomising between rules makes the whole convex hull of the operating points
        attainable, so the achievable region becomes the down-set of that hull,

            D = { q : q <= p for some p in conv(S) }  =  conv( union of boxes [0, p] ),

        (the second equality holds because scaling each coordinate of a box vertex by the
        same factors keeps it inside its box).  ``D`` is convex, so its volume is just the
        volume of the convex hull of all box vertices.

        ``vus_convex_hull - vus`` is the head-room available from better thresholding or
        randomisation alone, with the model itself left untouched.
        """
        return convex_hull_vus(self.S)

    # ---- geometry ----------------------------------------------------------------
    def envelope(self, resolution: int = 201):
        """Monotone envelope ``Z(u, v) = max{ S3 : S1 >= u, S2 >= v }``.

        Returns ``(u, v, Z)`` with ``Z`` of shape ``(resolution, resolution)`` indexed
        ``Z[iu, iv]``.  ``Z.mean()`` is a midpoint-rule estimate of the VUS, so this is
        literally "the surface whose volume is the VUS".
        """
        return monotone_envelope(self.S, resolution=resolution)

    def pareto_mask(self) -> np.ndarray:
        """Boolean mask of Pareto-optimal (non-dominated) operating points."""
        return _pareto_mask(self.S)

    # ---- evaluation at a rule ------------------------------------------------------
    def confusion_at(self, w) -> np.ndarray:
        """Raw 3x3 confusion matrix (rows = true class) for the rule with weights ``w``."""
        w = np.asarray(w, dtype=float)
        if w.shape != (3,) or np.any(w < 0) or not np.any(w > 0):
            raise ValueError("w must be 3 non-negative numbers, not all zero")
        pred = np.argmax(self.log_scores + np.log(np.clip(w / w.sum(), _EPS, None)), axis=1)
        cm = np.zeros((3, 3), dtype=np.int64)
        np.add.at(cm, (self.y_idx, pred), 1)
        return cm

    def sensitivities_at(self, w) -> np.ndarray:
        cm = self.confusion_at(w)
        return np.diag(cm) / cm.sum(axis=1)

    def select(self, criterion="youden", **kwargs):
        """Pick an operating point.  See :mod:`roc3.thresholds` for the criteria."""
        from .thresholds import select_operating_point

        return select_operating_point(self, criterion=criterion, **kwargs)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return (
            f"OperatingSurface(n_rules={len(self.S)}, classes={list(self.classes)}, "
            f"counts={list(self.counts)}, VUS={self.vus:.4f})"
        )


def roc_surface(
    y,
    scores,
    *,
    classes=None,
    score_type="proba",
    resolution: int = 160,
    max_cells: int = 4_000_000,
) -> OperatingSurface:
    """Compute the 3-class ROC surface.

    Parameters
    ----------
    y, scores, classes, score_type
        See :func:`prepare_scores`.  Exactly 3 classes are required.
    resolution : int
        Number of cut points per axis of the 2-D rule-parameter grid.  The number of
        candidate rules is roughly ``resolution**2``.  VUS converges from below as this
        grows; 160 is ample for n up to a few thousand (see ``docs/IDEAS_LOG.md``).
    max_cells : int
        Memory guard: the (grid-chunk x samples) working array is kept below this size.

    Returns
    -------
    OperatingSurface
    """
    y_idx, logp, classes, counts = prepare_scores(
        y, scores, classes=classes, score_type=score_type
    )
    if len(classes) != 3:
        raise ValueError(
            f"roc_surface needs exactly 3 classes, got {len(classes)}. "
            "Use roc3.metrics.hum() for the rank-based summary with other class counts."
        )

    d12 = logp[:, 0] - logp[:, 1]
    d13 = logp[:, 0] - logp[:, 2]
    d23 = logp[:, 1] - logp[:, 2]

    span = max(
        np.ptp(d12[np.isfinite(d12)]) if np.isfinite(d12).any() else 0.0,
        np.ptp(d13[np.isfinite(d13)]) if np.isfinite(d13).any() else 0.0,
        np.ptp(d23[np.isfinite(d23)]) if np.isfinite(d23).any() else 0.0,
    )
    pad = max(1.0, 2.0 * span)

    a_axis = _sweep_grid(d12, resolution, pad)
    b_axis = _sweep_grid(d13, resolution, pad)
    AA, BB = np.meshgrid(a_axis, b_axis, indexing="ij")
    A = AA.ravel()
    B = BB.ravel()
    G = A.size

    CM = _confusions_for_rules(d12, d13, d23, y_idx, counts, A, B, max_cells)
    S = np.einsum("gkk->gk", CM) / counts[None, :]
    W = _weights_from_log_ratios(A, B)

    return OperatingSurface(
        A=A, B=B, W=W, S=S, classes=classes, counts=counts,
        y_idx=y_idx, log_scores=logp, grid_shape=(a_axis.size, b_axis.size), CM=CM,
    )


def _confusions_for_rules(d12, d13, d23, y_idx, counts, A, B, max_cells=4_000_000):
    """Confusion counts ``(G, 3, 3)`` for the rules with log weight ratios ``(A, B)``.

    The rule's partition of the ``(a, b)`` plane is the same for every sample:

        predict 0  <=>  d12 >= a  and  d13 >= b
        predict 1  <=>  d12 <  a  and  d23 >= b - a
        predict 2  <=>  the remainder

    The inequalities are oriented so that exact ties resolve exactly as ``np.argmax``
    does (lowest class index wins), keeping the surface consistent with
    :meth:`OperatingSurface.confusion_at`.  Grid values sit strictly between distinct
    data values, so for continuous scores the orientation never matters.
    """
    A = np.asarray(A, dtype=float).ravel()
    B = np.asarray(B, dtype=float).ravel()
    G = A.size
    by_class = [(d12[y_idx == k], d13[y_idx == k], d23[y_idx == k]) for k in range(3)]
    chunk = max(1, int(max_cells // max(1, int(np.max(counts)))))
    CM = np.empty((G, 3, 3), dtype=np.int64)
    for lo in range(0, G, chunk):
        hi = min(G, lo + chunk)
        a = A[lo:hi, None]
        b = B[lo:hi, None]
        c = b - a
        for k, (dk12, dk13, dk23) in enumerate(by_class):
            n_p0 = ((dk12[None, :] >= a) & (dk13[None, :] >= b)).sum(1)
            n_p1 = ((dk12[None, :] < a) & (dk23[None, :] >= c)).sum(1)
            CM[lo:hi, k, 0] = n_p0
            CM[lo:hi, k, 1] = n_p1
            CM[lo:hi, k, 2] = counts[k] - n_p0 - n_p1
    return CM


def _weights_from_log_ratios(A, B) -> np.ndarray:
    """Normalised simplex weights from ``a = log(w2/w1)``, ``b = log(w3/w1)``."""
    L = np.stack([np.zeros_like(A), np.asarray(A, float), np.asarray(B, float)], axis=1)
    L = L - L.max(axis=1, keepdims=True)
    W = np.exp(L)
    return W / W.sum(axis=1, keepdims=True)


def simplex_lattice(n: int = 96) -> np.ndarray:
    """Regular barycentric lattice on the 2-simplex: ``(n+1)(n+2)/2`` weight vectors."""
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    keep = (i + j) <= n
    i, j = i[keep], j[keep]
    return np.stack([i, j, n - i - j], axis=1) / float(n)


def lattice_surface(surface: "OperatingSurface", n: int = 96) -> "OperatingSurface":
    """Re-evaluate the same model on a *regular* barycentric lattice of weights.

    :func:`roc_surface` places rules where the data changes them (best for the volume);
    the ternary panels instead want an evenly spaced mesh over the weight simplex so the
    contours are clean.  Log ratios are clipped to the range in which the rule can still
    change, so the lattice corners really are the "always predict k" rules.
    """
    logp = surface.log_scores
    d12 = logp[:, 0] - logp[:, 1]
    d13 = logp[:, 0] - logp[:, 2]
    d23 = logp[:, 1] - logp[:, 2]
    span = max(float(np.ptp(d12)), float(np.ptp(d13)), float(np.ptp(d23)), 1.0)
    pad = 2.0 * span

    W = simplex_lattice(n)
    lw = np.log(np.clip(W, 1e-300, None))
    A = np.clip(lw[:, 1] - lw[:, 0], -pad, pad)
    B = np.clip(lw[:, 2] - lw[:, 0], -pad, pad)
    CM = _confusions_for_rules(d12, d13, d23, surface.y_idx, surface.counts, A, B)
    S = np.einsum("gkk->gk", CM) / surface.counts[None, :]
    return OperatingSurface(
        A=A, B=B, W=W, S=S, classes=surface.classes, counts=surface.counts,
        y_idx=surface.y_idx, log_scores=logp, grid_shape=(n,), CM=CM,
    )


# --------------------------------------------------------------------------------------
# geometry: hypervolume of the down-set, Pareto front, monotone envelope
# --------------------------------------------------------------------------------------
class _Staircase:
    """Union of origin-anchored rectangles, maintained incrementally.

    Stores the 2-D Pareto-maximal points sorted by ``x`` ascending (hence ``y`` strictly
    descending).  ``area`` is the area of ``union_i [0, x_i] x [0, y_i]``, kept up to date
    as ``sum_i y_i * (x_i - x_{i-1})`` with ``x_{-1} = 0``.

    Amortised cost per insertion is O(log n) comparisons plus one C-level list splice;
    every point is inserted and deleted at most once.
    """

    __slots__ = ("xs", "ys", "area")

    def __init__(self):
        self.xs: list[float] = []
        self.ys: list[float] = []
        self.area = 0.0

    def dominated(self, x: float, y: float) -> bool:
        """True if some stored point has ``x' >= x`` and ``y' >= y``."""
        xs, ys = self.xs, self.ys
        i = bisect.bisect_left(xs, x)
        return i < len(xs) and ys[i] >= y

    def add(self, x: float, y: float) -> bool:
        """Insert ``(x, y)``; returns True if it was non-dominated (and thus stored)."""
        if x <= 0.0 or y <= 0.0:
            return False
        xs, ys = self.xs, self.ys
        n = len(xs)
        i = bisect.bisect_left(xs, x)          # first index with xs[i] >= x
        if i < n and ys[i] >= y:               # dominated by an existing point
            return False
        if i < n and xs[i] == x:               # same x, smaller y -> absorb into removal
            i += 1
        # Points dominated by (x, y) form the contiguous block [j, i).
        j = i
        while j > 0 and ys[j - 1] <= y:
            j -= 1
        for m in range(j, i):
            self.area -= ys[m] * (xs[m] - (xs[m - 1] if m > 0 else 0.0))
        if i < n:                              # successor's predecessor becomes x
            self.area += ys[i] * ((xs[i - 1] if i > 0 else 0.0) - x)
        del xs[j:i]
        del ys[j:i]
        xs.insert(j, x)
        ys.insert(j, y)
        self.area += y * (x - (xs[j - 1] if j > 0 else 0.0))
        return True


def hypervolume3d(points: np.ndarray) -> float:
    """Volume of ``union_i [0, x_i] x [0, y_i] x [0, z_i]`` for points in ``[0, 1]^3``.

    This is the hypervolume indicator with reference point at the origin, computed
    exactly by a dimension sweep over ``z``.  For two classes the analogous 2-D quantity
    is exactly the trapezoidal AUC, which is why this is the right generalisation.
    """
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")
    pts = pts[np.all(pts > 0.0, axis=1)]
    if pts.size == 0:
        return 0.0
    pts = np.unique(pts, axis=0)
    order = np.argsort(-pts[:, 2], kind="stable")
    pts = pts[order]
    zs = pts[:, 2]

    sc = _Staircase()
    vol = 0.0
    prev_z = None
    i, N = 0, len(pts)
    while i < N:
        z = zs[i]
        if prev_z is not None:
            vol += sc.area * (prev_z - z)
        j = i
        while j < N and zs[j] == z:
            sc.add(pts[j, 0], pts[j, 1])
            j += 1
        prev_z = z
        i = j
    vol += sc.area * prev_z
    return float(vol)


def convex_hull_vus(S: np.ndarray) -> float:
    """Volume of the down-set of ``conv(S)`` -- the VUS attainable with randomised rules.

    Always ``>= hypervolume3d(S)``.  See :attr:`OperatingSurface.vus_convex_hull`.
    """
    from itertools import product

    from scipy.spatial import ConvexHull
    from scipy.spatial.qhull import QhullError  # type: ignore[attr-defined]

    pts = np.unique(np.asarray(S, dtype=float), axis=0)
    pts = pts[_pareto_mask(pts)]                     # dominated boxes add nothing
    verts = np.vstack([
        pts * np.array(mask, dtype=float) for mask in product((0.0, 1.0), repeat=3)
    ])
    verts = np.unique(np.round(verts, 12), axis=0)
    try:
        return float(ConvexHull(verts).volume)
    except (QhullError, ValueError):                 # degenerate / coplanar cloud
        return hypervolume3d(pts)


def hypervolume3d_mc(points: np.ndarray, n_samples: int = 400_000, seed: int = 0):
    """Monte-Carlo cross-check of :func:`hypervolume3d`.  Returns ``(estimate, stderr)``."""
    pts = np.unique(np.asarray(points, dtype=float), axis=0)
    pts = pts[_pareto_mask(pts)]
    rng = np.random.default_rng(seed)
    hit = 0
    block = 20_000
    done = 0
    while done < n_samples:
        m = min(block, n_samples - done)
        u = rng.random((m, 3))
        dominated = np.zeros(m, dtype=bool)
        for lo in range(0, len(pts), 512):
            p = pts[lo : lo + 512]
            dominated |= np.any(np.all(u[:, None, :] <= p[None, :, :], axis=2), axis=1)
        hit += int(dominated.sum())
        done += m
    p = hit / n_samples
    return p, float(np.sqrt(max(p * (1 - p), 1e-12) / n_samples))


def _pareto_mask(S: np.ndarray) -> np.ndarray:
    """Mask of rows not strictly dominated by another row (maximisation in all 3 coords).

    Exact duplicates are all kept.  Sweep by ``S1`` descending while maintaining a 2-D
    staircase over ``(S2, S3)``: a point is dominated iff the staircase of already-seen
    points (which all have ``S1 >= ``) covers it.  ``O(n log n)``.
    """
    S = np.asarray(S, dtype=float)
    n = len(S)
    order = np.lexsort((-S[:, 2], -S[:, 1], -S[:, 0]))
    keep = np.zeros(n, dtype=bool)
    sc = _Staircase()
    prev = None
    prev_keep = False
    for idx in order:
        s1, s2, s3 = S[idx]
        if prev is not None and s1 == prev[0] and s2 == prev[1] and s3 == prev[2]:
            keep[idx] = prev_keep           # duplicates share the verdict
            continue
        # Shift into the strictly-positive quadrant so that zero coordinates still take
        # part in the domination test (the staircase ignores non-positive entries).
        ok = not sc.dominated(s2 + 1.0, s3 + 1.0)
        if ok:
            sc.add(s2 + 1.0, s3 + 1.0)
        keep[idx] = ok
        prev, prev_keep = (s1, s2, s3), ok
    return keep


def monotone_envelope(S: np.ndarray, resolution: int = 201):
    """Grid the monotone envelope ``Z(u, v) = max{ S3 : S1 >= u, S2 >= v }``.

    Returns ``(u, v, Z)`` where ``u``/``v`` are cell midpoints of ``[0, 1]`` and
    ``Z[iu, iv]`` is the best achievable class-3 sensitivity subject to the class-1 and
    class-2 requirements.  ``Z.mean()`` approximates the VUS.
    """
    S = np.asarray(S, dtype=float)
    R = int(resolution)
    mid = (np.arange(R) + 0.5) / R
    iu = np.floor(S[:, 0] * R - 0.5).astype(np.int64)
    iv = np.floor(S[:, 1] * R - 0.5).astype(np.int64)
    ok = (iu >= 0) & (iv >= 0)
    Z0 = np.zeros((R, R))
    if ok.any():
        np.maximum.at(Z0, (iu[ok], iv[ok]), S[ok, 2])
    Z = np.maximum.accumulate(Z0[::-1, :], axis=0)[::-1, :]
    Z = np.maximum.accumulate(Z[:, ::-1], axis=1)[:, ::-1]
    return mid, mid, Z
