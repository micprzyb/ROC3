"""
roc3.constrained
================

ROC analysis when a **logical constraint** forces the class probabilities to be ordered,

    0 < p1(x) < p2(x) < p3(x) < 1     for every x.

Everything here turns on a distinction that is easy to miss, because the two readings of
that sentence point in *opposite* directions.

Reading B — the constraint is imposed on the **model's output**
---------------------------------------------------------------
An architectural guarantee (a Semantic Probabilistic Layer, an ordered softmax, a
projection): whatever the world looks like, the model emits ordered probabilities.

**This is pure gauge.**  Both VUS estimators and the entire ROC surface are invariant
under class-wise rescaling ``p_k -> c_k p_k`` (only ratios enter any rule in the family,
and a rescaling is absorbed into the weight vector ``w``).  And *every* model can be put
into the ordered gauge by one such rescaling: set ``c`` so that every sample's
``log p1 - log p2`` and ``log p2 - log p3`` are pushed below zero.  So

    the output-ordering constraint costs nothing, buys nothing, and is invisible to VUS.

What it *does* change is where the useful rules sit: ``argmax`` degenerates to "always
class 3", and the informative weights are squeezed into a corner of the simplex.  Fix that
with :func:`gauge_normalize` before plotting -- it changes no number, only the coordinates.

Reading A — the constraint is a fact about the **world**
--------------------------------------------------------
The *true posterior* is ordered: ``pi_1 f_1(x) < pi_2 f_2(x) < pi_3 f_3(x)`` for all ``x``.
Now it bites, hard, and it bites every classifier — not just constrained ones.

Integrating the pointwise inequality over any decision region ``R_m`` gives, for the
row-stochastic confusion matrix ``C[k, m] = P(predict m | true k)``,

    pi_i * C[i, m]  <=  pi_j * C[j, m]        for all i < j and all m.          (*)

Nine linear inequalities.  Consequences, all proved in ``docs/CONSTRAINED.md``:

* **The perfect corner is unreachable.**  ``(1,1,1)`` violates (*).
* **Accuracy is a dead metric**: ``max_C sum_k pi_k C[k,k] = pi_3`` exactly, attained by
  the constant rule "always predict 3".  No classifier can beat it.
* **VUS is capped** at a value ``VUS_max(pi) < 1`` computable from the priors alone.
* In the two-class case the cap reduces to the closed form ``1 - pi_1/(2 pi_2)``, which
  is also what a bounded-likelihood-ratio argument gives.  The two derivations agree.

So the scale changes: the range is ``[1/6, VUS_max(pi)]``, not ``[1/6, 1]``, and
:func:`normalized_vus` rescales onto that.

This mirrors Boyd, Davis & Page (ICML 2012), "Unachievable region in precision-recall
space": a region of the evaluation space that the problem's structure -- not the model --
puts out of reach.
"""

from __future__ import annotations

import itertools

import numpy as np

from .core import CHANCE_VUS, convex_hull_vus, prepare_scores

__all__ = [
    "ORDER_CHAMBER_VERTICES",
    "check_order_constraint",
    "to_ordered_gauge",
    "gauge_normalize",
    "order_chamber_sample",
    "constrained_world",
    "confusion_polytope",
    "achievable_diagonals",
    "vus_ceiling",
    "ceiling_surface",
    "pairwise_auc_ceiling",
    "scalar_ceilings",
    "normalized_vus",
    "constrained_report",
    "format_constrained_report",
    "plot_ceiling",
]


# ======================================================================================
# Reading B -- the constraint as a gauge on the model's output
# ======================================================================================
def check_order_constraint(scores, *, score_type="proba", strict=True) -> dict:
    """Does the score matrix satisfy ``p1 < p2 < p3`` row by row?

    Returns the fraction of rows that comply, the worst violation, and the smallest
    margin ``min_x min(log p2 - log p1, log p3 - log p2)`` (positive means satisfied
    with room to spare).
    """
    s = np.asarray(scores, dtype=float)
    if score_type == "proba":
        logp = np.log(np.clip(s / s.sum(axis=1, keepdims=True), 1e-300, None))
    else:
        logp = s
    d12 = logp[:, 1] - logp[:, 0]        # want > 0
    d23 = logp[:, 2] - logp[:, 1]        # want > 0
    ok = (d12 > 0) & (d23 > 0) if strict else (d12 >= 0) & (d23 >= 0)
    return {
        "satisfied": bool(ok.all()),
        "fraction_satisfied": float(ok.mean()),
        "n_violations": int((~ok).sum()),
        "min_log_margin": float(min(d12.min(), d23.min())),
        "worst_row": int(np.argmin(np.minimum(d12, d23))),
    }


#: log(smallest positive float64).  Below this a probability cannot be represented.
_LOG_TINY = np.log(np.finfo(float).tiny)          # about -708


def to_ordered_gauge(scores, *, score_type="proba", margin=1.0, return_log=False):
    """Rescale the classes so the output satisfies ``p1 < p2 < p3`` **for every row**.

    Applies ``p_k -> c_k p_k`` with a single constant vector ``c``, which leaves the ROC
    surface and both VUS estimators *exactly* unchanged (only ratios matter to any rule
    in the family, and the change is absorbed into ``w``).  Demonstrates that the ordering
    constraint is a gauge choice rather than a restriction on ranking quality.

    Returns ``(P_ordered, c)``.  ``P_ordered`` is on the probability scale unless
    ``return_log=True``.

    Dynamic range
    -------------
    The gauge is exact in exact arithmetic, but it costs representable range: the shift
    needed is the *spread* of the model's log-odds, so a very confident model needs
    ``p1`` far below float64's floor.  The function raises in that case and tells you to
    use ``return_log=True`` and pass ``score_type='log'`` downstream, where the identity
    still holds exactly.
    """
    s = np.asarray(scores, dtype=float)
    if score_type == "proba":
        logp = np.log(np.clip(s / s.sum(axis=1, keepdims=True), 1e-300, None))
    else:
        logp = s.copy()
    a = logp[:, 0] - logp[:, 1]                    # want strictly negative
    c = logp[:, 1] - logp[:, 2]                    # want strictly negative
    shift_a = a.max() + margin
    shift_c = c.max() + margin
    a2, c2 = a - shift_a, c - shift_c
    L = np.stack([a2 + c2, c2, np.zeros_like(c2)], axis=1)
    L -= L.max(axis=1, keepdims=True)
    cvec = np.exp(np.clip(np.array([-shift_a - shift_c, -shift_c, 0.0]), _LOG_TINY, None))
    cvec = cvec / cvec.sum()
    if return_log:
        return L, cvec
    if L.min() < _LOG_TINY:
        raise ValueError(
            f"the ordered gauge needs log-probabilities down to {L.min():.0f}, below "
            f"float64's floor of {_LOG_TINY:.0f}. The model's log-odds spread is too "
            "wide to represent on the probability scale. Use return_log=True and pass "
            "score_type='log' downstream -- every identity still holds exactly there."
        )
    P = np.exp(L)
    P /= P.sum(axis=1, keepdims=True)
    return P, cvec


def gauge_normalize(scores, *, reference="geometric_mean", y=None, priors=None,
                    score_type="proba"):
    """Re-centre the score matrix so that plain ``argmax`` is a sensible default again.

    Divides column ``k`` by a reference value ``r_k``.  **No number changes** -- the ROC
    surface, the VUS and every operating point are identical -- but the weight vector is
    now measured relative to ``r`` instead of relative to 1, which puts the informative
    part of the weight simplex back in the middle where it can be read.

    ``reference``:
        ``'geometric_mean'``  per-column geometric mean over the sample (model-agnostic;
                              makes the mean log-ratio zero).
        ``'prior'``           the class prevalence (from ``y`` or ``priors``).  Then
                              ``argmax`` of the normalised scores is the *maximum
                              likelihood* rule ``argmax_k p_k(x)/pi_k`` -- the classical
                              prior-shift correction.
        array-like            an explicit reference vector.

    Returns ``(P_normalized, r)``.
    """
    s = np.asarray(scores, dtype=float)
    if score_type == "proba":
        logp = np.log(np.clip(s / s.sum(axis=1, keepdims=True), 1e-300, None))
    else:
        logp = s.copy()

    if isinstance(reference, str) and reference == "geometric_mean":
        logr = logp.mean(axis=0)
    elif isinstance(reference, str) and reference == "prior":
        if priors is None:
            if y is None:
                raise ValueError("reference='prior' needs y or priors")
            y = np.asarray(y)
            classes = np.unique(y)
            priors = np.array([(y == c).mean() for c in classes], dtype=float)
        logr = np.log(np.asarray(priors, dtype=float))
    else:
        logr = np.log(np.asarray(reference, dtype=float))

    L = logp - logr[None, :]
    L -= L.max(axis=1, keepdims=True)
    P = np.exp(L)
    P /= P.sum(axis=1, keepdims=True)
    return P, np.exp(logr)


# ======================================================================================
# Reading A -- the constraint as a fact about the world
# ======================================================================================
def _free_var_polytope(priors):
    """Halfspaces ``A x <= b`` for the feasible confusion matrices, in free coordinates.

    Free variables ``x = (a0, b0, a1, b1, a2, b2)`` with ``C[k,0]=a_k``, ``C[k,1]=b_k``,
    ``C[k,2] = 1 - a_k - b_k``.  Constraints: each row is a probability vector, plus the
    logical constraints ``pi_i C[i,m] <= pi_j C[j,m]`` for ``i < j`` and every ``m``.
    """
    pi = np.asarray(priors, dtype=float)
    pi = pi / pi.sum()
    K = len(pi)
    if K != 3:
        raise ValueError("confusion_polytope currently handles 3 classes; "
                         "use pairwise_auc_ceiling for the 2-class closed form")
    rows, rhs = [], []

    def idx(k, j):                      # j = 0 -> a_k, j = 1 -> b_k
        return 2 * k + j

    for k in range(3):                                  # simplex rows
        r = np.zeros(6); r[idx(k, 0)] = -1.0; rows.append(r); rhs.append(0.0)
        r = np.zeros(6); r[idx(k, 1)] = -1.0; rows.append(r); rhs.append(0.0)
        r = np.zeros(6); r[idx(k, 0)] = 1.0; r[idx(k, 1)] = 1.0
        rows.append(r); rhs.append(1.0)

    for i, j in itertools.combinations(range(3), 2):    # logical constraints
        for m in (0, 1):
            r = np.zeros(6)
            r[idx(i, m)] = pi[i]
            r[idx(j, m)] = -pi[j]
            rows.append(r); rhs.append(0.0)
        # m = 2:  pi_i (1 - a_i - b_i) - pi_j (1 - a_j - b_j) <= 0
        r = np.zeros(6)
        r[idx(i, 0)] = -pi[i]; r[idx(i, 1)] = -pi[i]
        r[idx(j, 0)] = pi[j]; r[idx(j, 1)] = pi[j]
        rows.append(r); rhs.append(float(pi[j] - pi[i]))

    return np.array(rows), np.array(rhs), pi


#: Vertices of the order chamber ``T = {p in the simplex : p1 <= p2 <= p3}``.
#: Every ordered probability vector is a convex combination of these three, so they are
#: the most extreme -- i.e. most informative -- posteriors the constraint permits.
#: Note what they say: the best evidence for class 1 the logic allows is the *uniform*
#: posterior, and for class 2 it is a coin flip against class 3.
ORDER_CHAMBER_VERTICES = np.array([[1 / 3, 1 / 3, 1 / 3],
                                   [0.0, 1 / 2, 1 / 2],
                                   [0.0, 0.0, 1.0]])


def order_chamber_sample(lam):
    """Map barycentric coordinates ``lam`` (rows summing to 1) into the order chamber.

    ``p = lam @ ORDER_CHAMBER_VERTICES`` is ordered ``p1 <= p2 <= p3`` by construction,
    which makes it the clean way to build a data-generating process whose *true
    posterior* obeys the logical constraint.
    """
    lam = np.asarray(lam, dtype=float)
    lam = lam / lam.sum(axis=-1, keepdims=True)
    return lam @ ORDER_CHAMBER_VERTICES


def _barycentric_lattice(m: int) -> np.ndarray:
    """Regular lattice of barycentric coordinates with ``m`` subdivisions."""
    i, j = np.meshgrid(np.arange(m + 1), np.arange(m + 1), indexing="ij")
    keep = (i + j) <= m
    i, j = i[keep], j[keep]
    return np.stack([i, j, m - i - j], axis=1) / float(m)


def constrained_world(n=40_000, *, atoms=None, weights=None, n_atoms=3, concentration=0.3,
                      eps=1e-4, seed=0):
    """A data-generating process whose **true posterior** satisfies ``p1 < p2 < p3``.

    Draw a latent atom, read off its ordered posterior ``p``, then draw ``y ~ Cat(p)``.
    Because ``p`` *is* the posterior by construction, this is a genuine "Reading A" world
    and the ideal observer is ``p`` itself -- so the VUS computed on ``(y, P)`` is the
    best any model could do on this world.

    Returns ``(y, P, atoms, weights)``.
    """
    rng = np.random.default_rng(seed)
    if atoms is None:
        if isinstance(n_atoms, str) and n_atoms == "lattice":
            lam = _barycentric_lattice(30)
        elif n_atoms <= 3:
            lam = np.eye(3)[:n_atoms]
        else:
            lam = rng.dirichlet(np.full(3, concentration), size=n_atoms)
        atoms = order_chamber_sample(lam)
    atoms = np.asarray(atoms, dtype=float)
    # Nudge to make the ordering STRICT.  The chamber's own vertices have ties
    # ((1/3,1/3,1/3) and (0,1/2,1/2)), so a uniform shrink is not enough -- the offset
    # has to be increasing in k.  This keeps the rows summing to 1 and positive.
    atoms = atoms * (1 - 6 * eps) + eps * np.array([1.0, 2.0, 3.0])

    if weights is None:
        weights = np.full(len(atoms), 1.0 / len(atoms))
    weights = np.asarray(weights, dtype=float)
    weights = weights / weights.sum()

    which = rng.choice(len(atoms), size=n, p=weights)
    P = atoms[which]
    u = rng.random(n)[:, None]
    y = (u > np.cumsum(P, axis=1)).sum(axis=1)
    return y, P, atoms, weights


def confusion_polytope(priors):
    """The feasible set of confusion matrices, as ``(A, b, priors)`` with ``A x <= b``."""
    return _free_var_polytope(priors)


def _vertices(A, b, tol=1e-9):
    """Vertices of ``{x : A x <= b}`` by enumerating d-subsets of tight constraints."""
    m, d = A.shape
    out = []
    for idx in itertools.combinations(range(m), d):
        M = A[list(idx)]
        if abs(np.linalg.det(M)) < 1e-10:
            continue
        try:
            x = np.linalg.solve(M, b[list(idx)])
        except np.linalg.LinAlgError:      # pragma: no cover
            continue
        if np.all(A @ x <= b + tol):
            out.append(x)
    if not out:
        return np.zeros((0, d))
    return np.unique(np.round(np.asarray(out), 9), axis=0)


def achievable_diagonals(priors):
    """Vertices of the achievable ``(S1, S2, S3)`` region under the logical constraint.

    The set of feasible confusion matrices is a polytope; the achievable sensitivity
    triples are its projection onto the diagonal, hence the convex hull of the projected
    vertices.  **No classifier of any kind can produce an operating point outside it.**
    """
    A, b, pi = _free_var_polytope(priors)
    V = _vertices(A, b)
    if len(V) == 0:                                   # pragma: no cover
        raise RuntimeError("empty feasible polytope; are the priors strictly ordered?")
    S = np.stack([V[:, 0], V[:, 3], 1.0 - V[:, 4] - V[:, 5]], axis=1)
    S = np.clip(S, 0.0, 1.0)
    S = np.round(S, 9) + 0.0          # normalise signed zeros so -0.0 does not print
    return np.unique(S, axis=0)


def vus_ceiling(priors, *, method="exact", resolution=121) -> float:
    """Largest VUS any classifier can attain when the true posterior is ordered.

    ``method='exact'`` projects the confusion polytope and takes the volume of its
    down-set (the region is convex, so this is exact).  ``method='lp'`` integrates the
    per-``(u,v)`` linear program on a grid -- slower, used as an independent check.
    """
    if method == "exact":
        return float(convex_hull_vus(achievable_diagonals(priors)))
    if method == "lp":
        _, _, Z = ceiling_surface(priors, resolution=resolution)
        return float(Z.mean())
    raise ValueError("method must be 'exact' or 'lp'")


def ceiling_surface(priors, *, resolution=81):
    """The ceiling as a surface: ``Zmax(u, v) = max{ S3 : S1 >= u, S2 >= v }``.

    One small linear program per grid cell.  Directly comparable to
    :func:`roc3.core.monotone_envelope` of a fitted model, so the two can be drawn
    together: the model's surface and the roof above it.
    """
    from scipy.optimize import linprog

    A, b, pi = _free_var_polytope(priors)
    R = int(resolution)
    mid = (np.arange(R) + 0.5) / R
    # maximise C[2,2] = 1 - a2 - b2   <=>   minimise a2 + b2
    cost = np.zeros(6); cost[4] = 1.0; cost[5] = 1.0
    Z = np.zeros((R, R))
    for iu, u in enumerate(mid):
        for iv, v in enumerate(mid):
            Aub = np.vstack([A, -np.eye(6)[0], -np.eye(6)[3]])
            bub = np.concatenate([b, [-u], [-v]])
            res = linprog(cost, A_ub=Aub, b_ub=bub, bounds=[(0, 1)] * 6,
                          method="highs")
            Z[iu, iv] = max(0.0, 1.0 - res.fun) if res.status == 0 else 0.0
    return mid, mid, Z


def pairwise_auc_ceiling(priors) -> dict:
    """Closed-form ceiling on each pairwise AUC: ``1 - pi_i / (2 pi_j)`` for ``i < j``.

    Derivation (``docs/CONSTRAINED.md`` §3): the constraint makes the likelihood ratio
    ``f_i / f_j`` bounded above by ``R = pi_j / pi_i``, and a bounded likelihood ratio
    caps the AUC at ``1 - 1/(2R)``.  The bound is tight.
    """
    pi = np.asarray(priors, dtype=float)
    pi = pi / pi.sum()
    out = {}
    for i, j in itertools.combinations(range(len(pi)), 2):
        out[(i, j)] = float(1.0 - pi[i] / (2.0 * pi[j]))
    return out


def scalar_ceilings(priors) -> dict:
    """Ceilings on accuracy, balanced accuracy and worst-class sensitivity.

    ``accuracy`` has the closed form ``pi_3`` -- the constant rule "always predict the
    last class" is optimal and unbeatable.  The other two come from small LPs.
    """
    from scipy.optimize import linprog

    A, b, pi = _free_var_polytope(priors)
    diag = np.array([[1, 0, 0, 0, 0, 0],        # S1 = a0
                     [0, 0, 0, 1, 0, 0],        # S2 = b1
                     [0, 0, 0, 0, -1, -1]], dtype=float)   # S3 = 1 - a2 - b2
    const = np.array([0.0, 0.0, 1.0])

    def maximise(c_diag):
        cost = -(c_diag @ diag)
        off = -(c_diag @ const)
        r = linprog(cost, A_ub=A, b_ub=b, bounds=[(0, 1)] * 6, method="highs")
        return float(-(r.fun) - off) if r.status == 0 else float("nan")

    # max_min_k S_k : maximise t s.t. S_k >= t  -> add t as a 7th variable
    A7 = np.hstack([A, np.zeros((len(A), 1))])
    for k in range(3):
        row = np.concatenate([-diag[k], [1.0]])
        A7 = np.vstack([A7, row])
    b7 = np.concatenate([b, const])
    c7 = np.zeros(7); c7[6] = -1.0
    r = linprog(c7, A_ub=A7, b_ub=b7, bounds=[(0, 1)] * 7, method="highs")

    return {
        "accuracy": float(pi[2]),
        "accuracy_lp": maximise(pi),
        "balanced_accuracy": maximise(np.ones(3) / 3.0),
        "min_sensitivity": float(r.x[6]) if r.status == 0 else float("nan"),
        "youden": maximise(np.ones(3)) - 1.0,
    }


def normalized_vus(vus, priors, *, ceiling=None) -> float:
    """Rescale VUS onto ``[0, 1]`` between chance and the ceiling the constraint allows.

    ``(VUS - 1/6) / (VUS_max(pi) - 1/6)``.  0 at chance, 1 at the best any classifier
    could do given the logical constraint.  Use this instead of the raw VUS whenever the
    constraint is a fact about the world -- otherwise every model looks mediocre for a
    reason that has nothing to do with the model.
    """
    c = vus_ceiling(priors) if ceiling is None else float(ceiling)
    if c <= CHANCE_VUS:                                  # pragma: no cover
        return float("nan")
    return float((vus - CHANCE_VUS) / (c - CHANCE_VUS))


# ======================================================================================
# reporting
# ======================================================================================
def constrained_report(y, scores, *, classes=None, score_type="proba",
                       priors=None, resolution=240) -> dict:
    """Everything the ordering constraint implies for this dataset and model."""
    from .core import roc_surface
    from .metrics import hand_till_m
    from .vus import forced_choice_profile

    y_idx, logp, classes, counts = prepare_scores(
        y, scores, classes=classes, score_type=score_type
    )
    pi = counts / counts.sum() if priors is None else np.asarray(priors, float)
    pi = pi / pi.sum()

    surf = roc_surface(y_idx, logp, classes=np.arange(3), score_type="log",
                       resolution=resolution)
    fc = forced_choice_profile(y_idx, logp, classes=np.arange(3), score_type="log")
    ceil = vus_ceiling(pi)
    cm_argmax = surf.confusion_at(np.ones(3))
    se_argmax = np.diag(cm_argmax) / counts

    # Reading A implies pi_1 < pi_2 < pi_3 (sum the master lemma over the predicted
    # class).  If the observed prevalences are not ordered, the constraint cannot be a
    # fact about the world -- it is a model-side gauge, or the model is misspecified.
    priors_ordered = bool(pi[0] < pi[1] < pi[2])

    return {
        "classes": list(classes),
        "priors": pi,
        "priors_ordered": priors_ordered,
        "order_constraint": check_order_constraint(logp, score_type="log"),
        "VUS": surf.vus,
        "VUS_3AFC": fc.vus,
        "VUS_ceiling": ceil,
        "VUS_normalized": normalized_vus(surf.vus, pi, ceiling=ceil),
        "VUS_3AFC_normalized": normalized_vus(fc.vus, pi, ceiling=ceil),
        "headroom": ceil - surf.vus,
        "pairwise_auc_ceiling": pairwise_auc_ceiling(pi),
        "hand_till_M": hand_till_m(y_idx, logp, classes=np.arange(3), score_type="log"),
        "scalar_ceilings": scalar_ceilings(pi),
        "argmax_sensitivity": se_argmax,
        "argmax_accuracy": float(np.trace(cm_argmax) / cm_argmax.sum()),
        "argmax_predicts_only": (int(np.unique(np.argmax(
            logp, axis=1))[0]) if len(np.unique(np.argmax(logp, axis=1))) == 1 else None),
        "surface": surf,
    }


def format_constrained_report(rep: dict) -> str:
    cls = [str(c) for c in rep["classes"]]
    pi = rep["priors"]
    sc = rep["scalar_ceilings"]
    oc = rep["order_constraint"]
    L = ["=" * 72,
         "ORDER-CONSTRAINED 3-CLASS ROC REPORT   (0 < p1 < p2 < p3 < 1)",
         "=" * 72,
         "classes : " + ",  ".join(f"{c} (pi={p:.3f})" for c, p in zip(cls, pi)),
         "",
         "Model output obeys the constraint?  "
         f"{oc['fraction_satisfied']:.1%} of rows"
         f"   (min log margin {oc['min_log_margin']:+.2e})",
         ""]
    if not rep["priors_ordered"]:
        L += ["!! prevalences are NOT ordered pi1 < pi2 < pi3, but an ordered TRUE",
              "   posterior would force them to be.  So the constraint here is a",
              "   property of the model's output (a gauge), not of the world -- the",
              "   ceilings below do not apply.  See docs/CONSTRAINED.md section 0.",
              ""]
    if rep["argmax_predicts_only"] is not None:
        k = rep["argmax_predicts_only"]
        L += [f"!! plain argmax predicts '{cls[k]}' for EVERY sample -- the default rule "
              f"is degenerate.",
              f"   accuracy {rep['argmax_accuracy']:.4f} = prevalence of '{cls[k]}'; "
              f"per-class Se {np.round(rep['argmax_sensitivity'], 3)}",
              "   Use the ROC surface: the model is only usable through re-weighting.",
              ""]
    L += ["Volume under the ROC surface",
          f"  VUS  (geometric)               : {rep['VUS']:.4f}",
          f"  VUS  (3AFC)                    : {rep['VUS_3AFC']:.4f}",
          f"  chance                         : {CHANCE_VUS:.4f}"]
    if not rep["priors_ordered"]:
        L += ["  CEILING                        : not applicable in gauge mode",
              "",
              "  The usual [1/6, 1] scale applies unchanged -- an output-ordering",
              "  constraint is a reparameterisation and costs nothing.",
              "=" * 72]
        return "\n".join(L)
    L += [f"  CEILING from the constraint    : {rep['VUS_ceiling']:.4f}"
          f"   <- not 1.000",
          f"  head-room left                 : {rep['headroom']:.4f}",
          "",
          "Re-normalised onto [chance, ceiling]",
          f"  VUS_normalized (geometric)     : {rep['VUS_normalized']:.4f}",
          f"  VUS_normalized (3AFC)          : {rep['VUS_3AFC_normalized']:.4f}",
          "",
          "What no classifier can exceed, given the constraint",
          f"  accuracy                       : {sc['accuracy']:.4f}"
          f"   (the constant rule 'always {cls[2]}' attains it)",
          f"  balanced accuracy              : {sc['balanced_accuracy']:.4f}",
          f"  worst-class sensitivity        : {sc['min_sensitivity']:.4f}",
          f"  generalised Youden             : {sc['youden']:.4f}",
          ""]
    L.append("Pairwise AUC ceilings  1 - pi_i/(2 pi_j)")
    for (i, j), v in rep["pairwise_auc_ceiling"].items():
        L.append(f"  {cls[i]} vs {cls[j]:<16}: {v:.4f}")
    L.append(f"  observed Hand-Till M           : {rep['hand_till_M']:.4f}")
    L.append("=" * 72)
    return "\n".join(L)


# ======================================================================================
# plotting
# ======================================================================================
def plot_ceiling(surf, priors, *, theme="light", path=None, resolution=81,
                 title=None, dpi=150):
    """Three panels: the model's surface under the ceiling, the two contour maps, the gap.

    The point of the figure is the roof.  Under the logical constraint the perfect corner
    is not merely unreached, it is unreachable, and the honest question is how much of the
    *attainable* volume the model has captured.
    """
    import matplotlib
    if path is not None:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .core import monotone_envelope
    from .plots import THEMES, _cmap, _style_axes

    th = THEMES[theme] if isinstance(theme, str) else theme
    cls = [str(c) for c in surf.classes]
    u, v, Zc = ceiling_surface(priors, resolution=resolution)
    _, _, Zm = monotone_envelope(surf.S, resolution=resolution)
    ceil = float(Zc.mean())
    vus = surf.vus

    fig = plt.figure(figsize=(17.0, 5.4), facecolor=th["page"])

    ax = fig.add_subplot(1, 3, 1, projection="3d")
    U, V = np.meshgrid(u, v, indexing="ij")
    ax.plot_surface(U, V, Zc, color=th["muted"], alpha=0.30, linewidth=0, shade=False)
    ax.plot_surface(U, V, Zm, cmap=_cmap(th), vmin=0, vmax=1, linewidth=0,
                    rstride=1, cstride=1, shade=False)
    ax.scatter([1], [1], [1], color=th["accent"], s=42, depthshade=False)
    ax.text(1, 1, 1.07, "perfect\n(unreachable)", color=th["accent"], fontsize=7.5)
    ax.set_xlabel(f"$S_1$ Se[{cls[0]}]", fontsize=8.5, color=th["ink2"], labelpad=6)
    ax.set_ylabel(f"$S_2$ Se[{cls[1]}]", fontsize=8.5, color=th["ink2"], labelpad=6)
    ax.set_zlabel(f"$S_3$ Se[{cls[2]}]", fontsize=8.5, color=th["ink2"], labelpad=4)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((0, 0, 0, 0))
        a._axinfo["grid"]["color"] = th["grid"]
    ax.tick_params(colors=th["muted"], labelsize=7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.view_init(elev=24, azim=-128)
    ax.set_facecolor(th["page"])
    ax.set_title(f"model surface under the logical ceiling\n"
                 f"VUS {vus:.3f}   ceiling {ceil:.3f}",
                 fontsize=10, color=th["ink"], pad=12)

    ax = fig.add_subplot(1, 3, 2)
    cf = ax.contourf(u, v, Zc.T, levels=np.linspace(0, 1, 11), cmap=_cmap(th))
    cs = ax.contour(u, v, Zc.T, levels=[0.2, 0.4, 0.6, 0.8, 0.95],
                    colors=th["surface"], linewidths=0.9)
    ax.clabel(cs, inline=True, fontsize=7, fmt="%.2f", colors=th["surface"])
    ax.plot([0, 1], [1, 0], color=th["muted"], linewidth=1.0)
    _style_axes(ax, th, grid=False)
    ax.set_xlabel(f"required $S_1$ = Se[{cls[0]}]", fontsize=9)
    ax.set_ylabel(f"required $S_2$ = Se[{cls[1]}]", fontsize=9)
    ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"the ceiling itself\nbest Se[{cls[2]}] any classifier could reach",
                 fontsize=10, color=th["ink"], pad=8)
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(colors=th["muted"], labelsize=7)
    cb.outline.set_edgecolor(th["axis"])

    ax = fig.add_subplot(1, 3, 3)
    gap = np.clip(Zc - Zm, 0, 1)
    cf = ax.contourf(u, v, gap.T, levels=np.linspace(0, max(0.05, gap.max()), 11),
                     cmap=_cmap(th))
    ax.plot([0, 1], [1, 0], color=th["muted"], linewidth=1.0)
    _style_axes(ax, th, grid=False)
    ax.set_xlabel(f"required $S_1$ = Se[{cls[0]}]", fontsize=9)
    ax.set_ylabel(f"required $S_2$ = Se[{cls[1]}]", fontsize=9)
    ax.set_aspect("equal"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(f"what the model is leaving on the table\n"
                 f"mean gap {gap.mean():.3f}   captured "
                 f"{(vus - CHANCE_VUS) / (ceil - CHANCE_VUS):.1%} of attainable",
                 fontsize=10, color=th["ink"], pad=8)
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(colors=th["muted"], labelsize=7)
    cb.outline.set_edgecolor(th["axis"])

    fig.suptitle(title or "Logical constraint 0 < p₁ < p₂ < p₃ < 1 — the reachable part "
                          "of ROC space", fontsize=14, color=th["ink"], x=0.02,
                 ha="left", y=1.02)
    fig.tight_layout()
    if path is not None:
        fig.savefig(path, dpi=dpi, facecolor=th["page"], bbox_inches="tight")
    return fig
