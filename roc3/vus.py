"""
roc3.vus
========

Rank-based ("forced choice") estimators of the volume under the ROC surface.

Mossman (1999): *the volume under the ROC surface equals the probability that the test
values let a decision maker correctly sort a trio of items containing a randomly selected
member from each of the three populations.*  That is the quantity computed here, and it
requires no threshold grid at all.

Formally, draw one case from each class, ``x1 in C1, x2 in C2, x3 in C3``, shuffle them,
and assign the three labels bijectively so as to maximise ``prod_k p_{sigma(k)}(x_k)``.

    VUS_3AFC = P( the maximising permutation is the identity )

Ties among maximising permutations share the credit equally.  Summing over all six
permutations gives 1, so the six win-probabilities form a *3AFC confusion profile*.

Two equivalent formulations are implemented and cross-validated against each other:

``method='exact'``
    Tie-aware argmax over the six permutation scores, over all ``n1*n2*n3`` triples.

``method='exact_fast'``
    The algebraic characterisation (assumes no exact ties).  With
    ``A = l1-l2, B = l1-l3, C = l2-l3`` the identity wins iff

        A(x1) > A(x2),   C(x2) > C(x3),   B(x1) > B(x3),
        A(x1) + C(x2) > B(x3),   B(x1) > A(x2) + C(x3).

    Conditions 1 and 3 depend on only two of the three indices, which allows aggressive
    pruning; this path is typically 5-20x faster.

``method='mc'``
    Monte-Carlo over sampled triples (the default for large samples).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from .core import CHANCE_VUS, prepare_scores

__all__ = [
    "PERMUTATIONS",
    "ForcedChoiceResult",
    "forced_choice_profile",
    "vus_forced_choice",
    "hum",
]

#: The six label assignments; index 0 is the identity ``(0, 1, 2)``.
PERMUTATIONS = list(itertools.permutations(range(3)))
_IDENTITY = 0


@dataclass
class ForcedChoiceResult:
    """Outcome of the 3-alternative-forced-choice analysis."""

    vus: float
    profile: np.ndarray            # (6,) win probability of each permutation, sums to 1
    permutations: list
    method: str
    n_triples: int
    mc_stderr: float | None = None

    @property
    def vus_adjusted(self) -> float:
        return (self.vus - CHANCE_VUS) / (1.0 - CHANCE_VUS)

    def profile_table(self, classes=None) -> str:
        names = list(classes) if classes is not None else [0, 1, 2]
        lines = ["3AFC profile  (assignment of the trio -> probability)"]
        order = np.argsort(-self.profile)
        for t in order:
            p = PERMUTATIONS[t]
            tag = "  <- correct" if t == _IDENTITY else ""
            lines.append(
                f"  ({names[0]},{names[1]},{names[2]}) -> "
                f"({names[p[0]]},{names[p[1]]},{names[p[2]]})   {self.profile[t]:.4f}{tag}"
            )
        return "\n".join(lines)


def _split_by_class(y_idx, logp):
    return [logp[y_idx == k] for k in range(3)]


def _exact_profile(L0, L1, L2, max_block=500_000):
    n0, n1, n2 = len(L0), len(L1), len(L2)
    acc = np.zeros(6)
    jc = max(1, int(max_block // max(n2, 1)))
    S = None
    for i in range(n0):
        li = L0[i]
        for jlo in range(0, n1, jc):
            Lj = L1[jlo : jlo + jc]
            m = len(Lj)
            if S is None or S.shape[1:] != (m, n2):
                S = np.empty((6, m, n2))
            for t, (a, b, c) in enumerate(PERMUTATIONS):
                np.add(Lj[:, b][:, None], L2[:, c][None, :], out=S[t])
                S[t] += li[a]
            mx = S.max(axis=0)
            eq = S == mx
            cnt = eq.sum(axis=0)
            acc += (eq / cnt).sum(axis=(1, 2))
    return acc / (n0 * n1 * n2)


def _mc_profile(L0, L1, L2, n_samples, seed):
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(L0), n_samples)
    j = rng.integers(0, len(L1), n_samples)
    k = rng.integers(0, len(L2), n_samples)
    Li, Lj, Lk = L0[i], L1[j], L2[k]
    S = np.empty((6, n_samples))
    for t, (a, b, c) in enumerate(PERMUTATIONS):
        S[t] = Li[:, a] + Lj[:, b] + Lk[:, c]
    mx = S.max(axis=0)
    eq = S == mx
    cnt = eq.sum(axis=0)
    prof = (eq / cnt).sum(axis=1) / n_samples
    p = prof[_IDENTITY]
    se = float(np.sqrt(max(p * (1.0 - p), 1e-12) / n_samples))
    return prof, se


def _exact_fast_vus(L0, L1, L2):
    """Strict-inequality count using the five algebraic conditions (no tie handling)."""
    A0, B0 = L0[:, 0] - L0[:, 1], L0[:, 0] - L0[:, 2]
    A1, C1 = L1[:, 0] - L1[:, 1], L1[:, 1] - L1[:, 2]
    B2, C2 = L2[:, 0] - L2[:, 2], L2[:, 1] - L2[:, 2]
    n0, n1, n2 = len(L0), len(L1), len(L2)
    total = 0
    for i in range(n0):
        mj = A0[i] > A1
        if not mj.any():
            continue
        mk = B0[i] > B2
        if not mk.any():
            continue
        Aj, Cj = A1[mj], C1[mj]
        Bk, Ck = B2[mk], C2[mk]
        ok = Cj[:, None] > Ck[None, :]
        ok &= (Cj[:, None] + A0[i]) > Bk[None, :]
        ok &= B0[i] > (Aj[:, None] + Ck[None, :])
        total += int(ok.sum())
    return total / (n0 * n1 * n2)


def forced_choice_profile(
    y,
    scores,
    *,
    classes=None,
    score_type="proba",
    method="auto",
    n_samples=500_000,
    seed=0,
    exact_limit=2_000_000,
) -> ForcedChoiceResult:
    """Full 3AFC analysis: VUS plus the six permutation win-probabilities.

    Parameters
    ----------
    method : {'auto', 'exact', 'exact_fast', 'mc'}
        ``'auto'`` uses ``'exact'`` when ``n1*n2*n3 <= exact_limit`` and ``'mc'`` otherwise.
    n_samples : int
        Number of sampled triples for the Monte-Carlo path.
    exact_limit : int
        Triple-count above which ``'auto'`` switches to Monte Carlo.
    """
    y_idx, logp, classes, counts = prepare_scores(
        y, scores, classes=classes, score_type=score_type
    )
    if len(classes) != 3:
        raise ValueError("forced_choice_profile requires exactly 3 classes; see hum()")
    L0, L1, L2 = _split_by_class(y_idx, logp)
    total = int(counts.prod())

    if method == "auto":
        method = "exact" if total <= exact_limit else "mc"

    if method == "exact":
        prof, se = _exact_profile(L0, L1, L2), None
    elif method == "mc":
        prof, se = _mc_profile(L0, L1, L2, n_samples, seed)
    elif method == "exact_fast":
        v = _exact_fast_vus(L0, L1, L2)
        prof = np.full(6, np.nan)
        prof[_IDENTITY] = v
        se = None
    else:
        raise ValueError("method must be 'auto', 'exact', 'exact_fast' or 'mc'")

    return ForcedChoiceResult(
        vus=float(prof[_IDENTITY]),
        profile=prof,
        permutations=PERMUTATIONS,
        method=method,
        n_triples=total,
        mc_stderr=se,
    )


def vus_forced_choice(y, scores, **kwargs) -> float:
    """Scalar 3AFC VUS.  Perfect 1, uninformative 1/6, adversarial 0."""
    return forced_choice_profile(y, scores, **kwargs).vus


# --------------------------------------------------------------------------------------
# generic k-class hypervolume under the manifold
# --------------------------------------------------------------------------------------
def hum(y, scores, *, classes=None, score_type="proba", n_samples=400_000, seed=0):
    """Hypervolume Under the Manifold for any number of classes ``C >= 2``.

    Generalises :func:`vus_forced_choice`: draw one case per class and assign the ``C``
    labels bijectively to maximise the total log score (a linear assignment problem).
    ``HUM`` is the probability that the identity assignment wins.

    Chance level is ``1/C!``; perfect is 1.  For ``C = 2`` this is exactly the AUC and for
    ``C = 3`` exactly the VUS.  There is no surface to plot beyond ``C = 3`` -- this is the
    scalar that survives.

    Returns
    -------
    dict with keys ``hum``, ``chance``, ``hum_adjusted``, ``stderr``.
    """
    y_idx, logp, classes, counts = prepare_scores(
        y, scores, classes=classes, score_type=score_type
    )
    C = len(classes)
    rng = np.random.default_rng(seed)
    by_class = [logp[y_idx == k] for k in range(C)]

    idx = [rng.integers(0, len(by_class[k]), n_samples) for k in range(C)]
    # M[s, r, c] = log score of class c for the case drawn from class r in sample s
    M = np.stack([by_class[k][idx[k]] for k in range(C)], axis=1)

    if C <= 7:
        perms = np.array(list(itertools.permutations(range(C))))
        rows = np.arange(C)
        S = np.stack([M[:, rows, p].sum(axis=1) for p in perms], axis=0)
        mx = S.max(axis=0)
        eq = S == mx
        cnt = eq.sum(axis=0)
        identity = int(np.flatnonzero((perms == rows).all(axis=1))[0])
        credit = eq[identity] / cnt
    else:  # pragma: no cover - exercised only for very many classes
        from scipy.optimize import linear_sum_assignment

        credit = np.empty(n_samples)
        for s in range(n_samples):
            r, c = linear_sum_assignment(M[s], maximize=True)
            credit[s] = float(np.array_equal(c, np.arange(C)))

    est = float(credit.mean())
    chance = 1.0 / _fact(C)
    return {
        "hum": est,
        "chance": chance,
        "hum_adjusted": (est - chance) / (1.0 - chance),
        "stderr": float(credit.std(ddof=1) / np.sqrt(n_samples)),
        "n_classes": C,
        "n_samples": n_samples,
    }


def _fact(n: int) -> float:
    out = 1.0
    for i in range(2, n + 1):
        out *= i
    return out
