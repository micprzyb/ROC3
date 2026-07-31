"""
roc3.datasets
=============

Synthetic 3-class problems with *known* ground truth, used for validation and demos.

The Gaussian generator returns the **true posterior probabilities** as well as the data,
so the estimators can be checked against an exact ideal observer rather than against a
fitted model.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "make_gaussian_3class",
    "make_perfect",
    "make_uninformative",
    "make_adversarial",
    "make_ordinal_3class",
    "make_asymmetric_clinical",
    "wine_probabilities",
]


def _softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def make_gaussian_3class(
    n_per_class=300,
    separation=2.0,
    *,
    n_features=2,
    priors=None,
    seed=0,
    means=None,
    return_true_posterior=True,
):
    """Three isotropic Gaussians on the vertices of an equilateral triangle.

    Returns ``(X, y, P)`` where ``P`` are the **exact** posterior probabilities (so the
    scorer is the ideal observer) unless ``return_true_posterior=False``, in which case
    ``P`` is ``None``.

    ``separation`` scales the distance between class means; 0 gives an uninformative
    problem (VUS -> 1/6), large values give a nearly perfect one (VUS -> 1).
    """
    rng = np.random.default_rng(seed)
    if means is None:
        ang = np.array([0.0, 2 * np.pi / 3, 4 * np.pi / 3])
        base = np.stack([np.cos(ang), np.sin(ang)], axis=1) * separation
        means = np.zeros((3, n_features))
        means[:, : min(2, n_features)] = base[:, : min(2, n_features)]
    means = np.asarray(means, dtype=float)

    if priors is None:
        counts = np.full(3, int(n_per_class))
    else:
        pri = np.asarray(priors, dtype=float)
        counts = np.maximum(1, np.round(pri / pri.sum() * 3 * n_per_class).astype(int))

    X = np.vstack([rng.normal(means[k], 1.0, size=(counts[k], n_features))
                   for k in range(3)])
    y = np.repeat(np.arange(3), counts)

    P = None
    if return_true_posterior:
        logpri = np.log(counts / counts.sum())
        # log N(x; mu_k, I) up to a constant common to all k
        z = X @ means.T - 0.5 * (means ** 2).sum(axis=1)[None, :] + logpri[None, :]
        P = _softmax(z)
    return X, y, P


def make_perfect(n_per_class=200, *, confidence=0.999, seed=0):
    """Scores that identify the class with certainty.  Every VUS estimator must return 1."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(3), n_per_class)
    P = np.full((len(y), 3), (1.0 - confidence) / 2.0)
    P[np.arange(len(y)), y] = confidence
    P += rng.random(P.shape) * 1e-9          # break exact ties
    return y, P / P.sum(axis=1, keepdims=True)


def make_uninformative(n_per_class=200, *, seed=0):
    """Scores drawn independently of the label.  Every VUS estimator must return ~1/6."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(3), n_per_class)
    P = rng.dirichlet(np.ones(3), size=len(y))
    return y, P


def make_adversarial(n_per_class=200, *, confidence=0.98, seed=0):
    """Cyclically wrong: class k is confidently labelled ``k+1``.  VUS should be ~0."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(3), n_per_class)
    P = np.full((len(y), 3), (1.0 - confidence) / 2.0)
    P[np.arange(len(y)), (y + 1) % 3] = confidence
    P += rng.random(P.shape) * 1e-9
    return y, P / P.sum(axis=1, keepdims=True)


def make_ordinal_3class(n_per_class=250, *, separation=1.4, seed=0, noise=1.0):
    """A single latent marker with three ordered classes (mild < moderate < severe)."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(3), n_per_class)
    s = rng.normal(loc=y * separation, scale=noise)
    return y, s


def make_asymmetric_clinical(n=(600, 400, 200), *, seed=0):
    """A deliberately lopsided, realistic-looking 3-class problem.

    Class 0 ("benign") is common and easy; class 1 ("indeterminate") is the middle,
    overlapping heavily with both neighbours; class 2 ("malignant") is rare and the one
    you must not miss.  Returns ``(X, y, P_true, class_names)``.

    This is the demo case: the plain ``argmax`` rule sacrifices the rare class, and the
    surface shows exactly what re-weighting buys and costs.
    """
    rng = np.random.default_rng(seed)
    means = np.array([[0.0, 0.0], [1.6, 0.5], [2.4, 2.1]])
    covs = np.array([
        [[1.0, 0.2], [0.2, 1.0]],
        [[1.5, 0.6], [0.6, 1.3]],
        [[1.1, -0.3], [-0.3, 1.4]],
    ])
    n = np.asarray(n, dtype=int)
    X = np.vstack([rng.multivariate_normal(means[k], covs[k], size=n[k]) for k in range(3)])
    y = np.repeat(np.arange(3), n)

    logpri = np.log(n / n.sum())
    logdens = np.empty((len(X), 3))
    for k in range(3):
        d = X - means[k]
        Si = np.linalg.inv(covs[k])
        logdens[:, k] = (-0.5 * np.einsum("ij,jk,ik->i", d, Si, d)
                         - 0.5 * np.log(np.linalg.det(covs[k])) + logpri[k])
    return X, y, _softmax(logdens), np.array(["benign", "indeterminate", "malignant"])


def wine_probabilities(*, seed=0, n_splits=5, model=None,
                       features=("alcohol", "alcalinity_of_ash")):
    """Out-of-fold predicted probabilities for the classic 3-class *wine* dataset.

    Returns ``(y, P, class_names)``.  Predictions are cross-validated, so the scores are
    honest.  By default only two of the thirteen features are used: with all thirteen the
    problem is essentially separable, the surface collapses onto the perfect corner and
    there is nothing to look at.  Pass ``features=None`` for the full feature set.
    """
    from sklearn.datasets import load_wine
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    data = load_wine()
    X, y = data.data, data.target
    if features is not None:
        cols = [list(data.feature_names).index(f) for f in features]
        X = X[:, cols]
    if model is None:
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=5000, random_state=seed),
        )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    P = cross_val_predict(model, X, y, cv=cv, method="predict_proba")
    return y, P, np.asarray(data.target_names)
