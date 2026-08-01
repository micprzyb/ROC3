"""
elasticity_lab.pricelevels
==========================

The **price-test model**, transported to observational data.

Instead of regressing ``log q`` on ``log p``, cut the price into ``K`` ordered levels and
predict *which level a sold unit came from*.  Under randomisation the resulting posterior
**is** the demand curve, normalised — the identity that
:mod:`roc3.pricetest` is built on.  Observationally the seller chose the price, so the
posterior is distorted by the seller's policy and has to be divided by it.

The construction, in four steps (``docs/PLAN_PRICETEST_MODELS.md`` §2.3)::

    A   pi_k(x)  = P(level = k | x)                        a multiclass classifier, unweighted
    B   b_k(x)   = P(level = k | x, a randomly chosen SOLD UNIT)   the same, weighted by qty
    C   log s_k(x) = log b_k(x) - log pi_k(x)              the demand curve, up to a constant
    D   eps(x)   = -slope of log s_k(x) against bar_r_k    weighted least squares over k

Step C is Lemma 1: the expected units at level ``k`` are (how often that level occurs) x
(units when it does) ``= pi_k s_k``, so ``b_k = pi_k s_k / sum_j pi_j s_j`` and the ratio
``b_k / pi_k`` recovers ``s_k`` up to a factor common to all ``k`` — which is an additive
constant in logs and drops out of a slope.

**It reduces to the price test exactly.**  If ``pi_k(x) = 1/K`` the propensity cancels and
``s ∝ b``.  So the same code analyses a real randomised test and this panel; the only
difference is whether step A is estimated or known.

What this buys over the rest of the zoo
---------------------------------------
The demand curve is ``K`` free points, so **curvature is visible**: the arc elasticities
between adjacent levels need not agree, and where they do not, constant-elasticity models
are misspecified.  Nothing else in :mod:`elasticity_lab.models` can see this.

What it costs
-------------
It is a ratio estimator, so it inherits every pathology of inverse-propensity weighting:
``b_k/pi_k`` is unstable wherever a level is rare for some ``x``.  ``clip`` bounds the
damage and :func:`overlap_diagnostic` measures it.  It is also **not** Neyman-orthogonal —
errors in ``pi_hat`` enter at first order, unlike :class:`~elasticity_lab.models.DoubleMLPartialling`.

The constraint
--------------
Monotone demand — ``s_1 >= s_2 >= ... >= s_K`` for levels ordered by increasing price — is
the user's ``0 < p_1 < p_2 < p_3 < 1`` written on the right object.  It must be imposed on
the **demand**, not on the raw posterior: observationally the posterior is distorted by the
seller's pricing policy, and nothing requires *that* to be monotone.  Equivalently, the
constraint says **every arc elasticity is non-negative** — which rules out by construction
the upward-sloping demand curves the unconstrained S-learner reports for 6.7% of real
product-weeks.

Two enforcement mechanisms, so we can tell whether the benefit is in the fitting or merely
in the repair:

``constraint="softplus"``
    Fit under the constraint.  ``log s_k(x) = -sum_{j<k} softplus(a_j + w_j'x)`` is
    decreasing by construction; the multinomial likelihood is maximised with ``log pi_hat``
    as a known offset and ``qty`` as the observation weight.  L-BFGS, analytic gradient.
``constraint="isotonic"``
    Repair afterwards.  Project ``log s_hat`` onto the decreasing cone by pool-adjacent
    violators, weighted by ``pi_hat``.  Model-agnostic and exact.

References
----------
Manski & Lerman (1977) on choice-based samples — the ``pi``-division is their correction.
Elkan (2001), Saerens et al. (2002) on prior correction.  Hirano & Imbens (2004) on the
generalised propensity score.  Barlow et al. (1972) for PAVA.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .features import CONTROL_FEATURES, OUTCOME, TREATMENT
from .models import ElasticityModel, N_THREADS

__all__ = ["PriceLevels", "PriceLevelClassifier", "isotonic_decreasing",
           "overlap_diagnostic", "vus_report", "gauge_transform"]

_LOG_FLOOR = 1e-300      # never 1e-12: see IDEAS_LOG G9 — a "safe" floor is lossy


# =======================================================================================
# levels
# =======================================================================================
@dataclass
class PriceLevels:
    """Cut the relative price into ``K`` ordered levels; level 1 is the **cheapest**.

    The cut points are quantiles of ``rel_price`` — log price minus the product's own
    trailing 12-week mean — so a "level" means *this product is unusually cheap/dear for
    itself this week*, not *this product is cheap*.  That is what makes the labels
    comparable across products, and it is why per-product cut points were rejected
    (``PLAN_PRICETEST_MODELS.md`` §10).

    Cut points and level prices are learned on the training frame and applied unchanged
    afterwards, exactly like any other fitted transform.
    """

    K: int = 3
    column: str = "rel_price"
    #: name of a column holding the level directly.  Set this when the design is a real
    #: price test and the arm is **known** rather than inferred — which is the whole point
    #: of running one.  Inferring levels from ``rel_price`` on such data is actively wrong:
    #: ``rel_price`` is measured against the product's *historical* prices, so on a panel
    #: where the test price replaces the observed one it is dominated by that history and
    #: recovers the assigned arm only about 60% of the time.
    level_col: str | None = None
    #: column whose per-level mean gives the x-axis of the demand curve.  With
    #: ``level_col`` set this should be the log price actually charged.
    price_col: str = "log_price"

    def fit(self, df: pd.DataFrame) -> "PriceLevels":
        if self.level_col is not None:
            lv = df[self.level_col].to_numpy(int)
            self.K = int(lv.max()) + 1
            self.cuts_ = None
            r = df[self.price_col].to_numpy(float)
        else:
            r = df[self.column].to_numpy(float)
            qs = np.linspace(0, 1, self.K + 1)[1:-1]
            self.cuts_ = np.quantile(r, qs)
            lv = self.transform(df)
        # bar_r_k: the mean log price inside level k.  These are the x-axis of the demand
        # curve, so they must be strictly increasing or the slope is undefined.  Only the
        # DIFFERENCES matter (the level is a free additive constant), so it is legitimate
        # that this is a raw log price under level_col and a relative one otherwise.
        self.bar_r_ = np.array([r[lv == k].mean() if (lv == k).any() else np.nan
                                for k in range(self.K)])
        if not np.all(np.diff(self.bar_r_) > 0):
            raise ValueError(f"level prices are not increasing: {self.bar_r_}; "
                             f"K={self.K} is too many for this price variation")
        self.counts_ = np.bincount(lv, minlength=self.K)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """0-based level index, ordered cheapest first."""
        if self.level_col is not None:
            return df[self.level_col].to_numpy(int)
        r = df[self.column].to_numpy(float)
        return np.searchsorted(self.cuts_, r, side="right").astype(int)

    def price_multipliers(self) -> np.ndarray:
        """``exp(bar_r_k)`` — the level prices as multipliers of the reference price.

        This is what :func:`roc3.pricetest.ceiling_from_elasticity` wants.
        """
        return np.exp(self.bar_r_)

    def describe(self) -> pd.DataFrame:
        return pd.DataFrame({"level": np.arange(1, self.K + 1),
                             "n": self.counts_,
                             "mean_rel_price": self.bar_r_,
                             "price_multiplier": self.price_multipliers()})


# =======================================================================================
# helpers
# =======================================================================================
def isotonic_decreasing(v: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Weighted projection of each row of ``v`` onto ``{v_1 >= v_2 >= ... >= v_K}``.

    Computed by the **max–min formula** rather than by iterative pooling::

        v*_k  =  min_{i <= k}  max_{j >= k}  Av(i, j)

    where ``Av(i,j)`` is the weight-average of ``v[i..j]``.  This is the exact ``L2``
    projection (Barlow et al. 1972, §1.2) — so "repair" means the *closest* decreasing
    curve, not an arbitrary one.

    Why not iterative pool-adjacent-violators?  Because the bookkeeping when two blocks
    merge is easy to get wrong in a vectorised implementation, and getting it wrong is
    silent: the first version here returned ``[1.97, 2.02, 2.02]`` for input
    ``[1, 2, 3]`` — not even decreasing — and left 11.9% of rows violating the constraint
    it was supposed to enforce.  With ``K`` between 3 and 5 there are at most 15 intervals,
    so the closed form is both cheaper and checkable by eye.
    """
    v = np.asarray(v, float)
    n, K = v.shape
    w = np.ones_like(v) if w is None else np.asarray(w, float)
    cw = np.concatenate([np.zeros((n, 1)), np.cumsum(w, axis=1)], axis=1)
    cv = np.concatenate([np.zeros((n, 1)), np.cumsum(w * v, axis=1)], axis=1)

    # Av[:, i, j] = weighted mean of v[i..j] inclusive
    Av = np.full((n, K, K), np.nan)
    for i in range(K):
        for j in range(i, K):
            den = cw[:, j + 1] - cw[:, i]
            Av[:, i, j] = np.where(den > 0, (cv[:, j + 1] - cv[:, i]) / np.maximum(den, 1e-300),
                                   v[:, i])

    out = np.empty_like(v)
    for k in range(K):
        inner = np.nanmax(Av[:, :k + 1, k:], axis=2)      # max over j >= k, for each i <= k
        out[:, k] = np.min(inner, axis=1)                 # min over i <= k
    return out


def gauge_transform(proba: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Rescale class ``k`` by ``c_k`` and renormalise — the gauge of ``CONSTRAINED.md``.

    The ROC surface and every VUS estimator are **bit-for-bit invariant** under this map.
    The arc elasticities are not: each shifts by ``-(log c_{k+1} - log c_k)/(r_{k+1} - r_k)``.
    That pair of facts is the whole answer to "how does VUS relate to elasticity quality" —
    see ``PLAN_PRICETEST_MODELS.md`` §4.1 — and :func:`gauge_check` in
    ``experiments/17_vus_elasticity_link.py`` verifies both numerically.
    """
    z = np.asarray(proba, float) * np.asarray(c, float)[None, :]
    return z / z.sum(axis=1, keepdims=True)


def overlap_diagnostic(pi: np.ndarray) -> dict:
    """How badly does inverse-propensity weighting hurt here?

    ``b_k/pi_k`` is unstable wherever some level is rare for some ``x``.  The share of rows
    whose smallest propensity is tiny is the honest summary; a large share means the
    estimate is being carried by a handful of extrapolations.
    """
    mn = pi.min(axis=1)
    return {"min_propensity_p1": float(np.percentile(mn, 1)),
            "min_propensity_p5": float(np.percentile(mn, 5)),
            "min_propensity_median": float(np.median(mn)),
            "pct_below_0.02": float(np.mean(mn < 0.02)),
            "pct_below_0.05": float(np.mean(mn < 0.05))}


def _kish_ess(w: np.ndarray) -> float:
    """Kish effective sample size — how many rows the weighted fit is really using."""
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / np.sum(w ** 2)) if w.sum() > 0 else 0.0


def _lgb_multiclass(params: dict, K: int, seed: int):
    import lightgbm as lgb
    p = dict(objective="multiclass", num_class=K, verbosity=-1, n_jobs=N_THREADS,
             random_state=seed, force_col_wise=True)
    p.update(params)
    return lgb.LGBMClassifier(**p)


# =======================================================================================
# the constrained fit
# =======================================================================================
def _fit_monotone_softplus(X, log_pi, level, weight, K, l2=1e-3, max_iter=400, seed=0):
    """Maximise the unit-weighted multinomial likelihood under monotone demand.

    Parameterisation (strictly decreasing in ``k`` by construction)::

        u_j(x)      = a_j + w_j'x                    j = 1 .. K-1
        log s_1(x)  = 0
        log s_k(x)  = -sum_{j<k} softplus(u_j(x))
        z_k(x)      = log pi_k(x) + log s_k(x)       pi enters as a known OFFSET
        b_k(x)      = softmax_k(z(x))

    Loss: ``-sum_i q_i log b_{level_i}(x_i) + l2 * ||w||^2``.

    The gradient is analytic.  With ``G = dLoss/dz`` (the usual softmax cross-entropy
    residual, weighted), ``dLoss/du_j = -(sum_{k>j} G_k) * sigmoid(u_j)`` — the suffix sum
    appears because ``u_j`` lowers every level above ``j`` and none below it.
    """
    from scipy.optimize import minimize
    from scipy.special import logsumexp, expit

    n, d = X.shape
    M = K - 1
    onehot = np.zeros((n, K))
    onehot[np.arange(n), level] = 1.0
    wq = weight / weight.mean()                      # scale-free, so l2 means the same thing

    def unpack(p):
        return p[:M * d].reshape(M, d), p[M * d:]

    def obj(p):
        W, a = unpack(p)
        U = X @ W.T + a                              # (n, M)
        S = np.logaddexp(0.0, U)                     # softplus, overflow-safe
        L = np.zeros((n, K))
        L[:, 1:] = -np.cumsum(S, axis=1)
        Z = log_pi + L
        lse = logsumexp(Z, axis=1)
        loss = float(np.sum(wq * (lse - Z[np.arange(n), level]))) + l2 * float(np.sum(W ** 2))

        P = np.exp(Z - lse[:, None])
        G = wq[:, None] * (P - onehot)               # dLoss/dZ, (n, K)
        # suffix sums over k > j, for j = 0..M-1  ->  columns 1..K-1 of the reverse cumsum
        suffix = np.cumsum(G[:, ::-1], axis=1)[:, ::-1][:, 1:]      # (n, M)
        dU = -suffix * expit(U)
        gW = dU.T @ X + 2.0 * l2 * W
        ga = dU.sum(axis=0)
        return loss, np.concatenate([gW.ravel(), ga])

    rng = np.random.default_rng(seed)
    p0 = np.concatenate([0.01 * rng.standard_normal(M * d), np.zeros(M)])
    res = minimize(obj, p0, jac=True, method="L-BFGS-B",
                   options={"maxiter": max_iter, "maxfun": 4 * max_iter})
    W, a = unpack(res.x)
    return {"W": W, "a": a, "success": bool(res.success), "n_iter": int(res.nit),
            "loss": float(res.fun)}


def _monotone_log_s(X, W, a, K):
    U = X @ W.T + a
    S = np.logaddexp(0.0, U)
    L = np.zeros((X.shape[0], K))
    L[:, 1:] = -np.cumsum(S, axis=1)
    return L


# =======================================================================================
# the model
# =======================================================================================
class PriceLevelClassifier(ElasticityModel):
    """Elasticity from a price-level classifier — the price-test model, observationally.

    Parameters
    ----------
    K : number of price levels, cheapest first.  ``K=3`` mirrors a −10%/0%/+10% test.
    constraint : ``"none"`` | ``"isotonic"`` | ``"softplus"`` — see the module docstring.
    use_propensity : divide by ``pi_hat``.  ``False`` is the **ablation**: it is the literal
        transposition of the price-test model to observational data, and it is wrong,
        because it mixes demand with the seller's pricing policy.  Under randomisation the
        two coincide, so the gap between them measures the confounding directly.
    clip : floor for ``pi_hat`` before dividing.  IPW's stability knob.
    unit_weights : how to weight rows when fitting classifier **B**.

        The identity of step B needs rows weighted by units sold.  Weighting by raw ``q``
        is theory-correct and, on this panel, unusable: quantities span several orders of
        magnitude, and the Kish effective sample size came out at **0.13% of rows** — an
        effective ``n`` of about 50 out of 38,000.  Every estimate was noise.

        The fix rests on an observation: *any reweighting that is constant within* ``x``
        *leaves the conditional posterior* ``b_k(x)`` *unchanged* — it only changes which
        ``x`` the classifier spends its capacity on.  So dividing by the product's mean
        quantity is (near enough) estimand-preserving for ``eps(x)`` while restoring the
        effective sample size.

        It does change the implicit weighting of the *average*: from unit-weighted to
        product-weighted.  That is the right change here, because the oracle mean this is
        graded against, ``true_elasticity.mean()``, is a **row** average, not a unit
        average (see ``MODELS.md`` §2 on the three quantities called "elasticity").

        ``"per_product"``  ``q / mean_i(q)``   — the default
        ``"units"``        raw ``q``           — theory-correct, unusable here
        ``"log"``          ``log1p(q)``        — a blunter variance fix, biased
        ``"none"``         all ones            — the ablation: ignores demand entirely
    weight_cap : winsorise the unit weights at this quantile (``None`` for no cap).
    level_weights : weights of the levels in the slope of step D.  ``"uniform"`` treats the
        demand curve as K equally-informative points; ``"propensity"`` weights each level by
        how often it actually occurs.
    """

    name = "pricelevel_clf"

    def __init__(self, K: int = 3, constraint: str = "none", level_col: str | None = None,
                 use_propensity: bool = True,
                 clip: float = 0.02, unit_weights: str = "per_product",
                 weight_cap: float | None = 0.99, level_weights: str = "uniform",
                 n_estimators: int = 300, learning_rate: float = 0.05,
                 num_leaves: int = 31, min_child_samples: int = 40,
                 subsample: float = 0.9, colsample_bytree: float = 0.8,
                 reg_lambda: float = 1.0, mono_l2: float = 1e-3, seed: int = 0):
        super().__init__(K=K, constraint=constraint, level_col=level_col,
                         use_propensity=use_propensity,
                         clip=clip, unit_weights=unit_weights, weight_cap=weight_cap,
                         level_weights=level_weights,
                         n_estimators=n_estimators, learning_rate=learning_rate,
                         num_leaves=num_leaves, min_child_samples=min_child_samples,
                         subsample=subsample, colsample_bytree=colsample_bytree,
                         reg_lambda=reg_lambda, mono_l2=mono_l2, seed=seed)

    # ---------------------------------------------------------------- internals
    def _lgb_params(self):
        p = {k: self.params[k] for k in
             ("n_estimators", "learning_rate", "num_leaves", "min_child_samples",
              "subsample", "colsample_bytree", "reg_lambda")}
        p["subsample_freq"] = 1
        return p

    def _X(self, df):
        return df[CONTROL_FEATURES].to_numpy(float)

    def _unit_weights(self, df):
        q = np.exp(df[OUTCOME].to_numpy(float))          # back to units
        scheme = self.params["unit_weights"]
        if scheme == "units":
            w = q
        elif scheme == "per_product":
            # divide by the product's own mean: constant within x, so b_k(x) is unchanged
            m = pd.Series(q).groupby(df["stock_code"].to_numpy()).transform("mean")
            w = q / np.maximum(m.to_numpy(float), 1e-12)
        elif scheme == "log":
            w = np.log1p(q)
        elif scheme == "none":
            w = np.ones_like(q)
        else:
            raise ValueError(f"unknown unit_weights={scheme!r}")
        cap = self.params["weight_cap"]
        if cap is not None:
            w = np.minimum(w, np.quantile(w, cap))
        return np.maximum(w, 1e-12)

    # ---------------------------------------------------------------- fit
    def fit(self, df: pd.DataFrame):
        self.levels_ = PriceLevels(K=self.params["K"],
                                   level_col=self.params["level_col"]).fit(df)
        K = self.levels_.K
        lv = self.levels_.transform(df)
        X = self._X(df)
        wq = self._unit_weights(df)
        self.ess_ = _kish_ess(wq)
        self.n_train_ = len(df)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # A -- the propensity: how often does the seller put this row at each level?
            self.clf_pi_ = _lgb_multiclass(self._lgb_params(), K, self.params["seed"])
            self.clf_pi_.fit(X, lv)
            # B -- the unit-weighted posterior: where did a randomly chosen SOLD UNIT
            # come from?  This is the choice-based sample of Manski & Lerman.
            self.clf_b_ = _lgb_multiclass(self._lgb_params(), K, self.params["seed"] + 1)
            self.clf_b_.fit(X, lv, sample_weight=wq)
            # a level-averaged outcome model, only so the class can predict log q at all
            self.base_ = self._fit_base(X, df[OUTCOME].to_numpy(float))

        if self.params["constraint"] == "softplus":
            self.mu_ = X.mean(0)
            self.sd_ = np.where(X.std(0) > 1e-12, X.std(0), 1.0)
            Z = (X - self.mu_) / self.sd_
            log_pi = np.log(self._propensity(X))
            self.mono_ = _fit_monotone_softplus(
                Z, log_pi, lv, wq, K, l2=self.params["mono_l2"], seed=self.params["seed"])

        self.avg_elasticity_ = float(np.mean(self.elasticity(df)))
        self.fitted_ = True
        return self

    def _fit_base(self, X, y):
        import lightgbm as lgb
        p = dict(objective="regression", verbosity=-1, n_jobs=N_THREADS,
                 random_state=self.params["seed"], force_col_wise=True)
        p.update(self._lgb_params())
        return lgb.LGBMRegressor(**p).fit(X, y)

    # ---------------------------------------------------------------- pieces
    def _propensity(self, X):
        pi = self.clf_pi_.predict_proba(X)
        c = self.params["clip"]
        if c and c > 0:
            pi = np.clip(pi, c, None)
            pi = pi / pi.sum(axis=1, keepdims=True)
        return pi

    def posterior(self, df: pd.DataFrame) -> np.ndarray:
        """``b_k(x)`` — the unit-weighted class posterior.  This is what VUS scores."""
        X = self._X(df)
        if self.params["constraint"] == "softplus":
            from scipy.special import softmax
            Z = (X - self.mu_) / self.sd_
            log_s = _monotone_log_s(Z, self.mono_["W"], self.mono_["a"], self.levels_.K)
            return softmax(np.log(self._propensity(X)) + log_s, axis=1)
        return np.clip(self.clf_b_.predict_proba(X), _LOG_FLOOR, None)

    def demand_curve(self, df: pd.DataFrame) -> np.ndarray:
        """``log s_k(x)`` for every row — the demand curve, up to an additive constant.

        Step C.  Returns an ``(n, K)`` array whose rows are, under the constraint,
        decreasing.
        """
        X = self._X(df)
        K = self.levels_.K
        if self.params["constraint"] == "softplus":
            Z = (X - self.mu_) / self.sd_
            return _monotone_log_s(Z, self.mono_["W"], self.mono_["a"], K)

        b = np.clip(self.clf_b_.predict_proba(X), _LOG_FLOOR, None)
        log_s = np.log(b)
        if self.params["use_propensity"]:
            log_s = log_s - np.log(self._propensity(X))
        if self.params["constraint"] == "isotonic":
            log_s = isotonic_decreasing(log_s, self._propensity(X))
        # centre each row: the common factor of Lemma 1 is not identified and must not be
        # allowed to masquerade as signal
        return log_s - log_s.mean(axis=1, keepdims=True)

    def _level_weights(self, X):
        if self.params["level_weights"] == "propensity":
            return self._propensity(X)
        return np.ones((X.shape[0], self.levels_.K)) / self.levels_.K

    def arc_elasticities(self, df: pd.DataFrame) -> np.ndarray:
        """``(n, K-1)`` — the elasticity between each adjacent pair of levels.

        Non-constant columns mean the demand curve is **curved**, which every
        constant-elasticity model in the zoo assumes away.  Under
        ``constraint != "none"`` every entry is guaranteed ``>= 0``.
        """
        log_s = self.demand_curve(df)
        gaps = np.diff(self.levels_.bar_r_)
        return -np.diff(log_s, axis=1) / gaps[None, :]

    # ---------------------------------------------------------------- interface
    def elasticity(self, df: pd.DataFrame) -> np.ndarray:
        """Step D: the negative slope of ``log s_k`` on ``bar_r_k``, weighted over ``k``."""
        log_s = self.demand_curve(df)
        r = self.levels_.bar_r_[None, :]
        w = self._level_weights(self._X(df))
        rbar = (w * r).sum(1, keepdims=True) / w.sum(1, keepdims=True)
        sbar = (w * log_s).sum(1, keepdims=True) / w.sum(1, keepdims=True)
        num = (w * (r - rbar) * (log_s - sbar)).sum(1)
        den = (w * (r - rbar) ** 2).sum(1)
        return -num / np.maximum(den, 1e-12)

    def predict_log_qty(self, df: pd.DataFrame) -> np.ndarray:
        """``E[y|X]`` plus this row's own level's deviation from the level-average demand."""
        X = self._X(df)
        log_s = self.demand_curve(df)
        pi = self._propensity(X)
        dev = log_s - (pi * log_s).sum(axis=1, keepdims=True)
        lv = self.levels_.transform(df)
        return self.base_.predict(X) + dev[np.arange(len(df)), lv]

    # ---------------------------------------------------------------- diagnostics
    def diagnostics(self, df: pd.DataFrame) -> dict:
        X = self._X(df)
        pi = self._propensity(X)
        arcs = self.arc_elasticities(df)
        out = {"K": self.levels_.K, "constraint": self.params["constraint"],
               "kish_ess": self.ess_, "kish_ess_frac": self.ess_ / max(self.n_train_, 1),
               "pct_arc_negative": float(np.mean(arcs < 0)),
               "arc_mean": arcs.mean(0).tolist()}
        out |= overlap_diagnostic(pi)
        if self.params["constraint"] == "softplus":
            out["mono_converged"] = self.mono_["success"]
            out["mono_iters"] = self.mono_["n_iter"]
        return out


# =======================================================================================
# VUS scoring
# =======================================================================================
def vus_report(model: PriceLevelClassifier, df: pd.DataFrame, *,
               resolution: int = 200) -> dict:
    """Score the classifier with the `roc3` machinery, and put the number in context.

    A **raw VUS is not interpretable on this task.**  Monotone demand is a fact about the
    world, so it caps the attainable VUS at a value that depends on the *level* of the
    elasticity: at zero elasticity the levels are indistinguishable in principle and the
    ceiling is exactly chance; the stronger the demand response, the more room there is.
    The only comparable number is therefore the position within ``[chance, ceiling]``,
    which is what ``normalized`` reports and what
    ``roc3.pricetest.audit`` calls ``fraction_of_attainable``.

    For ``K = 3`` the full ceiling machinery applies.  For ``K > 3`` there is no ceiling
    implementation, so the raw HUM and its chance level ``1/K!`` are reported and
    ``normalized`` is ``nan`` — stated rather than silently approximated.
    """
    import math

    import roc3
    from roc3.vus import hum

    K = model.levels_.K
    lv = model.levels_.transform(df)
    proba = model.posterior(df)
    out = {"K": K, "n": len(df), "chance": 1.0 / math.factorial(K)}

    h = hum(lv, proba)                    # returns a dict, not a float
    out["hum"] = float(h["hum"])
    out["hum_adjusted"] = float(h["hum_adjusted"])
    out["hum_stderr"] = float(h["stderr"])

    # The constraint is on DEMAND, not on the posterior (PLAN §3.1), so that is what gets
    # checked.  Equivalently: is every arc elasticity non-negative?
    arcs = model.arc_elasticities(df)
    out["monotone_rows_pct"] = 100.0 * float(np.mean(np.all(arcs >= -1e-12, axis=1)))
    out["pct_arc_negative"] = float(np.mean(arcs < 0))

    if K == 3:
        surf = roc3.roc_surface(lv, proba, resolution=resolution)
        out["vus_geometric"] = float(surf.vus)
        out["vus_3afc"] = float(roc3.vus_forced_choice(lv, proba))

        # The ceiling, computed from the demand response this model itself measured.  This
        # is the step that makes the number mean something: `audit` also flags a VUS above
        # the ceiling, which cannot happen honestly and indicates leakage.
        from roc3.pricetest import audit, ceiling_from_purchase_rates
        s = np.exp(model.demand_curve(df).mean(axis=0))      # mean demand curve, level order
        s = s / s.sum()
        pi_bar = model._propensity(model._X(df)).mean(axis=0)
        info = ceiling_from_purchase_rates(s, pi_bar)
        a = audit(out["vus_3afc"], s, pi_bar)
        out["vus_ceiling"] = float(info["vus_ceiling"])
        out["normalized"] = float(a["fraction_of_attainable"])
        out["verdict"] = a["verdict"]
        out["implied_demand_ratios"] = info["demand_ratios"]
    else:
        out["vus_ceiling"] = float("nan")
        out["normalized"] = float("nan")
        out["verdict"] = f"no ceiling implementation for K={K} (only K=3)"
    return out
