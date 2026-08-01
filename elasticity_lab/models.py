"""
elasticity_lab.models
=====================

Seven estimators of price elasticity, sharing one interface, spanning classical
econometrics through to causal machine learning.

======================================  ==========================================================
:class:`PooledLogLog`                   OLS of log q on log p.  The baseline that is always wrong.
:class:`TwoWayFixedEffects`             Product and week fixed effects; uses within-product variation only.
:class:`PoissonGLM`                     Count model with a log link, which is what quantities actually are.
:class:`LightGBMDemand`                 Flexible demand surface; elasticity by numerical differentiation (an S-learner).
:class:`DoubleMLPartialling`            Residualise y and p on controls with ML, then regress residual on residual.
:class:`RLearnerHeterogeneous`          The same orthogonal score, but the final stage learns eps(x).
:class:`ShrunkPerProduct`               Per-product slopes pulled toward the pooled mean by empirical Bayes.
======================================  ==========================================================

The common interface
--------------------
``fit(df)`` → ``self``; ``predict_log_qty(df)`` → array; ``elasticity(df)`` → per-row array;
``average_elasticity`` → float.  Every model reports elasticity as a **positive** number
meaning "a 1% price rise costs this many percent of demand".

Two implementation points that are easy to get wrong
----------------------------------------------------
**Perturbing price consistently.**  ``rel_price``, ``price_change`` and ``is_discounted``
are all functions of ``log_price``.  A finite-difference elasticity that moves
``log_price`` while leaving those fixed measures the wrong derivative — it holds the
discount depth constant while changing the price, which is not a price change at all.
:func:`perturb_price` moves all of them together.

**Cross-fitting.**  The orthogonal estimators need out-of-fold nuisance predictions or
they inherit the nuisance model's overfitting as bias.  Here the folds are formed **by
product**, so a nuisance model never sees the SKU it is predicting for; that is the
version of sample splitting appropriate to a panel.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from .features import CONTROL_FEATURES, PRICE_FEATURES, OUTCOME, TREATMENT

__all__ = ["ElasticityModel", "PooledLogLog", "TwoWayFixedEffects", "PoissonGLM",
           "LightGBMDemand", "DoubleMLPartialling", "RLearnerHeterogeneous",
           "ShrunkPerProduct", "perturb_price", "MODEL_REGISTRY", "build_model",
           "N_THREADS"]


# =======================================================================================
# helpers
# =======================================================================================
def perturb_price(df: pd.DataFrame, delta: float) -> pd.DataFrame:
    """Shift log price by ``delta``, moving every price-derived feature with it.

    Without this a finite-difference elasticity silently conditions on the discount depth
    while varying the price, and reports a derivative nobody asked for.
    """
    out = df.copy()
    out["log_price"] = out["log_price"] + delta
    out["rel_price"] = out["rel_price"] + delta
    out["price_change"] = out["price_change"] + delta
    out["is_discounted"] = (out["rel_price"] < -0.02).astype(float)
    return out


def _product_folds(df: pd.DataFrame, k: int, seed: int = 0) -> np.ndarray:
    codes = pd.unique(df["stock_code"])
    rng = np.random.default_rng(seed)
    assign = dict(zip(codes, rng.permutation(len(codes)) % k))
    return df["stock_code"].map(assign).to_numpy()


#: LightGBM threads.  **Not** ``-1``.  On a 24-core host ``n_jobs=-1`` measured 30.7 s for
#: a fit that takes 0.09 s at ``n_jobs=8`` — a 340x slowdown from OpenMP oversubscription
#: on data this small (~20k rows).  The trees are tiny; the threads spend their time
#: synchronising.  Override with the ``ELASTICITY_LAB_THREADS`` environment variable.
N_THREADS = int(os.environ.get("ELASTICITY_LAB_THREADS",
                               max(1, min(8, (os.cpu_count() or 8) // 2))))


def _lgb(params: dict, seed: int):
    import lightgbm as lgb
    p = dict(objective="regression", verbosity=-1, n_jobs=N_THREADS, random_state=seed,
             force_col_wise=True)
    p.update(params)
    return lgb.LGBMRegressor(**p)


# =======================================================================================
# base
# =======================================================================================
class ElasticityModel:
    """Common interface.  ``name`` and ``params`` are used by the tuning layer."""

    name = "base"

    def __init__(self, **params):
        self.params = params
        self.fitted_ = False

    def fit(self, df: pd.DataFrame) -> "ElasticityModel":  # pragma: no cover
        raise NotImplementedError

    def predict_log_qty(self, df: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def elasticity(self, df: pd.DataFrame) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    @property
    def average_elasticity(self) -> float:
        return float(getattr(self, "avg_elasticity_", np.nan))

    def __repr__(self) -> str:
        return f"{self.name}({', '.join(f'{k}={v}' for k, v in self.params.items())})"


# =======================================================================================
# 1. pooled log-log OLS
# =======================================================================================
class PooledLogLog(ElasticityModel):
    """``log q = a + b·log p + c'X``.  Constant elasticity, no product structure.

    With ``use_controls=False`` this is the textbook two-variable regression and is
    biased by every omitted channel at once (see
    :func:`elasticity_lab.simulate.naive_bias_decomposition`).  It is here as the
    reference point that everything else has to beat.
    """

    name = "pooled_loglog"

    def __init__(self, use_controls: bool = True, ridge: float = 0.0):
        super().__init__(use_controls=use_controls, ridge=ridge)

    def _X(self, df):
        cols = [TREATMENT] + (list(CONTROL_FEATURES) if self.params["use_controls"] else [])
        return df[cols].to_numpy(float), cols

    def fit(self, df):
        import statsmodels.api as sm
        X, self.cols_ = self._X(df)
        y = df[OUTCOME].to_numpy(float)
        Xc = sm.add_constant(X, has_constant="add")
        if self.params["ridge"] > 0:
            self.res_ = sm.OLS(y, Xc).fit_regularized(alpha=self.params["ridge"], L1_wt=0.0)
            self.beta_ = np.asarray(self.res_.params, float)
        else:
            self.res_ = sm.OLS(y, Xc).fit()
            self.beta_ = np.asarray(self.res_.params, float)
        self.avg_elasticity_ = -self.beta_[1]
        self.fitted_ = True
        return self

    def predict_log_qty(self, df):
        import statsmodels.api as sm
        X = df[self.cols_].to_numpy(float)
        return sm.add_constant(X, has_constant="add") @ self.beta_

    def elasticity(self, df):
        return np.full(len(df), self.avg_elasticity_)


# =======================================================================================
# 2. two-way fixed effects
# =======================================================================================
class TwoWayFixedEffects(ElasticityModel):
    """Product **and** week fixed effects, by iterative demeaning.

    Absorbs anything constant within a product (brand, quality, shelf position) and
    anything common to a week (season, promotions calendar, macro).  What remains is
    within-product, within-week price variation — much closer to a controlled comparison,
    and the standard workhorse in applied demand estimation.

    ``add_controls`` additionally partials out the time-varying controls linearly.

    Collinear controls must be dropped, not merely tolerated
    -------------------------------------------------------
    A control that is constant within week (``week_idx``, ``woy_sin``, ``is_q4``,
    ``market_log_qty``) is *exactly* collinear with the week effects; one that is constant
    within product (``prod_log_price_mean``, ``prod_n_weeks``) is exactly collinear with
    the product effects.  After demeaning, those columns are zero up to floating-point
    residue — and OLS by pseudo-inverse then hands them enormous coefficients fitted to
    that residue.  The elasticity survives (the pinv keeps the fit), but predictions do
    not: on this panel, leaving them in produced a held-out RMSE of **3.6e7** against a
    constant-mean baseline of 1.75.  :meth:`fit` therefore drops any column whose
    demeaned standard deviation falls below ``collinear_tol`` of its raw one, and records
    the survivors in ``kept_cols_``.

    The default tolerance of 2% is not arbitrary — on this panel the columns separate
    cleanly into three groups: exactly absorbed (ratio ≤ 2.5e-8), *nearly* absorbed
    (``market_log_qty``/``market_log_price``, 0.9–1.5%, which are leave-one-out week
    aggregates and so almost but not quite week-constant), and genuinely within-varying
    (≥ 23%).  The near-absorbed pair is the dangerous one: it survives a
    floating-point-scale test yet still gets a huge, unstable coefficient.

    Forecasting caveat
    ------------------
    A week effect for a *future* week is not identified — there is no data for it.
    :meth:`predict_log_qty` uses the mean of the last four estimated week effects, which
    is a persistence forecast and the best available default.  Treat this model as an
    inference tool that can also predict, not as a forecaster.
    """

    name = "twoway_fe"

    def __init__(self, add_controls: bool = True, n_iter: int = 12,
                 collinear_tol: float = 0.02):
        super().__init__(add_controls=add_controls, n_iter=n_iter,
                         collinear_tol=collinear_tol)

    @staticmethod
    def _absorb(v: np.ndarray, prod: np.ndarray, wk: np.ndarray, n_iter: int) -> np.ndarray:
        v = v.astype(float).copy()
        pdf = pd.DataFrame({"v": v, "p": prod, "w": wk})
        for _ in range(n_iter):
            pdf["v"] -= pdf.groupby("p", observed=True)["v"].transform("mean")
            pdf["v"] -= pdf.groupby("w", observed=True)["v"].transform("mean")
        return pdf["v"].to_numpy()

    def fit(self, df):
        import statsmodels.api as sm
        prod = df["stock_code"].to_numpy()
        wk = df["week"].to_numpy()
        n = self.params["n_iter"]
        y = self._absorb(df[OUTCOME].to_numpy(float), prod, wk, n)
        cols = [TREATMENT] + (list(CONTROL_FEATURES) if self.params["add_controls"] else [])

        absorbed, kept = [], []
        for c in cols:
            raw = df[c].to_numpy(float)
            a = self._absorb(raw, prod, wk, n)
            raw_sd = float(np.std(raw))
            if c != TREATMENT and np.std(a) <= self.params["collinear_tol"] * max(raw_sd, 1e-12):
                continue                       # absorbed by the fixed effects
            absorbed.append(a)
            kept.append(c)
        self.kept_cols_, self.dropped_cols_ = kept, [c for c in cols if c not in kept]
        X = np.column_stack(absorbed)
        self.res_ = sm.OLS(y, X).fit()
        self.beta_ = np.asarray(self.res_.params, float)
        self.avg_elasticity_ = -self.beta_[0]

        # Recover the levels so the model can predict.  The slope came from demeaned data,
        # so keep the controls centred at their training means; then split the remainder
        # r = y − (X − X̄)·beta into a grand mean, a product effect and a week effect.
        # Product and week means are not orthogonal in an unbalanced panel, so one pass
        # leaves a large remainder — alternate until they agree.
        Xr = df[kept].to_numpy(float)
        self.xbar_ = Xr.mean(axis=0)
        r = df[OUTCOME].to_numpy(float) - (Xr - self.xbar_) @ self.beta_
        self.grand_ = float(r.mean())
        resid = pd.Series(r - self.grand_)
        g = pd.Series(0.0, index=pd.Index(pd.unique(wk)))
        for _ in range(max(n, 5)):
            a = (resid - g.reindex(wk).to_numpy()).groupby(prod).mean()
            g = (resid - a.reindex(prod).to_numpy()).groupby(wk).mean()
        self.prod_fe_, self.week_fe_ = a, g
        self.recent_week_fe_ = float(g.sort_index().iloc[-4:].mean()) if len(g) else 0.0
        self.fitted_ = True
        return self

    def predict_log_qty(self, df):
        X = df[self.kept_cols_].to_numpy(float) - self.xbar_
        a = df["stock_code"].map(self.prod_fe_).fillna(0.0).to_numpy(float)
        g = (pd.Series(df["week"].to_numpy()).map(self.week_fe_)
             .fillna(self.recent_week_fe_).to_numpy(float))
        return self.grand_ + a + g + X @ self.beta_

    def elasticity(self, df):
        return np.full(len(df), self.avg_elasticity_)


# =======================================================================================
# 3. Poisson GLM
# =======================================================================================
class PoissonGLM(ElasticityModel):
    """``E[q] = exp(a + b·log p + c'X)``, fitted by Poisson maximum likelihood.

    Quantities are non-negative counts, and taking logs of them (as every other model
    here does) throws away the mean-variance relationship and mishandles small counts.
    A log-link Poisson keeps the same multiplicative interpretation — ``−b`` is still the
    elasticity — while modelling the outcome on its natural scale.  It is also the one
    model here that is consistent under a *multiplicative* error term, which is what a
    demand system with proportional shocks actually has.

    Implementation note: fitted by L-BFGS (scikit-learn's ``PoissonRegressor``) rather than
    IRLS.  Weekly quantities in this panel reach ~80,000 units, and IRLS weights equal the
    fitted mean, so the working Hessian becomes badly conditioned and the solver crawls.
    L-BFGS on the standardised design converges in seconds.  Features are standardised
    internally and the coefficients transformed back, so ``alpha`` penalises comparable
    quantities and the reported elasticity is on the original scale.
    """

    name = "poisson_glm"

    def __init__(self, use_controls: bool = True, alpha: float = 1e-8,
                 max_iter: int = 300):
        super().__init__(use_controls=use_controls, alpha=alpha, max_iter=max_iter)

    def fit(self, df):
        from sklearn.linear_model import PoissonRegressor

        cols = [TREATMENT] + (list(CONTROL_FEATURES) if self.params["use_controls"] else [])
        self.cols_ = cols
        X = df[cols].to_numpy(float)
        self.mu_ = X.mean(axis=0)
        self.sd_ = np.where(X.std(axis=0) > 1e-12, X.std(axis=0), 1.0)
        Z = (X - self.mu_) / self.sd_
        y = np.exp(df[OUTCOME].to_numpy(float))
        m = PoissonRegressor(alpha=self.params["alpha"], fit_intercept=True,
                             max_iter=self.params["max_iter"], tol=1e-7)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(Z, y)
        self.m_ = m
        # back out coefficients on the original scale
        self.coef_ = m.coef_ / self.sd_
        self.intercept_ = float(m.intercept_ - np.dot(m.coef_ / self.sd_, self.mu_))
        self.avg_elasticity_ = -float(self.coef_[0])
        self.n_iter_ = int(getattr(m, "n_iter_", -1))
        self.fitted_ = True
        return self

    def predict_log_qty(self, df):
        X = df[self.cols_].to_numpy(float)
        return self.intercept_ + X @ self.coef_    # log E[q]; vs log q up to Jensen

    def elasticity(self, df):
        return np.full(len(df), self.avg_elasticity_)


# =======================================================================================
# 4. gradient-boosted demand surface (S-learner)
# =======================================================================================
class LightGBMDemand(ElasticityModel):
    """One boosted model of ``log q`` on controls **and** price; elasticity by differencing.

    This is the *S-learner*: a single model covering all price levels.  It is the most
    natural thing a machine-learning practitioner reaches for, it predicts extremely well,
    and — as the tuning notebook shows — it is systematically the worst of the flexible
    methods at recovering elasticity, because nothing in its objective asks it to get the
    price derivative right.  Regularisation that helps prediction (heavy shrinkage, deep
    trees splitting on the many strong control features) actively flattens the price
    dimension.
    """

    name = "lgbm_demand"

    def __init__(self, n_estimators: int = 400, learning_rate: float = 0.05,
                 num_leaves: int = 63, min_child_samples: int = 40,
                 subsample: float = 0.9, colsample_bytree: float = 0.8,
                 reg_lambda: float = 1.0, delta: float = 0.05, seed: int = 0):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate,
                         num_leaves=num_leaves, min_child_samples=min_child_samples,
                         subsample=subsample, colsample_bytree=colsample_bytree,
                         reg_lambda=reg_lambda, delta=delta, seed=seed)

    def fit(self, df):
        self.cols_ = list(CONTROL_FEATURES) + list(PRICE_FEATURES)
        p = {k: v for k, v in self.params.items() if k not in ("delta", "seed")}
        p["subsample_freq"] = 1
        self.m_ = _lgb(p, self.params["seed"])
        self.m_.fit(df[self.cols_].to_numpy(float), df[OUTCOME].to_numpy(float))
        self.avg_elasticity_ = float(np.mean(self.elasticity(df)))
        self.fitted_ = True
        return self

    def predict_log_qty(self, df):
        return self.m_.predict(df[self.cols_].to_numpy(float))

    def elasticity(self, df):
        d = self.params["delta"]
        up = self.m_.predict(perturb_price(df, +d)[self.cols_].to_numpy(float))
        dn = self.m_.predict(perturb_price(df, -d)[self.cols_].to_numpy(float))
        return -(up - dn) / (2 * d)


# =======================================================================================
# 5. Double ML, partialling out (constant elasticity)
# =======================================================================================
class DoubleMLPartialling(ElasticityModel):
    """Robinson's partialling-out with machine-learned nuisances and cross-fitting.

    Fit ``E[log q | X]`` and ``E[log p | X]`` out of fold, take residuals, and regress one
    on the other::

        elasticity = − Cov(y − E[y|X],  p − E[p|X]) / Var(p − E[p|X])

    The estimate is *Neyman-orthogonal*: a small error in either nuisance model has only a
    second-order effect on the answer, which is exactly the property the S-learner lacks.
    Cross-fitting by product removes the own-observation bias that would otherwise creep
    in from a flexible nuisance model.
    """

    name = "dml_partialling"

    def __init__(self, n_estimators: int = 400, learning_rate: float = 0.05,
                 num_leaves: int = 63, min_child_samples: int = 40,
                 subsample: float = 0.9, colsample_bytree: float = 0.8,
                 reg_lambda: float = 1.0, n_folds: int = 5, seed: int = 0):
        super().__init__(n_estimators=n_estimators, learning_rate=learning_rate,
                         num_leaves=num_leaves, min_child_samples=min_child_samples,
                         subsample=subsample, colsample_bytree=colsample_bytree,
                         reg_lambda=reg_lambda, n_folds=n_folds, seed=seed)

    def _nuisance_params(self):
        p = {k: v for k, v in self.params.items() if k not in ("n_folds", "seed")}
        p["subsample_freq"] = 1
        return p

    def _crossfit(self, df):
        X = df[CONTROL_FEATURES].to_numpy(float)
        y = df[OUTCOME].to_numpy(float)
        t = df[TREATMENT].to_numpy(float)
        folds = _product_folds(df, self.params["n_folds"], self.params["seed"])
        y_hat = np.empty_like(y)
        t_hat = np.empty_like(t)
        for k in np.unique(folds):
            tr, te = folds != k, folds == k
            my = _lgb(self._nuisance_params(), self.params["seed"]).fit(X[tr], y[tr])
            mt = _lgb(self._nuisance_params(), self.params["seed"]).fit(X[tr], t[tr])
            y_hat[te] = my.predict(X[te])
            t_hat[te] = mt.predict(X[te])
        return y - y_hat, t - t_hat, y_hat, t_hat

    def fit(self, df):
        ry, rt, y_hat, t_hat = self._crossfit(df)
        denom = float(np.dot(rt, rt))
        self.theta_ = float(np.dot(rt, ry) / denom) if denom > 0 else np.nan
        self.avg_elasticity_ = -self.theta_
        # standard error of the orthogonal score
        psi = rt * (ry - self.theta_ * rt)
        self.se_ = float(np.sqrt((psi ** 2).sum()) / abs(denom)) if denom > 0 else np.nan
        # refit nuisances on everything, for out-of-sample prediction
        X = df[CONTROL_FEATURES].to_numpy(float)
        self.my_ = _lgb(self._nuisance_params(), self.params["seed"]).fit(
            X, df[OUTCOME].to_numpy(float))
        self.mt_ = _lgb(self._nuisance_params(), self.params["seed"]).fit(
            X, df[TREATMENT].to_numpy(float))
        self.resid_ = {"ry": ry, "rt": rt}
        self.fitted_ = True
        return self

    def predict_log_qty(self, df):
        X = df[CONTROL_FEATURES].to_numpy(float)
        return self.my_.predict(X) + self.theta_ * (
            df[TREATMENT].to_numpy(float) - self.mt_.predict(X))

    def elasticity(self, df):
        return np.full(len(df), self.avg_elasticity_)


# =======================================================================================
# 6. R-learner with heterogeneous elasticity
# =======================================================================================
class RLearnerHeterogeneous(DoubleMLPartialling):
    """Same orthogonal residuals, but the final stage learns ``eps(x)``.

    Nie & Wager's R-learner: minimise ``sum_i (ry_i − theta(x_i)·rt_i)^2``, which is a
    weighted regression of ``ry/rt`` on ``x`` with weights ``rt^2``.  Products whose price
    barely moved carry almost no weight, which is exactly right — they contain almost no
    information about their own elasticity.
    """

    name = "r_learner"

    def __init__(self, final_n_estimators: int = 300, final_learning_rate: float = 0.05,
                 final_num_leaves: int = 31, final_min_child_samples: int = 60,
                 min_weight_quantile: float = 0.05, **kw):
        super().__init__(**kw)
        self.params.update(final_n_estimators=final_n_estimators,
                           final_learning_rate=final_learning_rate,
                           final_num_leaves=final_num_leaves,
                           final_min_child_samples=final_min_child_samples,
                           min_weight_quantile=min_weight_quantile)

    def fit(self, df):
        super().fit(df)
        ry, rt = self.resid_["ry"], self.resid_["rt"]
        w = rt ** 2
        floor = np.quantile(w, self.params["min_weight_quantile"])
        keep = w > max(floor, 1e-12)
        pseudo = np.where(keep, ry / np.where(np.abs(rt) > 1e-9, rt, np.nan), 0.0)
        fp = dict(n_estimators=self.params["final_n_estimators"],
                  learning_rate=self.params["final_learning_rate"],
                  num_leaves=self.params["final_num_leaves"],
                  min_child_samples=self.params["final_min_child_samples"])
        self.final_ = _lgb(fp, self.params["seed"])
        Xc = df.loc[keep, CONTROL_FEATURES].to_numpy(float)
        self.final_.fit(Xc, pseudo[keep], sample_weight=w[keep])
        self.avg_elasticity_ = float(np.mean(self.elasticity(df)))
        return self

    def elasticity(self, df):
        return -self.final_.predict(df[CONTROL_FEATURES].to_numpy(float))


# =======================================================================================
# 7. per-product slopes with empirical-Bayes shrinkage
# =======================================================================================
class ShrunkPerProduct(ElasticityModel):
    """A separate within-product regression per SKU, shrunk toward the pooled mean.

    Estimating one elasticity per product is the obvious way to get heterogeneity, and on
    its own it is hopeless: a product with 20 weeks and little price movement produces a
    slope with an enormous standard error, occasionally of the wrong sign.  Empirical
    Bayes fixes that by pulling each slope toward the pooled mean in proportion to its own
    noise::

        shrunk_i  =  w_i * raw_i + (1 − w_i) * pooled ,     w_i = tau^2 / (tau^2 + se_i^2)

    with ``tau^2`` the estimated across-product variance of the true slopes.  Products
    whose price never moved end up at the pooled mean, which is the honest answer.
    """

    name = "shrunk_per_product"

    def __init__(self, min_obs: int = 12, min_price_sd: float = 0.02,
                 tau2: float | None = None, demean_week: bool = True):
        super().__init__(min_obs=min_obs, min_price_sd=min_price_sd, tau2=tau2,
                         demean_week=demean_week)

    def fit(self, df):
        d = df[["stock_code", "week", OUTCOME, TREATMENT]].copy()
        if self.params["demean_week"]:
            for c in (OUTCOME, TREATMENT):
                d[c] = d[c] - d.groupby("week", observed=True)[c].transform("mean")
        raw, se, n = {}, {}, {}
        for code, g in d.groupby("stock_code", observed=True):
            y = g[OUTCOME].to_numpy(float) - g[OUTCOME].mean()
            x = g[TREATMENT].to_numpy(float) - g[TREATMENT].mean()
            sxx = float(np.dot(x, x))
            if len(g) < self.params["min_obs"] or np.std(x) < self.params["min_price_sd"] \
                    or sxx <= 1e-10:
                continue
            b = float(np.dot(x, y) / sxx)
            resid = y - b * x
            dof = max(len(g) - 2, 1)
            raw[code] = b
            se[code] = float(np.sqrt(max((resid ** 2).sum() / dof / sxx, 1e-12)))
            n[code] = len(g)
        self.raw_ = pd.Series(raw, dtype=float)
        self.se_ = pd.Series(se, dtype=float)
        if len(self.raw_) == 0:
            self.pooled_ = 0.0
            self.shrunk_ = pd.Series(dtype=float)
            self.tau2_ = 0.0
        else:
            prec = 1.0 / self.se_ ** 2
            self.pooled_ = float((self.raw_ * prec).sum() / prec.sum())
            tau2 = self.params["tau2"]
            if tau2 is None:   # method-of-moments: total variance minus sampling variance
                tau2 = max(float(self.raw_.var(ddof=1) - (self.se_ ** 2).mean()), 1e-6)
            self.tau2_ = float(tau2)
            w = self.tau2_ / (self.tau2_ + self.se_ ** 2)
            self.shrunk_ = w * self.raw_ + (1 - w) * self.pooled_
            self.weights_ = w
        self.avg_elasticity_ = -float(self.shrunk_.mean()) if len(self.shrunk_) else np.nan
        self.level_ = df.groupby("stock_code", observed=True)[OUTCOME].mean()
        self.global_level_ = float(df[OUTCOME].mean())
        self.fitted_ = True
        return self

    def _slopes(self, df):
        return df["stock_code"].map(self.shrunk_).fillna(self.pooled_).to_numpy(float)

    def predict_log_qty(self, df):
        base = df["stock_code"].map(self.level_).fillna(self.global_level_).to_numpy(float)
        pm = df.groupby("stock_code", observed=True)[TREATMENT].transform("mean")
        return base + self._slopes(df) * (df[TREATMENT].to_numpy(float) - pm.to_numpy(float))

    def elasticity(self, df):
        return -self._slopes(df)


MODEL_REGISTRY: dict[str, type[ElasticityModel]] = {
    m.name: m for m in (PooledLogLog, TwoWayFixedEffects, PoissonGLM, LightGBMDemand,
                        DoubleMLPartialling, RLearnerHeterogeneous, ShrunkPerProduct)
}


def build_model(name: str, **params) -> ElasticityModel:
    """Instantiate by registry name; the tuning layer uses this."""
    if name not in MODEL_REGISTRY:
        raise KeyError(f"unknown model {name!r}; have {sorted(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**params)
