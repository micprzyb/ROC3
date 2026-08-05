"""Scalar elasticity models and custom objectives."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize_scalar
from scipy.special import expit, logsumexp
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from .core import DEFAULT_PI, DEFAULT_Z, as_arm_index, inverse_softplus, log_rr_from_scalar_beta, scalar_buyer_posterior, softplus, validate_doses, validate_propensities
FloatArray = NDArray[np.float64]

def _eval_rows_or_raise(train_rows: FloatArray, explicit: ArrayLike | None, n_eval: int, *, kind: str) -> FloatArray:
    validator = validate_propensities if kind == 'pi' else validate_doses
    if explicit is not None:
        return validator(explicit, n_eval)
    if np.allclose(train_rows, train_rows[[0]], atol=1e-12, rtol=1e-12):
        return validator(train_rows[0], n_eval)
    raise ValueError(f'eval_{kind} is required when training {kind} varies by row')

def scalar_flip_nll_rows(beta: ArrayLike, arm: ArrayLike, pi: ArrayLike=DEFAULT_PI, z: ArrayLike=DEFAULT_Z) -> FloatArray:
    t = as_arm_index(arm)
    b = np.asarray(beta, float).reshape(-1)
    if len(b) == 1 and len(t) > 1:
        b = np.full(len(t), b.item())
    if len(b) != len(t):
        raise ValueError('beta and arm must align')
    p = validate_propensities(pi, len(t))
    zz = validate_doses(z, len(t))
    logits = np.log(p) - b[:, None] * zz
    return logsumexp(logits, axis=1) - logits[np.arange(len(t)), t]

@dataclass(frozen=True)
class GlobalScalarFit:
    beta: float
    nll: float
    converged: bool

def fit_global_scalar_beta(arm: ArrayLike, pi: ArrayLike=DEFAULT_PI, z: ArrayLike=DEFAULT_Z, *, sample_weight: ArrayLike | None=None, beta_upper: float=30.0) -> GlobalScalarFit:
    t = as_arm_index(arm)
    p = validate_propensities(pi, len(t))
    zz = validate_doses(z, len(t))
    w = np.ones(len(t)) if sample_weight is None else np.asarray(sample_weight, float).reshape(-1)
    if len(w) != len(t) or np.any(w < 0) or (not np.any(w > 0)):
        raise ValueError('invalid sample_weight')

    def obj(beta):
        return float(np.average(scalar_flip_nll_rows(np.full(len(t), beta), t, p, zz), weights=w))
    r = minimize_scalar(obj, bounds=(0.0, beta_upper), method='bounded')
    return GlobalScalarFit(float(r.x), float(r.fun), bool(r.success))

def scalar_raw_grad_hess(raw_score: ArrayLike, arm: ArrayLike, pi: ArrayLike, z: ArrayLike, *, sample_weight: ArrayLike | None=None, hessian: str='fisher', hessian_floor: float=1e-08) -> tuple[FloatArray, FloatArray]:
    f = np.asarray(raw_score, float).reshape(-1)
    t = as_arm_index(arm)
    if len(f) != len(t):
        raise ValueError('raw_score and arm must align')
    p = validate_propensities(pi, len(t))
    zz = validate_doses(z, len(t))
    beta = softplus(f)
    d = expit(f)
    d2 = d * (1 - d)
    r = scalar_buyer_posterior(beta, p, zz)
    mu = np.sum(r * zz, axis=1)
    var = np.maximum(np.sum(r * zz * zz, axis=1) - mu * mu, 0.0)
    gb = zz[np.arange(len(t)), t] - mu
    grad = gb * d
    if hessian == 'fisher':
        hess = var * d * d
    elif hessian == 'exact':
        hess = var * d * d + gb * d2
    else:
        raise ValueError('hessian must be fisher or exact')
    if sample_weight is not None:
        w = np.asarray(sample_weight, float).reshape(-1)
        grad *= w
        hess *= w
    return (grad.astype(float), np.maximum(hess, hessian_floor).astype(float))

class XGBScalarFlipRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, params: Mapping[str, Any] | None=None, num_boost_round: int=3000, early_stopping_rounds: int=100, beta_upper_global_fit: float=30.0, hessian_floor: float=1e-08, random_state: int=2026) -> None:
        self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.beta_upper_global_fit = beta_upper_global_fit
        self.hessian_floor = hessian_floor
        self.random_state = random_state

    def fit(self, X: Any, y: ArrayLike, *, pi: ArrayLike=DEFAULT_PI, z: ArrayLike=DEFAULT_Z, sample_weight: ArrayLike | None=None, eval_set: tuple[Any, ArrayLike] | None=None, eval_pi: ArrayLike | None=None, eval_z: ArrayLike | None=None, eval_sample_weight: ArrayLike | None=None, verbose_eval: bool | int=False) -> 'XGBScalarFlipRegressor':
        import xgboost as xgb
        arm = as_arm_index(y)
        p = validate_propensities(pi, len(arm))
        zz = validate_doses(z, len(arm))
        w = np.ones(len(arm)) if sample_weight is None else np.asarray(sample_weight, float)
        if w.shape != (len(arm),):
            raise ValueError('sample_weight must align')
        gf = fit_global_scalar_beta(arm, p, zz, sample_weight=w, beta_upper=self.beta_upper_global_fit)
        self.global_beta_ = max(gf.beta, 1e-06)
        self.base_raw_ = float(inverse_softplus([self.global_beta_])[0])
        dtrain = xgb.DMatrix(X, label=arm.astype(float), weight=w, base_margin=np.full(len(arm), self.base_raw_), enable_categorical=True)
        data = {id(dtrain): (p, zz)}
        evals = [(dtrain, 'train')]
        if eval_set is not None:
            Xv, yv = eval_set
            av = as_arm_index(yv)
            pv = _eval_rows_or_raise(p, eval_pi, len(av), kind='pi')
            zv = _eval_rows_or_raise(zz, eval_z, len(av), kind='z')
            wv = np.ones(len(av)) if eval_sample_weight is None else np.asarray(eval_sample_weight, float)
            dvalid = xgb.DMatrix(Xv, label=av.astype(float), weight=wv, base_margin=np.full(len(av), self.base_raw_), enable_categorical=True)
            data[id(dvalid)] = (pv, zv)
            evals.append((dvalid, 'valid'))

        def obj(pred, dmat):
            pp, zdata = data[id(dmat)]
            return scalar_raw_grad_hess(pred, dmat.get_label().astype(int), pp, zdata, sample_weight=dmat.get_weight(), hessian='fisher', hessian_floor=self.hessian_floor)

        def metric(pred, dmat):
            pp, zdata = data[id(dmat)]
            rows = scalar_flip_nll_rows(softplus(pred), dmat.get_label().astype(int), pp, zdata)
            ww = dmat.get_weight()
            return ('flip_nll', float(np.average(rows, weights=ww if len(ww) else None)))
        params = {'tree_method': 'hist', 'max_depth': 3, 'eta': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.8, 'reg_lambda': 10.0, 'reg_alpha': 0.0, 'min_child_weight': 0.02, 'disable_default_eval_metric': True, 'seed': self.random_state, 'nthread': 1}
        if self.params:
            params.update(dict(self.params))
        self.booster_ = xgb.train(params, dtrain, num_boost_round=self.num_boost_round, evals=evals, obj=obj, custom_metric=metric, early_stopping_rounds=self.early_stopping_rounds if eval_set is not None else None, verbose_eval=verbose_eval)
        self.best_iteration_ = getattr(self.booster_, 'best_iteration', self.num_boost_round - 1)
        return self

    def predict(self, X: Any) -> FloatArray:
        import xgboost as xgb
        d = xgb.DMatrix(X, base_margin=np.full(len(X), self.base_raw_), enable_categorical=True)
        raw = self.booster_.predict(d, output_margin=True, iteration_range=(0, int(self.best_iteration_) + 1))
        return softplus(raw)

    def predict_log_rr(self, X: Any, z: ArrayLike=DEFAULT_Z) -> tuple[FloatArray, FloatArray]:
        return log_rr_from_scalar_beta(self.predict(X), z)

class LGBScalarFlipRegressor(BaseEstimator, RegressorMixin):

    def __init__(self, params: Mapping[str, Any] | None=None, num_boost_round: int=3000, early_stopping_rounds: int=100, beta_upper_global_fit: float=30.0, hessian_floor: float=1e-08, random_state: int=2026) -> None:
        self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.beta_upper_global_fit = beta_upper_global_fit
        self.hessian_floor = hessian_floor
        self.random_state = random_state

    def fit(self, X: Any, y: ArrayLike, *, pi: ArrayLike=DEFAULT_PI, z: ArrayLike=DEFAULT_Z, sample_weight: ArrayLike | None=None, eval_set: tuple[Any, ArrayLike] | None=None, eval_pi: ArrayLike | None=None, eval_z: ArrayLike | None=None, eval_sample_weight: ArrayLike | None=None, categorical_feature: Any='auto', verbose_eval: int=0) -> 'LGBScalarFlipRegressor':
        import lightgbm as lgb
        arm = as_arm_index(y)
        p = validate_propensities(pi, len(arm))
        zz = validate_doses(z, len(arm))
        w = np.ones(len(arm)) if sample_weight is None else np.asarray(sample_weight, float)
        gf = fit_global_scalar_beta(arm, p, zz, sample_weight=w, beta_upper=self.beta_upper_global_fit)
        self.global_beta_ = max(gf.beta, 1e-06)
        self.base_raw_ = float(inverse_softplus([self.global_beta_])[0])
        train = lgb.Dataset(X, label=arm.astype(float), weight=w, init_score=np.full(len(arm), self.base_raw_), categorical_feature=categorical_feature, free_raw_data=False)
        data = {id(train): (p, zz)}
        valid_sets = [train]
        valid_names = ['train']
        if eval_set is not None:
            Xv, yv = eval_set
            av = as_arm_index(yv)
            pv = _eval_rows_or_raise(p, eval_pi, len(av), kind='pi')
            zv = _eval_rows_or_raise(zz, eval_z, len(av), kind='z')
            wv = np.ones(len(av)) if eval_sample_weight is None else np.asarray(eval_sample_weight, float)
            valid = lgb.Dataset(Xv, label=av.astype(float), weight=wv, init_score=np.full(len(av), self.base_raw_), categorical_feature=categorical_feature, reference=train, free_raw_data=False)
            data[id(valid)] = (pv, zv)
            valid_sets.append(valid)
            valid_names.append('valid')

        def obj(pred, dset):
            pp, zdata = data[id(dset)]
            return scalar_raw_grad_hess(pred, dset.get_label().astype(int), pp, zdata, sample_weight=dset.get_weight(), hessian='fisher', hessian_floor=self.hessian_floor)

        def metric(pred, dset):
            pp, zdata = data[id(dset)]
            rows = scalar_flip_nll_rows(softplus(pred), dset.get_label().astype(int), pp, zdata)
            return ('flip_nll', float(np.average(rows, weights=dset.get_weight())), False)
        params = {'objective': obj, 'metric': 'None', 'boosting_type': 'gbdt', 'learning_rate': 0.03, 'num_leaves': 15, 'max_depth': 5, 'min_data_in_leaf': 100, 'min_sum_hessian_in_leaf': 0.005, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l2': 10.0, 'lambda_l1': 0.0, 'boost_from_average': False, 'verbosity': -1, 'seed': self.random_state, 'num_threads': 1, 'feature_pre_filter': False}
        if self.params:
            params.update(dict(self.params))
        callbacks = []
        if eval_set is not None:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
        if verbose_eval:
            callbacks.append(lgb.log_evaluation(verbose_eval))
        self.booster_ = lgb.train(params, train, num_boost_round=self.num_boost_round, valid_sets=valid_sets, valid_names=valid_names, feval=metric, callbacks=callbacks)
        self.best_iteration_ = self.booster_.best_iteration or self.num_boost_round
        return self

    def predict(self, X: Any) -> FloatArray:
        tree_raw = np.asarray(self.booster_.predict(X, raw_score=True, num_iteration=self.best_iteration_), float)
        return softplus(tree_raw + self.base_raw_)

    def predict_log_rr(self, X: Any, z: ArrayLike=DEFAULT_Z) -> tuple[FloatArray, FloatArray]:
        return log_rr_from_scalar_beta(self.predict(X), z)

class RandomForestMomentElasticity(BaseEstimator, RegressorMixin):
    """RF benchmark: inverse-propensity weighted conditional mean of log-price dose, inverted to beta."""

    def __init__(self, n_estimators: int=1000, max_depth: int | None=8, min_samples_leaf: int | float=0.02, max_features: str | float=0.7, max_samples: float | None=0.8, ccp_alpha: float=0.0, beta_max: float=30.0, n_jobs: int=-1, random_state: int=2026) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_samples = max_samples
        self.ccp_alpha = ccp_alpha
        self.beta_max = beta_max
        self.n_jobs = n_jobs
        self.random_state = random_state

    def fit(self, X: Any, y: ArrayLike, *, pi: ArrayLike=DEFAULT_PI, z: ArrayLike=DEFAULT_Z, sample_weight: ArrayLike | None=None) -> 'RandomForestMomentElasticity':
        arm = as_arm_index(y)
        p = validate_propensities(pi, len(arm))
        zz = validate_doses(z, len(arm))
        centered = zz - zz[:, [1]]
        observed = centered[np.arange(len(arm)), arm]
        base = np.ones(len(arm)) if sample_weight is None else np.asarray(sample_weight, float)
        ipw = base / p[np.arange(len(arm)), arm]
        ipw /= np.mean(ipw)
        self.model_ = RandomForestRegressor(n_estimators=self.n_estimators, criterion='squared_error', max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf, max_features=self.max_features, bootstrap=True, max_samples=self.max_samples, ccp_alpha=self.ccp_alpha, n_jobs=self.n_jobs, random_state=self.random_state)
        self.model_.fit(X, observed, sample_weight=ipw)
        return self

    @staticmethod
    def _mean_dose(beta: FloatArray, z: FloatArray) -> FloatArray:
        centered = z - z[:, [1]]
        logits = -beta[:, None] * centered
        probs = np.exp(logits - logsumexp(logits, axis=1, keepdims=True))
        return np.sum(probs * centered, axis=1)

    def predict(self, X: Any, *, z: ArrayLike=DEFAULT_Z) -> FloatArray:
        mu = np.asarray(self.model_.predict(X), float).reshape(-1)
        zz = validate_doses(z, len(mu))
        low = np.zeros(len(mu))
        high = np.full(len(mu), self.beta_max)
        mu0 = self._mean_dose(low, zz)
        mumax = self._mean_dose(high, zz)
        target = np.clip(mu, np.minimum(mumax, mu0), np.maximum(mumax, mu0))
        for _ in range(60):
            mid = (low + high) / 2
            mm = self._mean_dose(mid, zz)
            move = mm > target
            low = np.where(move, mid, low)
            high = np.where(move, high, mid)
        return (low + high) / 2

    def predict_log_rr(self, X: Any, z: ArrayLike=DEFAULT_Z) -> tuple[FloatArray, FloatArray]:
        return log_rr_from_scalar_beta(self.predict(X, z=z), z)
