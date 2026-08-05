"""Pairwise buyer models for H-vs-C and L-vs-C risk ratios."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit
from scipy.optimize import minimize_scalar
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from .calibration import PairwiseAffineCalibrator
from .core import DEFAULT_PI, as_arm_index, buyer_posterior_from_log_rr, project_log_rr_sign, stable_logit, validate_propensities
FloatArray = NDArray[np.float64]

def row_subset(X: Any, mask: ArrayLike) -> Any:
    m = np.asarray(mask)
    idx = np.flatnonzero(m) if m.dtype == bool else m.astype(int)
    return X.iloc[idx] if hasattr(X, 'iloc') else X[idx]

def _eval_propensities_or_raise(train_pi: FloatArray, eval_pi: ArrayLike | None, n_eval: int) -> FloatArray:
    if eval_pi is not None:
        return validate_propensities(eval_pi, n_eval)
    if np.allclose(train_pi, train_pi[[0]], atol=1e-12, rtol=1e-12):
        return validate_propensities(train_pi[0], n_eval)
    raise ValueError('eval_pi is required when training propensities vary by row')

def _global_pairwise_log_rr(arm: ArrayLike, pi: ArrayLike, contrast_arm: int, sample_weight: ArrayLike | None=None) -> float:
    t = as_arm_index(arm)
    p = validate_propensities(pi, len(t))
    mask = (t == contrast_arm) | (t == 1)
    y = (t[mask] == contrast_arm).astype(float)
    offset = np.log(p[mask, contrast_arm]) - np.log(p[mask, 1])
    w = np.ones(mask.sum()) if sample_weight is None else np.asarray(sample_weight, float)[mask]

    def objective(g: float) -> float:
        eta = offset + g
        return float(np.average(np.logaddexp(0.0, eta) - y * eta, weights=w))
    result = minimize_scalar(objective, bounds=(-20.0, 20.0), method='bounded')
    return float(result.x)

def pairwise_subset(arm: ArrayLike, pi: ArrayLike, contrast_arm: int) -> tuple[NDArray[np.bool_], NDArray[np.int64], FloatArray, FloatArray]:
    if contrast_arm not in (0, 2):
        raise ValueError('contrast_arm must be 0 or 2')
    t = as_arm_index(arm)
    p = validate_propensities(pi, len(t))
    mask = (t == contrast_arm) | (t == 1)
    y = (t[mask] == contrast_arm).astype(np.int64)
    assigned = p[mask, t[mask]]
    offset = np.log(p[mask, contrast_arm]) - np.log(p[mask, 1])
    return (mask, y, assigned, offset)

def _fit_with_sample_weight(estimator: Any, X: Any, y: ArrayLike, w: ArrayLike) -> Any:
    """Route sample weights through a sklearn Pipeline when necessary."""
    try:
        estimator.fit(X, y, sample_weight=w)
    except (TypeError, ValueError) as exc:
        if hasattr(estimator, 'steps'):
            last = estimator.steps[-1][0]
            estimator.fit(X, y, **{f'{last}__sample_weight': w})
        else:
            raise exc
    return estimator

class IPWPairwiseClassifier(BaseEstimator, ClassifierMixin):
    """Generic sklearn/RF model; inverse propensity weighting makes logit(p)=log RR."""

    def __init__(self, estimator: Any, contrast_arm: int) -> None:
        self.estimator = estimator
        self.contrast_arm = contrast_arm

    def fit(self, X: Any, y: ArrayLike, *, pi: ArrayLike=DEFAULT_PI, sample_weight: ArrayLike | None=None) -> 'IPWPairwiseClassifier':
        t = as_arm_index(y)
        p = validate_propensities(pi, len(t))
        mask, binary, assigned, _ = pairwise_subset(t, p, self.contrast_arm)
        base = np.ones(len(t)) if sample_weight is None else np.asarray(sample_weight, float)
        if base.shape != (len(t),):
            raise ValueError('sample_weight must align')
        w = base[mask] / assigned
        w = w / np.mean(w)
        self.model_ = clone(self.estimator)
        _fit_with_sample_weight(self.model_, row_subset(X, mask), binary, w)
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X: Any) -> FloatArray:
        p = np.asarray(self.model_.predict_proba(X), float)
        if p.ndim != 2 or p.shape[1] != 2:
            raise ValueError('base estimator must provide two-class probabilities')
        return p

    def predict_log_rr(self, X: Any) -> FloatArray:
        return stable_logit(self.predict_proba(X)[:, 1])

class XGBOffsetPairwiseClassifier(BaseEstimator, ClassifierMixin):
    """XGBoost binary likelihood with log(pi_t/pi_C) supplied as base_margin."""

    def __init__(self, contrast_arm: int, params: Mapping[str, Any] | None=None, num_boost_round: int=3000, early_stopping_rounds: int=100, random_state: int=2026) -> None:
        self.contrast_arm = contrast_arm
        self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

    def fit(self, X: Any, y: ArrayLike, *, pi: ArrayLike=DEFAULT_PI, sample_weight: ArrayLike | None=None, eval_set: tuple[Any, ArrayLike] | None=None, eval_pi: ArrayLike | None=None, eval_sample_weight: ArrayLike | None=None, verbose_eval: bool | int=False) -> 'XGBOffsetPairwiseClassifier':
        import xgboost as xgb
        t = as_arm_index(y)
        p = validate_propensities(pi, len(t))
        mask, binary, _, offset = pairwise_subset(t, p, self.contrast_arm)
        base = np.ones(len(t)) if sample_weight is None else np.asarray(sample_weight, float)
        if base.shape != (len(t),):
            raise ValueError('sample_weight must align')
        self.global_log_rr_ = _global_pairwise_log_rr(t, p, self.contrast_arm, base)
        dtrain = xgb.DMatrix(row_subset(X, mask), label=binary.astype(float), weight=base[mask], base_margin=offset + self.global_log_rr_, enable_categorical=True)
        evals = [(dtrain, 'train')]
        if eval_set is not None:
            Xv, yv = eval_set
            tv = as_arm_index(yv)
            pv = _eval_propensities_or_raise(p, eval_pi, len(tv))
            mv, ybv, _, ov = pairwise_subset(tv, pv, self.contrast_arm)
            wv = np.ones(len(tv)) if eval_sample_weight is None else np.asarray(eval_sample_weight, float)
            dvalid = xgb.DMatrix(row_subset(Xv, mv), label=ybv.astype(float), weight=wv[mv], base_margin=ov + self.global_log_rr_, enable_categorical=True)
            evals.append((dvalid, 'valid'))
        params = {'objective': 'binary:logistic', 'eval_metric': 'logloss', 'tree_method': 'hist', 'max_depth': 4, 'eta': 0.03, 'min_child_weight': 20.0, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0.0, 'reg_alpha': 0.0, 'reg_lambda': 10.0, 'seed': self.random_state, 'nthread': 1}
        if self.params:
            params.update(dict(self.params))
        self.booster_ = xgb.train(params, dtrain, num_boost_round=self.num_boost_round, evals=evals, early_stopping_rounds=self.early_stopping_rounds if eval_set is not None else None, verbose_eval=verbose_eval)
        self.best_iteration_ = getattr(self.booster_, 'best_iteration', self.num_boost_round - 1)
        self.classes_ = np.array([0, 1])
        return self

    def predict_log_rr(self, X: Any) -> FloatArray:
        import xgboost as xgb
        d = xgb.DMatrix(X, base_margin=np.zeros(len(X)), enable_categorical=True)
        return self.global_log_rr_ + np.asarray(self.booster_.predict(d, output_margin=True, iteration_range=(0, int(self.best_iteration_) + 1)), float)

    def predict_pair_probability(self, X: Any, pi_treatment: ArrayLike, pi_control: ArrayLike) -> FloatArray:
        return expit(np.log(np.asarray(pi_treatment, float)) - np.log(np.asarray(pi_control, float)) + self.predict_log_rr(X))

    def predict_proba(self, X: Any) -> FloatArray:
        p = self.predict_pair_probability(X, np.full(len(X), DEFAULT_PI[self.contrast_arm]), np.full(len(X), DEFAULT_PI[1]))
        return np.column_stack([1 - p, p])

class LGBOffsetPairwiseClassifier(BaseEstimator, ClassifierMixin):
    """LightGBM binary likelihood with log(pi_t/pi_C) supplied as init_score."""

    def __init__(self, contrast_arm: int, params: Mapping[str, Any] | None=None, num_boost_round: int=3000, early_stopping_rounds: int=100, random_state: int=2026) -> None:
        self.contrast_arm = contrast_arm
        self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state

    def fit(self, X: Any, y: ArrayLike, *, pi: ArrayLike=DEFAULT_PI, sample_weight: ArrayLike | None=None, eval_set: tuple[Any, ArrayLike] | None=None, eval_pi: ArrayLike | None=None, eval_sample_weight: ArrayLike | None=None, categorical_feature: Any='auto', verbose_eval: int=0) -> 'LGBOffsetPairwiseClassifier':
        import lightgbm as lgb
        t = as_arm_index(y)
        p = validate_propensities(pi, len(t))
        mask, binary, _, offset = pairwise_subset(t, p, self.contrast_arm)
        base = np.ones(len(t)) if sample_weight is None else np.asarray(sample_weight, float)
        self.global_log_rr_ = _global_pairwise_log_rr(t, p, self.contrast_arm, base)
        train = lgb.Dataset(row_subset(X, mask), label=binary.astype(float), weight=base[mask], init_score=offset + self.global_log_rr_, categorical_feature=categorical_feature, free_raw_data=False)
        valid_sets = [train]
        valid_names = ['train']
        if eval_set is not None:
            Xv, yv = eval_set
            tv = as_arm_index(yv)
            pv = _eval_propensities_or_raise(p, eval_pi, len(tv))
            mv, ybv, _, ov = pairwise_subset(tv, pv, self.contrast_arm)
            wv = np.ones(len(tv)) if eval_sample_weight is None else np.asarray(eval_sample_weight, float)
            valid = lgb.Dataset(row_subset(Xv, mv), label=ybv.astype(float), weight=wv[mv], init_score=ov + self.global_log_rr_, categorical_feature=categorical_feature, reference=train, free_raw_data=False)
            valid_sets.append(valid)
            valid_names.append('valid')
        params = {'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt', 'learning_rate': 0.03, 'num_leaves': 15, 'max_depth': 5, 'min_data_in_leaf': 150, 'min_sum_hessian_in_leaf': 5.0, 'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1, 'lambda_l1': 0.0, 'lambda_l2': 10.0, 'min_gain_to_split': 0.0, 'boost_from_average': False, 'feature_pre_filter': False, 'verbosity': -1, 'seed': self.random_state, 'num_threads': 1}
        if self.params:
            params.update(dict(self.params))
        callbacks = []
        if eval_set is not None:
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds, verbose=False))
        if verbose_eval:
            callbacks.append(lgb.log_evaluation(verbose_eval))
        self.booster_ = lgb.train(params, train, num_boost_round=self.num_boost_round, valid_sets=valid_sets, valid_names=valid_names, callbacks=callbacks)
        self.best_iteration_ = self.booster_.best_iteration or self.num_boost_round
        self.classes_ = np.array([0, 1])
        return self

    def predict_log_rr(self, X: Any) -> FloatArray:
        return self.global_log_rr_ + np.asarray(self.booster_.predict(X, raw_score=True, num_iteration=self.best_iteration_), float)

    def predict_pair_probability(self, X: Any, pi_treatment: ArrayLike, pi_control: ArrayLike) -> FloatArray:
        return expit(np.log(np.asarray(pi_treatment, float)) - np.log(np.asarray(pi_control, float)) + self.predict_log_rr(X))

    def predict_proba(self, X: Any) -> FloatArray:
        p = self.predict_pair_probability(X, np.full(len(X), DEFAULT_PI[self.contrast_arm]), np.full(len(X), DEFAULT_PI[1]))
        return np.column_stack([1 - p, p])

@dataclass
class PairwiseElasticityModel:
    high_model: Any
    low_model: Any
    high_calibrator: PairwiseAffineCalibrator | None = None
    low_calibrator: PairwiseAffineCalibrator | None = None
    enforce_monotonicity: bool = True

    def predict_log_rr(self, X: Any) -> tuple[FloatArray, FloatArray]:
        h = np.asarray(self.high_model.predict_log_rr(X), float)
        l = np.asarray(self.low_model.predict_log_rr(X), float)
        if self.high_calibrator is not None:
            h = self.high_calibrator.predict_log_rr(h)
        if self.low_calibrator is not None:
            l = self.low_calibrator.predict_log_rr(l)
        return project_log_rr_sign(h, l) if self.enforce_monotonicity else (h, l)

    def predict_buyer_posterior(self, X: Any, pi: ArrayLike=DEFAULT_PI) -> FloatArray:
        h, l = self.predict_log_rr(X)
        return buyer_posterior_from_log_rr(h, l, pi)

    def predict_risk_ratios(self, X: Any) -> FloatArray:
        h, l = self.predict_log_rr(X)
        return np.column_stack([np.exp(h), np.ones(len(h)), np.exp(l)])
