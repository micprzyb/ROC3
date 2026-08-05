"""Optuna search spaces tailored to weak buyer-side elasticity signal."""
from __future__ import annotations
from dataclasses import dataclass
import math
import optuna

@dataclass(frozen=True)
class LeafBounds:
    lower: int
    upper: int

def dynamic_leaf_bounds(n_pair_buyers: int, minority_fraction: float, *, min_minority_per_leaf: int=10, max_minority_per_leaf: int=100) -> LeafBounds:
    if n_pair_buyers < 50 or not 0 < minority_fraction < 1:
        raise ValueError('valid pair buyer count and minority fraction required')
    lower = max(20, int(math.ceil(min_minority_per_leaf / minority_fraction)))
    upper = max(lower, min(int(0.25 * n_pair_buyers), int(math.ceil(max_minority_per_leaf / minority_fraction))))
    return LeafBounds(lower, upper)

def suggest_lightgbm_pairwise(trial: optuna.Trial, *, n_pair_buyers: int, minority_fraction: float) -> dict:
    b = dynamic_leaf_bounds(n_pair_buyers, minority_fraction)
    depth = trial.suggest_int('max_depth', 2, 7)
    max_leaves = max(4, min(64, 2 ** depth - 1))
    leaves = trial.suggest_int('num_leaves', 4, max_leaves)
    gain_mode = trial.suggest_categorical('min_gain_mode', ['zero', 'positive'])
    gain = 0.0 if gain_mode == 'zero' else trial.suggest_float('min_gain_to_split', 1e-05, 1.0, log=True)
    return {'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.1, log=True), 'max_depth': depth, 'num_leaves': leaves, 'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', b.lower, b.upper, log=True), 'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 0.1, 50.0, log=True), 'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0), 'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0), 'bagging_freq': trial.suggest_int('bagging_freq', 1, 10), 'lambda_l1': trial.suggest_float('lambda_l1', 1e-08, 100.0, log=True), 'lambda_l2': trial.suggest_float('lambda_l2', 0.001, 100.0, log=True), 'min_gain_to_split': gain, 'max_bin': trial.suggest_categorical('max_bin', [63, 127, 255, 511]), 'feature_pre_filter': False, 'boost_from_average': False, 'verbosity': -1}

def suggest_xgboost_pairwise(trial: optuna.Trial, *, n_pair_buyers: int, minority_fraction: float) -> dict:
    grow = trial.suggest_categorical('grow_policy', ['depthwise', 'lossguide'])
    params = {'eta': trial.suggest_float('eta', 0.005, 0.1, log=True), 'grow_policy': grow, 'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 100.0, log=True), 'subsample': trial.suggest_float('subsample', 0.6, 1.0), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), 'reg_alpha': trial.suggest_float('reg_alpha', 1e-08, 100.0, log=True), 'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 100.0, log=True), 'max_bin': trial.suggest_categorical('max_bin', [64, 128, 256, 512]), 'max_delta_step': trial.suggest_categorical('max_delta_step', [0, 1, 2, 5]), 'tree_method': 'hist', 'scale_pos_weight': 1.0}
    if grow == 'depthwise':
        params['max_depth'] = trial.suggest_int('max_depth', 2, 7)
    else:
        params.update({'max_depth': 0, 'max_leaves': trial.suggest_int('max_leaves', 8, 64, log=True)})
    gamma_mode = trial.suggest_categorical('gamma_mode', ['zero', 'positive'])
    params['gamma'] = 0.0 if gamma_mode == 'zero' else trial.suggest_float('gamma', 1e-06, 10.0, log=True)
    return params

def suggest_random_forest_pairwise(trial: optuna.Trial, *, n_pair_buyers: int, minority_fraction: float) -> dict:
    b = dynamic_leaf_bounds(n_pair_buyers, minority_fraction)
    feature_mode = trial.suggest_categorical('max_features_mode', ['sqrt', 'log2', 'fraction'])
    max_features = trial.suggest_float('max_features', 0.2, 1.0) if feature_mode == 'fraction' else feature_mode
    ccp_mode = trial.suggest_categorical('ccp_mode', ['zero', 'positive'])
    impurity_mode = trial.suggest_categorical('impurity_mode', ['zero', 'positive'])
    return {'n_estimators': 1000, 'criterion': trial.suggest_categorical('criterion', ['log_loss', 'gini']), 'max_depth': trial.suggest_int('max_depth', 4, 16), 'min_samples_leaf': trial.suggest_int('min_samples_leaf', b.lower, b.upper, log=True), 'max_features': max_features, 'max_samples': trial.suggest_float('max_samples', 0.5, 1.0), 'min_impurity_decrease': 0.0 if impurity_mode == 'zero' else trial.suggest_float('min_impurity_decrease', 1e-08, 0.001, log=True), 'ccp_alpha': 0.0 if ccp_mode == 'zero' else trial.suggest_float('ccp_alpha', 1e-08, 0.01, log=True), 'bootstrap': True, 'class_weight': None, 'n_jobs': -1}

def suggest_logistic(trial: optuna.Trial) -> dict:
    """Modern scikit-learn elastic-net search.

    Since scikit-learn 1.8, ``l1_ratio`` is the unified way to choose L2
    (0), L1 (1), or an elastic-net mixture; the older ``penalty`` argument is
    deprecated.
    """
    return {
        'C': trial.suggest_float('C', 0.0001, 100.0, log=True),
        'l1_ratio': trial.suggest_float('l1_ratio', 0.0, 1.0),
        'solver': 'saga',
        'max_iter': 5000,
    }

def suggest_xgboost_scalar(trial: optuna.Trial, *, n_buyers: int) -> dict:
    return {'tree_method': 'hist', 'eta': trial.suggest_float('eta', 0.005, 0.05, log=True), 'max_depth': trial.suggest_int('max_depth', 1, 4), 'min_child_weight': trial.suggest_float('min_child_weight', 0.0001, 1.0, log=True), 'subsample': trial.suggest_float('subsample', 0.6, 1.0), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), 'gamma': trial.suggest_float('gamma', 1e-07, 1.0, log=True), 'reg_alpha': trial.suggest_float('reg_alpha', 1e-08, 100.0, log=True), 'reg_lambda': trial.suggest_float('reg_lambda', 0.001, 100.0, log=True), 'max_bin': trial.suggest_categorical('max_bin', [64, 128, 256]), 'nthread': 1}

def suggest_lightgbm_scalar(trial: optuna.Trial, *, n_buyers: int) -> dict:
    depth = trial.suggest_int('max_depth', 1, 4)
    return {'learning_rate': trial.suggest_float('learning_rate', 0.005, 0.05, log=True), 'max_depth': depth, 'num_leaves': trial.suggest_int('num_leaves', 2, min(15, 2 ** depth)), 'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', max(30, int(0.01 * n_buyers)), max(50, int(0.2 * n_buyers)), log=True), 'min_sum_hessian_in_leaf': trial.suggest_float('min_sum_hessian_in_leaf', 1e-05, 0.5, log=True), 'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0), 'bagging_fraction': trial.suggest_float('bagging_fraction', 0.6, 1.0), 'bagging_freq': 1, 'lambda_l1': trial.suggest_float('lambda_l1', 1e-08, 100.0, log=True), 'lambda_l2': trial.suggest_float('lambda_l2', 0.001, 100.0, log=True), 'feature_pre_filter': False, 'boost_from_average': False, 'verbosity': -1}

def suggest_random_forest_moment(trial: optuna.Trial) -> dict:
    return {'n_estimators': 1000, 'max_depth': trial.suggest_int('max_depth', 3, 12), 'min_samples_leaf': trial.suggest_float('min_samples_leaf', 0.01, 0.15, log=True), 'max_features': trial.suggest_float('max_features', 0.3, 1.0), 'max_samples': trial.suggest_float('max_samples', 0.6, 1.0), 'ccp_alpha': trial.suggest_float('ccp_alpha', 1e-09, 0.01, log=True), 'beta_max': trial.suggest_categorical('beta_max', [10.0, 20.0, 30.0]), 'n_jobs': -1}
