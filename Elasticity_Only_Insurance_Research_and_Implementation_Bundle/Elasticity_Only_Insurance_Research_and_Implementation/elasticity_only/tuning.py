"""Nested-CV Optuna helpers.

The optimization target is a proper buyer-side likelihood.  AUC, accuracy, and
policy value are intentionally excluded from hyperparameter search because they
are either insensitive to probability ratios or too noisy for inner-loop tuning.
"""
from __future__ import annotations
from pathlib import Path
from typing import Any, Callable
import inspect
import numpy as np
import optuna
from .metrics import pairwise_nll
from .pairwise import row_subset
from .scalar import scalar_flip_nll_rows

def _storage_url(storage: str | Path) -> str:
    text = str(storage)
    if '://' in text:
        if text.startswith('sqlite:///'):
            path = Path(text.removeprefix('sqlite:///')).expanduser()
            path.parent.mkdir(parents=True, exist_ok=True)
        return text
    path = Path(text).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f'sqlite:///{path}'


def _fit_with_supported_kwargs(model: Any, X: Any, y: Any, kwargs: dict[str, Any]) -> Any:
    """Call ``fit`` with only keyword arguments supported by the estimator.

    Native boosting wrappers accept validation data for early stopping, while
    scikit-learn and Random-Forest wrappers often do not.  Filtering by the
    actual signature keeps one CV objective usable across all model families
    without hiding TypeErrors raised inside an estimator.
    """
    signature = inspect.signature(model.fit)
    accepts_var_kwargs = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    supported = kwargs if accepts_var_kwargs else {
        key: value for key, value in kwargs.items() if key in signature.parameters
    }
    return model.fit(X, y, **supported)

def create_study(name: str, storage: str | Path, *, seed: int=2026, load_if_exists: bool=True, n_startup_trials: int=30) -> optuna.Study:
    """Create a persistent, reproducible minimization study."""
    sampler = optuna.samplers.TPESampler(seed=seed, n_startup_trials=n_startup_trials, multivariate=True, group=True)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=max(10, n_startup_trials // 2), n_warmup_steps=1, n_min_trials=5)
    return optuna.create_study(study_name=name, direction='minimize', storage=_storage_url(storage), sampler=sampler, pruner=pruner, load_if_exists=load_if_exists)

def pairwise_cv_objective(X: Any, arm: np.ndarray, pi: np.ndarray, groups: np.ndarray, contrast_arm: int, *, splits: list[tuple[np.ndarray, np.ndarray]], model_factory: Callable[[optuna.Trial], Any], sample_weight: np.ndarray | None=None) -> Callable[[optuna.Trial], float]:
    """Build a fold-prunable H/C or L/C conditional-NLL objective."""
    arm = np.asarray(arm)
    pi = np.asarray(pi, dtype=float)
    groups = np.asarray(groups)
    if len(arm) != len(pi) or len(arm) != len(groups):
        raise ValueError('arm, pi, and groups must align')

    def objective(trial: optuna.Trial) -> float:
        scores: list[float] = []
        for step, (train_index, valid_index) in enumerate(splits):
            model = model_factory(trial)
            fit_kwargs: dict[str, Any] = {'pi': pi[train_index], 'eval_set': (row_subset(X, valid_index), arm[valid_index]), 'eval_pi': pi[valid_index]}
            if sample_weight is not None:
                fit_kwargs['sample_weight'] = sample_weight[train_index]
                fit_kwargs['eval_sample_weight'] = sample_weight[valid_index]
            _fit_with_supported_kwargs(model, row_subset(X, train_index), arm[train_index], fit_kwargs)
            pred = np.asarray(model.predict_log_rr(row_subset(X, valid_index)), dtype=float)
            score = pairwise_nll(arm[valid_index], pi[valid_index], pred, contrast_arm, None if sample_weight is None else sample_weight[valid_index])
            scores.append(float(score))
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr('fold_scores', scores)
        trial.set_user_attr('fold_std', float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0)
        return float(np.mean(scores))
    return objective

def scalar_cv_objective(X: Any, arm: np.ndarray, pi: np.ndarray, z: np.ndarray, groups: np.ndarray, *, splits: list[tuple[np.ndarray, np.ndarray]], model_factory: Callable[[optuna.Trial], Any], sample_weight: np.ndarray | None=None) -> Callable[[optuna.Trial], float]:
    """Build a fold-prunable three-arm scalar-flip NLL objective."""
    arm = np.asarray(arm)
    pi = np.asarray(pi, dtype=float)
    z = np.asarray(z, dtype=float)
    groups = np.asarray(groups)
    if not len(arm) == len(pi) == len(z) == len(groups):
        raise ValueError('arm, pi, z, and groups must align')

    def objective(trial: optuna.Trial) -> float:
        scores: list[float] = []
        for step, (train_index, valid_index) in enumerate(splits):
            model = model_factory(trial)
            fit_kwargs: dict[str, Any] = {'pi': pi[train_index], 'z': z[train_index], 'eval_set': (row_subset(X, valid_index), arm[valid_index]), 'eval_pi': pi[valid_index], 'eval_z': z[valid_index]}
            if sample_weight is not None:
                fit_kwargs['sample_weight'] = sample_weight[train_index]
                fit_kwargs['eval_sample_weight'] = sample_weight[valid_index]
            _fit_with_supported_kwargs(model, row_subset(X, train_index), arm[train_index], fit_kwargs)
            beta = np.asarray(model.predict(row_subset(X, valid_index)), dtype=float)
            rows = scalar_flip_nll_rows(beta, arm[valid_index], pi[valid_index], z[valid_index])
            score = float(np.average(rows, weights=None if sample_weight is None else sample_weight[valid_index]))
            scores.append(score)
            trial.report(float(np.mean(scores)), step)
            if trial.should_prune():
                raise optuna.TrialPruned()
        trial.set_user_attr('fold_scores', scores)
        trial.set_user_attr('fold_std', float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0)
        return float(np.mean(scores))
    return objective

def optimize_study(study: optuna.Study, objective: Callable[[optuna.Trial], float], *, n_trials: int, n_jobs: int=1) -> optuna.Study:
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=True, show_progress_bar=False)
    return study
