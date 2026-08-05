import numpy as np
from sklearn.linear_model import LogisticRegression

from elasticity_only.pairwise import IPWPairwiseClassifier
from elasticity_only.pipeline import crossfit_pairwise_with_calibration
from elasticity_only.splits import make_buyer_group_folds
from elasticity_only.tuning import create_study, optimize_study, pairwise_cv_objective


def _fixture():
    rng = np.random.default_rng(123)
    n_groups = 90
    X = rng.normal(size=(n_groups, 3))
    arm = np.tile(np.array([0, 1, 2]), n_groups // 3)
    pi = np.tile([0.1, 0.8, 0.1], (n_groups, 1))
    groups = np.arange(n_groups)
    return X, arm, pi, groups


def _factory():
    return IPWPairwiseClassifier(LogisticRegression(C=0.1, max_iter=500), 0)


def test_crossfit_pipeline_contract():
    X, arm, pi, groups = _fixture()
    folds = make_buyer_group_folds(arm, groups, n_splits=3, random_state=4)
    result = crossfit_pairwise_with_calibration(
        X=X,
        arm=arm,
        pi=pi,
        groups=groups,
        outer_folds=folds,
        model_factory=_factory,
        contrast_arm=0,
        inner_splits=2,
    )
    assert np.isfinite(result.raw_log_rr).all()
    assert np.isfinite(result.calibrated_log_rr).all()
    assert len(result.calibration_summaries) == 3


def test_optuna_cv_objective_smoke(tmp_path):
    X, arm, pi, groups = _fixture()
    splits = make_buyer_group_folds(arm, groups, n_splits=2, random_state=9)

    def factory(trial):
        c = trial.suggest_float("C", 0.05, 0.2)
        return IPWPairwiseClassifier(LogisticRegression(C=c, max_iter=500), 0)

    study = create_study("smoke", tmp_path / "study.sqlite", n_startup_trials=1)
    objective = pairwise_cv_objective(
        X,
        arm,
        pi,
        groups,
        0,
        splits=splits,
        model_factory=factory,
    )
    optimize_study(study, objective, n_trials=1)
    assert len(study.trials) == 1
    assert np.isfinite(study.best_value)
