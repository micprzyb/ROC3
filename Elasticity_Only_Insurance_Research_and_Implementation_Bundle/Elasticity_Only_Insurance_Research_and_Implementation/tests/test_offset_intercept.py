import numpy as np

from elasticity_only.metrics import fit_global_pairwise_log_rr
from elasticity_only.pairwise import LGBOffsetPairwiseClassifier, XGBOffsetPairwiseClassifier


def test_offset_models_retain_global_intercept_without_splits():
    # This is a deterministic API/likelihood test, not an empirical model study.
    arm = np.array([0] * 8 + [1] * 80 + [2] * 12, dtype=int)
    pi = np.tile([0.1, 0.8, 0.1], (len(arm), 1))
    rng = np.random.default_rng(4)
    X = rng.normal(size=(len(arm), 2)).astype("float32")
    target = fit_global_pairwise_log_rr(arm, pi, 0).log_rr

    xgb = XGBOffsetPairwiseClassifier(
        0,
        params={
            "max_depth": 1,
            "eta": 0.1,
            "min_child_weight": 1e9,
            "reg_lambda": 0.0,
            "nthread": 1,
        },
        num_boost_round=1,
    ).fit(X, arm, pi=pi)

    lgb = LGBOffsetPairwiseClassifier(
        0,
        params={
            "max_depth": 1,
            "num_leaves": 2,
            "learning_rate": 0.1,
            "min_data_in_leaf": len(arm) + 1,
            "min_sum_hessian_in_leaf": 1e9,
            "lambda_l2": 0.0,
            "num_threads": 1,
        },
        num_boost_round=1,
    ).fit(X, arm, pi=pi)

    np.testing.assert_allclose(xgb.predict_log_rr(X[:5]), target, atol=2e-3)
    np.testing.assert_allclose(lgb.predict_log_rr(X[:5]), target, atol=2e-3)
