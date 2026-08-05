import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from elasticity_only.audit import arm_assignment_audit, dose_compliance_audit, numeric_covariate_balance
from elasticity_only.inspection import prediction_stability
from elasticity_only.pairwise import IPWPairwiseClassifier
from elasticity_only.preprocessing import FeatureSpecification, make_sklearn_preprocessor, prepare_native_categoricals
from elasticity_only.wrappers import PreprocessedPairwiseModel


def test_audit_and_preprocessing_contracts():
    n = 60
    frame = pd.DataFrame(
        {
            "x": np.linspace(0, 1, n),
            "cat": np.tile(["a", "b", "c"], 20),
            "base": np.full(n, 100.0),
            "displayed": np.tile([110.0, 100.0, 90.0], 20),
            "mult": np.tile([1.1, 1.0, 0.9], 20),
        }
    )
    arm = np.tile([0, 1, 2], 20)
    pi = np.tile([1 / 3, 1 / 3, 1 / 3], (n, 1))
    assert len(arm_assignment_audit(arm, pi)) == 3
    assert len(numeric_covariate_balance(frame, arm, ["x"])) == 2
    dose = dose_compliance_audit(frame.base, frame.displayed, frame.mult, arm)
    assert np.allclose(dose.absolute_error_mean, 0.0)

    spec = FeatureSpecification(("x",), ("cat",))
    prep = make_sklearn_preprocessor(spec, scale_numeric=False, one_hot_min_frequency=1)
    Xt = prep.fit_transform(frame)
    assert Xt.shape[0] == n
    native, others = prepare_native_categoricals(frame, [frame.iloc[:5].copy()], spec)
    assert str(native["cat"].dtype) == "category"
    assert len(others) == 1


def test_preprocessed_pairwise_and_stability():
    rng = np.random.default_rng(3)
    n = 90
    frame = pd.DataFrame({"x": rng.normal(size=n), "cat": np.tile(["a", "b", "c"], 30)})
    arm = np.tile(np.array([0, 1, 1, 1, 1, 1, 1, 1, 2]), 10)
    pi = np.tile([0.1, 0.8, 0.1], (n, 1))
    spec = FeatureSpecification(("x",), ("cat",))
    prep = make_sklearn_preprocessor(spec, scale_numeric=False, one_hot_min_frequency=1)
    base = IPWPairwiseClassifier(
        RandomForestClassifier(n_estimators=30, min_samples_leaf=3, random_state=1), 0
    )
    wrapped = PreprocessedPairwiseModel(prep, base).fit(frame, arm, pi=pi)
    score = wrapped.predict_log_rr(frame.iloc[:8])
    assert score.shape == (8,)
    stability = prediction_stability(np.column_stack([np.arange(30), np.arange(30) + 0.1]))
    assert stability["median_spearman"] > 0.99
