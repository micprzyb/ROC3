import numpy as np
from sklearn.linear_model import LogisticRegression

from elasticity_only.calibration import PairwiseAffineCalibrator
from elasticity_only.ensemble import ConvexLogRRStacker
from elasticity_only.multiclass import IPWMulticlassClassifier


def test_affine_calibrator_and_stacker_contracts():
    # Deterministic likelihood/API fixture only.
    raw = np.linspace(-1.0, 1.0, 60)
    offset = np.full(60, np.log(0.1 / 0.8))
    y = np.array(([0] * 42) + ([1] * 18), dtype=int)

    cal = PairwiseAffineCalibrator(direction="high").fit(raw, y, offset)
    pred = cal.predict_log_rr(raw)
    assert pred.shape == raw.shape
    assert np.all(pred <= 1e-12)
    assert cal.summary().slope >= 0

    matrix = np.column_stack([raw, 0.5 * raw, np.zeros_like(raw)])
    stack = ConvexLogRRStacker(direction="high").fit(matrix, y, offset)
    stacked = stack.predict_log_rr(matrix)
    assert stacked.shape == raw.shape
    assert np.all(stacked <= 1e-12)
    np.testing.assert_allclose(stack.summary().weights.sum(), 1.0)


def test_ipw_multiclass_api():
    rng = np.random.default_rng(12)
    n = 120
    X = rng.normal(size=(n, 4))
    arm = np.tile(np.array([0, 1, 1, 1, 1, 1, 1, 1, 2, 1]), 12)
    pi = np.tile([0.1, 0.8, 0.1], (n, 1))
    model = IPWMulticlassClassifier(
        LogisticRegression(max_iter=1000),
        enforce_monotonicity=True,
    ).fit(X, arm, pi=pi)
    h, l = model.predict_log_rr(X[:10])
    assert h.shape == l.shape == (10,)
    assert np.all(h <= 1e-12)
    assert np.all(l >= -1e-12)
    posterior = model.predict_buyer_posterior(X[:10], pi[:10])
    np.testing.assert_allclose(posterior.sum(axis=1), 1.0)
