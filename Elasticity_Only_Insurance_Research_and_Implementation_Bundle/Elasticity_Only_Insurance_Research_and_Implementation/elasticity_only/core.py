"""Core identities for elasticity-only Bayes-flip models.

This package contains no real-insurance data generator. Synthetic arrays may be
used only in software tests, never as evidence for model selection.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.special import expit, logsumexp
FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]
ARM_ORDER = ('H', 'C', 'L')
ARM_TO_INDEX = {'H': 0, 'C': 1, 'L': 2, '+10': 0, '0': 1, '-10': 2}
DEFAULT_PI = np.array([0.1, 0.8, 0.1], dtype=np.float64)
DEFAULT_MULTIPLIERS = np.array([1.1, 1.0, 0.9], dtype=np.float64)
DEFAULT_Z = np.log(DEFAULT_MULTIPLIERS)
EPS = 1e-12

def as_arm_index(arm: ArrayLike) -> IntArray:
    values = np.asarray(arm)
    if values.ndim != 1:
        raise ValueError('arm must be one-dimensional')
    if np.issubdtype(values.dtype, np.integer):
        out = values.astype(np.int64, copy=False)
    else:
        try:
            out = np.array([ARM_TO_INDEX[str(v).strip().upper()] for v in values], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f'unknown arm label {exc.args[0]!r}') from exc
    if np.any((out < 0) | (out > 2)):
        raise ValueError('arm indices must be 0, 1, or 2')
    return out

def _as_rows(x: ArrayLike, n: int | None, width: int, name: str) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape[0] != width:
            raise ValueError(f'{name} must have {width} columns')
        arr = arr[None, :] if n is None else np.broadcast_to(arr, (n, width)).copy()
    if arr.ndim != 2 or arr.shape[1] != width:
        raise ValueError(f'{name} must have shape (n,{width}) or ({width},)')
    if n is not None and arr.shape[0] != n:
        raise ValueError(f'{name} has {arr.shape[0]} rows, expected {n}')
    if not np.all(np.isfinite(arr)):
        raise ValueError(f'{name} contains non-finite values')
    return arr

def validate_propensities(pi: ArrayLike, n: int | None=None) -> FloatArray:
    arr = _as_rows(pi, n, 3, 'pi')
    if np.any(arr <= 0):
        raise ValueError('all propensities must be strictly positive')
    if not np.allclose(arr.sum(axis=1), 1.0, atol=1e-08, rtol=1e-08):
        raise ValueError('each propensity row must sum to one')
    return arr

def validate_doses(z: ArrayLike, n: int | None=None) -> FloatArray:
    return _as_rows(z, n, 3, 'z')

def softplus(x: ArrayLike) -> FloatArray:
    arr = np.asarray(x, dtype=np.float64)
    return np.log1p(np.exp(-np.abs(arr))) + np.maximum(arr, 0.0)

def inverse_softplus(y: ArrayLike) -> FloatArray:
    arr = np.asarray(y, dtype=np.float64)
    if np.any(arr <= 0):
        raise ValueError('inverse_softplus requires positive values')
    return arr + np.log(-np.expm1(-arr))

def stable_logit(p: ArrayLike, eps: float=1e-09) -> FloatArray:
    arr = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(arr) - np.log1p(-arr)

def buyer_posterior_from_log_rr(log_rr_high: ArrayLike, log_rr_low: ArrayLike, pi: ArrayLike=DEFAULT_PI) -> FloatArray:
    """Construct r_t=P(T=t | buyer,X) from q_H/q_C and q_L/q_C."""
    h = np.asarray(log_rr_high, dtype=np.float64).reshape(-1)
    l = np.asarray(log_rr_low, dtype=np.float64).reshape(-1)
    if h.shape != l.shape:
        raise ValueError('high and low log-risk-ratio arrays must align')
    p = validate_propensities(pi, len(h))
    logits = np.log(p) + np.column_stack([h, np.zeros_like(h), l])
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))

def risk_ratios_from_buyer_posterior(posterior: ArrayLike, pi: ArrayLike=DEFAULT_PI) -> FloatArray:
    """Exact Bayes flip: q_t/q_C=(r_t/pi_t)/(r_C/pi_C)."""
    r = _as_rows(posterior, None, 3, 'posterior')
    if np.any(r <= 0) or not np.allclose(r.sum(axis=1), 1.0, atol=1e-08):
        raise ValueError('posterior rows must be positive and sum to one')
    p = validate_propensities(pi, len(r))
    density = r / p
    return density / density[:, [1]]

def scalar_buyer_posterior(beta: ArrayLike, pi: ArrayLike=DEFAULT_PI, z: ArrayLike=DEFAULT_Z) -> FloatArray:
    b = np.asarray(beta, dtype=np.float64).reshape(-1)
    p = validate_propensities(pi, len(b))
    zz = validate_doses(z, len(b))
    logits = np.log(p) - b[:, None] * zz
    return np.exp(logits - logsumexp(logits, axis=1, keepdims=True))

def log_rr_from_scalar_beta(beta: ArrayLike, z: ArrayLike=DEFAULT_Z) -> tuple[FloatArray, FloatArray]:
    b = np.asarray(beta, dtype=np.float64).reshape(-1)
    zz = validate_doses(z, len(b))
    centered = zz - zz[:, [1]]
    return (-b * centered[:, 0], -b * centered[:, 2])

def log_rr_to_elasticities(log_rr_high: ArrayLike, log_rr_low: ArrayLike, z: ArrayLike=DEFAULT_Z) -> FloatArray:
    h = np.asarray(log_rr_high, dtype=np.float64).reshape(-1)
    l = np.asarray(log_rr_low, dtype=np.float64).reshape(-1)
    if h.shape != l.shape:
        raise ValueError('high and low arrays must align')
    zz = validate_doses(z, len(h))
    centered = zz - zz[:, [1]]
    if np.any(np.abs(centered[:, [0, 2]]) < EPS):
        raise ValueError('price doses must differ from control')
    return np.column_stack([-h / centered[:, 0], -l / centered[:, 2]])

def project_log_rr_sign(log_rr_high: ArrayLike, log_rr_low: ArrayLike, *, mode: Literal['hard', 'smooth']='hard', temperature: float=0.05) -> tuple[FloatArray, FloatArray]:
    h = np.asarray(log_rr_high, dtype=np.float64)
    l = np.asarray(log_rr_low, dtype=np.float64)
    if h.shape != l.shape:
        raise ValueError('high and low arrays must align')
    if mode == 'hard':
        return (np.minimum(h, 0.0), np.maximum(l, 0.0))
    if mode != 'smooth' or temperature <= 0:
        raise ValueError('mode must be hard/smooth and temperature positive')
    return (-temperature * softplus(-h / temperature), temperature * softplus(l / temperature))

def pairwise_buyer_probability(log_rr: ArrayLike, pi_treatment: ArrayLike, pi_control: ArrayLike) -> FloatArray:
    f = np.asarray(log_rr, dtype=np.float64)
    pt = np.asarray(pi_treatment, dtype=np.float64)
    pc = np.asarray(pi_control, dtype=np.float64)
    if f.shape != pt.shape or f.shape != pc.shape or np.any(pt <= 0) or np.any(pc <= 0):
        raise ValueError('aligned positive propensity arrays are required')
    return expit(np.log(pt) - np.log(pc) + f)

def scalar_projection_from_two_gaps(log_rr_high: ArrayLike, log_rr_low: ArrayLike, z: ArrayLike=DEFAULT_Z, weights: ArrayLike | None=None) -> FloatArray:
    h = np.asarray(log_rr_high, dtype=np.float64).reshape(-1)
    l = np.asarray(log_rr_low, dtype=np.float64).reshape(-1)
    if h.shape != l.shape:
        raise ValueError('high and low arrays must align')
    zz = validate_doses(z, len(h))
    d = np.column_stack([zz[:, 0] - zz[:, 1], zz[:, 2] - zz[:, 1]])
    a = np.column_stack([h, l])
    w = np.ones_like(a) if weights is None else _as_rows(weights, len(h), 2, 'weights')
    if np.any(w < 0):
        raise ValueError('weights must be nonnegative')
    numerator = -np.sum(w * d * a, axis=1)
    denominator = np.sum(w * d * d, axis=1)
    return np.maximum(numerator / np.maximum(denominator, EPS), 0.0)

@dataclass(frozen=True)
class RelativeContributionDecision:
    chosen_arm: IntArray
    relative_value: FloatArray
    relative_value_all_arms: FloatArray

def choose_price_from_relative_demand(rr_to_control: ArrayLike, contribution_if_bought: ArrayLike) -> RelativeContributionDecision:
    """Choose arm from relative demand only; q_C cancels from the argmax."""
    rr = _as_rows(rr_to_control, None, 3, 'rr_to_control')
    margin = _as_rows(contribution_if_bought, len(rr), 3, 'contribution')
    if np.any(rr <= 0) or np.any(margin[:, 1] <= 0):
        raise ValueError('positive risk ratios and control contribution required')
    score = rr * margin
    normalized = score / score[:, [1]]
    chosen = np.argmax(score, axis=1).astype(np.int64)
    return RelativeContributionDecision(chosen, normalized[np.arange(len(rr)), chosen], normalized)
