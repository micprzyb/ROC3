"""Elasticity-focused validation metrics for real randomized data."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit
from .calibration import binary_log_loss_rows
from .core import as_arm_index, validate_propensities
FloatArray = NDArray[np.float64]

def buyer_multiclass_nll(arm: ArrayLike, posterior: ArrayLike, sample_weight: ArrayLike | None=None) -> float:
    t = as_arm_index(arm)
    r = np.asarray(posterior, float)
    if r.shape != (len(t), 3):
        raise ValueError('posterior must be (n,3)')
    rows = -np.log(np.clip(r[np.arange(len(t)), t], 1e-15, 1.0))
    w = None if sample_weight is None else np.asarray(sample_weight, float)
    return float(np.average(rows, weights=w))

def pairwise_nll_rows(arm: ArrayLike, pi: ArrayLike, log_rr: ArrayLike, contrast_arm: int) -> FloatArray:
    t = as_arm_index(arm)
    p = validate_propensities(pi, len(t))
    f = np.asarray(log_rr, float).reshape(-1)
    if len(f) != len(t):
        raise ValueError('log_rr must align')
    mask = (t == contrast_arm) | (t == 1)
    y = (t[mask] == contrast_arm).astype(float)
    offset = np.log(p[mask, contrast_arm]) - np.log(p[mask, 1])
    return binary_log_loss_rows(y, offset + f[mask])

def pairwise_nll(arm: ArrayLike, pi: ArrayLike, log_rr: ArrayLike, contrast_arm: int, sample_weight: ArrayLike | None=None) -> float:
    t = as_arm_index(arm)
    rows = pairwise_nll_rows(t, pi, log_rr, contrast_arm)
    mask = (t == contrast_arm) | (t == 1)
    w = None if sample_weight is None else np.asarray(sample_weight, float)[mask]
    return float(np.average(rows, weights=w))

def equal_contrast_nll(arm: ArrayLike, pi: ArrayLike, log_rr_high: ArrayLike, log_rr_low: ArrayLike, sample_weight: ArrayLike | None=None) -> float:
    return 0.5 * (pairwise_nll(arm, pi, log_rr_high, 0, sample_weight) + pairwise_nll(arm, pi, log_rr_low, 2, sample_weight))

@dataclass(frozen=True)
class GlobalPairwiseFit:
    log_rr: float
    nll: float
    converged: bool

def fit_global_pairwise_log_rr(arm: ArrayLike, pi: ArrayLike, contrast_arm: int, *, sample_weight: ArrayLike | None=None, direction_constraint: bool=True) -> GlobalPairwiseFit:
    t = as_arm_index(arm)
    p = validate_propensities(pi, len(t))
    mask = (t == contrast_arm) | (t == 1)
    y = (t[mask] == contrast_arm).astype(float)
    off = np.log(p[mask, contrast_arm]) - np.log(p[mask, 1])
    w = np.ones(mask.sum()) if sample_weight is None else np.asarray(sample_weight, float)[mask]
    bounds = (-20.0, 0.0) if contrast_arm == 0 and direction_constraint else (0.0, 20.0) if contrast_arm == 2 and direction_constraint else (-20.0, 20.0)

    def obj(g):
        return float(np.average(binary_log_loss_rows(y, off + g), weights=w))
    res = minimize_scalar(obj, bounds=bounds, method='bounded')
    return GlobalPairwiseFit(float(res.x), float(res.fun), bool(res.success))

@dataclass(frozen=True)
class CalibrationSlope:
    intercept: float
    slope: float
    nll: float
    converged: bool

def pairwise_calibration_intercept_slope(arm: ArrayLike, pi: ArrayLike, predicted_log_rr: ArrayLike, contrast_arm: int, *, sample_weight: ArrayLike | None=None, slope_nonnegative: bool=True) -> CalibrationSlope:
    t = as_arm_index(arm)
    p = validate_propensities(pi, len(t))
    pred = np.asarray(predicted_log_rr, float)
    mask = (t == contrast_arm) | (t == 1)
    y = (t[mask] == contrast_arm).astype(float)
    off = np.log(p[mask, contrast_arm]) - np.log(p[mask, 1])
    score = pred[mask]
    w = np.ones(mask.sum()) if sample_weight is None else np.asarray(sample_weight, float)[mask]
    center = float(np.average(score, weights=w))
    score = score - center

    def obj(theta):
        slope = np.exp(theta[1]) if slope_nonnegative else theta[1]
        return float(np.average(binary_log_loss_rows(y, off + theta[0] + slope * score), weights=w))
    bounds = [(-20.0, 20.0), (np.log(1e-06), np.log(10.0))] if slope_nonnegative else [(-20.0, 20.0), (-10.0, 10.0)]
    res = minimize(obj, np.array([0.0, np.log(1.0) if slope_nonnegative else 1.0]), method='L-BFGS-B', bounds=bounds)
    slope = float(np.exp(res.x[1]) if slope_nonnegative else res.x[1])
    return CalibrationSlope(float(res.x[0] - slope * center), slope, float(res.fun), bool(res.success))

def grouped_pairwise_calibration(arm: ArrayLike, pi: ArrayLike, predicted_log_rr: ArrayLike, contrast_arm: int, *, n_groups: int=5) -> pd.DataFrame:
    t = as_arm_index(arm)
    p = validate_propensities(pi, len(t))
    score = np.asarray(predicted_log_rr, float)
    ranks = pd.Series(score).rank(method='first')
    groups = pd.qcut(ranks, q=n_groups, labels=False, duplicates='drop').to_numpy()
    rows = []
    for g in np.unique(groups):
        m = groups == g
        fit = fit_global_pairwise_log_rr(t[m], p[m], contrast_arm, direction_constraint=False)
        pair = m & ((t == contrast_arm) | (t == 1))
        y = (t[pair] == contrast_arm).astype(float)
        off = np.log(p[pair, contrast_arm]) - np.log(p[pair, 1])
        pred_prob = expit(off + score[pair])
        rows.append({'group': int(g), 'n_all_buyers': int(m.sum()), 'n_pair_buyers': int(pair.sum()), 'score_mean': float(np.mean(score[m])), 'observed_log_rr': fit.log_rr, 'observed_rr': float(np.exp(fit.log_rr)), 'mean_predicted_pair_probability': float(np.mean(pred_prob)), 'observed_pair_fraction': float(np.mean(y))})
    return pd.DataFrame(rows)

def elasticity_information_gain(arm: ArrayLike, pi: ArrayLike, model_high: ArrayLike, model_low: ArrayLike, *, sample_weight: ArrayLike | None=None) -> dict[str, float]:
    gh = fit_global_pairwise_log_rr(arm, pi, 0, sample_weight=sample_weight)
    gl = fit_global_pairwise_log_rr(arm, pi, 2, sample_weight=sample_weight)
    mh = pairwise_nll(arm, pi, model_high, 0, sample_weight)
    ml = pairwise_nll(arm, pi, model_low, 2, sample_weight)
    return {'high_millinats_per_pair_buyer': 1000 * (gh.nll - mh), 'low_millinats_per_pair_buyer': 1000 * (gl.nll - ml), 'equal_contrast_millinats': 500 * (gh.nll - mh + (gl.nll - ml))}

@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    n_clusters: int

def cluster_bootstrap_mean(values: ArrayLike, clusters: ArrayLike, *, n_bootstrap: int=1000, seed: int=2026) -> BootstrapInterval:
    v = np.asarray(values, float).reshape(-1)
    c = np.asarray(clusters)
    if len(v) != len(c):
        raise ValueError('values and clusters must align')
    unique, inv = np.unique(c, return_inverse=True)
    sums = np.bincount(inv, weights=v)
    counts = np.bincount(inv)
    rng = np.random.default_rng(seed)
    draws = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, len(unique), size=len(unique))
        draws[b] = sums[idx].sum() / counts[idx].sum()
    return BootstrapInterval(float(np.mean(v)), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), len(unique))

def paired_cluster_bootstrap_loss_difference(loss_a: ArrayLike, loss_b: ArrayLike, clusters: ArrayLike, *, n_bootstrap: int=1000, seed: int=2026) -> BootstrapInterval:
    return cluster_bootstrap_mean(np.asarray(loss_a, float) - np.asarray(loss_b, float), clusters, n_bootstrap=n_bootstrap, seed=seed)
