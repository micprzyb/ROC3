from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

EPS = 1e-12
H, C, L = 0, 1, 2


def normalize_rows(values: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(values, dtype=float)
    values = np.clip(values, EPS, np.inf)
    return values / values.sum(axis=1, keepdims=True)


def direct_q_to_buyer_posterior(
    q: NDArray[np.float64],
    propensity: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Convert direct-model counterfactual conversion probabilities
    q[i, t] into buyer-arm probabilities r[i, t].
    """
    q = np.asarray(q, dtype=float)
    propensity = np.asarray(propensity, dtype=float)

    if q.shape != propensity.shape or q.shape[1] != 3:
        raise ValueError("q and propensity must both have shape (n, 3).")

    if np.any((q <= 0.0) | (q >= 1.0)):
        raise ValueError("Direct probabilities must lie strictly between 0 and 1.")

    if np.any(propensity <= 0.0):
        raise ValueError("All randomized-arm propensities must be positive.")

    return normalize_rows(propensity * q)


def risk_ratios_to_buyer_posterior(
    rr_high: NDArray[np.float64],
    rr_low: NDArray[np.float64],
    propensity: NDArray[np.float64],
) -> NDArray[np.float64]:
    """
    Convert H/C and L/C risk ratios to a coherent three-arm buyer posterior.
    Arm order is H, C, L.
    """
    rr_high = np.asarray(rr_high, dtype=float)
    rr_low = np.asarray(rr_low, dtype=float)
    propensity = np.asarray(propensity, dtype=float)

    rr = np.column_stack(
        [
            np.clip(rr_high, EPS, np.inf),
            np.ones_like(rr_high),
            np.clip(rr_low, EPS, np.inf),
        ]
    )
    return normalize_rows(propensity * rr)


def posterior_to_risk_ratios(
    posterior: NDArray[np.float64],
    propensity: NDArray[np.float64],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Recover H/C and L/C risk ratios from buyer-arm probabilities.
    """
    posterior = normalize_rows(posterior)
    propensity = np.asarray(propensity, dtype=float)

    adjusted = posterior / np.clip(propensity, EPS, np.inf)
    rr_high = adjusted[:, H] / np.clip(adjusted[:, C], EPS, np.inf)
    rr_low = adjusted[:, L] / np.clip(adjusted[:, C], EPS, np.inf)
    return rr_high, rr_low


def joint_buyer_nll(
    posterior: NDArray[np.float64],
    arm: NDArray[np.int64],
    purchase: NDArray[np.int64],
) -> float:
    posterior = normalize_rows(posterior)
    arm = np.asarray(arm, dtype=int)
    purchase = np.asarray(purchase, dtype=int)

    mask = purchase == 1
    if not np.any(mask):
        raise ValueError("The test set contains no buyers.")

    selected = posterior[mask, arm[mask]]
    return float(-np.mean(np.log(np.clip(selected, EPS, 1.0))))


def pairwise_buyer_nll(
    rr: NDArray[np.float64],
    target_arm: int,
    propensity: NDArray[np.float64],
    arm: NDArray[np.int64],
    purchase: NDArray[np.int64],
) -> float:
    """
    target_arm must be H or L; control is always C.
    """
    if target_arm not in (H, L):
        raise ValueError("target_arm must be H=0 or L=2.")

    rr = np.asarray(rr, dtype=float)
    propensity = np.asarray(propensity, dtype=float)
    arm = np.asarray(arm, dtype=int)
    purchase = np.asarray(purchase, dtype=int)

    mask = (purchase == 1) & np.isin(arm, [target_arm, C])
    if not np.any(mask):
        raise ValueError("No pairwise buyers for the requested contrast.")

    numerator = propensity[mask, target_arm] * np.clip(rr[mask], EPS, np.inf)
    denominator = numerator + propensity[mask, C]
    probability = np.clip(numerator / denominator, EPS, 1.0 - EPS)

    target = (arm[mask] == target_arm).astype(float)
    loss = -(
        target * np.log(probability)
        + (1.0 - target) * np.log(1.0 - probability)
    )
    return float(np.mean(loss))


def ips_policy_value(
    recommended_arm: NDArray[np.int64],
    observed_arm: NDArray[np.int64],
    purchase: NDArray[np.int64],
    contribution_if_purchase: NDArray[np.float64],
    propensity: NDArray[np.float64],
) -> float:
    """
    contribution_if_purchase has shape (n, 3).
    """
    recommended_arm = np.asarray(recommended_arm, dtype=int)
    observed_arm = np.asarray(observed_arm, dtype=int)
    purchase = np.asarray(purchase, dtype=float)
    contribution_if_purchase = np.asarray(contribution_if_purchase, dtype=float)
    propensity = np.asarray(propensity, dtype=float)

    row = np.arange(len(observed_arm))
    match = observed_arm == recommended_arm

    observed_reward = (
        purchase
        * contribution_if_purchase[row, observed_arm]
    )
    observed_propensity = propensity[row, observed_arm]

    return float(
        np.mean(
            match
            * observed_reward
            / np.clip(observed_propensity, EPS, np.inf)
        )
    )
