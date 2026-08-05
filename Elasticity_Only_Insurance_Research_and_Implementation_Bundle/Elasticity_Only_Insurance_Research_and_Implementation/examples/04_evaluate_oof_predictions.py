"""Evaluate real out-of-fold elasticity predictions and pricing policies.

The prediction file must contain one row per eligible quote and be produced
strictly out of fold. It must have quote_id, log_rr_H, and log_rr_L columns.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from elasticity_only.core import as_arm_index, buyer_posterior_from_log_rr, choose_price_from_relative_demand
from elasticity_only.metrics import (
    buyer_multiclass_nll,
    elasticity_information_gain,
    equal_contrast_nll,
    grouped_pairwise_calibration,
    pairwise_calibration_intercept_slope,
)
from elasticity_only.policy import (
    cluster_bootstrap_policy_difference,
    evaluate_policy_ips,
    selective_policy,
)
from elasticity_only.schema import (
    contribution_matrix,
    eligible_rows,
    propensity_matrix,
    validate_experiment_table,
)

from _common import columns_from_config, load_config, load_frame


def read_predictions(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--predictions", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    frame = load_frame(config)
    columns = columns_from_config(config)
    validation = validate_experiment_table(frame, columns, require_policy_columns=True)
    if validation.has_errors:
        raise RuntimeError("Experiment table failed validation.")

    eligible = eligible_rows(frame, columns).copy()
    pred = read_predictions(Path(args.predictions))
    required = {columns.quote_id, "log_rr_H", "log_rr_L"}
    if not required.issubset(pred.columns):
        raise ValueError(f"Prediction file must contain {sorted(required)}")
    if pred[columns.quote_id].duplicated().any():
        raise ValueError("Prediction file contains duplicate quote_id values")
    merged = eligible.merge(pred[list(required)], on=columns.quote_id, how="left", validate="one_to_one")
    if merged[["log_rr_H", "log_rr_L"]].isna().any().any():
        raise ValueError("Every eligible quote must have a strictly out-of-fold prediction")

    arm = as_arm_index(merged[columns.arm])
    purchase = merged[columns.purchase].to_numpy(int)
    pi = propensity_matrix(merged, columns)
    log_h = merged["log_rr_H"].to_numpy(float)
    log_l = merged["log_rr_L"].to_numpy(float)
    buyers = purchase == 1
    posterior = buyer_posterior_from_log_rr(log_h[buyers], log_l[buyers], pi[buyers])

    metrics = {
        "buyer_multiclass_nll": buyer_multiclass_nll(arm[buyers], posterior),
        "equal_contrast_nll": equal_contrast_nll(arm[buyers], pi[buyers], log_h[buyers], log_l[buyers]),
        **elasticity_information_gain(arm[buyers], pi[buyers], log_h[buyers], log_l[buyers]),
    }
    out = Path(config["project"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([metrics]).to_csv(out / "elasticity_oof_metrics.csv", index=False)

    for label, contrast, score in (("H", 0, log_h), ("L", 2, log_l)):
        slope = pairwise_calibration_intercept_slope(
            arm[buyers], pi[buyers], score[buyers], contrast
        )
        pd.DataFrame([slope.__dict__]).to_csv(out / f"calibration_slope_{label}.csv", index=False)
        grouped_pairwise_calibration(
            arm[buyers], pi[buyers], score[buyers], contrast, n_groups=5
        ).to_csv(out / f"grouped_calibration_{label}.csv", index=False)

    rr = np.column_stack([np.exp(log_h), np.ones(len(merged)), np.exp(log_l)])
    contribution = contribution_matrix(merged, columns)
    decision = choose_price_from_relative_demand(rr, contribution)
    policy = decision.chosen_arm
    control = np.ones(len(policy), dtype=int)
    clusters = merged[columns.customer_id].to_numpy()

    policy_rows = []
    for name, chosen in (("model", policy), ("control", control)):
        policy_rows.append(
            {
                "policy": name,
                "ips": evaluate_policy_ips(chosen, arm, purchase, pi, contribution).ips,
                "snips": evaluate_policy_ips(chosen, arm, purchase, pi, contribution).snips,
                "fraction_H": float(np.mean(chosen == 0)),
                "fraction_C": float(np.mean(chosen == 1)),
                "fraction_L": float(np.mean(chosen == 2)),
            }
        )
    pd.DataFrame(policy_rows).to_csv(out / "policy_values.csv", index=False)

    boot = cluster_bootstrap_policy_difference(
        policy,
        control,
        arm,
        purchase,
        pi,
        contribution,
        clusters,
        n_bootstrap=int(config["policy"]["cluster_bootstrap_repetitions"]),
        seed=int(config["project"]["seed"]),
    )
    pd.DataFrame([boot.__dict__]).to_csv(out / "policy_lift_cluster_bootstrap.csv", index=False)

    rows = []
    fractions = config["policy"]["selective_adjustment_fractions"]
    for fraction in fractions:
        selective = selective_policy(
            policy,
            decision.relative_value - 1.0,
            fraction=float(fraction),
            default_arm=1,
        )
        rows.append(
            {
                "adjustment_fraction": fraction,
                "ips": evaluate_policy_ips(selective, arm, purchase, pi, contribution).ips,
                "snips": evaluate_policy_ips(selective, arm, purchase, pi, contribution).snips,
            }
        )
    pd.DataFrame(rows).to_csv(out / "selective_policy_curve.csv", index=False)
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(pd.DataFrame(policy_rows).to_string(index=False))


if __name__ == "__main__":
    main()
