"""Audit the real randomized experiment before fitting any model."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from elasticity_only.audit import arm_assignment_audit, dose_compliance_audit, numeric_covariate_balance
from elasticity_only.schema import eligible_rows, propensity_matrix, validate_experiment_table

from _common import columns_from_config, features_from_config, load_config, load_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    frame = load_frame(config)
    columns = columns_from_config(config)
    features = features_from_config(config)
    features.validate(frame)

    report = validate_experiment_table(frame, columns, require_policy_columns=True)
    output = Path(config["project"]["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    report.to_frame().to_csv(output / "data_validation_issues.csv", index=False)
    if report.has_errors:
        raise RuntimeError(
            "Experiment table failed validation. Inspect data_validation_issues.csv; "
            "do not fit elasticity models until all errors are resolved."
        )

    eligible = eligible_rows(frame, columns)
    pi = propensity_matrix(eligible, columns)
    assignment = arm_assignment_audit(eligible[columns.arm], pi)
    balance = numeric_covariate_balance(eligible, eligible[columns.arm], list(features.numeric))
    assignment.to_csv(output / "assignment_audit.csv", index=False)
    balance.to_csv(output / "numeric_covariate_balance.csv", index=False)

    dose_columns = [columns.base_premium, columns.displayed_premium, columns.assigned_multiplier]
    if all(c in eligible for c in dose_columns):
        dose = dose_compliance_audit(
            eligible[columns.base_premium],
            eligible[columns.displayed_premium],
            eligible[columns.assigned_multiplier],
            eligible[columns.arm],
        )
        dose.to_csv(output / "dose_compliance.csv", index=False)

    summary = pd.DataFrame(
        {
            "metric": ["rows", "eligible", "buyers", "customers"],
            "value": [report.n_rows, report.n_eligible, report.n_buyers, report.n_customers],
        }
    )
    summary.to_csv(output / "experiment_summary.csv", index=False)
    print(summary.to_string(index=False))
    print("\nArm quotes:", report.arm_quotes)
    print("Arm buyers:", report.arm_buyers)


if __name__ == "__main__":
    main()
