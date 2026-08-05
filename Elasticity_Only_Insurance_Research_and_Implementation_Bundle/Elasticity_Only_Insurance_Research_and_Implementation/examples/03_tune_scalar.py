"""Tune a one-parameter elasticity surface on real buyers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna

from elasticity_only.core import DEFAULT_Z, as_arm_index
from elasticity_only.optuna_spaces import (
    suggest_lightgbm_scalar,
    suggest_random_forest_moment,
    suggest_xgboost_scalar,
)
from elasticity_only.preprocessing import make_sklearn_preprocessor
from elasticity_only.scalar import (
    LGBScalarFlipRegressor,
    RandomForestMomentElasticity,
    XGBScalarFlipRegressor,
)
from elasticity_only.schema import buyer_rows, eligible_rows, propensity_matrix, validate_experiment_table
from elasticity_only.splits import make_forward_time_holdout, make_buyer_group_folds
from elasticity_only.tuning import create_study, scalar_cv_objective
from elasticity_only.wrappers import PreprocessedScalarModel

from _common import columns_from_config, features_from_config, load_config, load_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--family", required=True, choices=["rf_moment", "lightgbm", "xgboost"])
    parser.add_argument("--trials", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    frame = load_frame(config)
    columns = columns_from_config(config)
    features = features_from_config(config)
    features.validate(frame)
    validation = validate_experiment_table(frame, columns)
    if validation.has_errors:
        raise RuntimeError("Real experiment table failed validation; run 01_audit_real_data.py first.")

    eligible = eligible_rows(frame, columns).reset_index(drop=True)
    temporal = make_forward_time_holdout(
        eligible[columns.timestamp],
        eligible[columns.customer_id],
        holdout_fraction=float(config["splitting"]["final_time_holdout_fraction"]),
    )
    buyers = buyer_rows(eligible.iloc[temporal.development_index], columns).reset_index(drop=True)
    X = buyers.loc[:, features.all]
    arm = as_arm_index(buyers[columns.arm])
    pi = propensity_matrix(buyers, columns)
    groups = buyers[columns.customer_id].to_numpy()
    z = np.broadcast_to(DEFAULT_Z, (len(buyers), 3)).copy()
    splits = make_buyer_group_folds(
        arm,
        groups,
        n_splits=int(config["splitting"]["inner_group_folds"]),
        random_state=int(config["project"]["seed"]),
    )
    preprocessor = make_sklearn_preprocessor(
        features,
        scale_numeric=False,
        one_hot_min_frequency=config["features"].get("one_hot_min_frequency", 20),
        sparse_output=True,
    )
    seed = int(config["project"]["seed"])

    def model_factory(trial: optuna.Trial):
        if args.family == "rf_moment":
            estimator = RandomForestMomentElasticity(
                **suggest_random_forest_moment(trial), random_state=seed
            )
        elif args.family == "lightgbm":
            estimator = LGBScalarFlipRegressor(
                params=suggest_lightgbm_scalar(trial, n_buyers=len(buyers)),
                num_boost_round=5000,
                early_stopping_rounds=150,
                random_state=seed,
            )
        else:
            estimator = XGBScalarFlipRegressor(
                params=suggest_xgboost_scalar(trial, n_buyers=len(buyers)),
                num_boost_round=5000,
                early_stopping_rounds=150,
                random_state=seed,
            )
        return PreprocessedScalarModel(preprocessor, estimator)

    objective = scalar_cv_objective(
        X,
        arm,
        pi,
        z,
        groups,
        splits=splits,
        model_factory=model_factory,
    )
    out = Path(config["project"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    name = f"{config['project']['name']}_{args.family}_scalar"
    study = create_study(
        name=name,
        storage=config["project"]["optuna_storage"],
        seed=seed,
        load_if_exists=True,
    )
    key = f"scalar_{args.family}_trials"
    trials = args.trials or int(config["optuna"]["studies"].get(key, 120))
    study.optimize(objective, n_trials=trials, n_jobs=1)
    with open(out / f"best_scalar_{args.family}.json", "w", encoding="utf-8") as handle:
        json.dump({"value": study.best_value, "params": study.best_params}, handle, indent=2)
    study.trials_dataframe().to_csv(out / f"trials_scalar_{args.family}.csv", index=False)
    print("Best buyer multiclass NLL:", study.best_value)
    print(json.dumps(study.best_params, indent=2))


if __name__ == "__main__":
    main()
