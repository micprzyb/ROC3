"""Tune one buyer-side price contrast on the development period.

Run this separately for H/C and L/C, and separately for each model family.
The objective is pairwise conditional log loss with the logged propensity odds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from elasticity_only.core import as_arm_index
from elasticity_only.optuna_spaces import (
    suggest_lightgbm_pairwise,
    suggest_logistic,
    suggest_random_forest_pairwise,
    suggest_xgboost_pairwise,
)
from elasticity_only.pairwise import (
    IPWPairwiseClassifier,
    LGBOffsetPairwiseClassifier,
    XGBOffsetPairwiseClassifier,
)
from elasticity_only.preprocessing import make_sklearn_preprocessor
from elasticity_only.schema import buyer_rows, eligible_rows, propensity_matrix, validate_experiment_table
from elasticity_only.splits import make_forward_time_holdout, make_buyer_group_folds
from elasticity_only.tuning import create_study, pairwise_cv_objective
from elasticity_only.wrappers import PreprocessedPairwiseModel

from _common import columns_from_config, features_from_config, load_config, load_frame


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--family", required=True, choices=["logistic", "rf", "lightgbm", "xgboost"])
    parser.add_argument("--contrast", required=True, choices=["H", "L"])
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
    development = eligible.iloc[temporal.development_index]
    buyers = buyer_rows(development, columns).reset_index(drop=True)

    X = buyers.loc[:, features.all]
    arm = as_arm_index(buyers[columns.arm])
    pi = propensity_matrix(buyers, columns)
    groups = buyers[columns.customer_id].to_numpy()
    contrast_arm = 0 if args.contrast == "H" else 2
    pair = (arm == contrast_arm) | (arm == 1)
    n_pair = int(pair.sum())
    minority = float(np.mean(arm[pair] == contrast_arm))

    splits = make_buyer_group_folds(
        arm,
        groups,
        n_splits=int(config["splitting"]["inner_group_folds"]),
        random_state=int(config["project"]["seed"]),
    )
    preprocessor = make_sklearn_preprocessor(
        features,
        scale_numeric=args.family == "logistic",
        one_hot_min_frequency=config["features"].get("one_hot_min_frequency", 20),
        sparse_output=True,
    )

    seed = int(config["project"]["seed"])

    def model_factory(trial: optuna.Trial):
        if args.family == "logistic":
            params = suggest_logistic(trial)
            base = LogisticRegression(**params, random_state=seed, class_weight=None)
            estimator = IPWPairwiseClassifier(base, contrast_arm)
        elif args.family == "rf":
            params = suggest_random_forest_pairwise(
                trial, n_pair_buyers=n_pair, minority_fraction=minority
            )
            base = RandomForestClassifier(**params, random_state=seed)
            estimator = IPWPairwiseClassifier(base, contrast_arm)
        elif args.family == "lightgbm":
            params = suggest_lightgbm_pairwise(
                trial, n_pair_buyers=n_pair, minority_fraction=minority
            )
            estimator = LGBOffsetPairwiseClassifier(
                contrast_arm, params=params, num_boost_round=5000, early_stopping_rounds=150, random_state=seed
            )
        else:
            params = suggest_xgboost_pairwise(
                trial, n_pair_buyers=n_pair, minority_fraction=minority
            )
            estimator = XGBOffsetPairwiseClassifier(
                contrast_arm, params=params, num_boost_round=5000, early_stopping_rounds=150, random_state=seed
            )
        return PreprocessedPairwiseModel(preprocessor, estimator)

    objective = pairwise_cv_objective(
        X,
        arm,
        pi,
        groups,
        contrast_arm,
        splits=splits,
        model_factory=model_factory,
    )

    out = Path(config["project"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    name = f"{config['project']['name']}_{args.family}_{args.contrast}_pairwise"
    study = create_study(
        name=name,
        storage=config["project"]["optuna_storage"],
        seed=seed,
        load_if_exists=True,
    )
    default_trials = config["optuna"]["studies"].get(
        f"{args.family}_trials_per_contrast", 100
    )
    study.optimize(objective, n_trials=args.trials or int(default_trials), n_jobs=1)

    with open(out / f"best_{args.family}_{args.contrast}.json", "w", encoding="utf-8") as handle:
        json.dump(
            {"value": study.best_value, "params": study.best_params, "n_pair_buyers": n_pair},
            handle,
            indent=2,
        )
    study.trials_dataframe().to_csv(out / f"trials_{args.family}_{args.contrast}.csv", index=False)
    print("Best conditional NLL:", study.best_value)
    print(json.dumps(study.best_params, indent=2))


if __name__ == "__main__":
    main()
