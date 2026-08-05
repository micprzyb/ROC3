# Elasticity-only modeling for randomized insurance price tests

This is a **real-data-only** research and reference implementation for learning heterogeneous price response from a randomized three-arm insurance quote experiment.

It is built around the exact buyer-side identity

```text
P(T=t | buyer,X) ∝ pi_t(X) P(buy | T=t,X),
```

which identifies risk ratios to control without requiring a customer-level base-conversion model. For a tested-arm contribution matrix `g_t(X)`, the price decision is

```text
argmax_t g_t(X) * RR_tC(X),
```

because the common control conversion factor cancels.

## What is included

- `LLM_AGENT_INSTRUCTION.md`: copy-paste operating instruction for an LLM agent.
- `ELASTICITY_MODELING_REPORT.md`: research rationale and model recommendations.
- `DATA_CONTRACT.md`: mandatory real-experiment schema and leakage rules.
- `elasticity_only/`: Python implementation.
- `examples/`: command-line entry points for auditing, Optuna tuning, and held-out evaluation.
- `config/example_config.yaml`: full configuration template.
- `tests/`: deterministic identity, derivative, API, and offset tests. These are software tests, not evidence about an insurance portfolio.

The package deliberately contains no generated insurance dataset and no synthetic leaderboard.

## Installation

Create an isolated Python environment and install the package from this directory:

```bash
python -m pip install -e .
```

The implementation was verified with Python 3.13, scikit-learn 1.8.0, LightGBM 4.6.0, XGBoost 3.1.3, and Optuna 4.8.0. The declared requirements allow compatible recent versions; production use should lock exact versions after validation.

## Real-data workflow

1. Copy `config/example_config.yaml` and point `data.path` to the randomized quote table.
2. Complete the explicit feature lists. Do not select all columns automatically.
3. Audit the experiment:

```bash
python examples/01_audit_real_data.py --config config/real_run.yaml
```

4. Tune H/C and L/C separately for every pairwise family:

```bash
python examples/02_tune_pairwise.py --config config/real_run.yaml --family logistic --contrast H
python examples/02_tune_pairwise.py --config config/real_run.yaml --family logistic --contrast L
python examples/02_tune_pairwise.py --config config/real_run.yaml --family rf --contrast H
python examples/02_tune_pairwise.py --config config/real_run.yaml --family rf --contrast L
python examples/02_tune_pairwise.py --config config/real_run.yaml --family lightgbm --contrast H
python examples/02_tune_pairwise.py --config config/real_run.yaml --family lightgbm --contrast L
python examples/02_tune_pairwise.py --config config/real_run.yaml --family xgboost --contrast H
python examples/02_tune_pairwise.py --config config/real_run.yaml --family xgboost --contrast L
```

5. Tune the scalar elasticity challengers:

```bash
python examples/03_tune_scalar.py --config config/real_run.yaml --family rf_moment
python examples/03_tune_scalar.py --config config/real_run.yaml --family lightgbm
python examples/03_tune_scalar.py --config config/real_run.yaml --family xgboost
```

6. Build strictly out-of-fold predictions using the nested cross-fitting utilities, then evaluate them:

```bash
python examples/04_evaluate_oof_predictions.py \
  --config config/real_run.yaml \
  --predictions outputs/oof_predictions.parquet
```

The prediction file must contain `quote_id`, `log_rr_H`, and `log_rr_L` for every eligible quote, with no in-sample predictions.

## Main modules

- `core.py`: Bayes identities, risk ratios, elasticities, constraints, and contribution decisions.
- `pairwise.py`: propensity-offset LightGBM/XGBoost and inverse-propensity sklearn/Random-Forest models.
- `scalar.py`: global scalar fit, custom gradients/Hessians, boosting estimators, and Random-Forest moment inversion.
- `calibration.py`: propensity-aware affine calibration and heterogeneity shrinkage.
- `metrics.py`: proper buyer likelihoods, information gain, calibration, and clustered comparisons.
- `policy.py`: IPS/SNIPS pricing-policy evaluation without a conversion model.
- `optuna_spaces.py`: problem-specific search spaces.
- `tuning.py`: persistent Optuna studies and fold-prunable objectives.
- `pipeline.py`: nested OOF calibration workflow.
- `schema.py`, `audit.py`, `splits.py`: real-data validation and leakage-resistant splitting.

## Non-negotiable safeguards

- Use logged row-level propensities, not nominal 10/80/10 constants when they differ.
- Train elasticity shape on buyers, but keep all eligible quotes for experiment audit and randomized policy evaluation.
- Keep every customer or household in one fold.
- Reserve a final chronological holdout.
- Calibrate only from inner out-of-fold predictions.
- Keep `class_weight=None`; do not balance buyer arms unless the induced prior shift is reversed exactly.
- Optimize conditional log loss, not AUC, accuracy, VUS, or in-sample policy value.
- Do not report individual elasticity RMSE on real data.
- Retain the global scalar model as an anchor and reject personalization when it does not improve held-out likelihood and policy value.

## Verification

Run:

```bash
python -m compileall -q elasticity_only examples tests
pytest -q
```

The included QA report records the exact environment and results used for this release.
