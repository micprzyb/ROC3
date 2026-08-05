# QA and reproducibility report

## Scope

This release is a reference implementation and an agent playbook for a real randomized motor-insurance price experiment. It contains no generated insurance dataset and no synthetic model-performance leaderboard. Deterministic arrays and a temporary API fixture were used only to test identities, derivatives, library interfaces, and command-line plumbing.

## Environment

See `ENVIRONMENT.json` for the exact locally tested versions. The release was tested with Python 3.13, scikit-learn 1.8.0, LightGBM 4.6.0, XGBoost 3.1.3, and Optuna 4.8.0.

## Automated checks

- Python bytecode compilation completed for `elasticity_only/`, `examples/`, and `tests/`.
- `pytest -q` completed with **22 passed** and no failed tests.
- The only warnings were Optuna warnings that `TPESampler(multivariate=True, group=True)` remains an experimental interface.
- Statement coverage was **77%** over the package and test suite. See `COVERAGE.txt`.
- Native LightGBM and XGBoost pairwise-offset and scalar-custom-objective wrappers were included in smoke tests.
- Tests verify the Bayes identity, risk-ratio reconstruction, business decision cancellation, monotonic projections, offset recovery, custom gradients, policy estimators, split integrity, preprocessing, tuning helpers, calibration, multiclass reconstruction, stacking, and schema validation.

## End-to-end interface checks

The four example programs were executed against a temporary, deterministic API fixture created solely for software validation:

1. experiment audit;
2. one-trial pairwise Optuna tuning;
3. one-trial scalar Optuna tuning;
4. evaluation of out-of-fold prediction-file structure and metrics.

The fixture and its numerical model outputs are intentionally not included. They are not evidence about motor-insurance demand and must not be used to rank model families.

## Evidence boundary

What is verified:

- mathematical identities implemented by the code;
- gradient and Hessian consistency at tested points;
- compatibility with the locally installed library APIs;
- output-shape, validation, persistence, and split invariants.

What is not verified without the user's data:

- existence or strength of heterogeneous elasticity;
- which model family wins;
- business lift of any personalized pricing policy;
- temporal stability across tariff versions;
- adequacy of the supplied feature set;
- causal validity of an experiment whose operational logs have not been audited.

## Production-hardening limitations

This is a high-quality research reference, not a fully managed production service. Before deployment, add organization-specific security review, data-access controls, model registry integration, monitoring, protected-feature governance, deterministic dependency locking, distributed-training tests, and larger failure-path coverage. Run all model training inside the organization's approved environment and retain the exact input hash, feature manifest, folds, Optuna storage, model binaries, and out-of-fold predictions.

## Document QA

The final DOCX was rendered to **36 page PNGs** and a PDF with the canonical document renderer. Every page was visually inspected for clipping, overlap, broken tables, missing glyphs, and header/footer defects. The accessibility audit reported zero high-, medium-, or low-severity findings. After metadata scrubbing, all 36 rendered page PNG files were byte-identical to the visually inspected render.
