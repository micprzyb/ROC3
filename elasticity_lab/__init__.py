"""
elasticity_lab — price-elasticity modelling on public retail data.

Layers, in dependency order::

    data      UCI Online Retail II -> a clean product-week panel
    features  leakage-safe feature engineering, with controls and treatment separated
    splits    cross-validation designs that respect time and product identity
    simulate  a semi-synthetic benchmark with a KNOWN elasticity, for grading
    models    seven estimators, from pooled OLS to causal machine learning
    tuning    the Optuna layer: search spaces, objectives, nested evaluation
    evaluate  predictive and causal metrics, and model comparison

See ``docs/ELASTICITY_MODELS.md`` and the notebooks in ``notebooks/``.
"""
from .data import get_panel, load_transactions, clean_transactions, build_panel
from .features import (add_features, design_matrix, CONTROL_FEATURES, PRICE_FEATURES,
                       OUTCOME, TREATMENT)
from .splits import (RollingOriginSplit, PurgedRollingSplit, GroupedProductSplit,
                     NaiveKFold, describe_split, split_report)
from .simulate import (SyntheticConfig, make_synthetic_panel, theoretical_ols_bias,
                       naive_bias_decomposition,
                       elasticity_recovery_metrics)
from .models import (MODEL_REGISTRY, build_model, PooledLogLog, TwoWayFixedEffects,
                     PoissonGLM, LightGBMDemand, DoubleMLPartialling,
                     RLearnerHeterogeneous, ShrunkPerProduct)
from .tuning import (CausalScorer, cv_score, tune, objective_comparison, nested_cv,
                     SEARCH_SPACES, tuning_history)
from .evaluation import (predictive_metrics, gates, blp_calibration, policy_value,
                         oracle_metrics, evaluate_model, compare_models,
                         elasticity_summary)

__version__ = "0.1.0"
