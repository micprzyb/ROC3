"""Fold-local preprocessing wrappers for custom elasticity estimators."""
from __future__ import annotations
from typing import Any
from sklearn.base import BaseEstimator, clone
from .pairwise import LGBOffsetPairwiseClassifier, XGBOffsetPairwiseClassifier
from .scalar import LGBScalarFlipRegressor, XGBScalarFlipRegressor

class PreprocessedPairwiseModel(BaseEstimator):

    def __init__(self, preprocessor: Any, estimator: Any) -> None:
        self.preprocessor = preprocessor
        self.estimator = estimator

    def fit(self, X: Any, y: Any, *, pi: Any, sample_weight: Any=None, eval_set: tuple[Any, Any] | None=None, eval_pi: Any=None, eval_sample_weight: Any=None) -> 'PreprocessedPairwiseModel':
        self.preprocessor_ = clone(self.preprocessor)
        Xt = self.preprocessor_.fit_transform(X)
        Xv = self.preprocessor_.transform(eval_set[0]) if eval_set is not None else None
        self.estimator_ = clone(self.estimator)
        kwargs = {'pi': pi, 'sample_weight': sample_weight}
        if eval_set is not None and isinstance(self.estimator_, (LGBOffsetPairwiseClassifier, XGBOffsetPairwiseClassifier)):
            kwargs.update({'eval_set': (Xv, eval_set[1]), 'eval_pi': eval_pi, 'eval_sample_weight': eval_sample_weight})
        self.estimator_.fit(Xt, y, **kwargs)
        return self

    def predict_log_rr(self, X: Any):
        return self.estimator_.predict_log_rr(self.preprocessor_.transform(X))

class PreprocessedScalarModel(BaseEstimator):

    def __init__(self, preprocessor: Any, estimator: Any) -> None:
        self.preprocessor = preprocessor
        self.estimator = estimator

    def fit(self, X: Any, y: Any, *, pi: Any, z: Any, sample_weight: Any=None, eval_set: tuple[Any, Any] | None=None, eval_pi: Any=None, eval_z: Any=None, eval_sample_weight: Any=None) -> 'PreprocessedScalarModel':
        self.preprocessor_ = clone(self.preprocessor)
        Xt = self.preprocessor_.fit_transform(X)
        Xv = self.preprocessor_.transform(eval_set[0]) if eval_set is not None else None
        self.estimator_ = clone(self.estimator)
        kwargs = {'pi': pi, 'z': z, 'sample_weight': sample_weight}
        if eval_set is not None and isinstance(self.estimator_, (LGBScalarFlipRegressor, XGBScalarFlipRegressor)):
            kwargs.update({'eval_set': (Xv, eval_set[1]), 'eval_pi': eval_pi, 'eval_z': eval_z, 'eval_sample_weight': eval_sample_weight})
        self.estimator_.fit(Xt, y, **kwargs)
        return self

    def predict(self, X: Any):
        return self.estimator_.predict(self.preprocessor_.transform(X))

    def predict_log_rr(self, X: Any, z: Any):
        return self.estimator_.predict_log_rr(self.preprocessor_.transform(X), z=z)
