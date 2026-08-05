"""Multiclass buyer benchmark models."""
from __future__ import annotations
from typing import Any
import numpy as np
from numpy.typing import ArrayLike,NDArray
from sklearn.base import BaseEstimator,ClassifierMixin,clone
from .core import DEFAULT_PI,as_arm_index,buyer_posterior_from_log_rr,project_log_rr_sign,stable_logit,validate_propensities
from .pairwise import _fit_with_sample_weight
FloatArray=NDArray[np.float64]

class IPWMulticlassClassifier(BaseEstimator,ClassifierMixin):
    """IPW multiclass classifier targeting q_t/sum_j q_j among buyers."""
    def __init__(self,estimator:Any,enforce_monotonicity:bool=True)->None: self.estimator=estimator; self.enforce_monotonicity=enforce_monotonicity
    def fit(self,X:Any,y:ArrayLike,*,pi:ArrayLike=DEFAULT_PI,sample_weight:ArrayLike|None=None)->"IPWMulticlassClassifier":
        t=as_arm_index(y); p=validate_propensities(pi,len(t)); base=np.ones(len(t)) if sample_weight is None else np.asarray(sample_weight,float)
        w=base/p[np.arange(len(t)),t]; w/=np.mean(w); self.model_=clone(self.estimator); _fit_with_sample_weight(self.model_,X,t,w); self.classes_=np.array([0,1,2]); return self
    def predict_normalized_demand(self,X:Any)->FloatArray:
        proba=np.asarray(self.model_.predict_proba(X),float)
        ordered=np.zeros((len(proba),3),float)
        for j,c in enumerate(self.model_.classes_.astype(int)): ordered[:,c]=proba[:,j]
        ordered=np.clip(ordered,1e-9,None); ordered/=ordered.sum(1,keepdims=True); return ordered
    def predict_log_rr(self,X:Any)->tuple[FloatArray,FloatArray]:
        s=self.predict_normalized_demand(X); h=np.log(s[:,0])-np.log(s[:,1]); l=np.log(s[:,2])-np.log(s[:,1])
        return project_log_rr_sign(h,l) if self.enforce_monotonicity else (h,l)
    def predict_buyer_posterior(self,X:Any,pi:ArrayLike=DEFAULT_PI)->FloatArray:
        h,l=self.predict_log_rr(X); return buyer_posterior_from_log_rr(h,l,pi)
