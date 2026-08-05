"""Constrained stacking of out-of-fold log-risk-ratio predictions."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from numpy.typing import ArrayLike,NDArray
from scipy.optimize import minimize
from scipy.special import logsumexp
from .calibration import binary_log_loss_rows
from .core import project_log_rr_sign
FloatArray=NDArray[np.float64]
@dataclass(frozen=True)
class StackingSummary:
    weights:FloatArray; intercept:float; slope:float; nll:float; converged:bool
class ConvexLogRRStacker:
    def __init__(self,direction:Literal["high","low","none"]="none",max_slope:float=4.,l2_weight_penalty:float=1e-4)->None: self.direction=direction; self.max_slope=max_slope; self.l2_weight_penalty=l2_weight_penalty
    @staticmethod
    def _softmax(x): return np.exp(x-logsumexp(x))
    def _project(self,s):
        if self.direction=="high": return project_log_rr_sign(s,np.zeros_like(s))[0]
        if self.direction=="low": return project_log_rr_sign(np.zeros_like(s),s)[1]
        return s
    def fit(self,score_matrix:ArrayLike,y:ArrayLike,log_propensity_odds:ArrayLike,sample_weight:ArrayLike|None=None)->"ConvexLogRRStacker":
        scores=np.asarray(score_matrix,float); target=np.asarray(y,float); offset=np.asarray(log_propensity_odds,float); w=np.ones(len(target)) if sample_weight is None else np.asarray(sample_weight,float)
        if scores.ndim!=2 or len(scores)!=len(target): raise ValueError("score matrix must align")
        self.centers_=np.average(scores,axis=0,weights=w); centered=scores-self.centers_; k=scores.shape[1]
        def obj(theta):
            cw=self._softmax(theta[:k]); slope=np.exp(theta[k+1]); lr=self._project(theta[k]+slope*(centered@cw)); return float(np.average(binary_log_loss_rows(target,offset+lr),weights=w)+self.l2_weight_penalty*np.sum((cw-1/k)**2))
        res=minimize(obj,np.r_[np.zeros(k),0.,np.log(.5)],method="L-BFGS-B",bounds=[(-10,10)]*k+[(-20,20),(np.log(1e-6),np.log(self.max_slope))]); self.weights_=self._softmax(res.x[:k]); self.intercept_=float(res.x[k]); self.slope_=float(np.exp(res.x[k+1])); self.nll_=float(res.fun); self.converged_=bool(res.success); return self
    def predict_log_rr(self,score_matrix:ArrayLike)->FloatArray:
        s=np.asarray(score_matrix,float); return self._project(self.intercept_+self.slope_*((s-self.centers_)@self.weights_))
    def summary(self)->StackingSummary: return StackingSummary(self.weights_.copy(),self.intercept_,self.slope_,self.nll_,self.converged_)
