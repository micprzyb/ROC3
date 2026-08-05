"""Calibration and heterogeneity shrinkage for buyer-side log risk ratios."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize
from .core import project_log_rr_sign
FloatArray=NDArray[np.float64]

def binary_log_loss_rows(y:FloatArray,eta:FloatArray)->FloatArray:
    return np.logaddexp(0.0,eta)-y*eta

@dataclass(frozen=True)
class CalibrationFitSummary:
    intercept:float; slope:float; center:float; nll:float; converged:bool

class PairwiseAffineCalibrator:
    """Fit logRR_cal=intercept+slope*(raw-center), slope>=0, using the exact offset likelihood."""
    def __init__(self,direction:Literal["high","low","none"]="none",smooth_constraint:bool=True,constraint_temperature:float=.05,max_slope:float=4.0,intercept_bound:float=12.0)->None:
        self.direction=direction; self.smooth_constraint=smooth_constraint; self.constraint_temperature=constraint_temperature; self.max_slope=max_slope; self.intercept_bound=intercept_bound
    def _constrain(self,g:FloatArray)->FloatArray:
        if self.direction=="none": return g
        mode="smooth" if self.smooth_constraint else "hard"
        if self.direction=="high": return project_log_rr_sign(g,np.zeros_like(g),mode=mode,temperature=self.constraint_temperature)[0]
        if self.direction=="low": return project_log_rr_sign(np.zeros_like(g),g,mode=mode,temperature=self.constraint_temperature)[1]
        raise ValueError("direction must be high, low, or none")
    def fit(self,raw_log_rr:ArrayLike,y:ArrayLike,log_propensity_odds:ArrayLike,sample_weight:ArrayLike|None=None)->"PairwiseAffineCalibrator":
        raw=np.asarray(raw_log_rr,float).reshape(-1); yy=np.asarray(y,float).reshape(-1); offset=np.asarray(log_propensity_odds,float).reshape(-1)
        if not(len(raw)==len(yy)==len(offset)) or np.any(~np.isin(yy,[0.,1.])): raise ValueError("aligned binary calibration inputs required")
        w=np.ones(len(raw)) if sample_weight is None else np.asarray(sample_weight,float).reshape(-1)
        if len(w)!=len(raw) or np.any(w<0) or not np.any(w>0): raise ValueError("invalid sample weights")
        self.center_=float(np.average(raw,weights=w)); centered=raw-self.center_
        def global_obj(a): return float(np.average(binary_log_loss_rows(yy,offset+self._constrain(np.full(len(raw),a[0]))),weights=w))
        gf=minimize(global_obj,np.array([0.]),method="L-BFGS-B",bounds=[(-self.intercept_bound,self.intercept_bound)])
        def objective(theta):
            g=self._constrain(theta[0]+np.exp(theta[1])*centered)
            return float(np.average(binary_log_loss_rows(yy,offset+g),weights=w))
        res=minimize(objective,np.array([float(gf.x[0]),np.log(.5)]),method="L-BFGS-B",bounds=[(-self.intercept_bound,self.intercept_bound),(np.log(1e-5),np.log(self.max_slope))])
        self.intercept_=float(res.x[0]); self.slope_=float(np.exp(res.x[1])); self.nll_=float(res.fun); self.converged_=bool(res.success); return self
    def predict_log_rr(self,raw_log_rr:ArrayLike,*,hard_final_constraint:bool=True)->FloatArray:
        if not hasattr(self,"intercept_"): raise RuntimeError("calibrator is not fitted")
        g=self.intercept_+self.slope_*(np.asarray(raw_log_rr,float)-self.center_)
        if hard_final_constraint and self.direction=="high": return np.minimum(g,0.)
        if hard_final_constraint and self.direction=="low": return np.maximum(g,0.)
        return self._constrain(g)
    def summary(self)->CalibrationFitSummary:
        return CalibrationFitSummary(self.intercept_,self.slope_,self.center_,self.nll_,self.converged_)
