"""Nested out-of-fold fitting and calibration for pairwise elasticity models."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any,Callable
import numpy as np
from numpy.typing import ArrayLike,NDArray
from .calibration import CalibrationFitSummary,PairwiseAffineCalibrator
from .core import as_arm_index,validate_propensities
from .pairwise import pairwise_subset,row_subset
from .splits import make_buyer_group_folds
FloatArray=NDArray[np.float64]

@dataclass(frozen=True)
class CrossFitPairwiseResult:
    raw_log_rr:FloatArray
    calibrated_log_rr:FloatArray
    fold_id:NDArray[np.int64]
    calibration_summaries:tuple[CalibrationFitSummary,...]

def crossfit_pairwise_with_calibration(*,X:Any,arm:ArrayLike,pi:ArrayLike,groups:ArrayLike,outer_folds:list[tuple[np.ndarray,np.ndarray]],model_factory:Callable[[],Any],contrast_arm:int,inner_splits:int=4,base_sample_weight:ArrayLike|None=None,direction:str|None=None)->CrossFitPairwiseResult:
    """Outer OOF predictions; each calibrator is fitted only on inner OOF scores."""
    t=as_arm_index(arm); p=validate_propensities(pi,len(t)); g=np.asarray(groups); w=None if base_sample_weight is None else np.asarray(base_sample_weight,float)
    raw=np.full(len(t),np.nan); cal=np.full(len(t),np.nan); fold_id=np.full(len(t),-1,dtype=np.int64); summaries=[]
    direction=direction or ("high" if contrast_arm==0 else "low")
    for k,(outer_train,outer_valid) in enumerate(outer_folds):
        inner_local=make_buyer_group_folds(t[outer_train],g[outer_train],n_splits=inner_splits,random_state=8800+k); inner_oof=np.full(len(outer_train),np.nan)
        for inner_train_local,inner_valid_local in inner_local:
            tr=outer_train[inner_train_local]; va=outer_train[inner_valid_local]; model=model_factory(); kwargs={"pi":p[tr]}
            if w is not None: kwargs["sample_weight"]=w[tr]
            try:
                fit_kwargs=dict(kwargs,eval_set=(row_subset(X,va),t[va]),eval_pi=p[va])
                if w is not None: fit_kwargs["eval_sample_weight"]=w[va]
                model.fit(row_subset(X,tr),t[tr],**fit_kwargs)
            except TypeError:
                model.fit(row_subset(X,tr),t[tr],**kwargs)
            inner_oof[inner_valid_local]=model.predict_log_rr(row_subset(X,va))
        if np.any(~np.isfinite(inner_oof)): raise RuntimeError("inner OOF predictions are incomplete")
        pair_mask,y_pair,_,offset=pairwise_subset(t[outer_train],p[outer_train],contrast_arm); calibrator=PairwiseAffineCalibrator(direction=direction)
        calibrator.fit(inner_oof[pair_mask],y_pair,offset,None if w is None else w[outer_train][pair_mask]); summaries.append(calibrator.summary())
        final_model=model_factory(); kwargs={"pi":p[outer_train]}
        if w is not None: kwargs["sample_weight"]=w[outer_train]
        final_model.fit(row_subset(X,outer_train),t[outer_train],**kwargs); pred=np.asarray(final_model.predict_log_rr(row_subset(X,outer_valid)),float); raw[outer_valid]=pred; cal[outer_valid]=calibrator.predict_log_rr(pred); fold_id[outer_valid]=k
    if np.any(~np.isfinite(raw)) or np.any(fold_id<0): raise RuntimeError("outer OOF coverage is incomplete")
    return CrossFitPairwiseResult(raw,cal,fold_id,tuple(summaries))
