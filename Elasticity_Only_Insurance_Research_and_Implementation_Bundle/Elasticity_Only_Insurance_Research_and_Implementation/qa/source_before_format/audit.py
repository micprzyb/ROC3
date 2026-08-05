"""Experiment-integrity diagnostics. These run before any elasticity model."""
from __future__ import annotations
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from .core import as_arm_index,validate_propensities

def arm_assignment_audit(arm:ArrayLike,pi:ArrayLike)->pd.DataFrame:
    t=as_arm_index(arm); p=validate_propensities(pi,len(t)); rows=[]
    for j,label in enumerate(("H","C","L")):
        observed=int(np.sum(t==j)); expected=float(np.sum(p[:,j])); variance=float(np.sum(p[:,j]*(1-p[:,j]))); z=(observed-expected)/np.sqrt(variance) if variance>0 else np.nan
        rows.append({"arm":label,"observed_quotes":observed,"expected_quotes":expected,"standardized_assignment_residual":z})
    return pd.DataFrame(rows)

def numeric_covariate_balance(frame:pd.DataFrame,arm:ArrayLike,numeric_features:list[str])->pd.DataFrame:
    t=as_arm_index(arm); rows=[]
    for feature in numeric_features:
        x=pd.to_numeric(frame[feature],errors="coerce").to_numpy(float); pooled=np.nanstd(x,ddof=1)
        for j,label in ((0,"H_vs_C"),(2,"L_vs_C")):
            diff=np.nanmean(x[t==j])-np.nanmean(x[t==1]); smd=diff/pooled if pooled>0 else np.nan; rows.append({"feature":feature,"contrast":label,"standardized_mean_difference":smd,"missing_fraction":float(np.mean(~np.isfinite(x)))})
    return pd.DataFrame(rows)

def dose_compliance_audit(base_premium:ArrayLike,displayed_premium:ArrayLike,assigned_multiplier:ArrayLike,arm:ArrayLike)->pd.DataFrame:
    base=np.asarray(base_premium,float); displayed=np.asarray(displayed_premium,float); assigned=np.asarray(assigned_multiplier,float); t=as_arm_index(arm); realized=displayed/np.maximum(base,1e-12); error=realized-assigned; rows=[]
    for j,label in enumerate(("H","C","L")):
        m=t==j; rows.append({"arm":label,"n":int(m.sum()),"assigned_mean":float(np.mean(assigned[m])),"realized_mean":float(np.mean(realized[m])),"absolute_error_mean":float(np.mean(np.abs(error[m]))),"p99_absolute_error":float(np.quantile(np.abs(error[m]),.99))})
    return pd.DataFrame(rows)
