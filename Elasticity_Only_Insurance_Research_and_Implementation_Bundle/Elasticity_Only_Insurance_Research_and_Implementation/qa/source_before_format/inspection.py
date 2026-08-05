"""Held-out interpretation and stability diagnostics."""
from __future__ import annotations
from typing import Any,Callable
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .metrics import pairwise_nll

def permutation_importance_log_rr(model:Any,X:pd.DataFrame,arm:np.ndarray,pi:np.ndarray,contrast_arm:int,*,n_repeats:int=5,seed:int=2026)->pd.DataFrame:
    """Permutation importance measured by held-out pairwise NLL increase."""
    rng=np.random.default_rng(seed); baseline=pairwise_nll(arm,pi,model.predict_log_rr(X),contrast_arm); rows=[]
    for col in X.columns:
        changes=[]
        for _ in range(n_repeats):
            xp=X.copy(); xp[col]=rng.permutation(xp[col].to_numpy()); changes.append(pairwise_nll(arm,pi,model.predict_log_rr(xp),contrast_arm)-baseline)
        rows.append({"feature":col,"nll_increase_mean":float(np.mean(changes)),"nll_increase_sd":float(np.std(changes,ddof=1) if len(changes)>1 else 0.)})
    return pd.DataFrame(rows).sort_values("nll_increase_mean",ascending=False)

def prediction_stability(score_matrix:np.ndarray,*,top_fraction:float=.10)->dict[str,float]:
    scores=np.asarray(score_matrix,float)
    if scores.ndim!=2: raise ValueError("score_matrix must be (n,replicates)")
    cors=[]; jacc=[]; k=max(1,int(len(scores)*top_fraction))
    for a in range(scores.shape[1]):
        for b in range(a+1,scores.shape[1]):
            cors.append(spearmanr(scores[:,a],scores[:,b]).statistic); ia=set(np.argsort(scores[:,a])[-k:]); ib=set(np.argsort(scores[:,b])[-k:]); jacc.append(len(ia&ib)/len(ia|ib))
    return {"median_spearman":float(np.nanmedian(cors)),"median_top_fraction_jaccard":float(np.nanmedian(jacc)),"n_replica_pairs":len(cors)}
