"""Policy construction and randomized offline evaluation without a conversion model."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from numpy.typing import ArrayLike,NDArray
from .core import as_arm_index,choose_price_from_relative_demand,validate_propensities
FloatArray=NDArray[np.float64]; IntArray=NDArray[np.int64]

@dataclass(frozen=True)
class PolicyValue:
    ips: float
    snips: float
    match_rate: float
    effective_matched_weight: float
    n_quotes: int


@dataclass(frozen=True)
class PolicyDifferenceInterval:
    estimate: float
    lower: float
    upper: float
    n_clusters: int

def elasticity_policy(log_rr_high:ArrayLike,log_rr_low:ArrayLike,contribution_if_bought:ArrayLike)->IntArray:
    h=np.asarray(log_rr_high,float); l=np.asarray(log_rr_low,float); rr=np.column_stack([np.exp(h),np.ones(len(h)),np.exp(l)])
    return choose_price_from_relative_demand(rr,contribution_if_bought).chosen_arm

def evaluate_policy_ips(policy_arm:ArrayLike,observed_arm:ArrayLike,purchase:ArrayLike,pi:ArrayLike,contribution_if_bought:ArrayLike,*,quote_weight:ArrayLike|None=None)->PolicyValue:
    d=as_arm_index(policy_arm); t=as_arm_index(observed_arm); y=np.asarray(purchase,float); p=validate_propensities(pi,len(y)); contribution=np.asarray(contribution_if_bought,float)
    if contribution.shape!=(len(y),3) or len(d)!=len(y) or len(t)!=len(y): raise ValueError("policy inputs must align")
    qw=np.ones(len(y)) if quote_weight is None else np.asarray(quote_weight,float); match=t==d; assigned=p[np.arange(len(y)),t]; reward=y*contribution[np.arange(len(y)),t]; iw=qw*match/assigned
    ips=float(np.sum(iw*reward)/np.sum(qw)); denom=np.sum(iw); snips=float(np.sum(iw*reward)/denom) if denom>0 else np.nan; ess=float(denom**2/np.sum(iw**2)) if np.sum(iw**2)>0 else 0.
    return PolicyValue(ips,snips,float(np.average(match,weights=qw)),ess,len(y))

def uniform_policy(n:int,arm:int)->IntArray: return np.full(n,arm,dtype=np.int64)

def selective_policy(candidate_arm:ArrayLike,relative_gain:ArrayLike,*,fraction:float,default_arm:int=1)->IntArray:
    if not 0<=fraction<=1: raise ValueError("fraction must be in [0,1]")
    cand=as_arm_index(candidate_arm); gain=np.asarray(relative_gain,float); n=max(0,int(np.floor(len(cand)*fraction))); out=np.full(len(cand),default_arm,dtype=np.int64)
    if n: idx=np.argsort(gain)[-n:]; out[idx]=cand[idx]
    return out

def cluster_bootstrap_policy_difference(policy_a:ArrayLike,policy_b:ArrayLike,observed_arm:ArrayLike,purchase:ArrayLike,pi:ArrayLike,contribution_if_bought:ArrayLike,clusters:ArrayLike,*,n_bootstrap:int=1000,seed:int=2026)->PolicyDifferenceInterval:
    da=as_arm_index(policy_a); db=as_arm_index(policy_b); t=as_arm_index(observed_arm); y=np.asarray(purchase,float); p=validate_propensities(pi,len(y)); c=np.asarray(contribution_if_bought,float); cl=np.asarray(clusters)
    reward=y*c[np.arange(len(y)),t]; assigned=p[np.arange(len(y)),t]; row_a=(t==da)*reward/assigned; row_b=(t==db)*reward/assigned; diff=row_a-row_b; unique,inv=np.unique(cl,return_inverse=True); sums=np.bincount(inv,weights=diff); counts=np.bincount(inv); rng=np.random.default_rng(seed); draws=[]
    for _ in range(n_bootstrap):
        idx=rng.integers(0,len(unique),len(unique)); draws.append(sums[idx].sum()/counts[idx].sum())
    return PolicyDifferenceInterval(
        estimate=float(np.mean(diff)),
        lower=float(np.quantile(draws, .025)),
        upper=float(np.quantile(draws, .975)),
        n_clusters=len(unique),
    )
