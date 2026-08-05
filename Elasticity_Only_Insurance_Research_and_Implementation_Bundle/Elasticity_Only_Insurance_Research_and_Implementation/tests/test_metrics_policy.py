import numpy as np
from elasticity_only.core import buyer_posterior_from_log_rr
from elasticity_only.metrics import buyer_multiclass_nll,equal_contrast_nll,fit_global_pairwise_log_rr
from elasticity_only.policy import evaluate_policy_ips,uniform_policy

def test_metrics():
    arm=np.array([0,1,2,1,0,2,1,1]); pi=np.tile([.1,.8,.1],(len(arm),1)); h=np.full(len(arm),-.2); l=np.full(len(arm),.3); post=buyer_posterior_from_log_rr(h,l,pi); assert np.isfinite(buyer_multiclass_nll(arm,post)); assert np.isfinite(equal_contrast_nll(arm,pi,h,l)); assert fit_global_pairwise_log_rr(arm,pi,0).converged
def test_ips():
    obs=np.array([0,1,2,1,0,2]); y=np.array([1,1,0,0,1,1]); pi=np.tile([1/3]*3,(len(y),1)); c=np.ones((len(y),3)); val=evaluate_policy_ips(uniform_policy(len(y),0),obs,y,pi,c); assert abs(val.ips-np.mean((obs==0)*y*3))<1e-12


def test_cluster_bootstrap_policy_difference_contract():
    import numpy as np
    from elasticity_only.policy import cluster_bootstrap_policy_difference

    observed_arm = np.array([0, 1, 2, 0, 1, 2])
    purchase = np.array([1, 0, 1, 0, 1, 0])
    pi = np.tile([1 / 3, 1 / 3, 1 / 3], (6, 1))
    contribution = np.ones((6, 3))
    clusters = np.array([1, 1, 2, 2, 3, 3])
    result = cluster_bootstrap_policy_difference(
        np.ones(6, dtype=int),
        np.zeros(6, dtype=int),
        observed_arm,
        purchase,
        pi,
        contribution,
        clusters,
        n_bootstrap=50,
        seed=7,
    )
    assert result.n_clusters == 3
    assert np.isfinite(result.estimate)
    assert result.lower <= result.upper
