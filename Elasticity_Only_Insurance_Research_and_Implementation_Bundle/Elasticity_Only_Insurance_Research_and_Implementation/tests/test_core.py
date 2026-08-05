import numpy as np
from elasticity_only.core import DEFAULT_PI,buyer_posterior_from_log_rr,choose_price_from_relative_demand,log_rr_from_scalar_beta,risk_ratios_from_buyer_posterior,scalar_buyer_posterior

def test_bayes_flip_round_trip():
    h=np.array([-.3,-.1]); l=np.array([.4,.2]); post=buyer_posterior_from_log_rr(h,l,DEFAULT_PI); rr=risk_ratios_from_buyer_posterior(post,DEFAULT_PI)
    np.testing.assert_allclose(rr[:,0],np.exp(h)); np.testing.assert_allclose(rr[:,1],1); np.testing.assert_allclose(rr[:,2],np.exp(l))
def test_scalar_agreement():
    b=np.array([1.,3.,6.]); h,l=log_rr_from_scalar_beta(b); np.testing.assert_allclose(scalar_buyer_posterior(b),buyer_posterior_from_log_rr(h,l))
def test_policy_cancels_scale():
    rr=np.array([[.9,1,1.1],[.6,1,1.5]]); m=np.array([[120,100,80],[120,100,80]],float); d=choose_price_from_relative_demand(rr,m); np.testing.assert_array_equal(d.chosen_arm,[0,2])
