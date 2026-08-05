import numpy as np
from elasticity_only.core import DEFAULT_PI,DEFAULT_Z
from elasticity_only.scalar import RandomForestMomentElasticity,fit_global_scalar_beta,scalar_flip_nll_rows,scalar_raw_grad_hess

def test_gradient():
    arm=np.array([0,1,2,1,0]); pi=np.tile(DEFAULT_PI,(len(arm),1)); z=np.tile(DEFAULT_Z,(len(arm),1)); raw=np.array([.1,-.4,.7,0,.3]); grad,_=scalar_raw_grad_hess(raw,arm,pi,z); eps=1e-6; num=[]
    for i in range(len(raw)):
        plus=raw.copy(); minus=raw.copy(); plus[i]+=eps; minus[i]-=eps; bp=np.log1p(np.exp(plus)); bm=np.log1p(np.exp(minus)); num.append((scalar_flip_nll_rows(bp,arm,pi,z)[i]-scalar_flip_nll_rows(bm,arm,pi,z)[i])/(2*eps))
    np.testing.assert_allclose(grad,num,rtol=1e-5,atol=1e-6)
def test_global_positive(): assert fit_global_scalar_beta(np.array([2]*40+[1]*50+[0]*10)).beta>0
def test_rf_moment_center_invariant():
    m=RandomForestMomentElasticity(beta_max=20); beta=np.array([0.,2.,8.]); z=np.tile(DEFAULT_Z+2.5,(3,1)); means=m._mean_dose(beta,z); assert np.all(np.diff(means)<0); np.testing.assert_allclose(means,m._mean_dose(beta,np.tile(DEFAULT_Z,(3,1))))
