import numpy as np
from elasticity_only.pairwise import LGBOffsetPairwiseClassifier,XGBOffsetPairwiseClassifier
from elasticity_only.scalar import LGBScalarFlipRegressor,XGBScalarFlipRegressor

def toy():
    rng=np.random.default_rng(7); n=90; X=rng.normal(size=(n,4)).astype('float32'); arm=np.tile(np.array([0,1,1,1,1,1,1,1,2]),10); pi=np.tile([.1,.8,.1],(n,1)); return X,arm,pi
def test_pairwise_smoke():
    X,a,p=toy(); models=[XGBOffsetPairwiseClassifier(0,params={'max_depth':2,'eta':.1,'min_child_weight':1,'nthread':1},num_boost_round=5),LGBOffsetPairwiseClassifier(0,params={'max_depth':2,'num_leaves':4,'learning_rate':.1,'min_data_in_leaf':5,'min_sum_hessian_in_leaf':.01,'num_threads':1},num_boost_round=5)]
    for m in models: m.fit(X,a,pi=p); s=m.predict_log_rr(X[:5]); assert s.shape==(5,) and np.isfinite(s).all()
def test_scalar_smoke():
    X,a,p=toy(); models=[XGBScalarFlipRegressor(params={'max_depth':2,'eta':.1,'min_child_weight':1e-4,'nthread':1},num_boost_round=3),LGBScalarFlipRegressor(params={'max_depth':2,'num_leaves':4,'learning_rate':.1,'min_data_in_leaf':5,'min_sum_hessian_in_leaf':1e-5,'num_threads':1},num_boost_round=3)]
    for m in models: m.fit(X,a,pi=p); b=m.predict(X[:5]); assert b.shape==(5,) and np.isfinite(b).all() and np.all(b>=0)
