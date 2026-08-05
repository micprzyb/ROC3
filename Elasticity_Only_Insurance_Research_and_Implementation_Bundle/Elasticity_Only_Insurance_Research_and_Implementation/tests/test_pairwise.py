import numpy as np
from sklearn.ensemble import RandomForestClassifier
from elasticity_only.pairwise import IPWPairwiseClassifier

def test_rf_api():
    X=np.arange(60,dtype=float).reshape(20,3); arm=np.array([0,1,2,1,1]*4); pi=np.tile([.1,.8,.1],(len(arm),1)); m=IPWPairwiseClassifier(RandomForestClassifier(n_estimators=20,min_samples_leaf=2,random_state=1),0); m.fit(X,arm,pi=pi); s=m.predict_log_rr(X[:4]); assert s.shape==(4,) and np.isfinite(s).all()
