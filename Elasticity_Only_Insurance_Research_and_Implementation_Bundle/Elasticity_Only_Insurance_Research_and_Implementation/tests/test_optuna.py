import optuna
from elasticity_only.optuna_spaces import dynamic_leaf_bounds,suggest_lightgbm_pairwise

def test_bounds():
    b=dynamic_leaf_bounds(5000,.08); assert 100<=b.lower<=b.upper<=2500
def test_space():
    t=optuna.trial.FixedTrial({'max_depth':4,'num_leaves':12,'min_gain_mode':'zero','learning_rate':.03,'min_data_in_leaf':200,'min_sum_hessian_in_leaf':5.,'feature_fraction':.8,'bagging_fraction':.8,'bagging_freq':2,'lambda_l1':.01,'lambda_l2':10.,'max_bin':255}); assert suggest_lightgbm_pairwise(t,n_pair_buyers=5000,minority_fraction=.08)['num_leaves']==12
