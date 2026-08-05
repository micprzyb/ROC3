import pandas as pd
from elasticity_only.schema import validate_experiment_table
from elasticity_only.splits import make_buyer_group_folds,make_forward_time_holdout

def frame():
    n=12; return pd.DataFrame({'quote_id':[f'q{i}' for i in range(n)],'customer_id':[f'c{i//2}' for i in range(n)],'quote_timestamp':pd.date_range('2025-01-01',periods=n,freq='D',tz='UTC'),'eligible':1,'arm':['H','C','L','C']*3,'purchase':[1,1,1,0]*3,'pi_H':.1,'pi_C':.8,'pi_L':.1})
def test_schema_splits():
    f=frame(); assert not validate_experiment_table(f).has_errors; s=make_forward_time_holdout(f.quote_timestamp,f.customer_id,holdout_fraction=.25); assert len(s.development_index)+len(s.holdout_index)==len(f); b=f[f.purchase==1]; assert len(make_buyer_group_folds(b.arm,b.customer_id,n_splits=3))==3
