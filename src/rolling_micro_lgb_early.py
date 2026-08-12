"""Complete the early chronological fold for combined microstructure features."""
import numpy as np,pandas as pd,lightgbm as lgb
SEEDS=(13,42,77)
def cosine(y,p):
 y=np.asarray(y,np.float64);p=np.asarray(p,np.float64);y-=y.mean();p-=p.mean()
 return float(y@p/(np.linalg.norm(y)*np.linalg.norm(p)+1e-12))
def main():
 lab=pd.read_feather('data/train/label.feather').sort_values('sample_id').reset_index(drop=True)
 old=lab[['sample_id']]
 for name in ('market','order','transaction'):
  for suffix in ('','_v2'):
   old=old.merge(pd.read_parquet(f'features/{name}_train{suffix}.parquet'),on='sample_id',how='left')
 micro=pd.read_parquet('features/micro_v3/train.parquet').sort_values('sample_id').reset_index(drop=True)
 old_cols=[c for c in old if c!='sample_id'];micro_cols=[c for c in micro if c not in ('sample_id','month','target')]
 X=old[old_cols].join(micro[micro_cols].add_prefix('v3_'))
 tr=lab.month.to_numpy()<41;va=(lab.month.to_numpy()>=41)&(lab.month.to_numpy()<51);y=lab.target.to_numpy()[va];ps=[]
 params=dict(objective='regression',metric='l2',learning_rate=.025,num_leaves=48,min_data_in_leaf=500,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,lambda_l2=10,max_bin=63,force_col_wise=True,verbosity=-1,n_jobs=10)
 for seed in SEEDS:
  params['seed']=seed;m=lgb.train(params,lgb.Dataset(X.loc[tr],label=lab.target.loc[tr]),num_boost_round=600);p=m.predict(X.loc[va]);ps.append(p);print(seed,cosine(y,p),flush=True)
 avg=np.mean(ps,axis=0);print('avg3',cosine(y,avg),flush=True)
 np.savez('output/rolling_micro_lgb_early_preds.npz',sample_id=lab.sample_id.to_numpy()[va],target=y,**{f'seed{s}':p for s,p in zip(SEEDS,ps)},combined=avg)
if __name__=='__main__':main()
