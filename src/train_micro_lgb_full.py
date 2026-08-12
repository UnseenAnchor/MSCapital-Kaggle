"""Train full-data 3-seed combined old+micro LGB and predict test. Never submits."""
import gc,time,numpy as np,pandas as pd,lightgbm as lgb
SEEDS=(13,42,77)
def load_old(split):
 if split=='train':base=pd.read_feather('data/train/label.feather').sort_values('sample_id').reset_index(drop=True)
 else:base=pd.read_csv('data/submission.csv')[['sample_id']].sort_values('sample_id').reset_index(drop=True)
 for name in ('market','order','transaction'):
  for suffix in ('','_v2'):
   base=base.merge(pd.read_parquet(f'features/{name}_{split}{suffix}.parquet'),on='sample_id',how='left')
 return base
def unit(x):
 x=np.asarray(x,np.float64);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
def main():
 t0=time.time();train=load_old('train');test=load_old('test');micro_tr=pd.read_parquet('features/micro_v3/train.parquet').sort_values('sample_id').reset_index(drop=True);micro_te=pd.read_parquet('features/micro_v3/test.parquet').sort_values('sample_id').reset_index(drop=True)
 assert np.array_equal(train.sample_id.to_numpy(),micro_tr.sample_id.to_numpy());assert np.array_equal(test.sample_id.to_numpy(),micro_te.sample_id.to_numpy())
 oldcols=[c for c in train if c not in ('sample_id','month','target')];mcols=[c for c in micro_tr if c not in ('sample_id','month','target')]
 Xtr=train[oldcols].join(micro_tr[mcols].add_prefix('v3_'));Xte=test[oldcols].join(micro_te[mcols].add_prefix('v3_'));y=train.target.to_numpy();ids=test.sample_id.to_numpy();del train,test,micro_tr,micro_te;gc.collect();print('matrix',Xtr.shape,Xte.shape,'sec',time.time()-t0,flush=True)
 params=dict(objective='regression',metric='l2',learning_rate=.025,num_leaves=48,min_data_in_leaf=500,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,lambda_l2=10,max_bin=63,force_col_wise=True,verbosity=-1,n_jobs=10)
 ps=[];imp=[]
 for seed in SEEDS:
  params['seed']=seed;m=lgb.train(params,lgb.Dataset(Xtr,label=y),num_boost_round=600);p=m.predict(Xte);ps.append(p);imp.append(m.feature_importance('gain'));m.save_model(f'output/micro_lgb_full_seed{seed}.txt');print('seed',seed,'pred mean/std',p.mean(),p.std(),'sec',time.time()-t0,flush=True);del m;gc.collect()
 names=np.asarray(Xtr.columns);mean_imp=np.mean(imp,axis=0);pd.DataFrame({'feature':names,'gain':mean_imp}).sort_values('gain',ascending=False).to_csv('output/micro_lgb_feature_importance.csv',index=False)
 raw=np.mean(ps,axis=0);u=np.mean([unit(p) for p in ps],axis=0);pd.DataFrame({'sample_id':ids,'prediction':raw}).to_csv('output/submission_micro_lgb_full_raw.csv',index=False);pd.DataFrame({'sample_id':ids,'prediction':u}).to_csv('output/submission_micro_lgb_full_unit.csv',index=False);print('saved',len(ids),'raw',raw.mean(),raw.std(),'unit',u.mean(),u.std(),'total_sec',time.time()-t0)
if __name__=='__main__':main()
