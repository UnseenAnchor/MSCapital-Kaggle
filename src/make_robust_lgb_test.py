"""训练全量 robust LGB ensemble，生成候选，不上传。权重来自滚动验证：
base600 40% + conservative600 30% + randomized600 30%。"""
import os,time,numpy as np,pandas as pd,lightgbm as lgb
CFG={
 'base600':dict(num_leaves=127,max_depth=8,min_child_samples=1000,feature_fraction=.7,bagging_fraction=.8,bagging_freq=1,lambda_l2=2.,extra_trees=False),
 'cons600':dict(num_leaves=63,max_depth=7,min_child_samples=2000,feature_fraction=.9,bagging_fraction=.9,bagging_freq=1,lambda_l2=5.,extra_trees=False),
 'rand600':dict(num_leaves=63,max_depth=8,min_child_samples=2000,feature_fraction=.7,bagging_fraction=.8,bagging_freq=1,lambda_l2=5.,extra_trees=True),
}
def load_train():
 l=pd.read_feather('data/train/label.feather');ds=[l]
 for n in ['market','order','transaction']:
  for v in ['','_v2']:
   p=f'features/{n}_train{v}.parquet'
   if os.path.exists(p):ds.append(pd.read_parquet(p))
 d=ds[0]
 for x in ds[1:]:d=d.merge(x,on='sample_id',how='left')
 return d,l

def load_test():
 ds=[]
 for n in ['market','order','transaction']:
  for v in ['','_v2']:
   p=f'features/{n}_test{v}.parquet'
   if os.path.exists(p):ds.append(pd.read_parquet(p))
 d=ds[0]
 for x in ds[1:]:d=d.merge(x,on='sample_id',how='left')
 return d

def main():
 t=time.time();tr,_=load_train();te=load_test();cols=[c for c in tr if c not in ('month','sample_id','target')];ps=[]
 for n,cp in CFG.items():
  pa=dict(objective='regression',metric='l2',learning_rate=.03,verbosity=-1,n_jobs=10,seed=42,**cp);m=lgb.train(pa,lgb.Dataset(tr[cols],label=tr.target),num_boost_round=600);p=m.predict(te[cols]);ps.append(p);print(n,p.mean(),p.std(),flush=True)
 p=.4*ps[0]+.3*ps[1]+.3*ps[2];pd.DataFrame({'sample_id':te.sample_id,'prediction':p}).sort_values('sample_id').to_csv('output/submission_lgb_robust.csv',index=False);print('done',len(p),time.time()-t)
if __name__=='__main__':main()
