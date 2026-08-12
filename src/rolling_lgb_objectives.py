import os,numpy as np,pandas as pd,lightgbm as lgb
SPLITS={'early':(range(0,41),range(41,51)),'middle':(range(0,51),range(51,61)),'late':(range(0,62),range(62,71))}
CFG={
 'goss':dict(boosting_type='goss',top_rate=.2,other_rate=.1,objective='regression',num_leaves=127,max_depth=8,min_child_samples=1000,lambda_l2=2.),
 'huber':dict(boosting_type='gbdt',objective='huber',alpha=.9,num_leaves=127,max_depth=8,min_child_samples=1000,lambda_l2=2.),
 'huber_reg':dict(boosting_type='gbdt',objective='huber',alpha=.7,num_leaves=63,max_depth=7,min_child_samples=2000,lambda_l2=5.),
}
def c(y,p):y=y-y.mean();p=p-p.mean();return float(np.dot(y,p)/(np.linalg.norm(y)*np.linalg.norm(p)+1e-12))
def main():
 lab=pd.read_feather('data/train/label.feather');ds=[lab]
 for n in ['market','order','transaction']:
  for v in ['','_v2']:
   p=f'features/{n}_train{v}.parquet'
   if os.path.exists(p):ds.append(pd.read_parquet(p))
 df=ds[0]
 for d in ds[1:]:df=df.merge(d,on='sample_id',how='left')
 cols=[x for x in df if x not in ('month','sample_id','target')];rows=[]
 for sn,(tm,vm) in SPLITS.items():
  tr=df[df.month.isin(tm)];va=df[df.month.isin(vm)];y=va.target.to_numpy()
  for name,cp in CFG.items():
   pa=dict(metric='l2',learning_rate=.03,feature_fraction=.7,verbosity=-1,n_jobs=10,seed=42,**cp)
   if name != 'goss': pa.update(bagging_fraction=.8,bagging_freq=1)
   m=lgb.train(pa,lgb.Dataset(tr[cols],label=tr.target),num_boost_round=600);p=m.predict(va[cols]);rows.append((sn,name,c(y,p)));print(sn,name,rows[-1][-1],flush=True)
 out=pd.DataFrame(rows,columns=['split','model','cosine']);print(out.to_string(index=False));print(out.groupby('model').cosine.agg(['mean','std','min']).sort_values('mean',ascending=False));out.to_csv('output/rolling_lgb_objectives.csv',index=False)
if __name__=='__main__':main()
