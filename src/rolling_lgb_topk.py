"""严格滚动验证 LGB Top-K 特征：每个 split 只用训练集重要性选特征。"""
import os,time,numpy as np,pandas as pd,lightgbm as lgb
SPLITS={'early':(range(0,41),range(41,51)),'middle':(range(0,51),range(51,61)),'late':(range(0,62),range(62,71))}
def c(y,p):y=y-y.mean();p=p-p.mean();return float(np.dot(y,p)/(np.linalg.norm(y)*np.linalg.norm(p)+1e-12))
def main():
 lab=pd.read_feather('data/train/label.feather');ds=[lab]
 for n in ['market','order','transaction']:
  for v in ['','_v2']:
   p=f'features/{n}_train{v}.parquet'
   if os.path.exists(p):ds.append(pd.read_parquet(p))
 df=ds[0]
 for d in ds[1:]:df=df.merge(d,on='sample_id',how='left')
 cols=[x for x in df if x not in ('month','sample_id','target')];rows=[];tops={}
 for sn,(tm,vm) in SPLITS.items():
  tr=df[df.month.isin(tm)];va=df[df.month.isin(vm)];pa=dict(objective='regression',metric='l2',learning_rate=.03,num_leaves=127,max_depth=8,min_child_samples=1000,feature_fraction=.7,bagging_fraction=.8,bagging_freq=1,lambda_l2=2.,verbosity=-1,n_jobs=10,seed=42)
  full=lgb.train(pa,lgb.Dataset(tr[cols],label=tr.target),num_boost_round=600);imp=pd.Series(full.feature_importance(importance_type='gain'),index=cols).sort_values(ascending=False);tops[sn]=imp.index.tolist();y=va.target.to_numpy();rows.append((sn,'all',c(y,full.predict(va[cols],num_iteration=600))))
  for k in [16,32,48,64,80]:
   fs=tops[sn][:k];m=lgb.train(pa,lgb.Dataset(tr[fs],label=tr.target),num_boost_round=600);rows.append((sn,f'top{k}',c(y,m.predict(va[fs],num_iteration=600))))
  print(sn,'top features:',', '.join(tops[sn][:20]),flush=True)
 out=pd.DataFrame(rows,columns=['split','features','cosine']);out.to_csv('output/rolling_lgb_topk.csv',index=False);print(out.to_string(index=False));print('\nmean');print(out.groupby('features').cosine.agg(['mean','std','min']).sort_values('mean',ascending=False))
if __name__=='__main__':main()
