"""严格滚动验证：LGB 训练 target 轻度裁剪，测试极端值是否导致过拟合。"""
import os,numpy as np,pandas as pd,lightgbm as lgb
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
 cols=[x for x in df if x not in ('month','sample_id','target')];rows=[]
 for sn,(tm,vm) in SPLITS.items():
  tr=df[df.month.isin(tm)];va=df[df.month.isin(vm)];pa=dict(objective='regression',metric='l2',learning_rate=.03,num_leaves=127,max_depth=8,min_child_samples=1000,feature_fraction=.7,bagging_fraction=.8,bagging_freq=1,lambda_l2=2.,verbosity=-1,n_jobs=10,seed=42);X=tr[cols];y=tr.target.to_numpy();yv=va.target.to_numpy()
  for q in [None,.005,.01,.02,.05]:
   yy=y.copy()
   if q is not None:
    lo,hi=np.quantile(yy,[q,1-q]);yy=np.clip(yy,lo,hi)
   m=lgb.train(pa,lgb.Dataset(X,label=yy),num_boost_round=600);p=m.predict(va[cols]);rows.append((sn,'none' if q is None else f'clip{q}',c(yv,p),p.std()))
 out=pd.DataFrame(rows,columns=['split','mode','cosine','pred_std']);out.to_csv('output/rolling_lgb_clip.csv',index=False);print(out.to_string(index=False));print(out.groupby('mode').cosine.agg(['mean','std','min']).sort_values('mean',ascending=False))
if __name__=='__main__':main()
