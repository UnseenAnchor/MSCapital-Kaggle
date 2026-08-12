"""滚动验证 LGB ensemble：比较不同树配置/训练轮数的融合稳定性。"""
import os,time
import numpy as np,pandas as pd,lightgbm as lgb
SPLITS={'early':(range(0,41),range(41,51)),'middle':(range(0,51),range(51,61)),'late':(range(0,62),range(62,71))}
CFG={
 'base400':(dict(num_leaves=127,max_depth=8,min_child_samples=1000,feature_fraction=.7,bagging_fraction=.8,bagging_freq=1,lambda_l2=2.,extra_trees=False),400),
 'base600':(dict(num_leaves=127,max_depth=8,min_child_samples=1000,feature_fraction=.7,bagging_fraction=.8,bagging_freq=1,lambda_l2=2.,extra_trees=False),600),
 'cons600':(dict(num_leaves=63,max_depth=7,min_child_samples=2000,feature_fraction=.9,bagging_fraction=.9,bagging_freq=1,lambda_l2=5.,extra_trees=False),600),
 'rand600':(dict(num_leaves=63,max_depth=8,min_child_samples=2000,feature_fraction=.7,bagging_fraction=.8,bagging_freq=1,lambda_l2=5.,extra_trees=True),600),
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
 cols=[x for x in df if x not in ('month','sample_id','target')];allpred={};truth={};t0=time.time()
 for sn,(tm,vm) in SPLITS.items():
  tr=df[df.month.isin(tm)];va=df[df.month.isin(vm)];truth[sn]=va.target.to_numpy();allpred[sn]={}
  for name,(cp,r) in CFG.items():
   pa=dict(objective='regression',metric='l2',learning_rate=.03,verbosity=-1,n_jobs=10,seed=42,**cp)
   mod=lgb.train(pa,lgb.Dataset(tr[cols],label=tr.target),num_boost_round=r)
   allpred[sn][name]=mod.predict(va[cols]);print(sn,name,c(truth[sn],allpred[sn][name]),flush=True)
 np.savez('output/rolling_lgb_preds.npz',**{f'{s}_{n}':allpred[s][n] for s in allpred for n in allpred[s]})
 # equal and single candidates
 names=list(CFG); scores=[]
 for n in names:scores.append((n,np.mean([c(truth[s],allpred[s][n]) for s in SPLITS])))
 scores.append(('equal',np.mean([c(truth[s],sum(allpred[s][n] for n in names)/len(names)) for s in SPLITS])))
 # one-dimensional weight on base vs average diversity
 for a in np.arange(0,1.01,.1):
  scores.append((f'base{a:.1f}',np.mean([c(truth[s],a*allpred[s]['base600']+(1-a)*(allpred[s]['cons600']+allpred[s]['rand600'])/2) for s in SPLITS])))
 print('SUMMARY',sorted(scores,key=lambda z:z[1],reverse=True));print('seconds',time.time()-t0)
if __name__=='__main__':main()
