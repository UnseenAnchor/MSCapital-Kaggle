import os,numpy as np,pandas as pd,lightgbm as lgb
SPLITS={'early':(range(0,41),range(41,51)),'middle':(range(0,51),range(51,61)),'late':(range(0,62),range(62,71))};SEEDS=[13,42,77]
def c(y,p):y=y-y.mean();p=p-p.mean();return float(np.dot(y,p)/(np.linalg.norm(y)*np.linalg.norm(p)+1e-12))
def main():
 lab=pd.read_feather('data/train/label.feather');ds=[lab]
 for n in ['market','order','transaction']:
  for v in ['','_v2']:
   p=f'features/{n}_train{v}.parquet'
   if os.path.exists(p):ds.append(pd.read_parquet(p))
 df=ds[0]
 for d in ds[1:]:df=df.merge(d,on='sample_id',how='left')
 cols=[x for x in df if x not in ('month','sample_id','target')];rows=[];predstore={}
 for sn,(tm,vm) in SPLITS.items():
  tr=df[df.month.isin(tm)];va=df[df.month.isin(vm)];ps=[];y=va.target.to_numpy();predstore[sn]={}
  for seed in SEEDS:
   pa=dict(objective='regression',metric='l2',learning_rate=.03,num_leaves=127,max_depth=8,min_child_samples=1000,feature_fraction=.7,bagging_fraction=.8,bagging_freq=1,lambda_l2=2.,verbosity=-1,n_jobs=10,seed=seed);m=lgb.train(pa,lgb.Dataset(tr[cols],label=tr.target),num_boost_round=600);ps.append(m.predict(va[cols]));predstore[sn][f'seed{seed}']=ps[-1];rows.append((sn,f'seed{seed}',c(y,ps[-1])))
  for k in [2,3]:predstore[sn][f'avg{k}']=np.mean(ps[:k],0);rows.append((sn,f'avg{k}',c(y,predstore[sn][f'avg{k}'])))
  print(sn,[(x[1],round(x[2],5)) for x in rows if x[0]==sn],flush=True)
 out=pd.DataFrame(rows,columns=['split','model','cosine']);out.to_csv('output/rolling_lgb_seeds.csv',index=False);np.savez('output/rolling_lgb_seed_preds.npz',**{f'{s}_{n}':predstore[s][n] for s in predstore for n in predstore[s]});print(out.groupby('model').cosine.agg(['mean','std','min']).sort_values('mean',ascending=False))
if __name__=='__main__':main()
