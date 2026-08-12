"""Compare old92 vs micro342 vs combined features on middle/late chronological folds."""
import os,time,numpy as np,pandas as pd,lightgbm as lgb
FOLDS={'middle':(51,61),'late':(62,71)};SEEDS=[13,42,77]
def cos(y,p):y=y-y.mean();p=p-p.mean();return float(np.dot(y,p)/(np.linalg.norm(y)*np.linalg.norm(p)+1e-12))
def old():
 l=pd.read_feather('data/train/label.feather');d=l
 for n in ['market','order','transaction']:
  for v in ['','_v2']:d=d.merge(pd.read_parquet(f'features/{n}_train{v}.parquet'),on='sample_id',how='left')
 return d
def main():
 o=old();m=pd.read_parquet('features/micro_v3/train.parquet');meta=o[['sample_id','month','target']];oldc=[c for c in o if c not in meta];mc=[c for c in m if c not in meta];sets={'old92':o[oldc],'micro342':m[mc],'combined':o[oldc].join(m[mc].add_prefix('v3_'))};rows=[];store={};t0=time.time()
 for fn,(st,en) in FOLDS.items():
  tr=meta.month<st;va=(meta.month>=st)&(meta.month<en);y=meta.loc[va,'target'].to_numpy();store[fn]={}
  for sn,X in sets.items():
   ps=[]
   for seed in SEEDS:
    pa=dict(objective='regression',metric='l2',learning_rate=.025,num_leaves=48,min_data_in_leaf=500,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,lambda_l2=10,max_bin=63,force_col_wise=True,verbosity=-1,n_jobs=10,seed=seed);q=lgb.train(pa,lgb.Dataset(X.loc[tr],label=meta.loc[tr,'target']),num_boost_round=600);p=q.predict(X.loc[va]);ps.append(p);rows.append((fn,sn,f'seed{seed}',cos(y,p)));print(rows[-1],flush=True)
   avg=np.mean(ps,0);store[fn][sn]=avg;rows.append((fn,sn,'avg3',cos(y,avg)));print(rows[-1],flush=True)
 out=pd.DataFrame(rows,columns=['fold','features','model','cosine']);out.to_csv('output/rolling_micro_lgb.csv',index=False);np.savez('output/rolling_micro_lgb_preds.npz',**{f'{f}_{s}':p for f,v in store.items() for s,p in v.items()});print(out[out.model=='avg3'].pivot(index='features',columns='fold',values='cosine'));print('sec',time.time()-t0)
if __name__=='__main__':main()
