"""Leakage-safe long-horizon proxy CV and train-only feature ranking.
Train months 0-44 (~797k), validate months 45-70 (~460k). Never submits.
"""
import gc,time,numpy as np,pandas as pd,lightgbm as lgb

def cosine(y,p):
 y=np.asarray(y,np.float64);p=np.asarray(p,np.float64);y-=y.mean();p-=p.mean()
 return float(y@p/(np.linalg.norm(y)*np.linalg.norm(p)+1e-12))
def load_combined():
 lab=pd.read_feather('data/train/label.feather').sort_values('sample_id').reset_index(drop=True);d=lab[['sample_id']]
 for name in ('market','order','transaction'):
  for suffix in ('','_v2'):
   d=d.merge(pd.read_parquet(f'features/{name}_train{suffix}.parquet'),on='sample_id',how='left')
 micro=pd.read_parquet('features/micro_v3/train.parquet').sort_values('sample_id').reset_index(drop=True)
 assert np.array_equal(d.sample_id.to_numpy(),micro.sample_id.to_numpy())
 old=[c for c in d if c!='sample_id'];new=[c for c in micro if c not in ('sample_id','month','target')]
 X=d[old].join(micro[new].add_prefix('v3_'));return lab,X

def main():
 t0=time.time();lab,X=load_combined();mo=lab.month.to_numpy();tr=mo<45;va=mo>=45;y=lab.target.to_numpy();print('matrix',X.shape,'train',tr.sum(),'valid',va.sum(),flush=True)
 params=dict(objective='regression',metric='l2',learning_rate=.025,num_leaves=48,min_data_in_leaf=500,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,lambda_l2=10,max_bin=63,force_col_wise=True,verbosity=-1,n_jobs=10,seed=42)
 model=lgb.train(params,lgb.Dataset(X.loc[tr],label=y[tr]),num_boost_round=600);pred=model.predict(X.loc[va]);score=cosine(y[va],pred);print('proxy combined434',score,'sec',time.time()-t0,flush=True)
 imp=pd.DataFrame({'feature':X.columns,'gain':model.feature_importance('gain'),'split':model.feature_importance('split')}).sort_values('gain',ascending=False);imp.to_csv('output/proxy_lgb_trainonly_importance.csv',index=False)
 rows=[]
 for m in range(45,71):
  mm=mo[va]==m;rows.append((m,int(mm.sum()),cosine(y[va][mm],pred[mm])))
 out=pd.DataFrame(rows,columns=['month','n','cosine']);out.to_csv('output/proxy_lgb_monthly.csv',index=False);print(out.to_string(index=False));print('month mean/min/std',out.cosine.mean(),out.cosine.min(),out.cosine.std())
 np.savez('output/proxy_lgb_oof.npz',sample_id=lab.sample_id.to_numpy()[va],target=y[va],prediction=pred,month=mo[va]);print('\nTOP30\n',imp.head(30).to_string(index=False))
if __name__=='__main__':main()
