"""Nested chronological residual LGB against the self-model backbone. Never submits."""
import time,numpy as np,pandas as pd,lightgbm as lgb
from proxy_lgb_feature_select import load_combined

def unit(x):x=np.asarray(x,np.float64);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
def cosine(y,p):return float(unit(y)@unit(p))
def stats(y,p,mo):
 vals=[cosine(y[mo==m],p[mo==m]) for m in np.unique(mo)];return cosine(y,p),float(np.mean(vals)),float(np.min(vals)),float(np.std(vals))
def project_residual(y,p):
 y0=np.asarray(y,np.float64)-np.mean(y);p0=np.asarray(p,np.float64)-np.mean(p);beta=float(y0@p0/(p0@p0+1e-12));return y0-beta*p0,beta

def backbone(fold,ids):
 if fold=='proxy':
  l=np.load('output/proxy_lgb_oof.npz');v=np.load('output/multistream_v3_proxy_oof.npz');r=np.load('output/realmlp_multiseed_proxy_oof.npz');assert all(np.array_equal(ids,z['sample_id']) for z in [l,v,r]);return .4*unit(l['prediction'])+.5*unit(v['ens4_5_6'])+.1*unit(r['s42'])
 if fold=='middle':
  v=np.load('output/multistream_v3_middle_eff1024_oof.npz');r=np.load('output/realmlp_multiseed_rolling_oof.npz');assert np.array_equal(ids,v['sample_id']) and np.array_equal(ids,r['middle_sample_id']);return .4*unit(r['middle_lgb'])+.5*unit(v['ens4_5_6'])+.1*unit(r['middle_s42'])
 z=np.load('output/exact_public140_late_reconstruction.npz');assert np.array_equal(ids,z['sample_id']);return z['public140']
def main():
 t0=time.time();lab,X=load_combined();mo=lab.month.to_numpy();y=lab.target.to_numpy(np.float64);sid=lab.sample_id.to_numpy();params=dict(objective='regression',metric='l2',learning_rate=.025,num_leaves=32,min_data_in_leaf=500,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,lambda_l2=20,max_bin=63,force_col_wise=True,verbosity=-1,n_jobs=10,seed=42);out={}
 for fold,train_end,valid_end in [('proxy',45,71),('middle',51,61),('late',62,71)]:
  inner=train_end-14;fit=mo<inner;res=(mo>=inner)&(mo<train_end);va=(mo>=train_end)&(mo<valid_end);print('\n',fold,'basefit',fit.sum(),'resfit',res.sum(),'valid',va.sum(),flush=True)
  bm=lgb.train(params,lgb.Dataset(X.loc[fit],label=y[fit]),num_boost_round=600);pr=bm.predict(X.loc[res]);rv,beta=project_residual(y[res],pr);rm=lgb.train(params,lgb.Dataset(X.loc[res],label=rv),num_boost_round=600);q=rm.predict(X.loc[va]);ids=sid[va];base=backbone(fold,ids);yt=y[va];mv=mo[va];true_res,_=project_residual(yt,base);print('beta',beta,'res std',rv.std(),'q',stats(yt,q,mv),'corr_base',float(unit(q)@unit(base)),'partial',cosine(true_res,q),flush=True);print('base',stats(yt,base,mv),flush=True)
  for w in [.02,.05,.10,.15,.20]:print(' residual_weight',w,stats(yt,(1-w)*unit(base)+w*unit(q),mv),flush=True)
  out.update({f'{fold}_sample_id':ids,f'{fold}_target':yt,f'{fold}_month':mv,f'{fold}_base':base,f'{fold}_residual':q,f'{fold}_train_sample_id':sid[res],f'{fold}_train_residual_target':rv})
 np.savez('output/residual_lgb_rolling_oof.npz',**out);print('sec',time.time()-t0)
if __name__=='__main__':main()
