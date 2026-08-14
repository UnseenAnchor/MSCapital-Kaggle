"""Nested chronological Residual-RealMLP on train-only selected features. Never submits."""
import gc,time,math,random,numpy as np,pandas as pd,torch,lightgbm as lgb
from proxy_lgb_feature_select import load_combined
from train_realmlp_proxy_v4 import RealMLP,RobustSmooth,EMA,loss_fn,unit,cosine,metrics
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu');BS=1024;EPOCHS=12;TOPN=128;SEED=42;Y_SCALE=1000.
def project_residual(y,p):
 y0=np.asarray(y,np.float64)-np.mean(y);p0=np.asarray(p,np.float64)-np.mean(p);beta=float(y0@p0/(p0@p0+1e-12));return y0-beta*p0,beta
def backbone(fold,ids):
 if fold=='proxy':
  l=np.load('output/proxy_lgb_oof.npz');v=np.load('output/multistream_v3_proxy_oof.npz');r=np.load('output/realmlp_multiseed_proxy_oof.npz');assert all(np.array_equal(ids,z['sample_id']) for z in [l,v,r]);return .4*unit(l['prediction'])+.5*unit(v['ens4_5_6'])+.1*unit(r['s42'])
 if fold=='middle':
  v=np.load('output/multistream_v3_middle_eff1024_oof.npz');r=np.load('output/realmlp_multiseed_rolling_oof.npz');assert np.array_equal(ids,v['sample_id']) and np.array_equal(ids,r['middle_sample_id']);return .4*unit(r['middle_lgb'])+.5*unit(v['ens4_5_6'])+.1*unit(r['middle_s42'])
 z=np.load('output/exact_public140_late_reconstruction.npz');assert np.array_equal(ids,z['sample_id']);return z['public140']
def fold_stats(y,p,mo):
 vals=[cosine(y[mo==m],p[mo==m]) for m in np.unique(mo)];return cosine(y,p),float(np.mean(vals)),float(np.min(vals)),float(np.std(vals))
@torch.no_grad()
def predict(model,X,idx,bs=4096):
 model.eval();return np.concatenate([model(X[idx[i:i+bs]]).mean(1).float().cpu().numpy() for i in range(0,len(idx),bs)])
def main():
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);t0=time.time();lab,Xdf=load_combined();mo=lab.month.to_numpy();y=lab.target.to_numpy(np.float64);sid=lab.sample_id.to_numpy();params=dict(objective='regression',metric='l2',learning_rate=.025,num_leaves=32,min_data_in_leaf=500,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,lambda_l2=20,max_bin=63,force_col_wise=True,verbosity=-1,n_jobs=10,seed=SEED);saved={}
 for fold,train_end,valid_end in [('proxy',45,71),('middle',51,61),('late',62,71)]:
  inner=train_end-14;fit=mo<inner;tr=(mo>=inner)&(mo<train_end);va=(mo>=train_end)&(mo<valid_end);tri=np.flatnonzero(tr);vai=np.flatnonzero(va);print('\n',fold,'basefit',fit.sum(),'resfit',len(tri),'valid',len(vai),flush=True)
  bm=lgb.train(params,lgb.Dataset(Xdf.loc[fit],label=y[fit]),num_boost_round=600);pr=bm.predict(Xdf.loc[tr]);rv,beta=project_residual(y[tr],pr);rank=pd.DataFrame({'feature':Xdf.columns,'gain':bm.feature_importance('gain')})
  # Residual-specific ranking, fit only on the inner residual window.
  selector=lgb.train(params,lgb.Dataset(Xdf.loc[tr],label=rv),num_boost_round=600);rank=pd.DataFrame({'feature':Xdf.columns,'gain':selector.feature_importance('gain')}).sort_values('gain',ascending=False);cols=rank.feature.head(TOPN).tolist();X=Xdf[cols].replace([np.inf,-np.inf],np.nan).to_numpy(np.float64);sc=RobustSmooth().fit(X[tri]);X=torch.from_numpy(sc.transform(X)).to(DEVICE);target=torch.from_numpy((rv*Y_SCALE).astype(np.float32)).to(DEVICE)
  model=RealMLP(TOPN).to(DEVICE);opt=torch.optim.AdamW(model.parameters(),lr=8e-4,betas=(.9,.98),weight_decay=1e-5);ema=EMA(model);steps=math.ceil(len(tri)/BS)*EPOCHS;step=0;trit=torch.from_numpy(tri).to(DEVICE)
  for ep in range(1,EPOCHS+1):
   model.train();perm=torch.randperm(len(tri),device=DEVICE);tot=0;st=time.time()
   for i in range(0,len(tri),BS):
    progress=step/steps;lr=8e-4*(1 if progress<.6 else max((1-progress)/.4,.02));opt.param_groups[0]['lr']=lr;ii=perm[i:i+BS];loss,_,_=loss_fn(model(X[trit[ii]]),target[ii]);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();ema.update(model);tot+=float(loss)*len(ii);step+=1
   if ep in (4,6,9,11,12):print(' epoch',ep,'loss',tot/len(tri),'sec',round(time.time()-st),flush=True)
  em=RealMLP(TOPN).to(DEVICE);ema.apply_to(em);q=predict(em,X,vai)/Y_SCALE;ids=sid[va];base=backbone(fold,ids);yt=y[va];mv=mo[va];true_res,_=project_residual(yt,base);print('beta',beta,'q',fold_stats(yt,q,mv),'corr_base',float(unit(q)@unit(base)),'partial',cosine(true_res,q),flush=True);print('base',fold_stats(yt,base,mv),flush=True)
  for w in [.02,.05,.10,.15]:print(' residual_weight',w,fold_stats(yt,(1-w)*unit(base)+w*unit(q),mv),flush=True)
  saved.update({f'{fold}_sample_id':ids,f'{fold}_target':yt,f'{fold}_month':mv,f'{fold}_base':base,f'{fold}_residual':q});torch.save({'model':em.state_dict(),'cols':cols,'med':sc.med,'fac':sc.fac,'inner':inner,'train_end':train_end},f'output/residual_realmlp_{fold}_ema.pt');del X,model,em,bm,selector;torch.cuda.empty_cache();gc.collect()
 np.savez('output/residual_realmlp_rolling_oof.npz',**saved);print('total_sec',time.time()-t0)
if __name__=='__main__':main()
