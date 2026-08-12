"""Evaluate fixed 6/9/11 checkpoint ensembles across RealMLP seeds on proxy CV."""
import numpy as np,pandas as pd,torch
import train_realmlp_proxy_v4 as rv4
from train_realmlp_proxy_v3 import load_combined,RobustSmooth,unit,cosine,Y_SCALE,DEVICE
PREFIX={'s42':'realmlp_proxy_v4','s13':'realmlp_v4_proxy_s13','s77':'realmlp_v4_proxy_s77'}
def infer(path,Xall,va):
 z=torch.load(path,map_location='cpu');raw=Xall[z['cols']].replace([np.inf,-np.inf],np.nan).to_numpy(np.float64);sc=RobustSmooth();sc.med=z['med'];sc.fac=z['fac'];xt=torch.from_numpy(sc.transform(raw[va])).to(DEVICE);m=rv4.RealMLP(len(z['cols']),z['n_ens']).to(DEVICE);m.load_state_dict(z['model']);pm=rv4.predict(m,xt,member=True)/Y_SCALE;del xt,m;torch.cuda.empty_cache();return np.mean([unit(pm[:,i]) for i in range(pm.shape[1])],0)
def stats(y,p,mo):
 ms=[cosine(y[mo==m],p[mo==m]) for m in np.unique(mo)];return cosine(y,p),float(np.mean(ms)),float(np.min(ms)),float(np.std(ms))
def main():
 lab,X=load_combined();mo=lab.month.to_numpy();va=np.flatnonzero(mo>=45);y=lab.target.to_numpy()[va];pl=np.load('output/proxy_lgb_oof.npz')['prediction'];out={}
 for seed,prefix in PREFIX.items():
  ps=[infer(f'output/{prefix}_ep{e}.pt',X,va) for e in (6,9,11)];out[seed]=np.mean([unit(x) for x in ps],0);print(seed,stats(y,out[seed],mo[va]),flush=True)
 print('seed corr',np.corrcoef([unit(out[s]) for s in PREFIX]),flush=True)
 candidates={'s42':out['s42'],'s13':out['s13'],'s77':out['s77'],'avg42_13':np.mean([unit(out['s42']),unit(out['s13'])],0),'avg3':np.mean([unit(out[s]) for s in PREFIX],0)}
 rows=[]
 for name,p in candidates.items():
  single=stats(y,p,mo[va]);best=(-9,None,None)
  for w in np.arange(.1,.61,.05):
   q=(1-w)*unit(pl)+w*unit(p);st=stats(y,q,mo[va])
   if st[0]>best[0]:best=(st[0],w,st)
  rows.append((name,*single,best[1],*best[2]));print(name,'single',single,'blend',best,flush=True)
 np.savez('output/realmlp_multiseed_proxy_oof.npz',sample_id=lab.sample_id.to_numpy()[va],target=y,month=mo[va],lgb=pl,**candidates);pd.DataFrame(rows,columns=['model','cos','month_mean','month_min','month_std','weight','blend_cos','blend_month_mean','blend_month_min','blend_month_std']).to_csv('output/realmlp_multiseed_proxy.csv',index=False)
if __name__=='__main__':main()
