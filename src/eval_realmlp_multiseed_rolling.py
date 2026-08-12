"""Evaluate fixed checkpoint RealMLP seed ensembles on middle/late folds."""
import numpy as np,pandas as pd,torch
import train_realmlp_proxy_v4 as rv4
from train_realmlp_proxy_v3 import load_combined,RobustSmooth,unit,cosine,Y_SCALE,DEVICE
PREFIX={
 'middle':{'s42':'realmlp_v4_middle','s13':'realmlp_v4_middle_s13'},
 'late':{'s42':'realmlp_v4_late','s13':'realmlp_v4_late_s13'}}
def infer(path,Xall,va):
 z=torch.load(path,map_location='cpu');raw=Xall[z['cols']].replace([np.inf,-np.inf],np.nan).to_numpy(np.float64);sc=RobustSmooth();sc.med=z['med'];sc.fac=z['fac'];xt=torch.from_numpy(sc.transform(raw[va])).to(DEVICE);m=rv4.RealMLP(len(z['cols']),z['n_ens']).to(DEVICE);m.load_state_dict(z['model']);pm=rv4.predict(m,xt,member=True)/Y_SCALE;del xt,m;torch.cuda.empty_cache();return np.mean([unit(pm[:,i]) for i in range(pm.shape[1])],0)
def stats(y,p,mo):
 ms=[cosine(y[mo==m],p[mo==m]) for m in np.unique(mo)];return cosine(y,p),float(np.mean(ms)),float(np.min(ms)),float(np.std(ms))
def main():
 lab,X=load_combined();mo=lab.month.to_numpy();y=lab.target.to_numpy();zl=np.load('output/rolling_micro_lgb_preds.npz');rows=[];save={}
 for fold,lo,hi in [('middle',51,61),('late',62,71)]:
  va=np.flatnonzero((mo>=lo)&(mo<hi));yy=y[va];mm=mo[va];pl=zl[f'{fold}_combined'];seeds={}
  for seed,prefix in PREFIX[fold].items():
   eps=[infer(f'output/{prefix}_ep{e}.pt',X,va) for e in (6,9,11)];seeds[seed]=np.mean([unit(p) for p in eps],0)
  candidates={'s42':seeds['s42'],'s13':seeds['s13'],'avg42_13':np.mean([unit(seeds['s42']),unit(seeds['s13'])],0)}
  print('\n',fold,'seed_corr',float(unit(seeds['s42'])@unit(seeds['s13'])),'lgb',stats(yy,pl,mm),flush=True)
  for name,p in candidates.items():
   single=stats(yy,p,mm);fixed=stats(yy,.6*unit(pl)+.4*unit(p),mm);rows.append((fold,name,*single,*fixed));print(name,'single',single,'fixed_lgb60_real40',fixed,flush=True);save[f'{fold}_{name}']=p
  save[f'{fold}_sample_id']=lab.sample_id.to_numpy()[va];save[f'{fold}_target']=yy;save[f'{fold}_lgb']=pl
 np.savez('output/realmlp_multiseed_rolling_oof.npz',**save);pd.DataFrame(rows,columns=['fold','model','cos','month_mean','month_min','month_std','blend_cos','blend_month_mean','blend_month_min','blend_month_std']).to_csv('output/realmlp_multiseed_rolling.csv',index=False)
if __name__=='__main__':main()
