"""Evaluate fixed RealMLP checkpoint recipes against micro-LGB on middle and late folds."""
import numpy as np,pandas as pd,torch
import train_realmlp_proxy_v4 as rv4
from train_realmlp_proxy_v3 import load_combined,RobustSmooth,unit,cosine,Y_SCALE,DEVICE

def infer(prefix,ep,Xall,va):
 z=torch.load(f'output/{prefix}_ep{ep}.pt',map_location='cpu');raw=Xall[z['cols']].replace([np.inf,-np.inf],np.nan).to_numpy(np.float64);sc=RobustSmooth();sc.med=z['med'];sc.fac=z['fac'];xt=torch.from_numpy(sc.transform(raw[va])).to(DEVICE);m=rv4.RealMLP(len(z['cols']),z['n_ens']).to(DEVICE);m.load_state_dict(z['model']);pm=rv4.predict(m,xt,member=True)/Y_SCALE;del xt,m;torch.cuda.empty_cache();return np.mean([unit(pm[:,i]) for i in range(pm.shape[1])],0)
def stats(y,p,mo):
 a=[cosine(y[mo==m],p[mo==m]) for m in np.unique(mo)];return cosine(y,p),np.mean(a),np.min(a),np.std(a)
def main():
 lab,X=load_combined();mo=lab.month.to_numpy();y=lab.target.to_numpy();zl=np.load('output/rolling_micro_lgb_preds.npz');rows=[];save={}
 for fold,lo,hi,prefix in [('middle',51,61,'realmlp_v4_middle'),('late',62,71,'realmlp_v4_late')]:
  va=np.flatnonzero((mo>=lo)&(mo<hi));yy=y[va];mm=mo[va];pl=zl[f'{fold}_combined'];eps={e:infer(prefix,e,X,va) for e in (6,9,11)};recipes={'ep6':eps[6],'ep9':eps[9],'ep11':eps[11],'ens6_9_11':np.mean([unit(eps[e]) for e in (6,9,11)],0),'ens9_11':np.mean([unit(eps[e]) for e in (9,11)],0)}
  save[f'{fold}_sample_id']=lab.sample_id.to_numpy()[va];save[f'{fold}_target']=yy;save[f'{fold}_lgb']=pl
  print('\n',fold,'lgb',stats(yy,pl,mm),flush=True)
  for name,p in recipes.items():
   save[f'{fold}_{name}']=p;single=stats(yy,p,mm);best=(-9,None,None)
   for w in np.arange(.1,.61,.05):
    q=(1-w)*unit(pl)+w*unit(p);s=stats(yy,q,mm)
    if s[0]>best[0]:best=(s[0],w,s)
   rows.append((fold,name,*single,best[1],*best[2]));print(name,'single',single,'best',best,flush=True)
 np.savez('output/realmlp_rolling_oof.npz',**save);pd.DataFrame(rows,columns=['fold','model','cos','month_mean','month_min','month_std','weight','blend_cos','blend_month_mean','blend_month_min','blend_month_std']).to_csv('output/realmlp_rolling_blend.csv',index=False)
if __name__=='__main__':main()
