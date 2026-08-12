"""Select robust RealMLP checkpoints by proxy score, month floor and LGB marginal gain."""
import numpy as np,pandas as pd,torch
import train_realmlp_proxy_v4 as rv4
from train_realmlp_proxy_v3 import load_combined,RobustSmooth,unit,cosine,Y_SCALE,DEVICE

def infer(path,Xall,va):
 z=torch.load(path,map_location='cpu');X=Xall[z['cols']].replace([np.inf,-np.inf],np.nan).to_numpy(np.float64);sc=RobustSmooth();sc.med=z['med'];sc.fac=z['fac'];xt=torch.from_numpy(sc.transform(X[va])).to(DEVICE);m=rv4.RealMLP(len(z['cols']),z['n_ens']).to(DEVICE);m.load_state_dict(z['model']);p=rv4.predict(m,xt,member=True)/Y_SCALE;del xt,m;torch.cuda.empty_cache();return np.mean([unit(p[:,i]) for i in range(p.shape[1])],axis=0)
def stats(y,p,mo):
 ms=[cosine(y[mo==m],p[mo==m]) for m in np.unique(mo)];return cosine(y,p),float(np.mean(ms)),float(np.min(ms)),float(np.std(ms))
def main():
 lab,X=load_combined();mo=lab.month.to_numpy();va=np.flatnonzero(mo>=45);y=lab.target.to_numpy()[va];pl=np.load('output/proxy_lgb_oof.npz')['prediction'];z3=np.load('output/realmlp_proxy_early_oof.npz');pred={'v3ep1':z3['ep1'],'v3ep2':z3['ep2'],'v3ens12':z3['ensemble12']}
 for ep in (5,7,10,11,16):pred[f'v4ep{ep}']=infer(f'output/realmlp_proxy_v4_ep{ep}.pt',X,va)
 candidates=dict(pred)
 candidates['v4_7_10_11']=np.mean([unit(pred[f'v4ep{x}']) for x in (7,10,11)],0)
 candidates['cross_v3v4']=np.mean([unit(pred['v3ep1']),unit(pred['v3ep2']),unit(pred['v4ep10'])],0)
 rows=[]
 for name,p in candidates.items():
  base=stats(y,p,mo[va]);best=(-9,None,None)
  for w in np.arange(.05,.61,.05):
   q=(1-w)*unit(pl)+w*unit(p);s=stats(y,q,mo[va]);
   if s[0]>best[0]:best=(s[0],w,s)
  rows.append((name,*base,best[1],*best[2]));print(name,'single',base,'best_lgb_blend',best,flush=True)
 pd.DataFrame(rows,columns=['model','cos','month_mean','month_min','month_std','weight','blend_cos','blend_month_mean','blend_month_min','blend_month_std']).to_csv('output/realmlp_proxy_selection.csv',index=False)
 np.savez('output/realmlp_proxy_selected_oof.npz',sample_id=lab.sample_id.to_numpy()[va],target=y,month=mo[va],lgb=pl,**candidates)
if __name__=='__main__':main()
