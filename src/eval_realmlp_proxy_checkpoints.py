"""Evaluate early RealMLP proxy checkpoints and marginal value versus proxy LGB."""
import numpy as np,pandas as pd,torch
from train_realmlp_proxy_v3 import load_combined,RealMLP,RobustSmooth,DEVICE,predict,unit,cosine,Y_SCALE

def main():
 lab,Xall=load_combined();mo=lab.month.to_numpy();va=np.flatnonzero(mo>=45);y=lab.target.to_numpy()[va];ps=[]
 for ep in (1,2,3,4):
  z=torch.load(f'output/realmlp_proxy_v3_ep{ep}.pt',map_location='cpu');X=Xall[z['cols']].replace([np.inf,-np.inf],np.nan).to_numpy(np.float64);sc=RobustSmooth();sc.med=z['med'];sc.fac=z['fac'];X=torch.from_numpy(sc.transform(X[va])).to(DEVICE);m=RealMLP(len(z['cols']),z['n_ens']).to(DEVICE);m.load_state_dict(z['model']);members=predict(m,X,member=True)/Y_SCALE;p=np.mean([unit(members[:,i]) for i in range(members.shape[1])],0);ps.append(p);print('ep',ep,'mean',cosine(y,p),'members',[round(cosine(y,members[:,i]),5) for i in range(members.shape[1])],flush=True);del X,m;torch.cuda.empty_cache()
 pl=np.load('output/proxy_lgb_oof.npz')['prediction'];ens12=(unit(ps[0])+unit(ps[1]))/2;print('corr lgb/ep1/ep2',np.corrcoef([unit(pl),unit(ps[0]),unit(ps[1])]),flush=True)
 print('checkpoint12',cosine(y,ens12))
 for w in np.arange(0,0.51,.05):print('realmlp_weight',round(w,2),'score',cosine(y,(1-w)*unit(pl)+w*unit(ens12)))
 np.savez('output/realmlp_proxy_early_oof.npz',sample_id=lab.sample_id.to_numpy()[va],target=y,month=mo[va],ep1=ps[0],ep2=ps[1],ensemble12=ens12)
if __name__=='__main__':main()
