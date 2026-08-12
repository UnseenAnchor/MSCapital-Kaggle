"""Evaluate v3 effective-batch-1024 checkpoints on leakage-safe proxy CV."""
import os
os.environ.update(GRID_ROOT='features/grid_v3',GRID_VERSION='v3',MARKET_LEN='400',FLOW_LEN='120',D_MODEL='64',N_LAYERS='2',BATCH='128')
import numpy as np,pandas as pd,torch
from train_multistream_grid import arrays,DS,Net,DEVICE,ROOT,pred,cos
PREFIX='multistream_v3_proxy_eff1024'
def unit(x):x=np.asarray(x,np.float64);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
def stats(y,p,mo):
 vals=[float(unit(y[mo==m])@unit(p[mo==m])) for m in np.unique(mo)];return float(unit(y)@unit(p)),float(np.mean(vals)),float(np.min(vals)),float(np.std(vals))
def main():
 lab=pd.read_feather('data/train/label.feather').sort_values('sample_id');mo=lab.month.to_numpy();y=lab.target.to_numpy(np.float32);va=np.flatnonzero(mo>=45);A=arrays('train',len(lab));z=np.load(f'{ROOT}/norm_stats_{PREFIX}.npz');norm={k:(z[k+'_mean'],z[k+'_std']) for k in ('market','tx','order')};dl=torch.utils.data.DataLoader(DS(A,va,norm,y),256,shuffle=False,num_workers=0,pin_memory=True);out={'sample_id':lab.sample_id.to_numpy()[va],'target':y[va],'month':mo[va]};ps=[]
 for ep in (3,4,5,6,7,8):
  q=Net().to(DEVICE);q.load_state_dict(torch.load(f'output/{PREFIX}_ep{ep}.pt',map_location=DEVICE));p,t=pred(q,dl);out[f'ep{ep}']=p;ps.append(p);print('ep',ep,stats(t,p,mo[va]),flush=True);del q;torch.cuda.empty_cache()
 recipes={'ens4_5':np.mean([unit(out[f'ep{e}']) for e in (4,5)],0),'ens4_5_6':np.mean([unit(out[f'ep{e}']) for e in (4,5,6)],0),'ens3_4_5_6':np.mean([unit(out[f'ep{e}']) for e in (3,4,5,6)],0)}
 for k,p in recipes.items():out[k]=p;print(k,stats(y[va],p,mo[va]),flush=True)
 pl=np.load('output/proxy_lgb_oof.npz')['prediction'];zr=np.load('output/realmlp_multiseed_proxy_oof.npz');real=zr['avg42_13'];print('corr lgb/real/v3',np.corrcoef([unit(pl),unit(real),unit(out['ens4_5_6'])]),flush=True)
 for name,p in recipes.items():
  best=(-9,None,None)
  for w in np.arange(.1,.91,.05):
   q=(1-w)*unit(pl)+w*unit(p);s=stats(y[va],q,mo[va])
   if s[0]>best[0]:best=(s[0],w,s)
  print('lgb +',name,best,flush=True)
 np.savez('output/multistream_v3_proxy_oof.npz',**out)
if __name__=='__main__':main()
