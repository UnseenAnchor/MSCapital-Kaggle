import os
os.environ.update(GRID_ROOT='features/grid_v3',GRID_VERSION='v3',MARKET_LEN='400',FLOW_LEN='120',D_MODEL='64',N_LAYERS='2',BATCH='128')
PREFIX=os.environ.get('PREFIX','v3_cosine_proxy');TRAIN_END=int(os.environ.get('TRAIN_END','45'));VALID_END=int(os.environ.get('VALID_END','71'));OUT=os.environ.get('OUT','output/v3_cosine_proxy_oof.npz')
import numpy as np,pandas as pd,torch
from train_multistream_grid import arrays,DS,Net,DEVICE,pred,ROOT

def u(x):x=np.asarray(x,float);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
def c(y,p):return float(u(y)@u(p))
lab=pd.read_feather('data/train/label.feather').sort_values('sample_id');mo=lab.month.to_numpy();y=lab.target.to_numpy(np.float32);va=np.flatnonzero((mo>=TRAIN_END)&(mo<VALID_END));A=arrays('train',len(lab));z=np.load(f'{ROOT}/norm_stats_{PREFIX}.npz');norm={k:(z[k+'_mean'],z[k+'_std']) for k in ('market','tx','order')};dl=torch.utils.data.DataLoader(DS(A,va,norm,y),256,shuffle=False,num_workers=0,pin_memory=True);out={}
for ep in (3,4,5,6,7,8):
 q=Net().to(DEVICE);q.load_state_dict(torch.load(f'output/{PREFIX}_ep{ep}.pt',map_location=DEVICE));p,t=pred(q,dl);out[ep]=p;vals=[c(y[va][mo[va]==m],p[mo[va]==m]) for m in np.unique(mo[va])];print('ep',ep,c(t,p),np.mean(vals),min(vals),flush=True);del q;torch.cuda.empty_cache()
for es in [(4,5,6),(5,6,7),(4,5,6,7,8)]:
 p=np.mean([u(out[e]) for e in es],0);vals=[c(y[va][mo[va]==m],p[mo[va]==m]) for m in np.unique(mo[va])];print('ens',es,c(y[va],p),np.mean(vals),min(vals),flush=True)
np.savez(OUT,sample_id=lab.sample_id.to_numpy()[va],target=y[va],month=mo[va],**{f'ep{e}':p for e,p in out.items()})
