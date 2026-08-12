"""Evaluate a v3 effective-batch checkpoint ensemble on a chronological fold."""
import os
os.environ.update(GRID_ROOT='features/grid_v3',GRID_VERSION='v3',MARKET_LEN='400',FLOW_LEN='120',D_MODEL='64',N_LAYERS='2',BATCH='128')
import argparse,sys,numpy as np,pandas as pd,torch
_cli=sys.argv[1:];sys.argv=[sys.argv[0]]
from train_multistream_grid import arrays,DS,Net,DEVICE,ROOT,pred
sys.argv=[sys.argv[0],*_cli]

def unit(x):x=np.asarray(x,np.float64);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
def stats(y,p,mo):
 vals=[float(unit(y[mo==m])@unit(p[mo==m])) for m in np.unique(mo)];return float(unit(y)@unit(p)),float(np.mean(vals)),float(np.min(vals)),float(np.std(vals))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fold',choices=['middle','late'],required=True);ap.add_argument('--prefix',required=True);a=ap.parse_args();lo,hi=(51,61) if a.fold=='middle' else (62,71)
 lab=pd.read_feather('data/train/label.feather').sort_values('sample_id');mo=lab.month.to_numpy();y=lab.target.to_numpy(np.float32);va=np.flatnonzero((mo>=lo)&(mo<hi));A=arrays('train',len(lab));z=np.load(f'{ROOT}/norm_stats_{a.prefix}.npz');norm={k:(z[k+'_mean'],z[k+'_std']) for k in ('market','tx','order')};dl=torch.utils.data.DataLoader(DS(A,va,norm,y),256,shuffle=False,num_workers=0,pin_memory=True);out={'sample_id':lab.sample_id.to_numpy()[va],'target':y[va],'month':mo[va]}
 for ep in (3,4,5,6,7,8):
  q=Net().to(DEVICE);q.load_state_dict(torch.load(f'output/{a.prefix}_ep{ep}.pt',map_location=DEVICE));p,t=pred(q,dl);out[f'ep{ep}']=p;print('ep',ep,stats(t,p,mo[va]),flush=True);del q;torch.cuda.empty_cache()
 recipes={'ens4_5_6':(4,5,6),'ens3_4_5_6':(3,4,5,6),'ens5_6_8':(5,6,8)};zl=np.load('output/rolling_micro_lgb_preds.npz');pl=zl[f'{a.fold}_combined']
 for name,eps in recipes.items():
  p=np.mean([unit(out[f'ep{e}']) for e in eps],0);out[name]=p;print(name,stats(y[va],p,mo[va]),'corr_lgb',float(unit(pl)@unit(p)),flush=True)
  for w in (.4,.5,.6,.7):print(' lgb/v3',round(1-w,1),w,stats(y[va],(1-w)*unit(pl)+w*unit(p),mo[va]),flush=True)
 np.savez(f'output/{a.prefix}_oof.npz',**out)
if __name__=='__main__':main()
