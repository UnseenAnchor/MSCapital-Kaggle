"""Fast fold evaluation for aligned joint-stream v3 against base v3."""
import os,sys,argparse
_cli=sys.argv[1:];sys.argv=[sys.argv[0]]
os.environ.update(MODEL_VARIANT='joint',GRID_ROOT='features/grid_v3',GRID_VERSION='v3',MARKET_LEN='400',FLOW_LEN='120',D_MODEL='64',N_LAYERS='2',BATCH='256')
import numpy as np,pandas as pd,torch
from train_multistream_grid import arrays,load_ram_arrays,GPUBatchPrep,ram_batches,Net,DEVICE,ROOT
sys.argv=[sys.argv[0],*_cli]
def unit(x):x=np.asarray(x,np.float64);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
def stats(y,p,mo):
 v=[float(unit(y[mo==m])@unit(p[mo==m])) for m in np.unique(mo)];return float(unit(y)@unit(p)),float(np.mean(v)),float(np.min(v)),float(np.std(v))
@torch.no_grad()
def infer(q,A,prep,va,y):
 q.eval();out=[]
 for b in ram_batches(A,va,y,512,False,42):m,t,o,yy=prep.batch(b);out.append(q(m,t,o).float().cpu().numpy())
 return np.concatenate(out)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fold',choices=['middle','late'],required=True);ap.add_argument('--prefix',required=True);a=ap.parse_args();lo,hi=(51,61) if a.fold=='middle' else (62,71)
 lab=pd.read_feather('data/train/label.feather').sort_values('sample_id');mo=lab.month.to_numpy();y=lab.target.to_numpy(np.float32);va=np.flatnonzero((mo>=lo)&(mo<hi));A=load_ram_arrays(arrays('train',len(lab)));z=np.load(f'{ROOT}/norm_stats_{a.prefix}.npz');norm={k:(z[k+'_mean'],z[k+'_std']) for k in ('market','tx','order')};prep=GPUBatchPrep(norm);out={'sample_id':lab.sample_id.to_numpy()[va],'target':y[va],'month':mo[va]}
 for ep in (3,4,5,6,7):
  q=Net().to(DEVICE);q.load_state_dict(torch.load(f'output/{a.prefix}_ep{ep}.pt',map_location=DEVICE));p=infer(q,A,prep,va,y);out[f'ep{ep}']=p;print('ep',ep,stats(y[va],p,mo[va]),flush=True);del q;torch.cuda.empty_cache()
 joint=np.mean([unit(out[f'ep{e}']) for e in (4,5,6)],0);out['ens4_5_6']=joint;basez=np.load(f'output/multistream_v3_{a.fold}_eff1024_oof.npz');assert np.array_equal(basez['sample_id'],out['sample_id']);base=basez['ens4_5_6'];print('base',stats(y[va],base,mo[va]),'joint',stats(y[va],joint,mo[va]),'corr',float(unit(base)@unit(joint)),flush=True)
 for w in (.2,.4,.5,.6,.8):print('joint_weight',w,stats(y[va],(1-w)*unit(base)+w*unit(joint),mo[va]),flush=True)
 out['base']=base;out['mix40_60']=.4*unit(base)+.6*unit(joint);np.savez(f'output/{a.prefix}_oof.npz',**out)
if __name__=='__main__':main()
