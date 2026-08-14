"""Fast RAM-batched proxy evaluation for joint aligned-stream v3 checkpoints."""
import os,sys
os.environ.setdefault('MODEL_VARIANT','joint');os.environ.update(GRID_ROOT='features/grid_v3',GRID_VERSION='v3',MARKET_LEN='400',FLOW_LEN='120',D_MODEL='64',N_LAYERS='2',BATCH='256')
_argv=sys.argv;sys.argv=[sys.argv[0]]
import numpy as np,pandas as pd,torch
from train_multistream_grid import arrays,load_ram_arrays,GPUBatchPrep,ram_batches,Net,DEVICE,ROOT
sys.argv=_argv
PREFIX=os.environ.get('OUT_PREFIX','joint_v3_proxy_fast')
def unit(x):x=np.asarray(x,np.float64);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
def stats(y,p,mo):
 v=[float(unit(y[mo==m])@unit(p[mo==m])) for m in np.unique(mo)];return float(unit(y)@unit(p)),float(np.mean(v)),float(np.min(v)),float(np.std(v))
@torch.no_grad()
def infer(q,A,prep,va,y):
 q.eval();out=[]
 for b in ram_batches(A,va,y,512,False,42):m,t,o,yy=prep.batch(b);out.append(q(m,t,o).float().cpu().numpy())
 return np.concatenate(out)
def main():
 lab=pd.read_feather('data/train/label.feather').sort_values('sample_id');mo=lab.month.to_numpy();y=lab.target.to_numpy(np.float32);va=np.flatnonzero(mo>=45);A=load_ram_arrays(arrays('train',len(lab)));z=np.load(f'{ROOT}/norm_stats_{PREFIX}.npz');norm={k:(z[k+'_mean'],z[k+'_std']) for k in ('market','tx','order')};prep=GPUBatchPrep(norm);out={'sample_id':lab.sample_id.to_numpy()[va],'target':y[va],'month':mo[va]}
 for ep in (3,4,5,6,7):
  q=Net().to(DEVICE);q.load_state_dict(torch.load(f'output/{PREFIX}_ep{ep}.pt',map_location=DEVICE));p=infer(q,A,prep,va,y);out[f'ep{ep}']=p;print('ep',ep,stats(y[va],p,mo[va]),flush=True);del q;torch.cuda.empty_cache()
 rec={'ens4_5_6':(4,5,6),'ens3_4_5_6':(3,4,5,6),'ens5_6':(5,6)}
 base=np.load('output/multistream_v3_proxy_oof.npz');assert np.array_equal(base['sample_id'],out['sample_id']);b=base['ens4_5_6'];pl=np.load('output/proxy_lgb_oof.npz')['prediction']
 for name,eps in rec.items():
  p=np.mean([unit(out[f'ep{e}']) for e in eps],0);out[name]=p;print(name,stats(y[va],p,mo[va]),'corr_base',float(unit(p)@unit(b)),'corr_lgb',float(unit(p)@unit(pl)),flush=True)
  for w in (.2,.4,.5,.6,.8):print(' joint_weight_vs_base',w,stats(y[va],(1-w)*unit(b)+w*unit(p),mo[va]),flush=True)
 np.savez(f'output/{PREFIX}_oof.npz',**out)
if __name__=='__main__':main()
