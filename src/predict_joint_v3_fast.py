"""RAM-batched inference for aligned joint-stream v3 checkpoint ensemble. Never submits."""
import os,sys
os.environ.update(MODEL_VARIANT='joint',GRID_ROOT='features/grid_v3',GRID_VERSION='v3',MARKET_LEN='400',FLOW_LEN='120',D_MODEL='64',N_LAYERS='2',BATCH='256')
_argv=sys.argv;sys.argv=[sys.argv[0]]
import numpy as np,pandas as pd,torch
from train_multistream_grid import arrays,load_ram_arrays,GPUBatchPrep,ram_batches,Net,DEVICE,ROOT
sys.argv=_argv
PREFIX=os.environ.get('OUT_PREFIX','joint_v3_full_fast');OUT=os.environ.get('OUT','output/submission_joint_v3_fast_unit.csv')
def unit(x):x=np.asarray(x,np.float64);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
@torch.no_grad()
def infer(q,A,prep,idx,y):
 q.eval();out=[]
 for b in ram_batches(A,idx,y,512,False,42):m,t,o,yy=prep.batch(b);out.append(q(m,t,o).float().cpu().numpy())
 return np.concatenate(out)
def main():
 ids=pd.read_csv('data/submission.csv').sample_id.to_numpy();y=np.zeros(len(ids),np.float32);idx=np.arange(len(ids));A=load_ram_arrays(arrays('test',len(ids)));z=np.load(f'{ROOT}/norm_stats_{PREFIX}.npz');norm={k:(z[k+'_mean'],z[k+'_std']) for k in ('market','tx','order')};prep=GPUBatchPrep(norm);ps=[]
 for ep in (4,5,6):
  q=Net().to(DEVICE);q.load_state_dict(torch.load(f'output/{PREFIX}_ep{ep}.pt',map_location=DEVICE));p=infer(q,A,prep,idx,y);ps.append(p);pd.DataFrame({'sample_id':ids,'prediction':p}).to_csv(f'output/submission_{PREFIX}_ep{ep}.csv',index=False);print('ep',ep,p.mean(),p.std(),flush=True);del q;torch.cuda.empty_cache()
 p=np.mean([unit(x) for x in ps],0);pd.DataFrame({'sample_id':ids,'prediction':p}).to_csv(OUT,index=False);print('saved',OUT,len(p),p.mean(),p.std())
if __name__=='__main__':main()
