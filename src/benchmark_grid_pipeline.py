"""Benchmark legacy per-sample mmap loader against vectorized RAM/GPU preprocessing."""
import os,sys,time,queue,threading
os.environ.update(GRID_ROOT='features/grid_v3',GRID_VERSION='v3',MARKET_LEN='400',FLOW_LEN='120',D_MODEL='64',N_LAYERS='2')
_argv=sys.argv;sys.argv=[sys.argv[0]]
import numpy as np,pandas as pd,torch
from train_multistream_grid import arrays,norm_stats,DS,Net,DEVICE,lossfn
sys.argv=_argv
STEPS=int(os.environ.get('BENCH_STEPS','200'))

def load_ram(A):
 out={};t=time.time()
 for k,a in A.items():
  s=time.time();out[k]=np.array(a,dtype=np.float16,copy=True,order='C');print('RAM',k,out[k].shape,'GB',out[k].nbytes/2**30,'sec',time.time()-s,flush=True)
 print('RAM total sec',time.time()-t,flush=True);return out
class GPUPrep:
 def __init__(self,norm):self.norm={k:(torch.as_tensor(v[0],device=DEVICE),torch.as_tensor(v[1],device=DEVICE)) for k,v in norm.items()}
 def one(self,k,x):
  z=torch.from_numpy(x).to(DEVICE,non_blocking=False);pad=z.abs().sum(-1)==0;mu,sd=self.norm[k];z=torch.nan_to_num(torch.clamp((z.float()-mu)/sd,-8,8),nan=0.,posinf=8.,neginf=-8.);z[pad]=0;return z.transpose(1,2)
 def batch(self,b):return self.one('market',b[0]),self.one('tx',b[1]),self.one('order',b[2]),torch.from_numpy(b[3]).to(DEVICE)
def prefetch_batches(A,idx,y,bs,maxq=3):
 q=queue.Queue(maxq);stop=object()
 def work():
  try:
   for i in range(0,len(idx)-bs+1,bs):
    j=idx[i:i+bs];q.put((A['market'][j],A['tx'][j],A['order'][j],y[j]))
  finally:q.put(stop)
 threading.Thread(target=work,daemon=True).start()
 while True:
  z=q.get()
  if z is stop:break
  yield z
def run_legacy(A,idx,norm,y,bs):
 dl=torch.utils.data.DataLoader(DS(A,idx,norm,y),bs,shuffle=False,num_workers=0,pin_memory=True,drop_last=True);m=Net().to(DEVICE);opt=torch.optim.AdamW(m.parameters(),1e-4);torch.cuda.synchronize();t=time.time();n=0
 for a,b,c,d in dl:
  if n>=STEPS:break
  opt.zero_grad(set_to_none=True);loss=lossfn(m(a.to(DEVICE),b.to(DEVICE),c.to(DEVICE)),d.to(DEVICE));loss.backward();opt.step();n+=1
 torch.cuda.synchronize();return n/(time.time()-t),time.time()-t
def run_fast(A,idx,norm,y,bs):
 prep=GPUPrep(norm);m=Net().to(DEVICE);opt=torch.optim.AdamW(m.parameters(),1e-4);torch.cuda.synchronize();t=time.time();n=0
 for raw in prefetch_batches(A,idx,y,bs):
  if n>=STEPS:break
  a,b,c,d=prep.batch(raw);opt.zero_grad(set_to_none=True)
  with torch.cuda.amp.autocast():loss=lossfn(m(a,b,c),d)
  loss.backward();opt.step();n+=1
 torch.cuda.synchronize();return n/(time.time()-t),time.time()-t
def main():
 lab=pd.read_feather('data/train/label.feather').sort_values('sample_id');y=lab.target.to_numpy(np.float32);idx=np.flatnonzero(lab.month.to_numpy()<45);A=arrays('train',len(lab));norm=norm_stats(A,idx);small=idx[:max(STEPS*128+128,30000)];a=run_legacy(A,small,norm,y,128);print('LEGACY bs128 batches/s sec',a,flush=True);R=load_ram(A)
 for bs in (128,256,384,512):
  try:
   z=run_fast(R,idx[:max(STEPS*bs+bs,60000)],norm,y,bs);print('FAST bs',bs,'batches/s sec samples/s',z,z[0]*bs,flush=True)
  except torch.cuda.OutOfMemoryError:
   print('FAST bs',bs,'OOM',flush=True);torch.cuda.empty_cache()
if __name__=='__main__':main()
