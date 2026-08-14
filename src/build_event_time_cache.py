"""Add event-time, inter-arrival and real-event masks aligned to features/cache sequences."""
from pathlib import Path
import time,numpy as np,pyarrow.feather as pf
from numba import njit,prange
ROOT=Path('features/event_cache');ROOT.mkdir(parents=True,exist_ok=True);L=32
@njit(parallel=True)
def build_time(sec,offsets,counts):
 n=len(counts);out=np.zeros((n,L,4),np.float32)
 for i in prange(n):
  st=offsets[i];T=counts[i]
  if T<=0:continue
  for j in range(L):
   active=1
   if T>=L:
    k=(j*(T-1))//(L-1)
   else:
    pad=L-T
    if j<pad:k=0;active=0
    else:k=j-pad
   src=st+T-1-k # reverse raw old->new rows: cached orientation is recent->old
   s=sec[src]
   out[i,j,0]=s/60.0
   out[i,j,2]=active
   out[i,j,3]=k/max(T-1,1)
  for j in range(L-1):
   if out[i,j,2]>0 and out[i,j+1,2]>0:out[i,j,1]=np.log1p(abs(float(out[i,j+1,0]-out[i,j,0]))*60.0)/np.log(61.0)
 return out
def one(split,name):
 t=time.time();ids=np.load(f'features/cache/{split}_ids.npy');tab=pf.read_table(f'data/{split}/{name}.feather',columns=['sample_id','seconds_before_predict'],memory_map=True);sid=tab['sample_id'].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64,copy=False);sec=tab['seconds_before_predict'].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float32,copy=False);assert np.array_equal(ids,np.arange(len(ids))) and sid.min()==0 and sid.max()==len(ids)-1;counts=np.bincount(sid,minlength=len(ids)).astype(np.int64);offsets=np.empty(len(ids),np.int64);offsets[0]=0;np.cumsum(counts[:-1],out=offsets[1:]);assert offsets[-1]+counts[-1]==len(sid);out=build_time(sec,offsets,counts).astype(np.float16);dest=ROOT/f'{split}_{name}_time.npy';np.save(dest,out);print(split,name,'rows',len(sid),'shape',out.shape,'GB',out.nbytes/2**30,'sec',time.time()-t,'count q',np.quantile(counts,[0,.1,.5,.9,1]),flush=True)
def main():
 for split in ('train','test'):
  for name in ('transaction','order'):one(split,name)
if __name__=='__main__':main()
