"""Leakage-safe fast Proxy training for the public v9_big architecture. Never submits."""
import os,sys,time,queue,threading,copy
from pathlib import Path
import numpy as np,pandas as pd,torch
import torch.nn.functional as F
sys.path.insert(0,str(Path('research/lb0142').resolve()))
from lb0142.models_v9 import GridCfg,MultiStreamModel,cosine_init_scale
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu');ROOT='features/grid_v2';VER='v2';ML=200;FL=60;BS=int(os.environ.get('BATCH','256'));ACC=int(os.environ.get('ACCUM_STEPS','4'));EPOCHS=int(sys.argv[1]) if len(sys.argv)>1 else 12;PREFIX=os.environ.get('OUT_PREFIX','v9big_proxy_fast');SEED=42;TRAIN_END=int(os.environ.get('TRAIN_END','45'));VALID_END=int(os.environ.get('VALID_END','71'))
CFG=GridCfg(d_model=96,n_layers=3,cnn_channels=96,market_len=ML,flow_len=FL,batch_size=BS,num_workers=0)
torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True

def arrays(n):
 def mm(name,shape):return np.memmap(f'{ROOT}/train_{VER}_{name}.mmap',np.float16,'r',shape=shape)
 return {'market':mm(f'market_{ML}x11',(n,ML,11)),'market_count':mm(f'market_count_{ML}',(n,ML)),'tx':mm(f'tx_{FL}x7',(n,FL,7)),'tx_count':mm(f'tx_count_{FL}',(n,FL)),'order':mm(f'order_{FL}x10',(n,FL,10))}
def load_ram(A):
 t=time.time();out={k:np.array(v,dtype=np.float16,copy=True,order='C') for k,v in A.items()};print('RAM',sum(x.nbytes for x in out.values())/2**30,'GB',time.time()-t,'sec',flush=True);return out
def norm_stats(A,idx,nmax=50000):
 rng=np.random.default_rng(42);ii=np.sort(rng.choice(idx,min(nmax,len(idx)),replace=False));out={}
 for stream,ks in [('market',['market','market_count']),('tx',['tx','tx_count']),('order',['order'])]:
  ss=sq=None;n=0
  for j in range(0,len(ii),1024):
   z=[np.asarray(A[k][ii[j:j+1024]],np.float32) for k in ks];z=[x[...,None] if x.ndim==2 else x for x in z];x=np.concatenate(z,-1).reshape(-1,sum(a.shape[-1] for a in z));
   if ss is None:ss=np.zeros(x.shape[1]);sq=np.zeros(x.shape[1])
   ss+=x.sum(0);sq+=(x*x).sum(0);n+=len(x)
  mu=np.nan_to_num(ss/n);sd=np.sqrt(np.maximum(np.nan_to_num(sq/n-mu*mu,nan=1),1e-6));out[stream]=(mu.astype('f4'),sd.astype('f4'))
 return out
class Prep:
 def __init__(self,norm):self.norm={k:(torch.tensor(a,device=DEVICE),torch.tensor(b,device=DEVICE)) for k,(a,b) in norm.items()}
 def one(self,k,*xs):
  z=torch.cat([torch.from_numpy(x).to(DEVICE)[...,None] if x.ndim==2 else torch.from_numpy(x).to(DEVICE) for x in xs],-1).float();mu,sd=self.norm[k];return torch.nan_to_num(torch.clamp((z-mu)/sd,-8,8),nan=0.,posinf=8.,neginf=-8.).transpose(1,2)
 def batch(self,b):return self.one('market',b[0],b[1]),self.one('tx',b[2],b[3]),self.one('order',b[4]),torch.from_numpy(b[5]).to(DEVICE)
def batches(A,idx,y,bs,shuffle,seed,maxq=3):
 idx=np.asarray(idx).copy();
 if shuffle:np.random.default_rng(seed).shuffle(idx)
 q=queue.Queue(maxq);stop=object()
 def work():
  try:
   for i in range(0,len(idx),bs):
    j=idx[i:i+bs];q.put((A['market'][j],A['market_count'][j],A['tx'][j],A['tx_count'][j],A['order'][j],y[j]))
  finally:q.put(stop)
 threading.Thread(target=work,daemon=True).start()
 while True:
  z=q.get()
  if z is stop:break
  yield z
def cosine(y,p):
 y=np.asarray(y,float);p=np.asarray(p,float);y-=y.mean();p-=p.mean();return float(y@p/(np.linalg.norm(y)*np.linalg.norm(p)+1e-12))
@torch.no_grad()
def predict(model,A,idx,y,prep):
 model.eval();po=[];yo=[]
 for b in batches(A,idx,y,BS*2,False,SEED):
  m,t,o,yy=prep.batch(b)
  with torch.cuda.amp.autocast(enabled=DEVICE.type=='cuda'):p=model(m,t,o)
  po.append(p.float().cpu().numpy());yo.append(yy.cpu().numpy())
 return np.concatenate(po),np.concatenate(yo)
def main():
 np.random.seed(SEED);torch.manual_seed(SEED);lab=pd.read_feather('data/train/label.feather').sort_values('sample_id').reset_index(drop=True);mo=lab.month.to_numpy();y=lab.target.to_numpy('f4');tr=np.flatnonzero(mo<TRAIN_END);va=np.flatnonzero((mo>=TRAIN_END)&(mo<VALID_END));A=arrays(len(lab));norm=norm_stats(A,tr);np.savez(f'output/{PREFIX}_norm.npz',**{f'{k}_{s}':v[i] for k,v in norm.items() for i,s in enumerate(['mean','std'])});A=load_ram(A);prep=Prep(norm);model=MultiStreamModel(CFG).to(DEVICE);cosine_init_scale(model)
 try:opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4,fused=True)
 except TypeError:opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
 sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS);scaler=torch.cuda.amp.GradScaler(enabled=DEVICE.type=='cuda');print('model',sum(p.numel() for p in model.parameters())/1e6,'M train',len(tr),'val',len(va),'batch',BS,'accum',ACC,'effective',BS*ACC,flush=True)
 for ep in range(1,EPOCHS+1):
  model.train();opt.zero_grad(set_to_none=True);pending=0;tot=seen=0;t=time.time()
  for b in batches(A,tr,y,BS,True,SEED+ep):
   m,tx,o,yy=prep.batch(b)
   with torch.cuda.amp.autocast(enabled=DEVICE.type=='cuda'):
    p=model(m,tx,o);loss=1-F.cosine_similarity((p-p.mean())[None],(yy-yy.mean())[None],dim=1,eps=1e-8).mean()
   scaler.scale(loss/ACC).backward();pending+=1;tot+=float(loss)*len(yy);seen+=len(yy)
   if pending==ACC:
    scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1);scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True);pending=0
  if pending:
   for p in model.parameters():
    if p.grad is not None:p.grad.mul_(ACC/pending)
   scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1);scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
  sched.step();pv,yv=predict(model,A,va,y,prep);score=cosine(yv,pv);print(f'epoch {ep}/{EPOCHS} loss={tot/seen:.5f} proxy={score:.6f} sec={time.time()-t:.0f}',flush=True);torch.save(model.state_dict(),f'output/{PREFIX}_ep{ep}.pt');np.savez(f'output/{PREFIX}_ep{ep}_oof.npz',sample_id=lab.sample_id.to_numpy()[va],target=yv,month=mo[va],prediction=pv)
if __name__=='__main__':main()
