"""High-resolution MultiStream CNN-Transformer for MSCapital.
Grid features are adapted from the public Transformer baseline; architecture follows the
public LB0.142 multi-stream idea. Strict chronological validation, no Kaggle upload.
"""
import os,sys,time,json,random
from dataclasses import dataclass
import numpy as np,pandas as pd,torch
import torch.nn as nn
import torch.nn.functional as F

ROOT=os.environ.get('GRID_ROOT','features/grid_v2'); GRID_VERSION=os.environ.get('GRID_VERSION','v2'); DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS=int(sys.argv[1]) if len(sys.argv)>1 else 10; BATCH=int(os.environ.get('BATCH','256')); ACCUM=int(os.environ.get('ACCUM_STEPS','1'))
TRAIN_END=int(os.environ.get('TRAIN_END','62')); VALID_END=int(os.environ.get('VALID_END','71')); SEED=int(os.environ.get('SEED','42')); OUT_PREFIX=os.environ.get('OUT_PREFIX','multistream'); D_MODEL=int(os.environ.get('D_MODEL','64')); N_LAYERS=int(os.environ.get('N_LAYERS','2')); LR=float(os.environ.get('LR','0.001')); LAMBDA_COS=float(os.environ.get('LAMBDA_COS','1.0'))
RESUME=os.environ.get('RESUME_CHECKPOINT','');START_EPOCH=int(os.environ.get('START_EPOCH','0'))
M_LEN=int(os.environ.get('MARKET_LEN','200')); F_LEN=int(os.environ.get('FLOW_LEN','60')); M_CH,T_CH,O_CH=11,7,10

def paths(split):
 return {k:f'{ROOT}/{split}_{GRID_VERSION}_{k}_{L}x{C}.mmap' for k,L,C in [('market',M_LEN,M_CH),('tx',F_LEN,T_CH),('order',F_LEN,O_CH)]}
def arrays(split,n):
 p=paths(split);return {'market':np.memmap(p['market'],np.float16,'r',shape=(n,M_LEN,M_CH)),'tx':np.memmap(p['tx'],np.float16,'r',shape=(n,F_LEN,T_CH)),'order':np.memmap(p['order'],np.float16,'r',shape=(n,F_LEN,O_CH))}
def norm_stats(A,idx,nmax=50000):
 rng=np.random.default_rng(42);ii=np.sort(rng.choice(idx,min(nmax,len(idx)),replace=False));out={}
 for k,a in A.items():
  s=np.zeros(a.shape[-1]);sq=np.zeros(a.shape[-1]);n=0
  for j in range(0,len(ii),2048):
   x=np.asarray(a[ii[j:j+2048]],np.float32).reshape(-1,a.shape[-1]);s+=x.sum(0);sq+=(x*x).sum(0);n+=len(x)
  mu=np.nan_to_num(s/n,nan=0.0,posinf=0.0,neginf=0.0);sd=np.sqrt(np.maximum(np.nan_to_num(sq/n-mu*mu,nan=1.0,posinf=1.0,neginf=1.0),1e-6));out[k]=(mu.astype(np.float32),sd.astype(np.float32))
 return out
class DS(torch.utils.data.Dataset):
 def __init__(self,A,idx,norm,y=None):self.A,self.idx,self.norm,self.y=A,np.asarray(idx),norm,y
 def __len__(self):return len(self.idx)
 def one(self,k,j):
  x=np.asarray(self.A[k][j],np.float32).copy();pad=np.abs(x).sum(-1)==0;mu,sd=self.norm[k];x=np.nan_to_num(np.clip((x-mu)/sd,-8,8),nan=0.0,posinf=8.0,neginf=-8.0);x[pad]=0;return torch.from_numpy(x.T.copy())
 def __getitem__(self,i):
  j=self.idx[i];z=(self.one('market',j),self.one('tx',j),self.one('order',j))
  return z if self.y is None else (*z,torch.tensor(self.y[j],dtype=torch.float32))
class Conv(nn.Module):
 def __init__(self,a,b,k):
  super().__init__();self.n=nn.Sequential(nn.Conv1d(a,b,k,padding=k//2,bias=False),nn.BatchNorm1d(b),nn.GELU(),nn.Dropout(.1),nn.Conv1d(b,b,k,padding=k//2,bias=False),nn.BatchNorm1d(b),nn.GELU());self.s=nn.Identity() if a==b else nn.Conv1d(a,b,1,bias=False)
 def forward(self,x):return self.n(x)+self.s(x)
class Pool(nn.Module):
 def __init__(self,d):super().__init__();self.s=nn.Sequential(nn.Linear(d,d),nn.Tanh(),nn.Linear(d,1))
 def forward(self,x):w=torch.softmax(self.s(x).squeeze(-1),1);return torch.einsum('bt,btd->bd',w,x)
class Stream(nn.Module):
 def __init__(self,inc,L,d=64,layers=2):
  super().__init__();self.c=nn.Sequential(Conv(inc,d,5),Conv(d,d,3));el=nn.TransformerEncoderLayer(d,4,d*4,.1,'gelu',batch_first=True,norm_first=True);self.t=nn.TransformerEncoder(el,layers);self.pos=nn.Parameter(torch.randn(1,L,d)*.02);self.p=Pool(d);self.ep=nn.Sequential(nn.Linear(d*3,d),nn.GELU(),nn.LayerNorm(d))
 def forward(self,x):
  h=self.c(x).transpose(1,2);h=self.t(h+self.pos[:,:h.size(1)]);return self.ep(torch.cat([self.p(h),h[:,-1],h.mean(1)],-1))
class Net(nn.Module):
 def __init__(self,d=None):
  super().__init__();d=d or D_MODEL;self.m=Stream(M_CH,M_LEN,d,N_LAYERS);self.t=Stream(T_CH,F_LEN,d,max(1,N_LAYERS-1));self.o=Stream(O_CH,F_LEN,d,max(1,N_LAYERS-1));el=nn.TransformerEncoderLayer(d,4,d*4,.1,'gelu',batch_first=True,norm_first=True);self.cross=nn.TransformerEncoder(el,1);self.typ=nn.Parameter(torch.randn(1,3,d)*.02);self.h=nn.Sequential(nn.Linear(d*6,d*2),nn.GELU(),nn.Dropout(.1),nn.Linear(d*2,d),nn.GELU(),nn.Linear(d,1))
 def forward(self,m,t,o):
  z=[self.m(m),self.t(t),self.o(o)];raw=torch.cat(z,-1);mix=self.cross(torch.stack(z,1)+self.typ).flatten(1);return self.h(torch.cat([raw,mix],-1)).squeeze(-1)
def lossfn(p,y):
 p=torch.nan_to_num(p,nan=0.0,posinf=0.0,neginf=0.0);p0=p-p.mean();y0=y-y.mean();cos=1-F.cosine_similarity(p0[None],y0[None],dim=1,eps=1e-8).mean();return LAMBDA_COS*cos+(1-LAMBDA_COS)*F.smooth_l1_loss(p,y*1000.0)
def cos(y,p,center=False):
 if center:y=y-y.mean();p=p-p.mean()
 return float(np.dot(y,p)/(np.linalg.norm(y)*np.linalg.norm(p)+1e-12))
@torch.no_grad()
def pred(q,dl):
 q.eval();z=[];yy=[]
 for b in dl:
  m,t,o,*y=b;z.append(q(m.to(DEVICE),t.to(DEVICE),o.to(DEVICE)).float().cpu().numpy());
  if y:yy.append(y[0].numpy())
 return np.concatenate(z),np.concatenate(yy) if yy else None
def main():
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);lab=pd.read_feather('data/train/label.feather').sort_values('sample_id');n=len(lab);A=arrays('train',n);mo=lab.month.to_numpy();y=lab.target.to_numpy(np.float32);tr=np.flatnonzero(mo<TRAIN_END);va=np.flatnonzero((mo>=TRAIN_END)&(mo<VALID_END));norm=norm_stats(A,tr);np.savez(f'{ROOT}/norm_stats_{OUT_PREFIX}.npz',**{f'{k}_{z}':v[i] for k,v in norm.items() for i,z in enumerate(['mean','std'])})
 tl=torch.utils.data.DataLoader(DS(A,tr,norm,y),BATCH,shuffle=True,num_workers=0,pin_memory=True,drop_last=True);vl=torch.utils.data.DataLoader(DS(A,va,norm,y),BATCH*2,shuffle=False,num_workers=0,pin_memory=True);q=Net().to(DEVICE);opt=torch.optim.AdamW(q.parameters(),LR,weight_decay=1e-4);sc=torch.cuda.amp.GradScaler(enabled=DEVICE.type=='cuda');best=-1;start_epoch=START_EPOCH
 if RESUME:
  state=torch.load(RESUME,map_location=DEVICE)
  if isinstance(state,dict) and 'model' in state:
   q.load_state_dict(state['model']);opt.load_state_dict(state['optimizer']);sc.load_state_dict(state['scaler']);start_epoch=int(state.get('epoch',start_epoch))
  else:q.load_state_dict(state)
 sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
 for _ in range(start_epoch):sch.step()
 print(f'DEVICE={DEVICE}, prefix={OUT_PREFIX}, grid={GRID_VERSION} {M_LEN}/{F_LEN}, d={D_MODEL}, layers={N_LAYERS}, split=< {TRAIN_END} / [{TRAIN_END},{VALID_END}), params={sum(p.numel() for p in q.parameters())/1e6:.2f}M, micro_batch={BATCH}, accum={ACCUM}, effective_batch={BATCH*ACCUM}, train={len(tr)}, val={len(va)}, resume={RESUME or None}, start_epoch={start_epoch}',flush=True)
 for ep in range(start_epoch,EPOCHS):
  q.train();tot=nseen=0;st=time.time();opt.zero_grad(set_to_none=True);pending=0
  for bi,(m,t,o,yy) in enumerate(tl):
   m,t,o,yy=[z.to(DEVICE,non_blocking=True) for z in (m,t,o,yy)]
   with torch.cuda.amp.autocast(enabled=DEVICE.type=='cuda'):loss=lossfn(q(m,t,o),yy)
   if not torch.isfinite(loss):
    opt.zero_grad(set_to_none=True);pending=0;continue
   sc.scale(loss/ACCUM).backward();pending+=1;tot+=loss.item()*len(yy);nseen+=len(yy)
   if pending==ACCUM:
    sc.unscale_(opt);nn.utils.clip_grad_norm_(q.parameters(),1);sc.step(opt);sc.update();opt.zero_grad(set_to_none=True);pending=0
  if pending:
   # Correct the final partial accumulation to preserve gradient magnitude.
   scale=ACCUM/pending
   for param in q.parameters():
    if param.grad is not None:param.grad.mul_(scale)
   sc.unscale_(opt);nn.utils.clip_grad_norm_(q.parameters(),1);sc.step(opt);sc.update();opt.zero_grad(set_to_none=True)
  sch.step()
  if len(va):
   p,yt=pred(q,vl);raw=cos(yt,p);cen=cos(yt,p,True);metrics=f'raw={raw:.5f}, centered={cen:.5f}'
  else:
   raw=cen=float('nan');metrics='full_data_no_validation'
  print(f'epoch {ep+1}/{EPOCHS}: loss={tot/nseen:.5f}, {metrics}, sec={time.time()-st:.0f}',flush=True)
  torch.save(q.state_dict(),f'output/{OUT_PREFIX}_ep{ep+1}.pt')
  torch.save({'model':q.state_dict(),'optimizer':opt.state_dict(),'scaler':sc.state_dict(),'epoch':ep+1},f'output/{OUT_PREFIX}_train_state.pt')
  if len(va) and raw>best:best=raw;torch.save(q.state_dict(),f'output/{OUT_PREFIX}_best.pt')
 print('best_raw=',best)
if __name__=='__main__':main()
