"""Cross-scale delta v2+v3 self-anchor model. Proxy gate first; never submits."""
import os,time,queue,threading,random,numpy as np,pandas as pd,torch
import torch.nn as nn
import torch.nn.functional as F
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu');BS=128;ACC=8;EPOCHS=12;SEED=42;D=64;TRAIN_END=int(os.environ.get('TRAIN_END','45'));VALID_END=int(os.environ.get('VALID_END','71'));PREFIX=os.environ.get('OUT_PREFIX','multires_self_proxy');FULL_TRAIN=os.environ.get('FULL_TRAIN','0')=='1'
ROOT2='features/grid_v2';ROOT3='features/grid_v3';torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True

def unit(x):x=np.asarray(x,float);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
def cosine(y,p):return float(unit(y)@unit(p))
def paths(root,ver,ml,fl):return {'market':f'{root}/train_{ver}_market_{ml}x11.mmap','tx':f'{root}/train_{ver}_tx_{fl}x7.mmap','order':f'{root}/train_{ver}_order_{fl}x10.mmap'}
def load_arrays(root,ver,ml,fl,n):
 p=paths(root,ver,ml,fl);return {'market':np.array(np.memmap(p['market'],np.float16,'r',shape=(n,ml,11)),copy=True),'tx':np.array(np.memmap(p['tx'],np.float16,'r',shape=(n,fl,7)),copy=True),'order':np.array(np.memmap(p['order'],np.float16,'r',shape=(n,fl,10)),copy=True)}
def stats(A,idx):
 ii=np.sort(np.random.default_rng(42).choice(idx,min(50000,len(idx)),replace=False));out={}
 for k,a in A.items():
  x=A[k][ii].astype(np.float32).reshape(-1,a.shape[-1]);mu=x.mean(0);sd=np.maximum(x.std(0),1e-6);out[k]=(mu.astype('f4'),sd.astype('f4'))
 return out
def batches(A2,A3,idx,y,bs,shuffle,seed,maxq=3):
 ix=np.asarray(idx).copy()
 if shuffle:np.random.default_rng(seed).shuffle(ix)
 q=queue.Queue(maxq);stop=object()
 def work():
  try:
   for i in range(0,len(ix),bs):
    j=ix[i:i+bs];q.put((A2['market'][j],A2['tx'][j],A2['order'][j],A3['market'][j],A3['tx'][j],A3['order'][j],A3['market'][j,::2]-A2['market'][j],A3['tx'][j,::2]-A2['tx'][j],A3['order'][j,::2]-A2['order'][j],y[j]))
  finally:q.put(stop)
 threading.Thread(target=work,daemon=True).start()
 while True:
  z=q.get()
  if z is stop:break
  yield z
class Prep:
 def __init__(self,s2,s3):self.s=[(torch.tensor(a,device=DEVICE),torch.tensor(b,device=DEVICE)) for a,b in [s2['market'],s2['tx'],s2['order'],s3['market'],s3['tx'],s3['order'],s3['market_delta'],s3['tx_delta'],s3['order_delta']]]
 def one(self,x,i):
  z=torch.from_numpy(x).to(DEVICE).float();mu,sd=self.s[i];z=torch.nan_to_num(torch.clamp((z-mu)/sd,-8,8),nan=0.,posinf=8.,neginf=-8.);return z.transpose(1,2)
 def batch(self,b):return tuple(self.one(x,i) for i,x in enumerate(b[:9]))+(torch.from_numpy(b[9]).to(DEVICE),)
class Conv(nn.Module):
 def __init__(self,a,b,k):super().__init__();self.n=nn.Sequential(nn.Conv1d(a,b,k,padding=k//2,bias=False),nn.BatchNorm1d(b),nn.GELU(),nn.Dropout(.1),nn.Conv1d(b,b,k,padding=k//2,bias=False),nn.BatchNorm1d(b),nn.GELU());self.s=nn.Conv1d(a,b,1,bias=False)
 def forward(self,x):return self.n(x)+self.s(x)
class Pool(nn.Module):
 def __init__(self,d):super().__init__();self.s=nn.Sequential(nn.Linear(d,d),nn.Tanh(),nn.Linear(d,1))
 def forward(self,x):w=torch.softmax(self.s(x).squeeze(-1),1);return torch.einsum('bt,btd->bd',w,x)
class Stream(nn.Module):
 def __init__(self,inc,L):
  super().__init__();self.c=nn.Sequential(Conv(inc,D,5),Conv(D,D,3));el=nn.TransformerEncoderLayer(D,4,D*4,.1,'gelu',batch_first=True,norm_first=True);self.t=nn.TransformerEncoder(el,2);self.pos=nn.Parameter(torch.randn(1,L,D)*.02);self.p=Pool(D);self.ep=nn.Sequential(nn.Linear(D*3,D),nn.GELU(),nn.LayerNorm(D))
 def forward(self,x):
  h=self.c(x).transpose(1,2);h=self.t(h+self.pos[:,:h.size(1)]);return self.ep(torch.cat([self.p(h),h[:,-1],h.mean(1)],-1))
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.e=nn.ModuleList([Stream(11,200),Stream(7,60),Stream(10,60),Stream(11,400),Stream(7,120),Stream(10,120),Stream(11,200),Stream(7,60),Stream(10,60)]);el=nn.TransformerEncoderLayer(D,4,D*4,.1,'gelu',batch_first=True,norm_first=True);self.cross=nn.TransformerEncoder(el,2);self.typ=nn.Parameter(torch.randn(1,9,D)*.02);self.h=nn.Sequential(nn.Linear(D*18,D*4),nn.GELU(),nn.Dropout(.1),nn.Linear(D*4,D*2),nn.GELU(),nn.Linear(D*2,1))
 def forward(self,*x):
  z=[m(a) for m,a in zip(self.e,x)];raw=torch.cat(z,-1);mix=self.cross(torch.stack(z,1)+self.typ).flatten(1);return self.h(torch.cat([raw,mix],-1)).squeeze(-1)
def lossfn(p,y):
 p=torch.nan_to_num(p);return .7*(1-F.cosine_similarity((p-p.mean())[None],(y-y.mean())[None],dim=1,eps=1e-8).mean())+.3*F.smooth_l1_loss(p,y*1000.)
@torch.no_grad()
def infer(m,A2,A3,idx,y,prep):
 m.eval();out=[]
 for b in batches(A2,A3,idx,y,BS*2,False,SEED):
  z=prep.batch(b);out.append(m(*z[:-1]).float().cpu().numpy())
 return np.concatenate(out)
def main():
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);lab=pd.read_feather('data/train/label.feather').sort_values('sample_id').reset_index(drop=True);mo=lab.month.to_numpy();y=lab.target.to_numpy(np.float32);n=len(lab);tr=np.arange(n) if FULL_TRAIN else np.flatnonzero(mo<TRAIN_END);va=np.array([],dtype=np.int64) if FULL_TRAIN else np.flatnonzero((mo>=TRAIN_END)&(mo<VALID_END));print('loading grids',flush=True);A2=load_arrays(ROOT2,'v2',200,60,n);A3=load_arrays(ROOT3,'v3',400,120,n);s2=stats(A2,tr);s3=stats(A3,tr)
 for k in ('market','tx','order'):
  x=(A3[k][tr,::2].astype(np.float32)-A2[k][tr].astype(np.float32)).reshape(-1,A2[k].shape[-1]);s3[k+'_delta']=(x.mean(0).astype('f4'),np.maximum(x.std(0),1e-6).astype('f4'))
 prep=Prep(s2,s3);m=Net().to(DEVICE);print('PREFIX',PREFIX,'split',TRAIN_END,VALID_END,'RAM GB',(sum(a.nbytes for a in A2.values())+sum(a.nbytes for a in A3.values()))/2**30,'params',sum(p.numel() for p in m.parameters())/1e6,flush=True)
 try:opt=torch.optim.AdamW(m.parameters(),lr=6e-4,weight_decay=1e-4,fused=True)
 except TypeError:opt=torch.optim.AdamW(m.parameters(),lr=6e-4,weight_decay=1e-4)
 sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS);sc=torch.cuda.amp.GradScaler(enabled=DEVICE.type=='cuda');pred={}
 for ep in range(1,EPOCHS+1):
  m.train();opt.zero_grad(set_to_none=True);pending=0;tot=seen=0;st=time.time()
  for b in batches(A2,A3,tr,y,BS,True,SEED+ep):
   z=prep.batch(b)
   with torch.cuda.amp.autocast(enabled=DEVICE.type=='cuda'):loss=lossfn(m(*z[:-1]),z[-1])
   sc.scale(loss/ACC).backward();pending+=1;tot+=float(loss)*len(z[-1]);seen+=len(z[-1])
   if pending==ACC:
    sc.unscale_(opt);nn.utils.clip_grad_norm_(m.parameters(),1);sc.step(opt);sc.update();opt.zero_grad(set_to_none=True);pending=0
  if pending:
   for p in m.parameters():
    if p.grad is not None:p.grad.mul_(ACC/pending)
   sc.unscale_(opt);nn.utils.clip_grad_norm_(m.parameters(),1);sc.step(opt);sc.update();opt.zero_grad(set_to_none=True)
  sch.step()
  if ep in (4,5,6,7,8):
   torch.save(m.state_dict(),f'output/{PREFIX}_ep{ep}.pt')
   if len(va):pred[ep]=infer(m,A2,A3,va,y,prep);print('epoch',ep,'cos',cosine(y[va],pred[ep]),'sec',round(time.time()-st),flush=True)
   else:print('epoch',ep,'loss',tot/seen,'full_data_no_validation sec',round(time.time()-st),flush=True)
  else:print('epoch',ep,'loss',tot/seen,'sec',round(time.time()-st),flush=True)
 if len(va):
  for es in [(4,5,6),(5,6,7),(6,7,8),(4,5,6,7,8)]:
   p=np.mean([unit(pred[e]) for e in es],axis=0);vals=[cosine(y[va][mo[va]==q],p[mo[va]==q]) for q in np.unique(mo[va])];print('ens',es,cosine(y[va],p),np.mean(vals),min(vals),np.std(vals),flush=True)
 np.savez(f'output/{PREFIX}_oof.npz',sample_id=lab.sample_id.to_numpy()[va],target=y[va],month=mo[va],**{f'ep{e}':pred[e] for e in pred})
if __name__=='__main__':main()
