"""Raw event-order residual Transformer with nested chronological targets. Never submits."""
import gc,time,queue,threading,random,numpy as np,pandas as pd,torch
import torch.nn as nn
import torch.nn.functional as F
from residual_realmlp_rolling import backbone,project_residual,fold_stats
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu');BS=1024;EPOCHS=12;SEED=42;D=64
ROOT='features/cache';TROOT='features/event_cache';torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True

def unit(x):x=np.asarray(x,np.float64);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
def cosine(y,p):return float(unit(y)@unit(p))
def load_arrays(split):
 t=time.time();A={'tx':np.array(np.load(f'{ROOT}/{split}_tx.npy',mmap_mode='r'),copy=True),'order':np.array(np.load(f'{ROOT}/{split}_order.npy',mmap_mode='r'),copy=True),'tx_time':np.array(np.load(f'{TROOT}/{split}_transaction_time.npy',mmap_mode='r'),copy=True),'order_time':np.array(np.load(f'{TROOT}/{split}_order_time.npy',mmap_mode='r'),copy=True)};print(split,'RAM GB',sum(x.nbytes for x in A.values())/2**30,'sec',time.time()-t,flush=True);return A
def fit_stats(A,idx,nmax=50000):
 ii=np.sort(np.random.default_rng(42).choice(idx,min(nmax,len(idx)),replace=False));out={}
 for k in ('tx','order'):
  x=np.asarray(A[k][ii],np.float32);mask=np.asarray(A[k+'_time'][ii,:,2]>0);z=x[mask];out[k]=(z.mean(0).astype('f4'),np.maximum(z.std(0),1e-3).astype('f4'))
 return out
def batches(A,idx,y,bs,shuffle,seed,maxq=3):
 idx=np.asarray(idx).copy()
 if shuffle:np.random.default_rng(seed).shuffle(idx)
 q=queue.Queue(maxq);stop=object()
 def work():
  try:
   for i in range(0,len(idx),bs):
    j=idx[i:i+bs];q.put((A['tx'][j],A['order'][j],A['tx_time'][j],A['order_time'][j],y[j]))
  finally:q.put(stop)
 threading.Thread(target=work,daemon=True).start()
 while True:
  z=q.get()
  if z is stop:break
  yield z
class Prep:
 def __init__(self,stats):self.stats={k:(torch.tensor(a,device=DEVICE),torch.tensor(b,device=DEVICE)) for k,(a,b) in stats.items()}
 def one(self,k,v,t):
  v=torch.from_numpy(v).to(DEVICE).float();t=torch.from_numpy(t).to(DEVICE).float();mu,sd=self.stats[k];v=torch.clamp((v-mu)/sd,-8,8);mask=t[:,:,2:3];dv=F.pad(v[:,1:,:2]-v[:,:-1,:2],(0,0,1,0));inter=(v[:,:,1:2]*v[:,:,2:3]) if k=='tx' else (v[:,:,2:3]*v[:,:,3:4]);x=torch.cat([v,t,dv,inter],-1)*mask;return x.transpose(1,2),mask.squeeze(-1).bool()
 def batch(self,b):
  tx,tm=self.one('tx',b[0],b[2]);o,om=self.one('order',b[1],b[3]);y=torch.from_numpy(b[4]).to(DEVICE);return tx,tm,o,om,y
class Conv(nn.Module):
 def __init__(self,a,b,k):super().__init__();self.n=nn.Sequential(nn.Conv1d(a,b,k,padding=k//2,bias=False),nn.BatchNorm1d(b),nn.GELU(),nn.Dropout(.1),nn.Conv1d(b,b,k,padding=k//2,bias=False),nn.BatchNorm1d(b),nn.GELU());self.s=nn.Conv1d(a,b,1,bias=False)
 def forward(self,x):return self.n(x)+self.s(x)
class EventEncoder(nn.Module):
 def __init__(self,c):
  super().__init__();self.conv=nn.Sequential(Conv(c,D,3),Conv(D,D,3));el=nn.TransformerEncoderLayer(D,4,D*4,.1,'gelu',batch_first=True,norm_first=True);self.tr=nn.TransformerEncoder(el,2);self.pos=nn.Parameter(torch.randn(1,32,D)*.02);self.score=nn.Sequential(nn.Linear(D,D),nn.Tanh(),nn.Linear(D,1));self.out=nn.Sequential(nn.Linear(D*3,D),nn.GELU(),nn.LayerNorm(D))
 def forward(self,x,mask):
  h=self.conv(x).transpose(1,2)+self.pos;h=self.tr(h,src_key_padding_mask=~mask);s=self.score(h).squeeze(-1).masked_fill(~mask,-1e4);pool=torch.einsum('bt,btd->bd',torch.softmax(s,1),h);mean=(h*mask[:,:,None]).sum(1)/mask.sum(1,keepdim=True).clamp_min(1);recent=h[torch.arange(len(h),device=h.device),mask.float().argmax(1)];return self.out(torch.cat([pool,mean,recent],-1))
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.tx=EventEncoder(10);self.o=EventEncoder(11);el=nn.TransformerEncoderLayer(D,4,D*4,.1,'gelu',batch_first=True,norm_first=True);self.cross=nn.TransformerEncoder(el,1);self.typ=nn.Parameter(torch.randn(1,2,D)*.02);self.head=nn.Sequential(nn.Linear(D*4,D*2),nn.GELU(),nn.Dropout(.1),nn.Linear(D*2,D),nn.GELU(),nn.Linear(D,1))
 def forward(self,tx,tm,o,om):
  a=self.tx(tx,tm);b=self.o(o,om);raw=torch.cat([a,b],-1);mix=self.cross(torch.stack([a,b],1)+self.typ).flatten(1);return self.head(torch.cat([raw,mix],-1)).squeeze(-1)
def loss_fn(p,y):
 p0=p-p.mean();y0=y-y.mean();cos=1-F.cosine_similarity(p0[None],y0[None],dim=1,eps=1e-8).mean();return .8*cos+.2*F.smooth_l1_loss(p,y*1000.)
@torch.no_grad()
def infer(model,A,idx,prep):
 model.eval();out=[];dummy=np.zeros(len(A['tx']),np.float32)
 for b in batches(A,idx,dummy,BS*2,False,SEED):
  tx,tm,o,om,_=prep.batch(b)
  with torch.cuda.amp.autocast(enabled=DEVICE.type=='cuda'):p=model(tx,tm,o,om)
  out.append(p.float().cpu().numpy())
 return np.concatenate(out)
def main():
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);lab=pd.read_feather('data/train/label.feather').sort_values('sample_id').reset_index(drop=True);sid=lab.sample_id.to_numpy();mo=lab.month.to_numpy();y=lab.target.to_numpy(np.float64);pos=pd.Series(np.arange(len(lab)),index=sid);rz=np.load('output/residual_lgb_rolling_oof.npz');A=load_arrays('train');saved={};tall=time.time()
 for fold,train_end in [('proxy',45),('middle',51),('late',62)]:
  tri=pos.loc[rz[f'{fold}_train_sample_id']].to_numpy();vai=pos.loc[rz[f'{fold}_sample_id']].to_numpy();yt=np.zeros(len(lab),np.float32);yt[tri]=rz[f'{fold}_train_residual_target'].astype(np.float32);prep=Prep(fit_stats(A,np.flatnonzero(mo<train_end)));model=Net().to(DEVICE)
  try:opt=torch.optim.AdamW(model.parameters(),lr=6e-4,weight_decay=1e-4,fused=True)
  except TypeError:opt=torch.optim.AdamW(model.parameters(),lr=6e-4,weight_decay=1e-4)
  sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS);scaler=torch.cuda.amp.GradScaler(enabled=DEVICE.type=='cuda');preds={};print('\n',fold,'train',len(tri),'valid',len(vai),'params',sum(p.numel() for p in model.parameters())/1e6,flush=True)
  for ep in range(1,EPOCHS+1):
   model.train();tot=seen=0;st=time.time()
   for b in batches(A,tri,yt,BS,True,SEED+ep):
    tx,tm,o,om,yy=prep.batch(b)
    with torch.cuda.amp.autocast(enabled=DEVICE.type=='cuda'):loss=loss_fn(model(tx,tm,o,om),yy)
    opt.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1);scaler.step(opt);scaler.update();tot+=float(loss)*len(yy);seen+=len(yy)
   sched.step()
   if ep in (6,9,12):preds[ep]=infer(model,A,vai,prep);torch.save(model.state_dict(),f'output/event_residual_{fold}_ep{ep}.pt')
   print(' epoch',ep,'loss',tot/seen,'sec',round(time.time()-st),flush=True)
  q=np.mean([unit(preds[e]) for e in (6,9,12)],axis=0);ids=sid[vai];base=backbone(fold,ids);yv=y[vai];mv=mo[vai];true_res,_=project_residual(yv,base);print('ensemble',fold_stats(yv,q,mv),'corr_base',float(unit(q)@unit(base)),'partial',cosine(true_res,q),flush=True);print('base',fold_stats(yv,base,mv),flush=True)
  for w in [.02,.05,.10,.15,.20]:print(' residual_weight',w,fold_stats(yv,(1-w)*unit(base)+w*unit(q),mv),flush=True)
  saved.update({f'{fold}_sample_id':ids,f'{fold}_target':yv,f'{fold}_month':mv,f'{fold}_base':base,f'{fold}_residual':q});del model;torch.cuda.empty_cache();gc.collect()
 np.savez('output/event_residual_rolling_oof.npz',**saved);print('total_sec',time.time()-tall)
if __name__=='__main__':main()
