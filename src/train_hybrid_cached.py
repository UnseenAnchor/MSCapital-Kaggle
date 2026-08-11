"""Hybrid：缓存的三路 GRU 序列 + 93 个微观结构特征，GPU 时序验证。"""
import os,sys,time,numpy as np,pandas as pd,torch
import torch.nn as nn
from train_gru_cached import CACHE,DEVICE,BATCH,SCALE_Y
EPOCHS=int(sys.argv[1]) if len(sys.argv)>1 else 8

class DS(torch.utils.data.Dataset):
 def __init__(self,m,o,x,t,y,idx):self.m,self.o,self.x,self.t,self.y,self.idx=m,o,x,t,y,idx
 def __len__(self):return len(self.idx)
 def __getitem__(self,i):
  j=self.idx[i];return torch.from_numpy(self.m[j]),torch.from_numpy(self.o[j]),torch.from_numpy(self.x[j]),torch.from_numpy(self.t[j]),torch.tensor(self.y[j],dtype=torch.float32)
class E(nn.Module):
 def __init__(self,n,h):
  super().__init__();self.g=nn.GRU(n,h,2,batch_first=True,dropout=.1);self.h=nn.Sequential(nn.Linear(h,h),nn.ReLU())
 def forward(self,x):_,z=self.g(x);return self.h(z[-1])
class Net(nn.Module):
 def __init__(self,n_tab):
  super().__init__();self.m=E(11,96);self.o=E(4,64);self.x=E(3,64);self.tab=nn.Sequential(nn.Linear(n_tab,128),nn.LayerNorm(128),nn.GELU(),nn.Dropout(.15),nn.Linear(128,64),nn.GELU());self.g=nn.Linear(224,3);self.head=nn.Sequential(nn.Linear(288,160),nn.GELU(),nn.Dropout(.2),nn.Linear(160,64),nn.GELU(),nn.Linear(64,1))
 def forward(self,m,o,x,t):
  a,b,c=self.m(m),self.o(o),self.x(x);h=torch.cat([a,b,c],-1);q=torch.softmax(self.g(h),-1);seq=torch.cat([q[:,0:1]*a,q[:,1:2]*b,q[:,2:3]*c],-1);return self.head(torch.cat([seq,self.tab(t)],-1)).squeeze(-1)
def cos(y,p):y=y-y.mean();p=p-p.mean();return float(np.dot(y,p)/(np.linalg.norm(y)*np.linalg.norm(p)+1e-12))
def load_tab(ids):
 lab=pd.read_feather('data/train/label.feather');df=lab[['sample_id','month']]
 for n in ['market','order','transaction']:
  for v in ['','_v2']:
   p=f'features/{n}_train{v}.parquet'
   if os.path.exists(p):df=df.merge(pd.read_parquet(p),on='sample_id',how='left')
 cols=[c for c in df if c not in ('sample_id','month','target')]
 a=df.set_index('sample_id').reindex(ids)[cols].replace([np.inf,-np.inf],np.nan).to_numpy(np.float32)
 return a,cols,lab.set_index('sample_id').loc[ids,'month'].to_numpy(),lab.set_index('sample_id').loc[ids,'target'].to_numpy(np.float32)
def main():
 t0=time.time();ids=np.load(CACHE+'/train_ids.npy');m=np.load(CACHE+'/train_market.npy',mmap_mode='r');o=np.load(CACHE+'/train_order.npy',mmap_mode='r');x=np.load(CACHE+'/train_tx.npy',mmap_mode='r');tab,cols,mon,y=load_tab(ids)
 tr=np.flatnonzero(mon<62);va=np.flatnonzero(mon>=62);mu=np.nanmean(tab[tr],0);sd=np.nanstd(tab[tr],0);sd=np.maximum(sd,1e-6);tab=np.where(np.isnan(tab),mu,tab);tab=np.clip((tab-mu)/sd,-10,10).astype(np.float32)
 trl=torch.utils.data.DataLoader(DS(m,o,x,tab,y*SCALE_Y,tr),batch_size=BATCH,shuffle=True,num_workers=0,pin_memory=True);vl=torch.utils.data.DataLoader(DS(m,o,x,tab,y*SCALE_Y,va),batch_size=BATCH*2,shuffle=False,num_workers=0,pin_memory=True)
 net=Net(len(cols)).to(DEVICE);opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=2e-5);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS);loss=nn.MSELoss();best=-1
 print(f'DEVICE={DEVICE}, tab={len(cols)}, params={sum(p.numel() for p in net.parameters())/1e6:.2f}M, train={len(tr)}, val={len(va)}',flush=True)
 for ep in range(EPOCHS):
  net.train();tot=n=0;st=time.time()
  for bm,bo,bx,bt,by in trl:
   bm,bo,bx,bt,by=[z.to(DEVICE,non_blocking=True) for z in (bm,bo,bx,bt,by)];opt.zero_grad(set_to_none=True);z=loss(net(bm,bo,bx,bt),by);z.backward();torch.nn.utils.clip_grad_norm_(net.parameters(),1);opt.step();tot+=z.item()*len(by);n+=len(by)
  sch.step();net.eval();pp=[];yy=[]
  with torch.no_grad():
   for bm,bo,bx,bt,by in vl:pp.append((net(bm.to(DEVICE),bo.to(DEVICE),bx.to(DEVICE),bt.to(DEVICE)).cpu().numpy()/SCALE_Y));yy.append(by.numpy()/SCALE_Y)
  s=cos(np.concatenate(yy),np.concatenate(pp));print(f'epoch {ep+1}/{EPOCHS}: loss={tot/n:.5f}, val={s:.5f}, sec={time.time()-st:.0f}',flush=True);torch.save(net.state_dict(),f'output/hybrid_ep{ep+1}.pt')
  if s>best:best=s;torch.save(net.state_dict(),'output/hybrid_best.pt')
 print('best=',best,'total=',time.time()-t0)
if __name__=='__main__':main()
