"""修复时间方向的 GRU：现有 cache 是 recent->oldest，本脚本翻转为 oldest->recent。
使用 SmoothL1 + batch cosine，避免 MSE 被极端收益主导。不会覆盖旧 checkpoint。
"""
import sys,time,numpy as np,pandas as pd,torch
import torch.nn as nn
import torch.nn.functional as F
from train_gru_cached import FusionNet,CACHE,DEVICE,BATCH,SCALE_Y
EPOCHS=int(sys.argv[1]) if len(sys.argv)>1 else 12

class ChronoDS(torch.utils.data.Dataset):
 def __init__(self,m,o,x,y,idx):self.m,self.o,self.x,self.y,self.idx=m,o,x,y,idx
 def __len__(self):return len(self.idx)
 def __getitem__(self,i):
  j=self.idx[i]
  # cache was built with arr[::-1], flip back so final recurrent state is closest to prediction.
  return (torch.from_numpy(np.ascontiguousarray(self.m[j,::-1])),torch.from_numpy(np.ascontiguousarray(self.o[j,::-1])),torch.from_numpy(np.ascontiguousarray(self.x[j,::-1])),torch.tensor(self.y[j],dtype=torch.float32))

def loss_fn(p,y):
 p0=p-p.mean();y0=y-y.mean();cos=1-F.cosine_similarity(p0[None],y0[None],dim=1,eps=1e-8).mean()
 return .35*F.smooth_l1_loss(p,y)+.65*cos

def cosine(y,p):
 return float(np.dot(y,p)/(np.linalg.norm(y)*np.linalg.norm(p)+1e-12))
def main():
 ids=np.load(CACHE+'/train_ids.npy');m=np.load(CACHE+'/train_market.npy',mmap_mode='r');o=np.load(CACHE+'/train_order.npy',mmap_mode='r');x=np.load(CACHE+'/train_tx.npy',mmap_mode='r');lab=pd.read_feather('data/train/label.feather').set_index('sample_id');mon=lab.loc[ids].month.to_numpy();y=lab.loc[ids].target.to_numpy(np.float32)*SCALE_Y;tr=np.flatnonzero(mon<62);va=np.flatnonzero(mon>=62)
 tl=torch.utils.data.DataLoader(ChronoDS(m,o,x,y,tr),batch_size=BATCH,shuffle=True,num_workers=0,pin_memory=True);vl=torch.utils.data.DataLoader(ChronoDS(m,o,x,y,va),batch_size=BATCH*2,shuffle=False,num_workers=0,pin_memory=True)
 q=FusionNet().to(DEVICE);opt=torch.optim.AdamW(q.parameters(),lr=1.5e-3,weight_decay=2e-5);sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS);best=-1
 print(f'DEVICE={DEVICE}, train={len(tr)}, val={len(va)}, chronological=True',flush=True)
 for ep in range(EPOCHS):
  q.train();tot=n=0;st=time.time()
  for bm,bo,bx,by in tl:
   bm,bo,bx,by=[z.to(DEVICE,non_blocking=True) for z in (bm,bo,bx,by)];opt.zero_grad(set_to_none=True);loss=loss_fn(q(bm,bo,bx),by);loss.backward();torch.nn.utils.clip_grad_norm_(q.parameters(),1);opt.step();tot+=loss.item()*len(by);n+=len(by)
  sch.step();q.eval();pp=[];yy=[]
  with torch.no_grad():
   for bm,bo,bx,by in vl:pp.append(q(bm.to(DEVICE),bo.to(DEVICE),bx.to(DEVICE)).cpu().numpy());yy.append(by.numpy())
  p=np.concatenate(pp);t=np.concatenate(yy);raw=cosine(t,p);center=cosine(t-t.mean(),p-p.mean());print(f'epoch {ep+1}/{EPOCHS}: loss={tot/n:.5f}, raw={raw:.5f}, centered={center:.5f}, sec={time.time()-st:.0f}',flush=True)
  torch.save(q.state_dict(),f'output/gru_chrono_ep{ep+1}.pt')
  if raw>best:best=raw;torch.save(q.state_dict(),'output/gru_chrono_best.pt')
 print('best_raw=',best)
if __name__=='__main__':main()
