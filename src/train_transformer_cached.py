"""缓存序列上的轻量 Transformer 三路编码，验证超过融合基线后再提交。"""
import os, sys, time
import numpy as np, pandas as pd, torch
import torch.nn as nn
from train_gru_cached import CacheDataset, CACHE, DEVICE, BATCH, SCALE_Y

EPOCHS=int(sys.argv[1]) if len(sys.argv)>1 else 8
D=64

class Enc(nn.Module):
    def __init__(self, inp):
        super().__init__(); self.proj=nn.Linear(inp,D)
        layer=nn.TransformerEncoderLayer(D,4,128,.1,batch_first=True,norm_first=True)
        self.enc=nn.TransformerEncoder(layer,2)
        self.norm=nn.LayerNorm(D)
        self.pos=nn.Parameter(torch.zeros(1,64,D))
        nn.init.normal_(self.pos,std=.02)
    def forward(self,x):
        z=self.proj(x)+self.pos[:,:x.size(1)]
        valid=x.abs().sum(-1)>1e-7
        z=self.enc(z,src_key_padding_mask=~valid)
        w=valid.float().unsqueeze(-1)
        return self.norm((z*w).sum(1)/w.sum(1).clamp_min(1))

class Net(nn.Module):
    def __init__(self):
        super().__init__(); self.m=Enc(11); self.o=Enc(4); self.x=Enc(3)
        self.g=nn.Linear(192,3)
        self.head=nn.Sequential(nn.Linear(192,128),nn.GELU(),nn.Dropout(.2),nn.Linear(128,64),nn.GELU(),nn.Linear(64,1))
    def forward(self,m,o,x):
        a,b,c=self.m(m),self.o(o),self.x(x); h=torch.cat([a,b,c],-1); q=torch.softmax(self.g(h),-1)
        return self.head(torch.cat([q[:,0:1]*a,q[:,1:2]*b,q[:,2:3]*c],-1)).squeeze(-1)

def cos(y,p):
    y=y-y.mean(); p=p-p.mean(); return float(np.dot(y,p)/(np.linalg.norm(y)*np.linalg.norm(p)+1e-12))

def main():
    t0=time.time(); C=CACHE
    ids=np.load(C+'/train_ids.npy'); m=np.load(C+'/train_market.npy',mmap_mode='r'); o=np.load(C+'/train_order.npy',mmap_mode='r'); x=np.load(C+'/train_tx.npy',mmap_mode='r')
    lab=pd.read_feather('data/train/label.feather').set_index('sample_id'); mon=lab.loc[ids,'month'].to_numpy(); y=lab.loc[ids,'target'].to_numpy(np.float32)*SCALE_Y
    tr=np.flatnonzero(mon<62); va=np.flatnonzero(mon>=62)
    trl=torch.utils.data.DataLoader(CacheDataset(m,o,x,y,tr),batch_size=BATCH,shuffle=True,num_workers=0,pin_memory=True)
    val=torch.utils.data.DataLoader(CacheDataset(m,o,x,y,va),batch_size=BATCH*2,shuffle=False,num_workers=0,pin_memory=True)
    net=Net().to(DEVICE); opt=torch.optim.AdamW(net.parameters(),lr=1e-3,weight_decay=2e-5); sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS); lossfn=nn.MSELoss(); best=-1
    print(f'DEVICE={DEVICE}, params={sum(p.numel() for p in net.parameters())/1e6:.2f}M, train={len(tr)}, val={len(va)}',flush=True)
    for ep in range(EPOCHS):
        net.train(); total=n=0; st=time.time()
        for bm,bo,bx,by in trl:
            bm,bo,bx,by=bm.to(DEVICE,non_blocking=True),bo.to(DEVICE,non_blocking=True),bx.to(DEVICE,non_blocking=True),by.to(DEVICE,non_blocking=True)
            opt.zero_grad(set_to_none=True); loss=lossfn(net(bm,bo,bx),by); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(),1); opt.step(); total+=loss.item()*len(by); n+=len(by)
        sch.step(); net.eval(); pp=[]; yy=[]
        with torch.no_grad():
            for bm,bo,bx,by in val:
                pp.append((net(bm.to(DEVICE),bo.to(DEVICE),bx.to(DEVICE)).cpu().numpy()/SCALE_Y)); yy.append(by.numpy()/SCALE_Y)
        score=cos(np.concatenate(yy),np.concatenate(pp)); print(f'epoch {ep+1}/{EPOCHS}: loss={total/n:.5f}, val={score:.5f}, sec={time.time()-st:.0f}',flush=True)
        torch.save(net.state_dict(),f'output/transformer_ep{ep+1}.pt')
        if score>best:best=score;torch.save(net.state_dict(),'output/transformer_best.pt')
    print('best=',best,'total_sec=',time.time()-t0)
if __name__=='__main__':main()
