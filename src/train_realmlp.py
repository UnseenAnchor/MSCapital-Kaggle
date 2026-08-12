"""RealMLP batch ensemble for engineered microstructure features.
Inspired by the public LB0.142 pack, adapted to strict chronological validation.
No Kaggle upload is performed.
"""
import os,sys,time,copy,math,random
import numpy as np,pandas as pd,torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE='cuda' if torch.cuda.is_available() else 'cpu';EPOCHS=int(sys.argv[1]) if len(sys.argv)>1 else 10;BS=512

def load_data():
 lab=pd.read_feather('data/train/label.feather');ds=[lab]
 for n in ['market','order','transaction']:
  for v in ['','_v2']:
   p=f'features/{n}_train{v}.parquet'
   if os.path.exists(p):ds.append(pd.read_parquet(p))
 d=ds[0]
 for x in ds[1:]:d=d.merge(x,on='sample_id',how='left')
 cols=[c for c in d if c not in ('sample_id','month','target')]
 X=d[cols].replace([np.inf,-np.inf],np.nan).to_numpy(np.float64);y=d.target.to_numpy(np.float32);mo=d.month.to_numpy();return X,y,mo,cols
class RobustSmooth:
 def fit(self,X):
  self.med=np.nanmedian(X,0);q=np.nanquantile(X,.75,axis=0)-np.nanquantile(X,.25,axis=0);rg=np.nanmax(X,0)-np.nanmin(X,0);q=np.where(q==0,.5*rg,q);self.fac=np.where(q==0,0,1/(q+1e-30));return self
 def transform(self,X):
  X=np.where(np.isfinite(X),X,self.med);z=self.fac*(X-self.med);return (z/np.sqrt(1+(z/3)**2)).astype(np.float32)
class Scaling(nn.Module):
 def __init__(self,n,f):super().__init__();self.s=nn.Parameter(torch.ones(n,f))
 def forward(self,x):return x*self.s[None]
class PBLD(nn.Module):
 def __init__(self,n,f,h=16,out=3):
  super().__init__();self.w1=nn.Parameter(torch.randn(n,f,h));self.b1=nn.Parameter(torch.empty(n,f,h));self.w2=nn.Parameter(torch.randn(n,f,h,out-1)/np.sqrt(h));self.b2=nn.Parameter(torch.randn(n,f,out-1));nn.init.uniform_(self.b1,-np.pi,np.pi)
 def forward(self,x):
  e=x[...,None];p=torch.cos(2*np.pi*(e*self.w1[None]+self.b1[None]));z=F.gelu(torch.einsum('bnfh,nfhd->bnfd',p,self.w2)+self.b2[None]);return torch.cat([x[...,None],z],-1).flatten(2)
class NLinear(nn.Module):
 def __init__(self,n,i,o):super().__init__();self.i=i;self.w=nn.Parameter(torch.randn(n,i,o));self.b=nn.Parameter(torch.zeros(n,o))
 def forward(self,x):return torch.einsum('bni,nio->bno',x,self.w)/np.sqrt(self.i)+self.b
class RealMLP(nn.Module):
 def __init__(self,f,n=8):
  super().__init__();self.n=n;self.emb=PBLD(n,f);dim=f*3;self.scale=Scaling(n,dim);self.l1=NLinear(n,dim,256);self.l2=NLinear(n,256,256);self.l3=NLinear(n,256,64);self.head=NLinear(n,64,1);mask=torch.ones(n,dim,dtype=torch.bool)
  for i in range(n):mask[i,i::max(1,n//2)]=False
  self.register_buffer('mask',mask)
 def forward(self,x):
  x=x[:,None].expand(-1,self.n,-1);z=self.scale(self.emb(x)*self.mask[None]);z=F.gelu(self.l1(z));z=F.dropout(z,.01,self.training);z=F.gelu(self.l2(z));z=F.dropout(z,.01,self.training);z=F.gelu(self.l3(z));return self.head(z).squeeze(-1)
class EMA:
 def __init__(self,m,d=.998):self.d=d;self.s={k:v.detach().clone() for k,v in m.state_dict().items() if v.is_floating_point()}
 @torch.no_grad()
 def update(self,m):
  for k,v in m.state_dict().items():
   if k in self.s:self.s[k].mul_(self.d).add_(v.detach(),alpha=1-self.d)
 def state(self,m):
  z=m.state_dict();z.update(self.s);return z
def loss_fn(p,y):
 mse=((p-y[:,None])**2).mean();q=p.mean(1);q=q-q.mean();t=y-y.mean();cos=1-F.cosine_similarity(q[None],t[None],dim=1,eps=1e-8).mean();return mse+.05*cos
def cosine(y,p,center=False):
 if center:y=y-y.mean();p=p-p.mean()
 return float(np.dot(y,p)/(np.linalg.norm(y)*np.linalg.norm(p)+1e-12))
@torch.no_grad()
def predict(m,X):
 m.eval();z=[]
 for i in range(0,len(X),4096):z.append(m(torch.from_numpy(X[i:i+4096]).to(DEVICE)).mean(1).cpu().numpy())
 return np.concatenate(z)
def main():
 random.seed(42);np.random.seed(42);torch.manual_seed(42);X,y,mo,cols=load_data();tr=np.flatnonzero(mo<62);va=np.flatnonzero(mo>=62);sc=RobustSmooth().fit(X[tr]);X=sc.transform(X);Xt=torch.from_numpy(X).to(DEVICE);yt=torch.from_numpy(y).to(DEVICE);m=RealMLP(len(cols)).to(DEVICE);opt=torch.optim.AdamW(m.parameters(),lr=1e-3,betas=(.9,.98));ema=EMA(m);best=-1;steps=math.ceil(len(tr)/BS)*EPOCHS;step=0
 print(f'DEVICE={DEVICE}, feats={len(cols)}, params={sum(p.numel() for p in m.parameters())/1e6:.2f}M, train={len(tr)}, val={len(va)}',flush=True)
 for ep in range(EPOCHS):
  m.train();perm=torch.from_numpy(np.random.permutation(tr)).to(DEVICE);st=time.time();last=0
  for i in range(0,len(tr),BS):
   prog=step/steps;lr=1e-3*(1 if prog<.7 else max((1-prog)/.3,1e-3));opt.param_groups[0]['lr']=lr;ix=perm[i:i+BS];loss=loss_fn(m(Xt[ix]),yt[ix]);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();ema.update(m);last=loss.item();step+=1
  p=predict(m,X[va]);raw=cosine(y[va],p);cen=cosine(y[va],p,True);print(f'epoch {ep+1}/{EPOCHS}: loss={last:.5f}, raw={raw:.5f}, centered={cen:.5f}, sec={time.time()-st:.0f}',flush=True)
  if raw>best:best=raw;torch.save({'model':m.state_dict(),'cols':cols,'med':sc.med,'fac':sc.fac,'score':raw},'output/realmlp_best.pt')
 # evaluate EMA once
 em=RealMLP(len(cols)).to(DEVICE);em.load_state_dict(ema.state(m));p=predict(em,X[va]);print(f'EMA raw={cosine(y[va],p):.5f}, centered={cosine(y[va],p,True):.5f}; best={best:.5f}',flush=True);torch.save({'model':em.state_dict(),'cols':cols,'med':sc.med,'fac':sc.fac},'output/realmlp_ema.pt')
if __name__=='__main__':main()
