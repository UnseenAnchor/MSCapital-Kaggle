"""Late probe: RealMLP with train-only residual-quantized target auxiliary heads."""
import os,sys,time,math,random,json
import numpy as np,pandas as pd,torch
import torch.nn as nn,torch.nn.functional as F
from sklearn.cluster import KMeans
from proxy_lgb_feature_select import load_combined

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); EPOCHS=int(os.environ.get('EPOCHS','12')); BS=int(os.environ.get('BATCH','1024')); TOPN=int(os.environ.get('TOPN','128')); ENS=int(os.environ.get('N_ENS','8')); LAMBDA_RQ=float(os.environ.get('LAMBDA_RQ','0.1')); PREFIX=os.environ.get('OUT_PREFIX','realmlp_rq_late'); TRAIN_END=62; VALID_END=71; SCALE=1000.
def unit(x):x=np.asarray(x,float);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
def cosine(y,p):return float(unit(y)@unit(p))
def metrics(y,p,m):
 z=[cosine(y[m==q],p[m==q]) for q in np.unique(m)];return {'cosine':cosine(y,p),'month_mean':float(np.mean(z)),'month_min':float(np.min(z))}
class Smooth:
 def fit(self,X):
  self.med=np.nanmedian(X,0);q=np.nanquantile(X,.75,0)-np.nanquantile(X,.25,0);r=np.nanmax(X,0)-np.nanmin(X,0);q=np.where(q==0,.5*r,q);self.fac=np.where(q==0,0,1/(q+1e-30));return self
 def transform(self,X):
  X=np.where(np.isfinite(X),X,self.med);z=self.fac*(X-self.med);return (z/np.sqrt(1+(z/3)**2)).astype(np.float32)
class PBLD(nn.Module):
 def __init__(self,n,f,h=16,out=3):
  super().__init__();self.w1=nn.Parameter(torch.randn(n,f,h));self.b1=nn.Parameter(torch.empty(n,f,h));self.w2=nn.Parameter(torch.randn(n,f,h,out-1)/np.sqrt(h));self.b2=nn.Parameter(torch.randn(n,f,out-1));nn.init.uniform_(self.b1,-np.pi,np.pi)
 def forward(self,x):
  e=x[:,None,:,None];p=torch.cos(2*np.pi*(e*self.w1[None]+self.b1[None]));z=F.gelu(torch.einsum('bnfh,nfhd->bnfd',p,self.w2)+self.b2[None]);return torch.cat([x[:,None,:,None].expand(-1,self.w1.shape[0],-1,1),z],-1).flatten(2)
class NL(nn.Module):
 def __init__(self,n,i,o):super().__init__();self.i=i;self.w=nn.Parameter(torch.randn(n,i,o));self.b=nn.Parameter(torch.zeros(n,o))
 def forward(self,x):return torch.einsum('bni,nio->bno',x,self.w)/np.sqrt(self.i)+self.b
class Net(nn.Module):
 def __init__(self,f):
  super().__init__();self.n=ENS;self.emb=PBLD(ENS,f);d=f*3;self.scale=nn.Parameter(torch.ones(ENS,d));self.l1=NL(ENS,d,384);self.l2=NL(ENS,384,256);self.l3=NL(ENS,256,64);self.head=NL(ENS,64,1);self.codes=nn.ModuleList([NL(ENS,64,3) for _ in range(3)]);mask=torch.ones(ENS,d)
  for i in range(ENS):mask[i,i::max(1,ENS//2)]=0
  self.register_buffer('mask',mask)
 def features(self,x):
  z=self.emb(x)*self.mask[None]*self.scale[None];z=F.gelu(self.l1(z));z=F.dropout(z,.02,self.training);z=F.gelu(self.l2(z));z=F.dropout(z,.02,self.training);return F.gelu(self.l3(z))
 def forward(self,x):
  z=self.features(x);return self.head(z).squeeze(-1),[h(z) for h in self.codes]
class RQ:
 def __init__(self):self.ks=[]
 def fit(self,y):
  r=y.reshape(-1,1)
  for _ in range(3):
   k=KMeans(3,random_state=42,n_init=10).fit(r);self.ks.append(k);r=r-k.cluster_centers_[k.labels_]
  return self
 def encode(self,y):
  r=y.reshape(-1,1);out=[]
  for k in self.ks:
   c=k.predict(r);out.append(c);r=r-k.cluster_centers_[c]
  return np.stack(out,1)
def loss(p,logits,y,codes):
 ye=y[:,None].expand_as(p);mse=F.smooth_l1_loss(p,ye);yc=y-y.mean();pc=p-p.mean(0,keepdim=True);mc=1-F.cosine_similarity(pc.T,yc[None].expand(p.shape[1],-1),dim=1,eps=1e-8).mean();mp=p.mean(1);mc2=1-F.cosine_similarity((mp-mp.mean())[None],yc[None],dim=1,eps=1e-8).mean();rq=sum(F.cross_entropy(a.reshape(-1,3),codes[:,i,None].expand(-1,ENS).reshape(-1)) for i,a in enumerate(logits))/3;return mse+.03*mc+.02*mc2+LAMBDA_RQ*rq
@torch.no_grad()
def infer(m,X,bs=4096):
 m.eval();o=[]
 for i in range(0,len(X),bs):o.append(m(X[i:i+bs])[0].mean(1).float().cpu().numpy())
 return np.concatenate(o)
def main():
 random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42);t0=time.time();lab,Xall=load_combined();rank=pd.read_csv('output/proxy_lgb_trainonly_importance.csv');cols=rank.feature.head(TOPN).tolist();X=Xall[cols].replace([np.inf,-np.inf],np.nan).to_numpy(np.float64);del Xall;mo=lab.month.to_numpy();y0=lab.target.to_numpy(np.float32);tr=np.flatnonzero(mo<TRAIN_END);va=np.flatnonzero((mo>=TRAIN_END)&(mo<VALID_END));sc=Smooth().fit(X[tr]);X=sc.transform(X);codes=RQ().fit(y0[tr]*SCALE).encode(y0*SCALE);Xt=torch.from_numpy(X).to(DEVICE);yt=torch.from_numpy(y0*SCALE).to(DEVICE);ct=torch.from_numpy(codes).long().to(DEVICE);m=Net(TOPN).to(DEVICE);opt=torch.optim.AdamW(m.parameters(),lr=8e-4,betas=(.9,.98),weight_decay=1e-5);ema={k:v.detach().clone() for k,v in m.state_dict().items() if v.is_floating_point()};steps=math.ceil(len(tr)/BS)*EPOCHS;step=0;best=-9;hist=[];print(f'DEVICE={DEVICE} prefix={PREFIX} topn={TOPN} ens={ENS} rq={LAMBDA_RQ} train={len(tr)} val={len(va)}',flush=True)
 for ep in range(1,EPOCHS+1):
  m.train();perm=torch.from_numpy(np.random.permutation(tr)).to(DEVICE);tot=0;st=time.time()
  for i in range(0,len(tr),BS):
   ix=perm[i:i+BS];prog=step/steps;lr=8e-4*(1 if prog<.6 else max((1-prog)/.4,.02));opt.param_groups[0]['lr']=lr;p,lg=m(Xt[ix]);z=loss(p,lg,yt[ix],ct[ix]);opt.zero_grad(set_to_none=True);z.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();
   with torch.no_grad():
    for k,v in m.state_dict().items():
     if k in ema:ema[k].mul_(.998).add_(v,alpha=.002)
   tot+=z.item()*len(ix);step+=1
  pred=infer(m,Xt[va]);met=metrics(y0[va],pred,mo[va]);hist.append({'epoch':ep,'loss':tot/len(tr),**met});print('epoch',hist[-1],'sec',round(time.time()-st),flush=True);torch.save({'model':m.state_dict(),'cols':cols,'med':sc.med,'fac':sc.fac,'epoch':ep,'metrics':met},f'output/{PREFIX}_ep{ep}.pt');best=max(best,met['cosine'])
 m.load_state_dict(ema,strict=False);pred=infer(m,Xt[va]);met=metrics(y0[va],pred,mo[va]);np.savez(f'output/{PREFIX}_oof.npz',sample_id=lab.sample_id.to_numpy()[va],target=y0[va],month=mo[va],prediction=pred);torch.save({'model':m.state_dict(),'cols':cols,'med':sc.med,'fac':sc.fac,'metrics':met},f'output/{PREFIX}_ema.pt');json.dump({'best':best,'ema':met,'history':hist},open(f'output/{PREFIX}_metrics.json','w'),indent=2);print('EMA',met,'total_sec',round(time.time()-t0),flush=True)
if __name__=='__main__':main()
