"""Leakage-safe RealMLP proxy-CV on train-only selected microstructure features.
Train months 0-44, validate 45-70. Batch >=1024 and 16 epochs by default. Never submits.
"""
import os,sys,time,math,random,json
import numpy as np,pandas as pd,torch
import torch.nn as nn
import torch.nn.functional as F
from proxy_lgb_feature_select import load_combined

DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS=int(os.environ.get('EPOCHS',sys.argv[1] if len(sys.argv)>1 else 16));BS=int(os.environ.get('BATCH','1024'));TOPN=int(os.environ.get('TOPN','128'));SEED=int(os.environ.get('SEED','42'));N_ENS=int(os.environ.get('N_ENS','8'));Y_SCALE=1000.0
TRAIN_END=int(os.environ.get('TRAIN_END','45'));VALID_END=int(os.environ.get('VALID_END','71'));PREFIX=os.environ.get('OUT_PREFIX','realmlp_proxy_v4')
assert BS>=1024,'BATCH must be at least 1024'

def unit(x):
 x=np.asarray(x,np.float64);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
def cosine(y,p):return float(unit(y)@unit(p))
class RobustSmooth:
 def fit(self,X):
  self.med=np.nanmedian(X,0);q=np.nanquantile(X,.75,axis=0)-np.nanquantile(X,.25,axis=0);rg=np.nanmax(X,0)-np.nanmin(X,0);q=np.where(q==0,.5*rg,q);self.fac=np.where(q==0,0,1/(q+1e-30));return self
 def transform(self,X):
  X=np.where(np.isfinite(X),X,self.med);z=self.fac*(X-self.med);return (z/np.sqrt(1+(z/3)**2)).astype(np.float32)
class PBLD(nn.Module):
 def __init__(self,n,f,h=16,out=3):
  super().__init__();self.w1=nn.Parameter(torch.randn(n,f,h));self.b1=nn.Parameter(torch.empty(n,f,h));self.w2=nn.Parameter(torch.randn(n,f,h,out-1)/np.sqrt(h));self.b2=nn.Parameter(torch.randn(n,f,out-1));nn.init.uniform_(self.b1,-np.pi,np.pi)
 def forward(self,x):
  e=x[:,None,:,None];p=torch.cos(2*np.pi*(e*self.w1[None]+self.b1[None]));z=F.gelu(torch.einsum('bnfh,nfhd->bnfd',p,self.w2)+self.b2[None]);return torch.cat([x[:,None,:,None].expand(-1,self.w1.shape[0],-1,1),z],-1).flatten(2)
class NLinear(nn.Module):
 def __init__(self,n,i,o):super().__init__();self.i=i;self.w=nn.Parameter(torch.randn(n,i,o));self.b=nn.Parameter(torch.zeros(n,o))
 def forward(self,x):return torch.einsum('bni,nio->bno',x,self.w)/np.sqrt(self.i)+self.b
class RealMLP(nn.Module):
 def __init__(self,f,n=N_ENS):
  super().__init__();self.n=n;self.emb=PBLD(n,f);d=f*3;self.scale=nn.Parameter(torch.ones(n,d));self.l1=NLinear(n,d,384);self.l2=NLinear(n,384,256);self.l3=NLinear(n,256,64);self.head=NLinear(n,64,1);mask=torch.ones(n,d)
  for i in range(n):mask[i,i::max(1,n//2)]=0
  self.register_buffer('mask',mask)
 def forward(self,x):
  z=self.emb(x)*self.mask[None]*self.scale[None];z=F.gelu(self.l1(z));z=F.dropout(z,.02,self.training);z=F.gelu(self.l2(z));z=F.dropout(z,.02,self.training);z=F.gelu(self.l3(z));return self.head(z).squeeze(-1)
class EMA:
 def __init__(self,m,d=.998):self.d=d;self.s={k:v.detach().clone() for k,v in m.state_dict().items() if v.is_floating_point()}
 @torch.no_grad()
 def update(self,m):
  for k,v in m.state_dict().items():
   if k in self.s:self.s[k].mul_(self.d).add_(v.detach(),alpha=1-self.d)
 def apply_to(self,m):
  z=m.state_dict();z.update(self.s);m.load_state_dict(z)
def loss_fn(p,y):
 target=y[:,None].expand_as(p)
 mse=F.smooth_l1_loss(p,target)
 # Every member must generalize; cosine is auxiliary rather than the primary optimizer.
 yc=y-y.mean(); centered=p-p.mean(0,keepdim=True)
 member_cos=1-F.cosine_similarity(centered.T,yc[None].expand(p.shape[1],-1),dim=1,eps=1e-8).mean()
 mean=p.mean(1); mean_cos=1-F.cosine_similarity((mean-mean.mean())[None],yc[None],dim=1,eps=1e-8).mean()
 return mse+.03*member_cos+.02*mean_cos,mean_cos,mse
@torch.no_grad()
def predict(m,X,bs=4096,member=False):
 m.eval();out=[]
 for i in range(0,len(X),bs):out.append(m(X[i:i+bs]).float().cpu().numpy())
 z=np.concatenate(out);return z if member else z.mean(1)
def metrics(y,p,month):
 vals=[cosine(y[month==m],p[month==m]) for m in np.unique(month)];return dict(cosine=cosine(y,p),month_mean=float(np.mean(vals)),month_min=float(np.min(vals)),month_std=float(np.std(vals)))
def main():
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);t0=time.time();lab,Xall=load_combined();rank_file=os.environ.get('RANK_FILE','output/proxy_lgb_trainonly_importance.csv');rank=pd.read_csv(rank_file);cols=rank.feature.head(TOPN).tolist();X=Xall[cols].replace([np.inf,-np.inf],np.nan).to_numpy(np.float64);del Xall;mo=lab.month.to_numpy();y=lab.target.to_numpy(np.float32)*Y_SCALE;tr=np.flatnonzero(mo<TRAIN_END);va=np.flatnonzero((mo>=TRAIN_END)&(mo<VALID_END));sc=RobustSmooth().fit(X[tr]);X=sc.transform(X);Xt=torch.from_numpy(X).to(DEVICE);yt=torch.from_numpy(y).to(DEVICE);del X
 m=RealMLP(TOPN).to(DEVICE);opt=torch.optim.AdamW(m.parameters(),lr=8e-4,betas=(.9,.98),weight_decay=1e-5);ema=EMA(m);steps=math.ceil(len(tr)/BS)*EPOCHS;step=0;best=(-9,None);history=[]
 print(f'DEVICE={DEVICE} prefix={PREFIX} split=<{TRAIN_END}/[{TRAIN_END},{VALID_END}) topn={TOPN} ens={N_ENS} params={sum(p.numel() for p in m.parameters())/1e6:.2f}M batch={BS} epochs={EPOCHS} train={len(tr)} valid={len(va)}',flush=True)
 for ep in range(1,EPOCHS+1):
  m.train();perm=torch.from_numpy(np.random.permutation(tr)).to(DEVICE);st=time.time();tot=0
  for i in range(0,len(tr),BS):
   progress=step/steps;lr=8e-4*(1 if progress<.6 else max((1-progress)/.4,.02));opt.param_groups[0]['lr']=lr;ix=perm[i:i+BS];loss,lc,lm=loss_fn(m(Xt[ix]),yt[ix]);opt.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step();ema.update(m);tot+=loss.item()*len(ix);step+=1
  if len(va):
   p=predict(m,Xt[va]);met=metrics(y[va],p,mo[va])
  else:
   met={}
  history.append(dict(epoch=ep,loss=tot/len(tr),lr=lr,**met));print('epoch',ep,history[-1],'sec',round(time.time()-st),flush=True)
  torch.save({'model':m.state_dict(),'cols':cols,'med':sc.med,'fac':sc.fac,'epoch':ep,'metrics':met,'topn':TOPN,'n_ens':N_ENS,'train_end':TRAIN_END,'valid_end':VALID_END},f'output/{PREFIX}_ep{ep}.pt')
  if met and met['cosine']>best[0]:best=(met['cosine'],ep)
 em=RealMLP(TOPN).to(DEVICE);ema.apply_to(em)
 if len(va):
  pm=predict(em,Xt[va],member=True);p=pm.mean(1);emet=metrics(y[va],p,mo[va]);print('EMA',emet,'member cosine',[round(cosine(y[va],pm[:,i]),6) for i in range(N_ENS)],flush=True)
  np.savez(f'output/{PREFIX}_oof.npz',sample_id=lab.sample_id.to_numpy()[va],target=y[va]/Y_SCALE,month=mo[va],prediction=p/Y_SCALE,members=pm/Y_SCALE)
 else:
  emet={}
 torch.save({'model':em.state_dict(),'cols':cols,'med':sc.med,'fac':sc.fac,'metrics':emet,'topn':TOPN,'n_ens':N_ENS,'epochs':EPOCHS,'train_end':TRAIN_END,'valid_end':VALID_END},f'output/{PREFIX}_ema.pt');json.dump({'best':best,'ema':emet,'history':history},open(f'output/{PREFIX}_metrics.json','w'),indent=2);print('best',best,'total_sec',time.time()-t0)
if __name__=='__main__':main()
