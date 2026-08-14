"""Nested chronological residual CNN-Transformer on raw v2 grids. Never submits."""
import gc,time,random,numpy as np,pandas as pd,torch
import torch.nn.functional as F
from pathlib import Path
import sys
sys.path.insert(0,str(Path('research/lb0142').resolve()))
from lb0142.models_v9 import GridCfg,MultiStreamModel,cosine_init_scale
from train_v9big_fast import arrays,load_ram,norm_stats,Prep,batches,DEVICE
from residual_realmlp_rolling import backbone,project_residual,fold_stats
BS=256;ACC=4;EPOCHS=12;SEED=42;CFG=GridCfg(d_model=64,n_layers=2,cnn_channels=64,market_len=200,flow_len=60,batch_size=BS,num_workers=0)
torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
@torch.no_grad()
def infer(model,A,idx,prep):
 model.eval();out=[];dummy=np.zeros(len(A['market']),np.float32)
 for b in batches(A,idx,dummy,BS*2,False,SEED):
  m,t,o,_=prep.batch(b)
  with torch.cuda.amp.autocast(enabled=DEVICE.type=='cuda'):p=model(m,t,o)
  out.append(p.float().cpu().numpy())
 return np.concatenate(out)
def unit(x):x=np.asarray(x,np.float64);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
def cosine(y,p):return float(unit(y)@unit(p))
def main():
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);lab=pd.read_feather('data/train/label.feather').sort_values('sample_id').reset_index(drop=True);sid=lab.sample_id.to_numpy();mo=lab.month.to_numpy();y=lab.target.to_numpy(np.float64);pos=pd.Series(np.arange(len(lab)),index=sid);rz=np.load('output/residual_lgb_rolling_oof.npz');A=load_ram(arrays(len(lab)));saved={};tall=time.time()
 for fold,train_end in [('proxy',45),('middle',51),('late',62)]:
  tri=pos.loc[rz[f'{fold}_train_sample_id']].to_numpy();vai=pos.loc[rz[f'{fold}_sample_id']].to_numpy();ytarget=np.zeros(len(lab),np.float32);ytarget[tri]=rz[f'{fold}_train_residual_target'].astype(np.float32);norm=norm_stats(A,np.flatnonzero(mo<train_end));prep=Prep(norm);model=MultiStreamModel(CFG).to(DEVICE);cosine_init_scale(model)
  try:opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4,fused=True)
  except TypeError:opt=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
  sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS);scaler=torch.cuda.amp.GradScaler(enabled=DEVICE.type=='cuda');preds={};print('\n',fold,'res_train',len(tri),'valid',len(vai),'params',sum(p.numel() for p in model.parameters())/1e6,flush=True)
  for ep in range(1,EPOCHS+1):
   model.train();opt.zero_grad(set_to_none=True);pending=0;tot=seen=0;st=time.time()
   for b in batches(A,tri,ytarget,BS,True,SEED+ep):
    m,t,o,yy=prep.batch(b)
    with torch.cuda.amp.autocast(enabled=DEVICE.type=='cuda'):
     p=model(m,t,o);loss=1-F.cosine_similarity((p-p.mean())[None],(yy-yy.mean())[None],dim=1,eps=1e-8).mean()
    scaler.scale(loss/ACC).backward();pending+=1;tot+=float(loss)*len(yy);seen+=len(yy)
    if pending==ACC:
     scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1);scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True);pending=0
   if pending:
    for p in model.parameters():
     if p.grad is not None:p.grad.mul_(ACC/pending)
    scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1);scaler.step(opt);scaler.update();opt.zero_grad(set_to_none=True)
   sched.step()
   if ep in (4,5,6):preds[ep]=infer(model,A,vai,prep);torch.save(model.state_dict(),f'output/residual_sequence_{fold}_ep{ep}.pt')
   print(' epoch',ep,'loss',tot/seen,'sec',round(time.time()-st),flush=True)
  q=np.mean([unit(preds[e]) for e in (4,5,6)],axis=0);ids=sid[vai];base=backbone(fold,ids);yt=y[vai];mv=mo[vai];true_res,_=project_residual(yt,base);print('ensemble',fold_stats(yt,q,mv),'corr_base',float(unit(q)@unit(base)),'partial',cosine(true_res,q),flush=True);print('base',fold_stats(yt,base,mv),flush=True)
  for w in [.02,.05,.10,.15]:print(' residual_weight',w,fold_stats(yt,(1-w)*unit(base)+w*unit(q),mv),flush=True)
  saved.update({f'{fold}_sample_id':ids,f'{fold}_target':yt,f'{fold}_month':mv,f'{fold}_base':base,f'{fold}_residual':q});del model;torch.cuda.empty_cache();gc.collect()
 np.savez('output/residual_sequence_rolling_oof.npz',**saved);print('total_sec',time.time()-tall)
if __name__=='__main__':main()
