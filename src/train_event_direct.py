"""Direct-target raw event-order Transformer; evaluated only as a backbone complement."""
import gc,time,random,os,numpy as np,pandas as pd,torch
from train_event_residual import load_arrays,fit_stats,batches,Prep,Net,infer,unit,cosine,DEVICE,BS,EPOCHS,SEED
from residual_realmlp_rolling import backbone,project_residual,fold_stats

def loss_fn(p,y):
 p0=p-p.mean();y0=y-y.mean();cos=1-torch.nn.functional.cosine_similarity(p0[None],y0[None],dim=1,eps=1e-8).mean();return .8*cos+.2*torch.nn.functional.smooth_l1_loss(p,y*1000.)
def main():
 recent=int(os.environ.get('RECENT_MONTHS','0'));tag=f'event_direct_recent{recent}' if recent else 'event_direct';random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);lab=pd.read_feather('data/train/label.feather').sort_values('sample_id').reset_index(drop=True);sid=lab.sample_id.to_numpy();mo=lab.month.to_numpy();y=lab.target.to_numpy(np.float32);A=load_arrays('train');saved={};tall=time.time()
 for fold,train_end,valid_end in [('proxy',45,71),('middle',51,61),('late',62,71)]:
  tri=np.flatnonzero((mo<train_end)&((mo>=train_end-recent) if recent else True));vai=np.flatnonzero((mo>=train_end)&(mo<valid_end));prep=Prep(fit_stats(A,tri));model=Net().to(DEVICE)
  try:opt=torch.optim.AdamW(model.parameters(),lr=6e-4,weight_decay=1e-4,fused=True)
  except TypeError:opt=torch.optim.AdamW(model.parameters(),lr=6e-4,weight_decay=1e-4)
  sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS);scaler=torch.cuda.amp.GradScaler(enabled=DEVICE.type=='cuda');preds={};print('\n',fold,'train',len(tri),'valid',len(vai),flush=True)
  for ep in range(1,EPOCHS+1):
   model.train();tot=seen=0;st=time.time()
   for b in batches(A,tri,y,BS,True,SEED+ep):
    tx,tm,o,om,yy=prep.batch(b)
    with torch.cuda.amp.autocast(enabled=DEVICE.type=='cuda'):loss=loss_fn(model(tx,tm,o,om),yy)
    opt.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1);scaler.step(opt);scaler.update();tot+=float(loss)*len(yy);seen+=len(yy)
   sched.step()
   if ep in (6,9,12):preds[ep]=infer(model,A,vai,prep);torch.save(model.state_dict(),f'output/{tag}_{fold}_ep{ep}.pt')
   print(' epoch',ep,'loss',tot/seen,'sec',round(time.time()-st),flush=True)
  q=np.mean([unit(preds[e]) for e in (6,9,12)],axis=0);ids=sid[vai];base=backbone(fold,ids);yv=y[vai].astype(np.float64);mv=mo[vai];true_res,_=project_residual(yv,base);print('ensemble',fold_stats(yv,q,mv),'corr_base',float(unit(q)@unit(base)),'partial',cosine(true_res,q),flush=True);print('base',fold_stats(yv,base,mv),flush=True)
  for w in [.02,.05,.10,.15,.20]:print(' event_weight',w,fold_stats(yv,(1-w)*unit(base)+w*unit(q),mv),flush=True)
  saved.update({f'{fold}_sample_id':ids,f'{fold}_target':yv,f'{fold}_month':mv,f'{fold}_base':base,f'{fold}_event':q});del model;torch.cuda.empty_cache();gc.collect()
 np.savez(f'output/{tag}_rolling_oof.npz',**saved);print('total_sec',time.time()-tall)
if __name__=='__main__':main()
