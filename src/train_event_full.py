"""Full-data raw event-order Transformer training, fixed epochs 6/9/12."""
import time,random,numpy as np,pandas as pd,torch
from train_event_residual import load_arrays,fit_stats,batches,Prep,Net,DEVICE,BS,EPOCHS,SEED
from train_event_direct import loss_fn
def main():
 random.seed(SEED);np.random.seed(SEED);torch.manual_seed(SEED);torch.cuda.manual_seed_all(SEED);lab=pd.read_feather('data/train/label.feather').sort_values('sample_id').reset_index(drop=True);y=lab.target.to_numpy(np.float32);idx=np.arange(len(lab));A=load_arrays('train');stats=fit_stats(A,idx);np.savez('output/event_direct_full_norm.npz',**{f'{k}_{s}':v[i] for k,v in stats.items() for i,s in enumerate(['mean','std'])});prep=Prep(stats);model=Net().to(DEVICE)
 try:opt=torch.optim.AdamW(model.parameters(),lr=6e-4,weight_decay=1e-4,fused=True)
 except TypeError:opt=torch.optim.AdamW(model.parameters(),lr=6e-4,weight_decay=1e-4)
 sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS);scaler=torch.cuda.amp.GradScaler(enabled=DEVICE.type=='cuda');print('full train',len(idx),'params',sum(p.numel() for p in model.parameters())/1e6,flush=True)
 for ep in range(1,EPOCHS+1):
  model.train();tot=seen=0;st=time.time()
  for b in batches(A,idx,y,BS,True,SEED+ep):
   tx,tm,o,om,yy=prep.batch(b)
   with torch.cuda.amp.autocast(enabled=DEVICE.type=='cuda'):loss=loss_fn(model(tx,tm,o,om),yy)
   opt.zero_grad(set_to_none=True);scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1);scaler.step(opt);scaler.update();tot+=float(loss)*len(yy);seen+=len(yy)
  sched.step()
  if ep in (6,9,12):torch.save(model.state_dict(),f'output/event_direct_full_ep{ep}.pt')
  print('epoch',ep,'loss',tot/seen,'sec',round(time.time()-st),flush=True)
if __name__=='__main__':main()
