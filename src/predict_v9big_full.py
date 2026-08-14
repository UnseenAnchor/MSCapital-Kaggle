"""Predict fixed full-data v9_big checkpoint ensemble. Never submits itself."""
import sys
from pathlib import Path
import numpy as np,pandas as pd,torch
sys.path.insert(0,str(Path('research/lb0142').resolve()))
from lb0142.models_v9 import MultiStreamModel
from train_v9big_fast import CFG,ROOT,VER,ML,FL,DEVICE,Prep,load_ram,batches
PREFIX='v9big_full_fast';BS=256

def arrays_test(n):
 def mm(name,shape):return np.memmap(f'{ROOT}/test_{VER}_{name}.mmap',np.float16,'r',shape=shape)
 return {'market':mm(f'market_{ML}x11',(n,ML,11)),'market_count':mm(f'market_count_{ML}',(n,ML)),'tx':mm(f'tx_{FL}x7',(n,FL,7)),'tx_count':mm(f'tx_count_{FL}',(n,FL)),'order':mm(f'order_{FL}x10',(n,FL,10))}
def unit(x):x=np.asarray(x,np.float64);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
@torch.no_grad()
def infer(model,A,prep,n):
 model.eval();out=[];dummy=np.zeros(n,np.float32)
 for b in batches(A,np.arange(n),dummy,BS*2,False,42):
  m,t,o,_=prep.batch(b)
  with torch.cuda.amp.autocast(enabled=DEVICE.type=='cuda'):p=model(m,t,o)
  out.append(p.float().cpu().numpy())
 return np.concatenate(out)
def main():
 sub=pd.read_csv('data/submission.csv');n=len(sub);A=load_ram(arrays_test(n));z=np.load(f'output/{PREFIX}_norm.npz');norm={k:(z[k+'_mean'],z[k+'_std']) for k in ('market','tx','order')};prep=Prep(norm);preds=[]
 for ep in (4,5,6):
  model=MultiStreamModel(CFG).to(DEVICE);model.load_state_dict(torch.load(f'output/{PREFIX}_ep{ep}.pt',map_location=DEVICE));p=infer(model,A,prep,n);preds.append(p);print('ep',ep,'mean/std',p.mean(),p.std(),flush=True);del model;torch.cuda.empty_cache()
 ens=np.mean([unit(p) for p in preds],axis=0);out=pd.DataFrame({'sample_id':sub.sample_id,'prediction':ens});out.to_csv('output/submission_v9big_full_unit.csv',index=False);np.savez('output/v9big_full_test_preds.npz',ep4=preds[0],ep5=preds[1],ep6=preds[2],ensemble=ens);print(out.prediction.describe(),flush=True)
if __name__=='__main__':main()
