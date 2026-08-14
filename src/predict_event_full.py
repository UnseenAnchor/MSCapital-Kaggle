"""Predict pure event diagnostic and conservative 0.144-backbone enhancement."""
import hashlib,numpy as np,pandas as pd,torch
from train_event_residual import load_arrays,Prep,Net,infer,unit,DEVICE

def save(path,ids,p):pd.DataFrame({'sample_id':ids,'prediction':p}).to_csv(path,index=False);print(path,'sha256',hashlib.sha256(open(path,'rb').read()).hexdigest())
def main():
 sub=pd.read_csv('data/submission.csv');ids=np.load('features/cache/test_ids.npy');assert np.array_equal(ids,sub.sample_id);z=np.load('output/event_direct_full_norm.npz');stats={k:(z[k+'_mean'],z[k+'_std']) for k in ('tx','order')};prep=Prep(stats);A=load_arrays('test');preds=[]
 for ep in (6,9,12):
  m=Net().to(DEVICE);m.load_state_dict(torch.load(f'output/event_direct_full_ep{ep}.pt',map_location=DEVICE));p=infer(m,A,np.arange(len(ids)),prep);preds.append(p);print('ep',ep,'mean/std',p.mean(),p.std(),flush=True);del m;torch.cuda.empty_cache()
 event=np.mean([unit(p) for p in preds],axis=0);own=unit(pd.read_csv('output/candidate_v3_eff1024_conservative20.csv').prediction);ref=unit(pd.read_csv('research/lb0142/submission_ref_lb0142.csv').sort_values('sample_id').prediction);current=unit(pd.read_csv('output/candidate_ours40_public142_60.csv').prediction);self_enh=.98*own+.02*event;cand=.4*unit(self_enh)+.6*ref
 print('correlations event own/ref/current',float(event@own),float(event@ref),float(event@current),'candidate/current',float(unit(cand)@current),flush=True);save('output/diagnostic_event_direct_full_unit.csv',ids,event);save('output/candidate_0144_event_slot2.csv',ids,cand);np.savez('output/event_direct_full_test_preds.npz',ep6=preds[0],ep9=preds[1],ep12=preds[2],ensemble=event)
if __name__=='__main__':main()
