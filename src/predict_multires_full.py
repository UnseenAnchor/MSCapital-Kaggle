"""Predict full-data multi-resolution self-anchor checkpoints 5/6/7."""
import hashlib,numpy as np,pandas as pd,torch
from train_multires_self_anchor import Prep,Net,unit,DEVICE,BS,batches,stats,ROOT2,ROOT3

def load_split(root,ver,ml,fl,split,n):
 def one(name,c):return np.array(np.memmap(f'{root}/{split}_{ver}_{name}_{ml if name=="market" else fl}x{c}.mmap',np.float16,'r',shape=(n,ml if name=='market' else fl,c)),copy=True)
 return {'market':one('market',11),'tx':one('tx',7),'order':one('order',10)}
@torch.no_grad()
def infer(m,A2,A3,idx,prep):
 m.eval();dummy=np.zeros(len(idx),np.float32);out=[]
 idx=np.arange(len(A2['market']));dummy=np.zeros(len(A2['market']),np.float32)
 for b in batches(A2,A3,idx,dummy,BS*2,False,42):
  z=prep.batch(b);out.append(m(*z[:-1]).float().cpu().numpy())
 return np.concatenate(out)
def save(path,ids,p):pd.DataFrame({'sample_id':ids,'prediction':p}).to_csv(path,index=False);print(path,hashlib.sha256(open(path,'rb').read()).hexdigest())
def main():
 train_ids=pd.read_feather('data/train/label.feather').sort_values('sample_id').sample_id.to_numpy();n=len(train_ids);A2=load_split(ROOT2,'v2',200,60,'train',n);A3=load_split(ROOT3,'v3',400,120,'train',n);s2=stats(A2,np.arange(n));s3=stats(A3,np.arange(n));del A2,A3
 sub=pd.read_csv('data/submission.csv');ids=sub.sample_id.to_numpy();A2=load_split(ROOT2,'v2',200,60,'test',len(ids));A3=load_split(ROOT3,'v3',400,120,'test',len(ids));prep=Prep(s2,s3);pred={}
 for ep in (5,6,7):
  m=Net().to(DEVICE);m.load_state_dict(torch.load(f'output/multires_self_full_ep{ep}.pt',map_location=DEVICE));pred[ep]=infer(m,A2,A3,np.arange(len(ids)),prep);print('ep',ep,pred[ep].mean(),pred[ep].std(),flush=True);del m;torch.cuda.empty_cache()
 multi=np.mean([unit(pred[e]) for e in (5,6,7)],axis=0);v3=unit(pd.read_csv('output/submission_multistream_v3_eff1024_unit.csv').sort_values('sample_id').prediction);ref=unit(pd.read_csv('research/lb0142/submission_ref_lb0142.csv').sort_values('sample_id').prediction);base=unit(pd.read_csv('output/candidate_proxycv_realmlp5.csv').sort_values('sample_id').prediction);slot=unit(.5*v3+.5*multi);self_anchor=unit(.8*base+.2*slot);final=.4*self_anchor+.6*ref
 print('corr multi/v3/ref/self/final',multi@v3,multi@ref,self_anchor@ref,final@unit(pd.read_csv('output/candidate_ours40_public142_60.csv').sort_values('sample_id').prediction),flush=True)
 save('output/diagnostic_multires_self_full_unit.csv',ids,multi);save('output/candidate_top10_multires_self_anchor.csv',ids,final);np.savez('output/multires_self_full_test_preds.npz',ep5=pred[5],ep6=pred[6],ep7=pred[7],multi=multi,slot=slot,self_anchor=self_anchor,final=final)
if __name__=='__main__':main()
