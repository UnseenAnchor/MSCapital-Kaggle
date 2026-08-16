import hashlib,numpy as np,pandas as pd,torch
from train_crossscale_delta import Prep,Net,unit,DEVICE,BS,batches,stats,ROOT2,ROOT3

def load_split(root,ver,ml,fl,split,n):
 def one(name,c):return np.array(np.memmap(f'{root}/{split}_{ver}_{name}_{ml if name=="market" else fl}x{c}.mmap',np.float16,'r',shape=(n,ml if name=='market' else fl,c)),copy=True)
 return {'market':one('market',11),'tx':one('tx',7),'order':one('order',10)}
def add_delta_stats(s3,A2,A3,idx):
 for k in ('market','tx','order'):
  x=(A3[k][idx,::2].astype(np.float32)-A2[k][idx].astype(np.float32)).reshape(-1,A2[k].shape[-1]);s3[k+'_delta']=(x.mean(0).astype('f4'),np.maximum(x.std(0),1e-6).astype('f4'))
 return s3
@torch.no_grad()
def infer(m,A2,A3,prep):
 m.eval();idx=np.arange(len(A2['market']));dummy=np.zeros(len(idx),np.float32);out=[]
 for b in batches(A2,A3,idx,dummy,BS*2,False,42):
  z=prep.batch(b);out.append(m(*z[:-1]).float().cpu().numpy())
 return np.concatenate(out)
def save(path,ids,p):pd.DataFrame({'sample_id':ids,'prediction':p}).to_csv(path,index=False);print(path,hashlib.sha256(open(path,'rb').read()).hexdigest())
def main():
 ids_tr=pd.read_feather('data/train/label.feather').sort_values('sample_id').sample_id.to_numpy();n=len(ids_tr);A2=load_split(ROOT2,'v2',200,60,'train',n);A3=load_split(ROOT3,'v3',400,120,'train',n);idx=np.arange(n);s2=stats(A2,idx);s3=add_delta_stats(stats(A3,idx),A2,A3,idx);del A2,A3
 ids=pd.read_csv('data/submission.csv').sample_id.to_numpy();A2=load_split(ROOT2,'v2',200,60,'test',len(ids));A3=load_split(ROOT3,'v3',400,120,'test',len(ids));prep=Prep(s2,s3);pred={}
 for ep in (5,6,7):
  m=Net().to(DEVICE);m.load_state_dict(torch.load(f'output/crossscale_delta_full_ep{ep}.pt',map_location=DEVICE));pred[ep]=infer(m,A2,A3,prep);print('ep',ep,pred[ep].mean(),pred[ep].std());del m;torch.cuda.empty_cache()
 delta=np.mean([unit(pred[e]) for e in (5,6,7)],0);v3=unit(pd.read_csv('output/submission_multistream_v3_eff1024_unit.csv').sort_values('sample_id').prediction);ref=unit(pd.read_csv('research/lb0142/submission_ref_lb0142.csv').sort_values('sample_id').prediction);old=unit(pd.read_csv('output/candidate_independent_stack_public60_40.csv').sort_values('sample_id').prediction);slot=unit(.5*v3+.5*delta);final=unit(.6*ref+.4*slot)
 print('delta/v3',delta@v3,'delta/ref',delta@ref,'new/current0145',final@old);save('output/diagnostic_crossscale_delta_full_unit.csv',ids,delta);save('output/candidate_crossscale_delta_public60_40.csv',ids,final);np.savez('output/crossscale_delta_full_test_preds.npz',ep5=pred[5],ep6=pred[6],ep7=pred[7],delta=delta,slot=slot,final=final)
if __name__=='__main__':main()
