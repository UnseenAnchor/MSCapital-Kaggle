"""Predict full-data RealMLP checkpoint ensemble and make tabular blend. Never submits."""
import numpy as np,pandas as pd,torch
import train_realmlp_proxy_v4 as rv4
from train_realmlp_proxy_v3 import RobustSmooth,unit,Y_SCALE,DEVICE

def load_test(cols):
 base=pd.read_csv('data/submission.csv')[['sample_id']].sort_values('sample_id').reset_index(drop=True)
 d=base.copy()
 for name in ('market','order','transaction'):
  for suffix in ('','_v2'):
   d=d.merge(pd.read_parquet(f'features/{name}_test{suffix}.parquet'),on='sample_id',how='left')
 micro=pd.read_parquet('features/micro_v3/test.parquet').sort_values('sample_id').reset_index(drop=True);old=[c for c in d if c!='sample_id'];new=[c for c in micro if c!='sample_id'];X=d[old].join(micro[new].add_prefix('v3_'));return base.sample_id.to_numpy(),X[cols]
def infer(ep,ids,X):
 z=torch.load(f'output/realmlp_v4_full_ep{ep}.pt',map_location='cpu');raw=X.replace([np.inf,-np.inf],np.nan).to_numpy(np.float64);sc=RobustSmooth();sc.med=z['med'];sc.fac=z['fac'];xt=torch.from_numpy(sc.transform(raw)).to(DEVICE);m=rv4.RealMLP(len(z['cols']),z['n_ens']).to(DEVICE);m.load_state_dict(z['model']);pm=rv4.predict(m,xt,member=True)/Y_SCALE;del xt,m;torch.cuda.empty_cache();p=np.mean([unit(pm[:,i]) for i in range(pm.shape[1])],0);pd.DataFrame({'sample_id':ids,'prediction':p}).to_csv(f'output/submission_realmlp_v4_full_ep{ep}_unit.csv',index=False);return p
def main():
 z=torch.load('output/realmlp_v4_full_ep6.pt',map_location='cpu');ids,X=load_test(z['cols']);ps=[infer(e,ids,X) for e in (6,9,11)];p=np.mean([unit(x) for x in ps],0);pd.DataFrame({'sample_id':ids,'prediction':p}).to_csv('output/submission_realmlp_v4_unit.csv',index=False)
 l=pd.read_csv('output/submission_micro_lgb_full_unit.csv').sort_values('sample_id');assert np.array_equal(ids,l.sample_id.to_numpy());tab=.6*unit(l.prediction.to_numpy())+.4*unit(p);pd.DataFrame({'sample_id':ids,'prediction':tab}).to_csv('output/submission_tabular_lgb60_realmlp40_unit.csv',index=False);print('saved',len(ids),'checkpoint corr',np.corrcoef(ps),'lgb/real/tab corr',np.corrcoef([unit(l.prediction),unit(p),unit(tab)]))
if __name__=='__main__':main()
