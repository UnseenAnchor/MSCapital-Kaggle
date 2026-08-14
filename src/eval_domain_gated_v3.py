"""Evaluate pre-registered sample-wise domain gate between base and joint v3."""
import numpy as np,pandas as pd

def unit(x):x=np.asarray(x,np.float64);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
def stats(y,p,mo):
 v=[float(unit(y[mo==m])@unit(p[mo==m])) for m in np.unique(mo)];return float(unit(y)@unit(p)),float(np.mean(v)),float(np.min(v)),float(np.std(v))
def main():
 lab=pd.read_feather('data/train/label.feather').sort_values('sample_id');pos=pd.Series(np.arange(len(lab)),index=lab.sample_id);d=np.load('output/domain_scores.npz');assert np.array_equal(d['train_sample_id'],lab.sample_id.to_numpy());ds=d['train_score'];rows=[];saved={}
 specs=[('proxy','output/multistream_v3_proxy_oof.npz','output/joint_v3_proxy_fast_oof.npz'),('middle','output/multistream_v3_middle_eff1024_oof.npz','output/joint_v3_middle_fast_oof.npz'),('late','output/multistream_v3_late_eff1024_oof.npz','output/joint_v3_late_fast_oof.npz')]
 for fold,bp,jp in specs:
  b=np.load(bp);j=np.load(jp);assert np.array_equal(b['sample_id'],j['sample_id']);ids=b['sample_id'];ix=pos.loc[ids].to_numpy();s=ds[ix];y=b['target'];mo=b['month'];base=unit(b['ens4_5_6']);joint=unit(j['ens4_5_6']);fixed=.4*base+.6*joint;w=.4+.5*s;gate=(1-w)*base+w*joint
  print('\n',fold,'domain q',np.quantile(s,[0,.25,.5,.75,1]),flush=True);print('base',stats(y,base,mo),'fixed',stats(y,fixed,mo),'gated',stats(y,gate,mo),flush=True)
  edges=np.quantile(s,np.linspace(0,1,6))
  for q in range(5):
   m=(s>=edges[q])&(s<(edges[q+1] if q<4 else edges[q+1]+1));print(' q',q+1,'n',m.sum(),'score',s[m].mean(),'base',stats(y[m],base[m],mo[m])[0],'fixed',stats(y[m],fixed[m],mo[m])[0],'gate',stats(y[m],gate[m],mo[m])[0],flush=True)
  rows.append((fold,*stats(y,base,mo),*stats(y,fixed,mo),*stats(y,gate,mo)));saved[f'{fold}_sample_id']=ids;saved[f'{fold}_target']=y;saved[f'{fold}_month']=mo;saved[f'{fold}_score']=s;saved[f'{fold}_base']=base;saved[f'{fold}_joint']=joint;saved[f'{fold}_fixed']=fixed;saved[f'{fold}_gated']=gate
 pd.DataFrame(rows,columns=['fold','base_cos','base_mean','base_min','base_std','fixed_cos','fixed_mean','fixed_min','fixed_std','gate_cos','gate_mean','gate_min','gate_std']).to_csv('output/domain_gated_v3.csv',index=False);np.savez('output/domain_gated_v3_oof.npz',**saved)
if __name__=='__main__':main()
