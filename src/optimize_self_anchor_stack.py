"""Cross-fold fixed self-anchor stack from existing OOF/test predictions. No training or upload."""
import numpy as np,pandas as pd,itertools,hashlib

def u(x):x=np.asarray(x,np.float64);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
def c(y,p):return float(u(y)@u(p))
def metrics(y,p,m):
 a=[c(y[m==x],p[m==x]) for x in np.unique(m)];return c(y,p),float(np.mean(a)),float(np.min(a)),float(np.std(a))
def load_fold(name):
 if name=='proxy':
  l=np.load('output/proxy_lgb_oof.npz');v=np.load('output/multistream_v3_proxy_oof.npz');j=np.load('output/joint_v3_proxy_fast_oof.npz');r=np.load('output/realmlp_multiseed_proxy_oof.npz');e=np.load('output/event_direct_rolling_oof.npz');return {'ids':l['sample_id'],'y':l['target'],'m':l['month'],'lgb':l['prediction'],'real':r['s42'],'v3':v['ens4_5_6'],'joint':j['ens4_5_6'],'event':e['proxy_event']}
 if name=='middle':
  v=np.load('output/multistream_v3_middle_eff1024_oof.npz');j=np.load('output/joint_v3_middle_fast_oof.npz');r=np.load('output/realmlp_multiseed_rolling_oof.npz');e=np.load('output/event_direct_rolling_oof.npz');return {'ids':v['sample_id'],'y':v['target'],'m':pd.read_feather('data/train/label.feather').set_index('sample_id').loc[v['sample_id'],'month'].to_numpy(),'lgb':r['middle_lgb'],'real':r['middle_s42'],'v3':v['ens4_5_6'],'joint':j['ens4_5_6'],'event':e['middle_event']}
 z=np.load('output/multistream_v3_late_eff1024_oof.npz');j=np.load('output/joint_v3_late_fast_oof.npz');r=np.load('output/realmlp_multiseed_rolling_oof.npz');e=np.load('output/event_direct_rolling_oof.npz');return {'ids':z['sample_id'],'y':z['target'],'m':z['month'],'lgb':r['late_lgb'],'real':r['late_s42'],'v3':z['ens4_5_6'],'joint':j['ens4_5_6'],'event':e['late_event']}
def main():
 names=['lgb','real','v3','joint','event'];folds={x:load_fold(x) for x in ['proxy','middle','late']};
 for f,z in folds.items():
  z['p']=np.array([u(z[n]) for n in names]);print(f,{n:metrics(z['y'],z[n],z['m']) for n in names})
 best=[]
 # fixed weights in 0.02 grid; optimize robust cross-fold score, not one fold.
 grid=np.arange(0,1.001,.05)
 for a in grid:
  for b in grid:
   for d in grid:
    for e in grid:
     f=1-a-b-d-e
     if f < -1e-9:continue
     w=np.array([a,b,d,e,f]);vals=[]
     for z in folds.values():vals.append(metrics(z['y'],w@z['p'],z['m']))
     glob=np.array([x[0] for x in vals]);means=np.array([x[1] for x in vals]);mins=np.array([x[2] for x in vals]);score=.4*glob.mean()+.3*means.mean()+.2*mins.mean()-.1*glob.std();best.append((score,w,vals))
 best.sort(key=lambda x:x[0],reverse=True);print('\nTOP ROBUST');
 for row in best[:20]:print('score',row[0],'w',dict(zip(names,row[1].round(2))),'folds',row[2])
 # Test predictions for the top 5 distinct candidates.
 test_paths={'lgb':'output/submission_lgb_robust.csv','real':'output/submission_realmlp_v4_unit.csv','v3':'output/submission_multistream_v3_eff1024_unit.csv','joint':'output/submission_joint_v3_fast_unit.csv','event':'output/diagnostic_event_direct_full_unit.csv'};s=pd.read_csv('data/submission.csv');tp=np.array([u(pd.read_csv(test_paths[n]).sort_values('sample_id').prediction) for n in names]);ref=u(pd.read_csv('research/lb0142/submission_ref_lb0142.csv').sort_values('sample_id').prediction);cur=u(pd.read_csv('output/candidate_ours40_public142_60.csv').sort_values('sample_id').prediction);seen=set()
 for _,w,vals in best:
  key=tuple(np.round(w,2));
  if key in seen:continue
  seen.add(key);selfp=w@tp;final=.4*selfp+.6*ref;print('TEST w',dict(zip(names,w.round(2))),'corr_cur',u(final)@cur,'corr_ref',u(selfp)@ref,'self_std',selfp.std());
  if len(seen)>=10:break
if __name__=='__main__':main()
