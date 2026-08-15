"""Optimize a train-only independent self anchor including multi-resolution."""
import numpy as np,pandas as pd,itertools

def u(x):x=np.asarray(x,np.float64);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
def c(y,p):return float(u(y)@u(p))
def met(y,p,m):
 v=[c(y[m==q],p[m==q]) for q in np.unique(m)];return c(y,p),float(np.mean(v)),float(min(v)),float(np.std(v))
def load(kind):
 if kind=='proxy':
  x=np.load('output/proxy_lgb_oof.npz');v=np.load('output/multistream_v3_proxy_oof.npz');r=np.load('output/realmlp_multiseed_proxy_oof.npz');j=np.load('output/joint_v3_proxy_fast_oof.npz');q=np.load('output/multires_self_proxy_oof.npz');return x['sample_id'],x['target'],x['month'],[x['prediction'],r['s42'],v['ens4_5_6'],j['ens4_5_6'],np.mean([q['ep5'],q['ep6'],q['ep7']],0)]
 if kind=='middle':
  v=np.load('output/multistream_v3_middle_eff1024_oof.npz');r=np.load('output/realmlp_multiseed_rolling_oof.npz');j=np.load('output/joint_v3_middle_fast_oof.npz');q=np.load('output/multires_self_middle_oof.npz');ids=v['sample_id'];months=pd.read_feather('data/train/label.feather').set_index('sample_id').loc[ids,'month'].to_numpy();return ids,v['target'],months,[r['middle_lgb'],r['middle_s42'],v['ens4_5_6'],j['ens4_5_6'],np.mean([q['ep5'],q['ep6'],q['ep7']],0)]
 v=np.load('output/multistream_v3_late_eff1024_oof.npz');r=np.load('output/realmlp_multiseed_rolling_oof.npz');j=np.load('output/joint_v3_late_fast_oof.npz');q=np.load('output/multires_self_late_oof.npz');return v['sample_id'],v['target'],v['month'],[r['late_lgb'],r['late_s42'],v['ens4_5_6'],j['ens4_5_6'],np.mean([q['ep5'],q['ep6'],q['ep7']],0)]
def main():
 names=['lgb','real','v3','joint','multires'];folds={k:load(k) for k in ['proxy','middle','late']};A={}
 for k,(ids,y,m,ps) in folds.items():A[k]=(y,m,np.array([u(p) for p in ps]));print(k,{n:met(y,u(p),m) for n,p in zip(names,ps)})
 best=[]
 for w in itertools.product(np.arange(0,1.0001,.05),repeat=4):
  last=1-sum(w)
  if last < -1e-9:continue
  ww=np.array((*w,last));vals=[met(y,ww@p,m) for y,m,p in A.values()];g=np.array([x[0] for x in vals]);av=np.array([x[1] for x in vals]);lo=np.array([x[2] for x in vals]);score=.4*g.mean()+.3*av.mean()+.2*lo.mean()-.1*g.std();best.append((score,ww,vals))
 best.sort(key=lambda x:x[0],reverse=True);print('TOP')
 for score,w,vals in best[:30]:print(round(score,6),dict(zip(names,w.round(2))),[tuple(round(x,6) for x in z) for z in vals])
 paths=['output/submission_lgb_robust.csv','output/submission_realmlp_v4_unit.csv','output/submission_multistream_v3_eff1024_unit.csv','output/submission_joint_v3_fast_unit.csv','output/diagnostic_multires_self_full_unit.csv'];tp=np.array([u(pd.read_csv(p).sort_values('sample_id').prediction) for p in paths]);ref=u(pd.read_csv('research/lb0142/submission_ref_lb0142.csv').sort_values('sample_id').prediction);cur=u(pd.read_csv('output/candidate_ours40_public142_60.csv').sort_values('sample_id').prediction);seen=set()
 for _,w,_ in best:
  key=tuple(w.round(2));
  if key in seen:continue
  seen.add(key);selfp=w@tp;final=u(.6*ref+.4*selfp);print('TEST',dict(zip(names,w.round(2))),'self/ref',round(u(selfp)@ref,6),'new/current',round(final@cur,6),'selfstd',round(selfp.std(),8))
  if len(seen)>=15:break
if __name__=='__main__':main()
