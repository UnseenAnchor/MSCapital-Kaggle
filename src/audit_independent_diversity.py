import numpy as np,pandas as pd,itertools

def u(x):x=np.asarray(x,float);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
def c(y,p):return float(u(y)@u(p))
def met(y,p,m):
 z=[c(y[m==q],p[m==q]) for q in np.unique(m)];return c(y,p),np.mean(z),min(z)
def fold(k):
 if k=='proxy':
  a=np.load('output/proxy_lgb_oof.npz');v=np.load('output/multistream_v3_proxy_oof.npz');r=np.load('output/realmlp_multiseed_proxy_oof.npz');j=np.load('output/joint_v3_proxy_fast_oof.npz');q=np.load('output/multires_self_proxy_oof.npz');b=np.load('output/v3big_self_proxy_fast_oof.npz');return a['target'],a['month'],[a['prediction'],r['s42'],v['ens4_5_6'],j['ens4_5_6'],np.mean([q['ep5'],q['ep6'],q['ep7']],0),np.mean([b['ep5'],b['ep6'],b['ep7']],0)]
 if k=='middle':
  v=np.load('output/multistream_v3_middle_eff1024_oof.npz');r=np.load('output/realmlp_multiseed_rolling_oof.npz');j=np.load('output/joint_v3_middle_fast_oof.npz');q=np.load('output/multires_self_middle_oof.npz');b=np.load('output/multistream_v3big_middle_fast_oof.npz');return v['target'],pd.read_feather('data/train/label.feather').set_index('sample_id').loc[v['sample_id'],'month'].to_numpy(),[r['middle_lgb'],r['middle_s42'],v['ens4_5_6'],j['ens4_5_6'],np.mean([q['ep5'],q['ep6'],q['ep7']],0),b['ens4_5_6']]
 v=np.load('output/multistream_v3_late_eff1024_oof.npz');r=np.load('output/realmlp_multiseed_rolling_oof.npz');j=np.load('output/joint_v3_late_fast_oof.npz');q=np.load('output/multires_self_late_oof.npz');b=np.load('output/multistream_v3big_late_fast_oof.npz');return v['target'],v['month'],[r['late_lgb'],r['late_s42'],v['ens4_5_6'],j['ens4_5_6'],np.mean([q['ep5'],q['ep6'],q['ep7']],0),b['ens4_5_6']]
def main():
 names=['lgb','real','v3','joint','multi','v3big'];F={k:fold(k) for k in ['proxy','middle','late']}
 for k,(y,m,p) in F.items():print(k,{n:tuple(round(x,6) for x in met(y,u(a),m)) for n,a in zip(names,p)})
 best=[]
 for cuts in itertools.combinations(range(21+len(names)-1),len(names)-1):
  pts=(-1,)+cuts+(21+len(names)-1,);w=np.array([pts[i+1]-pts[i]-1 for i in range(len(names))])/20
  vals=[met(y,w@np.array([u(a) for a in p]),m) for y,m,p in F.values()];g=np.array([x[0] for x in vals]);avg=np.array([x[1] for x in vals]);lo=np.array([x[2] for x in vals]);score=.45*g.mean()+.35*avg.mean()+.2*lo.mean()-.1*g.std();best.append((score,w,vals))
 best.sort(key=lambda x:x[0],reverse=True);print('TOP')
 for x in best[:20]:print(round(x[0],6),dict(zip(names,x[1].round(2))),[tuple(round(a,6) for a in z) for z in x[2]])
 paths=['output/submission_lgb_robust.csv','output/submission_realmlp_v4_unit.csv','output/submission_multistream_v3_eff1024_unit.csv','output/submission_joint_v3_fast_unit.csv','output/diagnostic_multires_self_full_unit.csv','output/submission_multistream_v3big_stable_ep6.csv'];T=np.array([u(pd.read_csv(p).sort_values('sample_id').prediction) for p in paths]);ref=u(pd.read_csv('research/lb0142/submission_ref_lb0142.csv').sort_values('sample_id').prediction);cur=u(pd.read_csv('output/candidate_independent_stack_public60_40.csv').sort_values('sample_id').prediction);seen=set()
 for _,w,_ in best:
  key=tuple(w.round(2));
  if key in seen:continue
  seen.add(key);selfp=w@T;final=u(.6*ref+.4*selfp);print('TEST',dict(zip(names,w.round(2))),'new/0145',round(final@cur,6),'self/ref',round(selfp@ref,6))
  if len(seen)>=12:break
if __name__=='__main__':main()
