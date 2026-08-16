import numpy as np,pandas as pd
from sklearn.linear_model import Ridge

def u(x):x=np.asarray(x,float);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
def c(y,p):return float(u(y)@u(p))
def load(k):
 if k=='proxy':
  a=np.load('output/proxy_lgb_oof.npz');v=np.load('output/multistream_v3_proxy_oof.npz');r=np.load('output/realmlp_multiseed_proxy_oof.npz');j=np.load('output/joint_v3_proxy_fast_oof.npz');q=np.load('output/multires_self_proxy_oof.npz');return a['target'],a['month'],[a['prediction'],r['s42'],v['ens4_5_6'],j['ens4_5_6'],np.mean([q['ep5'],q['ep6'],q['ep7']],0)]
 if k=='middle':
  v=np.load('output/multistream_v3_middle_eff1024_oof.npz');r=np.load('output/realmlp_multiseed_rolling_oof.npz');j=np.load('output/joint_v3_middle_fast_oof.npz');q=np.load('output/multires_self_middle_oof.npz');return v['target'],v['month'],[r['middle_lgb'],r['middle_s42'],v['ens4_5_6'],j['ens4_5_6'],np.mean([q['ep5'],q['ep6'],q['ep7']],0)]
 v=np.load('output/multistream_v3_late_eff1024_oof.npz');r=np.load('output/realmlp_multiseed_rolling_oof.npz');j=np.load('output/joint_v3_late_fast_oof.npz');q=np.load('output/multires_self_late_oof.npz');return v['target'],v['month'],[r['late_lgb'],r['late_s42'],v['ens4_5_6'],j['ens4_5_6'],np.mean([q['ep5'],q['ep6'],q['ep7']],0)]
def main():
 names=['lgb','real','v3','joint','multi'];F={k:load(k) for k in ['proxy','middle','late']};X={};
 for k,(y,m,p) in F.items():X[k]=np.stack([u(a) for a in p],1);print(k,'base',c(y,.2*X[k][:,0]+.15*X[k][:,1]+.15*X[k][:,2]+.35*X[k][:,3]+.15*X[k][:,4]))
 for alpha in [1e-3,.01,.1,1,10,100]:
  print('alpha',alpha)
  for hold in F:
   train=[k for k in F if k!=hold];xt=np.concatenate([X[k] for k in train]);yt=np.concatenate([F[k][0] for k in train]);model=Ridge(alpha=alpha,fit_intercept=True).fit(xt,yt);p=model.predict(X[hold]);print(hold,'cos',c(F[hold][0],p),'coef',np.round(model.coef_,4))
 # fit loo average test weights and report fixed candidate-like average
 print('fixed OOF reference only; no test candidate generated')
if __name__=='__main__':main()
