"""Adversarial train-vs-test validation and test-similarity OOF audit. Never submits."""
import gc,time,numpy as np,pandas as pd,lightgbm as lgb
from sklearn.metrics import roc_auc_score
from proxy_lgb_feature_select import load_combined,cosine
SEED=2026;N_PER_DOMAIN=350000

def load_test(cols):
 base=pd.read_csv('data/submission.csv')[['sample_id']].sort_values('sample_id').reset_index(drop=True);d=base.copy()
 for name in ('market','order','transaction'):
  for suffix in ('','_v2'):d=d.merge(pd.read_parquet(f'features/{name}_test{suffix}.parquet'),on='sample_id',how='left')
 micro=pd.read_parquet('features/micro_v3/test.parquet').sort_values('sample_id').reset_index(drop=True);old=[c for c in d if c!='sample_id'];new=[c for c in micro if c!='sample_id'];X=d[old].join(micro[new].add_prefix('v3_'));return base.sample_id.to_numpy(),X[cols]
def unit(x):x=np.asarray(x,np.float64);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
def c(y,p):return float(unit(y)@unit(p))
def main():
 t0=time.time();lab,Xtr=load_combined();cols=list(Xtr.columns);test_ids,Xte=load_test(cols);rng=np.random.default_rng(SEED);it=rng.choice(len(Xtr),N_PER_DOMAIN,replace=False);ie=rng.choice(len(Xte),N_PER_DOMAIN,replace=False)
 X=pd.concat([Xtr.iloc[it],Xte.iloc[ie]],ignore_index=True);y=np.r_[np.zeros(len(it),np.int8),np.ones(len(ie),np.int8)];key=np.r_[lab.sample_id.to_numpy()[it],test_ids[ie]];va=(key*2654435761%10)<2;tr=~va
 params=dict(objective='binary',metric='auc',learning_rate=.04,num_leaves=32,min_data_in_leaf=500,feature_fraction=.75,bagging_fraction=.8,bagging_freq=1,lambda_l2=10,max_bin=63,force_col_wise=True,verbosity=-1,n_jobs=10,seed=SEED)
 m=lgb.train(params,lgb.Dataset(X.loc[tr],label=y[tr]),num_boost_round=500);pv=m.predict(X.loc[va]);auc=roc_auc_score(y[va],pv);print('domain AUC',auc,'fit',tr.sum(),'valid',va.sum(),flush=True)
 imp=pd.DataFrame({'feature':cols,'gain':m.feature_importance('gain')}).sort_values('gain',ascending=False);imp.to_csv('output/domain_feature_importance.csv',index=False);print('TOP25\n',imp.head(25).to_string(index=False),flush=True)
 # Refit balanced classifier, then score all rows in chunks.
 mf=lgb.train(params,lgb.Dataset(X,label=y),num_boost_round=500)
 def predict_chunks(df,n=200000):return np.concatenate([mf.predict(df.iloc[i:i+n]) for i in range(0,len(df),n)])
 ptr=predict_chunks(Xtr);pte=predict_chunks(Xte);np.savez('output/domain_scores.npz',train_sample_id=lab.sample_id.to_numpy(),train_month=lab.month.to_numpy(),train_score=ptr,test_sample_id=test_ids,test_score=pte)
 monthly=pd.DataFrame({'month':lab.month,'score':ptr}).groupby('month').score.agg(['mean','median','std']);monthly.to_csv('output/domain_monthly.csv');print('monthly\n',monthly.to_string(),flush=True);print('score train/test quantiles',np.quantile(ptr,[0,.1,.25,.5,.75,.9,1]),np.quantile(pte,[0,.1,.25,.5,.75,.9,1]),flush=True)
 # Audit existing proxy predictions by test-similarity quintile.
 zl=np.load('output/proxy_lgb_oof.npz');zb=np.load('output/multistream_v3_proxy_oof.npz');zj=np.load('output/joint_v3_proxy_oof.npz');zr=np.load('output/realmlp_multiseed_proxy_oof.npz');ids=zl['sample_id'];pos=pd.Series(np.arange(len(lab)),index=lab.sample_id).loc[ids].to_numpy();s=ptr[pos];target=zl['target'];models={'lgb':zl['prediction'],'v3':zb['ens4_5_6'],'joint':zj['ens4_5_6'],'real':zr['s42'],'v3mix':.4*unit(zb['ens4_5_6'])+.6*unit(zj['ens4_5_6'])};edges=np.quantile(s,np.linspace(0,1,6));rows=[]
 for qi in range(5):
  mask=(s>=edges[qi])&(s<edges[qi+1] if qi<4 else s<=edges[qi+1]);row={'quintile':qi+1,'n':mask.sum(),'score_mean':s[mask].mean()};row.update({k:c(target[mask],p[mask]) for k,p in models.items()});rows.append(row)
 out=pd.DataFrame(rows);out.to_csv('output/domain_oof_quintiles.csv',index=False);print('OOF by test similarity\n',out.to_string(index=False),flush=True);print('sec',time.time()-t0)
if __name__=='__main__':main()
