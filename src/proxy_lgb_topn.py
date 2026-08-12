"""Leakage-safe Top-N ablation using importance learned only on months 0-44."""
import numpy as np,pandas as pd,lightgbm as lgb
from proxy_lgb_feature_select import load_combined,cosine
TOPS=(64,96,128,160,224,320,434)
def main():
 lab,X=load_combined();imp=pd.read_csv('output/proxy_lgb_trainonly_importance.csv');mo=lab.month.to_numpy();tr=mo<45;va=mo>=45;y=lab.target.to_numpy();rows=[]
 params=dict(objective='regression',metric='l2',learning_rate=.025,num_leaves=48,min_data_in_leaf=500,feature_fraction=.8,bagging_fraction=.8,bagging_freq=1,lambda_l2=10,max_bin=63,force_col_wise=True,verbosity=-1,n_jobs=10,seed=13)
 for n in TOPS:
  cols=imp.feature.head(n).tolist();m=lgb.train(params,lgb.Dataset(X.loc[tr,cols],label=y[tr]),num_boost_round=600);p=m.predict(X.loc[va,cols]);monthly=[]
  for month in range(45,71):
   mm=mo[va]==month;monthly.append(cosine(y[va][mm],p[mm]))
  row=(n,cosine(y[va],p),float(np.mean(monthly)),float(np.min(monthly)),float(np.std(monthly)));rows.append(row);print(row,flush=True)
 pd.DataFrame(rows,columns=['topn','proxy_cos','month_mean','month_min','month_std']).to_csv('output/proxy_lgb_topn.csv',index=False)
if __name__=='__main__':main()
