"""Robust unit blend analysis on common middle/late OOF predictions."""
import numpy as np,pandas as pd

def unit(x):
 x=np.asarray(x,np.float64);x=x-x.mean();return x/(np.linalg.norm(x)+1e-12)
def cos(y,p):return float(unit(y)@unit(p))
def main():
 lab=pd.read_feather('data/train/label.feather').sort_values('sample_id').reset_index(drop=True);mo=lab.month.to_numpy();rows=[];monthly=[]
 zl=np.load('output/rolling_micro_lgb_preds.npz');zo=np.load('output/rolling_lgb_seed_preds.npz');zm=np.load('output/multistream_v2_multiseed_oof.npz')
 for fold,lo,hi,mskey in [('middle',51,61,'middle_avg3'),('late',62,71,'late_s42')]:
  mask=(mo>=lo)&(mo<hi);y=lab.target.to_numpy()[mask];old=zo[f'{fold}_avg3'];new=zl[f'{fold}_combined'];ms=zm[mskey]
  members=[unit(old),unit(new),unit(ms)]
  for wold in np.arange(0,.31,.05):
   for wnew in np.arange(.1,.71,.05):
    wms=1-wold-wnew
    if wms>=0:rows.append((fold,wold,wnew,wms,cos(y,wold*members[0]+wnew*members[1]+wms*members[2])))
  # conservative fixed candidates, and monthly stability
  for name,w in [('public_style',(0.1,0.0,0.9)),('balanced',(0.1,.35,.55)),('tab_heavy',(0.1,.50,.40)),('new_lgb_only',(0,.75,.25))]:
   p=sum(a*x for a,x in zip(w,members));scores=[]
   for month in range(lo,hi):
    mm=mo[mask]==month;scores.append(cos(y[mm],p[mm]));monthly.append((fold,name,month,scores[-1]))
   print(fold,name,'fold',cos(y,p),'month_mean',np.mean(scores),'month_min',np.min(scores),flush=True)
 r=pd.DataFrame(rows,columns=['fold','old','new','ms','cosine']);pivot=r.pivot_table(index=['old','new','ms'],columns='fold',values='cosine').dropna();pivot['mean']=pivot.mean(axis=1);pivot['min']=pivot[['middle','late']].min(axis=1);pivot['gap']=(pivot.late-pivot.middle).abs();pivot['robust']=pivot['mean']-.25*pivot['gap'];print('\nTOP ROBUST\n',pivot.sort_values('robust',ascending=False).head(15).to_string());pivot.reset_index().to_csv('output/oof_blend_grid.csv',index=False);pd.DataFrame(monthly,columns=['fold','candidate','month','cosine']).to_csv('output/oof_blend_monthly.csv',index=False)
if __name__=='__main__':main()
