"""Reconstruct pure ens5 and v10 components from the audited public LB0.142 pack."""
from pathlib import Path
import hashlib,numpy as np,pandas as pd
ROOT=Path('research/lb0142');OUT=Path('output')
NAMES=['v9_big','v9_ctrl','v9_deep','v9_v3grid','v9_v3grid_big']
def unit(x):x=np.asarray(x,np.float64);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
def main():
 sub=pd.read_csv('data/submission.csv');members=[]
 for name in NAMES:
  d=pd.read_csv(ROOT/'weights'/name/'submission.csv').sort_values('sample_id').reset_index(drop=True);assert np.array_equal(d.sample_id,sub.sample_id);members.append(unit(d.prediction))
 v=pd.read_csv(ROOT/'weights/v10/submission.csv').sort_values('sample_id').reset_index(drop=True);assert np.array_equal(v.sample_id,sub.sample_id);v10=unit(v.prediction);ens5=unit(np.mean(members,axis=0));ref=pd.read_csv(ROOT/'submission_ref_lb0142.csv').sort_values('sample_id').reset_index(drop=True);cur=pd.read_csv(OUT/'candidate_ours40_public142_60.csv')
 for name,p in [('public_ens5',ens5),('public_v10',v10)]:
  dest=OUT/f'diagnostic_{name}_unit.csv';pd.DataFrame({'sample_id':sub.sample_id,'prediction':p}).to_csv(dest,index=False);h=hashlib.sha256(dest.read_bytes()).hexdigest();print(name,'corr_current',float(unit(cur.prediction)@p),'corr_ref',float(unit(ref.prediction)@p),'sha256',h)
 print('ens5_v10_corr',float(ens5@v10),'ens5_member_norm',float(np.linalg.norm(np.mean(members,axis=0))))
if __name__=='__main__':main()
