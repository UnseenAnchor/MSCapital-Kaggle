"""Build a diagnostic candidate using a public-pack member substitution."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd

N=647896
PACK='output/public_lb0142/weights'
REPLACE=__import__('os').environ.get('REPLACE','v9_v3grid')
OUT=__import__('os').environ.get('OUT','output/candidate_public_anchor_variant.csv')
W5=np.array([.2]*5)
MEMBERS=['v9_big','v9_ctrl','v9_deep','v9_v3grid','v9_v3grid_big']
SELF={
'lgb':'output/submission_lgb_robust.csv',
'real':'output/submission_realmlp_v4_unit.csv',
'v3':'output/submission_multistream_coral_full_unit.csv',
'joint':'output/submission_joint_v3_fast_unit.csv',
'multires':'output/diagnostic_multires_self_full_unit.csv',
}
def unit(x):
 x=np.asarray(x,np.float64);x=x-x.mean();return x/(np.linalg.norm(x)+1e-12)
def pred(path):
 d=pd.read_csv(path).sort_values('sample_id');assert len(d)==N and np.array_equal(d.sample_id.to_numpy(),np.arange(N)),path
 x=d.prediction.to_numpy(np.float64);assert np.isfinite(x).all();return unit(x)
ids=np.arange(N)
ens=np.mean([pred(f'{PACK}/{m}/submission.csv') for m in MEMBERS],axis=0)
replacement=pred(f'{PACK}/{REPLACE}/submission.csv')
anchor=unit(.6*ens+.4*replacement)
selfp=unit(sum(w*pred(f) for w,f in zip([.176,.132,.132,.308,.132],SELF.values())) + .10*pred('output/submission_event_256_unit.csv') + .20*pred('output/submission_event_ssl_tt_full_supervised_unit.csv'))
final=unit(.6*anchor+.4*selfp)
out=pd.DataFrame({'sample_id':ids,'prediction':final});out.to_csv(OUT,index=False)
print('saved',OUT,'rows',len(out),'sha256',hashlib.sha256(Path(OUT).read_bytes()).hexdigest())
