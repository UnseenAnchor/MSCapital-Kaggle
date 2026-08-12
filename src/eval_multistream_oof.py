"""Evaluate selected v2 checkpoints on middle/late folds and save OOF predictions. No submission."""
import os
os.environ.setdefault('BATCH','1024')
import numpy as np,pandas as pd,torch
from train_multistream_grid import arrays,DS,Net,DEVICE,BATCH,pred,cos
ROOT='features/grid_v2'
def unit(x):
 x=np.asarray(x,np.float64);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
def infer(paths,norm_path,va,A,y):
 z=np.load(norm_path);norm={k:(z[k+'_mean'],z[k+'_std']) for k in ('market','tx','order')}
 dl=torch.utils.data.DataLoader(DS(A,va,norm,y),BATCH,shuffle=False,num_workers=0,pin_memory=True)
 ps=[]
 for path in paths:
  q=Net().to(DEVICE);q.load_state_dict(torch.load(path,map_location=DEVICE));p,t=pred(q,dl);ps.append(p);print(path,'center',cos(t,p,True),flush=True);del q;torch.cuda.empty_cache()
 return np.mean([unit(p) for p in ps],axis=0)
def main():
 lab=pd.read_feather('data/train/label.feather').sort_values('sample_id');y=lab.target.to_numpy(np.float32);mo=lab.month.to_numpy();A=arrays('train',len(lab));out={}
 va=np.flatnonzero((mo>=51)&(mo<61));out['middle_sample_id']=lab.sample_id.to_numpy()[va];out['middle_target']=y[va]
 specs={
  's42':([f'output/multistream_mid_ep{e}.pt' for e in (4,5,6)],f'{ROOT}/norm_stats_multistream_mid.npz'),
  's13':([f'output/multistream_mid_s13_ep{e}.pt' for e in (5,6)],f'{ROOT}/norm_stats_multistream_mid_s13.npz'),
  's77':([f'output/multistream_mid_s77_ep{e}.pt' for e in (4,5,6)],f'{ROOT}/norm_stats_multistream_mid_s77.npz')}
 for name,(paths,npth) in specs.items():out['middle_'+name]=infer(paths,npth,va,A,y);print('middle',name,cos(y[va],out['middle_'+name],True),flush=True)
 out['middle_avg3']=np.mean([out['middle_'+s] for s in specs],axis=0);print('middle avg3',cos(y[va],out['middle_avg3'],True),flush=True)
 va=np.flatnonzero((mo>=62)&(mo<71));out['late_sample_id']=lab.sample_id.to_numpy()[va];out['late_target']=y[va]
 old=np.load('output/multistream_val_preds.npz');out['late_s42']=np.mean([unit(old[f'ep{e}']) for e in (3,5,8,10)],axis=0);print('late s42',cos(y[va],out['late_s42'],True),flush=True)
 paths=[f'output/multistream_late_s13_ep{e}.pt' for e in (1,2,3)];out['late_s13']=infer(paths,f'{ROOT}/norm_stats_multistream_late_s13.npz',va,A,y);print('late s13',cos(y[va],out['late_s13'],True),flush=True)
 out['late_avg2']=(out['late_s42']+out['late_s13'])/2;print('late avg2',cos(y[va],out['late_avg2'],True),flush=True)
 np.savez('output/multistream_v2_multiseed_oof.npz',**out)
if __name__=='__main__':main()
