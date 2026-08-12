"""Predict test with selected high-resolution MultiStream checkpoints.
Creates member predictions and their unit-normalized checkpoint ensemble. No Kaggle upload.
"""
import os,numpy as np,pandas as pd,torch
from train_multistream_grid import arrays,DS,Net,DEVICE,BATCH,ROOT,pred
CHECKPOINTS=os.environ.get('CHECKPOINTS','output/multistream_ep5.pt,output/multistream_ep8.pt').split(',')
OUT=os.environ.get('OUT','output/submission_multistream_unit.csv'); NORM_PREFIX=os.environ.get('NORM_PREFIX','multistream')
def unit(x):x=x-x.mean();return x/(np.linalg.norm(x)+1e-12)
def main():
 ids=pd.read_csv('data/submission.csv').sample_id.to_numpy();A=arrays('test',len(ids));preferred=ROOT+f'/norm_stats_{NORM_PREFIX}.npz';norm_path=preferred if os.path.exists(preferred) else ROOT+'/norm_stats.npz';z=np.load(norm_path);norm={k:(z[k+'_mean'],z[k+'_std']) for k in ['market','tx','order']};dl=torch.utils.data.DataLoader(DS(A,np.arange(len(ids)),norm),BATCH*2,shuffle=False,num_workers=0,pin_memory=True);ps=[]
 for path in CHECKPOINTS:
  q=Net().to(DEVICE);q.load_state_dict(torch.load(path,map_location=DEVICE));p,_=pred(q,dl);ps.append(p);member=path.rsplit('/',1)[-1].replace('.pt','');pd.DataFrame({'sample_id':ids,'prediction':p}).to_csv(f'output/submission_{member}.csv',index=False);print(member,p.mean(),p.std())
 p=np.mean([unit(x) for x in ps],0);pd.DataFrame({'sample_id':ids,'prediction':p}).to_csv(OUT,index=False);print('ensemble',OUT,len(p),p.mean(),p.std())
if __name__=='__main__':main()
