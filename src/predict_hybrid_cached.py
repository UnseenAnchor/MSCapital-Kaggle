import os,numpy as np,pandas as pd,torch
from train_gru_cached import CACHE,DEVICE,BATCH,SCALE_Y,CacheDataset
from train_hybrid_cached import DS,Net
# base test cache
ids=np.load(CACHE+'/test_ids.npy');m=np.load(CACHE+'/test_market.npy',mmap_mode='r');o=np.load(CACHE+'/test_order.npy',mmap_mode='r');x=np.load(CACHE+'/test_tx.npy',mmap_mode='r')
# train/test tabular features, use train statistics
lab=pd.read_feather('data/train/label.feather'); trids=np.load(CACHE+'/train_ids.npy');
def read(split,ids):
 d=pd.DataFrame({'sample_id':ids})
 for n in ['market','order','transaction']:
  for v in ['','_v2']:
   p=f'features/{n}_{split}{v}.parquet'
   if os.path.exists(p):d=d.merge(pd.read_parquet(p),on='sample_id',how='left')
 return d.set_index('sample_id')
train=read('train',trids); test=read('test',ids); cols=[z for z in train if z not in ('month','target')]
a=train[cols].replace([np.inf,-np.inf],np.nan).to_numpy(np.float32);b=test[cols].replace([np.inf,-np.inf],np.nan).to_numpy(np.float32);mu=np.nanmean(a,0);sd=np.maximum(np.nanstd(a,0),1e-6);b=np.where(np.isnan(b),mu,b);b=np.clip((b-mu)/sd,-10,10).astype(np.float32)
ds=DS(m,o,x,b,np.zeros(len(ids),np.float32),np.arange(len(ids)));dl=torch.utils.data.DataLoader(ds,batch_size=BATCH*2,shuffle=False)
q=Net(len(cols)).to(DEVICE);q.load_state_dict(torch.load('output/hybrid_best.pt',map_location=DEVICE));q.eval();z=[]
with torch.no_grad():
 for bm,bo,bx,bt,by in dl:z.append(q(bm.to(DEVICE),bo.to(DEVICE),bx.to(DEVICE),bt.to(DEVICE)).cpu().numpy()/SCALE_Y)
p=np.concatenate(z);pd.DataFrame({'sample_id':ids,'prediction':p}).to_csv('output/submission_hybrid_cached.csv',index=False);print('done',len(p),p.mean(),p.std())
