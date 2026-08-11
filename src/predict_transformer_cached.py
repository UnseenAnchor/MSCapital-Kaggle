import numpy as np,pandas as pd,torch
from train_gru_cached import CacheDataset,CACHE,DEVICE,BATCH,SCALE_Y
from train_transformer_cached import Net
ids=np.load(CACHE+'/test_ids.npy');m=np.load(CACHE+'/test_market.npy',mmap_mode='r');o=np.load(CACHE+'/test_order.npy',mmap_mode='r');x=np.load(CACHE+'/test_tx.npy',mmap_mode='r')
ds=CacheDataset(m,o,x,np.zeros(len(ids),np.float32),np.arange(len(ids)));dl=torch.utils.data.DataLoader(ds,batch_size=BATCH*2,shuffle=False)
q=Net().to(DEVICE);q.load_state_dict(torch.load('output/transformer_best.pt',map_location=DEVICE));q.eval();z=[]
with torch.no_grad():
 for bm,bo,bx,by in dl:z.append(q(bm.to(DEVICE),bo.to(DEVICE),bx.to(DEVICE)).cpu().numpy()/SCALE_Y)
pd.DataFrame({'sample_id':ids,'prediction':np.concatenate(z)}).to_csv('output/submission_transformer_cached.csv',index=False)
print('done',len(ids),np.concatenate(z).mean(),np.concatenate(z).std())
