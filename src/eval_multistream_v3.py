import os,numpy as np,pandas as pd,torch
# configure module before import
os.environ.update(GRID_ROOT='features/grid_v3',GRID_VERSION='v3',MARKET_LEN='400',FLOW_LEN='120',D_MODEL='96',N_LAYERS='3',BATCH='128')
from train_multistream_grid import arrays,DS,Net,DEVICE,BATCH,ROOT,pred,cos
lab=pd.read_feather('data/train/label.feather').sort_values('sample_id');n=len(lab);A=arrays('train',n);mo=lab.month.to_numpy();y=lab.target.to_numpy(np.float32);va=np.flatnonzero(mo>=62);z=np.load(ROOT+'/norm_stats_multistream_v3big_stable.npz');norm={k:(z[k+'_mean'],z[k+'_std']) for k in ['market','tx','order']};dl=torch.utils.data.DataLoader(DS(A,va,norm,y),BATCH*2,shuffle=False,num_workers=0,pin_memory=True);out={'sample_id':lab.sample_id.to_numpy()[va],'target':y[va]}
for ep in [4,5,6]:
 q=Net().to(DEVICE);q.load_state_dict(torch.load(f'output/multistream_v3big_stable_ep{ep}.pt',map_location=DEVICE));p,t=pred(q,dl);out[f'ep{ep}']=p;print(ep,'raw',cos(t,p),'center',cos(t,p,True),'std',p.std())
np.savez('output/multistream_v3_val_preds.npz',**out)
