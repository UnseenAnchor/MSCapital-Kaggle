"""Save and compare validation predictions from selected MultiStream epochs."""
import numpy as np,pandas as pd,torch
from train_multistream_grid import arrays,norm_stats,DS,Net,DEVICE,BATCH,ROOT,cos,pred
lab=pd.read_feather('data/train/label.feather').sort_values('sample_id');n=len(lab);A=arrays('train',n);mo=lab.month.to_numpy();y=lab.target.to_numpy(np.float32);tr=np.flatnonzero(mo<62);va=np.flatnonzero(mo>=62)
z=np.load(ROOT+'/norm_stats.npz');norm={k:(z[k+'_mean'],z[k+'_std']) for k in ['market','tx','order']};dl=torch.utils.data.DataLoader(DS(A,va,norm,y),BATCH*2,shuffle=False,num_workers=0,pin_memory=True);out={'sample_id':lab.sample_id.to_numpy()[va],'target':y[va]}
for ep in [3,5,8,10]:
 q=Net().to(DEVICE);q.load_state_dict(torch.load(f'output/multistream_ep{ep}.pt',map_location=DEVICE));p,t=pred(q,dl);out[f'ep{ep}']=p;print(ep,'raw',cos(t,p),'center',cos(t,p,True))
for a in [.25,.5,.75]:
 p=a*out['ep5']+(1-a)*out['ep8'];print('ep5 weight',a,'raw',cos(y[va],p),'center',cos(y[va],p,True))
np.savez('output/multistream_val_preds.npz',**out)
