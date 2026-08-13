"""Proxy gate for v3 seed13 using fixed checkpoint 4/5/6 ensemble."""
import os
os.environ.update(GRID_ROOT='features/grid_v3',GRID_VERSION='v3',MARKET_LEN='400',FLOW_LEN='120',D_MODEL='64',N_LAYERS='2',BATCH='128')
import numpy as np,pandas as pd,torch
from train_multistream_grid import arrays,DS,Net,DEVICE,ROOT,pred

def unit(x):x=np.asarray(x,np.float64);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
def stats(y,p,mo):
 v=[float(unit(y[mo==m])@unit(p[mo==m])) for m in np.unique(mo)];return float(unit(y)@unit(p)),float(np.mean(v)),float(np.min(v)),float(np.std(v))
def main():
 lab=pd.read_feather('data/train/label.feather').sort_values('sample_id');mo=lab.month.to_numpy();y=lab.target.to_numpy(np.float32);va=np.flatnonzero(mo>=45);A=arrays('train',len(lab));prefix='multistream_v3_proxy_eff1024_s13';z=np.load(f'{ROOT}/norm_stats_{prefix}.npz');norm={k:(z[k+'_mean'],z[k+'_std']) for k in ('market','tx','order')};dl=torch.utils.data.DataLoader(DS(A,va,norm,y),256,shuffle=False,num_workers=0,pin_memory=True);o={'sample_id':lab.sample_id.to_numpy()[va],'target':y[va],'month':mo[va]}
 for ep in (4,5,6):
  q=Net().to(DEVICE);q.load_state_dict(torch.load(f'output/{prefix}_ep{ep}.pt',map_location=DEVICE));p,t=pred(q,dl);o[f'ep{ep}']=p;print('s13 ep',ep,stats(t,p,mo[va]),flush=True);del q;torch.cuda.empty_cache()
 s13=np.mean([unit(o[f'ep{e}']) for e in (4,5,6)],0);z42=np.load('output/multistream_v3_proxy_oof.npz');assert np.array_equal(z42['sample_id'],o['sample_id']);s42=z42['ens4_5_6'];print('s42',stats(y[va],s42,mo[va]),'s13',stats(y[va],s13,mo[va]),'corr',float(unit(s42)@unit(s13)),flush=True)
 for w in (0,.2,.4,.5,.6,.8,1.):print('s13_weight',w,stats(y[va],(1-w)*unit(s42)+w*unit(s13),mo[va]),flush=True)
 o['s13']=s13;o['s42']=s42;o['avg']=.5*unit(s42)+.5*unit(s13);np.savez('output/multistream_v3_multiseed_proxy_oof.npz',**o)
if __name__=='__main__':main()
