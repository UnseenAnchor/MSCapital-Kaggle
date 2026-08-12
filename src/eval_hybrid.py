import os,numpy as np,pandas as pd,torch,lightgbm as lgb
from train_gru_cached import FusionNet,CacheDataset,CACHE,DEVICE,BATCH,SCALE_Y
from train_transformer_cached import Net as TNet
from train_hybrid_cached import DS,Net as HNet,load_tab

def c(y,p):y=y-y.mean();p=p-p.mean();return float(np.dot(y,p)/(np.linalg.norm(y)*np.linalg.norm(p)+1e-12))
def unit(p):p=p-p.mean();return p/(np.linalg.norm(p)+1e-12)
def main():
 lab=pd.read_feather('data/train/label.feather');dfs=[lab]
 for n in ['market','order','transaction']:
  for v in ['','_v2']:
   p=f'features/{n}_train{v}.parquet'
   if os.path.exists(p):dfs.append(pd.read_parquet(p))
 df=dfs[0]
 for d in dfs[1:]:df=df.merge(d,on='sample_id',how='left')
 cols=[z for z in df if z not in ('month','sample_id','target')];tr=df[df.month<62];va=df[df.month>=62];pa={'objective':'regression','metric':'l2','learning_rate':.03,'num_leaves':127,'max_depth':8,'min_child_samples':1000,'feature_fraction':.7,'bagging_fraction':.8,'bagging_freq':1,'lambda_l2':2.,'verbosity':-1,'n_jobs':10,'seed':42}
 lm=lgb.train(pa,lgb.Dataset(tr[cols],label=tr.target),num_boost_round=376);pl=lm.predict(va[cols]);order=np.argsort(va.sample_id.to_numpy());y=va.target.to_numpy()[order];pl=pl[order]
 ids=np.load(CACHE+'/train_ids.npy');m=np.load(CACHE+'/train_market.npy',mmap_mode='r');o=np.load(CACHE+'/train_order.npy',mmap_mode='r');x=np.load(CACHE+'/train_tx.npy',mmap_mode='r');tab,tc,mon,yy=load_tab(ids);ti=np.flatnonzero(mon<62);vi=np.flatnonzero(mon>=62);mu=np.nanmean(tab[ti],0);sd=np.maximum(np.nanstd(tab[ti],0),1e-6);tab=np.where(np.isnan(tab),mu,tab);tab=np.clip((tab-mu)/sd,-10,10).astype(np.float32)
 dl=torch.utils.data.DataLoader(CacheDataset(m,o,x,yy*SCALE_Y,vi),batch_size=BATCH*2,shuffle=False);pred=[]
 for cls,path in [(FusionNet,'output/gru_cached_best.pt'),(TNet,'output/transformer_best.pt')]:
  q=cls().to(DEVICE);q.load_state_dict(torch.load(path,map_location=DEVICE));q.eval();z=[]
  with torch.no_grad():
   for bm,bo,bx,by in dl:z.append(q(bm.to(DEVICE),bo.to(DEVICE),bx.to(DEVICE)).cpu().numpy()/SCALE_Y)
  pred.append(np.concatenate(z))
 hdl=torch.utils.data.DataLoader(DS(m,o,x,tab,yy*SCALE_Y,vi),batch_size=BATCH*2,shuffle=False);q=HNet(len(tc)).to(DEVICE);q.load_state_dict(torch.load('output/hybrid_best.pt',map_location=DEVICE));q.eval();z=[]
 with torch.no_grad():
  for bm,bo,bx,bt,by in hdl:z.append(q(bm.to(DEVICE),bo.to(DEVICE),bx.to(DEVICE),bt.to(DEVICE)).cpu().numpy()/SCALE_Y)
 pred.append(np.concatenate(z))
 robust_z=np.load('output/rolling_lgb_preds.npz'); robust=.4*robust_z['late_base600']+.3*robust_z['late_cons600']+.3*robust_z['late_rand600']; robust=robust[order]
 seed_z=np.load('output/rolling_lgb_seed_preds.npz'); seedavg=seed_z['late_avg3'][order]
 print('L/G/T/H/RobustLGB/SeedAvg',*[f'{c(y,z):.5f}' for z in [pl,*pred,robust,seedavg]])
 old4=.3*pl+.3*pred[0]+.1*pred[1]+.3*pred[2]
 print('old four:',c(y,old4))
 print('robust four:',c(y,.3*robust+.3*pred[0]+.1*pred[1]+.3*pred[2]))
 print('seedavg four:',c(y,.3*seedavg+.3*pred[0]+.1*pred[1]+.3*pred[2]))
 ms_z=np.load('output/multistream_val_preds.npz'); assert np.array_equal(ms_z['sample_id'],va.sample_id.to_numpy()[order]); ms=.5*ms_z['ep5']+.5*ms_z['ep8']
 print('multistream checkpoint ensemble:',c(y,ms),'std',ms.std(),'old4 std',old4.std())
 print('UNIT-NORMALIZED BLEND')
 for a in np.arange(0,1.01,.1):print('old4 weight',round(a,1),c(y,a*unit(old4)+(1-a)*unit(ms)))
 # coarse unit-normalized 5-model search: LGB, GRU, Transformer, Hybrid, MultiStream
 models=[unit(pl),unit(pred[0]),unit(pred[1]),unit(pred[2]),unit(ms)];best5=[]
 for a in np.arange(0,1.01,.1):
  for b in np.arange(0,1.01-a,.1):
   for d in np.arange(0,1.01-a-b,.1):
    for e in np.arange(0,1.01-a-b-d,.1):
     f=1-a-b-d-e;best5.append((c(y,a*models[0]+b*models[1]+d*models[2]+e*models[3]+f*models[4]),a,b,d,e,f))
 print('best unit 5-model',sorted(best5,reverse=True)[:8])
 v3z=np.load('output/multistream_v3_val_preds.npz'); assert np.array_equal(v3z['sample_id'],va.sample_id.to_numpy()[order]); v3=.5*unit(v3z['ep5'])+.5*unit(v3z['ep6'])
 print('v3 unit checkpoint ensemble',c(y,v3))
 # Exact Public-0.136 candidate and conservative enhanced-LGB additions.
 micro_z=np.load('output/rolling_micro_lgb_preds.npz'); micro=unit(micro_z['late_combined'][order])
 public136=.1*models[0]+.2*models[1]+.2*models[3]+.25*unit(ms)+.25*v3
 print('public136 reconstructed',c(y,public136))
 for w in [.05,.10,.15,.20,.25,.30]:
  p=(1-w)*unit(public136)+w*micro
  print('public136 + micro',w,c(y,p))
 # Reconstruct current Public-0.137 candidate, then add fixed RealMLP checkpoints.
 public137=.9*unit(public136)+.1*micro
 rz=np.load('output/realmlp_rolling_oof.npz');assert np.array_equal(rz['late_sample_id'],va.sample_id.to_numpy()[order]);real=np.mean([unit(rz[f'late_ep{e}']) for e in (6,9,11)],axis=0)
 print('public137 reconstructed',c(y,public137),'realmlp',c(y,real),'corr',float(unit(public137)@unit(real)))
 for w in [.05,.10,.15,.20,.25,.30,.35,.40]:
  p=(1-w)*unit(public137)+w*unit(real)
  print('public137 + realmlp',w,c(y,p))
 public138=.95*unit(public137)+.05*unit(real)
 # New v3 effective-batch-1024 model, validated on proxy/middle/late.
 nvz=np.load('output/multistream_v3_late_eff1024_oof.npz');assert np.array_equal(nvz['sample_id'],va.sample_id.to_numpy()[order]);newv3=unit(nvz['ens4_5_6']);newv3_alt=unit(nvz['ens5_6_8'])
 print('public138 reconstructed',c(y,public138),'newv3',c(y,newv3),'newv3_alt',c(y,newv3_alt),'corr old/new',float(unit(v3)@newv3),'corr public/new',float(unit(public138)@newv3))
 for w in [.05,.10,.15,.20,.25,.30,.35,.40]:print('public138 + newv3',w,c(y,(1-w)*unit(public138)+w*newv3))
 # Replace or mix the old-v3 25% slot before applying the proven micro/RealMLP increments.
 for old_share in [0,.25,.5,.75,1.0]:
  v3slot=old_share*unit(v3)+(1-old_share)*newv3
  base=.1*models[0]+.2*models[1]+.2*models[3]+.25*unit(ms)+.25*unit(v3slot)
  cand137=.9*unit(base)+.1*micro;cand138=.95*unit(cand137)+.05*unit(real)
  print('v3slot old/new',old_share,1-old_share,'base',c(y,base),'final',c(y,cand138))
 # blend robust existing 4-model unit candidate with v2 and v3 multi-scale streams
 base4=.1*models[0]+.2*models[1]+.2*models[3]+.5*models[4]
 for a in np.arange(0,1.01,.1):
  for b in np.arange(0,1.01-a,.1):
   if abs((a+b)-.5)<1e-8 or a in [0,.2,.4,.6,.8,1.0]: print('multi-scale weights base/v2/v3',round(a,1),round(b,1),round(1-a-b,1),c(y,a*unit(base4)+b*unit(ms)+(1-a-b)*unit(v3)))
 for rr in [.1,.2,.3,.5,1.0]: print('robust replacement',rr,c(y,(.3*(1-rr))*pl+(.3*rr)*robust+.3*pred[0]+.1*pred[1]+.3*pred[2]))
 # Add H to fixed 3-way: search coarse simplex, reporting top
 best=[]
 for a in np.arange(0,1.01,.1):
  for b in np.arange(0,1.01-a,.1):
   for d in np.arange(0,1.01-a-b,.1):
    e=1-a-b-d;s=c(y,a*pl+b*pred[0]+d*pred[1]+e*pred[2]);best.append((s,a,b,d,e))
 for row in sorted(best,reverse=True)[:8]:print('blend',row)
if __name__=='__main__':main()
