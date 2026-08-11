import os,numpy as np,pandas as pd,torch,lightgbm as lgb
from train_gru_cached import FusionNet,CacheDataset,CACHE,DEVICE,BATCH,SCALE_Y
from train_transformer_cached import Net as TNet
from train_hybrid_cached import DS,Net as HNet,load_tab

def c(y,p):y=y-y.mean();p=p-p.mean();return float(np.dot(y,p)/(np.linalg.norm(y)*np.linalg.norm(p)+1e-12))
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
 pred.append(np.concatenate(z));print('L/G/T/H',*[f'{c(y,z):.5f}' for z in [pl,*pred]])
 # Add H to fixed 3-way: search coarse simplex, reporting top
 best=[]
 for a in np.arange(0,1.01,.1):
  for b in np.arange(0,1.01-a,.1):
   for d in np.arange(0,1.01-a-b,.1):
    e=1-a-b-d;s=c(y,a*pl+b*pred[0]+d*pred[1]+e*pred[2]);best.append((s,a,b,d,e))
 for row in sorted(best,reverse=True)[:8]:print('blend',row)
if __name__=='__main__':main()
