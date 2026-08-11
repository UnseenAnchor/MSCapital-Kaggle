import os,numpy as np,pandas as pd,torch,lightgbm as lgb
from train_gru_cached import FusionNet,CacheDataset,CACHE,DEVICE,BATCH,SCALE_Y
from train_transformer_cached import Net

def c(y,p):
 y=y-y.mean();p=p-p.mean();return float(np.dot(y,p)/(np.linalg.norm(y)*np.linalg.norm(p)+1e-12))
def main():
 lab=pd.read_feather('data/train/label.feather'); dfs=[lab]
 for n in ['market','order','transaction']:
  for v in ['','_v2']:
   p=f'features/{n}_train{v}.parquet'
   if os.path.exists(p):dfs.append(pd.read_parquet(p))
 df=dfs[0]
 for d in dfs[1:]:df=df.merge(d,on='sample_id',how='left')
 cols=[z for z in df if z not in ('month','sample_id','target')];tr=df[df.month<62];va=df[df.month>=62]
 pa={'objective':'regression','metric':'l2','learning_rate':.03,'num_leaves':127,'max_depth':8,'min_child_samples':1000,'feature_fraction':.7,'bagging_fraction':.8,'bagging_freq':1,'lambda_l2':2.,'verbosity':-1,'n_jobs':10,'seed':42}
 lm=lgb.train(pa,lgb.Dataset(tr[cols],label=tr.target),num_boost_round=376); pl=lm.predict(va[cols]);
 ids=np.load(CACHE+'/train_ids.npy');m=np.load(CACHE+'/train_market.npy',mmap_mode='r');o=np.load(CACHE+'/train_order.npy',mmap_mode='r');x=np.load(CACHE+'/train_tx.npy',mmap_mode='r');li=lab.set_index('sample_id'); mon=li.loc[ids].month.to_numpy();y=li.loc[ids].target.to_numpy(np.float32);vi=np.flatnonzero(mon>=62)
 dl=torch.utils.data.DataLoader(CacheDataset(m,o,x,y*SCALE_Y,vi),batch_size=BATCH*2,shuffle=False)
 out=[]
 for cls,path in [(FusionNet,'output/gru_cached_best.pt'),(Net,'output/transformer_best.pt')]:
  q=cls().to(DEVICE);q.load_state_dict(torch.load(path,map_location=DEVICE));q.eval();z=[]
  with torch.no_grad():
   for bm,bo,bx,by in dl:z.append(q(bm.to(DEVICE),bo.to(DEVICE),bx.to(DEVICE)).cpu().numpy()/SCALE_Y)
  out.append(np.concatenate(z))
 order=np.argsort(va.sample_id.to_numpy()); yy=va.target.to_numpy()[order]; pl=pl[order]
 print('base L/G/T',c(yy,pl),c(yy,out[0]),c(yy,out[1]))
 best=(-9,None)
 for a in np.arange(0,1.01,.1):
  for b in np.arange(0,1.01-a,.1):
   p=a*pl+b*out[0]+(1-a-b)*out[1];s=c(yy,p)
   if s>best[0]:best=(s,(a,b,1-a-b))
 print('best grid',best)
 for a,b in [(0.4,.4),(.5,.3),(.5,.4),(.6,.3)]:print(a,b,1-a-b,c(yy,a*pl+b*out[0]+(1-a-b)*out[1]))
if __name__=='__main__':main()
