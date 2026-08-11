import os, numpy as np, pandas as pd, torch, lightgbm as lgb
from train_gru_cached import FusionNet, CacheDataset, DEVICE, BATCH, SCALE_Y

def cos(y,p):
    return float(np.dot(y,p)/(np.linalg.norm(y)*np.linalg.norm(p)+1e-12))
def cos_center(y,p):
    return cos(y-y.mean(),p-p.mean())

def main():
    label=pd.read_feather('data/train/label.feather')
    dfs=[label]
    for n in ['market','order','transaction']:
        for v in ['','_v2']:
            p=f'features/{n}_train{v}.parquet'
            if os.path.exists(p): dfs.append(pd.read_parquet(p))
    df=dfs[0]
    for d in dfs[1:]: df=df.merge(d,on='sample_id',how='left')
    cols=[c for c in df if c not in ('month','sample_id','target')]
    tr=df[df.month<62]; va=df[df.month>=62]
    params={'objective':'regression','metric':'l2','learning_rate':.03,'num_leaves':127,'max_depth':8,'min_child_samples':1000,'feature_fraction':.7,'bagging_fraction':.8,'bagging_freq':1,'lambda_l2':2.,'verbosity':-1,'n_jobs':10,'seed':42}
    model=lgb.train(params,lgb.Dataset(tr[cols],label=tr.target),num_boost_round=376)
    pl=model.predict(va[cols]); y=va.target.to_numpy()
    print('LGB raw/center:',cos(y,pl),cos_center(y,pl),'std',pl.std())
    # GRU val
    C='features/cache'; ids=np.load(C+'/train_ids.npy'); m=np.load(C+'/train_market.npy',mmap_mode='r'); o=np.load(C+'/train_order.npy',mmap_mode='r'); x=np.load(C+'/train_tx.npy',mmap_mode='r')
    l2=label.set_index('sample_id'); months=l2.loc[ids].month.to_numpy(); yy=l2.loc[ids].target.to_numpy(np.float32)
    vi=np.flatnonzero(months>=62); ds=CacheDataset(m,o,x,yy*SCALE_Y,vi); dl=torch.utils.data.DataLoader(ds,batch_size=BATCH*2,shuffle=False)
    gm=FusionNet().to(DEVICE); gm.load_state_dict(torch.load('output/gru_cached_best.pt',map_location=DEVICE)); gm.eval(); pg=[]
    with torch.no_grad():
      for bm,bo,bx,by in dl: pg.append(gm(bm.to(DEVICE),bo.to(DEVICE),bx.to(DEVICE)).cpu().numpy()/SCALE_Y)
    pg=np.concatenate(pg)
    # ids order in va is sample_id ascending because label is sorted; align explicit
    order=np.argsort(va.sample_id.to_numpy()); assert np.array_equal(va.sample_id.to_numpy()[order],ids[vi])
    pl=pl[order]; y=y[order]
    print('GRU raw/center:',cos(y,pg),cos_center(y,pg),'std',pg.std())
    for a in np.arange(0,1.01,.1): print(f'blend lgb={a:.1f}: raw={cos(y,a*pl+(1-a)*pg):.5f} center={cos_center(y,a*pl+(1-a)*pg):.5f}')
if __name__=='__main__': main()
