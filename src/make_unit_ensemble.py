"""Make scale-invariant unit-normalized ensembles. This script never submits."""
import argparse,numpy as np,pandas as pd
def unit(x):x=np.asarray(x,np.float64);x=x-x.mean();return x/(np.linalg.norm(x)+1e-12)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--members',nargs='+',required=True);ap.add_argument('--weights',nargs='+',type=float,required=True);ap.add_argument('--out',required=True);a=ap.parse_args();assert len(a.members)==len(a.weights) and sum(a.weights)>0
 ds=[pd.read_csv(p).sort_values('sample_id') for p in a.members];ids=ds[0].sample_id.to_numpy();assert all(np.array_equal(ids,d.sample_id.to_numpy()) for d in ds);w=np.asarray(a.weights)/sum(a.weights);p=sum(x*unit(d.prediction.to_numpy()) for x,d in zip(w,ds));pd.DataFrame({'sample_id':ids,'prediction':p}).to_csv(a.out,index=False);print(a.out,'members',len(ds),'weights',w.tolist(),'mean',p.mean(),'std',p.std())
if __name__=='__main__':main()
