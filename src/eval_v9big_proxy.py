"""Evaluate fixed v9big Proxy checkpoint ensemble and low-correlation increments."""
import numpy as np,pandas as pd

def unit(x):x=np.asarray(x,np.float64);x-=x.mean();return x/(np.linalg.norm(x)+1e-12)
def c(y,p):return float(unit(y)@unit(p))
def stats(y,p,mo):
 vals=[c(y[mo==m],p[mo==m]) for m in np.unique(mo)];return c(y,p),float(np.mean(vals)),float(np.min(vals)),float(np.std(vals))
def main():
 zs=[np.load(f'output/v9big_proxy_fast_ep{e}_oof.npz') for e in [4,5,6]];ids=zs[0]['sample_id'];y=zs[0]['target'];mo=zs[0]['month'];p=np.mean([unit(z['prediction']) for z in zs],0)
 vb=np.load('output/multistream_v3_proxy_oof.npz');j=np.load('output/joint_v3_proxy_fast_oof.npz');r=np.load('output/realmlp_multiseed_proxy_oof.npz');l=np.load('output/proxy_lgb_oof.npz');assert all(np.array_equal(ids,z['sample_id']) for z in [vb,j,r,l])
 models={'lgb':l['prediction'],'real':r['s42'],'v3':vb['ens4_5_6'],'joint':j['ens4_5_6'],'v9big':p}
 for k,v in models.items():print(k,stats(y,v,mo),'corr_v9',unit(v)@unit(p))
 base=.4*unit(models['v3'])+.6*unit(models['joint']);print('base_v3joint',stats(y,base,mo),'corr_v9',unit(base)@unit(p))
 for w in [.1,.2,.3,.4,.5]:print('v9 weight',w,stats(y,(1-w)*unit(base)+w*unit(p),mo))
 # Existing broader own proxy approximation: LGB40 + v3/joint slot60.
 broad=.4*unit(models['lgb'])+.6*unit(base);print('broad',stats(y,broad,mo),'corr_v9',unit(broad)@unit(p))
 for w in [.1,.2,.3,.4]:print('broad+v9',w,stats(y,(1-w)*unit(broad)+w*unit(p),mo))
 np.savez('output/v9big_proxy_fast_oof.npz',sample_id=ids,target=y,month=mo,ep4=zs[0]['prediction'],ep5=zs[1]['prediction'],ep6=zs[2]['prediction'],ens4_5_6=p)
if __name__=='__main__':main()
