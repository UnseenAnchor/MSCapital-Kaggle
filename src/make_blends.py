import pandas as pd, numpy as np
l=pd.read_csv('output/submission_v2.csv').sort_values('sample_id')
g=pd.read_csv('output/submission_gru_cached.csv').sort_values('sample_id')
assert np.array_equal(l.sample_id,g.sample_id)
for a in [0.4,0.5,0.6]:
    p=a*l.prediction.to_numpy()+(1-a)*g.prediction.to_numpy()
    out=pd.DataFrame({'sample_id':l.sample_id,'prediction':p})
    path=f'output/submission_blend_{int(a*100)}.csv'; out.to_csv(path,index=False)
    print(path, 'mean',p.mean(),'std',p.std())
