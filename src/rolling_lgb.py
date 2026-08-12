"""滚动月份验证 LGB：只生成候选和报告，不上传 Kaggle。
目标：比较正则化、树复杂度、随机采样与训练轮数，降低单一验证区间过拟合。
"""
import os, time, json
import numpy as np
import pandas as pd
import lightgbm as lgb

SPLITS = {
    'early': (range(0, 41), range(41, 51)),
    'middle': (range(0, 51), range(51, 61)),
    'late': (range(0, 62), range(62, 71)),
}
ROUNDS = [200, 400, 600]
CONFIGS = {
    'baseline': dict(num_leaves=127, max_depth=8, min_child_samples=1000, feature_fraction=.7, bagging_fraction=.8, bagging_freq=1, lambda_l2=2., extra_trees=False),
    'conservative': dict(num_leaves=63, max_depth=7, min_child_samples=2000, feature_fraction=.9, bagging_fraction=.9, bagging_freq=1, lambda_l2=5., extra_trees=False),
    'regularized': dict(num_leaves=31, max_depth=6, min_child_samples=3000, feature_fraction=.9, bagging_fraction=.9, bagging_freq=1, lambda_l2=10., extra_trees=False),
    'randomized': dict(num_leaves=63, max_depth=8, min_child_samples=2000, feature_fraction=.7, bagging_fraction=.8, bagging_freq=1, lambda_l2=5., extra_trees=True),
}

def cosine(y, p, center=False):
    if center:
        y = y - y.mean(); p = p - p.mean()
    return float(np.dot(y, p) / (np.linalg.norm(y) * np.linalg.norm(p) + 1e-12))

def load():
    label = pd.read_feather('data/train/label.feather')
    dfs = [label]
    for n in ['market', 'order', 'transaction']:
        for v in ['', '_v2']:
            p = f'features/{n}_train{v}.parquet'
            if os.path.exists(p): dfs.append(pd.read_parquet(p))
    df = dfs[0]
    for d in dfs[1:]: df = df.merge(d, on='sample_id', how='left')
    cols = [c for c in df if c not in ('month', 'sample_id', 'target')]
    return df, cols

def main():
    t0 = time.time(); df, cols = load(); print('data', df.shape, 'features', len(cols), flush=True)
    rows = []
    for cname, cp in CONFIGS.items():
        params = dict(objective='regression', metric='l2', learning_rate=.03, verbosity=-1, n_jobs=10, seed=42, **cp)
        for sname, (trm, vam) in SPLITS.items():
            tr = df[df.month.isin(trm)]; va = df[df.month.isin(vam)]
            print(f'{cname}/{sname}: train={len(tr)} val={len(va)}', flush=True)
            model = lgb.train(params, lgb.Dataset(tr[cols], label=tr.target), num_boost_round=max(ROUNDS))
            y = va.target.to_numpy()
            for r in ROUNDS:
                p = model.predict(va[cols], num_iteration=r)
                rows.append(dict(config=cname, split=sname, rounds=r, raw=cosine(y,p), centered=cosine(y,p,True), pred_std=p.std()))
            del model, tr, va
    out = pd.DataFrame(rows)
    os.makedirs('output', exist_ok=True); out.to_csv('output/rolling_lgb_report.csv', index=False)
    print('\nMEAN BY CONFIG/ROUND (raw):')
    print(out.groupby(['config','rounds']).raw.agg(['mean','std','min']).sort_values('mean', ascending=False).head(15).to_string())
    print('\nSPLIT DETAIL:')
    print(out.sort_values(['config','rounds','split']).to_string(index=False))
    print('seconds', time.time()-t0)

if __name__ == '__main__': main()
