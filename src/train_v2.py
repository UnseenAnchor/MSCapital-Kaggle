"""训练 v2：合并 v1+v2 特征，时序验证，余弦评分"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import time, sys, os

TRAIN_MONTHS = (0, 62)
VAL_MONTHS = (62, 71)

def load():
    t0 = time.time()
    l = pd.read_feather('data/train/label.feather')
    dfs = [l]
    for name in ['market', 'order', 'transaction']:
        for v in ['', '_v2']:
            p = f'features/{name}_train{v}.parquet'
            if os.path.exists(p):
                dfs.append(pd.read_parquet(p))
    df = dfs[0]
    for d in dfs[1:]:
        df = df.merge(d, on='sample_id', how='left')
    print(f'加载 {df.shape}, {time.time()-t0:.0f}s')
    return df

def cosine(y_true, y_pred):
    yt = y_true - y_true.mean()
    yp = y_pred - y_pred.mean()
    return float(np.dot(yt, yp) / (np.linalg.norm(yt) * np.linalg.norm(yp) + 1e-12))

def main():
    df = load()
    feat_cols = [c for c in df.columns if c not in ('month', 'sample_id', 'target')]
    print(f'特征数: {len(feat_cols)}')

    tr = df[df.month.isin(range(*TRAIN_MONTHS))]
    va = df[df.month.isin(range(*VAL_MONTHS))]

    params = {
        'objective': 'regression', 'metric': 'l2', 'learning_rate': 0.03,
        'num_leaves': 127, 'max_depth': 8, 'min_child_samples': 1000,
        'feature_fraction': 0.7, 'bagging_fraction': 0.8, 'bagging_freq': 1,
        'lambda_l2': 2.0, 'verbosity': -1, 'n_jobs': 10, 'seed': 42,
    }
    dtr = lgb.Dataset(tr[feat_cols], label=tr['target'])
    dva = lgb.Dataset(va[feat_cols], label=va['target'])
    t0 = time.time()
    model = lgb.train(params, dtr, num_boost_round=5000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(150), lgb.log_evaluation(500)])
    print(f'训练 {time.time()-t0:.0f}s, best_iter={model.best_iteration}')

    pv = model.predict(va[feat_cols], num_iteration=model.best_iteration)
    print(f'Val cosine: {cosine(va["target"].values, pv):.5f}')
    va2 = va.copy(); va2['pred'] = pv
    for mo in range(*VAL_MONTHS):
        s = va2[va2.month == mo]
        if len(s) > 0:
            print(f'  month {mo}: {cosine(s["target"].values, s["pred"].values):.5f}')

    # 全量重训
    print('全量训练中...')
    all_d = lgb.Dataset(df[feat_cols], label=df['target'])
    model2 = lgb.train(params, all_d, num_boost_round=model.best_iteration * 2)
    model2.save_model('output/lgb_model_v2.txt')

    te_m = pd.read_parquet('features/market_test.parquet')
    te_o = pd.read_parquet('features/order_test.parquet')
    te_x = pd.read_parquet('features/transaction_test.parquet')
    te = te_m.merge(te_o, on='sample_id', how='left').merge(te_x, on='sample_id', how='left')
    for name in ['market', 'order', 'transaction']:
        p = f'features/{name}_test_v2.parquet'
        if os.path.exists(p):
            te = te.merge(pd.read_parquet(p), on='sample_id', how='left')
    print(f'test {te.shape}')
    te['prediction'] = model2.predict(te[feat_cols], num_iteration=model2.best_iteration)
    sub = te[['sample_id', 'prediction']].sort_values('sample_id')
    sub.to_csv('output/submission_v2.csv', index=False)
    print(f'submission_v2: {sub.shape}')

if __name__ == '__main__':
    main()
