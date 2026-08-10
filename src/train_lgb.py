"""训练 LightGBM baseline：时间序列验证（按月份切分），余弦相似度评分"""
import numpy as np
import pandas as pd
import lightgbm as lgb
import time, sys

TRAIN_MONTHS = (0, 62)   # 训练用月份 [0, 62)
VAL_MONTHS = (62, 71)    # 验证用月份 [62, 71)

def load():
    t0 = time.time()
    l = pd.read_feather('data/train/label.feather')
    m = pd.read_parquet('features/market_train.parquet')
    o = pd.read_parquet('features/order_train.parquet')
    x = pd.read_parquet('features/transaction_train.parquet')
    df = l.merge(m, on='sample_id', how='left').merge(o, on='sample_id', how='left').merge(x, on='sample_id', how='left')
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
    print(f'train {tr.shape}, val {va.shape}')

    params = {
        'objective': 'regression', 'metric': 'l2', 'learning_rate': 0.03,
        'num_leaves': 63, 'max_depth': 7, 'min_child_samples': 500,
        'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 1,
        'lambda_l2': 1.0, 'verbosity': -1, 'n_jobs': 10, 'seed': 42,
    }
    dtr = lgb.Dataset(tr[feat_cols], label=tr['target'])
    dva = lgb.Dataset(va[feat_cols], label=va['target'])
    t0 = time.time()
    model = lgb.train(params, dtr, num_boost_round=3000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(100), lgb.log_evaluation(200)])
    print(f'训练 {time.time()-t0:.0f}s, best_iter={model.best_iteration}')

    pv = model.predict(va[feat_cols], num_iteration=model.best_iteration)
    print(f'Val cosine: {cosine(va["target"].values, pv):.5f}')
    # 逐月
    va2 = va.copy(); va2['pred'] = pv
    for mo in range(*VAL_MONTHS):
        s = va2[va2.month == mo]
        if len(s) > 0:
            print(f'  month {mo}: {cosine(s["target"].values, s["pred"].values):.5f}')

    # 全量训练（含验证月份）后预测 test
    print('全量训练中...')
    all_d = lgb.Dataset(df[feat_cols], label=df['target'])
    model2 = lgb.train(params, all_d, num_boost_round=model.best_iteration * 2)
    model.save_model('output/lgb_model.txt')

    te_m = pd.read_parquet('features/market_test.parquet')
    te_o = pd.read_parquet('features/order_test.parquet')
    te_x = pd.read_parquet('features/transaction_test.parquet')
    te = te_m.merge(te_o, on='sample_id', how='left').merge(te_x, on='sample_id', how='left')
    print(f'test {te.shape}')
    te['prediction'] = model2.predict(te[feat_cols], num_iteration=model2.best_iteration)
    sub = te[['sample_id', 'prediction']].sort_values('sample_id')
    sub.to_csv('output/submission.csv', index=False)
    print(f'submission: {sub.shape}, 均值 {sub.prediction.mean():.6f}')

if __name__ == '__main__':
    main()
