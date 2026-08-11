"""一次性构建序列缓存：加载 feather → fillna → 归一化 → 固定长度采样 → npy 缓存
缓存后训练从内存数组直接加载，避免每次 24 分钟的数据读取。
"""
import numpy as np
import pandas as pd
import torch
import pyarrow.feather as pf
import time, os, sys

LEN_M, LEN_O, LEN_X = 64, 32, 32
M_COLS = ['transaction_avgprice', 'transaction_volume', 'transaction_count',
          'ask_price_1', 'ask_volume_1', 'bid_price_1', 'bid_volume_1',
          'ask_price_2', 'ask_volume_2', 'bid_price_2', 'bid_volume_2']
O_COLS = ['price', 'volume', 'side', 'order_action']
X_COLS = ['price', 'volume', 'side']

CACHE = 'features/cache'
os.makedirs(CACHE, exist_ok=True)


def load_grouped(split, key, cols):
    t0 = time.time()
    fname = 'transaction' if key == 'tx' else key
    t = pf.read_table(f'data/{split}/{fname}.feather')
    df = t.select(['sample_id'] + cols).to_pandas()
    grp = df.groupby('sample_id', sort=True)
    out = {sid: np.ascontiguousarray(g[cols].values, dtype=np.float32) for sid, g in grp}
    for sid in out:
        a = out[sid]
        if np.isnan(a).any():
            out[sid] = np.nan_to_num(a, nan=0.0)
    print(f'  [{split}/{key}] {len(out)} samples, {time.time()-t0:.0f}s', flush=True)
    return out


def estimate_stats(data, key, n=300):
    s = 0.0; sq = 0.0; c = 0
    rng = np.random.RandomState(0)
    sids = rng.choice(list(data[key].keys()), min(n, len(data[key])), replace=False)
    for sid in sids:
        a = data[key][sid]
        s += a.sum(0); sq += (a ** 2).sum(0); c += len(a)
    mu = s / c
    sd = np.sqrt(np.maximum(sq / c - mu ** 2, 1e-6))
    return mu, sd


def to_fixed(arr, L, C):
    """采样到固定长度 L（新在前），返回 (L, C)"""
    if arr is None or len(arr) == 0:
        return np.zeros((L, C), dtype=np.float32)
    T = len(arr)
    if T >= L:
        idx = np.linspace(0, T - 1, L).astype(np.int64)
    else:
        idx = np.concatenate([np.zeros(L - T, dtype=np.int64), np.arange(T)])
    return arr[::-1][idx].astype(np.float32)


def build(split):
    t0 = time.time()
    data = {}
    for key, cols, L in [('market', M_COLS, LEN_M), ('order', O_COLS, LEN_O), ('tx', X_COLS, LEN_X)]:
        data[key] = load_grouped(split, key, cols)
    # 归一化统计（train 统计）
    stats_path = f'{CACHE}/stats.npz'
    if os.path.exists(stats_path) and split == 'test':
        z = np.load(stats_path, allow_pickle=True)
        stats = {k: (z[f'{k}_mu'], z[f'{k}_sd']) for k in ['market', 'order', 'tx']}
    else:
        stats = {k: estimate_stats(data, k) for k in data}
        np.savez(stats_path, **{f'{k}_mu': stats[k][0] for k in stats},
                 **{f'{k}_sd': stats[k][1] for k in stats})
    for key in data:
        mu, sd = stats[key]
        for sid in data[key]:
            data[key][sid] = (data[key][sid] - mu) / sd
    print(f'归一化完成 {time.time()-t0:.0f}s', flush=True)

    # 组装为 (n, L, C) 数组 + ids
    ids = np.array(sorted(data['market'].keys()), dtype=np.int32)
    arrays = {}
    for key, L in [('market', LEN_M), ('order', LEN_O), ('tx', LEN_X)]:
        C = len(M_COLS) if key == 'market' else (len(O_COLS) if key == 'order' else len(X_COLS))
        arr = np.zeros((len(ids), L, C), dtype=np.float32)
        for i, sid in enumerate(ids):
            arr[i] = to_fixed(data[key].get(sid), L, C)
        arrays[key] = arr
        print(f'  [{split}/{key}] cache {arr.shape}, {time.time()-t0:.0f}s', flush=True)
    np.save(f'{CACHE}/{split}_ids.npy', ids)
    for key in arrays:
        np.save(f'{CACHE}/{split}_{key}.npy', arrays[key])
    print(f'[{split}] 缓存完成: {CACHE}, 总耗时 {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    build(sys.argv[1] if len(sys.argv) > 1 else 'train')
