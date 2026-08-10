"""Order + Transaction 特征工程：按 sample_id 聚合 1 分钟事件流特征"""
import numpy as np
import pandas as pd
import pyarrow.feather as pf
import time, os, sys

DATA = 'data'
OUT = 'features'
SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'train'
CHUNK = 12_000_000


def process_order(df: pd.DataFrame) -> pd.DataFrame:
    df['t'] = -df['seconds_before_predict']
    g = df.groupby('sample_id')
    feats = pd.DataFrame({'sample_id': df['sample_id'].unique()}).set_index('sample_id')
    def add(n, s): feats[n] = s
    add('o_n', g.size())
    add('o_vol_sum', g['volume'].sum())
    add('o_vol_mean', g['volume'].mean())
    add('o_price_mean', g['price'].mean())
    add('o_price_std', g['price'].std())
    add('o_price_last', g['price'].last())
    # side=1 占比（买/卖方向）
    add('o_side1_frac', g['side'].mean())
    add('o_side1_vol', df.assign(v=df['volume'] * (df['side'] == 1)).groupby('sample_id')['v'].sum())
    # order_action：0/1 占比（新增/撤单）
    add('o_act1_frac', g['order_action'].mean())
    # 方向×动作交叉
    add('o_buy_add_vol', df.assign(v=df['volume'] * ((df['side'] == 1) & (df['order_action'] == 1))).groupby('sample_id')['v'].sum())
    add('o_sell_add_vol', df.assign(v=df['volume'] * ((df['side'] == 0) & (df['order_action'] == 1))).groupby('sample_id')['v'].sum())
    add('o_buy_cancel_vol', df.assign(v=df['volume'] * ((df['side'] == 1) & (df['order_action'] == 0))).groupby('sample_id')['v'].sum())
    add('o_sell_cancel_vol', df.assign(v=df['volume'] * ((df['side'] == 0) & (df['order_action'] == 0))).groupby('sample_id')['v'].sum())
    # 事件强度（每秒事件数）
    add('o_rate', g.size() / (df.groupby('sample_id')['t'].max() - df.groupby('sample_id')['t'].min() + 1e-9))
    return feats.reset_index()


def process_tx(df: pd.DataFrame) -> pd.DataFrame:
    df['t'] = -df['seconds_before_predict']
    g = df.groupby('sample_id')
    feats = pd.DataFrame({'sample_id': df['sample_id'].unique()}).set_index('sample_id')
    def add(n, s): feats[n] = s
    add('x_n', g.size())
    add('x_vol_sum', g['volume'].sum())
    add('x_vol_mean', g['volume'].mean())
    add('x_price_mean', g['price'].mean())
    add('x_price_std', g['price'].std())
    add('x_price_first', g['price'].first())
    add('x_price_last', g['price'].last())
    add('x_price_ret', (g['price'].last() - g['price'].first()) / (g['price'].first() + 1e-9))
    add('x_price_min', g['price'].min())
    add('x_price_max', g['price'].max())
    add('x_price_range', g['price'].max() - g['price'].min())
    # 主动买卖压力：side=1 占比与成交量
    add('x_side1_frac', g['side'].mean())
    add('x_buy_vol', df.assign(v=df['volume'] * (df['side'] == 1)).groupby('sample_id')['v'].sum())
    add('x_sell_vol', df.assign(v=df['volume'] * (df['side'] == 0)).groupby('sample_id')['v'].sum())
    add('x_vol_imb', (df.assign(v=df['volume'] * (df['side'].astype(np.int8) * 2 - 1)).groupby('sample_id')['v'].sum())
                  / (g['volume'].sum() + 1e-9))
    add('x_rate', g.size() / (df.groupby('sample_id')['t'].max() - df.groupby('sample_id')['t'].min() + 1e-9))
    return feats.reset_index()


def main():
    t0 = time.time()
    for name, fn in [('order', process_order), ('transaction', process_tx)]:
        src = f'{DATA}/{SPLIT}/{name}.feather'
        print(f'[{SPLIT}/{name}] 读取 {src} ...')
        reader = pf.read_table(src)
        n = reader.num_rows
        parts = []
        for i in range(0, n, CHUNK):
            t1 = time.time()
            df = reader.slice(i, CHUNK).to_pandas()
            df['volume'] = df['volume'].astype(np.int32)
            parts.append(fn(df))
            print(f'  块 {i//CHUNK}: {df.shape[0]:,} 行, {time.time()-t1:.0f}s')
            del df
        out = pd.concat(parts, ignore_index=True)
        out = out.groupby('sample_id', as_index=False).last()
        path = f'{OUT}/{name}_{SPLIT}.parquet'
        out.to_parquet(path, index=False)
        print(f'完成: {path}, {out.shape}, {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
