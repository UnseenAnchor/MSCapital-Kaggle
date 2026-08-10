"""Market 表特征工程：按 sample_id 聚合 10 分钟盘口/成交特征"""
import numpy as np
import pandas as pd
import pyarrow.feather as pf
import time, os, sys

DATA = 'data'
OUT = 'features'
SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'train'
CHUNK = 8_000_000  # 每次读的行数


def process_chunk(df: pd.DataFrame) -> pd.DataFrame:
    """对一批 sample 计算特征。df 需含全部13列。"""
    # 中间价与价差
    df['mid'] = (df['bid_price_1'] + df['ask_price_1']) / 2
    df['spread'] = df['ask_price_1'] - df['bid_price_1']
    df['imb1'] = (df['bid_volume_1'] - df['ask_volume_1']) / (df['bid_volume_1'] + df['ask_volume_1'] + 1e-9)
    df['imb2'] = (df['bid_volume_2'] - df['ask_volume_2']) / (df['bid_volume_2'] + df['ask_volume_2'] + 1e-9)
    df['depth'] = (df['bid_volume_1'] + df['ask_volume_1'] + df['bid_volume_2'] + df['ask_volume_2'])
    # avgprice 相对 mid 偏离（用 mid 归一）
    df['avgprice_dev'] = (df['transaction_avgprice'] - df['mid']) / (df['mid'] + 1e-9)
    df['t'] = -df['seconds_before_predict']  # 负值，越接近0越新

    g = df.groupby('sample_id')
    feats = pd.DataFrame({'sample_id': df['sample_id'].unique()})
    feats = feats.set_index('sample_id')

    def add(name, series):
        feats[name] = series

    # --- 基础统计 ---
    add('m_n', g['mid'].count())
    add('m_mid_last', g['mid'].last())
    add('m_mid_mean', g['mid'].mean())
    add('m_mid_std', g['mid'].std())
    add('m_spread_mean', g['spread'].mean())
    add('m_spread_std', g['spread'].std())
    add('m_spread_last', g['spread'].last())
    add('m_imb1_mean', g['imb1'].mean())
    add('m_imb1_last', g['imb1'].last())
    add('m_imb2_mean', g['imb2'].mean())
    add('m_depth_mean', g['depth'].mean())
    add('m_depth_last', g['depth'].last())
    # 成交量
    add('m_vol_mean', g['transaction_volume'].mean())
    add('m_vol_sum', g['transaction_volume'].sum())
    add('m_cnt_mean', g['transaction_count'].mean())
    add('m_cnt_sum', g['transaction_count'].sum())
    add('m_avgprice_mean', g['transaction_avgprice'].mean())
    add('m_avgprice_dev_mean', g['avgprice_dev'].mean())
    add('m_avgprice_dev_last', g['avgprice_dev'].last())
    # 首末变化率（mid 从窗口起点到终点）
    first_mid = g['mid'].first()
    last_mid = g['mid'].last()
    add('m_mid_ret', (last_mid - first_mid) / (first_mid + 1e-9))
    add('m_mid_min', g['mid'].min())
    add('m_mid_max', g['mid'].max())
    add('m_mid_range', g['mid'].max() - g['mid'].min())
    # 时间加权：最近1/3 与整体均值差异
    df['third'] = np.where(df['t'] > -200, 'recent', 'old')
    piv = df.pivot_table(index='sample_id', columns='third', values='mid', aggfunc='mean')
    if 'recent' in piv and 'old' in piv:
        add('m_mid_recent_old', piv['recent'] - piv['old'])
    # 最近5条均价
    tail = df.sort_values(['sample_id', 't']).groupby('sample_id').tail(5)
    tg = tail.groupby('sample_id')
    add('m_mid_tail5_mean', tg['mid'].mean())
    add('m_imb1_tail5_mean', tg['imb1'].mean())
    return feats.reset_index()


def main():
    t0 = time.time()
    src = f'{DATA}/{SPLIT}/market.feather'
    print(f'[{SPLIT}] 读取 {src} ...')
    reader = pf.read_table(src)
    n = reader.num_rows
    print(f'总行数 {n:,}')
    parts = []
    for i in range(0, n, CHUNK):
        t1 = time.time()
        df = reader.slice(i, CHUNK).to_pandas()
        # 转小类型省内存
        for c in ['transaction_volume', 'transaction_count', 'ask_volume_1', 'bid_volume_1',
                  'ask_volume_2', 'bid_volume_2']:
            df[c] = df[c].astype(np.int32)
        feats = process_chunk(df)
        parts.append(feats)
        print(f'  块 {i//CHUNK}: {df.shape[0]:,} 行 -> {len(feats)} 特征, {time.time()-t1:.0f}s')
        del df
    out = pd.concat(parts, ignore_index=True)
    # 分块可能把同一 sample_id 拆到两块 -> 去重（后块覆盖前块）
    out = out.groupby('sample_id', as_index=False).last()
    os.makedirs(OUT, exist_ok=True)
    path = f'{OUT}/market_{SPLIT}.parquet'
    out.to_parquet(path, index=False)
    print(f'完成: {path}, {out.shape}, 耗时 {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
