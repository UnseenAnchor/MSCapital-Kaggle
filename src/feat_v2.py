"""增强特征 v2：订单流精细特征 + 多窗口价格变化率 + 时间加权聚合"""
import numpy as np
import pandas as pd
import pyarrow.feather as pf
import time, os, sys

DATA = 'data'
OUT = 'features'
SPLIT = sys.argv[1] if len(sys.argv) > 1 else 'train'
CHUNK = 8_000_000


def process_market_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Market v2：多窗口变化率 + 价格位置 + 波动"""
    df['mid'] = (df['bid_price_1'] + df['ask_price_1']) / 2
    df['spread'] = df['ask_price_1'] - df['bid_price_1']
    df['imb1'] = (df['bid_volume_1'] - df['ask_volume_1']) / (df['bid_volume_1'] + df['ask_volume_1'] + 1e-9)
    df['t'] = -df['seconds_before_predict']

    g = df.groupby('sample_id')
    feats = pd.DataFrame({'sample_id': df['sample_id'].unique()}).set_index('sample_id')
    def add(n, s): feats[n] = s

    # 各窗口末值 vs 起点变化率 (mid)：近 10s / 30s / 60s / 120s / 300s
    for w, name in [(10, 's10'), (30, 's30'), (60, 's60'), (120, 's120'), (300, 's300')]:
        sub = df[df['t'] >= -w]
        if len(sub) > 0:
            sg = sub.groupby('sample_id')
            last = sg['mid'].last()
            first = sg['mid'].first()
            add(f'm_ret_{name}', (last - first) / (first + 1e-9))
            add(f'm_imb_{name}', sg['imb1'].mean())
            add(f'm_spread_{name}', sg['spread'].mean())
    # 价格位置：mid 在 [min,max] 中的位置
    rng = (g['mid'].max() - g['mid'].min() + 1e-9)
    add('m_mid_pos', (g['mid'].last() - g['mid'].min()) / rng)
    # 波动：mid 相邻差分 std
    def diff_std(s):
        return s.diff().std()
    add('m_mid_diffstd', g['mid'].apply(diff_std))
    add('m_imb_std', g['imb1'].std())
    # 成交强度
    add('m_vol_last10', df[df['t'] >= -10].groupby('sample_id')['transaction_volume'].sum())
    add('m_cnt_last10', df[df['t'] >= -10].groupby('sample_id')['transaction_count'].sum())
    return feats.reset_index()


def process_order_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Order v2：买卖方向×动作矩阵、价格偏移、事件节奏"""
    df['t'] = -df['seconds_before_predict']
    df['signed_vol'] = df['volume'] * (df['side'].astype(np.int8) * 2 - 1)
    g = df.groupby('sample_id')
    feats = pd.DataFrame({'sample_id': df['sample_id'].unique()}).set_index('sample_id')
    def add(n, s): feats[n] = s

    # 订单流不平衡（signed volume 累加）
    add('o_flow_imb', g['signed_vol'].sum())
    add('o_flow_imb_abs', g['signed_vol'].sum().abs())
    # 近 20s 的订单流
    sub = df[df['t'] >= -20]
    if len(sub) > 0:
        add('o_flow_imb_20s', sub.groupby('sample_id')['signed_vol'].sum())
        add('o_n_20s', sub.groupby('sample_id').size())
    # 价格偏移：订单价格相对均值
    pm = g['price'].mean()
    add('o_price_dev_mean', (df['price'] - df['sample_id'].map(pm)).abs().groupby(df['sample_id']).mean())
    # 事件间隔：时间差中位数
    def median_gap(s):
        return s.sort_values().diff().median()
    add('o_med_gap', g['t'].apply(median_gap))
    # 大单占比（volume > 分位数）
    v90 = df['volume'].quantile(0.9)
    add('o_big_frac', (df['volume'] > v90).groupby(df['sample_id']).mean())
    # 方向切换频率
    def side_flips(s):
        return (s.diff() != 0).sum() / max(len(s) - 1, 1)
    add('o_side_flips', g['side'].apply(side_flips))
    return feats.reset_index()


def process_tx_v2(df: pd.DataFrame) -> pd.DataFrame:
    """Tx v2：成交节奏、买卖压力细分、价格冲击"""
    df['t'] = -df['seconds_before_predict']
    df['signed_vol'] = df['volume'] * (df['side'].astype(np.int8) * 2 - 1)
    g = df.groupby('sample_id')
    feats = pd.DataFrame({'sample_id': df['sample_id'].unique()}).set_index('sample_id')
    def add(n, s): feats[n] = s

    add('x_flow_imb', g['signed_vol'].sum())
    add('x_flow_imb_abs', g['signed_vol'].sum().abs())
    sub = df[df['t'] >= -20]
    if len(sub) > 0:
        add('x_flow_imb_20s', sub.groupby('sample_id')['signed_vol'].sum())
        add('x_n_20s', sub.groupby('sample_id').size())
    # 价格冲击：单笔均价 vs 成交价
    add('x_vwap', (df['price'] * df['volume']).groupby(df['sample_id']).sum() / (g['volume'].sum() + 1e-9))
    # 每笔平均成交量
    add('x_avg_trade_size', g['volume'].mean())
    # 首尾价格差（已有一阶，这里用相对 spread 归一化）
    def median_gap(s):
        return s.sort_values().diff().median()
    add('x_med_gap', g['t'].apply(median_gap))
    def side_flips(s):
        return (s.diff() != 0).sum() / max(len(s) - 1, 1)
    add('x_side_flips', g['side'].apply(side_flips))
    return feats.reset_index()


def main():
    t0 = time.time()
    os.makedirs(OUT, exist_ok=True)
    for key, fn in [('market', process_market_v2), ('order', process_order_v2), ('transaction', process_tx_v2)]:
        src = f'{DATA}/{SPLIT}/{key}.feather'
        print(f'[{SPLIT}/{key}] ...')
        reader = pf.read_table(src)
        n = reader.num_rows
        parts = []
        for i in range(0, n, CHUNK):
            t1 = time.time()
            df = reader.slice(i, CHUNK).to_pandas()
            for c in ['transaction_volume', 'transaction_count', 'ask_volume_1', 'bid_volume_1',
                      'ask_volume_2', 'bid_volume_2', 'volume']:
                if c in df.columns:
                    df[c] = df[c].astype(np.int32)
            parts.append(fn(df))
            del df
            print(f'  块 {i//CHUNK} ok ({time.time()-t1:.0f}s)')
        out = pd.concat(parts, ignore_index=True)
        out = out.groupby('sample_id', as_index=False).last()
        path = f'{OUT}/{key}_{SPLIT}_v2.parquet'
        out.to_parquet(path, index=False)
        print(f'  完成: {path}, {out.shape}, 累计 {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
