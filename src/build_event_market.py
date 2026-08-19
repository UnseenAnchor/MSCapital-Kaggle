"""Merged event->market-aligned microstructure cache.

For each sampled raw order/tx event (the exact 256 slots of event_cache_v2), join the
order-book snapshot at-or-just-before that event's timestamp (asof-join, carry-forward),
and express the event's price/volume relative to that book state. This gives the model the
classical order-flow->price microstructure signal (book state at every event).

Output channels (recent-first, padding=0):
  order  (10): price_mid_rel, vol_depth_rel, side, action, spread, imb1, micro,
               sec/60, inter_arrival, position
  tx     ( 9): price_mid_rel, vol_depth_rel, side, spread, imb1, micro,
               sec/60, inter_arrival, position
Time channels (sec/60, inter_arrival, position) reuse event_cache_v2 time (channels 0,1,3).
"""
import os, time
import numpy as np
import pyarrow.feather as pf
from numba import njit, prange

ROOT = os.environ.get('GRID_ROOT', 'features/grid_v2')
GRID_VERSION = os.environ.get('GRID_VERSION', 'v2')
SRC = os.environ.get('EVENT_ROOT', 'features/event_cache_v2')
L = int(os.environ.get('RAW_EVENT_LEN', '256'))

M_COLS = ['transaction_avgprice', 'transaction_volume', 'transaction_count',
          'ask_price_1', 'ask_volume_1', 'bid_price_1', 'bid_volume_1',
          'ask_price_2', 'ask_volume_2', 'bid_price_2', 'bid_volume_2']


@njit(parallel=True)
def join_market(m_off, m_cnt, m_sec, m_feat, ev_feat, slot_sec, slot_mask, out, is_order):
    n = len(m_cnt)
    for i in prange(n):
        st = m_off[i]; T = m_cnt[i]
        if T <= 0:
            continue
        for j in range(L):
            if slot_mask[i, j] == 0:
                continue
            s = slot_sec[i, j]
            # lower_bound: smallest market index in [st, st+T) with m_sec >= s
            lo, hi = st, st + T
            while lo < hi:
                mid = (lo + hi) >> 1
                if m_sec[mid] < s:
                    lo = mid + 1
                else:
                    hi = mid
            if lo >= st + T:
                lo = st + T - 1  # event newer than newest snapshot -> carry last
            v = m_feat[lo]
            txavg = v[0]
            a1, av1, b1, bv1 = v[3], v[4], v[5], v[6]
            mid = 0.5 * (a1 + b1)
            spread = (a1 - b1) / max(mid, 1e-6)
            depth = av1 + bv1 + 1.0
            imb1 = (bv1 - av1) / max(av1 + bv1, 1.0)
            if txavg != txavg or txavg <= 0:
                micro = 0.0  # no transaction in this book snapshot -> neutral
            else:
                micro = (txavg - mid) / max(mid, 1e-6)
            price = ev_feat[i, j, 0]
            vol = ev_feat[i, j, 1]
            side = ev_feat[i, j, 2]
            out[i, j, 0] = (price - mid) / max(mid, 1e-6)
            out[i, j, 1] = vol / depth
            out[i, j, 2] = side
            if is_order:
                out[i, j, 3] = ev_feat[i, j, 3]  # action
                out[i, j, 4] = spread
                out[i, j, 5] = imb1
                out[i, j, 6] = micro
            else:
                out[i, j, 3] = spread
                out[i, j, 4] = imb1
                out[i, j, 5] = micro


def load_market(split):
    tab = pf.read_table(f'data/{split}/market.feather', memory_map=True)
    sid = tab['sample_id'].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    sec = tab['seconds_before_predict'].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    feat = np.stack([tab[c].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float32, copy=False) for c in M_COLS], axis=1)
    n = int(sid.max()) + 1
    cnt = np.bincount(sid, minlength=n).astype(np.int64)
    off = np.empty(n, np.int64); off[0] = 0; np.cumsum(cnt[:-1], out=off[1:])
    order = np.lexsort((sec, sid))  # sample-major, seconds ascending within sample
    sid_s, sec_s, feat_s = sid[order], sec[order], feat[order]
    assert np.array_equal(sid_s, np.repeat(np.arange(n), cnt)), 'market must be sample-major'
    return off, cnt, sec_s, feat_s


def analyze(arr, mask):
    C = arr.shape[-1]
    s = np.zeros(C, np.float64); sq = np.zeros(C, np.float64); ntot = 0
    for j in range(0, arr.shape[0], 4096):
        x = np.asarray(arr[j:j+4096], np.float32)
        mm = mask[j:j+4096, :, None].astype(np.float32)
        s += (x * mm).sum((0, 1)); sq += (x * x * mm).sum((0, 1)); ntot += mm.sum()
    mu = (s / ntot).astype(np.float32)
    sd = np.sqrt(np.maximum(sq / ntot - mu * mu, 1e-6)).astype(np.float32)
    return mu, sd, int(ntot)


def build(split, name, is_order):
    t = time.time()
    feat = np.load(f'{SRC}/{split}_{name}_feat.npy', mmap_mode='r')
    tim = np.load(f'{SRC}/{split}_{name}_time.npy', mmap_mode='r')
    n = feat.shape[0]
    C_out = 10 if is_order else 9
    C_ev = 4 if is_order else 3
    ev_feat = np.asarray(feat[:, :, :C_ev], np.float32)
    slot_sec = np.asarray(tim[:, :, 0], np.float32) * 60.0
    slot_mask = np.asarray(tim[:, :, 2], np.float32) > 0.5
    off, cnt, m_sec, m_feat = load_market(split)
    out = np.zeros((n, L, C_out), np.float32)
    join_market(off, cnt, m_sec, m_feat, ev_feat, slot_sec, slot_mask, out, is_order)
    # clip extreme relatives
    out[:, :, 0] = np.clip(out[:, :, 0], -0.5, 0.5)      # price_mid_rel
    out[:, :, 1] = np.clip(out[:, :, 1], -5.0, 5.0)      # vol_depth_rel
    lo = 4 if is_order else 3
    out[:, :, lo:C_out - 3] = np.clip(out[:, :, lo:C_out - 3], -2.0, 2.0)  # spread/imb/micro
    # append time channels (sec/60, inter_arrival, position)
    out[:, :, C_out - 3] = np.asarray(tim[:, :, 0], np.float32)
    out[:, :, C_out - 2] = np.asarray(tim[:, :, 1], np.float32)
    out[:, :, C_out - 1] = np.asarray(tim[:, :, 3], np.float32)
    mask = slot_mask.astype(np.float32)
    print(f'{split} {name} merged {out.shape} sec {time.time()-t:.0f}', flush=True)
    return out, mask


def main():
    mu_sd = {}
    for split in ['train', 'test']:
        for name, is_order in [('order', True), ('transaction', False)]:
            key = 'rawordermkt' if is_order else 'rawtxmkt'
            arr, mask = build(split, name, is_order)
            C = arr.shape[2]
            if split == 'train':
                mu, sd, ntot = analyze(arr, mask)
                mu_sd[key] = (mu, sd)
                print('train', key, 'active', ntot, 'mu', mu.round(3), 'sd', sd.round(3), flush=True)
            else:
                mu, sd = mu_sd[key]
            out = np.memmap(f'{ROOT}/{split}_{GRID_VERSION}_{key}_{L}x{C}.mmap', np.float16, 'w+', shape=(arr.shape[0], L, C))
            for j in range(0, arr.shape[0], 4096):
                x = np.asarray(arr[j:j+4096], np.float32)
                mm = mask[j:j+4096, :, None].astype(np.float32)
                z = np.nan_to_num(np.clip((x - mu) / sd, -8, 8), nan=0.0, posinf=8.0, neginf=-8.0) * mm
                out[j:j+4096] = z.astype(np.float16)
            print('wrote', key, out.shape, flush=True)
            del arr, mask, out
    print('done')

if __name__ == '__main__':
    main()