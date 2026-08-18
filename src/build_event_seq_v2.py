"""Build full-length raw order/tx event cache (default L=256) with numba.
Fixes the L=32 truncation that limited the event Transformer's information coverage.
Features are stored RAW (un-normalized); the model's Prep.one fits stats on the fly."""
from pathlib import Path
import os, time, numpy as np, pyarrow.feather as pf
from numba import njit, prange
ROOT = Path('features/event_cache_v2'); ROOT.mkdir(parents=True, exist_ok=True)
L = int(os.environ.get('EVENT_LEN', '256'))

@njit(parallel=True)
def fill_feat(valmat, sid, offsets, counts, n, L, C, out):
    for i in prange(n):
        st = offsets[i]; T = counts[i]
        if T <= 0:
            continue
        if T >= L:
            # even-spaced, recent-first
            for j in range(L):
                k = (j * (T - 1)) // (L - 1)  # k=0..T-1, j=0 -> oldest, j=L-1 -> newest
                src = st + T - 1 - k  # recent-first orientation
                for c in range(C):
                    out[i, j, c] = valmat[src, c]
        else:
            pad = L - T
            for j in range(L):
                if j < pad:
                    continue  # keep zero padding (masked)
                k = j - pad  # 0..T-1
                src = st + T - 1 - k
                for c in range(C):
                    out[i, j, c] = valmat[src, c]


@njit(parallel=True)
def build_time(sec, offsets, counts, n, L, out):
    for i in prange(n):
        st = offsets[i]; T = counts[i]
        if T <= 0:
            continue
        for j in range(L):
            active = 1
            if T >= L:
                k = (j * (T - 1)) // (L - 1)
            else:
                pad = L - T
                if j < pad:
                    k = 0; active = 0
                else:
                    k = j - pad
            src = st + T - 1 - k
            out[i, j, 0] = sec[src] / 60.0
            out[i, j, 2] = active
            out[i, j, 3] = k / max(T - 1, 1)
        for j in range(L - 1):
            if out[i, j, 2] > 0 and out[i, j + 1, 2] > 0:
                out[i, j, 1] = np.log1p(abs(float(out[i, j + 1, 0] - out[i, j, 0])) * 60.0) / np.log(61.0)

def one(split, name, cols):
    t = time.time()
    tab = pf.read_table(f'data/{split}/{name}.feather', memory_map=True)
    sid = tab['sample_id'].combine_chunks().to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    sec = tab['seconds_before_predict'].combine_chunks().to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    valmat = np.stack([tab.column(c).combine_chunks().to_numpy(zero_copy_only=False).astype(np.float32, copy=False) for c in cols], axis=1)
    n = int(sid.max()) + 1
    counts = np.bincount(sid, minlength=n).astype(np.int64)
    offsets = np.empty(n, np.int64); offsets[0] = 0; np.cumsum(counts[:-1], out=offsets[1:])
    order = np.argsort(sid, kind='stable')
    sid_s = sid[order]; val_s = valmat[order]; sec_s = sec[order]
    assert np.array_equal(sid_s, np.repeat(np.arange(n), counts)), 'rows must be sample-major'
    C = len(cols)
    feat = np.zeros((n, L, C), np.float32)
    fill_feat(val_s, sid_s, offsets, counts, n, L, C, feat)
    np.save(f'features/event_cache_v2/{split}_{name}_feat.npy', feat)
    tim = np.zeros((n, L, 4), np.float32)
    build_time(sec_s, offsets, counts, n, L, tim)
    np.save(f'features/event_cache_v2/{split}_{name}_time.npy', tim.astype(np.float16))
    print(split, name, 'rows', n, 'shape', feat.shape, 'count q', np.quantile(counts, [0, .5, .9, .99, 1]).round(1), 'sec', round(time.time() - t), flush=True)

def main():
    one('train', 'order', ['price', 'volume', 'side', 'order_action'])
    one('train', 'transaction', ['price', 'volume', 'side'])
    one('test', 'order', ['price', 'volume', 'side', 'order_action'])
    one('test', 'transaction', ['price', 'volume', 'side'])

if __name__ == '__main__':
    main()