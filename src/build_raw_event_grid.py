"""Build mask-aware normalized raw order/tx event mmap caches for the strong grid model.
v3 currently aggregates order/tx into 60 one-second buckets (flow); these raw streams keep
per-event resolution (256 events, recent-first). Padding is zero; real events always have
price>0 so the grid `abs().sum(-1)==0` padding detector stays valid.
"""
import os, time, numpy as np
ROOT = os.environ.get('GRID_ROOT', 'features/grid_v2')
GRID_VERSION = os.environ.get('GRID_VERSION', 'v2')
SRC = os.environ.get('EVENT_ROOT', 'features/event_cache_v2')
L = int(os.environ.get('RAW_EVENT_LEN', '256'))

def active_mask(split, name):
    t = np.load(f'{SRC}/{split}_{name}_time.npy', mmap_mode='r')
    return (np.asarray(t[:, :, 2], np.float32) > 0.5)  # channel2 = activity

def stats(split, name, C):
    feat = np.load(f'{SRC}/{split}_{name}_feat.npy', mmap_mode='r')
    m = active_mask(split, name)
    s = np.zeros(C, np.float64); sq = np.zeros(C, np.float64); n = 0
    for j in range(0, feat.shape[0], 4096):
        x = np.asarray(feat[j:j + 4096], np.float32)
        mm = m[j:j + 4096, :, None].astype(np.float32)
        s += (x * mm).sum((0, 1)); sq += (x * x * mm).sum((0, 1)); n += mm.sum()
    mu = s / n; sd = np.sqrt(np.maximum(sq / n - mu * mu, 1e-6))
    return mu.astype(np.float32), sd.astype(np.float32), n

def build(split, name, C, mu, sd):
    feat = np.load(f'{SRC}/{split}_{name}_feat.npy', mmap_mode='r')
    m = active_mask(split, name)
    out = np.memmap(f'{ROOT}/{split}_{GRID_VERSION}_raw{name}_{L}x{C}.mmap', np.float16, 'w+',
                    shape=(feat.shape[0], L, C))
    for j in range(0, feat.shape[0], 4096):
        x = np.asarray(feat[j:j + 4096], np.float32)
        mm = m[j:j + 4096, :, None].astype(np.float32)
        z = np.nan_to_num(np.clip((x - mu) / sd, -8, 8), nan=0.0, posinf=8.0, neginf=-8.0) * mm
        out[j:j + 4096] = z.astype(np.float16)
    print(f'{split} raw{name} {out.shape} done', flush=True)

def main():
    t = time.time()
    for name, C in [('order', 4), ('transaction', 3)]:
        mu, sd, n = stats('train', name, C)
        print('train', name, 'active_events', int(n), 'mu', mu.round(3), 'sd', sd.round(3), flush=True)
        build('train', name, C, mu, sd)
        build('test', name, C, mu, sd)
    print('raw event grid built in %.0fs' % (time.time() - t), flush=True)

if __name__ == '__main__':
    main()