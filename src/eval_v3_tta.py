"""TTA (test-time augmentation) gate for v3: Gaussian-input-perturbation averaging.
Compare stored single-pass ep6 pred vs REPx TTA on the same fold (proxy/middle/late)."""
import os
os.environ.update(GRID_ROOT='features/grid_v3', GRID_VERSION='v3', MARKET_LEN='400', FLOW_LEN='120',
                  D_MODEL='64', N_LAYERS='2', BATCH='256')
import numpy as np, pandas as pd, torch
from train_multistream_grid import arrays, Net, DEVICE

PREFIX = 'multistream_v3_{fold}_eff1024'
SIG = 0.1
REPS = 4
EP = 6

def unit(x): x = np.asarray(x, np.float64); x -= x.mean(); return x / (np.linalg.norm(x) + 1e-12)
def stats(y, p, mo):
    v = [float(unit(y[mo == m]) @ unit(p[mo == m])) for m in np.unique(mo)]
    return float(unit(y) @ unit(p)), float(np.mean(v)), float(np.min(v))

def run(fold):
    z = np.load(f'features/grid_v3/norm_stats_{PREFIX.format(fold=fold)}.npz')
    mu = {k: torch.as_tensor(z[k + '_mean'], device=DEVICE) for k in ('market', 'tx', 'order')}
    sd = {k: torch.as_tensor(z[k + '_std'], device=DEVICE) for k in ('market', 'tx', 'order')}
    ofile = 'output/multistream_v3_proxy_oof.npz' if fold == 'proxy' else f'output/multistream_v3_{fold}_eff1024_oof.npz'
    oof = np.load(ofile)
    labt = pd.read_feather('data/train/label.feather').sort_values('sample_id')
    sid2pos = pd.Series(np.arange(len(labt)), index=labt.sample_id.to_numpy())
    va = sid2pos.loc[oof['sample_id']].to_numpy()
    y = labt.target.to_numpy(np.float32)[va]; mo = labt.month.to_numpy()[va]
    A = arrays('train', len(labt))
    q = Net().to(DEVICE)
    q.load_state_dict(torch.load(f'output/{PREFIX.format(fold=fold)}_ep{EP}.pt', map_location=DEVICE))
    q.eval()
    tta = np.zeros(len(va), np.float32)
    with torch.no_grad():
        for j in range(0, len(va), 1024):
            jj = va[j:j + 1024]
            acc = np.zeros(len(jj), np.float32)
            for r in range(REPS):
                rng = np.random.default_rng(1000 + r)
                inp = []
                for k in ('market', 'tx', 'order'):
                    x = np.asarray(A[k][jj], np.float32).copy()
                    pad = np.abs(x).sum(-1) == 0
                    x = np.nan_to_num(np.clip(x, -8, 8), nan=0.0, posinf=8.0, neginf=-8.0)
                    x[~pad] += rng.normal(0, SIG, x[~pad].shape).astype(np.float32)
                    x[pad] = 0
                    tz = torch.from_numpy(x).to(DEVICE)
                    out = torch.nan_to_num(torch.clamp((tz.float() - mu[k]) / sd[k], -8, 8), nan=0., posinf=8., neginf=-8.)
                    out[pad] = 0
                    inp.append(out.transpose(1, 2))
                acc += q(*inp).float().cpu().numpy()
            tta[j:j + 1024] = acc / REPS
    plain = oof[f'ep{EP}']
    print(f'[{fold}] plain g/av/min =', ['%.4f' % x for x in stats(y, plain, mo)], flush=True)
    print(f'[{fold}] TTAx{REPS} g/av/min =', ['%.4f' % x for x in stats(y, tta, mo)], flush=True)
    d = stats(y, tta, mo)[0] - stats(y, plain, mo)[0]
    print(f'[{fold}] TTA delta global = {d:+.6f}', flush=True)
    np.savez(f'output/v3_tta_{fold}.npz', tta=tta, plain=plain, y=y, month=mo)

def main():
    import time
    for fold in ['proxy', 'middle', 'late']:
        t0 = time.time(); run(fold); print(f'[{fold}] {time.time()-t0:.0f}s', flush=True)

if __name__ == '__main__':
    main()