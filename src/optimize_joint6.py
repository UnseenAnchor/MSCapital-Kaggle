"""Joint 6-member weight re-optimization INCLUDING event256 (never done: event was added at
fixed 12% with others scaled). Optimize over proxy/middle/late with the same scoring
used for the 5-member search, then refine with a finer grid + monthwise gate.
"""
import numpy as np, pandas as pd, itertools

def u(x): x = np.asarray(x, np.float64); x = x - x.mean(); return x / (np.linalg.norm(x) + 1e-12)
def c(y, p): return float(u(y) @ u(p))
def met(y, p, m):
    v = [c(y[m == q], p[m == q]) for q in np.unique(m)]
    return c(y, p), float(np.mean(v)), float(min(v)), float(np.std(v))

def load(fold):
    ps = []
    if fold == 'proxy':
        x = np.load('output/proxy_lgb_oof.npz'); v = np.load('output/multistream_v3_proxy_oof.npz')
        r = np.load('output/realmlp_multiseed_proxy_oof.npz'); j = np.load('output/joint_v3_proxy_fast_oof.npz')
        q = np.load('output/multires_self_proxy_oof.npz')
        ids, y, m = x['sample_id'], x['target'], x['month']
        ps = [x['prediction'], r['s42'], v['ens4_5_6'], j['ens4_5_6'], np.mean([q['ep5'], q['ep6'], q['ep7']], 0)]
    elif fold == 'middle':
        v = np.load('output/multistream_v3_middle_eff1024_oof.npz'); r = np.load('output/realmlp_multiseed_rolling_oof.npz')
        j = np.load('output/joint_v3_middle_fast_oof.npz'); q = np.load('output/multires_self_middle_oof.npz')
        ids, y = v['sample_id'], v['target']
        m = pd.read_feather('data/train/label.feather').set_index('sample_id').loc[ids, 'month'].to_numpy()
        ps = [r['middle_lgb'], r['middle_s42'], v['ens4_5_6'], j['ens4_5_6'], np.mean([q['ep5'], q['ep6'], q['ep7']], 0)]
        m = m.astype(np.int64)
    else:
        v = np.load('output/multistream_v3_late_eff1024_oof.npz'); r = np.load('output/realmlp_multiseed_rolling_oof.npz')
        j = np.load('output/joint_v3_late_fast_oof.npz'); q = np.load('output/multires_self_late_oof.npz')
        ids, y, m = v['sample_id'], v['target'], v['month']
        ps = [r['late_lgb'], r['late_s42'], v['ens4_5_6'], j['ens4_5_6'], np.mean([q['ep5'], q['ep6'], q['ep7']], 0)]
    ev = np.load(f'output/event_256_{fold}_oof.npz')
    ps.append(np.mean([ev['ep6'], ev['ep9'], ev['ep12']], 0))
    P = np.array([u(p) for p in ps])
    print(fold, 'shapes', P.shape, 'names', ['lgb', 'real', 'v3', 'joint', 'multires', 'event'], flush=True)
    return ids, y, m, P

def score_of(w, A):
    vals = [met(y, w @ P, m) for y, m, P in A.values()]
    g = np.array([x[0] for x in vals]); av = np.array([x[1] for x in vals]); lo = np.array([x[2] for x in vals])
    return .4 * g.mean() + .3 * av.mean() + .2 * lo.mean() - .1 * g.std(), vals

def main():
    names = ['lgb', 'real', 'v3', 'joint', 'multires', 'event']
    A = {k: load(k)[1:] for k in ['proxy', 'middle', 'late']}
    base = np.array([.176, .132, .132, .308, .132, .12]); base = base / base.sum()
    print('baseline (scaled) score', round(score_of(base, A)[0], 6), flush=True)
    # coarse simplex grid, step 0.1
    best = []
    for w in itertools.product(np.arange(0, 1.0001, .1), repeat=5):
        last = 1 - sum(w)
        if last < -1e-9: continue
        ww = np.array((*w, last))
        if ww.min() < 0: continue
        s, vals = score_of(ww, A)
        best.append((s, ww, vals))
    best.sort(key=lambda x: -x[0])
    print('coarse grid combos', len(best), 'top-1', round(best[0][0], 6), flush=True)
    for s, w, vals in best[:12]:
        print(round(s, 6), dict(zip(names, w.round(2))), [round(z[0], 5) for z in vals], flush=True)
    # local refinement: random walk around top-8
    rng = np.random.default_rng(7)
    refined = []
    for s0, w0, _ in best[:8]:
        w = w0.copy()
        for _ in range(4000):
            nw = np.clip(w + rng.normal(0, .02, 6), 0, 1)
            nw = nw / nw.sum()
            s, vals = score_of(nw, A)
            if s > s0:
                s0, w, vw = s, nw, vals
        refined.append((s0, w, vw))
    refined.sort(key=lambda x: -x[0])
    print('REFINED TOP-10 (joint incl event):', flush=True)
    for s, w, vals in refined[:10]:
        print(round(s, 6), dict(zip(names, w.round(3))), [z[0] for z in vals], flush=True)
    np.save('output/joint6_best_weights.npy', np.array([w for _, w, _ in refined[:10]]))
    for _, w, vals in refined[:3]:
        print('unit-sum final (6): ', np.round(w, 4), [(round(v[0], 5), round(v[1], 5), round(v[2], 5)) for v in vals], flush=True)

if __name__ == '__main__':
    main()