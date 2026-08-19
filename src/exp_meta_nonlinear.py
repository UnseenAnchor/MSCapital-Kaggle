"""Nonlinear LGB meta-stack over the 6 members, with HONEST cross-fold transfer:
meta fit on fold(s) A -> evaluate on fold(s) B (never in-fold).
The late fold is the most test-like, so proxy+middle -> late is the key gate.
"""
import numpy as np, pandas as pd
import lightgbm as lgb

def u(x): x = np.asarray(x, np.float64); x = x - x.mean(); return x / (np.linalg.norm(x) + 1e-12)
def corr(y, p): return float(u(y) @ u(p))
def gmet(y, p, m):
    v = [corr(y[m == q], p[m == q]) for q in np.unique(m)]
    return corr(y, p), float(np.mean(v)), float(min(v))

def load(fold):
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
        m = pd.read_feather('data/train/label.feather').set_index('sample_id').loc[ids, 'month'].to_numpy().astype(int)
        ps = [r['middle_lgb'], r['middle_s42'], v['ens4_5_6'], j['ens4_5_6'], np.mean([q['ep5'], q['ep6'], q['ep7']], 0)]
    else:
        v = np.load('output/multistream_v3_late_eff1024_oof.npz'); r = np.load('output/realmlp_multiseed_rolling_oof.npz')
        j = np.load('output/joint_v3_late_fast_oof.npz'); q = np.load('output/multires_self_late_oof.npz')
        ids, y, m = v['sample_id'], v['target'], v['month']
        ps = [r['late_lgb'], r['late_s42'], v['ens4_5_6'], j['ens4_5_6'], np.mean([q['ep5'], q['ep6'], q['ep7']], 0)]
    ev = np.load(f'output/event_256_{fold}_oof.npz')
    ps.append(np.mean([ev['ep6'], ev['ep9'], ev['ep12']], 0))
    P = np.array([u(p) for p in ps]).T
    return ids, y, m, P

BASE = np.array([.176, .132, .132, .308, .132, .12]); BASE = BASE / BASE.sum()

def main():
    K = {f: load(f) for f in ['proxy', 'middle', 'late']}
    for f in K:
        _, y, m, P = K[f]
        print(f, 'base blend g/av/min=', ['%.4f' % z for z in gmet(y, P @ BASE, m)], flush=True)
    combos = [('proxy', 'middle'), ('proxy', 'late'), ('middle', 'late'), ('proxy+middle', 'late'), ('middle+late', 'proxy'), ('proxy+late', 'middle'), ('proxy+middle+late', None)]
    for fitf, evf in combos:
        names = fitf.split('+')
        Y = np.concatenate([K[n][1] for n in names]); M = np.concatenate([K[n][2] for n in names]); X = np.concatenate([K[n][3] for n in names])
        Xm = np.column_stack([X, M / 100.0])
        for use_month in [False, True]:
            XX = Xm if use_month else X
            # nested month-based holdout for early stopping
            rng = np.random.default_rng(3)
            d = lgb.Dataset(XX, Y)
            mdl = lgb.train({'objective': 'regression', 'metric': 'l2', 'learning_rate': 0.05,
                             'num_leaves': 8, 'min_data_in_leaf': 4000, 'feature_fraction': 0.8,
                             'bagging_fraction': 0.8, 'bagging_freq': 1, 'verbosity': -1, 'seed': 3},
                            d, num_boost_round=600)
            if evf:
                ey, em, eP = K[evf][1], K[evf][2], K[evf][3]
                ep_base = eP @ BASE
                emeta = mdl.predict(np.column_stack([eP, em / 100.0]) if use_month else eP)
                d = corr(ey, emeta) - corr(ey, ep_base)
                g = gmet(ey, emeta, em)
                print(f'meta fit {fitf:16s} -> eval {evf:5s} month={use_month}: delta {d:+.6f}  g/av/min {g[0]:.4f}/{g[1]:.4f}/{g[2]:.4f}', flush=True)
            else:
                selfs = corr(Y, mdl.predict(XX))
                print(f'fit {fitf:16s} self-train corr {selfs:.4f} (in-fold, overfit bound)', flush=True)

if __name__ == '__main__':
    main()