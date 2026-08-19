"""Test crossscale-delta (Public 0.145, independently trained) as a 7th stack member.
Cheap: existing OOF. Weight chosen by fitting on proxy+middle, transfer-eval on late
(honest). Compare vs current 6-member baseline.
"""
import numpy as np, pandas as pd

def u(x): x = np.asarray(x, np.float64); x -= x.mean(); return x / (np.linalg.norm(x) + 1e-12)
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
    ev = np.load(f'output/event_256_{fold}_oof.npz'); ps.append(np.mean([ev['ep6'], ev['ep9'], ev['ep12']], 0))
    cs = np.load(f'output/crossscale_delta_{fold}_oof.npz')
    ps.append(np.mean([cs['ep4'], cs['ep5'], cs['ep6'], cs['ep7']], 0))  # 7th member (crossscale)
    return y, m, np.array([u(p) for p in ps])

BASE = np.array([.176, .132, .132, .308, .132, .12, 0.0])  # 7th weight 0
BASE /= BASE[:6].sum()  # renormalize 6-member part

def main():
    K = {f: load(f) for f in ['proxy', 'middle', 'late']}
    for f in K:
        y, m, P = K[f]
        cbase = BASE[:6] / BASE[:6].sum()
        print(f, '6-member base g/av/min=', ['%.4f' % z for z in gmet(y, cbase @ P[:6], m)],
              '| crossscale corr with v3/joint/event=', [round(float(c(P[2], P[6])), 3) if False else round(corr(u(P[2]), u(P[6])), 3) for _ in [0]], flush=True)
    # choose crossscale weight by fitting on proxy+middle, eval on late
    fy = np.concatenate([K['proxy'][0], K['middle'][0]]); fm = np.concatenate([K['proxy'][1], K['middle'][1]])
    FP = np.concatenate([K['proxy'][2], K['middle'][2]], 1)
    best = (-9, None)
    for w in np.arange(0.0, 0.30, 0.01):
        ww = np.array([0.176, .132, .132, .308, .132, .12, w]); ww = ww / ww.sum()
        s = gmet(fy, ww @ FP, fm)[0]
        if s > best[0]: best = (s, w)
    wb = best[1]
    print('best crossscale weight fit on proxy+middle:', round(wb, 3), 'score', round(best[0], 4), flush=True)
    for evf in ['late']:
        y, m, P = K[evf]
        w6 = np.array([0.176, .132, .132, .308, .132, .12]); w6 = w6 / w6.sum()
        bl = gmet(y, w6 @ P[:6], m)
        w7 = np.array([0.176, .132, .132, .308, .132, .12, wb]); w7 = w7 / w7.sum()
        nx = gmet(y, w7 @ P, m)
        print(f'eval {evf}: 6-member', ['%.4f' % z for z in bl], ' -> +crossscale', ['%.4f' % z for z in nx], flush=True)
    # also plain equal-ish inclusion
    y, m, P = K['late']
    w7 = np.array([.17, .13, .13, .30, .13, .12, .02]); w7 /= w7.sum()
    print('eval late w/ fixed crossscale 0.02 proxy/middle gener:', ['%.4f' % z for z in gmet(y, w7 @ P, m)], flush=True)

if __name__ == '__main__':
    main()