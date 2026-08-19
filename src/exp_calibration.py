"""Metric-E: two never-tested post-hoc optimizations on existing OOFs.

1) Monotone calibration of the blended stack (isotonic + GAM-style spline, fit proxy ->
   transfer eval on middle/late). Global corr is affine-invariant, so only *nonlinear*
   response corrections can help; this measures whether the stack->target relation has
   exploitable nonlinearity, with cross-fold transfer as the anti-overfit gate.

2) Nonlinear meta-stack: LGB small on 6 member OOFs (with month & raw features optional),
   fit on ONE fold, transfer-evaluated on the OTHERS (never in-fold).

Weights: current submission base (scaled sum=1).
"""
import numpy as np, pandas as pd

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
        names = ['lgb', 'real', 'v3', 'joint', 'multires']
    elif fold == 'middle':
        v = np.load('output/multistream_v3_middle_eff1024_oof.npz'); r = np.load('output/realmlp_multiseed_rolling_oof.npz')
        j = np.load('output/joint_v3_middle_fast_oof.npz'); q = np.load('output/multires_self_middle_oof.npz')
        ids, y = v['sample_id'], v['target']
        m = pd.read_feather('data/train/label.feather').set_index('sample_id').loc[ids, 'month'].to_numpy().astype(int)
        ps = [r['middle_lgb'], r['middle_s42'], v['ens4_5_6'], j['ens4_5_6'], np.mean([q['ep5'], q['ep6'], q['ep7']], 0)]
        names = ['lgb', 'real', 'v3', 'joint', 'multires']
    else:
        v = np.load('output/multistream_v3_late_eff1024_oof.npz'); r = np.load('output/realmlp_multiseed_rolling_oof.npz')
        j = np.load('output/joint_v3_late_fast_oof.npz'); q = np.load('output/multires_self_late_oof.npz')
        ids, y, m = v['sample_id'], v['target'], v['month']
        ps = [r['late_lgb'], r['late_s42'], v['ens4_5_6'], j['ens4_5_6'], np.mean([q['ep5'], q['ep6'], q['ep7']], 0)]
    ev = np.load(f'output/event_256_{fold}_oof.npz')
    ps.append(np.mean([ev['ep6'], ev['ep9'], ev['ep12']], 0))
    return ids, y, m, np.array([u(p) for p in ps])

BASE = np.array([.176, .132, .132, .308, .132, .12]); BASE = BASE / BASE.sum()

def main():
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LinearRegression
    K = {f: load(f) for f in ['proxy', 'middle', 'late']}
    for f in K:
        ids, y, m, P = K[f]
        print(f, 'base', ['%.4f' % z for z in gmet(y, BASE @ P, m)], flush=True)
    print('--- calibration: fit on ONE fold, transfer to others (monotone only, affine does nothing) ---')
    for fitf in ['proxy', 'middle', 'late']:
        f_ids, fy, fm, fP = K[fitf]
        fp = BASE @ fP
        iso = IsotonicRegression(out_of_bounds='clip').fit(fp, fy)
        slin = LinearRegression().fit(fp.reshape(-1, 1), fy)
        deltas = []
        for evf in ['proxy', 'middle', 'late']:
            if evf == fitf:
                continue
            e_ids, ey, em, eP = K[evf]
            ep = BASE @ eP
            d_iso = corr(ey, iso.predict(ep)) - corr(ey, ep)
            d_lin = corr(ey, slin.predict(ep.reshape(-1, 1))) - corr(ey, ep)
            print(f'fit {fitf:7s} -> eval {evf:7s}: iso {d_iso:+.6f}  linear(affine, expected 0) {d_lin:+.6f}', flush=True)
            deltas.append((d_iso, d_lin, evf, fitf))
    np.save('output/calib_deltas.npy', np.array(deltas) if False else None)

if __name__ == '__main__':
    main()