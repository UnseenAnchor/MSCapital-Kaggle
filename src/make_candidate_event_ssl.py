"""Generate the SSL-pretrained (train+test domain) Event256 candidate.

1. Three-fold OOF stack comparison: event_ssl_tt vs event256(orig) vs no-event,
   with event-weight sensitivity (12/15/20%).
2. Test candidate: 40% self-stack (event member = ssl_tt, weight w) + 60% public ref.

The self-stack 5-member base weights follow ITERATION_EVENT256_FULL_SEQUENCE:
    [.176, .132, .132, .308, .132]  (lgb, real, v3, joint, multires)
event member takes weight w out of the self-stack, others rescaled.

Run: python src/make_candidate_event_ssl.py
"""
import hashlib
import os
import numpy as np
import pandas as pd

PUBLIC_REF = "research/lb0142/submission_ref_lb0142.csv"
MEMBERS = {
    "lgb": "output/submission_lgb_robust.csv",
    "real": "output/submission_realmlp_v4_unit.csv",
    "v3": "output/submission_multistream_v3_eff1024_unit.csv",
    "joint": "output/submission_joint_v3_fast_unit.csv",
    "multires": "output/diagnostic_multires_self_full_unit.csv",
}
EVENT_ORIG = "output/submission_event_256_unit.csv"
EVENT_SSL = os.environ.get("EVENT_SSL_CSV", "output/submission_event_ssl_tt_full_unit.csv")
USE_PL_OOF = os.environ.get("USE_PL_OOF", "0") == "1"  # compare against pseudo-label-finetuned fold OOFs
W5 = np.array([.176, .132, .132, .308, .132])
WEIGHTS = [0.12, 0.15, 0.20]


def u(x):
    x = np.asarray(x, np.float64)
    x = x - x.mean()
    return x / (np.linalg.norm(x) + 1e-12)


def corr(y, p):
    return float(u(y) @ u(p))


def gmet(y, p, m):
    v = [corr(y[m == q], p[m == q]) for q in np.unique(m)]
    return corr(y, p), float(np.mean(v)), float(min(v))


def posmap(arr):
    return {int(s): i for i, s in enumerate(arr)}


def load_oof(fold):
    """Return (y, m, Pu[5 members], evu, slu) aligned to the member OOF ids."""
    if fold == "proxy":
        x = np.load("output/proxy_lgb_oof.npz")
        v = np.load("output/multistream_v3_proxy_oof.npz")
        r = np.load("output/realmlp_multiseed_proxy_oof.npz")
        j = np.load("output/joint_v3_proxy_fast_oof.npz")
        q = np.load("output/multires_self_proxy_oof.npz")
        ids, y, m = x["sample_id"], x["target"], x["month"]
        P = [x["prediction"], r["s42"], v["ens4_5_6"], j["ens4_5_6"],
             np.mean([q["ep5"], q["ep6"], q["ep7"]], 0)]
    elif fold == "middle":
        v = np.load("output/multistream_v3_middle_eff1024_oof.npz")
        r = np.load("output/realmlp_multiseed_rolling_oof.npz")
        j = np.load("output/joint_v3_middle_fast_oof.npz")
        q = np.load("output/multires_self_middle_oof.npz")
        ids, y = v["sample_id"], v["target"]
        m = pd.read_feather("data/train/label.feather").set_index("sample_id").loc[ids, "month"].to_numpy().astype(int)
        P = [r["middle_lgb"], r["middle_s42"], v["ens4_5_6"], j["ens4_5_6"],
             np.mean([q["ep5"], q["ep6"], q["ep7"]], 0)]
    else:
        v = np.load("output/multistream_v3_late_eff1024_oof.npz")
        r = np.load("output/realmlp_multiseed_rolling_oof.npz")
        j = np.load("output/joint_v3_late_fast_oof.npz")
        q = np.load("output/multires_self_late_oof.npz")
        ids, y, m = v["sample_id"], v["target"], v["month"]
        P = [r["late_lgb"], r["late_s42"], v["ens4_5_6"], j["ens4_5_6"],
             np.mean([q["ep5"], q["ep6"], q["ep7"]], 0)]
    ev = np.load(f"output/event_256_{fold}_oof.npz")
    sl = np.load(f"output/event_ssl_tt_{fold}_pl_oof.npz" if USE_PL_OOF else f"output/event_ssl_tt_{fold}_oof.npz")
    pm = posmap(ev["sample_id"]); qm = posmap(sl["sample_id"])
    ev_p = np.mean([ev["ep6"], ev["ep9"], ev["ep12"]], 0)[[pm[int(s)] for s in ids]]
    sl_p = np.mean([sl["ep6"], sl["ep9"], sl["ep12"]], 0)[[qm[int(s)] for s in ids]]
    return y, m, np.array([u(p) for p in P]), u(ev_p), u(sl_p)


def main():
    print("=" * 78)
    print("THREE-FOLD OOF: event_ssl_tt vs event256(orig) vs no-event")
    print("=" * 78)
    rows = []
    for fold in ["proxy", "middle", "late"]:
        y, m, Pu, evu, slu = load_oof(fold)
        w5 = W5 / W5.sum()
        base = w5 @ Pu
        evo = W5.sum()  # 1.0
        # 5-member base (no event)
        g0 = gmet(y, base, m)
        for w in [0.12, 0.15, 0.20]:
            ww = np.concatenate([W5, [w]]); ww = ww / ww.sum()
            g_ev = gmet(y, ww[:5] @ Pu + ww[5] * evu, m)
            g_sl = gmet(y, ww[:5] @ Pu + ww[5] * slu, m)
            rows.append((fold, w, g_ev, g_sl))
            d = np.array(g_sl) - np.array(g_ev)
            print(f"{fold:7s} w={w:.2f} | orig {g_ev[0]:.5f}/{g_ev[1]:.5f}/{g_ev[2]:.5f} "
                  f"| ssl_tt {g_sl[0]:.5f}/{g_sl[1]:.5f}/{g_sl[2]:.5f} "
                  f"| delta {d[0]:+.5f}/{d[1]:+.5f}/{d[2]:+.5f}", flush=True)
        print(f"        | 5-member base {g0[0]:.5f}/{g0[1]:.5f}/{g0[2]:.5f}", flush=True)

    # ---- test candidate ----
    print("=" * 78)
    print("TEST CANDIDATE: unit(0.4*self_stack(w) + 0.6*public_ref)")
    print("=" * 78)
    ref = u(pd.read_csv(PUBLIC_REF).sort_values("sample_id").prediction.to_numpy())
    member_u = {k: u(pd.read_csv(p).sort_values("sample_id").prediction.to_numpy()) for k, p in MEMBERS.items()}
    ev_o = u(pd.read_csv(EVENT_ORIG).sort_values("sample_id").prediction.to_numpy())
    sl_u = u(pd.read_csv(EVENT_SSL).sort_values("sample_id").prediction.to_numpy())
    assert len(ref) == len(ev_o) == len(sl_u) == 647896
    w5 = W5 / W5.sum()
    base = sum(wi * member_u[k] for k, wi in zip(MEMBERS, w5))
    for w in WEIGHTS:
        ww = np.concatenate([W5, [w]]); ww = ww / ww.sum()
        self_sl = sum(ww[i] * member_u[k] for i, k in enumerate(MEMBERS)) + ww[5] * sl_u
        self_ev = sum(ww[i] * member_u[k] for i, k in enumerate(MEMBERS)) + ww[5] * ev_o
        final = u(0.4 * self_sl + 0.6 * ref)
        dest = f"output/candidate_event_ssl_tt_public60_40_w{int(w*100)}.csv"
        out = pd.DataFrame({"sample_id": np.arange(len(final)), "prediction": final})
        out.to_csv(dest, index=False)
        h = hashlib.sha256(open(dest, "rb").read()).hexdigest()
        print(f"w={w:.2f} corr_vs_orig_candidate={float(u(final) @ u(0.4*self_ev + 0.6*ref)):.5f} "
              f"corr_self_vs_orig_event={float(u(self_sl) @ u(self_ev)):.4f} -> {dest} sha256={h[:16]}", flush=True)


if __name__ == "__main__":
    main()
