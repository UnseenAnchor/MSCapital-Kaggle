"""Fusion: strong grid model (market + flow) + raw per-event order/tx streams.
Reuses train_multistream_grid building blocks via import; raw event streams are
mask-aware pre-normalized (identity norm here). Strict chronological validation.
"""
import os, sys, time, threading, queue
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, 'src')
import train_multistream_grid as G

DEVICE = G.DEVICE
BATCH = int(os.environ.get('BATCH', '256'))
ACCUM = int(os.environ.get('ACCUM_STEPS', '1'))
EPOCHS = int(os.environ.get('EPOCHS', '10'))
TRAIN_END = int(os.environ.get('TRAIN_END', '45'))
VALID_END = int(os.environ.get('VALID_END', '71'))
SEED = int(os.environ.get('SEED', '42'))
OUT_PREFIX = os.environ.get('OUT_PREFIX', 'fusion_raw')
D_MODEL = int(os.environ.get('D_MODEL', '64'))
N_LAYERS = int(os.environ.get('N_LAYERS', '2'))
LR = float(os.environ.get('LR', '0.001'))
RO_LEN = int(os.environ.get('RAW_ORDER_LEN', '256'))
RT_LEN = int(os.environ.get('RAW_TX_LEN', '256'))
RO_CH, RT_CH = 4, 3

def raw_paths(split):
    return {k: f'{G.ROOT}/{split}_{G.GRID_VERSION}_{k}_{L}x{C}.mmap'
            for k, L, C in [('raworder', RO_LEN, RO_CH), ('rawtransaction', RT_LEN, RT_CH)]}

def load_all(split, n):
    A = G.arrays(split, n)
    p = raw_paths(split)
    A['raworder'] = np.memmap(p['raworder'], np.float16, 'r', shape=(n, RO_LEN, RO_CH))
    A['rawtransaction'] = np.memmap(p['rawtransaction'], np.float16, 'r', shape=(n, RT_LEN, RT_CH))
    return A

def to_ram(A):
    out = {}
    for k, a in A.items():
        out[k] = np.array(a, dtype=np.float16, copy=True, order='C')
    print('RAM loaded %.2fGB' % (sum(x.nbytes for x in out.values()) / 2**30), flush=True)
    return out

def norm_stats_grid(A, idx, nmax=50000):
    rng = np.random.default_rng(42)
    ii = np.sort(rng.choice(idx, min(nmax, len(idx)), replace=False))
    out = {}
    for k in ['market', 'tx', 'order']:
        a = A[k]
        s = np.zeros(a.shape[-1]); sq = np.zeros(a.shape[-1]); n = 0
        for j in range(0, len(ii), 2048):
            x = np.asarray(a[ii[j:j + 2048]], np.float32).reshape(-1, a.shape[-1])
            s += x.sum(0); sq += (x * x).sum(0); n += len(x)
        mu = np.nan_to_num(s / n, nan=0.0, posinf=0.0, neginf=0.0)
        sd = np.sqrt(np.maximum(np.nan_to_num(sq / n - mu * mu, nan=1.0, posinf=1.0, neginf=1.0), 1e-6))
        out[k] = (mu.astype(np.float32), sd.astype(np.float32))
    return out

class Prep:
    def __init__(self, norm):
        self.n = {k: (torch.as_tensor(v[0], device=DEVICE), torch.as_tensor(v[1], device=DEVICE)) for k, v in norm.items()}
    def one(self, k, x):
        z = torch.from_numpy(x).to(DEVICE)
        pad = z.abs().sum(-1) == 0
        if k in self.n:
            mu, sd = self.n[k]
            z = torch.nan_to_num(torch.clamp((z.float() - mu) / sd, -8, 8), nan=0., posinf=8., neginf=-8.)
        else:
            z = z.float()  # raw streams already normalized
        z[pad] = 0
        return z.transpose(1, 2)
    def batch(self, b):
        return (self.one('market', b[0]), self.one('tx', b[1]), self.one('order', b[2]),
                self.one('raworder', b[3]), self.one('rawtransaction', b[4]),
                torch.from_numpy(b[5]).to(DEVICE))

class Net(nn.Module):
    def __init__(self, d=None):
        super().__init__()
        d = d or D_MODEL
        self.m = G.Stream(G.M_CH, G.M_LEN, d, N_LAYERS)
        self.t = G.Stream(G.T_CH, G.F_LEN, d, max(1, N_LAYERS - 1))
        self.o = G.Stream(G.O_CH, G.F_LEN, d, max(1, N_LAYERS - 1))
        self.ro = G.Stream(RO_CH, RO_LEN, d, 1)
        self.rt = G.Stream(RT_CH, RT_LEN, d, 1)
        ntok = 5
        self.cross = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, d * 4, .1, 'gelu', batch_first=True, norm_first=True), 1)
        self.typ = nn.Parameter(torch.randn(1, ntok, d) * .02)
        self.h = nn.Sequential(nn.Linear(d * ntok * 2, d * 2), nn.GELU(), nn.Dropout(.1), nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, 1))
    def forward(self, m, t, o, ro, rt):
        z = [self.m(m), self.t(t), self.o(o), self.ro(ro), self.rt(rt)]
        raw = torch.cat(z, -1)
        mix = self.cross(torch.stack(z, 1) + self.typ).flatten(1)
        return self.h(torch.cat([raw, mix], -1)).squeeze(-1)

def ram_batches(A, idx, y, bs, shuffle, seed, drop_last=False):
    idx = np.asarray(idx).copy()
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(idx)
    q = queue.Queue(3)
    stop = object()
    def work():
        try:
            end = (len(idx) // bs) * bs if drop_last else len(idx)
            for i in range(0, end, bs):
                j = idx[i:min(i + bs, len(idx))]
                if len(j) < bs and drop_last:
                    break
                q.put((A['market'][j], A['tx'][j], A['order'][j], A['raworder'][j], A['rawtransaction'][j], y[j]))
        finally:
            q.put(stop)
    threading.Thread(target=work, daemon=True).start()
    while True:
        z = q.get()
        if z is stop:
            break
        yield z

def main():
    import random
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    lab = pd.read_feather('data/train/label.feather').sort_values('sample_id')
    n = len(lab)
    A = load_all('train', n)
    mo = lab.month.to_numpy(); y = lab.target.to_numpy(np.float32)
    tr = np.flatnonzero(mo < TRAIN_END); va = np.flatnonzero((mo >= TRAIN_END) & (mo < VALID_END))
    norm = norm_stats_grid(A, tr)
    A = to_ram(A)
    prep = Prep(norm)
    q = Net().to(DEVICE)
    try:
        opt = torch.optim.AdamW(q.parameters(), LR, weight_decay=1e-4, fused=DEVICE.type == 'cuda')
    except TypeError:
        opt = torch.optim.AdamW(q.parameters(), LR, weight_decay=1e-4)
    sc = torch.cuda.amp.GradScaler(enabled=DEVICE.type == 'cuda')
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    best = -1
    print('prefix', OUT_PREFIX, 'D', D_MODEL, 'layers', N_LAYERS, 'params %.2fM' % (sum(p.numel() for p in q.parameters()) / 1e6),
          'train', len(tr), 'val', len(va), 'batch', BATCH, 'accum', ACCUM, flush=True)
    for ep in range(EPOCHS):
        q.train(); tot = seen = 0; st = time.time(); opt.zero_grad(set_to_none=True); pending = 0
        for b in ram_batches(A, tr, y, BATCH, True, SEED + ep, drop_last=True):
            m, t, o, ro, rt, yy = prep.batch(b)
            with torch.cuda.amp.autocast(enabled=DEVICE.type == 'cuda'):
                loss = G.lossfn(q(m, t, o, ro, rt), yy)
            if not torch.isfinite(loss):
                opt.zero_grad(set_to_none=True); pending = 0; continue
            sc.scale(loss / ACCUM).backward(); pending += 1; tot += loss.item() * len(yy); seen += len(yy)
            if pending == ACCUM:
                sc.unscale_(opt); nn.utils.clip_grad_norm_(q.parameters(), 1); sc.step(opt); sc.update(); opt.zero_grad(set_to_none=True); pending = 0
        if pending:
            scale = ACCUM / pending
            for p in q.parameters():
                if p.grad is not None:
                    p.grad.mul_(scale)
            sc.unscale_(opt); nn.utils.clip_grad_norm_(q.parameters(), 1); sc.step(opt); sc.update(); opt.zero_grad(set_to_none=True)
        sch.step()
        q.eval(); po = []; yo = []
        with torch.no_grad():
            for b in ram_batches(A, va, y, BATCH * 2, False, SEED):
                m, t, o, ro, rt, yy = prep.batch(b)
                po.append(q(m, t, o, ro, rt).float().cpu().numpy()); yo.append(yy.cpu().numpy())
        p = np.concatenate(po); yt = np.concatenate(yo)
        raw = G.cos(yt, p); cen = G.cos(yt, p, True)
        print(f'epoch {ep + 1}/{EPOCHS}: loss={tot/seen:.5f} raw={raw:.5f} centered={cen:.5f} sec={time.time()-st:.0f}', flush=True)
        if raw > best:
            best = raw; torch.save(q.state_dict(), f'output/{OUT_PREFIX}_best.pt')
        torch.save(q.state_dict(), f'output/{OUT_PREFIX}_ep{ep + 1}.pt')
        np.savez(f'output/{OUT_PREFIX}_ep{ep + 1}_val.npz', pred=p, target=yt, month=mo[va], raw=raw, centered=cen)
    print('best_raw=', best)

if __name__ == '__main__':
    main()