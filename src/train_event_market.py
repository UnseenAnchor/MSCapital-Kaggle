"""Event-aligned microstructure model: raw order/tx events joined to the order-book state
at each event (book-relative price/volume). Two streams (ordermkt, txmkt) -> cross -> head.
Reuses train_multistream_grid Stream encoder. Strict chronological validation.
"""
import os, sys, time, threading, queue, random
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F
sys.path.insert(0, 'src')
import train_multistream_grid as G

DEVICE = G.DEVICE
BATCH = int(os.environ.get('BATCH', '256'))
ACCUM = int(os.environ.get('ACCUM_STEPS', '4'))
EPOCHS = int(os.environ.get('EPOCHS', '10'))
TRAIN_END = int(os.environ.get('TRAIN_END', '45'))
VALID_END = int(os.environ.get('VALID_END', '71'))
SEED = int(os.environ.get('SEED', '42'))
OUT_PREFIX = os.environ.get('OUT_PREFIX', 'eventmkt')
D_MODEL = int(os.environ.get('D_MODEL', '64'))
N_LAYERS = int(os.environ.get('N_LAYERS', '2'))
LR = float(os.environ.get('LR', '0.001'))
L = int(os.environ.get('RAW_EVENT_LEN', '256'))
OC, TC = 10, 9

def paths(split):
    return {k: f'{G.ROOT}/{split}_{G.GRID_VERSION}_{k}_{L}x{C}.mmap'
            for k, C in [('rawordermkt', OC), ('rawtxmkt', TC)]}

def load(split, n):
    p = paths(split)
    return {'o': np.memmap(p['rawordermkt'], np.float16, 'r', shape=(n, L, OC)),
            't': np.memmap(p['rawtxmkt'], np.float16, 'r', shape=(n, L, TC))}

def to_ram(A):
    out = {k: np.array(a, dtype=np.float16, copy=True, order='C') for k, a in A.items()}
    print('RAM %.2fGB' % (sum(x.nbytes for x in out.values()) / 2**30), flush=True)
    return out

class Prep:
    def batch(self, b):
        o = torch.from_numpy(b[0]).to(DEVICE).float().transpose(1, 2)
        t = torch.from_numpy(b[1]).to(DEVICE).float().transpose(1, 2)
        return o, t, torch.from_numpy(b[2]).to(DEVICE)

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        d = D_MODEL
        self.o = G.Stream(OC, L, d, N_LAYERS)
        self.t = G.Stream(TC, L, d, N_LAYERS)
        ntok = 2
        self.cross = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, 4, d * 4, .1, 'gelu', batch_first=True, norm_first=True), 1)
        self.typ = nn.Parameter(torch.randn(1, ntok, d) * .02)
        self.h = nn.Sequential(nn.Linear(d * ntok * 2, d * 2), nn.GELU(), nn.Dropout(.1), nn.Linear(d * 2, d), nn.GELU(), nn.Linear(d, 1))
    def forward(self, o, t):
        z = [self.o(o), self.t(t)]
        raw = torch.cat(z, -1)
        mix = self.cross(torch.stack(z, 1) + self.typ).flatten(1)
        return self.h(torch.cat([raw, mix], -1)).squeeze(-1)

def ram_batches(A, idx, y, bs, shuffle, seed, drop_last=False):
    idx = np.asarray(idx).copy()
    rng = np.random.default_rng(seed)
    if shuffle:
        rng.shuffle(idx)
    q = queue.Queue(3); stop = object()
    def work():
        try:
            end = (len(idx) // bs) * bs if drop_last else len(idx)
            for i in range(0, end, bs):
                j = idx[i:min(i + bs, len(idx))]
                if len(j) < bs and drop_last:
                    break
                q.put((A['o'][j], A['t'][j], y[j]))
        finally:
            q.put(stop)
    threading.Thread(target=work, daemon=True).start()
    while True:
        z = q.get()
        if z is stop:
            break
        yield z

def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    lab = pd.read_feather('data/train/label.feather').sort_values('sample_id')
    n = len(lab)
    A = to_ram(load('train', n))
    mo = lab.month.to_numpy(); y = lab.target.to_numpy(np.float32)
    tr = np.flatnonzero(mo < TRAIN_END); va = np.flatnonzero((mo >= TRAIN_END) & (mo < VALID_END))
    prep = Prep()
    q = Net().to(DEVICE)
    try:
        opt = torch.optim.AdamW(q.parameters(), LR, weight_decay=1e-4, fused=DEVICE.type == 'cuda')
    except TypeError:
        opt = torch.optim.AdamW(q.parameters(), LR, weight_decay=1e-4)
    sc = torch.cuda.amp.GradScaler(enabled=DEVICE.type == 'cuda')
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    best = -1
    print('prefix', OUT_PREFIX, 'D', D_MODEL, 'layers', N_LAYERS, 'C', OC, TC, 'L', L,
          'params %.2fM' % (sum(p.numel() for p in q.parameters()) / 1e6), 'train', len(tr), 'val', len(va),
          'batch', BATCH, 'accum', ACCUM, flush=True)
    for ep in range(EPOCHS):
        q.train(); tot = seen = 0; st = time.time(); opt.zero_grad(set_to_none=True); pending = 0
        for b in ram_batches(A, tr, y, BATCH, True, SEED + ep, drop_last=True):
            o, t, yy = prep.batch(b)
            with torch.cuda.amp.autocast(enabled=DEVICE.type == 'cuda'):
                loss = G.lossfn(q(o, t), yy)
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
                o, t, yy = prep.batch(b)
                po.append(q(o, t).float().cpu().numpy()); yo.append(yy.cpu().numpy())
        p = np.concatenate(po); yt = np.concatenate(yo)
        raw = G.cos(yt, p); cen = G.cos(yt, p, True)
        print(f'epoch {ep + 1}/{EPOCHS}: loss={tot/seen:.5f} raw={raw:.5f} centered={cen:.5f} sec={time.time()-st:.0f}', flush=True)
        np.savez(f'output/{OUT_PREFIX}_ep{ep + 1}_val.npz', pred=p, target=yt, month=mo[va], raw=raw, centered=cen)
        if raw > best:
            best = raw; torch.save(q.state_dict(), f'output/{OUT_PREFIX}_best.pt')
        torch.save(q.state_dict(), f'output/{OUT_PREFIX}_ep{ep + 1}.pt')
    print('best_raw=', best)

if __name__ == '__main__':
    main()