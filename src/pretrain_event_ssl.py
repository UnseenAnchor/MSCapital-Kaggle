"""SSL pretraining for the raw-event Transformer (Event256 input space).

Tasks (all on ACTIVE tokens only):
  1. masked feature reconstruction  (Huber on standardized raw features, 15% masked)
  2. side regression               (MSE on side channel)
  3. time-interval regression      (Huber on normalized log-delta)

Encoder submodules (conv/tr/pos) are structurally identical to the supervised
EventEncoder in train_event_v2.py, so weights transfer via matching state_dict keys.

Usage (typically called from train_event_ssl.py, but runnable standalone):
    FOLD=proxy python src/pretrain_event_ssl.py
"""
import os, time, random, queue, threading
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BS = int(os.environ.get("BS", "1024"))
SEED = int(os.environ.get("SEED", "42"))
D = int(os.environ.get("D_MODEL", "64"))
NL = int(os.environ.get("N_LAYERS", "2"))
L = int(os.environ.get("EVENT_LEN", "256"))
SSL_EPOCHS = int(os.environ.get("SSL_EPOCHS", "8"))
MASK_RATIO = float(os.environ.get("MASK_RATIO", "0.15"))
LR = float(os.environ.get("SSL_LR", "3e-4"))
TRAIN_END = int(os.environ.get("TRAIN_END", "45"))
VALID_END = int(os.environ.get("VALID_END", "71"))
TROOT = os.environ.get("EVENT_ROOT", "features/event_cache_v2")
OUT = os.environ.get("OUT_PREFIX", "event_ssl_proxy")
# Domain adaptation: include ALL test streams (no labels needed) in SSL pretraining.
# This is the key difference vs the train-only SSL probe: the encoder sees test-domain
# distributions, attacking the OOF->Public transfer gap directly.
INCLUDE_TEST = os.environ.get("SSL_INCLUDE_TEST", "0") == "1"

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def load_arrays(split):
    t = time.time()
    A = {
        "tx": np.array(np.load(f"{TROOT}/{split}_transaction_feat.npy", mmap_mode="r"), copy=True),
        "order": np.array(np.load(f"{TROOT}/{split}_order_feat.npy", mmap_mode="r"), copy=True),
        "tx_time": np.array(np.load(f"{TROOT}/{split}_transaction_time.npy", mmap_mode="r"), copy=True),
        "order_time": np.array(np.load(f"{TROOT}/{split}_order_time.npy", mmap_mode="r"), copy=True),
    }
    print(split, "RAM GB", round(sum(x.nbytes for x in A.values()) / 2**30, 2),
          "sec", round(time.time() - t), flush=True)
    return A


def fit_stats(A, idx, nmax=50000):
    ii = np.sort(np.random.default_rng(42).choice(idx, min(nmax, len(idx)), replace=False))
    out = {}
    for k in ("tx", "order"):
        x = np.asarray(A[k][ii], np.float32)
        mask = np.asarray(A[k + "_time"][ii, :, 2] > 0)
        z = x[mask]
        out[k] = (z.mean(0).astype("f4"), np.maximum(z.std(0), 1e-3).astype("f4"))
    return out


def batches(A, idx, bs, shuffle, seed, maxq=3, At=None, n_train=0):
    idx = np.asarray(idx).copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    q = queue.Queue(maxq)
    stop = object()

    def grab(src, j):
        return (src["tx"][j], src["order"][j], src["tx_time"][j], src["order_time"][j])

    def work():
        try:
            for i in range(0, len(idx), bs):
                j = idx[i:i + bs]
                if At is None:
                    q.put(grab(A, j))
                else:
                    tr = j[j < n_train]
                    te = j[j >= n_train] - n_train
                    if len(tr) and len(te):
                        b1, b2 = grab(A, tr), grab(At, te)
                        q.put(tuple(np.concatenate([a, b]) for a, b in zip(b1, b2)))
                    elif len(tr):
                        q.put(grab(A, tr))
                    else:
                        q.put(grab(At, te))
        finally:
            q.put(stop)

    threading.Thread(target=work, daemon=True).start()
    while True:
        z = q.get()
        if z is stop:
            break
        yield z


class Prep:
    """Same normalization as train_event_v2.Prep; returns token inputs and masks."""

    def __init__(self, stats):
        self.stats = {k: (torch.tensor(a, device=DEVICE), torch.tensor(b, device=DEVICE))
                      for k, (a, b) in stats.items()}

    def one(self, k, v, t):
        v = torch.from_numpy(v).to(DEVICE).float()
        t = torch.from_numpy(t).to(DEVICE).float()
        mu, sd = self.stats[k]
        vn = torch.clamp((v - mu) / sd, -8, 8)
        mask = t[:, :, 2] > 0
        return vn, t, mask  # (B,L,C) (B,L,4) (B,L) bool

    def batch(self, b):
        tx_v, tx_t, tx_m = self.one("tx", b[0], b[2])
        o_v, o_t, o_m = self.one("order", b[1], b[3])
        return tx_v, tx_t, tx_m, o_v, o_t, o_m


def build_token_input(vn, t, k):
    """Replicate train_event_v2.Prep.one feature assembly (without masking)."""
    dv = F.pad(vn[:, 1:, :2] - vn[:, :-1, :2], (0, 0, 1, 0))
    inter = (vn[:, :, 1:2] * vn[:, :, 2:3]) if k == "tx" else (vn[:, :, 2:3] * vn[:, :, 3:4])
    x = torch.cat([vn, t, dv, inter], -1)  # (B,L,D_in)
    return x


class Conv(nn.Module):
    def __init__(self, a, b, k):
        super().__init__()
        self.n = nn.Sequential(
            nn.Conv1d(a, b, k, padding=k // 2, bias=False), nn.BatchNorm1d(b), nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(b, b, k, padding=k // 2, bias=False), nn.BatchNorm1d(b), nn.GELU())
        self.s = nn.Conv1d(a, b, 1, bias=False)

    def forward(self, x):
        return self.n(x) + self.s(x)


class SSLEncoder(nn.Module):
    """Structurally identical front-end to supervised EventEncoder (conv/tr/pos)."""

    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(Conv(c, D, 3), Conv(D, D, 3))
        el = nn.TransformerEncoderLayer(D, 4, D * 4, 0.1, "gelu", batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(el, NL)
        self.pos = nn.Parameter(torch.randn(1, L, D) * 0.02)

    def forward(self, x, mask):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2) + self.pos
        return self.tr(h, src_key_padding_mask=~mask)  # (B,L,D)


class SSLHeads(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.recon = nn.Linear(D, c)     # reconstruct standardized raw features
        self.side = nn.Linear(D, 1)      # side regression
        self.interval = nn.Linear(D, 1)  # time delta regression

    def forward(self, h):
        return self.recon(h), self.side(h), self.interval(h)


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    lab = pd.read_feather("data/train/label.feather").sort_values("sample_id").reset_index(drop=True)
    mo = lab.month.to_numpy()
    A = load_arrays("train")
    tri = np.flatnonzero(mo < TRAIN_END)
    vai = np.flatnonzero((mo >= TRAIN_END) & (mo < VALID_END))
    At = None
    n_train = 0
    if INCLUDE_TEST:
        At = load_arrays("test")
        n_train = len(A["tx"])
        tri = np.concatenate([tri, n_train + np.arange(len(At["tx"]))])
        print("SSL includes test domain: train", n_train, "+ test", len(At["tx"]), flush=True)
    prep = Prep(fit_stats(A, np.flatnonzero(mo < TRAIN_END)))
    tx_enc = SSLEncoder(10).to(DEVICE)
    o_enc = SSLEncoder(11).to(DEVICE)
    tx_head = SSLHeads(3).to(DEVICE)
    o_head = SSLHeads(4).to(DEVICE)
    params = list(tx_enc.parameters()) + list(o_enc.parameters()) + list(tx_head.parameters()) + list(o_head.parameters())
    opt = torch.optim.AdamW(params, LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, SSL_EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
    print("SSL pretrain", "train", len(tri), "D", D, "layers", NL, "epochs", SSL_EPOCHS,
          "mask", MASK_RATIO, "prefix", OUT, flush=True)
    tall = time.time()
    for ep in range(1, SSL_EPOCHS + 1):
        tx_enc.train(); o_enc.train(); tx_head.train(); o_head.train()
        tot = seen = 0; st = time.time()
        for b in batches(A, tri, BS, True, SEED + ep, At=At, n_train=n_train):
            tx_v, tx_t, tx_m, o_v, o_t, o_m = prep.batch(b)
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                # random mask of active tokens (15%)
                rng = torch.rand(tx_m.shape, device=DEVICE)
                tx_masked = tx_m & (rng < MASK_RATIO)
                rng = torch.rand(o_m.shape, device=DEVICE)
                o_masked = o_m & (rng < MASK_RATIO)
                loss = torch.tensor(0.0, device=DEVICE)
                for enc, head, vn, t, act, msk, c in (
                        (tx_enc, tx_head, tx_v, tx_t, tx_m, tx_masked, 3),
                        (o_enc, o_head, o_v, o_t, o_m, o_masked, 4)):
                    x = build_token_input(vn, t, "tx" if c == 3 else "order")
                    x = x * act.unsqueeze(-1)          # zero out inactive tokens
                    x = x * (~msk).unsqueeze(-1)       # zero out masked tokens
                    h = enc(x, act)
                    r, s, iv = head(h)
                    # targets on masked+active tokens
                    r_l = F.smooth_l1_loss(r[msk], vn[msk])
                    s_l = F.mse_loss(s[msk].squeeze(-1), vn[msk][:, 2])
                    iv_l = F.smooth_l1_loss(iv[msk].squeeze(-1), t[msk][:, 1])
                    loss = loss + 1.0 * r_l + 0.3 * s_l + 0.2 * iv_l
                    if msk.any():
                        tot += float(loss.detach().item()); seen += int(msk.sum())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(
                list(tx_enc.parameters()) + list(o_enc.parameters()) +
                list(tx_head.parameters()) + list(o_head.parameters()), 1)
            scaler.step(opt); scaler.update()
        sched.step()
        print(" ssl_epoch", ep, "loss", tot / max(seen, 1), "sec", round(time.time() - st), flush=True)
    # save encoders for transfer into supervised Net
    torch.save(tx_enc.state_dict(), f"output/{OUT}_ssl_tx.pt")
    torch.save(o_enc.state_dict(), f"output/{OUT}_ssl_order.pt")
    print("ssl done total_sec", round(time.time() - tall), "saved output/{}_ssl_*.pt".format(OUT), flush=True)


if __name__ == "__main__":
    main()
