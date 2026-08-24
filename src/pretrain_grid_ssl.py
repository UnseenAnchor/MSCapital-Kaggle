"""Test-domain masked reconstruction pretraining for MultiStream grid encoders.

The saved encoder keys (c/t/pos) match train_multistream_grid.Stream so the
supervised v3 model can initialize from this unlabeled train+test pretraining.
"""
import os, time, queue, threading, random
import numpy as np, pandas as pd, torch
import torch.nn as nn
import torch.nn.functional as F

from train_multistream_grid import (
    DEVICE, D_MODEL as D, N_LAYERS as NL, M_CH, T_CH, O_CH,
    arrays, norm_stats, Conv, M_LEN, F_LEN,
)

BS = int(os.environ.get("SSL_BATCH", "256"))
EPOCHS = int(os.environ.get("SSL_EPOCHS", "8"))
MASK_RATIO = float(os.environ.get("SSL_MASK_RATIO", "0.15"))
LR = float(os.environ.get("SSL_LR", "3e-4"))
TRAIN_END = int(os.environ.get("TRAIN_END", "45"))
GRID_ROOT = os.environ.get("GRID_ROOT", "features/grid_v2")
GRID_VERSION = os.environ.get("GRID_VERSION", "v2")
OUT = os.environ.get("OUT_PREFIX", "grid_ssl")
INCLUDE_TEST = os.environ.get("SSL_INCLUDE_TEST", "1") == "1"
SEED = int(os.environ.get("SEED", "42"))


def make_paths(split):
    return {
        "market": f"{GRID_ROOT}/{split}_{GRID_VERSION}_market_{M_LEN}x{M_CH}.mmap",
        "tx": f"{GRID_ROOT}/{split}_{GRID_VERSION}_tx_{F_LEN}x{T_CH}.mmap",
        "order": f"{GRID_ROOT}/{split}_{GRID_VERSION}_order_{F_LEN}x{O_CH}.mmap",
    }


def load_labels():
    return pd.read_feather("data/train/label.feather").sort_values("sample_id").reset_index(drop=True)


def combined_batches(A, At, indices, bs, seed):
    idx = np.asarray(indices).copy()
    np.random.default_rng(seed).shuffle(idx)
    n = len(A["market"])
    q = queue.Queue(3)
    stop = object()

    def work():
        try:
            for i in range(0, len(idx), bs):
                j = idx[i:i + bs]
                tr = j[j < n]
                te = j[j >= n] - n
                pieces = []
                for key in ("market", "tx", "order"):
                    a = A[key][tr] if len(tr) else np.empty((0,) + A[key].shape[1:], dtype=np.float16)
                    b = At[key][te] if len(te) else np.empty((0,) + At[key].shape[1:], dtype=np.float16)
                    pieces.append(np.concatenate([a, b], axis=0))
                q.put(tuple(pieces))
        finally:
            q.put(stop)

    threading.Thread(target=work, daemon=True).start()
    while True:
        item = q.get()
        if item is stop:
            break
        yield item


class TokenEncoder(nn.Module):
    def __init__(self, inc, length):
        super().__init__()
        self.c = nn.Sequential(Conv(inc, D, 5), Conv(D, D, 3))
        el = nn.TransformerEncoderLayer(D, 4, D * 4, .1, "gelu", batch_first=True, norm_first=True)
        self.t = nn.TransformerEncoder(el, NL)
        self.pos = nn.Parameter(torch.randn(1, length, D) * .02)

    def forward(self, x, active):
        h = self.c(x.transpose(1, 2)).transpose(1, 2)
        h = self.t(h + self.pos[:, :h.size(1)], src_key_padding_mask=~active)
        return h


class GridSSL(nn.Module):
    def __init__(self):
        super().__init__()
        self.market = TokenEncoder(M_CH, M_LEN)
        self.tx = TokenEncoder(T_CH, F_LEN)
        self.order = TokenEncoder(O_CH, F_LEN)
        self.market_head = nn.Linear(D, M_CH)
        self.tx_head = nn.Linear(D, T_CH)
        self.order_head = nn.Linear(D, O_CH)


def normalize(x, stats, key):
    mu, sd = stats[key]
    z = torch.from_numpy(np.asarray(x, np.float32)).to(DEVICE)
    mu = torch.as_tensor(mu, device=DEVICE)
    sd = torch.as_tensor(sd, device=DEVICE)
    z = torch.nan_to_num(torch.clamp((z - mu) / sd, -8, 8), nan=0., posinf=8., neginf=-8.)
    active = z.abs().sum(-1) > 0
    return z, active


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    lab = load_labels(); n = len(lab)
    A = arrays("train", n)
    At = arrays("test", 647896) if INCLUDE_TEST else None
    train_idx = np.flatnonzero(lab.month.to_numpy() < TRAIN_END)
    stats = norm_stats(A, train_idx)
    all_idx = train_idx if At is None else np.concatenate([train_idx, n + np.arange(647896)])
    q = GridSSL().to(DEVICE)
    opt = torch.optim.AdamW(q.parameters(), LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
    print("grid SSL", "train", len(train_idx), "+ test" if At is not None else "", 647896 if At is not None else "", "epochs", EPOCHS, "batch", BS, "prefix", OUT, flush=True)
    for ep in range(1, EPOCHS + 1):
        q.train(); total = seen = 0; st = time.time()
        for raw in combined_batches(A, At, all_idx, BS, SEED + ep):
            vals = {}; masks = {}
            for key, x in zip(("market", "tx", "order"), raw):
                vals[key], masks[key] = normalize(x, stats, key)
            with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                loss = torch.zeros((), device=DEVICE)
                for key, enc, head in (("market", q.market, q.market_head), ("tx", q.tx, q.tx_head), ("order", q.order, q.order_head)):
                    active = masks[key]
                    masked = active & (torch.rand(active.shape, device=DEVICE) < MASK_RATIO)
                    inp = vals[key] * (~masked).unsqueeze(-1)
                    h = enc(inp, active)
                    pred = head(h)
                    loss = loss + F.smooth_l1_loss(pred[masked], vals[key][masked])
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(q.parameters(), 1); scaler.step(opt); scaler.update()
            total += float(loss) * len(raw[0]); seen += len(raw[0])
        sched.step()
        print("epoch", ep, "loss", total / max(seen, 1), "sec", round(time.time() - st), flush=True)
    for key in ("market", "tx", "order"):
        enc = getattr(q, key)
        torch.save(enc.state_dict(), f"output/{OUT}_ssl_{key}.pt")
    print("saved SSL encoders", OUT, flush=True)


if __name__ == "__main__":
    main()
