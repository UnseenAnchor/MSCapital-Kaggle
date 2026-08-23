"""Event256 supervised finetune with optional SSL-pretrained encoder init.

Mirrors src/train_event_v2.py (D=64, NL=2, EPOCHS=12, LAMBDA=0.8, ckpts 6/9/12,
direct target, unit-mean ensemble) so results are directly comparable to the
frozen Event256 baseline (proxy 0.13113 / middle 0.12541 / late 0.14388).

If {OUT_PREFIX}_ssl_tx.pt / _ssl_order.pt exist, the encoder front-ends
(conv/tr/pos) are initialized from them before supervised training.

Usage:
    FOLD=proxy OUT_PREFIX=event_ssl_proxy python src/train_event_ssl.py
"""
import gc, os, time, queue, threading, random
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
EPOCHS = int(os.environ.get("EPOCHS", "12"))
LAMBDA = float(os.environ.get("LAMBDA_COS", "0.8"))
TRAIN_END = int(os.environ.get("TRAIN_END", "45"))
VALID_END = int(os.environ.get("VALID_END", "71"))
TROOT = os.environ.get("EVENT_ROOT", "features/event_cache_v2")
PREFIX = os.environ.get("OUT_PREFIX", "event_ssl_proxy")
USE_SSL = os.environ.get("USE_SSL", "1") == "1"
FULL = os.environ.get("FULL", "0") == "1"  # full-data supervised training (no eval, for test prediction)
# Pseudo-label (PL) finetune on TEST streams with an ensemble prediction csv.
# Attacks the train->test transfer gap at the output level (vs SSL which only used test inputs).
PL_EPOCHS = int(os.environ.get("PL_EPOCHS", "0"))  # >0 enables PL stage
# Stable teacher downloaded from Kaggle ref 55666656; avoid mutable candidate files overwritten by later experiments.
PL_TARGET = os.environ.get("PL_TARGET", "output/teacher_submission_55666656.csv")
PL_LR = float(os.environ.get("PL_LR", "2e-4"))
PL_ONLY = os.environ.get("PL_ONLY", "0") == "1"  # skip supervised stage; resume from a supervised checkpoint
PL_RESUME_EPOCH = int(os.environ.get("PL_RESUME_EPOCH", "12"))
PL_HEAD_ONLY = os.environ.get("PL_HEAD_ONLY", "0") == "1"  # freeze encoders/cross; train head only
PL_TAG = os.environ.get("PL_TAG", "pl")
_pl_save = os.environ.get("PL_SAVE_EPOCHS", "")
PL_SAVE_EPOCHS = tuple(int(x) for x in _pl_save.split(",") if x) if _pl_save else (6, 9, 12)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def unit(x):
    x = np.asarray(x, np.float64); x = x - x.mean()
    return x / (np.linalg.norm(x) + 1e-12)


def cosine(y, p):
    return float(unit(y) @ unit(p))


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


def batches(A, idx, y, bs, shuffle, seed, maxq=3):
    idx = np.asarray(idx).copy()
    if shuffle:
        np.random.default_rng(seed).shuffle(idx)
    q = queue.Queue(maxq)
    stop = object()

    def work():
        try:
            for i in range(0, len(idx), bs):
                j = idx[i:i + bs]
                q.put((A["tx"][j], A["order"][j], A["tx_time"][j], A["order_time"][j], y[j]))
        finally:
            q.put(stop)

    threading.Thread(target=work, daemon=True).start()
    while True:
        z = q.get()
        if z is stop:
            break
        yield z


class Prep:
    def __init__(self, stats):
        self.stats = {k: (torch.tensor(a, device=DEVICE), torch.tensor(b, device=DEVICE))
                      for k, (a, b) in stats.items()}

    def one(self, k, v, t):
        v = torch.from_numpy(v).to(DEVICE).float()
        t = torch.from_numpy(t).to(DEVICE).float()
        mu, sd = self.stats[k]
        v = torch.clamp((v - mu) / sd, -8, 8)
        mask = t[:, :, 2:3]
        dv = F.pad(v[:, 1:, :2] - v[:, :-1, :2], (0, 0, 1, 0))
        inter = (v[:, :, 1:2] * v[:, :, 2:3]) if k == "tx" else (v[:, :, 2:3] * v[:, :, 3:4])
        x = torch.cat([v, t, dv, inter], -1) * mask
        return x.transpose(1, 2), mask.squeeze(-1).bool()

    def batch(self, b):
        tx, tm = self.one("tx", b[0], b[2])
        o, om = self.one("order", b[1], b[3])
        y = torch.from_numpy(b[4]).to(DEVICE)
        return tx, tm, o, om, y


class Conv(nn.Module):
    def __init__(self, a, b, k):
        super().__init__()
        self.n = nn.Sequential(
            nn.Conv1d(a, b, k, padding=k // 2, bias=False), nn.BatchNorm1d(b), nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(b, b, k, padding=k // 2, bias=False), nn.BatchNorm1d(b), nn.GELU())
        self.s = nn.Conv1d(a, b, 1, bias=False)

    def forward(self, x):
        return self.n(x) + self.s(x)


class EventEncoder(nn.Module):
    """Identical to train_event_v2.EventEncoder: SSL weights transfer by key match."""

    def __init__(self, c, L):
        super().__init__()
        self.conv = nn.Sequential(Conv(c, D, 3), Conv(D, D, 3))
        el = nn.TransformerEncoderLayer(D, 4, D * 4, 0.1, "gelu", batch_first=True, norm_first=True)
        self.tr = nn.TransformerEncoder(el, NL)
        self.pos = nn.Parameter(torch.randn(1, L, D) * 0.02)
        self.score = nn.Sequential(nn.Linear(D, D), nn.Tanh(), nn.Linear(D, 1))
        self.out = nn.Sequential(nn.Linear(D * 3, D), nn.GELU(), nn.LayerNorm(D))

    def forward(self, x, mask):
        h = self.conv(x).transpose(1, 2) + self.pos
        h = self.tr(h, src_key_padding_mask=~mask)
        s = self.score(h).squeeze(-1).masked_fill(~mask, -1e4)
        pool = torch.einsum("bt,btd->bd", torch.softmax(s, 1), h)
        mean = (h * mask[:, :, None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
        recent = h[torch.arange(len(h), device=h.device), mask.float().argmax(1)]
        return self.out(torch.cat([pool, mean, recent], -1))


class Net(nn.Module):
    def __init__(self, L=256):
        super().__init__()
        self.tx = EventEncoder(10, L)
        self.o = EventEncoder(11, L)
        el = nn.TransformerEncoderLayer(D, 4, D * 4, 0.1, "gelu", batch_first=True, norm_first=True)
        self.cross = nn.TransformerEncoder(el, 2)
        self.typ = nn.Parameter(torch.randn(1, 2, D) * 0.02)
        self.head = nn.Sequential(
            nn.Linear(D * 4, D * 2), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(D * 2, D), nn.GELU(), nn.Linear(D, 1))

    def forward(self, tx, tm, o, om):
        a = self.tx(tx, tm)
        b = self.o(o, om)
        raw = torch.cat([a, b], -1)
        mix = self.cross(torch.stack([a, b], 1) + self.typ).flatten(1)
        return self.head(torch.cat([raw, mix], -1)).squeeze(-1)


def loss_fn(p, y):
    p0 = p - p.mean(); y0 = y - y.mean()
    cos = 1 - F.cosine_similarity(p0[None], y0[None], dim=1, eps=1e-8).mean()
    return LAMBDA * cos + (1 - LAMBDA) * F.smooth_l1_loss(p, y * 1000.0)


@torch.no_grad()
def infer(model, A, idx, prep):
    model.eval()
    out = []
    dummy = np.zeros(len(A["tx"]), np.float32)
    for b in batches(A, idx, dummy, BS * 2, False, SEED):
        tx, tm, o, om, _ = prep.batch(b)
        with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
            p = model(tx, tm, o, om)
        out.append(p.float().cpu().numpy())
    return np.concatenate(out)


def load_ssl_weights(model):
    """Transfer conv/tr/pos from SSL encoders into supervised encoders."""
    for attr, fname in (("tx", f"output/{PREFIX}_ssl_tx.pt"), ("o", f"output/{PREFIX}_ssl_order.pt")):
        import os as _os
        if not _os.path.exists(fname):
            print("WARN missing", fname, "-> skip SSL init for", attr, flush=True)
            continue
        sd = torch.load(fname, map_location=DEVICE)
        enc = getattr(model, attr)
        matched = {k: v for k, v in sd.items() if k in enc.state_dict()}
        enc.load_state_dict(matched, strict=False)
        print(f"SSL init {attr}: {len(matched)}/{len(sd)} keys transferred", flush=True)


def main():
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    lab = pd.read_feather("data/train/label.feather").sort_values("sample_id").reset_index(drop=True)
    sid = lab.sample_id.to_numpy()
    mo = lab.month.to_numpy()
    y = lab.target.to_numpy(np.float32)
    A = load_arrays("train")
    if FULL:
        tri = np.arange(len(A["tx"]))
        vai = np.array([], dtype=np.int64)
    else:
        tri = np.flatnonzero(mo < TRAIN_END)
        vai = np.flatnonzero((mo >= TRAIN_END) & (mo < VALID_END))
    prep = Prep(fit_stats(A, tri))
    model = Net(A["tx"].shape[1]).to(DEVICE)
    if PL_ONLY:
        # resume from existing supervised ep12 checkpoint (no re-run of supervised stage)
        resume_path = f"output/{PREFIX}_ep{PL_RESUME_EPOCH}.pt"
        model.load_state_dict(torch.load(resume_path, map_location=DEVICE))
        print("PL_ONLY: resumed from", resume_path, flush=True)
        if len(vai) > 0:
            old = np.load(f"output/{PREFIX}_oof.npz")
            g = cosine(y[vai], np.mean([unit(old[f"ep{e}"]) for e in (6, 9, 12)], 0))
            print(f"supervised ens(6,9,12) reference: {g:.6f}", flush=True)
        else:
            g = 0.0
            print("FULL mode: no validation reference", flush=True)
        preds = {}
    elif USE_SSL:
        load_ssl_weights(model)
    print("PREFIX", PREFIX, "train", len(tri), "valid", len(vai), "D", D, "layers", NL,
          "epochs", EPOCHS, "ssl", USE_SSL, "params", round(sum(p.numel() for p in model.parameters()) / 1e6, 2),
          flush=True)
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4, fused=True)
    except TypeError:
        opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
    preds = {}
    if not PL_ONLY:
        try:
            opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4, fused=True)
        except TypeError:
            opt = torch.optim.AdamW(model.parameters(), lr=6e-4, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EPOCHS)
        scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
        for ep in range(1, EPOCHS + 1):
            model.train()
            tot = seen = 0
            st = time.time()
            for b in batches(A, tri, y, BS, True, SEED + ep):
                tx, tm, o, om, yy = prep.batch(b)
                with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                    loss = loss_fn(model(tx, tm, o, om), yy)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                scaler.step(opt); scaler.update()
                tot += float(loss) * len(yy); seen += len(yy)
            sched.step()
            if ep in (6, 9, 12):
                torch.save(model.state_dict(), f"output/{PREFIX}_ep{ep}.pt")
                if len(vai) > 0:
                    preds[ep] = infer(model, A, vai, prep)
                    print(" epoch", ep, "cos", cosine(y[vai], preds[ep]), "sec", round(time.time() - st), flush=True)
                else:
                    print(" epoch", ep, "checkpoint_saved(no_val)", "sec", round(time.time() - st), flush=True)
            else:
                print(" epoch", ep, "loss", tot / seen, "sec", round(time.time() - st), flush=True)
        m = mo[vai]
        if len(vai) == 0:
            print("FULL training done; checkpoints saved (no validation)", flush=True)
            return
        q = np.mean([unit(preds[e]) for e in (6, 9, 12)], 0)
        g = cosine(y[vai], q)
        per = [cosine(y[vai][m == x], q[m == x]) for x in np.unique(m)]
        print("RESULT ens(6,9,12) global", round(g, 6), "month_mean", round(float(np.mean(per)), 6),
              "worst", round(float(min(per)), 6), flush=True)
        base = {"proxy": 0.13113, "middle": 0.12541, "late": 0.14388}
        fold = {45: "proxy", 51: "middle", 62: "late"}.get(TRAIN_END, "?")
        if fold in base:
            print(f"vs_baseline {fold}: {g - base[fold]:+.6f} (baseline {base[fold]})", flush=True)
        np.savez(f"output/{PREFIX}_oof.npz", sample_id=sid[vai], target=y[vai], month=m,
                 **{f"ep{e}": p for e, p in preds.items()})

    # ---- pseudo-label finetune stage (test streams + ensemble pseudo-labels) ----
    if PL_EPOCHS > 0:
        m = mo[vai]
        pl = pd.read_csv(PL_TARGET).sort_values("sample_id")
        assert len(pl) == 647896, f"pseudo-label rows {len(pl)}"
        y_pl = pl.prediction.to_numpy(np.float32)
        At = load_arrays("test")
        n_test = len(At["tx"])
        if not PL_SAVE_EPOCHS or max(PL_SAVE_EPOCHS) > PL_EPOCHS:
            raise ValueError(f"PL_SAVE_EPOCHS={PL_SAVE_EPOCHS} incompatible with PL_EPOCHS={PL_EPOCHS}")
        if PL_HEAD_ONLY:
            for p in model.parameters():
                p.requires_grad_(False)
            for p in model.head.parameters():
                p.requires_grad_(True)
        trainable = [p for p in model.parameters() if p.requires_grad]
        print("PL finetune: test", n_test, "epochs", PL_EPOCHS, "lr", PL_LR,
              "head_only", PL_HEAD_ONLY, "save", PL_SAVE_EPOCHS, "tag", PL_TAG,
              "trainable", sum(p.numel() for p in trainable), "target", PL_TARGET, flush=True)
        opt = torch.optim.AdamW(trainable, lr=PL_LR, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, PL_EPOCHS)
        scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == "cuda")
        pl_preds = {}
        for ep in range(1, PL_EPOCHS + 1):
            if PL_HEAD_ONLY:
                model.eval()
                model.head.train()
            else:
                model.train()
            tot = seen = 0
            st = time.time()
            for b in batches(At, np.arange(n_test), y_pl, BS, True, SEED + 1000 + ep):
                tx, tm, o, om, yy = prep.batch(b)
                with torch.cuda.amp.autocast(enabled=DEVICE.type == "cuda"):
                    loss = loss_fn(model(tx, tm, o, om), yy)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                scaler.step(opt); scaler.update()
                tot += float(loss) * len(yy); seen += len(yy)
            sched.step()
            if ep in PL_SAVE_EPOCHS:
                torch.save(model.state_dict(), f"output/{PREFIX}_{PL_TAG}{ep}.pt")
                if len(vai) > 0:
                    pl_preds[ep] = infer(model, A, vai, prep)
                    print(" pl_epoch", ep, "cos", cosine(y[vai], pl_preds[ep]),
                          "sec", round(time.time() - st), flush=True)
                else:
                    print(" pl_epoch", ep, "checkpoint_saved(no_val)", "sec", round(time.time() - st), flush=True)
            else:
                print(" pl_epoch", ep, "loss", tot / seen, "sec", round(time.time() - st), flush=True)
        if len(vai) == 0:
            print("PL finetune done (FULL); pl checkpoints saved", flush=True)
            return
        q2 = np.mean([unit(pl_preds[e]) for e in PL_SAVE_EPOCHS], 0)
        g2 = cosine(y[vai], q2)
        per2 = [cosine(y[vai][m == x], q2[m == x]) for x in np.unique(m)]
        print(f"PL RESULT ens{PL_SAVE_EPOCHS} global", round(g2, 6),
              "month_mean", round(float(np.mean(per2)), 6),
              "worst", round(float(min(per2)), 6), flush=True)
        print(f"PL vs supervised: {g2 - g:+.6f}", flush=True)
        np.savez(f"output/{PREFIX}_{PL_TAG}_oof.npz", sample_id=sid[vai], target=y[vai], month=m,
                 **{f"ep{e}": p for e, p in pl_preds.items()})


if __name__ == "__main__":
    main()
