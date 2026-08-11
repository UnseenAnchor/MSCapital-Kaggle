"""深度学习序列模型 v1：GRU 三路编码（market/order/transaction）→ 门控融合 → 回归
GPU 训练，时序验证（0-61 train / 62-70 val）。
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pyarrow.feather as pf
import time, sys

torch.manual_seed(42)
np.random.seed(42)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'DEVICE: {DEVICE}', flush=True)

LEN_M, LEN_O, LEN_X = 64, 32, 32   # 采样长度
BATCH, EPOCHS = 1024, int(sys.argv[1]) if len(sys.argv) > 1 else 3

M_COLS = ['transaction_avgprice', 'transaction_volume', 'transaction_count',
          'ask_price_1', 'ask_volume_1', 'bid_price_1', 'bid_volume_1',
          'ask_price_2', 'ask_volume_2', 'bid_price_2', 'bid_volume_2']
O_COLS = ['price', 'volume', 'side', 'order_action']
X_COLS = ['price', 'volume', 'side']


def load_grouped(split, key, cols):
    """读取 feather → {sample_id: ndarray (T, C)}，按 seconds_before_predict 升序"""
    t0 = time.time()
    fname = 'transaction' if key == 'tx' else key
    src = f'data/{split}/{fname}.feather'
    t = pf.read_table(src)
    # 只取需要的列 + 排序键
    sel = ['sample_id'] + cols
    df = t.select(sel).to_pandas()
    # seconds_before_predict 升序 → 用原始顺序（feather 已按行序排列，假定已排序）
    grp = df.groupby('sample_id', sort=True)
    out = {sid: np.ascontiguousarray(g[cols].values, dtype=np.float32) for sid, g in grp}
    # 缺失值填充（如 transaction_avgprice 约 28% 缺失）
    for sid in out:
        a = out[sid]
        if np.isnan(a).any():
            out[sid] = np.nan_to_num(a, nan=0.0)
    print(f'  [{split}/{key}] {len(out)} samples, {time.time()-t0:.0f}s', flush=True)
    return out


def estimate_stats(data, key, n_samples=200):
    """用部分样本估计均值/方差（按列）"""
    s = 0.0; sq = 0.0; c = 0
    rng = np.random.RandomState(0)
    sids = rng.choice(list(data[key].keys()), min(n_samples, len(data[key])), replace=False)
    for sid in sids:
        a = data[key][sid]
        s += a.sum(0); sq += (a ** 2).sum(0); c += len(a)
    mu = s / c
    sd = np.sqrt(np.maximum(sq / c - mu ** 2, 1e-6))
    return mu, sd


class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, data, ids, targets=None):
        self.data, self.ids, self.targets = data, ids, targets

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _seq(arr, L, C):
        if arr is None or len(arr) == 0:
            return torch.zeros(L, C)
        T = len(arr)
        if T >= L:
            idx = np.linspace(0, T - 1, L).astype(np.int64)
        else:
            idx = np.concatenate([np.zeros(L - T, dtype=np.int64), np.arange(T)])
        return torch.from_numpy(arr[::-1][idx]).float()  # 新在前

    def __getitem__(self, idx):
        sid = self.ids[idx]
        m = self._seq(self.data['market'].get(sid), LEN_M, len(M_COLS))
        o = self._seq(self.data['order'].get(sid), LEN_O, len(O_COLS))
        x = self._seq(self.data['tx'].get(sid), LEN_X, len(X_COLS))
        if self.targets is not None:
            return m, o, x, float(self.targets[sid])
        return m, o, x


class GRUEncoder(nn.Module):
    def __init__(self, in_dim, hid, layers=2):
        super().__init__()
        self.gru = nn.GRU(in_dim, hid, num_layers=layers, batch_first=True,
                          dropout=0.1 if layers > 1 else 0)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU())

    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])


class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_m = GRUEncoder(len(M_COLS), 96)
        self.enc_o = GRUEncoder(len(O_COLS), 64)
        self.enc_x = GRUEncoder(len(X_COLS), 64)
        self.gate = nn.Linear(96 + 64 + 64, 3)
        self.fc = nn.Sequential(nn.Linear(96 + 64 + 64, 128), nn.ReLU(), nn.Dropout(0.2),
                                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, m, o, x):
        hm, ho, hx = self.enc_m(m), self.enc_o(o), self.enc_x(x)
        h = torch.cat([hm, ho, hx], dim=-1)
        g = torch.softmax(self.gate(h), dim=-1)
        hg = torch.cat([g[:, :1] * hm, g[:, 1:2] * ho, g[:, 2:] * hx], dim=-1)
        return self.fc(hg).squeeze(-1)


def cosine(y_true, y_pred):
    yt = y_true - y_true.mean(); yp = y_pred - y_pred.mean()
    return float(np.dot(yt, yp) / (np.linalg.norm(yt) * np.linalg.norm(yp) + 1e-12))


def main():
    t0 = time.time()
    label = pd.read_feather('data/train/label.feather')
    print('加载序列数据...', flush=True)
    tr_data = {}
    te_data = {}
    for key, cols in [('market', M_COLS), ('order', O_COLS), ('tx', X_COLS)]:
        tr_data[key] = load_grouped('train', key, cols)
        te_data[key] = load_grouped('test', key, cols)

    # 归一化（用 train 统计）
    stats = {}
    for key in ['market', 'order', 'tx']:
        mu, sd = estimate_stats(tr_data, key)
        stats[key] = (mu, sd)
        for d in [tr_data, te_data]:
            for sid in d[key]:
                d[key][sid] = (d[key][sid] - mu) / sd
    print(f'归一化完成, {time.time()-t0:.0f}s', flush=True)

    # 时间加权：seconds_before_predict 越小越新，这里简化用位置加权在采样中已体现
    tr_ids = label[label.month < 62].sample_id.values
    va_ids = label[label.month >= 62].sample_id.values
    tr_tgt = label.set_index('sample_id').target
    print(f'train ids {len(tr_ids)}, val ids {len(va_ids)}', flush=True)

    tr_ds = SeqDataset(tr_data, tr_ids, tr_tgt)
    va_ds = SeqDataset(tr_data, va_ids, tr_tgt)
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=2, persistent_workers=False)
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=BATCH * 2, shuffle=False, num_workers=2)

    model = FusionNet().to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'模型参数量: {n_params/1e6:.2f}M', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()

    for ep in range(EPOCHS):
        model.train()
        t1 = time.time(); tot = 0.0; nb = 0
        for m, o, x, y in tr_loader:
            m, o, x, y = m.to(DEVICE), o.to(DEVICE), x.to(DEVICE), y.to(DEVICE).float()
            opt.zero_grad()
            loss = loss_fn(model(m, o, x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * len(y); nb += len(y)
        sched.step()
        # 验证
        model.eval()
        preds = []; tgts = []
        with torch.no_grad():
            for m, o, x, y in va_loader:
                m, o, x = m.to(DEVICE), o.to(DEVICE), x.to(DEVICE)
                preds.append(model(m, o, x).cpu().numpy())
                tgts.append(y.numpy())
        pv = np.concatenate(preds); tv = np.concatenate(tgts)
        print(f'epoch {ep+1}/{EPOCHS}: loss {tot/nb:.6f}, val cosine {cosine(tv, pv):.5f}, {time.time()-t1:.0f}s', flush=True)
        torch.save(model.state_dict(), f'output/gru_ep{ep+1}.pt')

    # 预测 test
    te_ids = te_data['market'].keys()
    te_tgt = None
    te_ds = SeqDataset(te_data, np.array(sorted(te_ids)))
    te_loader = torch.utils.data.DataLoader(te_ds, batch_size=BATCH * 2, shuffle=False)
    model.eval()
    preds = []
    with torch.no_grad():
        for m, o, x in te_loader:
            m, o, x = m.to(DEVICE), o.to(DEVICE), x.to(DEVICE)
            preds.append(model(m, o, x).cpu().numpy())
    sub = pd.DataFrame({'sample_id': sorted(te_ids), 'prediction': np.concatenate(preds)})
    sub.to_csv('output/submission_gru.csv', index=False)
    print(f'submission_gru: {sub.shape}, 总耗时 {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
