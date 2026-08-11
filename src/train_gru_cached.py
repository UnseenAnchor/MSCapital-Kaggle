"""从 features/cache 直接训练 GRU 三路序列模型，避免重复读取 10GB+ feather。"""
import os, sys, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

CACHE = 'features/cache'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
BATCH = 2048
SCALE_Y = 1000.0
LEN_M, LEN_O, LEN_X = 64, 32, 32
np.random.seed(42); torch.manual_seed(42)
print(f'DEVICE={DEVICE}, epochs={EPOCHS}, batch={BATCH}', flush=True)

class CacheDataset(torch.utils.data.Dataset):
    def __init__(self, m, o, x, y, indices):
        self.m, self.o, self.x, self.y, self.indices = m, o, x, y, indices
    def __len__(self): return len(self.indices)
    def __getitem__(self, i):
        j = self.indices[i]
        return (torch.from_numpy(self.m[j]), torch.from_numpy(self.o[j]),
                torch.from_numpy(self.x[j]), torch.tensor(self.y[j], dtype=torch.float32))

class GRUEncoder(nn.Module):
    def __init__(self, in_dim, hid):
        super().__init__()
        self.gru = nn.GRU(in_dim, hid, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Sequential(nn.Linear(hid, hid), nn.ReLU())
    def forward(self, x):
        _, h = self.gru(x)
        return self.head(h[-1])

class FusionNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_m = GRUEncoder(11, 96)
        self.enc_o = GRUEncoder(4, 64)
        self.enc_x = GRUEncoder(3, 64)
        self.gate = nn.Linear(224, 3)
        self.fc = nn.Sequential(nn.Linear(224, 128), nn.ReLU(), nn.Dropout(.2),
                                nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, m, o, x):
        hm, ho, hx = self.enc_m(m), self.enc_o(o), self.enc_x(x)
        h = torch.cat([hm, ho, hx], -1)
        g = torch.softmax(self.gate(h), -1)
        return self.fc(torch.cat([g[:, :1]*hm, g[:, 1:2]*ho, g[:, 2:]*hx], -1)).squeeze(-1)

def cosine(y, p):
    y = y - y.mean(); p = p - p.mean()
    return float(np.dot(y, p) / (np.linalg.norm(y)*np.linalg.norm(p)+1e-12))

def main():
    t0 = time.time()
    ids = np.load(f'{CACHE}/train_ids.npy')
    m = np.load(f'{CACHE}/train_market.npy', mmap_mode='r')
    o = np.load(f'{CACHE}/train_order.npy', mmap_mode='r')
    x = np.load(f'{CACHE}/train_tx.npy', mmap_mode='r')
    label = pd.read_feather('data/train/label.feather').set_index('sample_id')
    months = label.loc[ids, 'month'].to_numpy()
    y = label.loc[ids, 'target'].to_numpy(dtype=np.float32)
    tr = np.flatnonzero(months < 62)
    va = np.flatnonzero(months >= 62)
    print(f'cache loaded in {time.time()-t0:.1f}s; train={len(tr)}, val={len(va)}', flush=True)
    tr_ds = CacheDataset(m, o, x, y*SCALE_Y, tr)
    va_ds = CacheDataset(m, o, x, y*SCALE_Y, va)
    tr_loader = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH, shuffle=True,
                                             num_workers=0, pin_memory=True)
    va_loader = torch.utils.data.DataLoader(va_ds, batch_size=BATCH*2, shuffle=False,
                                             num_workers=0, pin_memory=True)
    model = FusionNet().to(DEVICE)
    resume = len(sys.argv) > 2 and sys.argv[2] == 'resume'
    if resume and os.path.exists('output/gru_cached_best.pt'):
        model.load_state_dict(torch.load('output/gru_cached_best.pt', map_location=DEVICE))
        print('resumed from output/gru_cached_best.pt', flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=8e-4 if resume else 2e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    loss_fn = nn.MSELoss()
    best = -1
    for ep in range(EPOCHS):
        model.train(); total = 0.; n = 0; start = time.time()
        for bm, bo, bx, by in tr_loader:
            bm, bo, bx, by = bm.to(DEVICE, non_blocking=True), bo.to(DEVICE, non_blocking=True), bx.to(DEVICE, non_blocking=True), by.to(DEVICE, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(bm, bo, bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); total += loss.item()*len(by); n += len(by)
        sched.step()
        model.eval(); pred=[]; true=[]
        with torch.no_grad():
            for bm, bo, bx, by in va_loader:
                pred.append((model(bm.to(DEVICE), bo.to(DEVICE), bx.to(DEVICE)).cpu().numpy()/SCALE_Y))
                true.append(by.numpy()/SCALE_Y)
        p, t = np.concatenate(pred), np.concatenate(true)
        score = cosine(t, p)
        print(f'epoch {ep+1}/{EPOCHS}: loss={total/n:.6f}, val_cosine={score:.5f}, sec={time.time()-start:.0f}', flush=True)
        torch.save(model.state_dict(), f'output/gru_cached_ep{ep+1}.pt')
        if score > best:
            best = score; torch.save(model.state_dict(), 'output/gru_cached_best.pt')
    print(f'best val cosine={best:.5f}', flush=True)

if __name__ == '__main__': main()
