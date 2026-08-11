"""用缓存和指定 checkpoint 生成 test GRU 预测。"""
import os, numpy as np, pandas as pd, torch
from train_gru_cached import FusionNet, CacheDataset, CACHE, DEVICE, BATCH, SCALE_Y

MODEL = 'output/gru_cached_best.pt'
def main():
    ids = np.load(f'{CACHE}/test_ids.npy')
    m = np.load(f'{CACHE}/test_market.npy', mmap_mode='r')
    o = np.load(f'{CACHE}/test_order.npy', mmap_mode='r')
    x = np.load(f'{CACHE}/test_tx.npy', mmap_mode='r')
    ds = CacheDataset(m, o, x, np.zeros(len(ids), dtype=np.float32), np.arange(len(ids)))
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH*2, shuffle=False, num_workers=0)
    model = FusionNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL, map_location=DEVICE))
    model.eval(); pred=[]
    with torch.no_grad():
        for bm, bo, bx, _ in loader:
            pred.append((model(bm.to(DEVICE), bo.to(DEVICE), bx.to(DEVICE)).cpu().numpy()/SCALE_Y))
    out = pd.DataFrame({'sample_id': ids, 'prediction': np.concatenate(pred)})
    out.to_csv('output/submission_gru_cached.csv', index=False)
    print(out.shape, 'mean=', out.prediction.mean(), 'std=', out.prediction.std())
if __name__ == '__main__': main()
