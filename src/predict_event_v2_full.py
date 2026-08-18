"""Full-data prediction for the 256-length raw-event Transformer (event_256)."""
import os, sys, time, numpy as np, pandas as pd, torch
sys.path.insert(0, 'src')
os.environ.update(D_MODEL='64', N_LAYERS='2')
from train_event_v2 import DEVICE, BS, SEED, D, NL, load_arrays, fit_stats, batches, Prep, Net, infer, unit

def main():
    PREFIX = os.environ.get('PREFIX', 'event_256_full')
    CHECKPOINTS = os.environ.get('CHECKPOINTS', '6,9,12').split(',')
    lab = pd.read_feather('data/train/label.feather').sort_values('sample_id')
    n_train = len(lab)
    A_train = load_arrays('train')
    tri = np.arange(n_train)
    prep = Prep(fit_stats(A_train, tri))
    del A_train; torch.cuda.empty_cache()
    A_test = load_arrays('test')
    n_test = A_test['tx'].shape[0]
    dummy = np.zeros(n_test, np.float32)
    acc = np.zeros(n_test, np.float64)
    for ep in CHECKPOINTS:
        model = Net(A_test['tx'].shape[1]).to(DEVICE)
        model.load_state_dict(torch.load(f'output/{PREFIX}_ep{ep}.pt', map_location=DEVICE))
        p = infer(model, A_test, np.arange(n_test), prep)
        acc += unit(p)
        print(f'ep{ep} done, mean={p.mean():.5f}', flush=True)
        del model; torch.cuda.empty_cache()
    pred = unit(acc)
    out = pd.DataFrame({'sample_id': np.arange(n_test), 'prediction': pred})
    dest = os.environ.get('OUT', f'output/submission_{PREFIX}_unit.csv')
    out.to_csv(dest, index=False)
    import hashlib
    print('saved', dest, hashlib.sha256(open(dest, 'rb').read()).hexdigest())

if __name__ == '__main__':
    main()