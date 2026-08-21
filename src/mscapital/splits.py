"""Chronological month-based splits used across Event256/SSL experiments."""
import numpy as np
import pandas as pd

# Fold endpoints (month index). Verified against src/train_event_v2.py env vars.
FOLDS = {
    "proxy":  dict(train_end=45, valid_end=71),
    "middle": dict(train_end=51, valid_end=71),
    "late":   dict(train_end=62, valid_end=71),
}

def load_label(path="data/train/label.feather"):
    """Load label table sorted by sample_id. Returns (sample_id, month, target)."""
    lab = pd.read_feather(path).sort_values("sample_id").reset_index(drop=True)
    return lab.sample_id.to_numpy(), lab.month.to_numpy(), lab.target.to_numpy(np.float64)

def split_by_month(month, train_end, valid_end=71):
    """Return (train_idx, valid_idx) with strict chronological separation."""
    tri = np.flatnonzero(month < train_end)
    vai = np.flatnonzero((month >= train_end) & (month < valid_end))
    return tri, vai

def fold_split(month, fold):
    """Return (train_idx, valid_idx) for a named fold."""
    if fold not in FOLDS:
        raise KeyError(f"unknown fold {fold!r}; have {sorted(FOLDS)}")
    return split_by_month(month, **FOLDS[fold])

def assert_disjoint(a, b, name="train/valid"):
    """Raise if two index sets overlap."""
    inter = np.intersect1d(a, b)
    if len(inter):
        raise AssertionError(f"{name} overlap: {len(inter)} samples")
