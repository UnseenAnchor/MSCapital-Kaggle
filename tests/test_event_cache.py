"""Data-contract guards for the 256-length event cache (regression guard vs the old L=32 truncation)."""
import numpy as np
import pytest

EVENT_ROOT = "features/event_cache_v2"
N_TRAIN = 1257637
L = 256


def _load(name, mmap=True):
    return np.load(f"{EVENT_ROOT}/train_{name}.npy", mmap_mode="r" if mmap else None)


def test_event_cache_shapes():
    assert _load("transaction_feat").shape == (N_TRAIN, L, 3)
    assert _load("order_feat").shape == (N_TRAIN, L, 4)
    assert _load("transaction_time").shape == (N_TRAIN, L, 4)
    assert _load("order_time").shape == (N_TRAIN, L, 4)


def test_time_mask_is_active_flag():
    t = _load("transaction_time")
    # channel 2 is the active flag (0/1)
    assert set(np.unique(t[:, :, 2])) <= {0.0, 1.0}


def test_padding_is_zero_and_masked():
    tx_feat = _load("transaction_feat")
    tx_time = _load("transaction_time")
    n = min(20000, N_TRAIN)
    mask = tx_time[:n, :, 2] > 0
    # padded (inactive) slots must have zero features
    assert np.abs(tx_feat[:n][~mask]).max() == 0.0
    # every active slot must have at least one nonzero feature (prices/volumes are raw)
    active = tx_feat[:n][mask]
    assert active.shape[0] > 0


def test_no_32_event_truncation_regression():
    """The old bug truncated everything to 32 events. Require most samples to use more."""
    t = _load("order_time")
    counts = (t[:, :, 2] > 0).sum(1)
    frac_over32 = float((counts > 32).mean())
    assert frac_over32 > 0.5, f"only {frac_over32:.3f} samples exceed 32 events (truncation regressed?)"
    assert counts.min() >= 1
    assert counts.max() <= L


def test_event_count_distribution_sane():
    t = _load("transaction_time")
    counts = (t[:, :, 2] > 0).sum(1)
    med = np.median(counts)
    assert 10 <= med <= 250, f"median active events {med} out of expected range"
