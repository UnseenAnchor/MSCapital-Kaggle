"""Chronological split guards: no leakage, correct fold endpoints."""
import numpy as np
from mscapital.splits import FOLDS, load_label, fold_split, assert_disjoint


def test_month_range_and_no_overlap():
    _, month, _ = load_label()
    assert month.min() >= 0 and month.max() < 71
    for fold, cfg in FOLDS.items():
        tri, vai = fold_split(month, fold)
        assert_disjoint(tri, vai, f"{fold} train/valid")
        # chronological: every train month < every valid month
        assert month[tri].max() < month[vai].min(), f"{fold} not chronological"
        assert month[tri].max() < cfg["train_end"]
        assert month[vai].min() >= cfg["train_end"]


def test_fold_endpoints_match_documented():
    assert FOLDS["proxy"]["train_end"] == 45
    assert FOLDS["middle"]["train_end"] == 51
    assert FOLDS["late"]["train_end"] == 62
    assert all(cfg["valid_end"] == 71 for cfg in FOLDS.values())


def test_fold_sizes_positive():
    _, month, _ = load_label()
    for fold in FOLDS:
        tri, vai = fold_split(month, fold)
        assert len(tri) > 0 and len(vai) > 0
