"""OOF alignment guards: sample_id, target and month must line up with labels."""
import numpy as np
import pandas as pd
import pytest
from mscapital.splits import load_label

FOLDS = ["proxy", "middle", "late"]


def _load_oof(fold):
    z = np.load(f"output/event_256_{fold}_oof.npz")
    lab = pd.read_feather("data/train/label.feather").sort_values("sample_id").reset_index(drop=True)
    return z, lab


@pytest.mark.parametrize("fold", FOLDS)
def test_oof_sample_id_alignment(fold):
    z, lab = _load_oof(fold)
    sid = z["sample_id"]
    assert len(sid) == len(np.unique(sid)), f"{fold}: duplicate sample_id"
    assert np.array_equal(sid, np.sort(sid)), f"{fold}: sample_id not sorted"
    # must equal the label rows at those positions
    lab_sid = lab.sample_id.to_numpy()
    assert np.array_equal(lab_sid[sid], sid), f"{fold}: label row mismatch"


@pytest.mark.parametrize("fold", FOLDS)
def test_oof_target_matches_label(fold):
    z, lab = _load_oof(fold)
    lab_t = lab.target.to_numpy(np.float64)
    np.testing.assert_allclose(z["target"], lab_t[z["sample_id"]], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize("fold", FOLDS)
def test_oof_has_expected_members(fold):
    z, _ = _load_oof(fold)
    for k in ("ep6", "ep9", "ep12", "month"):
        assert k in z, f"{fold}: missing key {k}"
    assert np.isfinite(z["ep6"]).all() and np.isfinite(z["ep9"]).all() and np.isfinite(z["ep12"]).all()
