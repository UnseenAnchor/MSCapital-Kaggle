"""Artifact guards: submission shape/NaN checks and file hashing."""
import hashlib
import numpy as np
import pandas as pd

EXPECTED_SUBMISSION_ROWS = 647896

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def check_submission(path, n_rows=EXPECTED_SUBMISSION_ROWS):
    """Validate a submission CSV. Raises AssertionError on any violation.

    Returns the DataFrame on success.
    """
    df = pd.read_csv(path)
    assert len(df) == n_rows, f"rows {len(df)} != {n_rows}"
    assert list(df.columns) == ["sample_id", "prediction"], f"columns {list(df.columns)}"
    sid = df.sample_id.to_numpy()
    assert np.array_equal(sid, np.arange(n_rows)), "sample_id must be 0..N-1 in order"
    pred = df.prediction.to_numpy(np.float64)
    assert np.isfinite(pred).all(), "prediction contains NaN/Inf"
    return df
