"""Submission contract guards."""
from mscapital.artifacts import check_submission


def test_submission_contract():
    df = check_submission("data/submission.csv")  # raises on violation
    assert len(df) == 647896


def test_submission_prediction_range():
    df = check_submission("data/submission.csv")
    p = df.prediction.to_numpy()
    assert p.min() >= -1.0 and p.max() <= 1.0, "unit-normalized predictions expected in [-1, 1]"
