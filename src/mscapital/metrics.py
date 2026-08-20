"""Cosine-similarity evaluation, matching the competition metric."""
import numpy as np

def unit(x):
    """Unit-normalize (center + L2 normalize)."""
    x = np.asarray(x, np.float64)
    x = x - x.mean()
    n = np.linalg.norm(x)
    return x / (n + 1e-12)

def cosine(y, p):
    """Global cosine between prediction and target."""
    return float(unit(y) @ unit(p))

def fold_stats(y, p, month):
    """Return dict with global, month-mean, and worst-month cosine.

    month: array of month ids parallel to y/p.
    """
    y = np.asarray(y, np.float64)
    p = np.asarray(p, np.float64)
    month = np.asarray(month)
    g = cosine(y, p)
    per = [cosine(y[month == m], p[month == m]) for m in np.unique(month)]
    return dict(global_cosine=g, month_mean=float(np.mean(per)), worst_month=float(np.min(per)))

def ensemble_unit_mean(preds):
    """Unit-normalize each member prediction then average."""
    return np.mean([unit(p) for p in preds], axis=0)
