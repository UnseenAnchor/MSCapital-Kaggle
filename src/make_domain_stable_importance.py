"""Create one fixed domain-stable feature ranking for the RealMLP probe.

The target ranking is train-only. Domain gain is from the existing adversarial
train-vs-test classifier and uses no target labels. No fold/weight sweep is
performed: target_gain / sqrt(1 + domain_gain / median_domain_gain).
"""
from pathlib import Path
import numpy as np
import pandas as pd

TARGET = Path("output/proxy_lgb_trainonly_importance.csv")
DOMAIN = Path("output/domain_feature_importance.csv")
DEST = Path("output/domain_stable_importance.csv")
TOPN = 128

t = pd.read_csv(TARGET)[["feature", "gain", "split"]].rename(columns={"gain": "target_gain"})
d = pd.read_csv(DOMAIN)[["feature", "gain"]].rename(columns={"gain": "domain_gain"})
x = t.merge(d, on="feature", how="left", validate="one_to_one")
if x.domain_gain.isna().any():
    raise ValueError("domain importance missing target-ranked features")
med = float(x.domain_gain.median())
x["domain_penalty"] = np.sqrt(1.0 + x.domain_gain / med)
x["stable_score"] = x.target_gain / x.domain_penalty
x = x.sort_values(["stable_score", "target_gain", "feature"], ascending=[False, False, True]).reset_index(drop=True)
x["stable_rank"] = np.arange(1, len(x) + 1)
x.to_csv(DEST, index=False)
print("saved", DEST, "rows", len(x), "top", TOPN, "median_domain_gain", med)
print(x.head(TOPN)[["feature", "target_gain", "domain_gain", "stable_score"]].head(20).to_string(index=False))
print("top128 domain_gain median", float(x.head(TOPN).domain_gain.median()), "target median", float(x.head(TOPN).target_gain.median()))
