"""Build and audit the diversity-preserving dual-Event candidate.

Keeps both original Event256 and supervised test-domain SSL Event256 instead
of replacing one with the other.  Candidate weights are deliberately rounded
and fixed; the deterministic leave-one-fold grid below is an audit only.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

# This candidate deliberately uses supervised SSL OOFs, never pseudo-label OOFs.
os.environ["USE_PL_OOF"] = "0"
from make_candidate_event_ssl import MEMBERS, PUBLIC_REF, W5, gmet, load_oof, u

EVENT_ORIG = os.environ.get("EVENT_ORIG_CSV", "output/submission_event_256_unit.csv")
EVENT_SSL = os.environ.get(
    "EVENT_SSL_CSV", "output/submission_event_ssl_tt_full_supervised_unit.csv"
)
BEST_PUBLIC = os.environ.get("BEST_PUBLIC_CSV", "output/best_submission_55601441.csv")
SSL_PUBLIC = os.environ.get("SSL_PUBLIC_CSV", "output/teacher_submission_55666656.csv")
DEST = os.environ.get("DEST", "output/candidate_dual_event_public60_40_e10_s20.csv")
EVENT_ORIG_WEIGHT = float(os.environ.get("EVENT_ORIG_WEIGHT", "0.10"))
EVENT_SSL_WEIGHT = float(os.environ.get("EVENT_SSL_WEIGHT", "0.20"))
FOLDS = ("proxy", "middle", "late")
GRID = np.round(np.arange(0.0, 0.401, 0.01), 2)

# Exact inputs consumed indirectly by make_candidate_event_ssl.load_oof.
OOF_INPUTS = sorted(
    {
        "data/train/label.feather",
        "output/proxy_lgb_oof.npz",
        "output/multistream_v3_proxy_oof.npz",
        "output/multistream_v3_middle_eff1024_oof.npz",
        "output/multistream_v3_late_eff1024_oof.npz",
        "output/realmlp_multiseed_proxy_oof.npz",
        "output/realmlp_multiseed_rolling_oof.npz",
        "output/joint_v3_proxy_fast_oof.npz",
        "output/joint_v3_middle_fast_oof.npz",
        "output/joint_v3_late_fast_oof.npz",
        "output/multires_self_proxy_oof.npz",
        "output/multires_self_middle_oof.npz",
        "output/multires_self_late_oof.npz",
        "output/event_256_proxy_oof.npz",
        "output/event_256_middle_oof.npz",
        "output/event_256_late_oof.npz",
        "output/event_ssl_tt_proxy_oof.npz",
        "output/event_ssl_tt_middle_oof.npz",
        "output/event_ssl_tt_late_oof.npz",
    }
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_prediction(path: str) -> tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path).sort_values("sample_id")
    ids = df["sample_id"].to_numpy()
    pred = df["prediction"].to_numpy(np.float64)
    expected = np.arange(647896)
    if len(ids) != len(expected) or not np.array_equal(ids, expected):
        raise ValueError(f"invalid or incomplete submission ids: {path}")
    if not np.isfinite(pred).all():
        raise ValueError(f"non-finite predictions: {path}")
    return ids, pred


def stack_prediction(data: tuple, event_orig_weight: float, event_ssl_weight: float) -> np.ndarray:
    _, _, members, event_orig, event_ssl = data
    weights = np.r_[W5, event_orig_weight, event_ssl_weight]
    weights /= weights.sum()
    return (
        weights[:5] @ members
        + weights[5] * event_orig
        + weights[6] * event_ssl
    )


def metrics(data: tuple, event_orig_weight: float, event_ssl_weight: float) -> np.ndarray:
    y, month, *_ = data
    return np.asarray(gmet(y, stack_prediction(data, event_orig_weight, event_ssl_weight), month))


def loo_audit(fold_data: dict[str, tuple], baseline: dict[str, np.ndarray]) -> list[dict]:
    """Select on two folds, report the held-out fold; never changes final weights.

    Constraints on both selection folds: global/month mean >= baseline and
    worst-month delta >= -0.0003.  Deterministic tie-break (descending): mean
    global delta, minimum global delta, minimum month delta, minimum worst
    delta, then lower total event weight, lower original weight, lower SSL weight.
    """
    results = []
    for held_out in FOLDS:
        train_folds = tuple(f for f in FOLDS if f != held_out)
        candidates = []
        for event_orig_weight in GRID:
            for event_ssl_weight in GRID:
                if event_orig_weight + event_ssl_weight == 0:
                    continue
                deltas = {
                    fold: metrics(fold_data[fold], event_orig_weight, event_ssl_weight)
                    - baseline[fold]
                    for fold in FOLDS
                }
                global_delta = np.asarray([deltas[f][0] for f in train_folds])
                month_delta = np.asarray([deltas[f][1] for f in train_folds])
                worst_delta = np.asarray([deltas[f][2] for f in train_folds])
                if global_delta.min() < 0 or month_delta.min() < 0 or worst_delta.min() < -0.0003:
                    continue
                key = (
                    float(global_delta.mean()),
                    float(global_delta.min()),
                    float(month_delta.min()),
                    float(worst_delta.min()),
                    -float(event_orig_weight + event_ssl_weight),
                    -float(event_orig_weight),
                    -float(event_ssl_weight),
                )
                candidates.append((key, event_orig_weight, event_ssl_weight, deltas))
        if not candidates:
            raise RuntimeError(f"no feasible LOO weights for held-out fold {held_out}")
        _, selected_orig, selected_ssl, deltas = max(candidates, key=lambda row: row[0])
        results.append(
            {
                "held_out": held_out,
                "selection_folds": list(train_folds),
                "selected_event_orig_weight": float(selected_orig),
                "selected_event_ssl_weight": float(selected_ssl),
                "selection_fold_deltas": {
                    fold: deltas[fold].tolist() for fold in train_folds
                },
                "held_out_delta": deltas[held_out].tolist(),
            }
        )
    return results


def month_sign_audit(data: tuple, baseline_prediction: np.ndarray, dual_prediction: np.ndarray) -> dict:
    y, month, *_ = data
    month_values = np.unique(month)
    deltas = np.asarray(
        [
            float(
                u(y[month == value]) @ u(dual_prediction[month == value])
                - u(y[month == value]) @ u(baseline_prediction[month == value])
            )
            for value in month_values
        ]
    )
    return {
        "months": int(len(deltas)),
        "positive": int((deltas > 0).sum()),
        "negative": int((deltas < 0).sum()),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "min_delta": float(deltas.min()),
        "max_delta": float(deltas.max()),
        "negative_months": [
            {"month": int(value), "delta": float(delta)}
            for value, delta in zip(month_values, deltas)
            if delta < 0
        ],
    }


def main() -> None:
    fold_data = {fold: load_oof(fold) for fold in FOLDS}
    baseline = {fold: metrics(fold_data[fold], 0.20, 0.0) for fold in FOLDS}
    ssl_only = {fold: metrics(fold_data[fold], 0.0, 0.20) for fold in FOLDS}
    dual = {
        fold: metrics(fold_data[fold], EVENT_ORIG_WEIGHT, EVENT_SSL_WEIGHT)
        for fold in FOLDS
    }

    print("fold     original(g/m/w)          ssl(g/m/w)               dual(g/m/w)              dual-vs-original")
    fold_audit = {}
    month_audit = {}
    for fold in FOLDS:
        delta = dual[fold] - baseline[fold]
        fmt = lambda values: "/".join(f"{value:.6f}" for value in values)
        print(
            f"{fold:7s} {fmt(baseline[fold])}  {fmt(ssl_only[fold])}  "
            f"{fmt(dual[fold])}  {fmt(delta)}"
        )
        if np.min(delta) <= 0:
            raise RuntimeError(f"three-fold gate failed for {fold}: {delta}")
        base_pred = stack_prediction(fold_data[fold], 0.20, 0.0)
        dual_pred = stack_prediction(fold_data[fold], EVENT_ORIG_WEIGHT, EVENT_SSL_WEIGHT)
        fold_audit[fold] = {
            "original": baseline[fold].tolist(),
            "ssl_only": ssl_only[fold].tolist(),
            "dual": dual[fold].tolist(),
            "dual_vs_original": delta.tolist(),
        }
        month_audit[fold] = month_sign_audit(fold_data[fold], base_pred, dual_pred)

    loo = loo_audit(fold_data, baseline)
    print("LOO audit (grid=0.00..0.40 step .01; deterministic tie-break):")
    for row in loo:
        delta = "/".join(f"{value:+.6f}" for value in row["held_out_delta"])
        print(
            f" hold={row['held_out']} select=({row['selected_event_orig_weight']:.2f},"
            f"{row['selected_event_ssl_weight']:.2f}) held_delta={delta}"
        )
    for fold in FOLDS:
        row = month_audit[fold]
        print(
            f" month_sign {fold}: {row['positive']}/{row['months']} positive, "
            f"min={row['min_delta']:+.6f}"
        )

    ref_ids, ref = read_prediction(PUBLIC_REF)
    member_predictions = []
    for path in MEMBERS.values():
        ids, pred = read_prediction(path)
        if not np.array_equal(ids, ref_ids):
            raise ValueError(f"sample_id mismatch: {path}")
        member_predictions.append(u(pred))
    ids, event_orig = read_prediction(EVENT_ORIG)
    if not np.array_equal(ids, ref_ids):
        raise ValueError(f"sample_id mismatch: {EVENT_ORIG}")
    ids, event_ssl = read_prediction(EVENT_SSL)
    if not np.array_equal(ids, ref_ids):
        raise ValueError(f"sample_id mismatch: {EVENT_SSL}")

    weights = np.r_[W5, EVENT_ORIG_WEIGHT, EVENT_SSL_WEIGHT]
    weights /= weights.sum()
    self_dual = sum(weights[i] * pred for i, pred in enumerate(member_predictions))
    self_dual += weights[5] * u(event_orig) + weights[6] * u(event_ssl)
    final = u(0.40 * self_dual + 0.60 * u(ref))

    output = pd.DataFrame({"sample_id": ref_ids, "prediction": final})
    destination = Path(DEST)
    destination.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(destination, index=False)
    candidate_hash = sha256_file(destination)

    correlations = {}
    for label, path in (("best_55601441", BEST_PUBLIC), ("ssl_55666656", SSL_PUBLIC)):
        ids, prediction = read_prediction(path)
        if not np.array_equal(ids, ref_ids):
            raise ValueError(f"sample_id mismatch: {path}")
        correlations[label] = float(u(final) @ u(prediction))

    test_inputs = {
        "public_ref": PUBLIC_REF,
        **{f"member_{name}": path for name, path in MEMBERS.items()},
        "event_orig": EVENT_ORIG,
        "event_ssl": EVENT_SSL,
        "best_public_comparison": BEST_PUBLIC,
        "ssl_public_comparison": SSL_PUBLIC,
    }
    manifest = {
        "schema_version": 1,
        "candidate": str(destination),
        "candidate_sha256": candidate_hash,
        "formula": "unit(0.40*self_dual + 0.60*public_ref)",
        "base_weights": W5.tolist(),
        "event_orig_weight": EVENT_ORIG_WEIGHT,
        "event_ssl_weight": EVENT_SSL_WEIGHT,
        "normalized_self_stack_weights": weights.tolist(),
        "fold_audit": fold_audit,
        "loo_audit": loo,
        "month_sign_audit": month_audit,
        "test_correlations": correlations,
        "test_input_sha256": {label: sha256_file(path) for label, path in test_inputs.items()},
        "oof_input_sha256": {path: sha256_file(path) for path in OOF_INPUTS},
    }
    manifest_path = destination.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    print(f"saved {destination} rows={len(output)} sha256={candidate_hash}")
    print(f"manifest {manifest_path} sha256={sha256_file(manifest_path)}")
    for label, correlation in correlations.items():
        print(f"corr_vs_{label}={correlation:.8f}")


if __name__ == "__main__":
    main()
