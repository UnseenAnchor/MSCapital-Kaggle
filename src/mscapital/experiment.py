"""Experiment manifest recording: git commit, config, hashes, fold, metrics.

Writes one JSON manifest per experiment so every run is reproducible and
attributable (per the execution plan phase-4 contract).
"""
import json, os, subprocess, time, hashlib
from pathlib import Path


def git_head():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, errors="ignore").strip()
    except Exception:
        return "n/a"


def git_dirty():
    try:
        out = subprocess.check_output(["git", "status", "--porcelain"], text=True, errors="ignore").strip()
        return bool(out)
    except Exception:
        return True


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record_manifest(prefix, fold, metrics, config_path=None, extra=None, manifest_dir="output/manifests"):
    Path(manifest_dir).mkdir(parents=True, exist_ok=True)
    m = {
        "prefix": prefix,
        "fold": fold,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_commit": git_head(),
        "git_dirty": git_dirty(),
        "config_sha256": file_sha256(config_path) if config_path and os.path.exists(config_path) else None,
        "metrics": metrics,
        "extra": extra or {},
    }
    dest = f"{manifest_dir}/{prefix}_{fold}.json"
    with open(dest, "w") as f:
        json.dump(m, f, indent=2, ensure_ascii=False)
    return dest
