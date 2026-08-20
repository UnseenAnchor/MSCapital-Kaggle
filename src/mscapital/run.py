"""Unified experiment entrypoint.

    python -m mscapital.run --config configs/event_ssl_proxy.yaml [--fold proxy] [--skip-ssl]

Reads a YAML config, maps it to environment variables, and launches the
matching training script (currently the Event256 SSL/supervised pipeline).
"""
import argparse, os, subprocess, sys

# config key -> env var for the Event256/SSL family
ENV_MAP = {
    "event_root": "EVENT_ROOT",
    "length": "EVENT_LEN",
    "d_model": "D_MODEL",
    "n_layers": "N_LAYERS",
    "supervised_epochs": "EPOCHS",
    "lambda_cos": "LAMBDA_COS",
    "batch": "BS",
    "seed": "SEED",
    "lr": "LR",
    "weight_decay": "WEIGHT_DECAY",
}
SSL_ENV_MAP = {
    "epochs": "SSL_EPOCHS",
    "mask_ratio": "MASK_RATIO",
    "lr": "SSL_LR",
}
FOLD_ENV = {"proxy": 45, "middle": 51, "late": 62}
SCRIPT = os.path.join(os.path.dirname(__file__), "..", "..", "src", "train_event_ssl.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--fold", default="proxy", choices=sorted(FOLD_ENV))
    ap.add_argument("--skip-ssl", action="store_true", help="skip SSL init (pure supervised baseline run)")
    ap.add_argument("--pretrain-only", action="store_true", help="only run SSL pretraining")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    env = dict(os.environ)
    env["OUT_PREFIX"] = f"{args.fold}_event_ssl" if not args.skip_ssl else f"{args.fold}_event_ssl_nosssl"
    env["TRAIN_END"] = str(FOLD_ENV[args.fold])
    for k, v in ENV_MAP.items():
        if k in cfg:
            env[v] = str(cfg[k])
    ssl = cfg.get("ssl", {})
    for k, v in SSL_ENV_MAP.items():
        if k in ssl:
            env[v] = str(ssl[k])
    if args.skip_ssl:
        env["USE_SSL"] = "0"
    if args.pretrain_only:
        cmd = [sys.executable, os.path.join(os.path.dirname(SCRIPT), "pretrain_event_ssl.py")]
    else:
        cmd = [sys.executable, SCRIPT]
    print("launch", " ".join(cmd), "with fold", args.fold, flush=True)
    sys.exit(subprocess.call(cmd, env=env))


if __name__ == "__main__":
    main()
