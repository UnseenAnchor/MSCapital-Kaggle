"""Submit a prediction file via Kaggle REST API using the IAP bearer token.

Reads ~/.kaggle/access_token (KGAT_...) as Authorization: Bearer.
Works with the new Kaggle auth flow (no kaggle.json needed).

Usage: python src/submit_bearer.py <csv> <description>
"""
import json
import os
import ssl
import sys
import urllib.error
import urllib.request

COMP = "ms-capital-real-financial-market-forecasting"

_xsrf = {"cookie": None, "header": None}


def _load_xsrf():
    """Kaggle's new API requires X-XSRF-TOKEN (from the XSRF-TOKEN cookie)."""
    if _xsrf["header"]:
        return
    import urllib.parse
    req = urllib.request.Request(
        f"https://www.kaggle.com/competitions/{COMP}",
        headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=60, context=_ctx()) as r:
        cookies = []
        for k, v in r.headers.items():
            if k.lower() == "set-cookie":
                cookies.append(v)
    val = None
    for raw in cookies:
        for part in raw.split(";"):
            part = part.strip()
            if part.startswith("XSRF-TOKEN="):
                val = urllib.parse.unquote(part[len("XSRF-TOKEN="):])
                break
        if val:
            break
    _xsrf["cookie"] = val
    _xsrf["header"] = val
    print("   xsrf loaded:", (val or "")[:16] + "..." if val else "NONE", flush=True)


def _ctx():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl._create_unverified_context()


def api(path, data=None, method=None):
    token = open(os.path.expanduser("~/.kaggle/access_token")).read().strip()
    body = json.dumps(data).encode() if data is not None else None
    headers = {"Authorization": f"Bearer {token}",
               "Content-Type": "application/json"}
    if path not in ("competitions/list",) and _xsrf["header"]:
        headers["X-XSRF-TOKEN"] = _xsrf["header"]
        if _xsrf["cookie"]:
            headers["Cookie"] = f"XSRF-TOKEN={_xsrf['cookie']}"
    req = urllib.request.Request(
        f"https://www.kaggle.com/api/v1/{path}",
        data=body,
        method=method or ("POST" if body else "GET"),
        headers=headers,
    )
    try:
        with urllib.request.urlopen(req, timeout=600, context=_ctx()) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        print("HTTP", e.code, e.read().decode()[:600], flush=True)
        raise


def submit(file_path, message):
    f = file_path
    print("1. start upload", os.path.basename(f), os.path.getsize(f), flush=True)
    _load_xsrf()
    start = api("competitions/submission-url", {
        "competition_name": COMP,
        "file_name": os.path.basename(f),
        "content_length": os.path.getsize(f),
        "last_modified_epoch_seconds": int(os.path.getmtime(f)),
    })
    print("   token:", str(start.get("token"))[:40], "...", flush=True)
    print("2. PUT to signed url", flush=True)
    with open(f, "rb") as fh:
        data = fh.read()
    req = urllib.request.Request(start["createUrl"], data=data, method="PUT",
                                 headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=600, context=_ctx()) as r:
        print("   upload status:", r.status, flush=True)
    print("3. create submission", flush=True)
    resp = api("competitions/submission-create", {
        "competition_name": COMP,
        "blob_file_tokens": start["token"],
        "submission_description": message,
    })
    print("SUBMITTED ref:", resp.get("ref"), "| message:", resp.get("message"), flush=True)
    return resp.get("ref")


if __name__ == "__main__":
    f = sys.argv[1] if len(sys.argv) > 1 else "output/submission.csv"
    m = sys.argv[2] if len(sys.argv) > 2 else "auto_submit"
    submit(f, m)
