import gc
import os
import time
from pathlib import Path

os.environ.setdefault("POLARS_MAX_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.dataset as ds
    import pyarrow.feather as feather
    import pyarrow.ipc as ipc
except ModuleNotFoundError:
    pa = None
    ds = None
    feather = None
    ipc = None

try:
    import lightgbm as lgb
except ModuleNotFoundError:
    lgb = None


DATA = Path(os.environ.get(
    "DATA_DIR",
    "data",
))
OUT_CSV = os.environ.get("OUT_CSV", "output/submission_micro_v3.csv")

MARKET_WINDOWS = tuple(int(x) for x in os.environ.get("MARKET_WINDOWS", "30,60,300,600").split(",") if x.strip())
FLOW_WINDOWS = tuple(int(x) for x in os.environ.get("FLOW_WINDOWS", "10,30,60").split(",") if x.strip())
USE_TRANSACTION = os.environ.get("USE_TRANSACTION", "1") == "1"
USE_ORDER = os.environ.get("USE_ORDER", "1") == "1"
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "65536"))
N_THREADS = int(os.environ.get("N_THREADS", "10"))
VALID_START_MONTH = int(os.environ.get("VALID_START_MONTH", "62"))
NUM_BOOST_ROUND = int(os.environ.get("NUM_BOOST_ROUND", "3500"))
EARLY_STOPPING = int(os.environ.get("EARLY_STOPPING", "180"))
SEED = int(os.environ.get("SEED", "7"))
EPS = 1e-6

MARKET_COLUMNS = [
    "sample_id",
    "seconds_before_predict",
    "transaction_avgprice",
    "transaction_volume",
    "transaction_count",
    "ask_price_1",
    "ask_price_2",
    "bid_price_1",
    "bid_price_2",
    "ask_volume_1",
    "ask_volume_2",
    "bid_volume_1",
    "bid_volume_2",
]
TRANSACTION_COLUMNS = ["sample_id", "seconds_before_predict", "price", "volume", "side"]
ORDER_COLUMNS = ["sample_id", "seconds_before_predict", "price", "volume", "side", "order_action"]


def require_runtime():
    if pa is None or feather is None or ipc is None:
        raise ModuleNotFoundError("pyarrow is required.")
    if lgb is None:
        raise ModuleNotFoundError("lightgbm is required.")


def cos_uncenter(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def lgb_cos_metric(preds, dataset):
    return "cos_uncenter", cos_uncenter(preds, dataset.get_label()), True


def read_train_label():
    tab = feather.read_table(
        str(DATA / "train" / "label.feather"),
        columns=["sample_id", "month", "target"],
        memory_map=False,
    )
    sid = tab["sample_id"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    month = tab["month"].to_numpy(zero_copy_only=False).astype(np.int16, copy=False)
    target = tab["target"].to_numpy(zero_copy_only=False).astype(np.float32, copy=False)
    order = np.argsort(sid)
    return sid[order], month[order], target[order]


def read_submission_ids():
    ids = []
    with open(DATA / "submission.csv", "r") as f:
        next(f)
        for line in f:
            if line:
                ids.append(int(line.split(",", 1)[0]))
    return np.asarray(ids, dtype=np.int64)


def build_indexer(sample_ids):
    max_id = int(sample_ids.max())
    if max_id <= 25_000_000:
        indexer = np.full(max_id + 1, -1, dtype=np.int32)
        indexer[sample_ids] = np.arange(sample_ids.size, dtype=np.int32)
        return "direct", indexer

    mapping = {int(s): i for i, s in enumerate(sample_ids)}
    return "dict", mapping


def map_ids(batch_ids, indexer):
    kind, payload = indexer
    if kind == "direct":
        out = np.full(batch_ids.size, -1, dtype=np.int32)
        ok = (batch_ids >= 0) & (batch_ids < payload.size)
        out[ok] = payload[batch_ids[ok]]
        return out

    return np.fromiter((payload.get(int(x), -1) for x in batch_ids), dtype=np.int32, count=batch_ids.size)


def iter_record_batches(path, columns):
    if ds is not None:
        try:
            dataset = ds.dataset(str(path), format="ipc")
            scanner = dataset.scanner(columns=columns, batch_size=BATCH_SIZE, use_threads=False)
            for i, batch in enumerate(scanner.to_batches(), start=1):
                yield batch, i, None
            return
        except Exception as exc:
            print(f"  dataset scanner fallback for {path.name}: {type(exc).__name__}: {exc}", flush=True)

    source = pa.memory_map(str(path), "r")
    try:
        try:
            reader = ipc.RecordBatchFileReader(source)
            for i in range(reader.num_record_batches):
                batch = reader.get_batch(i)
                try:
                    batch = batch.select(columns)
                except Exception:
                    pass
                yield batch, i + 1, reader.num_record_batches
        except pa.ArrowInvalid:
            source.seek(0)
            reader = ipc.open_stream(source)
            for i, batch in enumerate(reader, start=1):
                try:
                    batch = batch.select(columns)
                except Exception:
                    pass
                yield batch, i, None
    finally:
        source.close()


def col_np(batch, name, dtype=None):
    arr = batch.column(batch.schema.get_field_index(name)).to_numpy(zero_copy_only=False)
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def add_bincount(dst, idx, values=None):
    if values is None:
        bc = np.bincount(idx, minlength=dst.size)
    else:
        bc = np.bincount(idx, weights=values, minlength=dst.size)
    if bc.dtype != dst.dtype:
        bc = bc.astype(dst.dtype, copy=False)
    dst += bc


def update_latest(idx, sec, value_arrays, best_sec, best_values, choose_min=True):
    if idx.size == 0:
        return

    key_sec = sec if choose_min else -sec
    order = np.lexsort((key_sec, idx))
    idx_s = idx[order]
    first = np.empty(idx_s.size, dtype=bool)
    first[0] = True
    first[1:] = idx_s[1:] != idx_s[:-1]
    pos = order[first]
    uidx = idx[pos]

    better = sec[pos] < best_sec[uidx] if choose_min else sec[pos] > best_sec[uidx]
    if not np.any(better):
        return

    uidx = uidx[better]
    pos = pos[better]
    best_sec[uidx] = sec[pos]
    for src, dst in zip(value_arrays, best_values):
        dst[uidx] = src[pos]


def finalize_mean(sum_arr, count_arr):
    out = np.full(sum_arr.shape, np.nan, dtype=np.float32)
    ok = count_arr > 0
    out[ok] = (sum_arr[ok] / count_arr[ok]).astype(np.float32)
    return out


def finalize_std(sum_arr, sq_arr, count_arr):
    out = np.full(sum_arr.shape, np.nan, dtype=np.float32)
    ok = count_arr > 1
    mean = sum_arr[ok] / count_arr[ok]
    var = np.maximum(sq_arr[ok] / count_arr[ok] - mean * mean, 0.0)
    out[ok] = np.sqrt(var).astype(np.float32)
    return out


def safe_div(num, den):
    out = np.full(np.asarray(num).shape, np.nan, dtype=np.float32)
    den_arr = np.asarray(den)
    ok = np.abs(den_arr) > EPS
    out[ok] = (np.asarray(num)[ok] / den_arr[ok]).astype(np.float32)
    return out


def ratio0(num, den):
    return (np.asarray(num) / (np.asarray(den) + EPS)).astype(np.float32)


def log_return(last, first):
    return np.where((last > EPS) & (first > EPS), np.log(last / first), np.nan).astype(np.float32)


def print_progress(split, name, batch_no, total_batches, rows_seen, t0):
    if batch_no % 200 != 0:
        return
    if total_batches is None:
        label = f"batch {batch_no}"
    else:
        label = f"batch {batch_no}/{total_batches}"
    print(f"  {split} {name} {label}, rows={rows_seen:,}, elapsed={time.time() - t0:.1f}s", flush=True)


def aggregate_market(split, sample_ids):
    n = sample_ids.size
    indexer = build_indexer(sample_ids)
    t0 = time.time()

    count = np.zeros(n, dtype=np.int32)
    mid_sum = np.zeros(n, dtype=np.float64)
    mid_sq = np.zeros(n, dtype=np.float64)
    mid_min = np.full(n, np.inf, dtype=np.float32)
    mid_max = np.full(n, -np.inf, dtype=np.float32)
    spread1_sum = np.zeros(n, dtype=np.float64)
    spread1_sq = np.zeros(n, dtype=np.float64)
    spread2_sum = np.zeros(n, dtype=np.float64)
    imb1_sum = np.zeros(n, dtype=np.float64)
    imb1_sq = np.zeros(n, dtype=np.float64)
    imb2_sum = np.zeros(n, dtype=np.float64)
    imb2_sq = np.zeros(n, dtype=np.float64)
    micro1_sum = np.zeros(n, dtype=np.float64)
    micro2_sum = np.zeros(n, dtype=np.float64)
    log_depth_sum = np.zeros(n, dtype=np.float64)
    log_depth_sq = np.zeros(n, dtype=np.float64)
    slope_sum = np.zeros(n, dtype=np.float64)
    txv_sum = np.zeros(n, dtype=np.float64)
    txc_sum = np.zeros(n, dtype=np.float64)
    amount_sum = np.zeros(n, dtype=np.float64)

    latest_sec = np.full(n, np.inf, dtype=np.float32)
    latest_mid = np.full(n, np.nan, dtype=np.float32)
    latest_micro1 = np.full(n, np.nan, dtype=np.float32)
    latest_spread1 = np.full(n, np.nan, dtype=np.float32)
    latest_imb1 = np.full(n, np.nan, dtype=np.float32)
    latest_imb2 = np.full(n, np.nan, dtype=np.float32)
    latest_log_depth = np.full(n, np.nan, dtype=np.float32)
    oldest_sec = np.full(n, -np.inf, dtype=np.float32)
    oldest_mid = np.full(n, np.nan, dtype=np.float32)

    win = {}
    for w in MARKET_WINDOWS:
        win[w] = {
            "count": np.zeros(n, dtype=np.int32),
            "mid_sum": np.zeros(n, dtype=np.float64),
            "mid_sq": np.zeros(n, dtype=np.float64),
            "spread1_sum": np.zeros(n, dtype=np.float64),
            "imb1_sum": np.zeros(n, dtype=np.float64),
            "imb2_sum": np.zeros(n, dtype=np.float64),
            "micro1_sum": np.zeros(n, dtype=np.float64),
            "log_depth_sum": np.zeros(n, dtype=np.float64),
            "txv_sum": np.zeros(n, dtype=np.float64),
            "txc_sum": np.zeros(n, dtype=np.float64),
            "amount_sum": np.zeros(n, dtype=np.float64),
        }

    rows_seen = 0
    for batch, batch_no, total_batches in iter_record_batches(DATA / split / "market.feather", MARKET_COLUMNS):
        sid = col_np(batch, "sample_id", np.int64)
        idx = map_ids(sid, indexer)
        valid = idx >= 0
        if not np.all(valid):
            idx = idx[valid]

        sec = col_np(batch, "seconds_before_predict", np.float32)
        ask1 = col_np(batch, "ask_price_1", np.float32)
        ask2 = col_np(batch, "ask_price_2", np.float32)
        bid1 = col_np(batch, "bid_price_1", np.float32)
        bid2 = col_np(batch, "bid_price_2", np.float32)
        askv1 = col_np(batch, "ask_volume_1", np.float64)
        askv2 = col_np(batch, "ask_volume_2", np.float64)
        bidv1 = col_np(batch, "bid_volume_1", np.float64)
        bidv2 = col_np(batch, "bid_volume_2", np.float64)
        avgpx = col_np(batch, "transaction_avgprice", np.float64)
        txv = col_np(batch, "transaction_volume", np.float64)
        txc = col_np(batch, "transaction_count", np.float64)

        if not np.all(valid):
            sec = sec[valid]
            ask1 = ask1[valid]
            ask2 = ask2[valid]
            bid1 = bid1[valid]
            bid2 = bid2[valid]
            askv1 = askv1[valid]
            askv2 = askv2[valid]
            bidv1 = bidv1[valid]
            bidv2 = bidv2[valid]
            avgpx = avgpx[valid]
            txv = txv[valid]
            txc = txc[valid]

        mid = ((ask1 + bid1) * 0.5).astype(np.float32, copy=False)
        spread1 = ((ask1 - bid1) / (mid + EPS)).astype(np.float32, copy=False)
        spread2 = ((ask2 - bid2) / (mid + EPS)).astype(np.float32, copy=False)
        depth1 = askv1 + bidv1
        depth2 = depth1 + askv2 + bidv2
        bid_depth = bidv1 + bidv2
        ask_depth = askv1 + askv2
        imb1 = ((bidv1 - askv1) / (depth1 + EPS)).astype(np.float32, copy=False)
        imb2 = ((bid_depth - ask_depth) / (depth2 + EPS)).astype(np.float32, copy=False)
        micro1 = ((ask1 * bidv1 + bid1 * askv1) / (depth1 + EPS)).astype(np.float32, copy=False)
        micro2 = (
            (ask1 * bidv1 + ask2 * bidv2 + bid1 * askv1 + bid2 * askv2) / (depth2 + EPS)
        ).astype(np.float32, copy=False)
        log_depth = np.log1p(depth2).astype(np.float32, copy=False)
        slope = (((ask2 - ask1) + (bid1 - bid2)) / (mid + EPS)).astype(np.float32, copy=False)
        amount = avgpx * txv

        add_bincount(count, idx)
        add_bincount(mid_sum, idx, mid)
        add_bincount(mid_sq, idx, mid.astype(np.float64) * mid)
        np.minimum.at(mid_min, idx, mid)
        np.maximum.at(mid_max, idx, mid)
        add_bincount(spread1_sum, idx, spread1)
        add_bincount(spread1_sq, idx, spread1.astype(np.float64) * spread1)
        add_bincount(spread2_sum, idx, spread2)
        add_bincount(imb1_sum, idx, imb1)
        add_bincount(imb1_sq, idx, imb1.astype(np.float64) * imb1)
        add_bincount(imb2_sum, idx, imb2)
        add_bincount(imb2_sq, idx, imb2.astype(np.float64) * imb2)
        add_bincount(micro1_sum, idx, micro1)
        add_bincount(micro2_sum, idx, micro2)
        add_bincount(log_depth_sum, idx, log_depth)
        add_bincount(log_depth_sq, idx, log_depth.astype(np.float64) * log_depth)
        add_bincount(slope_sum, idx, slope)
        add_bincount(txv_sum, idx, txv)
        add_bincount(txc_sum, idx, txc)
        add_bincount(amount_sum, idx, amount)

        update_latest(
            idx,
            sec,
            [mid, micro1, spread1, imb1, imb2, log_depth],
            latest_sec,
            [latest_mid, latest_micro1, latest_spread1, latest_imb1, latest_imb2, latest_log_depth],
            choose_min=True,
        )
        update_latest(idx, sec, [mid], oldest_sec, [oldest_mid], choose_min=False)

        for w in MARKET_WINDOWS:
            mask = sec <= w
            if not np.any(mask):
                continue
            j = idx[mask]
            ww = win[w]
            mid_w = mid[mask]
            add_bincount(ww["count"], j)
            add_bincount(ww["mid_sum"], j, mid_w)
            add_bincount(ww["mid_sq"], j, mid_w.astype(np.float64) * mid_w)
            add_bincount(ww["spread1_sum"], j, spread1[mask])
            add_bincount(ww["imb1_sum"], j, imb1[mask])
            add_bincount(ww["imb2_sum"], j, imb2[mask])
            add_bincount(ww["micro1_sum"], j, micro1[mask])
            add_bincount(ww["log_depth_sum"], j, log_depth[mask])
            add_bincount(ww["txv_sum"], j, txv[mask])
            add_bincount(ww["txc_sum"], j, txc[mask])
            add_bincount(ww["amount_sum"], j, amount[mask])

        rows_seen += batch.num_rows
        print_progress(split, "market", batch_no, total_batches, rows_seen, t0)
        del sid, idx, valid, sec, ask1, ask2, bid1, bid2, askv1, askv2, bidv1, bidv2, avgpx, txv, txc
        del mid, spread1, spread2, depth1, depth2, bid_depth, ask_depth, imb1, imb2, micro1, micro2
        del log_depth, slope, amount

    features = {
        "m_count": count.astype(np.float32),
        "m_mid_last": latest_mid,
        "m_mid_first": oldest_mid,
        "m_mid_mean": finalize_mean(mid_sum, count),
        "m_mid_std": finalize_std(mid_sum, mid_sq, count),
        "m_mid_range": (mid_max - mid_min).astype(np.float32),
        "m_micro1_last": latest_micro1,
        "m_micro1_mean": finalize_mean(micro1_sum, count),
        "m_micro2_mean": finalize_mean(micro2_sum, count),
        "m_spread_last": latest_spread1,
        "m_spread_mean": finalize_mean(spread1_sum, count),
        "m_spread_std": finalize_std(spread1_sum, spread1_sq, count),
        "m_spread2_mean": finalize_mean(spread2_sum, count),
        "m_imb1_last": latest_imb1,
        "m_imb1_mean": finalize_mean(imb1_sum, count),
        "m_imb1_std": finalize_std(imb1_sum, imb1_sq, count),
        "m_imb2_last": latest_imb2,
        "m_imb2_mean": finalize_mean(imb2_sum, count),
        "m_imb2_std": finalize_std(imb2_sum, imb2_sq, count),
        "m_log_depth_last": latest_log_depth,
        "m_log_depth_mean": finalize_mean(log_depth_sum, count),
        "m_log_depth_std": finalize_std(log_depth_sum, log_depth_sq, count),
        "m_slope_mean": finalize_mean(slope_sum, count),
        "m_txv_sum": txv_sum.astype(np.float32),
        "m_txc_sum": txc_sum.astype(np.float32),
        "m_vwap": safe_div(amount_sum, txv_sum),
    }
    features["m_ret_full"] = log_return(latest_mid, oldest_mid)
    features["m_micro1_rel_last"] = (latest_micro1 / (latest_mid + EPS) - 1.0).astype(np.float32)
    features["m_vwap_rel"] = (features["m_vwap"] / (latest_mid + EPS) - 1.0).astype(np.float32)
    features["m_mid_mean_rel"] = (features["m_mid_mean"] / (latest_mid + EPS) - 1.0).astype(np.float32)

    for w in MARKET_WINDOWS:
        ww = win[w]
        features[f"m_count_{w}"] = ww["count"].astype(np.float32)
        features[f"m_mid_mean_{w}"] = finalize_mean(ww["mid_sum"], ww["count"])
        features[f"m_mid_std_{w}"] = finalize_std(ww["mid_sum"], ww["mid_sq"], ww["count"])
        features[f"m_spread_mean_{w}"] = finalize_mean(ww["spread1_sum"], ww["count"])
        features[f"m_imb1_mean_{w}"] = finalize_mean(ww["imb1_sum"], ww["count"])
        features[f"m_imb2_mean_{w}"] = finalize_mean(ww["imb2_sum"], ww["count"])
        features[f"m_micro1_mean_{w}"] = finalize_mean(ww["micro1_sum"], ww["count"])
        features[f"m_log_depth_mean_{w}"] = finalize_mean(ww["log_depth_sum"], ww["count"])
        features[f"m_txv_sum_{w}"] = ww["txv_sum"].astype(np.float32)
        features[f"m_txc_sum_{w}"] = ww["txc_sum"].astype(np.float32)
        features[f"m_vwap_{w}"] = safe_div(ww["amount_sum"], ww["txv_sum"])
        features[f"m_mid_mean_rel_{w}"] = (features[f"m_mid_mean_{w}"] / (latest_mid + EPS) - 1.0).astype(np.float32)
        features[f"m_micro1_rel_{w}"] = (features[f"m_micro1_mean_{w}"] / (latest_mid + EPS) - 1.0).astype(np.float32)
        features[f"m_vwap_rel_{w}"] = (features[f"m_vwap_{w}"] / (latest_mid + EPS) - 1.0).astype(np.float32)
        features[f"m_txv_share_{w}"] = ratio0(ww["txv_sum"], txv_sum + 1.0)
        features[f"m_txc_share_{w}"] = ratio0(ww["txc_sum"], txc_sum + 1.0)

    print(f"  {split} market done: rows={rows_seen:,}, features={len(features)}, time={time.time() - t0:.1f}s", flush=True)
    return features


def aggregate_transaction(split, sample_ids):
    n = sample_ids.size
    indexer = build_indexer(sample_ids)
    t0 = time.time()

    count = np.zeros(n, dtype=np.int32)
    vol_sum = np.zeros(n, dtype=np.float64)
    vol_sq_sum = np.zeros(n, dtype=np.float64)
    amount_sum = np.zeros(n, dtype=np.float64)
    price_sum = np.zeros(n, dtype=np.float64)
    price_sq_sum = np.zeros(n, dtype=np.float64)
    price_min = np.full(n, np.inf, dtype=np.float32)
    price_max = np.full(n, -np.inf, dtype=np.float32)
    buy_vol = np.zeros(n, dtype=np.float64)
    sell_vol = np.zeros(n, dtype=np.float64)
    buy_count = np.zeros(n, dtype=np.float64)
    sell_count = np.zeros(n, dtype=np.float64)
    signed_vol = np.zeros(n, dtype=np.float64)
    signed_amount = np.zeros(n, dtype=np.float64)
    log_vol_sum = np.zeros(n, dtype=np.float64)
    latest_sec = np.full(n, np.inf, dtype=np.float32)
    latest_price = np.full(n, np.nan, dtype=np.float32)
    latest_signed = np.full(n, np.nan, dtype=np.float32)
    oldest_sec = np.full(n, -np.inf, dtype=np.float32)
    oldest_price = np.full(n, np.nan, dtype=np.float32)

    win = {}
    for w in FLOW_WINDOWS:
        win[w] = {
            "count": np.zeros(n, dtype=np.int32),
            "vol_sum": np.zeros(n, dtype=np.float64),
            "amount_sum": np.zeros(n, dtype=np.float64),
            "price_sum": np.zeros(n, dtype=np.float64),
            "price_sq_sum": np.zeros(n, dtype=np.float64),
            "buy_vol": np.zeros(n, dtype=np.float64),
            "sell_vol": np.zeros(n, dtype=np.float64),
            "buy_count": np.zeros(n, dtype=np.float64),
            "sell_count": np.zeros(n, dtype=np.float64),
            "signed_vol": np.zeros(n, dtype=np.float64),
            "signed_amount": np.zeros(n, dtype=np.float64),
        }

    rows_seen = 0
    for batch, batch_no, total_batches in iter_record_batches(DATA / split / "transaction.feather", TRANSACTION_COLUMNS):
        sid = col_np(batch, "sample_id", np.int64)
        idx = map_ids(sid, indexer)
        valid = idx >= 0
        if not np.all(valid):
            idx = idx[valid]

        sec = col_np(batch, "seconds_before_predict", np.float32)
        price = col_np(batch, "price", np.float32)
        volume = col_np(batch, "volume", np.float64)
        side = col_np(batch, "side", np.int8)
        if not np.all(valid):
            sec = sec[valid]
            price = price[valid]
            volume = volume[valid]
            side = side[valid]

        sgn = np.where(side == 0, 1.0, -1.0).astype(np.float32)
        amount = price.astype(np.float64) * volume
        buy = np.where(side == 0, volume, 0.0)
        sell = np.where(side == 1, volume, 0.0)
        buy_n = (side == 0).astype(np.float32)
        sell_n = (side == 1).astype(np.float32)
        sv = sgn.astype(np.float64) * volume
        sa = sgn.astype(np.float64) * amount
        log_vol = np.log1p(volume).astype(np.float32, copy=False)

        add_bincount(count, idx)
        add_bincount(vol_sum, idx, volume)
        add_bincount(vol_sq_sum, idx, volume * volume)
        add_bincount(amount_sum, idx, amount)
        add_bincount(price_sum, idx, price)
        add_bincount(price_sq_sum, idx, price.astype(np.float64) * price)
        np.minimum.at(price_min, idx, price)
        np.maximum.at(price_max, idx, price)
        add_bincount(buy_vol, idx, buy)
        add_bincount(sell_vol, idx, sell)
        add_bincount(buy_count, idx, buy_n)
        add_bincount(sell_count, idx, sell_n)
        add_bincount(signed_vol, idx, sv)
        add_bincount(signed_amount, idx, sa)
        add_bincount(log_vol_sum, idx, log_vol)
        update_latest(idx, sec, [price, sgn], latest_sec, [latest_price, latest_signed], choose_min=True)
        update_latest(idx, sec, [price], oldest_sec, [oldest_price], choose_min=False)

        for w in FLOW_WINDOWS:
            mask = sec <= w
            if not np.any(mask):
                continue
            j = idx[mask]
            ww = win[w]
            p_w = price[mask]
            add_bincount(ww["count"], j)
            add_bincount(ww["vol_sum"], j, volume[mask])
            add_bincount(ww["amount_sum"], j, amount[mask])
            add_bincount(ww["price_sum"], j, p_w)
            add_bincount(ww["price_sq_sum"], j, p_w.astype(np.float64) * p_w)
            add_bincount(ww["buy_vol"], j, buy[mask])
            add_bincount(ww["sell_vol"], j, sell[mask])
            add_bincount(ww["buy_count"], j, buy_n[mask])
            add_bincount(ww["sell_count"], j, sell_n[mask])
            add_bincount(ww["signed_vol"], j, sv[mask])
            add_bincount(ww["signed_amount"], j, sa[mask])

        rows_seen += batch.num_rows
        print_progress(split, "transaction", batch_no, total_batches, rows_seen, t0)
        del sid, idx, valid, sec, price, volume, side, sgn, amount, buy, sell, buy_n, sell_n, sv, sa, log_vol

    features = {
        "t_count": count.astype(np.float32),
        "t_price_last": latest_price,
        "t_price_first": oldest_price,
        "t_price_mean": finalize_mean(price_sum, count),
        "t_price_std": finalize_std(price_sum, price_sq_sum, count),
        "t_price_range": (price_max - price_min).astype(np.float32),
        "t_vol_sum": vol_sum.astype(np.float32),
        "t_vol_mean": finalize_mean(vol_sum, count),
        "t_vol_std": finalize_std(vol_sum, vol_sq_sum, count),
        "t_amount_sum": amount_sum.astype(np.float32),
        "t_vwap": safe_div(amount_sum, vol_sum),
        "t_buy_vol": buy_vol.astype(np.float32),
        "t_sell_vol": sell_vol.astype(np.float32),
        "t_buy_count": buy_count.astype(np.float32),
        "t_sell_count": sell_count.astype(np.float32),
        "t_signed_vol": signed_vol.astype(np.float32),
        "t_signed_amount": signed_amount.astype(np.float32),
        "t_log_vol_mean": finalize_mean(log_vol_sum, count),
        "t_last_signed_side": latest_signed,
    }
    features["t_ret_full"] = log_return(latest_price, oldest_price)
    features["t_buy_vol_share"] = ratio0(buy_vol, vol_sum + 1.0)
    features["t_buy_trade_ratio"] = ratio0(buy_count, count + 1.0)
    features["t_signed_ratio"] = ratio0(signed_vol, vol_sum + 1.0)
    features["t_vol_imb"] = ratio0(buy_vol - sell_vol, vol_sum + 1.0)
    features["t_signed_amount_per_vol"] = ratio0(signed_amount, vol_sum + 1.0)

    for w in FLOW_WINDOWS:
        ww = win[w]
        features[f"t_count_{w}"] = ww["count"].astype(np.float32)
        features[f"t_vol_sum_{w}"] = ww["vol_sum"].astype(np.float32)
        features[f"t_price_mean_{w}"] = finalize_mean(ww["price_sum"], ww["count"])
        features[f"t_price_std_{w}"] = finalize_std(ww["price_sum"], ww["price_sq_sum"], ww["count"])
        features[f"t_vwap_{w}"] = safe_div(ww["amount_sum"], ww["vol_sum"])
        features[f"t_buy_vol_{w}"] = ww["buy_vol"].astype(np.float32)
        features[f"t_sell_vol_{w}"] = ww["sell_vol"].astype(np.float32)
        features[f"t_buy_count_{w}"] = ww["buy_count"].astype(np.float32)
        features[f"t_sell_count_{w}"] = ww["sell_count"].astype(np.float32)
        features[f"t_signed_vol_{w}"] = ww["signed_vol"].astype(np.float32)
        features[f"t_signed_amount_{w}"] = ww["signed_amount"].astype(np.float32)
        features[f"t_signed_ratio_{w}"] = ratio0(ww["signed_vol"], ww["vol_sum"] + 1.0)
        features[f"t_vol_imb_{w}"] = ratio0(ww["buy_vol"] - ww["sell_vol"], ww["vol_sum"] + 1.0)
        features[f"t_buy_vol_share_{w}"] = ratio0(ww["buy_vol"], ww["vol_sum"] + 1.0)
        features[f"t_buy_trade_ratio_{w}"] = ratio0(ww["buy_count"], ww["count"] + 1.0)
        features[f"t_vol_share_{w}"] = ratio0(ww["vol_sum"], vol_sum + 1.0)
        features[f"t_count_share_{w}"] = ratio0(ww["count"], count + 1.0)

    print(f"  {split} transaction done: rows={rows_seen:,}, features={len(features)}, time={time.time() - t0:.1f}s", flush=True)
    return features


def aggregate_order(split, sample_ids):
    n = sample_ids.size
    indexer = build_indexer(sample_ids)
    t0 = time.time()

    count = np.zeros(n, dtype=np.int32)
    vol_sum = np.zeros(n, dtype=np.float64)
    vol_sq_sum = np.zeros(n, dtype=np.float64)
    price_sum = np.zeros(n, dtype=np.float64)
    price_sq_sum = np.zeros(n, dtype=np.float64)
    signed_vol = np.zeros(n, dtype=np.float64)
    action_vol = np.zeros(n, dtype=np.float64)
    pressure = np.zeros(n, dtype=np.float64)
    new_vol = np.zeros(n, dtype=np.float64)
    cancel_vol = np.zeros(n, dtype=np.float64)
    buy_new = np.zeros(n, dtype=np.float64)
    sell_new = np.zeros(n, dtype=np.float64)
    buy_cancel = np.zeros(n, dtype=np.float64)
    sell_cancel = np.zeros(n, dtype=np.float64)
    log_vol_sum = np.zeros(n, dtype=np.float64)
    latest_sec = np.full(n, np.inf, dtype=np.float32)
    latest_price = np.full(n, np.nan, dtype=np.float32)
    latest_side = np.full(n, np.nan, dtype=np.float32)
    latest_action = np.full(n, np.nan, dtype=np.float32)

    win = {}
    for w in FLOW_WINDOWS:
        win[w] = {
            "count": np.zeros(n, dtype=np.int32),
            "vol_sum": np.zeros(n, dtype=np.float64),
            "signed_vol": np.zeros(n, dtype=np.float64),
            "action_vol": np.zeros(n, dtype=np.float64),
            "pressure": np.zeros(n, dtype=np.float64),
            "new_vol": np.zeros(n, dtype=np.float64),
            "cancel_vol": np.zeros(n, dtype=np.float64),
            "buy_new": np.zeros(n, dtype=np.float64),
            "sell_new": np.zeros(n, dtype=np.float64),
            "buy_cancel": np.zeros(n, dtype=np.float64),
            "sell_cancel": np.zeros(n, dtype=np.float64),
            "price_sum": np.zeros(n, dtype=np.float64),
            "price_sq_sum": np.zeros(n, dtype=np.float64),
        }

    rows_seen = 0
    for batch, batch_no, total_batches in iter_record_batches(DATA / split / "order.feather", ORDER_COLUMNS):
        sid = col_np(batch, "sample_id", np.int64)
        idx = map_ids(sid, indexer)
        valid = idx >= 0
        if not np.all(valid):
            idx = idx[valid]

        sec = col_np(batch, "seconds_before_predict", np.float32)
        price = col_np(batch, "price", np.float32)
        volume = col_np(batch, "volume", np.float64)
        side = col_np(batch, "side", np.int8)
        action = col_np(batch, "order_action", np.int8)
        if not np.all(valid):
            sec = sec[valid]
            price = price[valid]
            volume = volume[valid]
            side = side[valid]
            action = action[valid]

        sgn = np.where(side == 0, 1.0, -1.0).astype(np.float32)
        act = np.where(action == 0, 1.0, -1.0).astype(np.float32)
        is_new = action == 0
        is_cancel = action == 1
        is_buy = side == 0
        is_sell = side == 1
        sv = sgn.astype(np.float64) * volume
        av = act.astype(np.float64) * volume
        pres = sgn.astype(np.float64) * act.astype(np.float64) * volume
        nv = np.where(is_new, volume, 0.0)
        cv = np.where(is_cancel, volume, 0.0)
        bn = np.where(is_buy & is_new, volume, 0.0)
        sn = np.where(is_sell & is_new, volume, 0.0)
        bc = np.where(is_buy & is_cancel, volume, 0.0)
        sc = np.where(is_sell & is_cancel, volume, 0.0)
        log_vol = np.log1p(volume).astype(np.float32, copy=False)

        add_bincount(count, idx)
        add_bincount(vol_sum, idx, volume)
        add_bincount(vol_sq_sum, idx, volume * volume)
        add_bincount(price_sum, idx, price)
        add_bincount(price_sq_sum, idx, price.astype(np.float64) * price)
        add_bincount(signed_vol, idx, sv)
        add_bincount(action_vol, idx, av)
        add_bincount(pressure, idx, pres)
        add_bincount(new_vol, idx, nv)
        add_bincount(cancel_vol, idx, cv)
        add_bincount(buy_new, idx, bn)
        add_bincount(sell_new, idx, sn)
        add_bincount(buy_cancel, idx, bc)
        add_bincount(sell_cancel, idx, sc)
        add_bincount(log_vol_sum, idx, log_vol)
        update_latest(idx, sec, [price, sgn, act], latest_sec, [latest_price, latest_side, latest_action], choose_min=True)

        for w in FLOW_WINDOWS:
            mask = sec <= w
            if not np.any(mask):
                continue
            j = idx[mask]
            ww = win[w]
            p_w = price[mask]
            add_bincount(ww["count"], j)
            add_bincount(ww["vol_sum"], j, volume[mask])
            add_bincount(ww["signed_vol"], j, sv[mask])
            add_bincount(ww["action_vol"], j, av[mask])
            add_bincount(ww["pressure"], j, pres[mask])
            add_bincount(ww["new_vol"], j, nv[mask])
            add_bincount(ww["cancel_vol"], j, cv[mask])
            add_bincount(ww["buy_new"], j, bn[mask])
            add_bincount(ww["sell_new"], j, sn[mask])
            add_bincount(ww["buy_cancel"], j, bc[mask])
            add_bincount(ww["sell_cancel"], j, sc[mask])
            add_bincount(ww["price_sum"], j, p_w)
            add_bincount(ww["price_sq_sum"], j, p_w.astype(np.float64) * p_w)

        rows_seen += batch.num_rows
        print_progress(split, "order", batch_no, total_batches, rows_seen, t0)
        del sid, idx, valid, sec, price, volume, side, action, sgn, act, is_new, is_cancel, is_buy, is_sell
        del sv, av, pres, nv, cv, bn, sn, bc, sc, log_vol

    features = {
        "o_count": count.astype(np.float32),
        "o_price_last": latest_price,
        "o_price_mean": finalize_mean(price_sum, count),
        "o_price_std": finalize_std(price_sum, price_sq_sum, count),
        "o_vol_sum": vol_sum.astype(np.float32),
        "o_vol_mean": finalize_mean(vol_sum, count),
        "o_vol_std": finalize_std(vol_sum, vol_sq_sum, count),
        "o_signed_vol": signed_vol.astype(np.float32),
        "o_action_vol": action_vol.astype(np.float32),
        "o_pressure": pressure.astype(np.float32),
        "o_new_vol": new_vol.astype(np.float32),
        "o_cancel_vol": cancel_vol.astype(np.float32),
        "o_buy_new": buy_new.astype(np.float32),
        "o_sell_new": sell_new.astype(np.float32),
        "o_buy_cancel": buy_cancel.astype(np.float32),
        "o_sell_cancel": sell_cancel.astype(np.float32),
        "o_log_vol_mean": finalize_mean(log_vol_sum, count),
        "o_last_side": latest_side,
        "o_last_action": latest_action,
    }
    features["o_signed_ratio"] = ratio0(signed_vol, vol_sum + 1.0)
    features["o_action_ratio"] = ratio0(action_vol, vol_sum + 1.0)
    features["o_pressure_ratio"] = ratio0(pressure, vol_sum + 1.0)
    features["o_cancel_ratio"] = ratio0(cancel_vol, vol_sum + 1.0)
    features["o_new_imb"] = ratio0(buy_new - sell_new, new_vol + 1.0)
    features["o_cancel_imb"] = ratio0(buy_cancel - sell_cancel, cancel_vol + 1.0)

    for w in FLOW_WINDOWS:
        ww = win[w]
        features[f"o_count_{w}"] = ww["count"].astype(np.float32)
        features[f"o_vol_sum_{w}"] = ww["vol_sum"].astype(np.float32)
        features[f"o_price_mean_{w}"] = finalize_mean(ww["price_sum"], ww["count"])
        features[f"o_price_std_{w}"] = finalize_std(ww["price_sum"], ww["price_sq_sum"], ww["count"])
        features[f"o_signed_vol_{w}"] = ww["signed_vol"].astype(np.float32)
        features[f"o_action_vol_{w}"] = ww["action_vol"].astype(np.float32)
        features[f"o_pressure_{w}"] = ww["pressure"].astype(np.float32)
        features[f"o_new_vol_{w}"] = ww["new_vol"].astype(np.float32)
        features[f"o_cancel_vol_{w}"] = ww["cancel_vol"].astype(np.float32)
        features[f"o_signed_ratio_{w}"] = ratio0(ww["signed_vol"], ww["vol_sum"] + 1.0)
        features[f"o_action_ratio_{w}"] = ratio0(ww["action_vol"], ww["vol_sum"] + 1.0)
        features[f"o_pressure_ratio_{w}"] = ratio0(ww["pressure"], ww["vol_sum"] + 1.0)
        features[f"o_cancel_ratio_{w}"] = ratio0(ww["cancel_vol"], ww["vol_sum"] + 1.0)
        features[f"o_new_imb_{w}"] = ratio0(ww["buy_new"] - ww["sell_new"], ww["new_vol"] + 1.0)
        features[f"o_cancel_imb_{w}"] = ratio0(ww["buy_cancel"] - ww["sell_cancel"], ww["cancel_vol"] + 1.0)
        features[f"o_vol_share_{w}"] = ratio0(ww["vol_sum"], vol_sum + 1.0)
        features[f"o_count_share_{w}"] = ratio0(ww["count"], count + 1.0)

    print(f"  {split} order done: rows={rows_seen:,}, features={len(features)}, time={time.time() - t0:.1f}s", flush=True)
    return features


def merge_features(base, extra):
    base.update(extra)
    del extra
    gc.collect()
    return base


def window_pairs(windows):
    ws = sorted(set(windows))
    pairs = []
    if len(ws) >= 2:
        pairs.extend((ws[i], ws[i + 1]) for i in range(len(ws) - 1))
        pairs.extend((w, ws[-1]) for w in ws[:-1])
    seen = set()
    out = []
    for pair in pairs:
        if pair not in seen and pair[0] < pair[1]:
            seen.add(pair)
            out.append(pair)
    return out


def add_window_delta_features(features):
    for short, long in window_pairs(MARKET_WINDOWS):
        features[f"m_imb1_delta_{short}_{long}"] = (
            features[f"m_imb1_mean_{short}"] - features[f"m_imb1_mean_{long}"]
        ).astype(np.float32)
        features[f"m_imb2_delta_{short}_{long}"] = (
            features[f"m_imb2_mean_{short}"] - features[f"m_imb2_mean_{long}"]
        ).astype(np.float32)
        features[f"m_spread_delta_{short}_{long}"] = (
            features[f"m_spread_mean_{short}"] - features[f"m_spread_mean_{long}"]
        ).astype(np.float32)
        features[f"m_mid_rel_delta_{short}_{long}"] = (
            features[f"m_mid_mean_rel_{short}"] - features[f"m_mid_mean_rel_{long}"]
        ).astype(np.float32)
        features[f"m_txv_ratio_{short}_{long}"] = ratio0(
            features[f"m_txv_sum_{short}"], features[f"m_txv_sum_{long}"] + 1.0
        )

    if USE_TRANSACTION and "t_signed_ratio" in features:
        for short, long in window_pairs(FLOW_WINDOWS):
            features[f"t_signed_delta_{short}_{long}"] = (
                features[f"t_signed_ratio_{short}"] - features[f"t_signed_ratio_{long}"]
            ).astype(np.float32)
            features[f"t_buy_trade_delta_{short}_{long}"] = (
                features[f"t_buy_trade_ratio_{short}"] - features[f"t_buy_trade_ratio_{long}"]
            ).astype(np.float32)
            features[f"t_vol_imb_delta_{short}_{long}"] = (
                features[f"t_vol_imb_{short}"] - features[f"t_vol_imb_{long}"]
            ).astype(np.float32)
            features[f"t_vol_ratio_{short}_{long}"] = ratio0(
                features[f"t_vol_sum_{short}"], features[f"t_vol_sum_{long}"] + 1.0
            )

    if USE_ORDER and "o_pressure_ratio" in features:
        for short, long in window_pairs(FLOW_WINDOWS):
            features[f"o_pressure_delta_{short}_{long}"] = (
                features[f"o_pressure_ratio_{short}"] - features[f"o_pressure_ratio_{long}"]
            ).astype(np.float32)
            features[f"o_cancel_delta_{short}_{long}"] = (
                features[f"o_cancel_ratio_{short}"] - features[f"o_cancel_ratio_{long}"]
            ).astype(np.float32)
            features[f"o_new_imb_delta_{short}_{long}"] = (
                features[f"o_new_imb_{short}"] - features[f"o_new_imb_{long}"]
            ).astype(np.float32)
            features[f"o_cancel_imb_delta_{short}_{long}"] = (
                features[f"o_cancel_imb_{short}"] - features[f"o_cancel_imb_{long}"]
            ).astype(np.float32)
            features[f"o_vol_ratio_{short}_{long}"] = ratio0(
                features[f"o_vol_sum_{short}"], features[f"o_vol_sum_{long}"] + 1.0
            )

    return features


def add_cross_features(features):
    m_mid = features["m_mid_last"]
    features["x_m_spread_imb_last"] = (features["m_spread_last"] * features["m_imb1_last"]).astype(np.float32)
    features["x_m_spread_imb_mean"] = (features["m_spread_mean"] * features["m_imb1_mean"]).astype(np.float32)

    if USE_TRANSACTION and "t_vwap" in features:
        features["x_t_vwap_vs_mid"] = (features["t_vwap"] / (m_mid + EPS) - 1.0).astype(np.float32)
        features["x_t_last_vs_mid"] = (features["t_price_last"] / (m_mid + EPS) - 1.0).astype(np.float32)
        features["x_t_signed_x_mimb"] = (features["t_signed_ratio"] * features["m_imb1_last"]).astype(np.float32)
        features["x_t_vol_to_m_txv"] = ratio0(features["t_vol_sum"], features["m_txv_sum"] + 1.0)
        for w in FLOW_WINDOWS:
            features[f"x_t_signed_x_mimb_{w}"] = (features[f"t_signed_ratio_{w}"] * features["m_imb1_last"]).astype(
                np.float32
            )
            features[f"x_t_vwap_vs_mid_{w}"] = (features[f"t_vwap_{w}"] / (m_mid + EPS) - 1.0).astype(np.float32)
            if w in MARKET_WINDOWS:
                features[f"x_t_vwap_vs_m_vwap_{w}"] = (
                    features[f"t_vwap_{w}"] / (features[f"m_vwap_{w}"] + EPS) - 1.0
                ).astype(np.float32)
                features[f"x_t_signed_x_mimb_mean_{w}"] = (
                    features[f"t_signed_ratio_{w}"] * features[f"m_imb1_mean_{w}"]
                ).astype(np.float32)

    if USE_ORDER and "o_pressure_ratio" in features:
        features["x_o_pressure_x_mimb"] = (features["o_pressure_ratio"] * features["m_imb1_last"]).astype(np.float32)
        features["x_o_new_minus_cancel_imb"] = (features["o_new_imb"] - features["o_cancel_imb"]).astype(np.float32)
        features["x_o_new_to_m_txv"] = ratio0(features["o_new_vol"], features["m_txv_sum"] + 1.0)
        features["x_o_cancel_to_m_txv"] = ratio0(features["o_cancel_vol"], features["m_txv_sum"] + 1.0)
        for w in FLOW_WINDOWS:
            features[f"x_o_pressure_x_mimb_{w}"] = (
                features[f"o_pressure_ratio_{w}"] * features["m_imb1_last"]
            ).astype(np.float32)
            features[f"x_o_new_minus_cancel_imb_{w}"] = (
                features[f"o_new_imb_{w}"] - features[f"o_cancel_imb_{w}"]
            ).astype(np.float32)
            if w in MARKET_WINDOWS:
                features[f"x_o_pressure_x_mimb_mean_{w}"] = (
                    features[f"o_pressure_ratio_{w}"] * features[f"m_imb1_mean_{w}"]
                ).astype(np.float32)

    if USE_TRANSACTION and USE_ORDER and "t_signed_ratio" in features and "o_pressure_ratio" in features:
        features["x_net_flow_pressure"] = (features["t_signed_ratio"] + features["o_pressure_ratio"]).astype(np.float32)
        features["x_t_minus_o_pressure"] = (features["t_signed_ratio"] - features["o_pressure_ratio"]).astype(np.float32)
        features["x_o_cancel_to_t_vol"] = ratio0(features["o_cancel_vol"], features["t_vol_sum"] + 1.0)
        features["x_o_new_to_t_vol"] = ratio0(features["o_new_vol"], features["t_vol_sum"] + 1.0)
        for w in FLOW_WINDOWS:
            features[f"x_net_flow_pressure_{w}"] = (
                features[f"t_signed_ratio_{w}"] + features[f"o_pressure_ratio_{w}"]
            ).astype(np.float32)
            features[f"x_t_minus_o_pressure_{w}"] = (
                features[f"t_signed_ratio_{w}"] - features[f"o_pressure_ratio_{w}"]
            ).astype(np.float32)
            features[f"x_o_cancel_to_t_vol_{w}"] = ratio0(features[f"o_cancel_vol_{w}"], features[f"t_vol_sum_{w}"] + 1.0)
            features[f"x_o_new_to_t_vol_{w}"] = ratio0(features[f"o_new_vol_{w}"], features[f"t_vol_sum_{w}"] + 1.0)

    return features


def build_features(split, sample_ids):
    print(f"\n=== build {split} ===", flush=True)
    features = aggregate_market(split, sample_ids)
    if USE_TRANSACTION:
        features = merge_features(features, aggregate_transaction(split, sample_ids))
    if USE_ORDER:
        features = merge_features(features, aggregate_order(split, sample_ids))
    features = add_window_delta_features(features)
    features = add_cross_features(features)
    print(f"  {split} final features={len(features)}", flush=True)
    return features


def make_matrix(features, feature_names, row_mask=None):
    if row_mask is None:
        n = next(iter(features.values())).shape[0]
        rows = slice(None)
    else:
        n = int(row_mask.sum())
        rows = row_mask

    x = np.empty((n, len(feature_names)), dtype=np.float32)
    for j, name in enumerate(feature_names):
        x[:, j] = features[name][rows]
    x[~np.isfinite(x)] = np.nan
    return x


def train_and_predict():
    train_ids, months, target = read_train_label()
    print(f"train samples={train_ids.size:,}", flush=True)

    train_features = build_features("train", train_ids)
    feature_names = list(train_features.keys())
    print(f"n_features={len(feature_names)}", flush=True)

    tr_mask = months < VALID_START_MONTH
    va_mask = ~tr_mask
    x_tr = make_matrix(train_features, feature_names, tr_mask)
    y_tr = target[tr_mask]
    x_va = make_matrix(train_features, feature_names, va_mask)
    y_va = target[va_mask]
    del train_features, months, target, tr_mask, va_mask
    gc.collect()
    print(f"X_train={x_tr.shape}, X_valid={x_va.shape}", flush=True)

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.025,
        "num_leaves": 48,
        "min_data_in_leaf": 500,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 10.0,
        "max_bin": 63,
        "force_col_wise": True,
        "num_threads": N_THREADS,
        "verbose": -1,
        "seed": SEED,
    }

    dtr = lgb.Dataset(x_tr, y_tr, params=params, free_raw_data=True)
    dva = lgb.Dataset(x_va, y_va, reference=dtr, params=params, free_raw_data=True)
    dtr.construct()
    dva.construct()
    del x_tr, y_tr
    gc.collect()

    t0 = time.time()
    model = lgb.train(
        params,
        dtr,
        num_boost_round=NUM_BOOST_ROUND,
        valid_sets=[dva],
        valid_names=["valid"],
        feval=lgb_cos_metric,
        callbacks=[
            lgb.early_stopping(EARLY_STOPPING, first_metric_only=False),
            lgb.log_evaluation(period=100),
        ],
    )
    pred_va = model.predict(x_va, num_iteration=model.best_iteration)
    print(
        f"valid_cos={cos_uncenter(pred_va, y_va):.6f}, "
        f"best_iter={model.best_iteration}, train_time={time.time() - t0:.1f}s",
        flush=True,
    )
    del x_va, y_va, pred_va, dtr, dva
    gc.collect()

    test_ids = read_submission_ids()
    print(f"test samples={test_ids.size:,}", flush=True)
    test_features = build_features("test", test_ids)
    missing = [name for name in feature_names if name not in test_features]
    if missing:
        raise ValueError(f"test missing features: {missing[:10]}")
    x_te = make_matrix(test_features, feature_names)
    del test_features
    gc.collect()

    pred = model.predict(x_te, num_iteration=model.best_iteration)
    del x_te, model
    gc.collect()
    return test_ids, pred


def write_submission(ids, pred):
    path = Path(OUT_CSV)
    with open(path, "w") as f:
        f.write("sample_id,prediction\n")
        for sid, p in zip(ids, pred):
            f.write(f"{int(sid)},{float(p)}\n")
    print(f"saved {path}, rows={len(ids):,}", flush=True)


def materialize_features():
    import pandas as pd
    out_dir = Path(os.environ.get("FEATURE_OUT", "features/micro_v3"))
    out_dir.mkdir(parents=True, exist_ok=True)
    train_ids, months, target = read_train_label()
    test_ids = read_submission_ids()
    for split, ids in [("train", train_ids), ("test", test_ids)]:
        feat = build_features(split, ids)
        columns = {"sample_id": ids}
        if split == "train":
            columns.update({"month": months, "target": target})
        columns.update(feat)
        frame = pd.DataFrame(columns)
        path = out_dir / f"{split}.parquet"
        frame.to_parquet(path, index=False)
        print(f"saved {path}: {frame.shape}", flush=True)
        del frame, feat
        gc.collect()


def main():
    require_runtime()
    print(
        "config: "
        f"market_windows={MARKET_WINDOWS}, flow_windows={FLOW_WINDOWS}, "
        f"use_transaction={USE_TRANSACTION}, use_order={USE_ORDER}, "
        f"batch_size={BATCH_SIZE}, valid_start={VALID_START_MONTH}, "
        f"num_boost_round={NUM_BOOST_ROUND}, n_threads={N_THREADS}",
        flush=True,
    )
    if os.environ.get("FEATURE_ONLY", "0") == "1":
        materialize_features()
        return
    ids, pred = train_and_predict()
    write_submission(ids, pred)


if __name__ == "__main__":
    main()
