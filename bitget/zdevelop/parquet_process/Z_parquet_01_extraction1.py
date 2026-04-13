#Z_parquet_A0_extraction.py
# -----------------------------
import os
import time
import re
import requests
import pandas as pd
from datetime import datetime, timezone

# ---------------- CONFIG ----------------
BASE_URL               = "https://api.bitget.com"
PRODUCT_TYPE           = "usdt-futures"

TIMEFRAME              = "1Dutc"
LIMIT                  = 200
DATA_FOLDER            = "crypto_2026_short"
START_DATE             = "2025-01-01"
END_DATE               = None            # e.g. "2024-06-01" to limit download; None = today
REQUEST_TIMEOUT        = 20
SLEEP_BETWEEN_REQUESTS = 0.06
MAX_ITER_PER_SYMBOL    = 2000
MAX_RETRIES            = 3

SELECTED_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
# ----------------------------------------

MS_90_DAYS = 90 * 24 * 60 * 60 * 1000


# ---------------- UTILITIES ----------------

def sanitize_filename(name):
    safe = re.sub(r'[^\w\-_\. ]', '_', name)
    return safe.strip()


def parse_timeframe_to_ms(tf):
    s = str(tf).strip().lower().replace('utc', '')
    m = re.match(r'^(\d+)([mhdwM])$', s)
    if not m:
        raise ValueError(f"Unrecognized timeframe: '{tf}'")
    n, u = int(m.group(1)), m.group(2)
    mapping = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800, 'M': 2592000}
    if u not in mapping:
        raise ValueError(f"Unsupported timeframe unit: '{tf}'")
    return n * mapping[u] * 1000


def detect_gaps(df, gran_ms, symbol):
    """Detects and logs unexpected gaps in timestamp sequence."""
    if df.empty or len(df) < 2:
        return
    ts_ms = df["timestamp"].astype("int64") // 10**6
    diffs = ts_ms.diff().dropna()
    expected = gran_ms
    gaps = diffs[diffs > expected * 1.5]
    if not gaps.empty:
        for idx, gap_ms in gaps.items():
            gap_start = df["timestamp"].iloc[idx - 1]
            gap_end   = df["timestamp"].iloc[idx]
            gap_days  = gap_ms / (1000 * 86400)
            print(f"  ⚠ GAP detected in {symbol}: {gap_start} → {gap_end} ({gap_days:.1f} days missing)")


# ---------------- API WRAPPERS ----------------

def _http_get(url, params=None, timeout=REQUEST_TIMEOUT, max_retries=MAX_RETRIES):
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 502, 503, 504) or r.status_code >= 500:
                time.sleep(0.5 * attempt)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            time.sleep(0.5 * attempt)
    raise Exception("Max retries exceeded")


def get_futures_symbols_from_api(product_type=PRODUCT_TYPE):
    url = f"{BASE_URL}/api/v2/mix/market/contracts"
    try:
        r = _http_get(url, params={'productType': product_type})
        data = r.json().get('data') or []
        symbols = []
        for item in data:
            s = item.get('symbol') or item.get('contract') or item.get('symbolName')
            if s:
                symbols.append(str(s))
        return sorted(set(symbols))
    except Exception as e:
        print(f"⚠️ Error fetching symbols: {e}")
        return []


def _call_history_candles(symbol, granularity, limit=LIMIT, startTime=None, endTime=None):
    url = f"{BASE_URL}/api/v2/mix/market/history-candles"
    params = {
        "symbol":      symbol,
        "granularity": granularity,
        "limit":       limit,
        "productType": PRODUCT_TYPE,
    }
    if startTime is not None:
        params["startTime"] = str(int(startTime))
    if endTime is not None:
        params["endTime"] = str(int(endTime))
    try:
        r = _http_get(url, params=params)
        j = r.json()
        if isinstance(j, dict) and j.get("code") not in (None, "00000"):
            return []
        data = j.get("data") if isinstance(j, dict) else j
        return data if isinstance(data, list) else []
    except Exception as e:
        print(f"  ⚠ API error (symbol={symbol} start={startTime} end={endTime}): {e}")
        return []


def to_dataframe_from_api(data):
    if not data:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume_base", "volume_quote"])
    clean = []
    for row in data:
        if not row or len(row) < 7:
            continue
        try:
            clean.append([int(row[0]), row[1], row[2], row[3], row[4], row[5], row[6]])
        except Exception:
            continue
    df = pd.DataFrame(clean, columns=["timestamp", "open", "high", "low", "close", "volume_base", "volume_quote"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    return df.sort_values("timestamp").reset_index(drop=True)


# ---------------- DOWNLOAD FUNCTIONS ----------------

def find_earliest_available_timestamp(symbol, gran_ms, timeframe, max_iters=500):
    end = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    earliest_found = None
    prev_end = None
    for _ in range(max_iters):
        data = _call_history_candles(symbol, timeframe, limit=LIMIT, endTime=end)
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        if not data:
            return earliest_found
        timestamps = [int(item[0]) for item in data if item]
        if not timestamps:
            return earliest_found
        min_ts  = min(timestamps)
        new_end = min_ts - gran_ms
        if prev_end is not None and new_end == prev_end:
            return earliest_found or min_ts
        prev_end      = end
        earliest_found = min(earliest_found, min_ts) if earliest_found else min_ts
        if new_end < 0 or new_end >= end:
            return earliest_found
        end = new_end
    return earliest_found


def download_candles_from_start(symbol, start_ms, gran_ms, timeframe, max_iters=2000, end_ms=None):
    all_rows       = []
    now_ms         = end_ms or int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    current_start  = int(start_ms)
    window_ms      = min(gran_ms * LIMIT, MS_90_DAYS)
    no_progress    = 0
    prev_start     = None

    while current_start < now_ms and len(all_rows) // max(1, LIMIT) < max_iters:
        window_end = min(current_start + window_ms, now_ms)
        data       = _call_history_candles(symbol, timeframe, limit=LIMIT,
                                           startTime=current_start, endTime=window_end)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

        if not data:
            next_start = window_end + gran_ms
            no_progress = no_progress + 1 if next_start <= current_start else 0
            current_start = next_start
            if no_progress >= 3:
                break
            continue

        valid_rows, timestamps = [], []
        for row in data:
            try:
                timestamps.append(int(row[0]))
                valid_rows.append(row)
            except Exception:
                continue

        if not valid_rows:
            current_start = window_end + gran_ms
            continue

        all_rows.extend(valid_rows)
        max_ts        = max(timestamps)
        current_start = max_ts + gran_ms if max_ts > current_start else window_end + gran_ms
        no_progress   = no_progress + 1 if prev_start and current_start <= prev_start else 0
        prev_start    = current_start
        if no_progress >= 5:
            break

    df = to_dataframe_from_api(all_rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    return df


# ---------------- INCREMENTAL LOGIC ----------------

def load_existing_parquet(parquet_path):
    """Loads existing parquet. Returns DataFrame with tz-aware timestamps or empty DataFrame."""
    if not os.path.exists(parquet_path):
        return pd.DataFrame()
    try:
        df = pd.read_parquet(parquet_path)
        if "timestamp" in df.columns:
            if df["timestamp"].dt.tz is None:
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        return df
    except Exception as e:
        print(f"  ⚠ Could not load existing parquet ({parquet_path}): {e}")
        return pd.DataFrame()


def merge_and_deduplicate(df_old, df_new):
    """Concatenates old and new data, deduplicates by timestamp, sorts."""
    if df_old.empty:
        return df_new
    if df_new.empty:
        return df_old
    df = pd.concat([df_old, df_new], ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"], keep="first")
    return df.sort_values("timestamp").reset_index(drop=True)


def get_resume_start_ms(df_existing, gran_ms):
    """Returns the start_ms for incremental download: last timestamp + 1 candle."""
    if df_existing.empty:
        return None
    last_ts = df_existing["timestamp"].max()
    return int(last_ts.timestamp() * 1000) + gran_ms


# ---------------- SAVE FUNCTIONS ----------------

def save_parquet(df, path):
    df_save = df.copy()
    if df_save["timestamp"].dt.tz is not None:
        df_save["timestamp"] = df_save["timestamp"].dt.tz_localize(None)
    df_save.to_parquet(path, index=False)


def save_csv(df, path):
    """Regenerates CSV from final merged DataFrame (parquet is source of truth)."""
    df_save = df.copy()
    if df_save["timestamp"].dt.tz is not None:
        df_save["timestamp"] = df_save["timestamp"].dt.tz_localize(None)
    df_save.to_csv(path, index=False)


# ---------------- SYMBOL PROCESSOR ----------------

def process_symbol(sym, start_ms_requested, gran_ms, timeframe, data_folder=DATA_FOLDER, end_ms=None):
    parquet_path = os.path.join(data_folder, sanitize_filename(f"{sym}_{timeframe}.parquet"))
    csv_path     = os.path.join(data_folder, sanitize_filename(f"{sym}_{timeframe}.csv"))

    # --- Load existing data ---
    df_existing = load_existing_parquet(parquet_path)

    if not df_existing.empty:
        resume_ms = get_resume_start_ms(df_existing, gran_ms)
        last_date = df_existing["timestamp"].max().strftime('%Y-%m-%d %H:%M:%S')
        print(f"  📂 Existing data found ({len(df_existing)} candles, last: {last_date}). Resuming from there.")
        df_new = download_candles_from_start(sym, resume_ms, gran_ms, timeframe, end_ms=end_ms)
    else:
        print(f"  🆕 No existing data. Downloading from {START_DATE}.")
        df_new = download_candles_from_start(sym, start_ms_requested, gran_ms, timeframe, end_ms=end_ms)

        # If no data from START_DATE, search for earliest available
        if not df_new.empty:
            start_dt  = pd.to_datetime(START_DATE).tz_localize("UTC")
            first_ts  = df_new["timestamp"].min()
            if first_ts > start_dt:
                print(f"  ⚠ No data from {START_DATE}. Searching for first available candle...")
                earliest_ts = find_earliest_available_timestamp(sym, gran_ms, timeframe)
                if earliest_ts is None:
                    print(f"  ❌ No candles found for {sym}. Skipping.")
                    return
                earliest_dt = datetime.fromtimestamp(earliest_ts / 1000, tz=timezone.utc)
                print(f"  ✅ First candle: {earliest_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC. Downloading from there.")
                df_new = download_candles_from_start(sym, earliest_ts, gran_ms, timeframe, end_ms=end_ms)
                if df_new.empty:
                    print(f"  ❌ Download failed for {sym}. Skipping.")
                    return

    if df_new.empty and df_existing.empty:
        print(f"  ❌ No data available for {sym}. Skipping.")
        return

    new_candles = len(df_new)

    # --- Merge ---
    df_final = merge_and_deduplicate(df_existing, df_new)

    # --- Cast numeric columns ---
    for col in ["open", "high", "low", "close", "volume_base", "volume_quote"]:
        df_final[col] = pd.to_numeric(df_final[col], errors="coerce")

    # --- Gap detection ---
    detect_gaps(df_final, gran_ms, sym)

    # --- Save parquet (source of truth) ---
    os.makedirs(data_folder, exist_ok=True)
    save_parquet(df_final, parquet_path)

    # --- Regenerate CSV from final parquet ---
    save_csv(df_final, csv_path)

    print(f"  💾 Saved {len(df_final)} candles total (+{new_candles} new) → {os.path.basename(parquet_path)}")


# ---------------- MAIN ----------------

def process_all_symbols(start_date_str=START_DATE, timeframe=TIMEFRAME, end_date_str=END_DATE):
    try:
        start_dt = pd.to_datetime(start_date_str)
        start_dt = start_dt.tz_localize("UTC") if start_dt.tzinfo is None else start_dt.tz_convert("UTC")
    except Exception:
        start_dt = pd.to_datetime(start_date_str, utc=True)

    end_ms = None
    if end_date_str:
        try:
            end_dt = pd.to_datetime(end_date_str)
            end_dt = end_dt.tz_localize("UTC") if end_dt.tzinfo is None else end_dt.tz_convert("UTC")
            end_ms = int(end_dt.timestamp() * 1000)
            print(f"📅 END_DATE set to {end_dt.isoformat()} — download will stop here.")
        except Exception as e:
            print(f"⚠️ Could not parse END_DATE '{end_date_str}': {e}. Ignoring.")

    start_ms_requested = int(start_dt.timestamp() * 1000)
    gran_ms = parse_timeframe_to_ms(timeframe)
    os.makedirs(DATA_FOLDER, exist_ok=True)

    symbols = get_futures_symbols_from_api(PRODUCT_TYPE)
    if not symbols:
        print("⚠️ No symbols retrieved. Aborting.")
        return

    if SELECTED_SYMBOLS:
        symbols = [s for s in symbols if s in SELECTED_SYMBOLS]
        if not symbols:
            print("⚠️ No symbols matched from SELECTED_SYMBOLS. Aborting.")
            return
        print(f"📋 Selected symbols: {symbols}")

    print(f"🔁 Downloading candles from {start_dt.isoformat()} for {len(symbols)} symbol(s).\n")

    for i, sym in enumerate(symbols, start=1):
        print(f"[{i}/{len(symbols)}] Processing {sym} ...")
        process_symbol(sym, start_ms_requested, gran_ms, timeframe, DATA_FOLDER, end_ms=end_ms)

    print("\n🏁 Process complete.")


# ---------------- ENTRY POINT ----------------

if __name__ == "__main__":
    t0 = time.time()
    process_all_symbols(START_DATE, TIMEFRAME)
    elapsed = time.time() - t0
    m, s = divmod(elapsed, 60)
    print(f"\n⏱ Total execution time: {int(m)}m {int(s)}s")