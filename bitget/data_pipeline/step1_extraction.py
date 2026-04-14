# step1_extraction.py
# =============================================================================
# Step 1 — Extraction — downloads OHLCV candles from Bitget API.
# Supports incremental download: resumes from last saved timestamp.
# =============================================================================
import logging
import os
import re
import sys
import time
from datetime import datetime, timezone

import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "broker_api")))
from api_client import (
    _call_history_candles,
    to_dataframe_from_api,
    get_futures_symbols_from_api,
)

logger = logging.getLogger("pipeline.step1")

# =============================================================================
# CONSTANTS
# =============================================================================
LIMIT                  = 200
SLEEP_BETWEEN_REQUESTS = 0.06
MS_90_DAYS             = 90 * 24 * 60 * 60 * 1000

# =============================================================================
# UTILITIES
# =============================================================================

def sanitize_filename(name: str) -> str:
    return re.sub(r'[^\w\-_\. ]', '_', name).strip()


def parse_timeframe_to_ms(tf: str) -> int:
    s = str(tf).strip().lower().replace('utc', '')
    m = re.match(r'^(\d+)([mhdwM])$', s)
    if not m:
        raise ValueError(f"Unrecognized timeframe: '{tf}'")
    n, u = int(m.group(1)), m.group(2)
    mapping = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800, 'M': 2592000}
    if u not in mapping:
        raise ValueError(f"Unsupported timeframe unit: '{tf}'")
    return n * mapping[u] * 1000


def detect_gaps(df: pd.DataFrame, gran_ms: int, symbol: str) -> None:
    if df.empty or len(df) < 2:
        return
    ts_ms = df["timestamp"].astype("int64") // 10**6
    diffs = ts_ms.diff().dropna()
    gaps  = diffs[diffs > gran_ms * 1.5]
    for idx, gap_ms in gaps.items():
        gap_start = df["timestamp"].iloc[idx - 1]
        gap_end   = df["timestamp"].iloc[idx]
        gap_days  = gap_ms / (1000 * 86400)
        logger.warning(f"  ⚠ GAP in {symbol}: {gap_start} → {gap_end} ({gap_days:.1f} days missing)")


def validate_append_border(df_old: pd.DataFrame, df_new: pd.DataFrame, gran_ms: int, symbol: str) -> None:
    if df_old.empty or df_new.empty:
        return
    last_old  = df_old["timestamp"].max()
    first_new = df_new["timestamp"].min()
    expected  = last_old + pd.Timedelta(milliseconds=gran_ms)
    diff_ms   = int((first_new - last_old).total_seconds() * 1000)

    if first_new == expected:
        logger.debug(f"  ✅ Border OK: {last_old} → {first_new}")
    elif first_new < expected:
        overlap = int((expected - first_new).total_seconds() * 1000) // gran_ms
        logger.warning(f"  ⚠ Border OVERLAP in {symbol}: {overlap} candle(s) at junction ({last_old} / {first_new})")
    else:
        gap_candles = diff_ms // gran_ms - 1
        logger.warning(f"  ⚠ Border GAP in {symbol}: {gap_candles} candle(s) missing at junction ({last_old} → {first_new})")

# =============================================================================
# DOWNLOAD
# =============================================================================

def find_earliest_available_timestamp(symbol: str, gran_ms: int, timeframe: str, max_iters: int = 500) -> int | None:
    end            = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    earliest_found = None
    prev_end       = None
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
        prev_end       = end
        earliest_found = min(earliest_found, min_ts) if earliest_found else min_ts
        if new_end < 0 or new_end >= end:
            return earliest_found
        end = new_end
    return earliest_found


def download_candles_from_start(
    symbol: str,
    start_ms: int,
    gran_ms: int,
    timeframe: str,
    end_ms: int | None = None,
    max_iters: int = 2000,
) -> pd.DataFrame:
    all_rows      = []
    now_ms        = end_ms or int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    current_start = int(start_ms)
    window_ms     = min(gran_ms * LIMIT, MS_90_DAYS)
    no_progress   = 0
    prev_start    = None

    while current_start < now_ms and len(all_rows) // max(1, LIMIT) < max_iters:
        window_end = min(current_start + window_ms, now_ms)
        data       = _call_history_candles(symbol, timeframe, limit=LIMIT,
                                           startTime=current_start, endTime=window_end)
        time.sleep(SLEEP_BETWEEN_REQUESTS)

        if not data:
            next_start  = window_end + gran_ms
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

# =============================================================================
# INCREMENTAL LOGIC
# =============================================================================

def _load_existing_parquet(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_parquet(path)
        if "timestamp" in df.columns and df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
        return df
    except Exception as e:
        logger.warning(f"  ⚠ Could not load existing parquet ({os.path.basename(path)}): {e}")
        return pd.DataFrame()


def _merge_and_deduplicate(df_old: pd.DataFrame, df_new: pd.DataFrame) -> pd.DataFrame:
    if df_old.empty:
        return df_new
    if df_new.empty:
        return df_old
    df = pd.concat([df_old, df_new], ignore_index=True)
    return df.drop_duplicates(subset=["timestamp"], keep="first").sort_values("timestamp").reset_index(drop=True)


def _get_resume_start_ms(df: pd.DataFrame, gran_ms: int) -> int:
    return int(df["timestamp"].max().timestamp() * 1000) + gran_ms


def _save_parquet(df: pd.DataFrame, path: str) -> None:
    df_save = df.copy()
    if df_save["timestamp"].dt.tz is not None:
        df_save["timestamp"] = df_save["timestamp"].dt.tz_localize(None)
    df_save.to_parquet(path, index=False)


def _save_csv(df: pd.DataFrame, path: str) -> None:
    df_save = df.copy()
    if df_save["timestamp"].dt.tz is not None:
        df_save["timestamp"] = df_save["timestamp"].dt.tz_localize(None)
    df_save.to_csv(path, index=False)

# =============================================================================
# SYMBOL PROCESSOR
# =============================================================================

def _process_symbol(
    sym: str,
    start_ms: int,
    gran_ms: int,
    timeframe: str,
    output_dir: str,
    end_ms: int | None,
    start_date: str,
) -> None:
    parquet_path = os.path.join(output_dir, sanitize_filename(f"{sym}_{timeframe}.parquet"))
    csv_path     = os.path.join(output_dir, sanitize_filename(f"{sym}_{timeframe}.csv"))

    df_existing = _load_existing_parquet(parquet_path)

    if not df_existing.empty:
        resume_ms = _get_resume_start_ms(df_existing, gran_ms)
        last_date = df_existing["timestamp"].max().strftime('%Y-%m-%d %H:%M:%S')
        logger.info(f"  📂 Existing data ({len(df_existing)} candles, last: {last_date}). Resuming.")
        df_new = download_candles_from_start(sym, resume_ms, gran_ms, timeframe, end_ms=end_ms)
    else:
        logger.info(f"  🆕 No existing data. Downloading from {start_date}.")
        df_new = download_candles_from_start(sym, start_ms, gran_ms, timeframe, end_ms=end_ms)

        if not df_new.empty:
            start_dt = pd.to_datetime(start_date).tz_localize("UTC")
            if df_new["timestamp"].min() > start_dt:
                logger.info(f"  ⚠ No data from {start_date}. Searching for first available candle...")
                earliest_ts = find_earliest_available_timestamp(sym, gran_ms, timeframe)
                if earliest_ts is None:
                    logger.info(f"  ❌ No candles found for {sym}. Skipping.")
                    return
                earliest_dt = datetime.fromtimestamp(earliest_ts / 1000, tz=timezone.utc)
                logger.info(f"  ✅ First candle: {earliest_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC.")
                df_new = download_candles_from_start(sym, earliest_ts, gran_ms, timeframe, end_ms=end_ms)
                if df_new.empty:
                    logger.info(f"  ❌ Download failed for {sym}. Skipping.")
                    return

    if df_new.empty and df_existing.empty:
        logger.info(f"  ❌ No data available for {sym}. Skipping.")
        return

    validate_append_border(df_existing, df_new, gran_ms, sym)

    df_final = _merge_and_deduplicate(df_existing, df_new)

    for col in ["open", "high", "low", "close", "volume_base", "volume_quote"]:
        df_final[col] = pd.to_numeric(df_final[col], errors="coerce")

    detect_gaps(df_final, gran_ms, sym)

    _save_parquet(df_final, parquet_path)
    _save_csv(df_final, csv_path)

    new_candles = len(df_new) if not df_new.empty else 0
    logger.info(f"  💾 {len(df_final)} candles total (+{new_candles} new) → {os.path.basename(parquet_path)}")

# =============================================================================
# RUN
# =============================================================================

def run(config: dict) -> bool:
    start_date       = config["start_date"]
    end_date         = config.get("end_date")
    timeframe        = config["timeframe"]
    selected_symbols = config.get("selected_symbols")
    output_dir: str  = config["raw_dir"]

    try:
        start_dt = pd.to_datetime(start_date)
        start_dt = start_dt.tz_localize("UTC") if start_dt.tzinfo is None else start_dt.tz_convert("UTC")
    except Exception:
        start_dt = pd.to_datetime(start_date, utc=True)

    end_ms = None
    if end_date:
        try:
            end_dt = pd.to_datetime(end_date)
            end_dt = end_dt.tz_localize("UTC") if end_dt.tzinfo is None else end_dt.tz_convert("UTC")
            end_ms = int(end_dt.timestamp() * 1000)
            logger.info(f"📅 END_DATE: {end_dt.isoformat()}")
        except Exception as e:
            logger.warning(f"⚠️ Could not parse END_DATE '{end_date}': {e}. Ignoring.")

    start_ms = int(start_dt.timestamp() * 1000)
    gran_ms  = parse_timeframe_to_ms(timeframe)
    os.makedirs(output_dir, exist_ok=True)

    symbols = get_futures_symbols_from_api()
    if not symbols:
        logger.warning("⚠️ No symbols retrieved. Aborting.")
        return False

    if selected_symbols:
        symbols = [s for s in symbols if s in selected_symbols]
        if not symbols:
            logger.warning("⚠️ No symbols matched. Aborting.")
            return False
        logger.info(f"📋 Symbols: {symbols}")

    logger.info(f"🔁 Downloading [{timeframe}] from {start_dt.isoformat()} for {len(symbols)} symbol(s).")

    for i, sym in enumerate(symbols, start=1):
        logger.info(f"\n[{i}/{len(symbols)}] {sym}")
        _process_symbol(sym, start_ms, gran_ms, timeframe, output_dir, end_ms, start_date)

    return True

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _config = {
        "start_date":       "2025-01-01",
        "end_date":         None,
        "timeframe":        "1Dutc",
        "selected_symbols": ["BTCUSDT", "ETHUSDT"],
        "raw_dir":          os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "01_raw"),
    }
    t0 = time.time()
    run(_config)
    m, s = divmod(time.time() - t0, 60)
    logger.info(f"\n⏱ Total: {int(m)}m {int(s)}s")
