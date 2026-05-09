# data_pipeline/integrity.py
# =============================================================================
# Data Integrity — validates OHLCV data at different pipeline stages.
#
# run_raw()     — validates raw extracted data (diagnostic only, never aborts)
# run_clean()   — validates cleaned data (aborts pipeline if errors found)
# run_highlow() — validates high_time/low_time are within bar interval (diagnostic only)
# =============================================================================
import logging
import os
import re

import pandas as pd

logger = logging.getLogger("pipeline.integrity")

# =============================================================================
# CONSTANTS
# =============================================================================
OHLC_COLS   = ["open", "high", "low", "close"]
VOLUME_COLS = ["volume_base", "volume_quote"]


# =============================================================================
# UTILITIES
# =============================================================================

def _parse_timeframe_to_ms(tf: str) -> int:
    s = str(tf).strip().lower().replace('utc', '')
    m = re.match(r'^(\d+)([mhdwM])$', s)
    if not m:
        return 86400 * 1000
    n, u = int(m.group(1)), m.group(2)
    mapping = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800, 'M': 2592000}
    return n * mapping.get(u, 86400) * 1000

def _list_parquet_files(folder: str, timeframe: str = "", selected_symbols: list | None = None) -> list[str]:
    if not os.path.exists(folder):
        return []
    return sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.endswith(".parquet")
        and (not timeframe or f.endswith(f"_{timeframe}.parquet"))
        and (not selected_symbols or any(f.startswith(s) for s in selected_symbols))
    ])


def _symbol_from_path(filepath: str) -> str:
    return os.path.splitext(os.path.basename(filepath))[0].rsplit("_", 1)[0]


# =============================================================================
# OHLCV CHECKS — shared by run_raw and run_clean
# =============================================================================

def _check_ohlcv(df: pd.DataFrame, symbol: str, critical: bool) -> int:
    """
    Validates OHLCV quality.
    critical=True  → uses ❌ prefix (run_clean)
    critical=False → uses ⚠ prefix (run_raw)
    """
    issues = 0
    prefix = "❌" if critical else "⚠"

    # NaN in OHLCV
    for col in OHLC_COLS + VOLUME_COLS:
        if col in df.columns:
            n = df[col].isna().sum()
            if n > 0:
                issues += n
                logger.info(f"  {prefix} [{symbol}] NaN in '{col}': {n} rows")

    # Zero OHLC
    for col in OHLC_COLS:
        if col in df.columns:
            n = (df[col] == 0).sum()
            if n > 0:
                issues += n
                logger.info(f"  {prefix} [{symbol}] Zero values in '{col}': {n} rows")

    # Zero volumes
    for col in VOLUME_COLS:
        if col in df.columns:
            n = (df[col] == 0).sum()
            if n > 0:
                issues += n
                logger.info(f"  {prefix} [{symbol}] Zero volume in '{col}': {n} rows")

    # OHLC coherence
    if all(c in df.columns for c in OHLC_COLS):
        mask = (df["low"] > df["open"])  | (df["low"] > df["close"]) | \
               (df["high"] < df["open"]) | (df["high"] < df["close"])
        n = mask.sum()
        if n > 0:
            issues += n
            logger.info(f"  {prefix} [{symbol}] Incoherent OHLC: {n} rows")

    if issues == 0:
        logger.debug(f"  ✅ [{symbol}] Integrity passed")

    return issues


# =============================================================================
# HIGH/LOW CHECK
# =============================================================================

def _check_highlow(df: pd.DataFrame, symbol: str, gran_ms: int) -> int:
    violations = 0

    if "high_time" not in df.columns or "low_time" not in df.columns:
        logger.warning(f"  ⚠ [{symbol}] Missing high_time/low_time columns")
        return 1

    if "timestamp" not in df.columns:
        df = df.reset_index()
        if "timestamp" not in df.columns:
            logger.warning(f"  ⚠ [{symbol}] No timestamp column found")
            return 1

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["high_time"] = pd.to_datetime(df["high_time"])
    df["low_time"]  = pd.to_datetime(df["low_time"])
    bar_end         = df["timestamp"] + pd.Timedelta(milliseconds=gran_ms)

    bad_high = df[
        df["high_time"].notna() &
        ((df["high_time"] < df["timestamp"]) | (df["high_time"] >= bar_end))
    ]
    if not bad_high.empty:
        violations += len(bad_high)
        logger.info(f"  ⚠ [{symbol}] high_time out of bar range: {len(bad_high)} rows")
        logger.debug(f"\n{bad_high[['timestamp','high_time']].head(5)}")

    bad_low = df[
        df["low_time"].notna() &
        ((df["low_time"] < df["timestamp"]) | (df["low_time"] >= bar_end))
    ]
    if not bad_low.empty:
        violations += len(bad_low)
        logger.info(f"  ⚠ [{symbol}] low_time out of bar range: {len(bad_low)} rows")
        logger.debug(f"\n{bad_low[['timestamp','low_time']].head(5)}")

    if violations == 0:
        logger.debug(f"  ✅ [{symbol}] High/Low integrity passed")

    return violations


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

def run_raw(config: dict) -> bool:
    """Validates raw extracted data — diagnostic only, never aborts pipeline."""
    input_dir: str = config["raw_dir"]
    timeframe: str = config.get("timeframe", "")
    
    # run_raw
    selected_symbols = config.get("selected_symbols") or []
    files = _list_parquet_files(input_dir, timeframe, selected_symbols)

    if not files:
        logger.warning(f"⚠ No parquet files found in {input_dir}")
        return True

    logger.info(f"🔍 Raw integrity check — {len(files)} file(s)")
    total = 0
    for filepath in files:
        try:
            df = pd.read_parquet(filepath)
        except Exception as e:
            logger.warning(f"  ❌ Could not read {os.path.basename(filepath)}: {e}")
            continue
        total += _check_ohlcv(df, _symbol_from_path(filepath), critical=False)
        
        # Gap detection
        if "timestamp" in df.columns:
            tf = os.path.splitext(os.path.basename(filepath))[0].rsplit("_", 1)[-1]
            gran_ms = _parse_timeframe_to_ms(tf)
            ts_ms   = pd.to_datetime(df["timestamp"]).astype("int64") // 10**6
            diffs   = ts_ms.diff().dropna()
            gaps    = diffs[diffs > gran_ms * 1.5]
            for idx, gap_ms in gaps.items():
                gap_start = pd.to_datetime(df["timestamp"].iloc[idx - 1])
                gap_end   = pd.to_datetime(df["timestamp"].iloc[idx])
                gap_days  = gap_ms / (1000 * 86400)
                total    += 1
                logger.info(f"  ⚠ [{_symbol_from_path(filepath)}] GAP: {gap_start} → {gap_end} ({gap_days:.1f} days)")

    if total == 0:
        logger.info("✅ Raw integrity check passed — no issues found")
    else:
        logger.info(f"⚠ Raw integrity check found {total} issue(s) — cleaning step will follow")

    return True


def run_clean(config: dict) -> bool:
    """Validates cleaned data — aborts pipeline if errors found."""
    input_dir: str = config["clean_dir"]
    timeframe: str = config.get("timeframe", "")
    selected_symbols = config.get("selected_symbols") or []
    files = _list_parquet_files(input_dir, timeframe, selected_symbols)

    if not files:
        logger.warning(f"⚠ No parquet files found in {input_dir}")
        return False

    logger.info(f"🔍 Clean integrity check — {len(files)} file(s)")
    total = 0
    for filepath in files:
        try:
            df = pd.read_parquet(filepath)
        except Exception as e:
            logger.warning(f"  ❌ Could not read {os.path.basename(filepath)}: {e}")
            total += 1
            continue
        total += _check_ohlcv(df, _symbol_from_path(filepath), critical=True)

    if total == 0:
        logger.info("✅ Clean integrity check passed")
        return True

    logger.info(f"❌ Clean integrity check FAILED — {total} critical error(s). Pipeline aborted.")
    return False


def run_highlow(config: dict) -> bool:
    """Validates high_time/low_time are within bar interval — diagnostic only, never aborts."""
    input_dir: str = config["highlow_dir"]
    # run_highlow
    selected_symbols = config.get("selected_symbols") or []
    files = _list_parquet_files(input_dir, selected_symbols=selected_symbols)

    if not files:
        logger.warning(f"⚠ No parquet files found in {input_dir}")
        return True

    logger.info(f"🔍 High/Low integrity check — {len(files)} file(s)")
    total = 0
    for filepath in files:
        tf      = os.path.splitext(os.path.basename(filepath))[0].rsplit("_", 1)[-1]
        gran_ms = _parse_timeframe_to_ms(tf)
        try:
            df = pd.read_parquet(filepath)
        except Exception as e:
            logger.warning(f"  ❌ Could not read {os.path.basename(filepath)}: {e}")
            continue
        total += _check_highlow(df, _symbol_from_path(filepath), gran_ms)

    if total == 0:
        logger.info("✅ High/Low integrity check passed — no violations found")
    else:
        logger.info(f"⚠ High/Low integrity check found {total} violation(s)")

    return True

def run_coverage(config: dict, tolerance_days: int = 30) -> bool:
    """
    Validates that all timeframes for each symbol start at roughly the same date.
    Uses 1Dutc as reference. Alerts if any timeframe starts more than tolerance_days later.
    Diagnostic only — never aborts pipeline.
    """
    input_dir: str = config["raw_dir"]
    # run_coverage
    selected_symbols = config.get("selected_symbols") or []
    files = _list_parquet_files(input_dir, selected_symbols=selected_symbols)

    if not files:
        logger.warning(f"⚠ No parquet files found in {input_dir}")
        return True

    # Group files by symbol
    symbol_files: dict[str, dict[str, str]] = {}
    for filepath in files:
        symbol   = _symbol_from_path(filepath)
        filename = os.path.basename(filepath)
        tf       = os.path.splitext(filename)[0].replace(f"{symbol}_", "", 1)
        symbol_files.setdefault(symbol, {})[tf] = filepath

    logger.info(f"🔍 Coverage integrity check — {len(symbol_files)} symbol(s)")

    issues = 0
    for symbol, tf_files in sorted(symbol_files.items()):
        ref_path = tf_files.get("1Dutc")
        if not ref_path:
            logger.debug(f"  ⚠ [{symbol}] No 1Dutc reference — skipping coverage check")
            continue

        try:
            df_ref   = pd.read_parquet(ref_path)
            ref_min  = pd.to_datetime(df_ref["timestamp"]).min()
        except Exception as e:
            logger.warning(f"  ⚠ [{symbol}] Could not read 1Dutc reference: {e}")
            continue

        for tf, filepath in sorted(tf_files.items()):
            if tf == "1Dutc":
                continue
            try:
                df      = pd.read_parquet(filepath)
                tf_min  = pd.to_datetime(df["timestamp"]).min()
                diff_days = (tf_min - ref_min).days
                if diff_days > tolerance_days:
                    issues += 1
                    logger.info(
                        f"  ⚠ [{symbol}] {tf} starts {diff_days}d after 1Dutc "
                        f"({tf_min.date()} vs {ref_min.date()})"
                    )
                else:
                    logger.debug(f"  ✅ [{symbol}] {tf} coverage OK (diff: {diff_days}d)")
            except Exception as e:
                logger.warning(f"  ⚠ [{symbol}] Could not read {tf}: {e}")

    if issues == 0:
        logger.info("✅ Coverage integrity check passed — all timeframes aligned")
    else:
        logger.info(f"⚠ Coverage integrity check found {issues} misaligned timeframe(s)")
        
# =============================================================================
#     # DEBUG — remove before production
#     print(f"\n{'Symbol':<14} {'Coverage by timeframe'}")
#     print("-" * 80)
#     for symbol, tf_files in sorted(symbol_files.items()):
#         ref_path = tf_files.get("1Dutc")
#         if not ref_path:
#             continue
#         try:
#             df_ref  = pd.read_parquet(ref_path)
#             ref_min = pd.to_datetime(df_ref["timestamp"]).min()
#         except Exception:
#             continue
#         parts = [f"1Dutc: {ref_min.date()} (ref)"]
#         for tf, filepath in sorted(tf_files.items()):
#             if tf == "1Dutc":
#                 continue
#             try:
#                 df        = pd.read_parquet(filepath)
#                 tf_min    = pd.to_datetime(df["timestamp"]).min()
#                 diff_days = (tf_min - ref_min).days
#                 status    = "✅" if diff_days <= tolerance_days else f"⚠(+{diff_days}d)"
#                 parts.append(f"{tf}: {tf_min.date()} {status}")
#             except Exception:
#                 parts.append(f"{tf}: ERROR ❌")
#         print(f"  {symbol:<14} {' | '.join(parts)}")
#     print()
# =============================================================================
    
    return True
# =============================================================================
# 
# def debug_highlow(config: dict, symbol: str, n_rows: int = 10) -> None:
#     files = _list_parquet_files(config["highlow_dir"])
#     matches = [f for f in files if _symbol_from_path(f) == symbol]
#     if not matches:
#         print(f"No files found for symbol: {symbol}")
#         return
#     for filepath in matches:
#         tf      = os.path.splitext(os.path.basename(filepath))[0].rsplit("_", 1)[-1]
#         gran_ms = _parse_timeframe_to_ms(tf)
#         df      = pd.read_parquet(filepath)
#         if "timestamp" not in df.columns:
#             df = df.reset_index()
#         df["timestamp"] = pd.to_datetime(df["timestamp"])
#         df["high_time"] = pd.to_datetime(df["high_time"])
#         df["low_time"]  = pd.to_datetime(df["low_time"])
#         df["bar_end"]   = df["timestamp"] + pd.Timedelta(milliseconds=gran_ms)
#         print(f"\n── {symbol} [{tf}] — first {n_rows} rows ──")
#         print(df[["timestamp", "bar_end", "high_time", "low_time"]].head(n_rows).to_string(index=False))
# =============================================================================
# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _base = os.path.dirname(os.path.abspath(__file__))
    _config = {
        "raw_dir":            os.path.join(_base, "data", "01_raw"),
        "clean_dir":          os.path.join(_base, "data", "02_clean"),
        "highlow_dir":        os.path.join(_base, "data", "03_highlow"),
        "timeframe":          "1Dutc",
        "timeframes_highlow": ["4H", "1H"],
    }
    run_raw(_config)
    run_clean(_config)
    run_highlow(_config)
    run_coverage(_config)
    #debug_highlow(_config, symbol="BTCUSDT")
