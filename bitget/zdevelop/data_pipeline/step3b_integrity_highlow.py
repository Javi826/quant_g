# step3b_integrity_highlow.py
# -----------------------------
import logging
import os
import re

import pandas as pd

logger = logging.getLogger("pipeline.step3b")


# ---------------- UTILITIES ----------------

def _parse_timeframe_to_ms(tf: str) -> int:
    s = str(tf).strip().lower().replace('utc', '')
    m = re.match(r'^(\d+)([mhdwM])$', s)
    if not m:
        return 86400 * 1000
    n, u = int(m.group(1)), m.group(2)
    mapping = {'m': 60, 'h': 3600, 'd': 86400, 'w': 604800, 'M': 2592000}
    return n * mapping.get(u, 86400) * 1000


# ---------------- CHECK ----------------

def _check_symbol(df: pd.DataFrame, symbol: str, gran_ms: int) -> int:
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


# ---------------- RUN ----------------

def run(config: dict) -> bool:
    input_dir: str = config["highlow_dir"]
    tf_high: str   = config["timeframes_highlow"][0]
    gran_ms: int   = _parse_timeframe_to_ms(tf_high)

    files = [f for f in os.listdir(input_dir) if f.endswith(".parquet")] \
        if os.path.exists(input_dir) else []

    if not files:
        logger.warning(f"⚠ No parquet files found in {input_dir}")
        return True

    logger.info(f"🔍 High/Low integrity check — {len(files)} file(s)")

    total_violations = 0
    for filename in sorted(files):
        symbol   = os.path.splitext(filename)[0].rsplit("_", 1)[0]
        filepath = os.path.join(input_dir, filename)
        try:
            df = pd.read_parquet(filepath)
        except Exception as e:
            logger.warning(f"  ❌ Could not read {filename}: {e}")
            continue

        total_violations += _check_symbol(df, symbol, gran_ms)

    if total_violations == 0:
        logger.info("✅ High/Low integrity check passed — no violations found")
    else:
        logger.info(f"⚠ High/Low integrity check found {total_violations} violation(s)")

    return True


# ---------------- ENTRY POINT ----------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _base = os.path.dirname(os.path.abspath(__file__))
    _config = {
        "highlow_dir":        os.path.join(_base, "data", "03_highlow"),
        "timeframes_highlow": ["15m", "5m"],
    }
    run(_config)
