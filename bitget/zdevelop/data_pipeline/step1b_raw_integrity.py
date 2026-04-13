# step1b_raw_integrity.py
# -----------------------------
import logging
import os
import re

import pandas as pd

logger = logging.getLogger("pipeline.step1b_raw")

OHLC_COLS   = ["open", "high", "low", "close"]
VOLUME_COLS = ["volume_base", "volume_quote"]


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

def _check_symbol(df: pd.DataFrame, symbol: str) -> int:
    issues = 0

    # NaN in OHLCV
    for col in OHLC_COLS + VOLUME_COLS:
        if col in df.columns:
            n = df[col].isna().sum()
            if n > 0:
                issues += n
                logger.info(f"  ⚠ [{symbol}] NaN in '{col}': {n} rows")

    # Zero OHLC
    for col in OHLC_COLS:
        if col in df.columns:
            n = (df[col] == 0).sum()
            if n > 0:
                issues += n
                logger.info(f"  ⚠ [{symbol}] Zero values in '{col}': {n} rows")

    # Zero volumes
    for col in VOLUME_COLS:
        if col in df.columns:
            n = (df[col] == 0).sum()
            if n > 0:
                issues += n
                logger.info(f"  ⚠ [{symbol}] Zero volume in '{col}': {n} rows")

    # OHLC coherence
    if all(c in df.columns for c in OHLC_COLS):
        mask = (df["low"] > df["open"])  | (df["low"] > df["close"]) | \
               (df["high"] < df["open"]) | (df["high"] < df["close"])
        n = mask.sum()
        if n > 0:
            issues += n
            logger.info(f"  ⚠ [{symbol}] Incoherent OHLC: {n} rows")

    if issues == 0:
        logger.debug(f"  ✅ [{symbol}] No issues found")

    return issues


# ---------------- RUN ----------------

def run(config: dict) -> bool:
    input_dir: str = config["raw_dir"]
    timeframe: str = config["timeframe"]

    files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".parquet")
    ]) if os.path.exists(input_dir) else []

    if not files:
        logger.warning(f"⚠ No parquet files found in {input_dir}")
        return True

    logger.info(f"🔍 Raw integrity check — {len(files)} file(s)")

    total_issues = 0
    for filepath in files:
        symbol = os.path.splitext(os.path.basename(filepath))[0].rsplit("_", 1)[0]
        try:
            df = pd.read_parquet(filepath)
        except Exception as e:
            logger.warning(f"  ❌ Could not read {os.path.basename(filepath)}: {e}")
            continue

        total_issues += _check_symbol(df, symbol)

    if total_issues == 0:
        logger.info("✅ Raw integrity check passed — no issues found")
    else:
        logger.info(f"⚠ Raw integrity check found {total_issues} issue(s) — cleaning step will follow")

    return True


# ---------------- ENTRY POINT ----------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _config = {
        "raw_dir":   os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "01_raw"),
        "timeframe": "1Dutc",
    }
    run(_config)
