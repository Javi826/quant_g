# step1b_clean_integrity.py
# -----------------------------
import logging
import os

import pandas as pd

logger = logging.getLogger("pipeline.step1b_clean")

OHLC_COLS   = ["open", "high", "low", "close"]
VOLUME_COLS = ["volume_base", "volume_quote"]


# ---------------- CHECK ----------------

def _check_symbol(df: pd.DataFrame, symbol: str) -> int:
    errors = 0

    # NaN or zero in OHLC
    for col in OHLC_COLS:
        if col in df.columns:
            n_nan  = df[col].isna().sum()
            n_zero = (df[col] == 0).sum()
            if n_nan > 0:
                errors += n_nan
                logger.info(f"  ❌ [{symbol}] NaN in '{col}': {n_nan} rows after cleaning")
            if n_zero > 0:
                errors += n_zero
                logger.info(f"  ❌ [{symbol}] Zero in '{col}': {n_zero} rows after cleaning")

    # NaN in volumes
    for col in VOLUME_COLS:
        if col in df.columns:
            n = df[col].isna().sum()
            if n > 0:
                errors += n
                logger.info(f"  ❌ [{symbol}] NaN in '{col}': {n} rows after cleaning")

    # OHLC coherence
    if all(c in df.columns for c in OHLC_COLS):
        mask = (df["low"] > df["open"])  | (df["low"] > df["close"]) | \
               (df["high"] < df["open"]) | (df["high"] < df["close"])
        n = mask.sum()
        if n > 0:
            errors += n
            logger.info(f"  ❌ [{symbol}] Incoherent OHLC: {n} rows after cleaning")

    if errors == 0:
        logger.debug(f"  ✅ [{symbol}] Clean integrity passed")

    return errors


# ---------------- RUN ----------------

def run(config: dict) -> bool:
    input_dir: str = config["clean_dir"]

    files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".parquet")
    ]) if os.path.exists(input_dir) else []

    if not files:
        logger.warning(f"⚠ No parquet files found in {input_dir}")
        return False

    logger.info(f"🔍 Clean integrity check — {len(files)} file(s)")

    total_errors = 0
    for filepath in files:
        symbol = os.path.splitext(os.path.basename(filepath))[0].rsplit("_", 1)[0]
        try:
            df = pd.read_parquet(filepath)
        except Exception as e:
            logger.warning(f"  ❌ Could not read {os.path.basename(filepath)}: {e}")
            total_errors += 1
            continue

        total_errors += _check_symbol(df, symbol)

    if total_errors == 0:
        logger.info("✅ Clean integrity check passed")
        return True

    logger.info(f"❌ Clean integrity check FAILED — {total_errors} critical error(s). Pipeline aborted.")
    return False


# ---------------- ENTRY POINT ----------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _config = {
        "clean_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "02_clean"),
        "timeframe": "1Dutc",
    }
    run(_config)
