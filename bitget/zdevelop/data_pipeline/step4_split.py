# step4_split.py
# -----------------------------
import logging
import os

import pandas as pd

logger = logging.getLogger("pipeline.step4")


# ---------------- SPLIT ----------------

def _split_is_oos(df: pd.DataFrame, is_start: str, is_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "timestamp" not in df.columns:
        df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    start  = pd.to_datetime(is_start)
    end    = pd.to_datetime(is_end)
    df_is  = df[(df["timestamp"] >= start) & (df["timestamp"] < end)].copy()
    df_oos = df[df["timestamp"] >= end].copy()

    return df_is, df_oos


def _save(df: pd.DataFrame, path: str) -> None:
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    df.to_parquet(path, index=True)


# ---------------- RUN ----------------

def run(config: dict) -> bool:
    input_dir: str = config["highlow_dir"]
    is_dir: str    = config["is_dir"]
    oos_dir: str   = config["oos_dir"]
    is_start: str  = config["is_start"]
    is_end: str    = config["is_end"]

    os.makedirs(is_dir, exist_ok=True)
    os.makedirs(oos_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".parquet")]) \
        if os.path.exists(input_dir) else []

    if not files:
        logger.warning(f"⚠ No parquet files found in {input_dir}")
        return False

    logger.info(f"✂️  IS/OOS split — {len(files)} file(s)")
    logger.info(f"   IS : {is_start} → {is_end}")
    logger.info(f"   OOS: {is_end} → end of file")

    skipped_is = skipped_oos = errors = processed = 0

    for filename in files:
        symbol   = os.path.splitext(filename)[0].rsplit("_", 1)[0]
        filepath = os.path.join(input_dir, filename)
        try:
            df = pd.read_parquet(filepath)
        except Exception as e:
            logger.warning(f"  ❌ Could not read {filename}: {e}")
            errors += 1
            continue

        try:
            df_is, df_oos = _split_is_oos(df, is_start, is_end)
            has_is  = not df_is.empty
            has_oos = not df_oos.empty

            if has_is:
                _save(df_is, os.path.join(is_dir, filename))
            else:
                skipped_is += 1
                logger.debug(f"  ⚠ [{symbol}] No IS data")

            if has_oos:
                _save(df_oos, os.path.join(oos_dir, filename))
            else:
                skipped_oos += 1
                logger.debug(f"  ⚠ [{symbol}] No OOS data")

            if has_is or has_oos:
                processed += 1
                logger.info(f"  ✅ [{symbol}] IS: {len(df_is)} rows | OOS: {len(df_oos)} rows")

        except Exception as e:
            logger.warning(f"  ❌ [{symbol}] Error: {e}")
            errors += 1

    logger.info(f"\n  Processed: {processed} | Skipped IS: {skipped_is} | Skipped OOS: {skipped_oos} | Errors: {errors}")

    if errors:
        logger.warning(f"⚠ Step 4 completed with {errors} error(s)")
        return False

    logger.info("✅ IS/OOS split complete")
    return True


# ---------------- ENTRY POINT ----------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _base = os.path.dirname(os.path.abspath(__file__))
    _config = {
        "highlow_dir": os.path.join(_base, "data", "03_highlow"),
        "is_dir":      os.path.join(_base, "data", "04_split", "IS"),
        "oos_dir":     os.path.join(_base, "data", "04_split", "OOS"),
        "is_start":    "2022-01-01",
        "is_end":      "2024-12-31",
    }
    run(_config)
