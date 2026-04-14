# step7_split.py
# =============================================================================
# Step 7 — IS/OOS Split — splits data into In-Sample and Out-of-Sample sets.
#
# SPLIT_MODE = "expanding"
#   IS  : from START_DATE until (today - WINDOW_OOS_MONTHS)
#   OOS : last WINDOW_OOS_MONTHS up to today
#   Each run the IS grows as more data becomes available.
#
# SPLIT_MODE = "rolling"
#   Run 1: IS = START_DATE → START_DATE + WINDOW_IS_MONTHS
#           OOS = IS_end → IS_end + WINDOW_OOS_MONTHS
#   Run 2: IS shifts forward by WINDOW_OOS_MONTHS
#           IS = Run1_IS_start + WINDOW_OOS_MONTHS → same duration
#           OOS = IS_end → IS_end + WINDOW_OOS_MONTHS
#   Each run the entire window slides forward by WINDOW_OOS_MONTHS.
#
# SPLIT_REFERENCE_DATE
#   Use None to always split relative to today (normal production use).
#   Set to a past date (e.g. "2025-10-01") to simulate how the split
#   would have looked at that point in time — useful for backtesting
#   or reconstructing historical train/test sets.
#
# Output folders are dated: 04_split/YYYY-MM/IS and 04_split/YYYY-MM/OOS
# =============================================================================
import logging
import os
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta

import pandas as pd

logger = logging.getLogger("pipeline.step7")


# =============================================================================
# WINDOW CALCULATION
# =============================================================================

def _compute_windows(config: dict) -> tuple[str, str, str, str]:
    """Returns (is_start, is_end, oos_start, oos_end) as ISO date strings."""
    split_mode   = config.get("split_mode", "expanding")
    window_is    = config.get("window_is_months", 12)
    window_oos   = config.get("window_oos_months", 3)
    ref_date_str = config.get("split_reference_date", None)
    start_date   = config.get("start_date", "2020-01-01")

    # Reference date — today or override
    if ref_date_str:
        ref = pd.to_datetime(ref_date_str).to_pydatetime()
    else:
        ref = datetime.now(tz=timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)

    if split_mode == "expanding":
        is_start  = pd.to_datetime(start_date).to_pydatetime()
        oos_end   = ref
        oos_start = ref - relativedelta(months=window_oos)
        is_end    = oos_start

    elif split_mode == "rolling":
        # Determine which run we are on based on existing split folders
        split_dir   = config.get("split_dir", "")
        run_number  = _get_run_number(split_dir)
        offset      = relativedelta(months=window_oos * run_number)
        is_start    = pd.to_datetime(start_date).to_pydatetime() + offset
        is_end      = is_start + relativedelta(months=window_is)
        oos_start   = is_end
        oos_end     = oos_start + relativedelta(months=window_oos)
    else:
        raise ValueError(f"Unknown SPLIT_MODE: '{split_mode}'. Use 'expanding' or 'rolling'.")

    return (
        is_start.strftime("%Y-%m-%d"),
        is_end.strftime("%Y-%m-%d"),
        oos_start.strftime("%Y-%m-%d"),
        oos_end.strftime("%Y-%m-%d"),
    )


def _get_run_number(split_dir: str) -> int:
    """Counts existing dated split folders to determine current run number."""
    if not os.path.exists(split_dir):
        return 0
    existing = [
        d for d in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, d)) and len(d) == 7 and d[4] == "-"
    ]
    return len(existing)


# =============================================================================
# SPLIT & SAVE
# =============================================================================

def _split(df: pd.DataFrame, is_start: str, is_end: str, oos_start: str, oos_end: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "timestamp" not in df.columns:
        df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    df_is  = df[(df["timestamp"] >= pd.to_datetime(is_start))  & (df["timestamp"] < pd.to_datetime(is_end))].copy()
    df_oos = df[(df["timestamp"] >= pd.to_datetime(oos_start)) & (df["timestamp"] < pd.to_datetime(oos_end))].copy()

    return df_is, df_oos


def _save(df: pd.DataFrame, path: str, export_csv: bool = False) -> None:
    df_save = df.copy()
    if "timestamp" in df_save.columns:
        df_save = df_save.set_index("timestamp")
    df_save.to_parquet(path, index=True)
    if export_csv:
        df_save.to_csv(os.path.splitext(path)[0] + ".csv", index=True)


# =============================================================================
# RUN
# =============================================================================

def run(config: dict) -> bool:
    input_dir: str   = config["highlow_dir"]
    split_dir: str   = config["split_dir"]
    export_csv: bool = config.get("export_csv", False)
    split_mode: str  = config.get("split_mode", "expanding")

    is_start, is_end, oos_start, oos_end = _compute_windows(config)

    period_label = oos_start[:7]
    is_dir  = os.path.join(split_dir, period_label, "IS")
    oos_dir = os.path.join(split_dir, period_label, "OOS")
    os.makedirs(is_dir, exist_ok=True)
    os.makedirs(oos_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".parquet")]) \
        if os.path.exists(input_dir) else []

    if not files:
        logger.warning(f"⚠ No parquet files found in {input_dir}")
        return False

    logger.info(f"✂️  IS/OOS split [{split_mode}] — {len(files)} file(s)")
    logger.info(f"   IS  : {is_start} → {is_end}")
    logger.info(f"   OOS : {oos_start} → {oos_end}")
    logger.info(f"   Period: {period_label}")

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
            df_is, df_oos = _split(df, is_start, is_end, oos_start, oos_end)
            has_is  = not df_is.empty
            has_oos = not df_oos.empty

            if has_is:
                _save(df_is, os.path.join(is_dir, filename), export_csv)
            else:
                skipped_is += 1
                logger.debug(f"  ⚠ [{symbol}] No IS data")

            if has_oos:
                _save(df_oos, os.path.join(oos_dir, filename), export_csv)
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
        logger.warning(f"⚠ Step 7 completed with {errors} error(s)")
        return False

    logger.info("✅ IS/OOS split complete")
    return True


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _base = os.path.dirname(os.path.abspath(__file__))
    _config = {
        "highlow_dir":          os.path.join(_base, "data", "03_highlow"),
        "split_dir":            os.path.join(_base, "data", "04_split"),
        "split_mode":           "expanding",
        "window_is_months":     12,
        "window_oos_months":    3,
        "start_date":           "2025-01-01",
        "split_reference_date": None,
        "export_csv":           False,
    }
    run(_config)