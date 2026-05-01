# step7_split.py
# =============================================================================
# Step 7 — IS/OOS Split — splits data into In-Sample and Out-of-Sample sets.
#
# SPLIT_MODE = "expanding"
#   IS  : from START_DATE until (today - WINDOW_OOS_MONTHS)
#   OOS : last WINDOW_OOS_MONTHS up to today
#   Each monthly run the IS grows as more data is available.
#
# SPLIT_MODE = "rolling"
#   OOS always ends at today and lasts WINDOW_OOS_MONTHS.
#   IS ends where OOS starts. IS_ROLLING_MONTHS controls how much
#   the IS start advances on each consecutive run.
#   Run 1: IS = START_DATE → (today - WINDOW_OOS_MONTHS)
#   Run 2: IS start advances by IS_ROLLING_MONTHS
#   State is persisted in rolling_state.csv so each run knows where to resume.
#
# SPLIT_REFERENCE_DATE
#   Controls the IS/OOS cut point calculation (step 7 only) — does NOT affect download.
#   None        → split calculated relative to today (normal monthly production use)
#   "YYYY-MM-DD"→ simulate how the split would have looked at that past date.
#                 Useful for backtesting or reconstructing historical train/test sets.
#   Example: data downloaded up to 2026-04-14, SPLIT_REFERENCE_DATE = "2025-10-01"
#            → IS/OOS calculated as if today were 2025-10-01, ignoring later data
#
# Output structure:
#   04_split/
#       expanding/
#           IS/  crypto_2022-01_2026-01_IS/  ← parquets here
#           OOS/ crypto_2026-01_2026-04_OOS/ ← parquets here
#       rolling/
#           rolling_state.csv
#           IS/  crypto_2022-01_2026-01_IS/
#           OOS/ crypto_2026-01_2026-04_OOS/
# =============================================================================
import csv
import logging
import os
from datetime import datetime, timezone
from dateutil.relativedelta import relativedelta
from shared_config import REGIME_REFERENCE_SYMBOL

import pandas as pd

logger = logging.getLogger("pipeline.step7")

ROLLING_STATE_FILE  = "rolling_state.csv"
REFERENCE_SYMBOL_TF = "1Dutc"   # Used to read available data range for preview


# =============================================================================
# ROLLING STATE
# =============================================================================

def _load_rolling_state(mode_dir: str) -> dict | None:
    """Loads rolling window state from CSV. Returns None if no state exists."""
    path = os.path.join(mode_dir, ROLLING_STATE_FILE)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", newline="") as f:
            reader = csv.DictReader(f)
            rows   = list(reader)
            return rows[0] if rows else None
    except Exception as e:
        logger.warning(f"  ⚠ Could not load rolling state: {e}")
        return None


def _save_rolling_state(mode_dir: str, is_start: str, is_end: str, oos_start: str, oos_end: str) -> None:
    """Persists rolling window state to CSV for next run."""
    state = {
        "last_run":  datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "is_start":  is_start,
        "is_end":    is_end,
        "oos_start": oos_start,
        "oos_end":   oos_end,
    }
    path = os.path.join(mode_dir, ROLLING_STATE_FILE)
    try:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=state.keys())
            writer.writeheader()
            writer.writerow(state)
        logger.debug(f"  Rolling state saved → {ROLLING_STATE_FILE}")
    except Exception as e:
        logger.warning(f"  ⚠ Could not save rolling state: {e}")


# =============================================================================
# WINDOW CALCULATION
# =============================================================================

def _compute_windows(config: dict, mode_dir: str) -> tuple[str, str, str, str]:
    """Returns (is_start, is_end, oos_start, oos_end) as ISO date strings."""
    split_mode      = config.get("split_mode", "expanding")
    window_oos      = config.get("window_oos_months", 3)
    is_rolling      = config.get("is_rolling_months", 3)
    ref_date_str    = config.get("split_reference_date", None)
    start_date      = config.get("start_date", "2020-01-01")

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
        # OOS always ends at ref and lasts WINDOW_OOS_MONTHS
        oos_end   = ref
        oos_start = ref - relativedelta(months=window_oos)
        is_end    = oos_start

        state = _load_rolling_state(mode_dir)
        if state is None:
            logger.info("  Rolling state not found — starting from run 1.")
            is_start = pd.to_datetime(start_date).to_pydatetime()
        else:
            logger.info(f"  Rolling state loaded — last run: {state['last_run']}")
            is_start = pd.to_datetime(state["is_start"]).to_pydatetime() + relativedelta(months=is_rolling)
    else:
        raise ValueError(f"Unknown SPLIT_MODE: '{split_mode}'. Use 'expanding' or 'rolling'.")

    return (
        is_start.strftime("%Y-%m-%d"),
        is_end.strftime("%Y-%m-%d"),
        oos_start.strftime("%Y-%m-%d"),
        oos_end.strftime("%Y-%m-%d"),
    )


# =============================================================================
# FOLDER NAMING
# =============================================================================

def _make_folder_name(is_start: str, is_end: str, oos_start: str, oos_end: str, subset: str) -> str:
    """
    Builds descriptive folder name.
    IS  → crypto_2022-01_2026-01_IS
    OOS → crypto_2026-01_2026-04_OOS
    """
    if subset == "IS":
        start = is_start[:7]
        end   = is_end[:7]
    else:
        start = oos_start[:7]
        end   = oos_end[:7]
    return f"crypto_{start}_{end}_{subset}"


# =============================================================================
# DATA RANGE READER — for preview
# =============================================================================

def _get_data_range(raw_dir: str) -> tuple[str, str] | None:
    """Reads min/max date from BTCUSDT 1Dutc parquet for preview calculation."""
    filename = f"{REGIME_REFERENCE_SYMBOL}_{REFERENCE_SYMBOL_TF}.parquet"
    # Try clean dir first, then raw
    for folder in [raw_dir.replace("01_raw", "02_clean"), raw_dir]:
        path = os.path.join(folder, filename)
        if os.path.exists(path):
            try:
                df = pd.read_parquet(path)
                if "timestamp" not in df.columns:
                    df = df.reset_index()
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                return (
                    df["timestamp"].min().strftime("%Y-%m-%d"),
                    df["timestamp"].max().strftime("%Y-%m-%d"),
                )
            except Exception:
                pass
    return None


# =============================================================================
# SPLIT PREVIEW
# =============================================================================

def print_split_preview(config: dict) -> bool:
    """
    Prints IS/OOS split preview before execution and asks for confirmation.
    Returns True if user confirms, False if user aborts.
    """
    split_mode   = config.get("split_mode", "expanding")
    window_oos   = config.get("window_oos_months", 3)
    is_rolling   = config.get("is_rolling_months", 3)
    start_date   = config.get("start_date", "2020-01-01")
    ref_date_str = config.get("split_reference_date", None)
    split_dir    = config.get("split_dir", "")
    raw_dir      = config.get("raw_dir", "")

    mode_dir = os.path.join(split_dir, split_mode)
    os.makedirs(mode_dir, exist_ok=True)

    # Compute windows
    is_start, is_end, oos_start, oos_end = _compute_windows(config, mode_dir)

    # IS/OOS durations
    is_start_dt  = pd.to_datetime(is_start).to_pydatetime()
    is_end_dt    = pd.to_datetime(is_end).to_pydatetime()
    oos_start_dt = pd.to_datetime(oos_start).to_pydatetime()
    oos_end_dt   = pd.to_datetime(oos_end).to_pydatetime()

    is_months  = (is_end_dt.year - is_start_dt.year) * 12 + (is_end_dt.month - is_start_dt.month)
    oos_months = (oos_end_dt.year - oos_start_dt.year) * 12 + (oos_end_dt.month - oos_start_dt.month)

    # Folder names
    is_folder  = _make_folder_name(is_start, is_end, oos_start, oos_end, "IS")
    oos_folder = _make_folder_name(is_start, is_end, oos_start, oos_end, "OOS")

    # Data range
    data_range = _get_data_range(raw_dir)

    ref_label = ref_date_str if ref_date_str else datetime.now(tz=timezone.utc).strftime("%Y-%m-%d") + " (today)"

    print(f"\n{'='*60}")
    print(f"  📊 Split preview — {ref_label}")
    print(f"{'='*60}")
    print(f"  Mode              : {split_mode}")
    if data_range:
        print(f"  Data available    : {data_range[0]} → {data_range[1]}")
    print(f"  START_DATE        : {start_date}")
    print(f"  WINDOW_OOS_MONTHS : {window_oos}")
    if split_mode == "rolling":
        state = _load_rolling_state(mode_dir)
        last_is_start = state["is_start"] if state else "n/a (run 1)"
        print(f"  IS_ROLLING_MONTHS : {is_rolling}")
        print(f"  Last run IS start : {last_is_start}")
    print(f"")
    print(f"  IS  : {is_start} → {is_end}  ({is_months} months)")
    print(f"  OOS : {oos_start} → {oos_end}  ({oos_months} months available)")
    print(f"")
    print(f"  📁 Output folders:")
    print(f"  IS  → {os.path.join(split_mode, 'IS',  is_folder)}/")
    print(f"  OOS → {os.path.join(split_mode, 'OOS', oos_folder)}/")
    print(f"{'='*60}")

    answer = input("\n  Continue? [y/n]: ").strip().lower()
    return answer == "y"


# =============================================================================
# GET LATEST SPLIT FOLDERS — utility for downstream scripts
# =============================================================================

def get_latest_split_folders(split_dir: str, mode: str = "expanding") -> dict | None:
    """
    Returns paths to the most recent IS and OOS folders for the given mode.
    Useful for downstream scripts (e.g. wfo_mc_parity.py) to auto-resolve data paths.

    Returns:
        {"IS": "/path/to/IS/crypto_..._IS", "OOS": "/path/to/OOS/crypto_..._OOS"}
        or None if no folders found.
    """
    mode_dir = os.path.join(split_dir, mode)
    result   = {}

    for subset in ["IS", "OOS"]:
        subset_dir = os.path.join(mode_dir, subset)
        if not os.path.exists(subset_dir):
            return None
        folders = sorted([
            f for f in os.listdir(subset_dir)
            if os.path.isdir(os.path.join(subset_dir, f)) and f.startswith("crypto_")
        ])
        if not folders:
            return None
        result[subset] = os.path.join(subset_dir, folders[-1])

    return result


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

    mode_dir = os.path.join(split_dir, split_mode)
    os.makedirs(mode_dir, exist_ok=True)

    is_start, is_end, oos_start, oos_end = _compute_windows(config, mode_dir)

    is_folder_name  = _make_folder_name(is_start, is_end, oos_start, oos_end, "IS")
    oos_folder_name = _make_folder_name(is_start, is_end, oos_start, oos_end, "OOS")
    is_dir  = os.path.join(mode_dir, "IS",  is_folder_name)
    oos_dir = os.path.join(mode_dir, "OOS", oos_folder_name)
    os.makedirs(is_dir, exist_ok=True)
    os.makedirs(oos_dir, exist_ok=True)

    files = sorted([f for f in os.listdir(input_dir) if f.endswith(".parquet")]) \
        if os.path.exists(input_dir) else []

    if not files:
        logger.warning(f"⚠ No parquet files found in {input_dir}")
        return False

    logger.info(f"✂️  IS/OOS split [{split_mode}] — {len(files)} file(s)")
    logger.info(f"   IS  : {is_start} → {is_end}  →  {is_folder_name}")
    logger.info(f"   OOS : {oos_start} → {oos_end}  →  {oos_folder_name}")

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

    if split_mode == "rolling":
        _save_rolling_state(mode_dir, is_start, is_end, oos_start, oos_end)

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
        "raw_dir":              os.path.join(_base, "data", "01_raw"),
        "split_mode":           "expanding",
        "window_oos_months":    3,
        "is_rolling_months":    3,
        "start_date":           "2025-01-01",
        "split_reference_date": None,
        "export_csv":           False,
    }
    run(_config)