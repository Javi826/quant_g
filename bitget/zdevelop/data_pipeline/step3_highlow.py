# step3_highlow.py
# -----------------------------
import logging
import os

import pandas as pd
from tqdm import tqdm

logger = logging.getLogger("pipeline.step3")


# ---------------- UTILITIES ----------------

def _parse_filename(filename: str) -> tuple[str, str]:
    stem  = os.path.splitext(filename)[0]
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return stem, ""


def _read_parquet(filepath: str) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(filepath)
        if "timestamp" in df.columns:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
        elif not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        df.columns = df.columns.str.lower()
        return df
    except Exception as e:
        logger.warning(f"  ❌ Could not read {os.path.basename(filepath)}: {e}")
        return None


def _write_parquet(df: pd.DataFrame, filepath: str) -> None:
    df_out = df.reset_index()
    if "index" in df_out.columns:
        df_out = df_out.rename(columns={"index": "timestamp"})
    df_out.to_parquet(filepath, index=False)


# ---------------- CORE ----------------

def _find_timestamp_extremum(df_high: pd.DataFrame, df_low: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = df_high.copy()
    df = df.loc[df_low.index[0]:]
    df["low_time"]  = pd.NaT
    df["high_time"] = pd.NaT

    for i in tqdm(range(len(df) - 1), desc=f"{symbol}", leave=False):
        start    = df.index[i]
        end      = df.index[i + 1]
        intrabar = df_low.loc[start:end].iloc[:-1]
        if intrabar.empty:
            continue
        try:
            df.loc[start, "high_time"] = intrabar["high"].idxmax()
            df.loc[start, "low_time"]  = intrabar["low"].idxmin()
        except Exception as e:
            logger.debug(f"  ⚠ [{symbol}] Error at {start}: {e}")

    df    = df.iloc[:-1]
    valid = df[["low_time", "high_time"]].notna().all(axis=1).sum()
    total = len(df)
    pct   = valid / total * 100 if total > 0 else 0
    logger.info(f"  [{symbol}] Valid rows: {valid}/{total} ({pct:.1f}%)")

    return df


# ---------------- RUN ----------------

def run(config: dict) -> bool:
    input_dir: str  = config["clean_dir"]
    output_dir: str = config["highlow_dir"]
    tf_pair: list   = config["timeframes_highlow"]
    os.makedirs(output_dir, exist_ok=True)

    if len(tf_pair) != 2:
        logger.warning("⚠ 'timeframes_highlow' must be [higher_tf, intrabar_tf]")
        return False

    tf_high, tf_low = tf_pair[0], tf_pair[1]

    files = [f for f in os.listdir(input_dir) if f.endswith(".parquet")] \
        if os.path.exists(input_dir) else []

    if not files:
        logger.warning(f"⚠ No parquet files found in {input_dir}")
        return False

    # Index by (symbol, timeframe)
    file_index: dict[tuple[str, str], str] = {}
    for f in files:
        sym, tf = _parse_filename(f)
        if sym and tf:
            file_index[(sym, tf)] = os.path.join(input_dir, f)

    symbols = sorted({sym for sym, _ in file_index})
    logger.info(f"📊 High/Low timestamps — {len(symbols)} symbol(s) | {tf_high} → {tf_low}")

    errors = 0
    for sym in symbols:
        key_high = (sym, tf_high)
        key_low  = (sym, tf_low)

        if key_high not in file_index:
            logger.warning(f"  ⚠ [{sym}] Missing {tf_high} file. Skipping.")
            continue
        if key_low not in file_index:
            logger.warning(f"  ⚠ [{sym}] Missing {tf_low} file. Skipping.")
            continue

        df_high = _read_parquet(file_index[key_high])
        df_low  = _read_parquet(file_index[key_low])

        if df_high is None or df_low is None:
            errors += 1
            continue

        try:
            df_result  = _find_timestamp_extremum(df_high, df_low, sym)
            out_path   = os.path.join(output_dir, os.path.basename(file_index[key_high]))
            _write_parquet(df_result, out_path)
            logger.info(f"  💾 [{sym}] Saved → {os.path.basename(out_path)}")
        except Exception as e:
            logger.warning(f"  ❌ [{sym}] Failed: {e}")
            errors += 1

    if errors:
        logger.warning(f"⚠ Step 3 completed with {errors} error(s)")
        return False

    logger.info("✅ High/Low timestamps complete")
    return True


# ---------------- ENTRY POINT ----------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    _base = os.path.dirname(os.path.abspath(__file__))
    _config = {
        "clean_dir":          os.path.join(_base, "data", "02_clean"),
        "highlow_dir":        os.path.join(_base, "data", "03_highlow"),
        "timeframes_highlow": ["15m", "5m"],
    }
    run(_config)
