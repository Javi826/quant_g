# data_main.py
# -----------------------------
import logging
import os
import time

import step1_extraction
import step1b_raw_integrity
import step2_cleaning
import step1b_clean_integrity
import step3_highlow
import step3b_integrity_highlow
import step4_split

# ---------------- LOGGING ----------------
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger("pipeline")

# ---------------- FOLDERS ----------------
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")

RAW_DIR     = os.path.join(DATA_DIR, "01_raw")
CLEAN_DIR   = os.path.join(DATA_DIR, "02_clean")
HIGHLOW_DIR = os.path.join(DATA_DIR, "03_highlow")
SPLIT_DIR   = os.path.join(DATA_DIR, "04_split")
IS_DIR      = os.path.join(SPLIT_DIR, "IS")
OOS_DIR     = os.path.join(SPLIT_DIR, "OOS")

DEBUG_RAW_INTEGRITY_DIR     = os.path.join(DATA_DIR, "debug_01b_raw_integrity")
DEBUG_CLEAN_INTEGRITY_DIR   = os.path.join(DATA_DIR, "debug_01b_clean_integrity")
DEBUG_HIGHLOW_INTEGRITY_DIR = os.path.join(DATA_DIR, "debug_03b_highlow_integrity")

# ---------------- PIPELINE CONFIG ----------------
MODE       = "extraction"   # "extraction" | "pipeline"
DEBUG_MODE = True

TIMEFRAMES       = ["1Dutc", "15m", "5m"]
SELECTED_SYMBOLS = ["BTCUSDT", "ETHUSDT"]
START_DATE       = "2025-01-01"
END_DATE         = None

RUN_STEP3  = True
RUN_STEP3B = True
RUN_STEP4  = True

TIMEFRAMES_HIGHLOW = ["15m", "5m"]

IS_START = "2022-01-01"
IS_END   = "2024-12-31"


# ---------------- HELPERS ----------------

def _make_dirs() -> None:
    for d in [RAW_DIR, CLEAN_DIR, HIGHLOW_DIR, IS_DIR, OOS_DIR]:
        os.makedirs(d, exist_ok=True)
    if DEBUG_MODE:
        for d in [DEBUG_RAW_INTEGRITY_DIR, DEBUG_CLEAN_INTEGRITY_DIR, DEBUG_HIGHLOW_INTEGRITY_DIR]:
            os.makedirs(d, exist_ok=True)


def _run_step(name: str, fn, config: dict) -> bool:
    logger.info(f"\n{'='*60}")
    logger.info(f"  {name}")
    logger.info(f"{'='*60}")
    t0 = time.time()
    try:
        result  = fn(config)
        elapsed = time.time() - t0
        m, s    = divmod(elapsed, 60)
        logger.info(f"  ✅ {name} completed in {int(m)}m {int(s)}s")
        return result if isinstance(result, bool) else True
    except Exception as e:
        elapsed = time.time() - t0
        logger.info(f"  ❌ {name} FAILED after {elapsed:.1f}s: {e}")
        return False


def _build_config(timeframe: str | None = None) -> dict:
    return {
        "start_date":                    START_DATE,
        "end_date":                      END_DATE,
        "timeframe":                     timeframe,
        "selected_symbols":              SELECTED_SYMBOLS,
        "timeframes_highlow":            TIMEFRAMES_HIGHLOW,
        "is_start":                      IS_START,
        "is_end":                        IS_END,
        "raw_dir":                       RAW_DIR,
        "clean_dir":                     CLEAN_DIR,
        "highlow_dir":                   HIGHLOW_DIR,
        "is_dir":                        IS_DIR,
        "oos_dir":                       OOS_DIR,
        "debug_mode":                    DEBUG_MODE,
        "debug_raw_integrity_dir":       DEBUG_RAW_INTEGRITY_DIR,
        "debug_clean_integrity_dir":     DEBUG_CLEAN_INTEGRITY_DIR,
        "debug_highlow_integrity_dir":   DEBUG_HIGHLOW_INTEGRITY_DIR,
    }


# ---------------- MODES ----------------

def _run_extraction() -> None:
    logger.info(f"📋 Timeframes to extract: {TIMEFRAMES}\n")
    for tf in TIMEFRAMES:
        logger.info(f"\n{'#'*60}")
        logger.info(f"  TIMEFRAME: {tf}")
        logger.info(f"{'#'*60}")
        config = _build_config(timeframe=tf)

        ok = _run_step(f"STEP 1 — Extraction [{tf}]", step1_extraction.run, config)
        if not ok:
            logger.info(f"❌ Extraction failed for {tf}. Skipping to next timeframe.")
            continue

        _run_step(f"STEP 1B RAW — Integrity [{tf}]", step1b_raw_integrity.run, config)

        ok = _run_step(f"STEP 2 — Cleaning [{tf}]", step2_cleaning.run, config)
        if not ok:
            logger.info(f"❌ Cleaning failed for {tf}. Skipping to next timeframe.")
            continue

        ok = _run_step(f"STEP 1B CLEAN — Integrity [{tf}]", step1b_clean_integrity.run, config)
        if not ok:
            logger.info(f"❌ Clean integrity failed for {tf}. Skipping to next timeframe.")


def _run_pipeline() -> None:
    config = _build_config()

    if RUN_STEP3:
        ok = _run_step("STEP 3 — High/Low Timestamps", step3_highlow.run, config)
        if not ok:
            logger.info("❌ Pipeline aborted at STEP 3.")
            return

    if RUN_STEP3B:
        _run_step("STEP 3B — High/Low Integrity", step3b_integrity_highlow.run, config)

    if RUN_STEP4:
        ok = _run_step("STEP 4 — IS/OOS Split", step4_split.run, config)
        if not ok:
            logger.info("❌ Pipeline aborted at STEP 4.")


# ---------------- MAIN ----------------

def main() -> None:
    logger.info(f"\n🚀 Starting data pipeline — MODE: {MODE}")
    _make_dirs()
    t_total = time.time()

    if MODE == "extraction":
        _run_extraction()
    elif MODE == "pipeline":
        _run_pipeline()
    else:
        logger.info(f"❌ Unknown MODE '{MODE}'. Use 'extraction' or 'pipeline'.")
        return

    elapsed = time.time() - t_total
    m, s = divmod(elapsed, 60)
    logger.info(f"\n🏁 Pipeline completed in {int(m)}m {int(s)}s")


if __name__ == "__main__":
    main()
