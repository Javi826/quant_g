# data_main.py
# =============================================================================
# Data Pipeline Orchestrator — single execution mode.
# Runs all steps sequentially: extraction → highlow → split.
# Extraction is incremental: only new candles are downloaded each run.
# =============================================================================
import logging
import os
import time

import step0_symbol_selection
import step1_extraction
import integrity
import step3_cleaning
import step5_highlow
import step7_split

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger("pipeline")

# =============================================================================
# FOLDERS
# =============================================================================
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")

RAW_DIR     = os.path.join(DATA_DIR, "01_raw")
CLEAN_DIR   = os.path.join(DATA_DIR, "02_clean")
HIGHLOW_DIR = os.path.join(DATA_DIR, "03_highlow")
SPLIT_DIR   = os.path.join(DATA_DIR, "04_split")

DEBUG_RAW_INTEGRITY_DIR     = os.path.join(DATA_DIR, "debug_02_raw_integrity")
DEBUG_CLEAN_INTEGRITY_DIR   = os.path.join(DATA_DIR, "debug_04_clean_integrity")
DEBUG_HIGHLOW_INTEGRITY_DIR = os.path.join(DATA_DIR, "debug_06_highlow_integrity")

# =============================================================================
# PIPELINE CONFIG
# =============================================================================
DEBUG_MODE = True
EXPORT_CSV = False   # Set True to export CSV alongside parquet at each step

# =============================================================================
# SYMBOL SELECTION
# =============================================================================
SYMBOL_MODE        = "manual"               # "manual" → use SELECTED_SYMBOLS list
                                            # "auto"   → rank all symbols by avg daily volume
                                            #            and pick top N_SYMBOLS_DOWNLOAD
SELECTED_SYMBOLS   = ["BTCUSDT", "ETHUSDT"] # only used when SYMBOL_MODE = "manual"
N_SYMBOLS_DOWNLOAD = 50                     # only used when SYMBOL_MODE = "auto"

# =============================================================================
# EXTRACTION
# =============================================================================
TIMEFRAMES = ["1Dutc", "4H", "1H"]
START_DATE = "2025-01-01"
END_DATE   = None   # None = download up to today
                    # Set e.g. "2025-06-01" to limit download (useful for testing incremental)

# =============================================================================
# HIGH/LOW TIMESTAMPS
# =============================================================================
TIMEFRAMES_HIGHLOW = [["1Dutc", "1H"], ["4H", "1H"]]   # list of [higher_tf, intrabar_tf] pairs

# =============================================================================
# IS/OOS SPLIT
# =============================================================================
# SPLIT_MODE = "expanding"
#   IS  : from START_DATE until (today - WINDOW_OOS_MONTHS)
#   OOS : last WINDOW_OOS_MONTHS up to today
#   Each monthly run the IS grows as more data is available.
#
# SPLIT_MODE = "rolling"
#   IS  : fixed WINDOW_IS_MONTHS duration, slides forward WINDOW_OOS_MONTHS each run
#   OOS : next WINDOW_OOS_MONTHS after IS end
#   Run 1: IS = START_DATE → START_DATE + WINDOW_IS_MONTHS
#   Run 2: IS = START_DATE + WINDOW_OOS_MONTHS → same duration
#   Run 3: IS = START_DATE + 2*WINDOW_OOS_MONTHS → same duration ...
#
# SPLIT_REFERENCE_DATE
#   None  → split is always calculated relative to today (normal monthly use)
#   "YYYY-MM-DD" → simulate how the split would look at a past date,
#                  useful for backtesting or reconstructing historical splits

SPLIT_MODE           = "expanding"
WINDOW_IS_MONTHS     = 12    # only used when SPLIT_MODE = "rolling"
WINDOW_OOS_MONTHS    = 3
SPLIT_REFERENCE_DATE = None


# =============================================================================
# HELPERS
# =============================================================================

def _make_dirs() -> None:
    for d in [RAW_DIR, CLEAN_DIR, HIGHLOW_DIR, SPLIT_DIR]:
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


def _build_config(timeframe: str | None = None, selected_symbols: list | None = None) -> dict:
    return {
        "start_date":                    START_DATE,
        "end_date":                      END_DATE,
        "timeframe":                     timeframe,
        "selected_symbols":              selected_symbols or SELECTED_SYMBOLS,
        "symbol_mode":                   SYMBOL_MODE,
        "n_symbols_download":            N_SYMBOLS_DOWNLOAD,
        "timeframes_highlow":            TIMEFRAMES_HIGHLOW,
        "split_mode":                    SPLIT_MODE,
        "window_is_months":              WINDOW_IS_MONTHS,
        "window_oos_months":             WINDOW_OOS_MONTHS,
        "split_reference_date":          SPLIT_REFERENCE_DATE,
        "export_csv":                    EXPORT_CSV,
        "raw_dir":                       RAW_DIR,
        "clean_dir":                     CLEAN_DIR,
        "highlow_dir":                   HIGHLOW_DIR,
        "split_dir":                     SPLIT_DIR,
        "debug_mode":                    DEBUG_MODE,
        "debug_raw_integrity_dir":       DEBUG_RAW_INTEGRITY_DIR,
        "debug_clean_integrity_dir":     DEBUG_CLEAN_INTEGRITY_DIR,
        "debug_highlow_integrity_dir":   DEBUG_HIGHLOW_INTEGRITY_DIR,
    }


# =============================================================================
# PIPELINE
# =============================================================================

def _run_pipeline() -> None:

    # Step 0 — Symbol selection (resolves symbol list once for all timeframes)
    config_s0 = _build_config()
    ok = _run_step("STEP 0 — Symbol Selection", step0_symbol_selection.run, config_s0)
    if not ok:
        logger.info("❌ Pipeline aborted at STEP 0.")
        return
    selected_symbols = config_s0["selected_symbols"]

    # Steps 1-4 — Extraction + integrity + cleaning per timeframe
    logger.info(f"\n📋 Timeframes to extract: {TIMEFRAMES}\n")
    for tf in TIMEFRAMES:
        logger.info(f"\n{'#'*60}")
        logger.info(f"  TIMEFRAME: {tf}")
        logger.info(f"{'#'*60}")
        config = _build_config(timeframe=tf, selected_symbols=selected_symbols)

        ok = _run_step(f"STEP 1 — Extraction [{tf}]", step1_extraction.run, config)
        if not ok:
            logger.info(f"❌ Extraction failed for {tf}. Skipping to next timeframe.")
            continue

        _run_step(f"STEP 2 — Raw Integrity [{tf}]", integrity.run_raw, config)

        ok = _run_step(f"STEP 3 — Cleaning [{tf}]", step3_cleaning.run, config)
        if not ok:
            logger.info(f"❌ Cleaning failed for {tf}. Skipping to next timeframe.")
            continue

        ok = _run_step(f"STEP 4 — Clean Integrity [{tf}]", integrity.run_clean, config)
        if not ok:
            logger.info(f"❌ Clean integrity failed for {tf}. Skipping to next timeframe.")

    # Steps 5-7 — High/Low + integrity + IS/OOS split
    config = _build_config(selected_symbols=selected_symbols)

    ok = _run_step("STEP 5 — High/Low Timestamps", step5_highlow.run, config)
    if not ok:
        logger.info("❌ Pipeline aborted at STEP 5.")
        return

    _run_step("STEP 6 — High/Low Integrity", integrity.run_highlow, config)

    ok = _run_step("STEP 7 — IS/OOS Split", step7_split.run, config)
    if not ok:
        logger.info("❌ Pipeline aborted at STEP 7.")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    logger.info("\n🚀 Starting data pipeline")
    _make_dirs()
    t_total = time.time()
    _run_pipeline()
    elapsed = time.time() - t_total
    m, s = divmod(elapsed, 60)
    logger.info(f"\n🏁 Pipeline completed in {int(m)}m {int(s)}s")


if __name__ == "__main__":
    main()