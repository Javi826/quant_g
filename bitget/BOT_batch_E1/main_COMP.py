#BOT_batch/main_COMP.py
"""
DSR vs StepM — full brute universe comparison, single standalone script.
"""
import os
import sys
import time
import logging
import numpy as np
from scipy.stats import pearsonr, spearmanr
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_batch")))

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL = logging.INFO
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_batch.main_comp")
logger.setLevel(LOG_LEVEL)

DSR_LOG_LEVEL = logging.INFO
logging.getLogger("BOT_batch.pipeline.dsr").setLevel(DSR_LOG_LEVEL)
logging.getLogger("BOT_batch.pipeline.backtest_runner").setLevel(DSR_LOG_LEVEL)

STEPM_LOG_LEVEL = logging.INFO
logging.getLogger("BOT_batch.pipeline.stepm").setLevel(STEPM_LOG_LEVEL)

logging.getLogger("joblib").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from shared_batchs.symbols.universe import filter_symbols, select_universe
from shared_batchs.setup.config_paths import DATA_FOLDER_IS
from shared_batchs.rule_mining.rule_generator import MAX_DEPTH as RULE_MAX_DEPTH
from shared_batchs.rule_mining.rule_runner import _build_rule_dicts
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_batchs.setup.config_backtest import MIN_PRICE, ORDER_AMOUNT
from shared_batchs.pipeline import backtest_runner as backtest_module
from shared_batchs.pipeline.dsr import pipe_dsr
from shared_batchs.pipeline.stepm import (
    pipe_stepm, STEPM_ALPHA, WHITE_PVALUE_TH, WHITE_N_BOOTSTRAP, WHITE_BLOCK_SIZE,
)

# =============================================================================
# UNIVERSE / SEARCH SPACE CONFIGURATION
# =============================================================================
DTYPE  = np.float32
N_JOBS = -1  # -1 = use all available cores, for both the backtest search and the StepM bootstrap

TIMEFRAMES = ["1H", "4H", "6Hutc", "12Hutc"]
TIMEFRAMES = ["12Hutc"]
N_SYMBOLS  = 10

PARAM_GRID = {
    "SELL_AFTER": [50],
    "TP_PCT":     [2, 4, 6, 8, 10],
    "SL_PCT":     [2, 4, 6, 8, 10],
}

# =============================================================================
# DSR vs STEPM — full brute universe comparison, no other pipeline stages
# =============================================================================
DSR_TH = 0.8

STEPM_ALPHA_RUN     = STEPM_ALPHA
STEPM_PVALUE_TH_RUN = WHITE_PVALUE_TH
STEPM_N_BOOTSTRAP   = WHITE_N_BOOTSTRAP
STEPM_BLOCK_SIZE    = WHITE_BLOCK_SIZE


# =============================================================================
# COMPARISON — build the raw universe once, hand it to both pipes, compare.
# =============================================================================
def compare_dsr_vs_stepm_from_raw(
    raw_results: list,
    dsr_th: float,
    n_combos: int,
    stepm_alpha: float | None = None,
    stepm_pvalue_th: float | None = None,
    n_bootstrap: int | None = None,
    block_size: int | None = None,
    timeframe: str = "",
    n_jobs: int = -1,
) -> dict:

    dsr_results = pipe_dsr(
        raw_results = raw_results,
        dsr_th      = dsr_th,
        n_combos    = n_combos,
        timeframe   = timeframe,
    )
    dsr_by_id = {r["rule_id"]: r for r in dsr_results}

    stepm_results = pipe_stepm(
        raw_results     = raw_results,
        stepm_alpha     = stepm_alpha,
        stepm_pvalue_th = stepm_pvalue_th,
        n_bootstrap     = n_bootstrap,
        block_size      = block_size,
        n_jobs          = n_jobs,
        timeframe       = timeframe,
    )
    stepm_by_id = {r["rule_id"]: r for r in stepm_results}

    _print_comparison(raw_results, dsr_by_id, stepm_by_id, timeframe)

    return {"dsr_by_id": dsr_by_id, "stepm_by_id": stepm_by_id}


def compare_dsr_vs_stepm(
    rules: list,
    ohlcv_arr: dict,
    param_grid: dict,
    order_amount: int,
    dtype,
    dsr_th: float,
    stepm_alpha: float | None = None,
    stepm_pvalue_th: float | None = None,
    n_bootstrap: int | None = None,
    block_size: int | None = None,
    timeframe: str = "",
    n_jobs: int = -1,
) -> dict:

    # pipe_backtesting (backtest_runner.py) has no n_jobs parameter of its
    # own — it always uses the module-level BACKTEST_N_JOBS constant
    # internally. We override that constant for the duration of this call so
    # n_jobs actually reaches the backtest search step too, then restore it,
    # so we don't leave backtest_runner.py permanently mutated as a side
    # effect of this script.
    original_n_jobs = backtest_module.BACKTEST_N_JOBS
    backtest_module.BACKTEST_N_JOBS = n_jobs
    try:
        raw_results, n_combos = backtest_module.pipe_backtesting(
            rules        = rules,
            ohlcv_arr    = ohlcv_arr,
            param_grid   = param_grid,
            order_amount = order_amount,
            dtype        = dtype,
            timeframe    = timeframe,
        )
    finally:
        backtest_module.BACKTEST_N_JOBS = original_n_jobs

    return compare_dsr_vs_stepm_from_raw(
        raw_results     = raw_results,
        dsr_th          = dsr_th,
        n_combos        = n_combos,
        stepm_alpha     = stepm_alpha,
        stepm_pvalue_th = stepm_pvalue_th,
        n_bootstrap     = n_bootstrap,
        block_size      = block_size,
        timeframe       = timeframe,
        n_jobs          = n_jobs,
    )


# =============================================================================
# REPORTING
# =============================================================================
def _print_comparison(raw_results: list, dsr_by_id: dict, stepm_by_id: dict, timeframe: str) -> None:

    n_total        = len(raw_results)
    n_dsr_passed   = 0
    n_stepm_passed = 0
    n_agreement    = 0
    n_both_passed  = 0
    rows           = []
    dsr_vals, stepm_vals = [], []

    for r in raw_results:
        rid      = r["rule_id"]
        dsr_r    = dsr_by_id.get(rid, {})
        stepm_r  = stepm_by_id.get(rid, {})

        dsr_val  = dsr_r.get("dsr", 0.0)
        dsr_ok   = bool(dsr_r.get("passed_dsr", False))
        stepm_p  = stepm_r.get("stepm_p")
        stepm_ok = bool(stepm_r.get("passed_stepm", False))

        n_dsr_passed   += int(dsr_ok)
        n_stepm_passed += int(stepm_ok)
        n_agreement    += int(dsr_ok == stepm_ok)
        n_both_passed  += int(dsr_ok and stepm_ok)

        if stepm_p is not None:
            rows.append((rid, dsr_val, dsr_ok, stepm_p, stepm_ok))
            dsr_vals.append(dsr_val)
            stepm_vals.append(stepm_p)

    rows.sort(key=lambda row: row[3])  # by StepM p-value, ascending

    logger.info(f"\n{'─' * 70}")
    logger.info(f"  DSR vs STEPM ── {timeframe}")
    logger.info(f"{'─' * 70}")

    id_width = max((len(row[0]) for row in rows), default=8) + 2
    logger.info(f"  top {min(20, len(rows))} by STEPM p-value")
    logger.info(f"{'RULE_ID':<{id_width}}{'DSR':<10}{'DSR_OK':<10}{'STEPM_p':<10}{'STEPM_OK':<10}")
    logger.info(f"{'─' * 70}")
    for rule_id, dsr_val, dsr_ok, stepm_p, stepm_ok in rows[:10]:
        dsr_mark   = "✅" if dsr_ok else "❌"
        stepm_mark = "✅" if stepm_ok else "❌"
        logger.info(f"{rule_id:<{id_width}}{dsr_val:<10.4f}{dsr_mark:<10}{stepm_p:<10.4f}{stepm_mark:<10}")

    _print_correlation(dsr_vals, stepm_vals, timeframe)

    pct_dsr       = n_dsr_passed   / n_total * 100.0 if n_total else 0.0
    pct_stepm     = n_stepm_passed / n_total * 100.0 if n_total else 0.0
    pct_agreement = n_agreement    / n_total * 100.0 if n_total else 0.0
    pct_both      = n_both_passed  / n_total * 100.0 if n_total else 0.0

    logger.info(f"{'─' * 70}")
    logger.info(f"  DSR        ── {n_dsr_passed}/{n_total} passed ({pct_dsr:.2f}%)")
    logger.info(f"  STEPM      ── {n_stepm_passed}/{n_total} passed ({pct_stepm:.2f}%)")
    logger.info(f"  AGREEMENT  ── {n_agreement}/{n_total} rules match ({pct_agreement:.2f}%)")
    logger.info(f"  OK-OK      ── {n_both_passed}/{n_total} both pass ({pct_both:.2f}%)")
    logger.info(f"{'─' * 70}\n")


def _print_correlation(dsr_vals: list, stepm_vals: list, timeframe: str) -> None:
    if len(dsr_vals) < 3:
        logger.warning(f"COMPARE ── {timeframe} ── not enough rules to compute correlation")
        return

    dsr_arr, stepm_arr = np.asarray(dsr_vals), np.asarray(stepm_vals)

    logger.info(f"{'─' * 70}")
    logger.info(f"  DSR vs STEPM_p CORRELATION ── {timeframe}")

    pearson_r, pearson_p   = pearsonr(dsr_arr, stepm_arr)
    spearman_r, spearman_p = spearmanr(dsr_arr, stepm_arr)
    logger.info(f"  [ALL RULES]              n={len(dsr_vals)}")
    logger.info(f"    Pearson  r = {pearson_r:.4f}  (p={pearson_p:.4g})")
    logger.info(f"    Spearman r = {spearman_r:.4f}  (p={spearman_p:.4g})")

    # Second cut: excluding p=1.0 (the saturated tail, indistinguishable
    # from noise by construction) — this is where the real ranking signal,
    # if any, tends to concentrate.
    not_saturated = stepm_arr < 1.0
    n_not_saturated = int(not_saturated.sum())
    if n_not_saturated >= 3:
        pearson_r2, pearson_p2   = pearsonr(dsr_arr[not_saturated], stepm_arr[not_saturated])
        spearman_r2, spearman_p2 = spearmanr(dsr_arr[not_saturated], stepm_arr[not_saturated])
        logger.info(f"  [EXCLUDING STEPM_p=1.0]  n={n_not_saturated}")
        logger.info(f"    Pearson  r = {pearson_r2:.4f}  (p={pearson_p2:.4g})")
        logger.info(f"    Spearman r = {spearman_r2:.4f}  (p={spearman_p2:.4g})")
    else:
        logger.info(f"  [EXCLUDING STEPM_p=1.0]  not enough rules (n={n_not_saturated})")

    logger.info(f"{'─' * 70}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    start = time.time()

    logger.info(f"\n{'─' * 115}")
    logger.info(f"  DSR vs STEPM — FULL BRUTE UNIVERSE COMPARISON")
    logger.info(f"{'─' * 115}")
    logger.info(f"  TIMEFRAMES     : {TIMEFRAMES}")
    logger.info(f"  N_SYMBOLS      : {N_SYMBOLS}")
    logger.debug(f"  MAX_DEPTH      : {RULE_MAX_DEPTH}")
    logger.info(f"  PARAM_GRID     : {PARAM_GRID}")
    logger.info(f"  DSR_TH         : {DSR_TH}")
    logger.info(f"  STEPM_ALPHA    : {STEPM_ALPHA_RUN}")
    logger.info(f"  STEPM_PVALUE_TH: {STEPM_PVALUE_TH_RUN}")
    logger.info(f"  N_BOOTSTRAP    : {STEPM_N_BOOTSTRAP}")
    logger.info(f"  BLOCK_SIZE     : {STEPM_BLOCK_SIZE}")
    logger.info(f"{'─' * 115}\n")

    # -------------------------------------------------------------------
    # DATA LOADING — cheap, sequential across timeframes.
    # -------------------------------------------------------------------
    ohlcv_data_by_timeframe = {}
    ohlcv_arr_by_timeframe  = {}

    for timeframe in TIMEFRAMES:
        ohlcv_is = select_universe(
            data_folder_is    = DATA_FOLDER_IS,
            timeframe         = timeframe,
            min_price         = MIN_PRICE,
            filter_symbols_fn = filter_symbols,
        )
        ohlcv_data_by_timeframe[timeframe] = ohlcv_is
        ohlcv_arr_by_timeframe[timeframe]  = prepare_ohlcv_arrays(ohlcv_is)

    # -------------------------------------------------------------------
    # DSR vs STEPM — one comparison per timeframe, on the full brute
    # universe (neither pipe pre-filters what the other sees).
    # -------------------------------------------------------------------
    comparisons_by_timeframe = {}

    for timeframe in TIMEFRAMES:
        rules_for_timeframe = _build_rule_dicts(
            ohlcv_data_by_timeframe[timeframe], timeframe, RULE_MAX_DEPTH,
        )

        comparisons_by_timeframe[timeframe] = compare_dsr_vs_stepm(
            rules           = rules_for_timeframe,
            ohlcv_arr       = ohlcv_arr_by_timeframe[timeframe],
            param_grid      = PARAM_GRID,
            order_amount    = ORDER_AMOUNT,
            dtype           = DTYPE,
            dsr_th          = DSR_TH,
            stepm_alpha     = STEPM_ALPHA_RUN,
            stepm_pvalue_th = STEPM_PVALUE_TH_RUN,
            n_bootstrap     = STEPM_N_BOOTSTRAP,
            block_size      = STEPM_BLOCK_SIZE,
            timeframe       = timeframe,
            n_jobs          = N_JOBS,
        )

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")