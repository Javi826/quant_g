#BOT_batch/main_NULL.py
"""
DSR vs StepM — null calibration.
Bars are permuted per symbol to destroy any real predictive structure before
rule generation and backtesting. Under this null, any rule that "passes" is
by construction a false positive. Reuses pipe_dsr / pipe_stepm unmodified.
"""
import os
import sys
import time
import logging
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_batch")))

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL = logging.INFO
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_batch.main_null")
logger.setLevel(LOG_LEVEL)

logging.getLogger("BOT_batch.pipeline.dsr").setLevel(logging.WARNING)
logging.getLogger("BOT_batch.pipeline.stepm").setLevel(logging.WARNING)
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
from shared_batchs.pipeline import stepm as stepm_module
from shared_batchs.pipeline.stepm import pipe_stepm

# =============================================================================
# UNIVERSE / SEARCH SPACE CONFIGURATION — mirrors main_COMP.py
# =============================================================================
DTYPE  = np.float32
N_JOBS = -1

TIMEFRAMES = ["12Hutc"]
N_SYMBOLS  = 10

PARAM_GRID = {
    "SELL_AFTER": [50],
    "TP_PCT":     [2, 4, 6, 8, 10],
    "SL_PCT":     [2, 4, 6, 8, 10],
}

DSR_TH = 0.80

# =============================================================================
# NULL CALIBRATION CONFIG
# =============================================================================
N_NULL_RUNS      = 10     # independent null realizations per timeframe
PERM_BLOCK_SIZE  = 300      # 1 = full iid bar shuffle (strongest null); >1 = block shuffle
PERM_SEED        = 12345
SHARE_PERMUTATION_ACROSS_SYMBOLS = True  # True = same block_order for all symbols (preserves real cross-symbol comovement); False = independent per symbol (destroys it too)

STEPM_K_MODE_NULL       = "percentile"  # "absolute" (k=1, proven guarantee) or "percentile" (production config, unproven for multi-step)
STEPM_K_FWE_NULL        = 1             # used only when STEPM_K_MODE_NULL == "absolute"
STEPM_K_PERCENTILE_NULL = 0.001         # used only when STEPM_K_MODE_NULL == "percentile" — mirrors production STEPM_K_PERCENTILE

# =============================================================================
# BAR PERMUTATION — destroys temporal structure, keeps each bar internally
# consistent (open/high/low/close/volume moved together, never mixed).
# =============================================================================
def _permute_ohlcv(ohlcv_is: dict, block_size: int, rng: np.random.Generator) -> dict:
    permuted = {}
    shared_row_order = None
    for symbol, df in ohlcv_is.items():
        n_obs = len(df)
        n_blocks = int(np.ceil(n_obs / block_size))

        if SHARE_PERMUTATION_ACROSS_SYMBOLS and shared_row_order is not None:
            row_order = shared_row_order
        else:
            block_order = rng.permutation(n_blocks)
            row_order = np.empty(n_obs, dtype=np.int64)
            pos = 0
            for block_idx in block_order:
                start = block_idx * block_size
                end = min(start + block_size, n_obs)
                length = end - start
                row_order[pos:pos + length] = np.arange(start, end)
                pos += length
            if SHARE_PERMUTATION_ACROSS_SYMBOLS:
                shared_row_order = row_order

        # Rows (open/high/low/close/volume/low_time/high_time) move together;
        # the original index (timestamp sequence) is kept intact.
        permuted_df = df.iloc[row_order].copy()
        permuted_df.index = df.index
        permuted[symbol] = permuted_df
    return permuted

# =============================================================================
# ONE NULL REALIZATION — permute, rebuild rules, backtest, DSR + StepM(k=1)
# =============================================================================
def _run_one_null_iteration(
    ohlcv_is: dict,
    timeframe: str,
    seed: int,
) -> dict:

    rng = np.random.default_rng(seed)
    ohlcv_is_null = _permute_ohlcv(ohlcv_is, PERM_BLOCK_SIZE, rng)
    ohlcv_arr_null = prepare_ohlcv_arrays(ohlcv_is_null)

    rules = _build_rule_dicts(ohlcv_is_null, timeframe, RULE_MAX_DEPTH)

    original_n_jobs = backtest_module.BACKTEST_N_JOBS
    backtest_module.BACKTEST_N_JOBS = N_JOBS
    try:
        raw_results, n_combos, matrix_arr, col_names = backtest_module.pipe_backtesting(
            rules        = rules,
            ohlcv_arr    = ohlcv_arr_null,
            param_grid   = PARAM_GRID,
            order_amount = ORDER_AMOUNT,
            dtype        = DTYPE,
            timeframe    = timeframe,
        )
    finally:
        backtest_module.BACKTEST_N_JOBS = original_n_jobs

    dsr_results = pipe_dsr(
        raw_results = raw_results,
        matrix_arr  = matrix_arr,
        dsr_th      = DSR_TH,
        n_combos    = n_combos,
        timeframe   = timeframe,
    )

    original_k_mode = stepm_module.STEPM_K_MODE
    original_k_fwe  = stepm_module.STEPM_K_FWE
    stepm_module.STEPM_K_MODE = STEPM_K_MODE_NULL
    if STEPM_K_MODE_NULL == "absolute":
        stepm_module.STEPM_K_FWE = STEPM_K_FWE_NULL
    try:
        stepm_results = pipe_stepm(
            raw_results        = raw_results,
            matrix_arr         = matrix_arr,
            col_names          = col_names,
            stepm_k_percentile = STEPM_K_PERCENTILE_NULL if STEPM_K_MODE_NULL == "percentile" else None,
            timeframe          = timeframe,
        )
    finally:
        stepm_module.STEPM_K_MODE = original_k_mode
        stepm_module.STEPM_K_FWE  = original_k_fwe

    dsr_pass_ids   = {r["rule_id"] for r in dsr_results   if r["passed_dsr"]}
    stepm_pass_ids = {r["rule_id"] for r in stepm_results if r["passed_stepm"]}

    return {
        "n_total":      len(raw_results),
        "n_dsr_pass":   len(dsr_pass_ids),
        "n_stepm_pass": len(stepm_pass_ids),
        "n_overlap":    len(dsr_pass_ids & stepm_pass_ids),
        "n_union":      len(dsr_pass_ids | stepm_pass_ids),
    }

# =============================================================================
# REPORTING
# =============================================================================
def _print_null_summary(timeframe: str, iterations: list) -> None:
    n_total = iterations[0]["n_total"]

    dsr_pass   = [it["n_dsr_pass"]   for it in iterations]
    stepm_pass = [it["n_stepm_pass"] for it in iterations]
    overlap    = [it["n_overlap"]    for it in iterations]
    jaccard    = [
        it["n_overlap"] / it["n_union"] if it["n_union"] > 0 else 0.0
        for it in iterations
    ]

    logger.info(f"\n{'─' * 70}")
    logger.info(f"  NULL CALIBRATION ── {timeframe} ── n_runs={len(iterations)} ── n_total={n_total}")
    logger.info(f"{'─' * 70}")
    logger.info(f"  DSR   passed   ── mean={np.mean(dsr_pass):.2f}   ({[f'{x/n_total:.4%}' for x in dsr_pass]})")
    logger.info(f"  STEPM passed   ── mean={np.mean(stepm_pass):.2f} ({[f'{x/n_total:.4%}' for x in stepm_pass]})")
    logger.info(f"  Overlap        ── mean={np.mean(overlap):.2f}")
    logger.info(f"  Jaccard        ── mean={np.mean(jaccard):.4f}")
    logger.info(f"{'─' * 70}")
    logger.info(f"  Expected under nominal FWE control (StepM, k=1): ~{0.05 * n_total:.2f} false positives at most (alpha=0.05)")
    logger.info(f"{'─' * 70}\n")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    start = time.time()

    logger.info(f"\n{'─' * 115}")
    logger.info(f"  DSR vs STEPM — NULL CALIBRATION (permuted bars, no real edge by construction)")
    logger.info(f"{'─' * 115}")
    logger.info(f"  TIMEFRAMES        : {TIMEFRAMES}")
    logger.info(f"  N_SYMBOLS         : {N_SYMBOLS}")
    logger.info(f"  PARAM_GRID        : {PARAM_GRID}")
    logger.info(f"  DSR_TH            : {DSR_TH}")
    logger.info(f"  N_NULL_RUNS       : {N_NULL_RUNS}")
    logger.info(f"  PERM_BLOCK_SIZE   : {PERM_BLOCK_SIZE}")
    logger.info(f"  SHARE_PERM_ACROSS_SYMBOLS : {SHARE_PERMUTATION_ACROSS_SYMBOLS}")
    if STEPM_K_MODE_NULL == "absolute":
        logger.info(f"  STEPM_K_MODE_NULL : {STEPM_K_MODE_NULL} (k={STEPM_K_FWE_NULL})")
    else:
        logger.info(f"  STEPM_K_MODE_NULL : {STEPM_K_MODE_NULL} (k_percentile={STEPM_K_PERCENTILE_NULL})")
    logger.info(f"{'─' * 115}\n")

    for timeframe in TIMEFRAMES:
        ohlcv_is = select_universe(
            data_folder_is    = DATA_FOLDER_IS,
            timeframe         = timeframe,
            min_price         = MIN_PRICE,
            filter_symbols_fn = filter_symbols,
        )

        iterations = []
        for run_idx in range(N_NULL_RUNS):
            seed = PERM_SEED + run_idx
            logger.info(f"NULL ── {timeframe} ── run {run_idx + 1}/{N_NULL_RUNS} ── seed={seed}")
            iterations.append(_run_one_null_iteration(ohlcv_is, timeframe, seed))

        _print_null_summary(timeframe, iterations)

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")