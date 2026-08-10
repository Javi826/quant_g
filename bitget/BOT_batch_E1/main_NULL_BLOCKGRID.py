#BOT_batch/main_NULL_BLOCKGRID.py

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
logger = logging.getLogger("BOT_batch.main_null_blockgrid")
logger.setLevel(LOG_LEVEL)

logging.getLogger("BOT_batch.pipeline.stepM").setLevel(logging.WARNING)
logging.getLogger("joblib").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from shared_batchs.symbols.universe import filter_symbols, select_universe
from shared_batchs.setup.config_paths import DATA_FOLDER_IS
from shared_batchs.rule_mining.rule_generator import MAX_DEPTH as RULE_MAX_DEPTH
from shared_batchs.rule_mining.rule_runner import _build_rule_dicts
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_batchs.setup.config_backtest import MIN_PRICE, ORDER_AMOUNT
from shared_batchs.pipeline import backtest_runner as backtest_module
from shared_batchs.pipeline import stepm as stepm_module
from shared_batchs.pipeline.stepM import pipe_stepm, RANDOM_SEED

# =============================================================================
# UNIVERSE / SEARCH SPACE CONFIGURATION — mirrors main_COMP.py
# =============================================================================
DTYPE  = np.float32
N_JOBS = -1

TIMEFRAMES = ["12Hutc"]
N_SYMBOLS  = 10

PARAM_GRID = {
    "SELL_AFTER": [50],
    "TP_PCT":     [2,4,6,8,10],
    "SL_PCT":     [2,4,6,8,10],
}

# =============================================================================
# GRID CONFIG
# =============================================================================
B_TRUE_GRID    = [5,10,20,50,100,150]   # dependence range of the simulated world
B_ASSUMED_GRID = [5,10,20,50,100,150]   # block size StepM uses for its critical values

N_RUNS_PER_CELL = 100
GRID_SEED       = 777

# The grid costs len(B_TRUE_GRID) * len(B_ASSUMED_GRID) * N_RUNS_PER_CELL StepM
# calls, so the universe is capped here. Absolute rates do not transfer to the
# full universe, but the relative sensitivity to block size does.
NULL_MAX_COLUMNS = 20000

# k = 1 keeps the criterion unambiguous: any rejection is a violation. Grid
# results for k > 1 would confound block-size effects with the k-FWER relaxation.
STEPM_ALPHA_NULL = 0.05
STEPM_K_FWE_NULL = 1

# =============================================================================
# NULL POPULATION — demean every column so theta_s = 0 by construction while
# leaving the dependence structure of the real data intact.
# =============================================================================
def _build_null_population(matrix_arr: np.ndarray, col_names: list, chunk_size: int = 5000) -> tuple:
    if NULL_MAX_COLUMNS is not None and matrix_arr.shape[1] > NULL_MAX_COLUMNS:
        rng = np.random.default_rng(GRID_SEED)
        keep = np.sort(rng.choice(matrix_arr.shape[1], size=NULL_MAX_COLUMNS, replace=False))
        matrix_arr = np.ascontiguousarray(matrix_arr[:, keep])
        col_names = [col_names[i] for i in keep]

    n_cols = matrix_arr.shape[1]
    for start in range(0, n_cols, chunk_size):
        end = min(start + chunk_size, n_cols)
        chunk = matrix_arr[:, start:end]
        chunk -= chunk.mean(axis=0, dtype=np.float64).astype(matrix_arr.dtype)[None, :]

    return matrix_arr, col_names

# =============================================================================
# OUTER DRAW — circular block bootstrap over rows, block length b_true.
# =============================================================================
def _circular_block_resample(population: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    n_obs = population.shape[0]
    n_blocks = int(np.ceil(n_obs / block_size))

    starts = rng.integers(0, n_obs, size=n_blocks, dtype=np.int64)
    row_idx = (starts[:, None] + np.arange(block_size, dtype=np.int64)[None, :]).ravel()[:n_obs] % n_obs

    return np.ascontiguousarray(population[row_idx])

# =============================================================================
# ONE GRID CELL — N runs at a fixed (b_true, b_assumed) pair
# =============================================================================
def _run_cell(
    population: np.ndarray,
    col_names: list,
    raw_results: list,
    b_true: int,
    b_assumed: int,
    timeframe: str,
) -> dict:

    original_k_mode = stepm_module.STEPM_K_MODE
    original_k_fwe  = stepm_module.STEPM_K_FWE
    stepm_module.STEPM_K_MODE = "absolute"
    stepm_module.STEPM_K_FWE  = STEPM_K_FWE_NULL

    rejections = np.empty(N_RUNS_PER_CELL, dtype=np.int64)
    try:
        for run_idx in range(N_RUNS_PER_CELL):
            rng = np.random.default_rng(GRID_SEED + 1000 * b_true + run_idx)
            sample_arr = _circular_block_resample(population, b_true, rng)

            stepm_results = pipe_stepm(
                raw_results = raw_results,
                matrix_arr  = sample_arr,
                col_names   = col_names,
                stepm_alpha = STEPM_ALPHA_NULL,
                block_size  = b_assumed,
                seed        = RANDOM_SEED + run_idx,
                timeframe   = timeframe,
            )
            rejections[run_idx] = sum(1 for r in stepm_results if r["passed_stepm"])
    finally:
        stepm_module.STEPM_K_MODE = original_k_mode
        stepm_module.STEPM_K_FWE  = original_k_fwe

    n_violations = int((rejections >= STEPM_K_FWE_NULL).sum())

    return {
        "b_true":       b_true,
        "b_assumed":    b_assumed,
        "n_violations": n_violations,
        "rate":         n_violations / N_RUNS_PER_CELL,
        "median_rej":   float(np.median(rejections)),
        "max_rej":      int(rejections.max()),
    }

# =============================================================================
# REPORTING
# =============================================================================
def _wilson_interval(n_success: int, n_trials: int, z: float = 1.96) -> tuple:
    """Wilson score interval — well behaved near 0, unlike the normal approximation."""
    if n_trials == 0:
        return 0.0, 1.0
    p_hat  = n_success / n_trials
    denom  = 1.0 + z ** 2 / n_trials
    center = (p_hat + z ** 2 / (2 * n_trials)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n_trials + z ** 2 / (4 * n_trials ** 2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _print_grid(timeframe: str, cells: dict, n_cols: int) -> None:
    cell_width = 14

    logger.info(f"\n{'─' * 100}")
    logger.info(f"  BLOCK SIZE ROBUSTNESS GRID ── {timeframe}")
    logger.info(f"{'─' * 100}")
    logger.info(f"  columns tested   : {n_cols}")
    logger.info(f"  runs per cell    : {N_RUNS_PER_CELL}")
    logger.info(f"  criterion        : empirical FWER (k={STEPM_K_FWE_NULL}) vs alpha={STEPM_ALPHA_NULL}")
    logger.info(f"  rows = b_true (simulated dependence) ── cols = b_assumed (StepM's block size)")
    logger.info(f"{'─' * 100}")

    header = f"  {'b_true':<10}" + "".join(f"{f'b_asm={b}':<{cell_width}}" for b in B_ASSUMED_GRID)
    logger.info(header)
    logger.info(f"{'─' * 100}")

    for b_true in B_TRUE_GRID:
        row = f"  {b_true:<10}"
        for b_assumed in B_ASSUMED_GRID:
            cell = cells[(b_true, b_assumed)]
            mark = "" if cell["rate"] <= STEPM_ALPHA_NULL else " *"
            row += f"{cell['rate']:.2f}{mark:<{cell_width - 4}}"
        logger.info(row)

    logger.info(f"{'─' * 100}")
    logger.info(f"  (* marks cells whose empirical rate exceeds nominal alpha)")
    logger.info(f"{'─' * 100}\n")

    logger.info(f"  median / max rejections per cell")
    logger.info(f"{'─' * 100}")
    logger.info(f"  {'b_true':<10}" + "".join(f"{f'b_asm={b}':<{cell_width}}" for b in B_ASSUMED_GRID))
    logger.info(f"{'─' * 100}")
    for b_true in B_TRUE_GRID:
        row = f"  {b_true:<10}"
        for b_assumed in B_ASSUMED_GRID:
            cell = cells[(b_true, b_assumed)]
            label = f"{cell['median_rej']:.0f}/{cell['max_rej']}"
            row += f"{label:<{cell_width}}"
        logger.info(row)
    logger.info(f"{'─' * 100}\n")

    logger.info(f"  VERDICT PER CANDIDATE b_assumed (worst case across all b_true)")
    logger.info(f"{'─' * 100}")
    for b_assumed in B_ASSUMED_GRID:
        rates = [cells[(b_true, b_assumed)]["rate"] for b_true in B_TRUE_GRID]
        worst_idx = int(np.argmax(rates))
        worst_b_true = B_TRUE_GRID[worst_idx]
        worst_cell = cells[(worst_b_true, b_assumed)]
        lo, hi = _wilson_interval(worst_cell["n_violations"], N_RUNS_PER_CELL)
        status = "ROBUST" if lo <= STEPM_ALPHA_NULL else "NOT ROBUST"
        logger.info(
            f"  b_assumed={b_assumed:<5} worst rate={max(rates):.2f} at b_true={worst_b_true:<5} "
            f"95% CI [{lo:.3f}, {hi:.3f}] ── {status}"
        )
    logger.info(f"{'─' * 100}\n")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    start = time.time()

    n_cells = len(B_TRUE_GRID) * len(B_ASSUMED_GRID)

    logger.info(f"\n{'─' * 115}")
    logger.info(f"  STEPM BLOCK SIZE ROBUSTNESS — outer draw decoupled from inner bootstrap")
    logger.info(f"{'─' * 115}")
    logger.info(f"  TIMEFRAMES        : {TIMEFRAMES}")
    logger.info(f"  PARAM_GRID        : {PARAM_GRID}")
    logger.info(f"  B_TRUE_GRID       : {B_TRUE_GRID}")
    logger.info(f"  B_ASSUMED_GRID    : {B_ASSUMED_GRID}")
    logger.info(f"  N_RUNS_PER_CELL   : {N_RUNS_PER_CELL}")
    logger.info(f"  TOTAL STEPM CALLS : {n_cells * N_RUNS_PER_CELL}")
    logger.info(f"  NULL_MAX_COLUMNS  : {NULL_MAX_COLUMNS}")
    logger.info(f"  STEPM_ALPHA_NULL  : {STEPM_ALPHA_NULL}  (k={STEPM_K_FWE_NULL})")
    logger.info(f"{'─' * 115}\n")

    for timeframe in TIMEFRAMES:
        # -------------------------------------------------------------------
        # SINGLE BACKTEST — the real PnL matrix is the basis of the null
        # population, so the expensive stages run once per timeframe.
        # -------------------------------------------------------------------
        ohlcv_is = select_universe(
            data_folder_is    = DATA_FOLDER_IS,
            timeframe         = timeframe,
            min_price         = MIN_PRICE,
            filter_symbols_fn = filter_symbols,
        )
        ohlcv_arr = prepare_ohlcv_arrays(ohlcv_is)
        rules = _build_rule_dicts(ohlcv_is, timeframe, RULE_MAX_DEPTH)

        original_n_jobs = backtest_module.BACKTEST_N_JOBS
        backtest_module.BACKTEST_N_JOBS = N_JOBS
        try:
            raw_results, n_combos, matrix_arr, col_names = backtest_module.pipe_backtesting(
                rules        = rules,
                ohlcv_arr    = ohlcv_arr,
                param_grid   = PARAM_GRID,
                order_amount = ORDER_AMOUNT,
                dtype        = DTYPE,
                timeframe    = timeframe,
            )
        finally:
            backtest_module.BACKTEST_N_JOBS = original_n_jobs

        population, col_names = _build_null_population(matrix_arr, col_names)
        logger.info(
            f"GRID ── {timeframe} ── population built ── {population.shape[0]} obs x "
            f"{population.shape[1]} columns ── max |column mean| = "
            f"{np.abs(population.mean(axis=0, dtype=np.float64)).max():.3e}\n"
        )

        # -------------------------------------------------------------------
        # GRID SWEEP
        # -------------------------------------------------------------------
        cells = {}
        cell_idx = 0
        for b_true in B_TRUE_GRID:
            for b_assumed in B_ASSUMED_GRID:
                cell_idx += 1
                cell_start = time.time()
                cells[(b_true, b_assumed)] = _run_cell(
                    population, col_names, raw_results, b_true, b_assumed, timeframe,
                )
                cell = cells[(b_true, b_assumed)]
                logger.info(
                    f"GRID ── cell {cell_idx}/{n_cells} ── b_true={b_true} b_assumed={b_assumed} ── "
                    f"rate={cell['rate']:.2f} ── median/max rej={cell['median_rej']:.0f}/{cell['max_rej']} ── "
                    f"{int(time.time() - cell_start)}s"
                )

        _print_grid(timeframe, cells, population.shape[1])

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")