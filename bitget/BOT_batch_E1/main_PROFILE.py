#BOT_batch/main_PROFILE.py
"""
Cross-sectional profile test — where, if anywhere, does the observed universe
depart from the data-snooping null?

White's Reality Check compares the SINGLE LARGEST studentized statistic against
the bootstrap distribution of the maximum. That is a detector for sparse
alternatives: powerful when one or a few strong signals hide in noise, blind to
a diffuse shift spread across many moderate ones.

This script reuses the exact same bootstrap output and reads a whole profile of
order statistics off it: the q-th largest observed z against the bootstrap
distribution of the q-th largest, for q spanning several orders of magnitude,
plus the mean, the median and the fraction of positive z. The q=1 row
reproduces the standard global White p-value and serves as a sanity check.

The point is diagnostic. A profile that departs from the null only at q=1 means
one lucky rule. A profile that departs across the body means a diffuse shift.
A profile flat against the null everywhere means there is nothing to find.

TWO THINGS THIS TEST DOES NOT DO
  1. It does not control any familywise criterion across the profile. Each row
     is a marginal p-value; reading the smallest of them as if it were the
     result of a single pre-registered test is exactly the snooping this whole
     pipeline exists to avoid. Decide which row matters BEFORE looking.
  2. It does not neutralize market exposure. The null is theta <= 0 against a
     benchmark fixed at zero, so rejecting means "the average rule beats cash",
     not "there is edge beyond long exposure". For long-only entries in a
     trending market those are very different claims, and separating them needs
     a random-entry benchmark, not this script.
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
logger = logging.getLogger("BOT_batch.main_profile")
logger.setLevel(LOG_LEVEL)

logging.getLogger("BOT_batch.pipeline.stepM").setLevel(logging.INFO)
logging.getLogger("BOT_batch.pipeline.backtest_runner").setLevel(logging.INFO)
logging.getLogger("joblib").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from shared_batchs.symbols.universe import filter_symbols, select_universe
from shared_batchs.setup.config_paths import DATA_FOLDER_IS
from shared_batchs.rule_mining.rule_generator import MAX_DEPTH as RULE_MAX_DEPTH
from shared_batchs.rule_mining.rule_runner import _build_rule_dicts
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_batchs.setup.config_backtest import MIN_PRICE, ORDER_AMOUNT
from shared_batchs.pipeline import backtest_runner as backtest_module
from shared_batchs.pipeline.stepM import (
    compute_deviation_matrix,
    compute_global_pvalue,
    WHITE_BLOCK_SIZE,
    WHITE_N_BOOTSTRAP,
    RANDOM_SEED,
)

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

# =============================================================================
# PROFILE CONFIG
# =============================================================================
# Block size is the one variable this whole line of investigation showed the
# result is sensitive to, so it is explicit here rather than inherited silently.
PROFILE_BLOCK_SIZE  = WHITE_BLOCK_SIZE
PROFILE_N_BOOTSTRAP = WHITE_N_BOOTSTRAP
PROFILE_SEED        = RANDOM_SEED

# Ranks of the order statistics to profile, from the top down. Any rank beyond
# the surviving column count is dropped automatically.
RANK_GRID = [1, 5, 10, 50, 100, 500, 1000, 5000, 10000, 50000]

REPLICA_ROW_CHUNK = 50     # bootstrap replicas per partition call (memory bound)
COLUMN_CHUNK_SIZE = 5000   # columns per chunk for the finiteness sweep


# =============================================================================
# FINITENESS SWEEP — guards against -inf leaking in from zero-variance
# bootstrap replicas before any order statistic is taken.
# =============================================================================
def finite_column_mask(deviations: np.ndarray, chunk_size: int = COLUMN_CHUNK_SIZE) -> np.ndarray:
    n_cols = deviations.shape[1]
    mask   = np.ones(n_cols, dtype=bool)
    for start in range(0, n_cols, chunk_size):
        end = min(start + chunk_size, n_cols)
        mask[start:end] = np.isfinite(deviations[:, start:end]).all(axis=0)
    return mask


# =============================================================================
# PROFILE STATISTICS — one shared definition, applied to the observed vector
# and to every bootstrap replica so the comparison is apples to apples.
# =============================================================================
def _partition_indices(n_cols: int, ranks: list) -> tuple:
    """Descending rank q maps to ascending partition index n_cols - q."""
    usable   = [q for q in ranks if 1 <= q <= n_cols]
    indices  = [n_cols - q for q in usable]
    median_idx = n_cols // 2
    kth      = sorted(set(indices + [median_idx]))
    return usable, indices, median_idx, kth


def observed_profile(z_stat: np.ndarray, ranks: list) -> dict:
    n_cols = z_stat.size
    usable, indices, median_idx, kth = _partition_indices(n_cols, ranks)

    partitioned = np.partition(z_stat, kth)
    return {
        "ranks":         usable,
        "order_stats":   np.array([partitioned[i] for i in indices], dtype=np.float64),
        "mean":          float(z_stat.mean(dtype=np.float64)),
        "median":        float(partitioned[median_idx]),
        "fraction_pos":  float((z_stat > 0).mean()),
    }


def bootstrap_profile(deviations: np.ndarray, ranks: list,
                      row_chunk: int = REPLICA_ROW_CHUNK) -> dict:
    """
    Same statistics, computed per bootstrap replica. Returns one array of length
    n_bootstrap per statistic, which is the null distribution to compare against.
    """
    n_bootstrap, n_cols = deviations.shape
    usable, indices, median_idx, kth = _partition_indices(n_cols, ranks)

    order_stats  = np.empty((n_bootstrap, len(usable)), dtype=np.float64)
    mean_by_rep  = np.empty(n_bootstrap, dtype=np.float64)
    median_by_rep = np.empty(n_bootstrap, dtype=np.float64)
    frac_by_rep  = np.empty(n_bootstrap, dtype=np.float64)

    for start in range(0, n_bootstrap, row_chunk):
        end   = min(start + row_chunk, n_bootstrap)
        chunk = deviations[start:end]

        partitioned = np.partition(chunk, kth, axis=1)
        for col_idx, part_idx in enumerate(indices):
            order_stats[start:end, col_idx] = partitioned[:, part_idx]

        median_by_rep[start:end] = partitioned[:, median_idx]
        mean_by_rep[start:end]   = chunk.mean(axis=1, dtype=np.float64)
        frac_by_rep[start:end]   = (chunk > 0).mean(axis=1)

    return {
        "ranks":        usable,
        "order_stats":  order_stats,
        "mean":         mean_by_rep,
        "median":       median_by_rep,
        "fraction_pos": frac_by_rep,
    }


def profile_pvalues(observed: dict, bootstrap: dict) -> dict:
    """One-sided p-value: how often the null reproduces something this extreme."""
    order_p = np.array([
        float(np.mean(bootstrap["order_stats"][:, i] >= observed["order_stats"][i]))
        for i in range(len(observed["ranks"]))
    ])
    return {
        "ranks":        observed["ranks"],
        "order_p":      order_p,
        "mean_p":       float(np.mean(bootstrap["mean"] >= observed["mean"])),
        "median_p":     float(np.mean(bootstrap["median"] >= observed["median"])),
        "fraction_p":   float(np.mean(bootstrap["fraction_pos"] >= observed["fraction_pos"])),
    }


# =============================================================================
# REPORTING
# =============================================================================
def _print_profile(timeframe: str, observed: dict, bootstrap: dict, pvalues: dict,
                   n_cols: int, global_p: float) -> None:
    logger.info(f"\n{'─' * 100}")
    logger.info(f"  CROSS-SECTIONAL PROFILE ── {timeframe}")
    logger.info(f"{'─' * 100}")
    logger.info(f"  columns tested   : {n_cols}")
    logger.info(f"  block size       : {PROFILE_BLOCK_SIZE}")
    logger.info(f"  bootstrap        : {PROFILE_N_BOOTSTRAP} replicas")
    logger.info(f"{'─' * 100}")
    logger.info(f"  {'RANK q':<12}{'OBSERVED z':<16}{'NULL MEAN':<16}{'NULL P95':<16}{'P-VALUE':<12}")
    logger.info(f"{'─' * 100}")

    for i, rank in enumerate(pvalues["ranks"]):
        observed_value = observed["order_stats"][i]
        null_column    = bootstrap["order_stats"][:, i]
        logger.info(
            f"  {rank:<12}{observed_value:<16.4f}{null_column.mean():<16.4f}"
            f"{np.quantile(null_column, 0.95):<16.4f}{pvalues['order_p'][i]:<12.4f}"
        )

    logger.info(f"{'─' * 100}")
    logger.info(f"  {'STATISTIC':<12}{'OBSERVED':<16}{'NULL MEAN':<16}{'NULL P95':<16}{'P-VALUE':<12}")
    logger.info(f"{'─' * 100}")
    logger.info(
        f"  {'mean z':<12}{observed['mean']:<16.4f}{bootstrap['mean'].mean():<16.4f}"
        f"{np.quantile(bootstrap['mean'], 0.95):<16.4f}{pvalues['mean_p']:<12.4f}"
    )
    logger.info(
        f"  {'median z':<12}{observed['median']:<16.4f}{bootstrap['median'].mean():<16.4f}"
        f"{np.quantile(bootstrap['median'], 0.95):<16.4f}{pvalues['median_p']:<12.4f}"
    )
    logger.info(
        f"  {'frac z>0':<12}{observed['fraction_pos']:<16.4f}{bootstrap['fraction_pos'].mean():<16.4f}"
        f"{np.quantile(bootstrap['fraction_pos'], 0.95):<16.4f}{pvalues['fraction_p']:<12.4f}"
    )
    logger.info(f"{'─' * 100}")
    logger.info(f"  SANITY CHECK ── q=1 p-value should match the global White p-value")
    logger.info(f"    profile q=1     : {pvalues['order_p'][0]:.4f}")
    logger.info(f"    compute_global  : {global_p:.4f}")
    logger.info(f"{'─' * 100}")
    logger.info(f"  Per-row p-values are MARGINAL. No familywise correction is applied")
    logger.info(f"  across the profile, so the smallest one is not a test result.")
    logger.info(f"{'─' * 100}\n")


# =============================================================================
# DRIVER
# =============================================================================
def run_profile(matrix_arr: np.ndarray, col_names: list, timeframe: str, n_jobs: int) -> dict:
    if matrix_arr is None or matrix_arr.shape[1] < 2:
        logger.warning(f"PROFILE ── {timeframe} ── insufficient columns, skipping")
        return {}

    bootstrap_result = compute_deviation_matrix(
        matrix_arr, col_names,
        n_bootstrap    = PROFILE_N_BOOTSTRAP,
        block_size     = PROFILE_BLOCK_SIZE,
        seed           = PROFILE_SEED,
        n_jobs         = n_jobs,
        progress_label = timeframe,
    )

    deviations = bootstrap_result["studentized_deviations"]
    z_stat     = bootstrap_result["z_stat"]

    keep = finite_column_mask(deviations)
    if not keep.all():
        logger.info(
            f"PROFILE ── {timeframe} ── dropping {int((~keep).sum())} columns with "
            f"non-finite studentized deviations"
        )
        deviations = np.ascontiguousarray(deviations[:, keep])
        z_stat     = z_stat[keep]

    n_cols = z_stat.size
    if n_cols < 2:
        logger.warning(f"PROFILE ── {timeframe} ── nothing left after the finiteness sweep, skipping")
        return {}

    global_result = compute_global_pvalue(deviations, z_stat)

    observed  = observed_profile(z_stat, RANK_GRID)
    bootstrap = bootstrap_profile(deviations, RANK_GRID)
    pvalues   = profile_pvalues(observed, bootstrap)

    _print_profile(timeframe, observed, bootstrap, pvalues, n_cols, global_result["global_p"])

    return {"observed": observed, "bootstrap": bootstrap, "pvalues": pvalues, "n_cols": n_cols}


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    start = time.time()

    logger.info(f"\n{'─' * 115}")
    logger.info(f"  CROSS-SECTIONAL PROFILE TEST — order statistics of z against the snooping null")
    logger.info(f"{'─' * 115}")
    logger.info(f"  TIMEFRAMES          : {TIMEFRAMES}")
    logger.info(f"  N_SYMBOLS           : {N_SYMBOLS}")
    logger.info(f"  PARAM_GRID          : {PARAM_GRID}")
    logger.info(f"  PROFILE_BLOCK_SIZE  : {PROFILE_BLOCK_SIZE}")
    logger.info(f"  PROFILE_N_BOOTSTRAP : {PROFILE_N_BOOTSTRAP}")
    logger.info(f"  RANK_GRID           : {RANK_GRID}")
    logger.info(f"{'─' * 115}\n")

    results_by_timeframe = {}

    for timeframe in TIMEFRAMES:
        ohlcv_is  = select_universe(
            data_folder_is    = DATA_FOLDER_IS,
            timeframe         = timeframe,
            min_price         = MIN_PRICE,
            filter_symbols_fn = filter_symbols,
        )
        ohlcv_arr = prepare_ohlcv_arrays(ohlcv_is)
        rules     = _build_rule_dicts(ohlcv_is, timeframe, RULE_MAX_DEPTH)

        original_n_jobs = backtest_module.BACKTEST_N_JOBS
        backtest_module.BACKTEST_N_JOBS = N_JOBS
        try:
            _, _, matrix_arr, col_names = backtest_module.pipe_backtesting(
                rules        = rules,
                ohlcv_arr    = ohlcv_arr,
                param_grid   = PARAM_GRID,
                order_amount = ORDER_AMOUNT,
                dtype        = DTYPE,
                timeframe    = timeframe,
            )
        finally:
            backtest_module.BACKTEST_N_JOBS = original_n_jobs

        results_by_timeframe[timeframe] = run_profile(
            matrix_arr = matrix_arr,
            col_names  = col_names,
            timeframe  = timeframe,
            n_jobs     = N_JOBS,
        )

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")