#BOT_batch/main_NULL_BLOCKGRID.py
"""
StepM block-size robustness under a constructed null.

Every column of the real daily P&L matrix is demeaned, so theta_s = 0 for all
trials by construction: no rule has an edge. Synthetic datasets are then drawn
from that null population with a circular block bootstrap (block length
b_true), and the production StepM machinery is run on each draw with block
length b_assumed. Any rejection is a false rejection, so the fraction of draws
with at least k rejections IS the empirical k-FWER.

The grid decouples the dependence range of the simulated world (b_true) from
the block size StepM assumes (b_assumed), which answers the minimax question
"how badly can the FWER degrade for each candidate b_assumed, across plausible
dependence ranges" rather than "which b is right under one fixed DGP".

Two design notes worth keeping in mind when reading the output:

  - The outer draw uses the CIRCULAR block bootstrap, not the moving block
    bootstrap. This is deliberate: the circular version has exactly zero mean
    bias, so theta_s = 0 survives the resampling. Moving blocks would reinject
    a small edge through edge effects and corrupt the null.

  - Rejections are counted over the tested COLUMN family, taken straight from
    the stepdown p-values. Counting them through raw_results instead would only
    see the columns that happen to be some rule's best_combo, which is a small
    fraction of the family whenever the population is subsampled, and would
    bias the measured FWER downward.
"""
import io
import os
import sys
import time
import logging
import contextlib
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
    stepwise_reality_check_pvalues,
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
# GRID CONFIG
# =============================================================================
# b_true controls the outer draw. Note what it does and does not do: with a
# daily P&L whose autocorrelation dies at lag 1, longer blocks cannot manufacture
# long-range LINEAR dependence that is absent from the data. What they do
# preserve is long contiguous stretches of the real path — volatility
# clustering and other higher-order dependence the ACF is blind to but which
# still moves the Sharpe through its denominator. That is the axis being probed.
B_TRUE_GRID    = [5, 20, 50, 150]
B_ASSUMED_GRID = [5, 20, 50, 150]

N_RUNS_PER_CELL = 300
GRID_SEED       = 777

NULL_N_BOOTSTRAP = 300     # inner bootstrap replicas per StepM call

# Absolute FWER depends on the size of the tested family, so the grid is run at
# one column count and then a separate diagnostic re-runs the selected
# b_assumed across several column counts to expose any degradation in S.
NULL_MAX_COLUMNS    = 20000
COLUMN_SCALING_GRID = [2500, 5000, 10000, 20000]
RUN_COLUMN_SCALING  = True

# k = 1 keeps the criterion unambiguous: any rejection is a violation. Grid
# results for k > 1 would confound block-size effects with the k-FWER relaxation.
STEPM_ALPHA_NULL = 0.05
STEPM_K_FWE_NULL = 1

MIN_NONZERO_DAYS = 30      # columns with fewer active days are excluded


# =============================================================================
# PROGRESS SUPPRESSION — StepM writes a tqdm bar per call; thousands of calls
# would drown the log. tqdm targets stderr, so redirecting it is enough.
# =============================================================================
@contextlib.contextmanager
def _suppress_progress():
    with contextlib.redirect_stderr(io.StringIO()):
        yield


# =============================================================================
# NULL POPULATION — demean every column so theta_s = 0 by construction while
# leaving the dependence structure of the real data intact.
# =============================================================================
def _demean_inplace(matrix: np.ndarray, chunk_size: int = 5000) -> np.ndarray:
    n_cols = matrix.shape[1]
    for start in range(0, n_cols, chunk_size):
        end = min(start + chunk_size, n_cols)
        chunk = matrix[:, start:end]
        chunk -= chunk.mean(axis=0, dtype=np.float64).astype(matrix.dtype)[None, :]
    return matrix


def _eligible_columns(matrix: np.ndarray, chunk_size: int = 5000) -> np.ndarray:
    """Columns with enough active days and nonzero variance."""
    n_cols  = matrix.shape[1]
    active  = np.zeros(n_cols, dtype=bool)
    for start in range(0, n_cols, chunk_size):
        end = min(start + chunk_size, n_cols)
        chunk = matrix[:, start:end]
        enough_days = (chunk != 0).sum(axis=0) >= MIN_NONZERO_DAYS
        has_variance = chunk.std(axis=0, ddof=1) > 0
        active[start:end] = enough_days & has_variance
    return np.flatnonzero(active)


def _subsample_columns(matrix: np.ndarray, col_names: list, eligible: np.ndarray,
                       n_columns: int, rng: np.random.Generator) -> tuple:
    take = min(n_columns, eligible.size)
    keep = np.sort(rng.choice(eligible, size=take, replace=False))
    return np.ascontiguousarray(matrix[:, keep]), [col_names[i] for i in keep]


# =============================================================================
# OUTER DRAW — circular block bootstrap over rows, block length b_true.
# =============================================================================
def _circular_block_resample(population: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    n_obs    = population.shape[0]
    n_blocks = int(np.ceil(n_obs / block_size))

    starts  = rng.integers(0, n_obs, size=n_blocks, dtype=np.int64)
    offsets = np.arange(block_size, dtype=np.int64)
    row_idx = (starts[:, None] + offsets[None, :]).ravel()[:n_obs] % n_obs

    return np.ascontiguousarray(population[row_idx])


# =============================================================================
# ONE STEPM CALL — rejections counted over the tested column family
# =============================================================================
def _count_rejections(sample_arr: np.ndarray, col_names: list, block_size: int,
                      alpha: float, k_fwe: int, seed: int, n_jobs: int) -> int:
    """
    Returns the number of columns rejected by the stepdown, or -1 if the
    stepdown could not resolve (recorded rather than raised, so one bad draw
    does not abort a multi-hour grid).
    """
    try:
        with _suppress_progress():
            bootstrap_result = compute_deviation_matrix(
                sample_arr, col_names,
                n_bootstrap    = NULL_N_BOOTSTRAP,
                block_size     = block_size,
                seed           = seed,
                n_jobs         = n_jobs,
                progress_label = "",
            )
            pvals = stepwise_reality_check_pvalues(
                bootstrap_result["studentized_deviations"],
                bootstrap_result["z_stat"],
                alpha = alpha,
                k     = k_fwe,
            )
    except (RuntimeError, ValueError) as exc:
        logger.warning(f"GRID ── stepdown failed (block={block_size}, seed={seed}): {exc}")
        return -1

    return int((pvals <= alpha).sum())


# =============================================================================
# ONE GRID CELL — N runs at a fixed (b_true, b_assumed) pair
# =============================================================================
def _run_cell(population: np.ndarray, col_names: list, b_true_idx: int,
              b_true: int, b_assumed: int, n_jobs: int) -> dict:
    """
    Outer draws are seeded from b_true_idx only, so every b_assumed in the same
    row sees the SAME synthetic datasets. That pairing removes draw-to-draw
    noise from within-row comparisons.
    """
    rejections = np.empty(N_RUNS_PER_CELL, dtype=np.int64)

    for run_idx in range(N_RUNS_PER_CELL):
        outer_rng  = np.random.default_rng(GRID_SEED + 1_000_000 * b_true_idx + run_idx)
        sample_arr = _circular_block_resample(population, b_true, outer_rng)

        rejections[run_idx] = _count_rejections(
            sample_arr = sample_arr,
            col_names  = col_names,
            block_size = b_assumed,
            alpha      = STEPM_ALPHA_NULL,
            k_fwe      = STEPM_K_FWE_NULL,
            seed       = RANDOM_SEED + run_idx,
            n_jobs     = n_jobs,
        )

    resolved     = rejections[rejections >= 0]
    n_failed     = int((rejections < 0).sum())
    n_violations = int((resolved >= STEPM_K_FWE_NULL).sum())
    n_valid      = int(resolved.size)

    return {
        "b_true":       b_true,
        "b_assumed":    b_assumed,
        "n_violations": n_violations,
        "n_valid":      n_valid,
        "n_failed":     n_failed,
        "rate":         n_violations / n_valid if n_valid else np.nan,
        "median_rej":   float(np.median(resolved)) if n_valid else np.nan,
        "max_rej":      int(resolved.max()) if n_valid else -1,
    }


# =============================================================================
# STATISTICS
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


def _classify(n_success: int, n_trials: int, alpha: float) -> tuple:
    """
    Three-way verdict read off the CONSERVATIVE end of the interval: control is
    only claimed when the upper bound clears alpha, so an underpowered cell
    reports INCONCLUSIVE instead of silently passing.
    """
    lo, hi = _wilson_interval(n_success, n_trials)
    if hi <= alpha:
        status = "CONTROLLED"
    elif lo > alpha:
        status = "VIOLATED"
    else:
        status = "INCONCLUSIVE"
    return lo, hi, status


def _runs_needed_to_resolve(alpha: float, z: float = 1.96) -> int:
    """Runs needed for a Wilson half-width narrow enough to separate alpha from 2*alpha."""
    return int(np.ceil(z ** 2 * alpha * (1.0 - alpha) / (alpha / 2.0) ** 2))


# =============================================================================
# REPORTING
# =============================================================================
def _print_grid(timeframe: str, cells: dict, n_cols: int) -> None:
    cell_width = 16

    logger.info(f"\n{'─' * 110}")
    logger.info(f"  BLOCK SIZE ROBUSTNESS GRID ── {timeframe}")
    logger.info(f"{'─' * 110}")
    logger.info(f"  columns tested   : {n_cols}")
    logger.info(f"  runs per cell    : {N_RUNS_PER_CELL}")
    logger.info(f"  inner bootstrap  : {NULL_N_BOOTSTRAP} replicas")
    logger.info(f"  criterion        : empirical FWER (k={STEPM_K_FWE_NULL}) vs alpha={STEPM_ALPHA_NULL}")
    logger.info(f"  rows = b_true (simulated dependence) ── cols = b_assumed (StepM's block size)")
    logger.info(f"{'─' * 110}")

    logger.info(f"  {'b_true':<10}" + "".join(f"{f'b_asm={b}':<{cell_width}}" for b in B_ASSUMED_GRID))
    logger.info(f"{'─' * 110}")
    for b_true in B_TRUE_GRID:
        row = f"  {b_true:<10}"
        for b_assumed in B_ASSUMED_GRID:
            cell = cells[(b_true, b_assumed)]
            _, hi, status = _classify(cell["n_violations"], cell["n_valid"], STEPM_ALPHA_NULL)
            flag  = {"CONTROLLED": "", "INCONCLUSIVE": "?", "VIOLATED": "*"}[status]
            label = f"{cell['rate']:.3f}{flag}"
            row += f"{label:<{cell_width}}"
        logger.info(row)

    logger.info(f"{'─' * 110}")
    logger.info(f"  * empirical rate significantly above alpha ── ? interval straddles alpha")
    logger.info(f"{'─' * 110}\n")

    logger.info(f"  median / max rejections per cell")
    logger.info(f"{'─' * 110}")
    logger.info(f"  {'b_true':<10}" + "".join(f"{f'b_asm={b}':<{cell_width}}" for b in B_ASSUMED_GRID))
    logger.info(f"{'─' * 110}")
    for b_true in B_TRUE_GRID:
        row = f"  {b_true:<10}"
        for b_assumed in B_ASSUMED_GRID:
            cell  = cells[(b_true, b_assumed)]
            label = f"{cell['median_rej']:.0f}/{cell['max_rej']}"
            row += f"{label:<{cell_width}}"
        logger.info(row)
    logger.info(f"{'─' * 110}\n")


def _print_verdict(cells: dict, n_cols: int) -> int:
    logger.info(f"  VERDICT PER CANDIDATE b_assumed (worst case across all b_true)")
    logger.info(f"{'─' * 110}")

    worst_by_candidate = {}
    for b_assumed in B_ASSUMED_GRID:
        rates        = [cells[(b_true, b_assumed)]["rate"] for b_true in B_TRUE_GRID]
        worst_b_true = B_TRUE_GRID[int(np.nanargmax(rates))]
        worst_cell   = cells[(worst_b_true, b_assumed)]
        lo, hi, status = _classify(worst_cell["n_violations"], worst_cell["n_valid"], STEPM_ALPHA_NULL)

        worst_by_candidate[b_assumed] = (worst_cell["rate"], hi)
        logger.info(
            f"  b_assumed={b_assumed:<5} worst rate={worst_cell['rate']:.3f} at b_true={worst_b_true:<5} "
            f"95% CI [{lo:.3f}, {hi:.3f}] ── {status}"
        )

    logger.info(f"{'─' * 110}")
    logger.info(f"  Verdicts are CONDITIONAL ON THE TESTED FAMILY SIZE ({n_cols} columns).")
    logger.info(f"  Estimating the max over a larger family is harder, so control observed")
    logger.info(f"  here does not automatically carry to the full universe.")
    needed = _runs_needed_to_resolve(STEPM_ALPHA_NULL)
    logger.info(f"  Runs needed to separate alpha from 2*alpha: ~{needed} (current: {N_RUNS_PER_CELL})")
    logger.info(f"{'─' * 110}\n")

    # Selected candidate: lowest worst-case upper bound, tie-broken by rate.
    return min(B_ASSUMED_GRID, key=lambda b: (worst_by_candidate[b][1], worst_by_candidate[b][0]))


def _print_column_scaling(timeframe: str, b_assumed: int, scaling: dict) -> None:
    logger.info(f"\n{'─' * 110}")
    logger.info(f"  FAMILY-SIZE SCALING ── {timeframe} ── b_assumed={b_assumed}")
    logger.info(f"{'─' * 110}")
    logger.info(f"  Does the empirical FWER drift as the tested family grows?")
    logger.info(f"{'─' * 110}")
    logger.info(f"  {'COLUMNS':<12}{'B_TRUE':<10}{'RATE':<10}{'95% CI':<22}{'MEDIAN/MAX REJ':<18}{'STATUS':<14}")
    logger.info(f"{'─' * 110}")
    for (n_columns, b_true), cell in scaling.items():
        lo, hi, status = _classify(cell["n_violations"], cell["n_valid"], STEPM_ALPHA_NULL)
        interval = f"[{lo:.3f}, {hi:.3f}]"
        rejects  = f"{cell['median_rej']:.0f}/{cell['max_rej']}"
        logger.info(
            f"  {n_columns:<12}{b_true:<10}{cell['rate']:<10.3f}{interval:<22}{rejects:<18}{status:<14}"
        )
    logger.info(f"{'─' * 110}\n")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    start = time.time()

    n_cells     = len(B_TRUE_GRID) * len(B_ASSUMED_GRID)
    total_calls = n_cells * N_RUNS_PER_CELL

    logger.info(f"\n{'─' * 115}")
    logger.info(f"  STEPM BLOCK SIZE ROBUSTNESS — outer draw decoupled from inner bootstrap")
    logger.info(f"{'─' * 115}")
    logger.info(f"  TIMEFRAMES          : {TIMEFRAMES}")
    logger.info(f"  PARAM_GRID          : {PARAM_GRID}")
    logger.info(f"  B_TRUE_GRID         : {B_TRUE_GRID}")
    logger.info(f"  B_ASSUMED_GRID      : {B_ASSUMED_GRID}")
    logger.info(f"  N_RUNS_PER_CELL     : {N_RUNS_PER_CELL}")
    logger.info(f"  NULL_N_BOOTSTRAP    : {NULL_N_BOOTSTRAP}")
    logger.info(f"  TOTAL STEPM CALLS   : {total_calls}")
    logger.info(f"  NULL_MAX_COLUMNS    : {NULL_MAX_COLUMNS}")
    logger.info(f"  COLUMN_SCALING_GRID : {COLUMN_SCALING_GRID if RUN_COLUMN_SCALING else 'disabled'}")
    logger.info(f"  STEPM_ALPHA_NULL    : {STEPM_ALPHA_NULL}  (k={STEPM_K_FWE_NULL})")
    logger.info(f"{'─' * 115}\n")

    for timeframe in TIMEFRAMES:
        # -------------------------------------------------------------------
        # SINGLE BACKTEST — the real PnL matrix is the basis of the null
        # population, so the expensive stages run once per timeframe.
        # -------------------------------------------------------------------
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

        # -------------------------------------------------------------------
        # NULL POPULATION
        # -------------------------------------------------------------------
        full_population = _demean_inplace(matrix_arr)
        eligible        = _eligible_columns(full_population)
        column_rng      = np.random.default_rng(GRID_SEED)

        population, grid_col_names = _subsample_columns(
            full_population, col_names, eligible, NULL_MAX_COLUMNS, column_rng,
        )
        max_abs_mean = np.abs(population.mean(axis=0, dtype=np.float64)).max()
        logger.info(
            f"GRID ── {timeframe} ── population built ── {population.shape[0]} obs x "
            f"{population.shape[1]} columns ── eligible pool={eligible.size} ── "
            f"max |column mean| = {max_abs_mean:.3e}\n"
        )

        # -------------------------------------------------------------------
        # GRID SWEEP
        # -------------------------------------------------------------------
        cells    = {}
        cell_idx = 0
        for b_true_idx, b_true in enumerate(B_TRUE_GRID):
            for b_assumed in B_ASSUMED_GRID:
                cell_idx  += 1
                cell_start = time.time()

                cells[(b_true, b_assumed)] = _run_cell(
                    population    = population,
                    col_names     = grid_col_names,
                    b_true_idx    = b_true_idx,
                    b_true        = b_true,
                    b_assumed     = b_assumed,
                    n_jobs        = N_JOBS,
                )

                cell         = cells[(b_true, b_assumed)]
                cell_elapsed = time.time() - cell_start
                logger.info(
                    f"GRID ── cell {cell_idx}/{n_cells} ── b_true={b_true} b_assumed={b_assumed} ── "
                    f"rate={cell['rate']:.3f} ── median/max rej={cell['median_rej']:.0f}/{cell['max_rej']} ── "
                    f"failed={cell['n_failed']} ── {int(cell_elapsed)}s"
                )

                if cell_idx == 1:
                    projected = int(cell_elapsed * n_cells)
                    logger.info(
                        f"GRID ── projected grid runtime ── "
                        f"{projected // 3600} h {(projected % 3600) // 60} min "
                        f"(abort now if that is unacceptable)\n"
                    )

        _print_grid(timeframe, cells, population.shape[1])
        selected_b = _print_verdict(cells, population.shape[1])
        logger.info(f"  SELECTED b_assumed (lowest worst-case upper bound) : {selected_b}\n")

        # -------------------------------------------------------------------
        # FAMILY-SIZE SCALING — same b_assumed, growing column counts
        # -------------------------------------------------------------------
        if RUN_COLUMN_SCALING:
            worst_b_true = max(
                B_TRUE_GRID,
                key=lambda bt: cells[(bt, selected_b)]["rate"],
            )
            scaling = {}
            for n_columns in COLUMN_SCALING_GRID:
                scale_rng = np.random.default_rng(GRID_SEED + 31 * n_columns)
                scale_population, scale_names = _subsample_columns(
                    full_population, col_names, eligible, n_columns, scale_rng,
                )
                scale_start = time.time()
                scaling[(n_columns, worst_b_true)] = _run_cell(
                    population = scale_population,
                    col_names  = scale_names,
                    b_true_idx = B_TRUE_GRID.index(worst_b_true),
                    b_true     = worst_b_true,
                    b_assumed  = selected_b,
                    n_jobs     = N_JOBS,
                )
                cell = scaling[(n_columns, worst_b_true)]
                logger.info(
                    f"SCALING ── columns={n_columns} b_true={worst_b_true} b_assumed={selected_b} ── "
                    f"rate={cell['rate']:.3f} ── {int(time.time() - scale_start)}s"
                )

            _print_column_scaling(timeframe, selected_b, scaling)

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")