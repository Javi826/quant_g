#shared_batchs/pipeline/fdr_pi0.py
import time
import logging
import numpy as np
from numba import njit, prange
from tqdm import tqdm

logger = logging.getLogger("BOT_batch.pipeline.fdr_pi0")

# =============================================================================
# CONFIG — Fama-French (2010) joint MOVING-BLOCK bootstrap, cross-sectional
# percentiles. Block bootstrap (not IID) to preserve day-to-day
# autocorrelation in strategy P&L under the null resample.
# =============================================================================
FF_N_BOOTSTRAP = 1000
FF_RANDOM_SEED = 42
FF_BLOCK_SIZE  = 10  # fixed block length — mirrors stepM.py WHITE_BLOCK_SIZE
FF_PERCENTILES = np.array([1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 96, 97, 98, 99])

# =============================================================================
# ACTIVE-DAY CONFIG — a column's n is its count of nonzero (traded) days,
# not the row count of the shared matrix. Rows are structural zeros on days a
# given combo did not trade, so they must never dilute its mean/variance/n.
# =============================================================================
MIN_ACTIVE_DAYS_PER_COLUMN = 30  # real-data column filter, tune to strategy frequency
MIN_ACTIVE_DAYS_PER_RUN    = 30  # same floor applied inside every bootstrap run

# =============================================================================
# MEMORY-CHUNKING CONFIG — replicas processed in batches SEQUENTIALLY, in a
# single process. This is NOT for CPU parallelism (the numba kernel below
# already parallelizes over all physical cores via prange), only to bound
# peak RAM: total_sum/total_sumsq/total_count and the percentile scratch
# array are each (chunk_size, n_cols), not (n_bootstrap, n_cols).
#
# IMPORTANT: do NOT wrap this loop in joblib.Parallel / multiprocessing.
# Doing so double-parallelizes (N worker processes x M numba threads each,
# fighting over the same physical cores) and adds shared-memory attach/
# detach overhead per batch for no benefit, since the kernel is already
# using every core within one process. This was the main cause of the
# slowdown reported when this module briefly used joblib here.
# =============================================================================
BOOTSTRAP_CHUNK_SIZE = 500
METRIC_LABEL_WIDTH   = 12


def _format_thousands(value: int) -> str:
    return format(value, ",").replace(",", ".")


def _active_day_stats(matrix_arr: np.ndarray) -> tuple:
    """Per-column active-day mask, count, sum, and sum-of-squares over
    nonzero (traded) days only. Rows with a structural zero (no trade that
    day for that column) are excluded from all three."""
    active_mask = matrix_arr != 0.0
    counts = active_mask.sum(axis=0).astype(np.int64)
    sums   = np.where(active_mask, matrix_arr, 0.0).sum(axis=0, dtype=np.float64)
    sumsq  = np.where(active_mask, matrix_arr * matrix_arr, 0.0).sum(axis=0, dtype=np.float64)
    return active_mask, counts, sums, sumsq


def _tstat_from_moments(counts: np.ndarray, sums: np.ndarray, sumsq: np.ndarray, min_active_days: int) -> np.ndarray:
    """t(alpha) = mean / (std / sqrt(n)), n being the column's own count of
    active days. Columns below min_active_days are NaN — insufficient
    observations to estimate precision, not just insufficient variance."""
    n = counts.astype(np.float64)
    valid = counts >= min_active_days

    with np.errstate(divide="ignore", invalid="ignore"):
        means = np.where(valid, sums / n, np.nan)
        var   = np.where(valid, (sumsq - n * means * means) / np.maximum(n - 1.0, 1.0), np.nan)

    var  = np.where(np.isfinite(var), np.maximum(var, 0.0), np.nan)  # guard tiny negative fp error
    stds = np.sqrt(var)

    with np.errstate(divide="ignore", invalid="ignore"):
        tstat = np.where(stds > 0, (means / stds) * np.sqrt(n), np.nan)
    return tstat


def _demean_active_days(matrix_arr: np.ndarray, active_mask: np.ndarray, means_active: np.ndarray) -> np.ndarray:
    """Force the null of zero true performance: subtract each column's own
    mean, computed over its active days only, from its active days only.
    Structural zeros (no trade) stay exactly zero — demeaning them would
    fabricate trading activity on days the combo never traded."""
    adjusted = np.where(active_mask, matrix_arr - means_active[None, :], 0.0)
    return adjusted.astype(np.float64)


# =============================================================================
# MOVING BLOCK BOOTSTRAP — mirrors stepM.py's _generate_block_starts exactly.
# NOTE: duplicated rather than imported because it's a private helper in
# another pipeline module. If it ever changes in stepM.py, update here too —
# ideally this belongs in a shared utils module (e.g.
# shared_batchs/utils/block_bootstrap.py) so both pipelines stay in sync.
# =============================================================================
def _generate_block_starts(n_obs: int, block_size: int, n_replicas: int, rng: np.random.Generator):

    n_blocks_needed = int(np.ceil(n_obs / block_size))
    n_block_starts  = n_obs - block_size + 1

    starts = rng.integers(0, n_block_starts, size=(n_replicas, n_blocks_needed), dtype=np.int32)

    len_last = n_obs - (n_blocks_needed - 1) * block_size
    starts_full = starts[:, :-1] if n_blocks_needed > 1 else starts[:, :0]
    starts_last = starts[:, -1]

    return starts_full, starts_last, len_last, n_blocks_needed


def _compute_prefix_sums(matrix_adjusted: np.ndarray, active_mask: np.ndarray) -> tuple:
    """Prefix sums over the FULL null-adjusted matrix (rows already
    demeaned/zeroed as in _demean_active_days). Computed once per pipe_pi0
    call — block starts change per replica, these do not."""
    n_obs, n_cols = matrix_adjusted.shape
    x64 = matrix_adjusted.astype(np.float64, copy=False)

    ps = np.empty((n_obs + 1, n_cols), dtype=np.float64)
    ps[0] = 0.0
    np.cumsum(x64, axis=0, out=ps[1:])

    ps2 = np.empty_like(ps)
    ps2[0] = 0.0
    np.cumsum(x64 * x64, axis=0, out=ps2[1:])

    # Third prefix sum, specific to fdr_pi0.py: active-day COUNT, since a
    # column's n varies with its own sparsity, unlike stepM.py where every
    # column shares the same fixed n_obs.
    ps_cnt = np.empty((n_obs + 1, n_cols), dtype=np.int64)
    ps_cnt[0] = 0
    np.cumsum(active_mask.astype(np.int64), axis=0, out=ps_cnt[1:])

    return ps, ps2, ps_cnt


@njit(parallel=True, fastmath=False, cache=True)
def _block_bootstrap_moments_numba(
    ps: np.ndarray,
    ps2: np.ndarray,
    ps_cnt: np.ndarray,
    starts_full: np.ndarray,
    starts_last: np.ndarray,
    block_size: int,
    len_last: int,
) -> tuple:
    """Sum, sum-of-squares, and active-day count per (run, column) for a
    batch of moving-block resamples, via prefix-sum block differences.

    LOOP ORDER IS THE PERFORMANCE-CRITICAL PART: replica (r, outer,
    parallel) -> block (b, middle) -> column (c, INNERMOST). For a fixed
    block (start, end), ps[end, :] and ps[start, :] are each a single
    contiguous row in memory (ps is row-major, shape (n_obs+1, n_cols)), so
    iterating c innermost reads/writes contiguous memory. The earlier
    version had c OUTER and b INNER: for a fixed column, varying the block
    jumps by a full row-stride (n_cols elements) in ps on every access —
    effectively a cache miss per read once n_cols is large. Do not swap
    this back without re-checking the memory layout.
    """
    n_replicas    = starts_last.shape[0]
    n_blocks_full = starts_full.shape[1]
    n_cols        = ps.shape[1]

    total_sum   = np.zeros((n_replicas, n_cols), dtype=np.float64)
    total_sumsq = np.zeros((n_replicas, n_cols), dtype=np.float64)
    total_count = np.zeros((n_replicas, n_cols), dtype=np.int64)

    for r in prange(n_replicas):
        for b in range(n_blocks_full):
            start = starts_full[r, b]
            end   = start + block_size
            for c in range(n_cols):
                total_sum[r, c]   += ps[end, c]   - ps[start, c]
                total_sumsq[r, c] += ps2[end, c]  - ps2[start, c]
                total_count[r, c] += ps_cnt[end, c] - ps_cnt[start, c]

        start_last = starts_last[r]
        end_last   = start_last + len_last
        for c in range(n_cols):
            total_sum[r, c]   += ps[end_last, c]   - ps[start_last, c]
            total_sumsq[r, c] += ps2[end_last, c]  - ps2[start_last, c]
            total_count[r, c] += ps_cnt[end_last, c] - ps_cnt[start_last, c]

    return total_sum, total_sumsq, total_count


def _bootstrap_tstat_batch(
    ps: np.ndarray,
    ps2: np.ndarray,
    ps_cnt: np.ndarray,
    starts_full: np.ndarray,
    starts_last: np.ndarray,
    block_size: int,
    len_last: int,
    min_active_days: int,
) -> np.ndarray:
    """Cross-sectional t(alpha) per (run, column) for one batch of bootstrap
    replicas, each column's n taken from its own active-day count within
    the run's resampled blocks."""
    total_sum, total_sumsq, total_count = _block_bootstrap_moments_numba(
        ps, ps2, ps_cnt, starts_full, starts_last, block_size, len_last,
    )
    return _tstat_from_moments(total_count, total_sum, total_sumsq, min_active_days)


def _run_joint_bootstrap(
    matrix_adjusted: np.ndarray,
    active_mask: np.ndarray,
    real_percentiles: np.ndarray,
    percentiles: np.ndarray,
    n_bootstrap: int,
    chunk_size: int,
    seed: int,
    min_active_days_per_run: int,
    block_size: int = FF_BLOCK_SIZE,
) -> tuple:
    """Joint (same resampled day blocks shared across all columns) MOVING
    BLOCK bootstrap under the null of zero true performance.

    SINGLE PROCESS BY DESIGN. All n_bootstrap block-start draws are
    generated once upfront (one RNG). Replicas are then processed in
    sequential chunks of `chunk_size` purely to bound peak RAM (the
    tstat_batch / percentile scratch is (chunk_size, n_cols), not
    (n_bootstrap, n_cols)) — NOT for CPU parallelism. The numba kernel
    already parallelizes across every physical core via prange; wrapping
    this loop in joblib/multiprocessing on top of that oversubscribes the
    CPU (N worker processes each trying to use all cores) and adds
    shared-memory attach/detach overhead per batch with no offsetting
    benefit. If profiling later shows the kernel is NOT saturating all
    cores on this machine (e.g. numba defaulting to a low thread count),
    the fix is numba.set_num_threads(), not reintroducing multiprocessing
    here.
    """
    n_obs = matrix_adjusted.shape[0]

    ps, ps2, ps_cnt = _compute_prefix_sums(matrix_adjusted, active_mask)

    rng = np.random.default_rng(seed)
    starts_full, starts_last, len_last, n_blocks_needed = _generate_block_starts(
        n_obs, block_size, n_bootstrap, rng,
    )

    percentile_sum   = np.zeros(percentiles.shape[0], dtype=np.float64)
    below_actual_cnt = np.zeros(percentiles.shape[0], dtype=np.int64)
    n_valid_runs     = 0

    n_batches = int(np.ceil(n_bootstrap / chunk_size))
    for start in tqdm(range(0, n_bootstrap, chunk_size), desc="FF BOOTSTRAP", total=n_batches, dynamic_ncols=True):
        n_runs = min(chunk_size, n_bootstrap - start)
        end    = start + n_runs

        tstat_batch = _bootstrap_tstat_batch(
            ps, ps2, ps_cnt,
            starts_full[start:end], starts_last[start:end],
            block_size, len_last, min_active_days_per_run,
        )
        batch_percentiles = np.nanpercentile(tstat_batch, percentiles, axis=1).T  # (n_runs, n_pct)

        percentile_sum   += batch_percentiles.sum(axis=0)
        below_actual_cnt += (batch_percentiles < real_percentiles[None, :]).sum(axis=0)
        n_valid_runs     += n_runs

    avg_sim_percentiles = percentile_sum / n_valid_runs
    pct_below_actual    = 100.0 * below_actual_cnt / n_valid_runs
    return avg_sim_percentiles, pct_below_actual


def _log_ff_report(
    percentiles: np.ndarray,
    real_percentiles: np.ndarray,
    sim_percentiles: np.ndarray,
    pct_below_actual: np.ndarray,
    n_cols_built: int,
    n_dropped: int,
    n_bootstrap: int,
    min_active_days: int,
    block_size: int,
    timeframe: str,
) -> None:
    logger.info(f"\n{'─' * 70}")
    logger.info(f"  FAMA-FRENCH (2010) JOINT BLOCK BOOTSTRAP — t(α) percentiles ── {timeframe}")
    logger.info(f"{'─' * 70}")
    logger.info(
        f"  columns (rules) : {_format_thousands(n_cols_built)}   "
        f"dropped (< {min_active_days} active days) : {_format_thousands(n_dropped)}   "
        f"bootstrap runs : {_format_thousands(n_bootstrap)}   "
        f"block size : {block_size}"
    )
    logger.info(f"{'─' * 70}")
    logger.info(f"  {'Pct':>5} │ {'Sim':>8} {'Act':>8} {'%<Act':>8}")
    for i, pct in enumerate(percentiles):
        logger.info(f"  {pct:>5.0f} │ {sim_percentiles[i]:>8.2f} {real_percentiles[i]:>8.2f} {pct_below_actual[i]:>7.2f}%")
    logger.info(f"{'─' * 70}")
    logger.info(
        "  Act far below Sim in the left tail and/or far above in the right tail signals "
        "genuine skill (positive or negative) beyond what pure luck would produce."
    )
    logger.info(f"{'─' * 70}\n")


# =============================================================================
# PIPE FF BOOTSTRAP — standalone orchestration, only needs the raw daily P&L
# =============================================================================
# NOTE ON THE ROW GRID: matrix_arr's rows are NOT a continuous calendar grid.
# Upstream compaction (_compact_matrix in backtest_runner.py) drops any row
# where every column is zero, keeping only days on which at least one column
# traded. Individual columns are still sparse within those rows. This module
# uses each column's own active-day count as its n, so the row grid's exact
# shape does not bias the statistic — but n_obs here means "days with any
# activity across the whole batch", not "trading days in the calendar year".
#
# NOTE ON BLOCK BOOTSTRAP: block resampling is applied over this same
# (non-calendar) row grid. Blocks of consecutive rows preserve whatever
# day-to-day autocorrelation exists in that grid; they do NOT reconstruct
# calendar-adjacency across the gaps introduced by _compact_matrix.
# =============================================================================
def pipe_pi0(
    matrix_arr: np.ndarray,
    col_names: np.ndarray = None,
    n_bootstrap: int = FF_N_BOOTSTRAP,
    percentiles: np.ndarray = FF_PERCENTILES,
    chunk_size: int = BOOTSTRAP_CHUNK_SIZE,
    seed: int = FF_RANDOM_SEED,
    enabled: bool = True,
    timeframe: str = "",
    min_active_days: int = MIN_ACTIVE_DAYS_PER_COLUMN,
    min_active_days_per_run: int = MIN_ACTIVE_DAYS_PER_RUN,
    block_size: int = FF_BLOCK_SIZE,
) -> dict:

    if not enabled:
        logger.info(f"FF BOOTSTRAP ── {timeframe} ── disabled, skipping")
        return None

    if matrix_arr is None or matrix_arr.shape[1] < 2:
        logger.warning(f"FF BOOTSTRAP ── {timeframe} ── insufficient columns — skipping")
        return None

    n_obs_check = matrix_arr.shape[0]
    if block_size > n_obs_check:
        raise ValueError(
            f"FF BOOTSTRAP ── {timeframe} ── block_size ({block_size}) exceeds "
            f"n_obs ({n_obs_check}); cannot form a single block."
        )

    start = time.time()

    _, counts, sums, sumsq = _active_day_stats(matrix_arr)
    real_tstat = _tstat_from_moments(counts, sums, sumsq, min_active_days)

    finite_mask = np.isfinite(real_tstat)
    n_kept = int(finite_mask.sum())
    if n_kept == 0:
        raise ValueError(
            f"FF BOOTSTRAP ── {timeframe} ── no column has >= {min_active_days} active days "
            f"with finite t(alpha)."
        )

    kept_idx    = np.flatnonzero(finite_mask)
    matrix_kept = matrix_arr[:, kept_idx]
    real_tstat  = real_tstat[kept_idx]
    counts_kept = counts[kept_idx]
    sums_kept   = sums[kept_idx]

    active_mask_kept  = matrix_kept != 0.0
    means_active_kept = sums_kept / counts_kept.astype(np.float64)

    matrix_adjusted  = _demean_active_days(matrix_kept, active_mask_kept, means_active_kept)
    real_percentiles = np.percentile(real_tstat, percentiles)

    sim_percentiles, pct_below_actual = _run_joint_bootstrap(
        matrix_adjusted, active_mask_kept, real_percentiles, percentiles,
        n_bootstrap, chunk_size, seed, min_active_days_per_run, block_size,
    )

    n_dropped = matrix_arr.shape[1] - n_kept
    _log_ff_report(
        percentiles, real_percentiles, sim_percentiles, pct_below_actual,
        matrix_arr.shape[1], n_dropped, n_bootstrap, min_active_days, block_size, timeframe,
    )

    result = {
        "percentiles":      percentiles,
        "real_percentiles": real_percentiles,
        "sim_percentiles":  sim_percentiles,
        "pct_below_actual": pct_below_actual,
        "real_tstat":       real_tstat,
        "kept_idx":         kept_idx,
        "n_cols_built":     matrix_arr.shape[1],
        "n_dropped":        n_dropped,
        "n_bootstrap":      n_bootstrap,
        "block_size":       block_size,
        "timeframe":        timeframe,
    }
    if col_names is not None:
        result["kept_columns"] = np.asarray(col_names)[kept_idx]

    elapsed = int(time.time() - start)
    logger.info(f"FF BOOTSTRAP ── {timeframe} ── elapsed {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")

    return result