#shared_batchs/pipeline/FF_test.py
import time
import logging
import numpy as np
from numba import njit, prange
from tqdm import tqdm

logger = logging.getLogger("BOT_batch.pipeline.FF_test")

# =============================================================================
# CONFIG — Fama-French (2010) joint MOVING-BLOCK bootstrap, cross-sectional
# =============================================================================
FF_N_BOOTSTRAP = 1000
FF_RANDOM_SEED = 42
FF_BLOCK_SIZE  = 10  # fixed block length — mirrors stepM.py WHITE_BLOCK_SIZE
FF_PERCENTILES = np.array([1,5,10,50,95,99,99.99])

# =============================================================================
# ACTIVE-DAY CONFIG — a column's n is its count of nonzero (traded) days,
# =============================================================================
MIN_ACTIVE_DAYS_PER_COLUMN = 30  # real-data column filter, tune to strategy frequency
MIN_ACTIVE_DAYS_PER_RUN    = 30  # same floor applied inside every bootstrap run

# =============================================================================
# COLUMN-CHUNKING CONFIG — bootstrap kernel processes FF_COLUMN_CHUNK_SIZE
# =============================================================================
FF_COLUMN_CHUNK_SIZE = 256

# =============================================================================
# MEMORY-CHUNKING CONFIG — percentile phase processes replicas in batches
# =============================================================================
BOOTSTRAP_CHUNK_SIZE  = 500
COLUMN_CHUNK_SIZE     = 5000    # column chunk size for the cheap moments pass (Pass 1)
METRIC_LABEL_WIDTH    = 12

def _format_thousands(value: int) -> str:
    return format(value, ",").replace(",", ".")
# =============================================================================
# PASS 1 — REAL MOMENTS, COLUMN-CHUNKED
# =============================================================================
def _real_moments_chunked(matrix_arr: np.ndarray, chunk_size: int = COLUMN_CHUNK_SIZE) -> tuple:
    n_obs, n_cols = matrix_arr.shape
    counts = np.empty(n_cols, dtype=np.int64)
    sums   = np.empty(n_cols, dtype=np.float64)
    sumsq  = np.empty(n_cols, dtype=np.float64)

    for start in range(0, n_cols, chunk_size):
        end   = min(start + chunk_size, n_cols)
        chunk = matrix_arr[:, start:end]
        active_mask = chunk != 0.0
        counts[start:end] = active_mask.sum(axis=0)
        sums[start:end]   = np.where(active_mask, chunk, 0.0).sum(axis=0, dtype=np.float64)
        sumsq[start:end]  = np.where(active_mask, chunk * chunk, 0.0).sum(axis=0, dtype=np.float64)

    return counts, sums, sumsq

def _tstat_from_moments(counts: np.ndarray, sums: np.ndarray, sumsq: np.ndarray, min_active_days: int) -> np.ndarray:

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

# =============================================================================
# MOVING BLOCK BOOTSTRAP — mirrors stepM.py's _generate_block_starts exactly.
# =============================================================================
def _generate_block_starts(n_obs: int, block_size: int, n_replicas: int, rng: np.random.Generator):

    n_blocks_needed = int(np.ceil(n_obs / block_size))
    n_block_starts  = n_obs - block_size + 1

    starts = rng.integers(0, n_block_starts, size=(n_replicas, n_blocks_needed), dtype=np.int32)

    len_last = n_obs - (n_blocks_needed - 1) * block_size
    starts_full = starts[:, :-1] if n_blocks_needed > 1 else starts[:, :0]
    starts_last = starts[:, -1]

    return starts_full, starts_last, len_last, n_blocks_needed

# =============================================================================
# PASS 2 — TRANSPOSED PREFIX SUMS PER COLUMN CHUNK
# =============================================================================
def _compute_prefix_sums_chunk(matrix_chunk: np.ndarray, active_mask_chunk: np.ndarray) -> tuple:
    n_obs, chunk_cols = matrix_chunk.shape
    x64 = matrix_chunk.astype(np.float64, copy=False)

    ps = np.empty((n_obs + 1, chunk_cols), dtype=np.float64)
    ps[0] = 0.0
    np.cumsum(x64, axis=0, out=ps[1:])

    ps2 = np.empty((n_obs + 1, chunk_cols), dtype=np.float64)
    ps2[0] = 0.0
    np.cumsum(x64 * x64, axis=0, out=ps2[1:])

    ps_cnt = np.empty((n_obs + 1, chunk_cols), dtype=np.int32)
    ps_cnt[0] = 0
    np.cumsum(active_mask_chunk.astype(np.int32), axis=0, out=ps_cnt[1:])

    return np.ascontiguousarray(ps.T), np.ascontiguousarray(ps2.T), np.ascontiguousarray(ps_cnt.T)

@njit(parallel=True, fastmath=False, cache=True)
def _block_bootstrap_tstat_numba_colwise(
    ps_t: np.ndarray,       # (chunk_cols, n_obs+1) float64 — row-contiguous per column
    ps2_t: np.ndarray,      # (chunk_cols, n_obs+1) float64
    ps_cnt_t: np.ndarray,   # (chunk_cols, n_obs+1) int32
    starts_full: np.ndarray,   # (n_replicas, n_blocks_full) int32
    starts_last: np.ndarray,   # (n_replicas,) int32
    block_size: int,
    len_last: int,
    min_active_days: int,
) -> np.ndarray:

    n_cols_chunk  = ps_t.shape[0]
    n_replicas    = starts_last.shape[0]
    n_blocks_full = starts_full.shape[1]

    tstat = np.empty((n_replicas, n_cols_chunk), dtype=np.float32)

    for c in prange(n_cols_chunk):
        row_ps  = ps_t[c]
        row_ps2 = ps2_t[c]
        row_cnt = ps_cnt_t[c]

        for r in range(n_replicas):
            s   = 0.0
            sq  = 0.0
            cnt = 0
            for b in range(n_blocks_full):
                start = starts_full[r, b]
                end   = start + block_size
                s   += row_ps[end]    - row_ps[start]
                sq  += row_ps2[end]   - row_ps2[start]
                cnt += row_cnt[end]   - row_cnt[start]

            start_last = starts_last[r]
            end_last   = start_last + len_last
            s   += row_ps[end_last]  - row_ps[start_last]
            sq  += row_ps2[end_last] - row_ps2[start_last]
            cnt += row_cnt[end_last] - row_cnt[start_last]

            if cnt >= min_active_days:
                n    = float(cnt)
                mean = s / n
                var  = (sq - n * mean * mean) / max(n - 1.0, 1.0)
                if var < 0.0:
                    var = 0.0
                std = np.sqrt(var)
                if std > 0.0:
                    tstat[r, c] = np.float32((mean / std) * np.sqrt(n))
                else:
                    tstat[r, c] = np.float32(np.nan)
            else:
                tstat[r, c] = np.float32(np.nan)

    return tstat

def _build_bootstrap_tstat_matrix(
    matrix_adjusted: np.ndarray,
    active_mask: np.ndarray,
    starts_full: np.ndarray,
    starts_last: np.ndarray,
    block_size: int,
    len_last: int,
    min_active_days_per_run: int,
    column_chunk_size: int = FF_COLUMN_CHUNK_SIZE,
) -> np.ndarray:

    n_obs, n_cols = matrix_adjusted.shape
    n_bootstrap   = starts_last.shape[0]

    tstat_full = np.empty((n_bootstrap, n_cols), dtype=np.float32)

    n_chunks = int(np.ceil(n_cols / column_chunk_size))
    for start in tqdm(range(0, n_cols, column_chunk_size), desc="FF BOOTSTRAP", total=n_chunks, dynamic_ncols=True):
        end = min(start + column_chunk_size, n_cols)

        ps_t, ps2_t, ps_cnt_t = _compute_prefix_sums_chunk(
            matrix_adjusted[:, start:end], active_mask[:, start:end],
        )
        tstat_full[:, start:end] = _block_bootstrap_tstat_numba_colwise(
            ps_t, ps2_t, ps_cnt_t, starts_full, starts_last, block_size, len_last, min_active_days_per_run,
        )

    return tstat_full

def _demean_active_days(matrix_arr: np.ndarray, active_mask: np.ndarray, means_active: np.ndarray) -> np.ndarray:

    adjusted = np.where(active_mask, matrix_arr - means_active[None, :], 0.0)
    return adjusted.astype(np.float64)

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
    n_sample_replicas: int = 0,
) -> tuple:

    n_obs = matrix_adjusted.shape[0]

    rng = np.random.default_rng(seed)
    starts_full, starts_last, len_last, n_blocks_needed = _generate_block_starts(
        n_obs, block_size, n_bootstrap, rng,
    )

    # ---- Fase A: full t(alpha) matrix, column-chunked ----------------------
    tstat_full = _build_bootstrap_tstat_matrix(
        matrix_adjusted, active_mask, starts_full, starts_last,
        block_size, len_last, min_active_days_per_run,
    )

    # ---- Sample of raw null replicas, kept only for plotting purposes ------
    sample_replicas = tstat_full[:n_sample_replicas].copy() if n_sample_replicas > 0 else None

    # ---- Fase B: cross-sectional percentiles, replica-chunked --------------
    percentile_sum   = np.zeros(percentiles.shape[0], dtype=np.float64)
    below_actual_cnt = np.zeros(percentiles.shape[0], dtype=np.int64)
    n_valid_runs      = 0

    for start in range(0, n_bootstrap, chunk_size):
        end    = min(start + chunk_size, n_bootstrap)
        n_runs = end - start

        tstat_batch        = tstat_full[start:end]
        batch_percentiles  = np.nanpercentile(tstat_batch, percentiles, axis=1).T  # (n_runs, n_pct)

        percentile_sum   += batch_percentiles.sum(axis=0)
        below_actual_cnt += (batch_percentiles < real_percentiles[None, :]).sum(axis=0)
        n_valid_runs     += n_runs

    avg_sim_percentiles = percentile_sum / n_valid_runs
    pct_below_actual    = 100.0 * below_actual_cnt / n_valid_runs
    return avg_sim_percentiles, pct_below_actual, sample_replicas

def _log_ff_report(
    percentiles: np.ndarray,
    real_percentiles: np.ndarray,
    sim_percentiles: np.ndarray,
    pct_below_actual: np.ndarray,
    n_ge_percentile: np.ndarray,
    n_cols_built: int,
    n_dropped: int,
    n_bootstrap: int,
    min_active_days: int,
    block_size: int,
    timeframe: str,
) -> None:
    logger.info(f"\n{'─' * 85}")
    logger.info(f"  FAMA-FRENCH (2010) JOINT BLOCK BOOTSTRAP — t(α) percentiles ── {timeframe}")
    logger.info(f"{'─' * 85}")
    logger.info(
        f"  columns (rule × combo) : {_format_thousands(n_cols_built)}   "
        f"dropped (< {min_active_days} active days) : {_format_thousands(n_dropped)}   "
        f"bootstrap runs : {_format_thousands(n_bootstrap)}   "
        f"block size : {block_size}"
    )
    logger.info(f"{'─' * 85}")
    logger.info(f"  {'Pct':>5} │ {'N≥Pct':>9} │ {'Sim':>8} {'Real':>8} {'%<Real':>8}")
    for i, pct in enumerate(percentiles):
        logger.info(
            f"  {pct:>5.0f} │ {_format_thousands(int(n_ge_percentile[i])):>9} │ "
            f"{sim_percentiles[i]:>8.2f} {real_percentiles[i]:>8.2f} {pct_below_actual[i]:>7.2f}%"
        )
    logger.info(f"{'─' * 85}")
    logger.info("  Pct    : percentile level being compared across columns")
    logger.info("  N≥Pct  : how many real columns reach or exceed that percentile")
    logger.info("  Sim    : average value at that percentile across bootstrap replicas (pure luck)")
    logger.info("  Real   : actual value at that percentile in the real data")
    logger.info("  %<Real : share of bootstrap replicas that fell below Real at that percentile")
    logger.info(f"{'─' * 85}")
    logger.info(
        "  Real far below Sim in the left tail and/or far above in the right tail signals "
        "genuine skill (positive or negative) beyond what pure luck would produce."
    )
    logger.info(f"{'─' * 85}\n")


# =============================================================================
# PIPE FF BOOTSTRAP — standalone orchestration, only needs the raw daily P&L
# =============================================================================
def pipe_FF_test(
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
    n_sample_replicas: int = 0,
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

    counts, sums, sumsq = _real_moments_chunked(matrix_arr)
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

    sorted_tstat_asc = np.sort(real_tstat)
    n_ge_percentile  = sorted_tstat_asc.shape[0] - np.searchsorted(sorted_tstat_asc, real_percentiles, side="left")

    sim_percentiles, pct_below_actual, sim_tstat_sample = _run_joint_bootstrap(
        matrix_adjusted, active_mask_kept, real_percentiles, percentiles,
        n_bootstrap, chunk_size, seed, min_active_days_per_run, block_size,
        n_sample_replicas,
    )

    n_dropped = matrix_arr.shape[1] - n_kept
    _log_ff_report(
        percentiles, real_percentiles, sim_percentiles, pct_below_actual, n_ge_percentile,
        matrix_arr.shape[1], n_dropped, n_bootstrap, min_active_days, block_size, timeframe,
    )

    result = {
        "percentiles":      percentiles,
        "real_percentiles": real_percentiles,
        "sim_percentiles":  sim_percentiles,
        "pct_below_actual": pct_below_actual,
        "real_tstat":       real_tstat,
        "sim_tstat_sample": sim_tstat_sample,
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