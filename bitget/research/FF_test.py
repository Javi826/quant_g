#sets/FF_test_new.py
import os
import time
import logging
import numpy as np
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from numba import njit, prange
from tqdm import tqdm

logger = logging.getLogger("BOT_batch.pipeline.FF_test")

# =============================================================================
# CONFIG — Fama-French (2010) joint MOVING-BLOCK bootstrap, cross-sectional
# =============================================================================
FF_N_BOOTSTRAP = 1500
FF_RANDOM_SEED = 42
FF_BLOCK_SIZE  = 10  # fixed block length — mirrors stepM.py WHITE_BLOCK_SIZE
FF_PERCENTILES = np.array([10,50,90,99,99.99])

# =============================================================================
# ACTIVE-DAY CONFIG — a column's n is its count of nonzero (traded) days,
# =============================================================================
MIN_ACTIVE_DAYS_PER_COLUMN = 30  # real-data column filter, tune to strategy frequency
MIN_ACTIVE_DAYS_PER_RUN    = 30  # same floor applied inside every bootstrap run

# =============================================================================
# COLUMN-CHUNKING CONFIG — the bootstrap kernel processes FF_COLUMN_CHUNK_SIZE
# =============================================================================
FF_COLUMN_CHUNK_SIZE = 1024

# =============================================================================
# MOMENTS-CHUNKING CONFIG — column width handled by one prange iteration of
# the real-moments pass. Small enough that the three accumulators stay in L1.
# =============================================================================
FF_MOMENTS_CHUNK_SIZE = 512

# =============================================================================
# MEMORY-CHUNKING CONFIG — percentile phase processes replicas in batches
# =============================================================================
BOOTSTRAP_CHUNK_SIZE  = 500
METRIC_LABEL_WIDTH    = 12

# =============================================================================
# PERCENTILE THREADING CONFIG — replica rows are reduced independently, so the
# cross-sectional percentile phase is spread across threads. 0 = auto-detect.
# =============================================================================
FF_PERCENTILE_N_THREADS = 0
FF_PERCENTILE_MAX_THREADS = 32

# =============================================================================
# PROFILING CONFIG — per-phase wall-clock breakdown. Purely observational:
# no phase is skipped, reordered or altered when enabled.
# =============================================================================
FF_PROFILE = True

def _format_thousands(value: int) -> str:
    return format(value, ",").replace(",", ".")

# =============================================================================
# PHASE TIMER — accumulates wall time per named section across repeated calls,
# so the per-chunk sections inside the bootstrap loop roll up into one total.
# =============================================================================
class _PhaseTimer:

    def __init__(self, enabled: bool = FF_PROFILE):
        self.enabled = enabled
        self.totals  = {}
        self.counts  = {}
        self.order   = []

    @contextmanager
    def section(self, name: str):
        if not self.enabled:
            yield
            return
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, time.perf_counter() - t0)

    def record(self, name: str, elapsed: float) -> None:
        if not self.enabled:
            return
        if name not in self.totals:
            self.totals[name] = 0.0
            self.counts[name] = 0
            self.order.append(name)
        self.totals[name] += elapsed
        self.counts[name] += 1

    def log_report(self, total_elapsed: float, timeframe: str = "") -> None:
        if not self.enabled or not self.order:
            return

        accounted  = sum(self.totals.values())
        name_width = max(len(name) for name in self.order)
        share      = lambda seconds: 100.0 * seconds / total_elapsed if total_elapsed > 0 else 0.0

        logger.debug(f"\n{'─' * 85}")
        logger.debug(f"  FF BOOTSTRAP — PHASE PROFILE ── {timeframe}")
        logger.debug(f"{'─' * 85}")
        logger.debug(f"  {'phase':<{name_width}} │ {'calls':>8} │ {'seconds':>10} │ {'% total':>8}")
        for name in self.order:
            logger.debug(
                f"  {name:<{name_width}} │ {self.counts[name]:>8} │ "
                f"{self.totals[name]:>10.2f} │ {share(self.totals[name]):>7.1f}%"
            )
        logger.debug(f"{'─' * 85}")
        logger.debug(f"  accounted {accounted:.2f} s of {total_elapsed:.2f} s total ({share(accounted):.1f}%)")
        logger.debug(f"{'─' * 85}\n")

# =============================================================================
# PASS 1 — REAL MOMENTS, PARALLEL OVER COLUMN CHUNKS
# =============================================================================
@njit(parallel=True, fastmath=False, cache=True)
def _real_moments_numba(
    matrix_arr: np.ndarray,
    counts: np.ndarray,
    sums: np.ndarray,
    sumsq: np.ndarray,
    chunk_size: int,
) -> None:

    n_obs, n_cols = matrix_arr.shape
    n_chunks      = (n_cols + chunk_size - 1) // chunk_size

    for t in prange(n_chunks):
        col_start = t * chunk_size
        col_end   = min(col_start + chunk_size, n_cols)
        width     = col_end - col_start

        cnt_acc = np.zeros(width, dtype=np.int64)
        s_acc   = np.zeros(width, dtype=np.float64)
        q_acc   = np.zeros(width, dtype=np.float64)

        # ---- Row-major traversal: the read stays sequential within each row,
        #      which keeps the walk TLB-friendly however wide the matrix is.
        for i in range(n_obs):
            for j in range(width):
                v = matrix_arr[i, col_start + j]
                if v != 0.0:
                    cnt_acc[j] += 1
                s_acc[j] += v
                q_acc[j] += v * v

        for j in range(width):
            counts[col_start + j] = cnt_acc[j]
            sums[col_start + j]   = s_acc[j]
            sumsq[col_start + j]  = q_acc[j]

def _real_moments(matrix_arr: np.ndarray, chunk_size: int = FF_MOMENTS_CHUNK_SIZE) -> tuple:
    n_cols = matrix_arr.shape[1]
    counts = np.empty(n_cols, dtype=np.int64)
    sums   = np.empty(n_cols, dtype=np.float64)
    sumsq  = np.empty(n_cols, dtype=np.float64)
    _real_moments_numba(matrix_arr, counts, sums, sumsq, chunk_size)
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
# CHUNK PACK BUFFER — narrow contiguous staging area, allocated once per run.
#
# It serves two purposes at once. First, the kernel walks one column at a time,
# so it reads with a stride equal to its source's row stride; slicing the full
# P&L matrix directly would make that stride n_cols x itemsize, which at a
# million columns puts every element of a column on its own page and thrashes
# the TLB. Packing pins the stride at chunk x itemsize, making kernel
# throughput independent of matrix width. Second, when the active-day filter
# has dropped columns, the gather of the surviving ones happens here, so the
# kernel never needs the index map and no full-size filtered copy is ever
# materialized.
# =============================================================================
class _PackBuffer:

    def __init__(self, n_obs: int, max_cols: int, dtype: np.dtype):
        self.matrix = np.empty((n_obs, max_cols), dtype=dtype)

    def pack(self, matrix_arr: np.ndarray, cols: slice | np.ndarray, n_cols: int) -> np.ndarray:
        out = self.matrix[:, :n_cols]
        np.copyto(out, matrix_arr[:, cols])
        return out

    @property
    def nbytes(self) -> int:
        return self.matrix.nbytes

# =============================================================================
# BOOTSTRAP KERNEL — DEMEAN AND PREFIX SUMS FUSED INTO THE PARALLEL REGION
## =============================================================================
@njit(parallel=True, fastmath=False, cache=True)
def _block_bootstrap_tstat_numba_fused(
    matrix_chunk: np.ndarray,  # (n_obs, chunk_cols) raw P&L, packed contiguous
    means_chunk: np.ndarray,   # (chunk_cols,) float64 — active-day mean
    starts_full: np.ndarray,   # (n_replicas, n_blocks_full) int32
    starts_last: np.ndarray,   # (n_replicas,) int32
    block_size: int,
    len_last: int,
    min_active_days: int,
) -> np.ndarray:

    n_obs          = matrix_chunk.shape[0]
    n_cols_chunk   = matrix_chunk.shape[1]
    n_replicas     = starts_last.shape[0]
    n_blocks_full  = starts_full.shape[1]
    n_block_starts = n_obs - block_size + 1

    tstat = np.empty((n_replicas, n_cols_chunk), dtype=np.float32)

    for c in prange(n_cols_chunk):
        mean_c = means_chunk[c]

        # ---- Demean and prefix sums in one pass, thread-locally.
        row_ps  = np.empty(n_obs + 1, dtype=np.float64)
        row_ps2 = np.empty(n_obs + 1, dtype=np.float64)
        row_cnt = np.empty(n_obs + 1, dtype=np.int32)

        row_ps[0]  = 0.0
        row_ps2[0] = 0.0
        row_cnt[0] = 0
        for i in range(n_obs):
            raw = matrix_chunk[i, c]
            if raw != 0.0:
                v   = np.float64(raw) - mean_c
                inc = 1
            else:
                v   = 0.0
                inc = 0
            row_ps[i + 1]  = row_ps[i] + v
            row_ps2[i + 1] = row_ps2[i] + v * v
            row_cnt[i + 1] = row_cnt[i] + inc

        # ---- Per-block sums, hoisted out of the replica loop so the hot path
        #      is 3 reads instead of 6 reads plus 3 subtractions.
        block_s   = np.empty(n_block_starts, dtype=np.float64)
        block_sq  = np.empty(n_block_starts, dtype=np.float64)
        block_cnt = np.empty(n_block_starts, dtype=np.int32)

        for k in range(n_block_starts):
            block_s[k]   = row_ps[k + block_size]  - row_ps[k]
            block_sq[k]  = row_ps2[k + block_size] - row_ps2[k]
            block_cnt[k] = row_cnt[k + block_size] - row_cnt[k]

        for r in range(n_replicas):
            s   = 0.0
            sq  = 0.0
            cnt = 0
            for b in range(n_blocks_full):
                k    = starts_full[r, b]
                s   += block_s[k]
                sq  += block_sq[k]
                cnt += block_cnt[k]

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
    matrix_arr: np.ndarray,
    kept_idx: np.ndarray,
    all_kept: bool,
    means_active: np.ndarray,
    starts_full: np.ndarray,
    starts_last: np.ndarray,
    block_size: int,
    len_last: int,
    min_active_days_per_run: int,
    column_chunk_size: int = FF_COLUMN_CHUNK_SIZE,
    profiler: "_PhaseTimer" = None,
) -> np.ndarray:

    n_obs       = matrix_arr.shape[0]
    n_kept      = means_active.shape[0]
    n_bootstrap = starts_last.shape[0]
    profiler    = profiler or _PhaseTimer(enabled=False)

    tstat_full = np.empty((n_bootstrap, n_kept), dtype=np.float32)

    column_chunk_size = min(column_chunk_size, n_kept)
    pack = _PackBuffer(n_obs, column_chunk_size, matrix_arr.dtype)
    logger.debug(f"FF BOOTSTRAP ── pack buffer {pack.nbytes / 1e6:.0f} MB")

    n_chunks = int(np.ceil(n_kept / column_chunk_size))
    for chunk_idx, start in enumerate(
        tqdm(range(0, n_kept, column_chunk_size), desc="FF BOOTSTRAP      ", total=n_chunks, dynamic_ncols=True)
    ):
        end   = min(start + column_chunk_size, n_kept)
        width = end - start

        with profiler.section("A1 pack chunk"):
            cols         = slice(start, end) if all_kept else kept_idx[start:end]
            matrix_chunk = pack.pack(matrix_arr, cols, width)

        # ---- The first chunk absorbs numba's JIT compilation on a cache miss,
        #      so it is reported separately to keep the steady-state cost clean.
        kernel_label = "A2 fused kernel (1st, incl JIT)" if chunk_idx == 0 else "A3 fused kernel"
        with profiler.section(kernel_label):
            tstat_full[:, start:end] = _block_bootstrap_tstat_numba_fused(
                matrix_chunk, means_active[start:end], starts_full, starts_last,
                block_size, len_last, min_active_days_per_run,
            )

    return tstat_full

# =============================================================================
# THREADED ROW-WISE NANPERCENTILE
# =============================================================================
def _resolve_percentile_threads(n_rows: int, n_threads: int = FF_PERCENTILE_N_THREADS) -> int:
    if n_threads <= 0:
        n_threads = min(FF_PERCENTILE_MAX_THREADS, os.cpu_count() or 1)
    return max(1, min(n_threads, n_rows))

def _nanpercentile_row_range(
    tstat_batch: np.ndarray,
    percentiles: np.ndarray,
    out: np.ndarray,
    row_start: int,
    row_end: int,
) -> int:

    n_all_nan = 0
    for r in range(row_start, row_end):
        row    = tstat_batch[r]
        finite = row[~np.isnan(row)]
        if finite.size == 0:
            out[r] = np.nan
            n_all_nan += 1
        else:
            out[r] = np.percentile(finite, percentiles, overwrite_input=True)
    return n_all_nan

def _nanpercentile_rows_threaded(
    tstat_batch: np.ndarray,
    percentiles: np.ndarray,
    n_threads: int = FF_PERCENTILE_N_THREADS,
) -> np.ndarray:

    n_rows = tstat_batch.shape[0]
    n_pct  = percentiles.shape[0]

    # ---- Probe the exact output dtype NumPy would produce rather than
    #      assuming it: percentile promotes float32 input to float64 through
    #      the float64 interpolation weight, and the Phase-B accumulators are
    #      sensitive to that width.
    out_dtype = np.percentile(np.zeros(2, dtype=tstat_batch.dtype), percentiles).dtype
    out       = np.empty((n_rows, n_pct), dtype=out_dtype)

    n_threads = _resolve_percentile_threads(n_rows, n_threads)
    if n_threads == 1:
        n_all_nan = _nanpercentile_row_range(tstat_batch, percentiles, out, 0, n_rows)
    else:
        bounds = np.linspace(0, n_rows, n_threads + 1).astype(np.int64)
        with ThreadPoolExecutor(max_workers=n_threads) as pool:
            futures = [
                pool.submit(_nanpercentile_row_range, tstat_batch, percentiles, out, int(lo), int(hi))
                for lo, hi in zip(bounds[:-1], bounds[1:]) if hi > lo
            ]
            n_all_nan = sum(future.result() for future in futures)

    if n_all_nan:
        logger.warning(f"FF BOOTSTRAP ── {n_all_nan} all-NaN replica row(s) encountered")

    return out

def _run_joint_bootstrap(
    matrix_arr: np.ndarray,
    kept_idx: np.ndarray,
    all_kept: bool,
    means_active: np.ndarray,
    real_percentiles: np.ndarray,
    percentiles: np.ndarray,
    n_bootstrap: int,
    chunk_size: int,
    seed: int,
    min_active_days_per_run: int,
    block_size: int = FF_BLOCK_SIZE,
    n_sample_replicas: int = 0,
    profiler: "_PhaseTimer" = None,
) -> tuple:

    n_obs    = matrix_arr.shape[0]
    profiler = profiler or _PhaseTimer(enabled=False)

    with profiler.section("A0 block starts"):
        rng = np.random.default_rng(seed)
        starts_full, starts_last, len_last, n_blocks_needed = _generate_block_starts(
            n_obs, block_size, n_bootstrap, rng,
        )

    # ---- Fase A: full t(alpha) matrix, column-chunked ----------------------
    tstat_full = _build_bootstrap_tstat_matrix(
        matrix_arr, kept_idx, all_kept, means_active, starts_full, starts_last,
        block_size, len_last, min_active_days_per_run,
        profiler = profiler,
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

        tstat_batch = tstat_full[start:end]

        with profiler.section("B1 nanpercentile"):
            batch_percentiles = _nanpercentile_rows_threaded(tstat_batch, percentiles)  # (n_runs, n_pct)

        with profiler.section("B2 accumulate"):
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
    logger.info("  Sim    : average value at that percentile across bootstrap replicas (pure luck)")
    logger.info("  Real   : actual value at that percentile in the real data")
    logger.info("  %<Real : n of bootstrap replicas that are < Real at that percentile")
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

    start    = time.time()
    profiler = _PhaseTimer()

    with profiler.section("P1 real moments"):
        counts, sums, sumsq = _real_moments(matrix_arr)
        real_tstat = _tstat_from_moments(counts, sums, sumsq, min_active_days)

    finite_mask = np.isfinite(real_tstat)
    n_kept = int(finite_mask.sum())
    if n_kept == 0:
        raise ValueError(
            f"FF BOOTSTRAP ── {timeframe} ── no column has >= {min_active_days} active days "
            f"with finite t(alpha)."
        )

    # ---- Only per-column scalars are materialized here. The demeaned matrix
    #      and the active-day mask are derived inside the kernel instead.
    with profiler.section("P2 column stats"):
        kept_idx     = np.flatnonzero(finite_mask)
        all_kept     = n_kept == matrix_arr.shape[1]
        real_tstat   = real_tstat[kept_idx]
        means_active = sums[kept_idx] / counts[kept_idx].astype(np.float64)

    with profiler.section("P4 real percentiles"):
        real_percentiles = np.percentile(real_tstat, percentiles)

        sorted_tstat_asc = np.sort(real_tstat)
        n_ge_percentile  = sorted_tstat_asc.shape[0] - np.searchsorted(sorted_tstat_asc, real_percentiles, side="left")

    sim_percentiles, pct_below_actual, sim_tstat_sample = _run_joint_bootstrap(
        matrix_arr, kept_idx, all_kept, means_active, real_percentiles, percentiles,
        n_bootstrap, chunk_size, seed, min_active_days_per_run, block_size,
        n_sample_replicas,
        profiler = profiler,
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

    total_elapsed = time.time() - start
    profiler.log_report(total_elapsed, timeframe)

    elapsed = int(total_elapsed)
    logger.info(f"FF BOOTSTRAP ── {timeframe} ── elapsed {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")

    return result