#shared_batchs/pipeline/fdr.py
"""
Standalone FDR pipeline module — Benjamini & Yekutieli (2001), "The control
of the false discovery rate in multiple testing under dependency", Annals of
Statistics 29(4). Fully self-contained: runs its own moving-block bootstrap
over the per-column Sharpe ratio, with no dependency on stepM.py.
"""
import time
import logging
import numpy as np
from multiprocessing.shared_memory import SharedMemory
from joblib import Parallel, delayed
from tqdm import tqdm
from numba import njit, prange
from statsmodels.stats.multitest import multipletests
from shared_batchs.utils.paralelization import arrays_to_shared_memory, arrays_from_shared_memory, compact_columns_inplace
logger = logging.getLogger("BOT_batch.pipeline.fdr")

# =============================================================================
# BOOTSTRAP CONFIG — own copy, decoupled from stepM.py by design
# =============================================================================
FDR_N_BOOTSTRAP      = 1000
FDR_BLOCK_SIZE       = 10      # fixed block length for the moving-block bootstrap
FDR_N_JOBS           = -1
RANDOM_SEED          = 42
SHARPE_PERIODS_YEAR  = 365.0

# =============================================================================
# MEMORY-CHUNKING CONFIG — bounds peak RAM without changing any result
# =============================================================================
COLUMN_CHUNK_SIZE    = 5000    # columns processed per chunk for chunked reductions
REPLICA_CHUNK        = 100     # bootstrap replicas processed per gather chunk
BOOTSTRAP_BATCH_SIZE = 100     # columns processed per parallel batch

# =============================================================================
# BY CONFIG — Benjamini & Yekutieli (2001), Theorem 1.3: same step-up rule as
# Benjamini-Hochberg, with q replaced by q / c(m), c(m) = sum_{j=1}^{m} 1/j.
# Controls FDR under arbitrary dependence between p-values, at the cost of
# being more conservative than plain BH.
# =============================================================================
 # significance level (gamma) used for the pass/fail decision

# =============================================================================
# CHUNKED REDUCTIONS — column-wise mean/std without materializing a full-size
# intermediate buffer
# =============================================================================
def _mean_std_by_column_chunks(arr: np.ndarray, ddof: int = 0, chunk_size: int = COLUMN_CHUNK_SIZE):
    n_cols = arr.shape[1]
    means = np.empty(n_cols, dtype=np.float64)
    stds  = np.empty(n_cols, dtype=np.float64)
    for start in range(0, n_cols, chunk_size):
        end = min(start + chunk_size, n_cols)
        chunk = arr[:, start:end]
        means[start:end] = chunk.mean(axis=0, dtype=np.float64)
        stds[start:end]  = chunk.std(axis=0, ddof=ddof, dtype=np.float64)
    return means, stds


def _std_by_column_chunks(arr: np.ndarray, ddof: int = 0, chunk_size: int = COLUMN_CHUNK_SIZE) -> np.ndarray:
    n_cols = arr.shape[1]
    stds = np.empty(n_cols, dtype=np.float64)
    for start in range(0, n_cols, chunk_size):
        end = min(start + chunk_size, n_cols)
        stds[start:end] = arr[:, start:end].std(axis=0, ddof=ddof, dtype=np.float64)
    return stds

# =============================================================================
# STATISTIC — annualized Sharpe per trial column
# =============================================================================
def _sharpe_per_column(matrix_arr: np.ndarray) -> np.ndarray:

    means, stds = _mean_std_by_column_chunks(matrix_arr, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = (means / stds) * np.sqrt(SHARPE_PERIODS_YEAR)

    return np.where(stds > 0, sharpe, -np.inf)

# =============================================================================
# MOVING BLOCK BOOTSTRAP — PREFIX-SUM FORMULATION
# =============================================================================
def _generate_block_starts(n_obs: int, block_size: int, n_replicas: int, rng: np.random.Generator):

    n_blocks_needed = int(np.ceil(n_obs / block_size))
    n_block_starts  = n_obs - block_size + 1

    starts = rng.integers(0, n_block_starts, size=(n_replicas, n_blocks_needed), dtype=np.int32)

    len_last = n_obs - (n_blocks_needed - 1) * block_size
    starts_full = starts[:, :-1] if n_blocks_needed > 1 else starts[:, :0]
    starts_last = starts[:, -1]

    return starts_full, starts_last, len_last, n_blocks_needed

@njit(parallel=True, fastmath=False, cache=True)
def _block_bootstrap_sums_numba(
    ps: np.ndarray,
    ps2: np.ndarray,
    starts_full: np.ndarray,
    starts_last: np.ndarray,
    block_size: int,
    len_last: int,
) -> tuple:
    n_replicas    = starts_last.shape[0]
    n_blocks_full = starts_full.shape[1]
    batch_size    = ps.shape[1]

    total_sum   = np.empty((n_replicas, batch_size), dtype=np.float64)
    total_sumsq = np.empty((n_replicas, batch_size), dtype=np.float64)

    for r in prange(n_replicas):
        start_last = starts_last[r]
        end_last   = start_last + len_last
        for c in range(batch_size):
            s  = 0.0
            sq = 0.0
            for b in range(n_blocks_full):
                start = starts_full[r, b]
                end   = start + block_size
                s  += ps[end, c]  - ps[start, c]
                sq += ps2[end, c] - ps2[start, c]
            s  += ps[end_last, c]  - ps[start_last, c]
            sq += ps2[end_last, c] - ps2[start_last, c]
            total_sum[r, c]   = s
            total_sumsq[r, c] = sq

    return total_sum, total_sumsq

def _bootstrap_deviations_batch_prefix(
    batch_values: np.ndarray,
    starts_full: np.ndarray,
    starts_last: np.ndarray,
    block_size: int,
    len_last: int,
    n_obs: int,
    real_sharpe_batch: np.ndarray,
    replica_chunk: int = REPLICA_CHUNK,
) -> np.ndarray:

    x64 = batch_values.astype(np.float64, copy=False)

    ps = np.empty((n_obs + 1, x64.shape[1]), dtype=np.float64)
    ps[0] = 0.0
    np.cumsum(x64, axis=0, out=ps[1:])

    ps2 = np.empty_like(ps)
    ps2[0] = 0.0
    np.cumsum(x64 * x64, axis=0, out=ps2[1:])

    total_sum, total_sumsq = _block_bootstrap_sums_numba(
        ps, ps2, starts_full, starts_last, block_size, len_last,
    )

    means = total_sum / n_obs
    var = (total_sumsq - n_obs * means * means) / (n_obs - 1)
    np.maximum(var, 0.0, out=var)  # guard tiny negative fp error before sqrt
    stds = np.sqrt(var)

    with np.errstate(divide="ignore", invalid="ignore"):
        boot_sharpe = (means / stds) * np.sqrt(SHARPE_PERIODS_YEAR)
    boot_sharpe = np.where(stds > 0, boot_sharpe, -np.inf)

    return boot_sharpe - real_sharpe_batch[None, :]
def _bootstrap_deviations_batch_prefix_shm(
    shm_metadata: dict,
    start: int,
    end: int,
    block_size: int,
    len_last: int,
    n_obs: int,
    real_sharpe_batch: np.ndarray,
) -> None:

    base_arrays, shm_handles = arrays_from_shared_memory(shm_metadata)
    try:
        matrix       = base_arrays["fdr"]["matrix"]
        starts_full  = base_arrays["fdr"]["starts_full"]
        starts_last  = base_arrays["fdr"]["starts_last"]
        deviations   = base_arrays["fdr"]["deviations"]
        batch_values = matrix[:, start:end]
        deviations[:, start:end] = _bootstrap_deviations_batch_prefix(
            batch_values, starts_full, starts_last, block_size, len_last, n_obs, real_sharpe_batch,
        )
    finally:
        for shm in shm_handles:
            shm.close()

# =============================================================================
# BOOTSTRAP ORCHESTRATION — own version, no SPA: BY has no notion of a
# joint/max-based null, so there is nothing to recenter.
# =============================================================================
def compute_deviation_matrix(
    matrix_arr: np.ndarray,
    col_names: list,
    n_bootstrap: int = FDR_N_BOOTSTRAP,
    block_size: int = FDR_BLOCK_SIZE,
    seed: int = RANDOM_SEED,
    n_jobs: int = None,
    progress_label: str = "",
) -> dict:

    n_jobs = n_jobs if n_jobs is not None else FDR_N_JOBS

    col_names_arr = np.asarray(col_names)
    real_sharpe   = _sharpe_per_column(matrix_arr)

    finite_mask = np.isfinite(real_sharpe)
    if finite_mask.all():
        kept_columns = col_names_arr
    else:
        n_keep       = compact_columns_inplace(finite_mask, matrix_arr, real_sharpe, col_names_arr, chunk_size=COLUMN_CHUNK_SIZE)
        matrix_arr   = matrix_arr[:, :n_keep]
        real_sharpe  = real_sharpe[:n_keep]
        kept_columns = col_names_arr[:n_keep]

    n_obs  = matrix_arr.shape[0]
    n_cols = matrix_arr.shape[1]

    rng = np.random.default_rng(seed)
    starts_full, starts_last, len_last, _n_blocks_needed = _generate_block_starts(
        n_obs, block_size, n_bootstrap, rng,
    )

    matrix_arr32 = matrix_arr  # already float32 — no extra copy needed

    n_batches = int(np.ceil(n_cols / BOOTSTRAP_BATCH_SIZE))
    batch_bounds = [
        (i * BOOTSTRAP_BATCH_SIZE, min((i + 1) * BOOTSTRAP_BATCH_SIZE, n_cols))
        for i in range(n_batches)
    ]

    deviations_dtype  = np.float32
    deviations_nbytes = n_bootstrap * n_cols * np.dtype(deviations_dtype).itemsize

    shm_list, shm_metadata = arrays_to_shared_memory({
        "fdr": {"matrix": matrix_arr32, "starts_full": starts_full, "starts_last": starts_last},
    })
    deviations_shm = SharedMemory(create=True, size=max(deviations_nbytes, 1))
    shm_list.append(deviations_shm)
    shm_metadata["fdr"]["deviations"] = {
        "name":  deviations_shm.name,
        "shape": (n_bootstrap, n_cols),
        "dtype": str(np.dtype(deviations_dtype)),
    }
    deviations_shared = np.ndarray((n_bootstrap, n_cols), dtype=deviations_dtype, buffer=deviations_shm.buf)

    try:
        desc = f"FDR BOOTSTRAP {progress_label} ({BOOTSTRAP_BATCH_SIZE} cols/batch)".strip()
        list(tqdm(
            Parallel(n_jobs=n_jobs, return_as="generator")(
                delayed(_bootstrap_deviations_batch_prefix_shm)(
                    shm_metadata, start, end, block_size, len_last, n_obs, real_sharpe[start:end],
                )
                for start, end in batch_bounds
            ),
            desc=desc,
            total=n_batches,
            dynamic_ncols=True,
        ))
        # Own copy, decoupled from the shared-memory segment released below.
        deviations = deviations_shared.copy()
    finally:
        for shm in shm_list:
            shm.close()
            shm.unlink()

    sigma_hat = _std_by_column_chunks(deviations, ddof=1)

    valid_se = sigma_hat > 0
    if not valid_se.all():
        n_keep       = compact_columns_inplace(valid_se, deviations, real_sharpe, sigma_hat, kept_columns, chunk_size=COLUMN_CHUNK_SIZE)
        deviations   = deviations[:, :n_keep]
        real_sharpe  = real_sharpe[:n_keep]
        sigma_hat    = sigma_hat[:n_keep]
        kept_columns = kept_columns[:n_keep]

    deviations /= sigma_hat[None, :]
    studentized_deviations = deviations
    z_stat = real_sharpe / sigma_hat

    return {
        "real_sharpe":            real_sharpe,
        "sigma_hat":              sigma_hat,
        "studentized_deviations": studentized_deviations,
        "z_stat":                 z_stat,
        "kept_columns":           kept_columns,
    }

# =============================================================================
# INDIVIDUAL P-VALUES — two-sided, per-column, from each column's own
# bootstrap null.
# =============================================================================
def compute_individual_pvalues(
    studentized_deviations: np.ndarray,
    z_stat: np.ndarray,
    chunk_size: int = COLUMN_CHUNK_SIZE,
) -> np.ndarray:

    n_bootstrap, n_cols = studentized_deviations.shape
    abs_z_stat = np.abs(z_stat)
    p_values = np.empty(n_cols, dtype=np.float64)

    for start in range(0, n_cols, chunk_size):
        end = min(start + chunk_size, n_cols)
        exceed = np.abs(studentized_deviations[:, start:end]) >= abs_z_stat[None, start:end]
        p_values[start:end] = exceed.sum(axis=0) / n_bootstrap

    return p_values

# =============================================================================
# BY CORRECTION
# =============================================================================
def compute_by_correction(p_values: np.ndarray, alpha: float) -> np.ndarray:
    _, p_adjusted, _, _ = multipletests(p_values, alpha=alpha, method="fdr_by")
    return p_adjusted

# =============================================================================
# PIPE FDR — orchestration layer, mirroring dsr.py's pipe_dsr / stepM.py's pipe_stepm
# =============================================================================
def empty_fdr_fields() -> dict:
    """Placeholder FDR fields for rules that were never evaluated (pipe skipped)."""
    return {
        "passed_fdr": True,
        "fdr_p":      None,
    }
def pipe_fdr(
    raw_results: list,
    matrix_arr: np.ndarray,
    col_names: list,
    fdr_alpha: float,
    n_bootstrap: int = None,
    block_size: int = None,
    n_jobs: int = None,
    seed: int = None,
    timeframe: str = "",
) -> list:

    n_bootstrap = n_bootstrap if n_bootstrap is not None else FDR_N_BOOTSTRAP
    block_size  = block_size  if block_size  is not None else FDR_BLOCK_SIZE
    n_jobs      = n_jobs      if n_jobs      is not None else FDR_N_JOBS
    seed        = seed        if seed        is not None else RANDOM_SEED

    start = time.time()

    if matrix_arr is None:
        logger.warning(f"FDR ── {timeframe} ── insufficient data — skipping, passing all rules through untouched")
        return [{**r, **empty_fdr_fields()} for r in raw_results]

    if matrix_arr.shape[1] < 2:
        logger.warning(f"FDR ── {timeframe} ── insufficient columns — skipping, passing all rules through untouched")
        return [{**r, **empty_fdr_fields()} for r in raw_results]

    bootstrap_result = compute_deviation_matrix(
        matrix_arr, col_names, n_bootstrap=n_bootstrap, block_size=block_size,
        seed=seed, n_jobs=n_jobs, progress_label=timeframe,
    )
    kept_columns           = bootstrap_result["kept_columns"]
    studentized_deviations = bootstrap_result["studentized_deviations"]
    z_stat                  = bootstrap_result["z_stat"]

    raw_p_values      = compute_individual_pvalues(studentized_deviations, z_stat)
    adjusted_p_values = compute_by_correction(raw_p_values, alpha=fdr_alpha)

    p_by_col = dict(zip(kept_columns, adjusted_p_values))

    n_passed = 0
    results  = []
    for r in raw_results:
        best_combo_id = r.get("best_combo_id")
        col_name      = f"{r['rule_id']}__{best_combo_id}" if best_combo_id else None
        fdr_p         = p_by_col.get(col_name, float("nan"))
        passed        = bool(np.isfinite(fdr_p) and fdr_p <= fdr_alpha)
        n_passed     += int(passed)

        results.append({
            **r,
            "passed_fdr": passed,
            "fdr_p":      float(fdr_p) if np.isfinite(fdr_p) else None,
        })

    logger.info(f"FDR ── {timeframe} ── alpha={fdr_alpha:.2f} ── {n_passed}/{len(raw_results)} rules pass")

    elapsed = int(time.time() - start)
    logger.info(f"FDR ── {timeframe} ── elapsed {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")

    return results