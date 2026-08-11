#shared_batchs/pipeline/stepM.py
import time
import logging
import numpy as np
from multiprocessing.shared_memory import SharedMemory
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from shared_batchs.utils.paralelization import arrays_to_shared_memory, arrays_from_shared_memory, compact_columns_inplace
from shared_batchs.utils.reporting import print_stepm_matrix_debug, print_stepm_real_variance_filter_debug, print_stepm_block_starts_debug
from shared_batchs.utils.reporting import print_stepm_bootstrap_replicas_debug, print_stepm_se_filter_debug, print_stepm_studentization_debug
from shared_batchs.utils.reporting import print_stepm_pvalue_quantile_equivalence_debug, print_stepm_monotonicity_debug, print_stepm_brc_equivalence_debug
logger = logging.getLogger("BOT_batch.pipeline.stepM")

# =============================================================================
# STATISTICAL TEST CONFIG — bootstrap sizing, annualization, reproducibility
# =============================================================================
STEPM_ALPHA         = 0.05     # significance level used inside the Romano-Wolf stepdown search
WHITE_PVALUE_TH     = STEPM_ALPHA
WHITE_N_BOOTSTRAP   = 1000
WHITE_BLOCK_SIZE    = 20   # fixed block length — mirrors montecarlo.py BLOCK_SIZE
STEPM_USE_SPA       = True
# =============================================================================
# STEPDOWN / K-FWE CONFIG — Romano-Wolf rejection rule and convergence cap
# =============================================================================
STEPM_K_MODE         = "percentile"  # "absolute" or "percentile"
STEPM_K_FWE          = 1             # used when STEPM_K_MODE == "absolute"
STEPM_MAX_ITERATIONS = 500           # safety cap on stepdown iterations

# =============================================================================
# MEMORY-CHUNKING CONFIG — bounds peak RAM without changing any result
# =============================================================================
REPLICA_CHUNK       = 100      # replicas processed per gather chunk inside the bootstrap prefix-sum step
COLUMN_CHUNK_SIZE   = 5000     # columns processed per chunk for chunked reductions/compaction over dense matrices
PARTITION_ROW_CHUNK = 50       # bootstrap replicas processed per np.partition call in the stepdown

# =============================================================================
# PARALLELISM + FIX
# =============================================================================
STEPM_N_JOBS         = -1
BOOTSTRAP_BATCH_SIZE = 100    
RANDOM_SEED          = 42
SHARPE_PERIODS_YEAR  = 365.0

# =============================================================================
# CHUNKED REDUCTIONS — column-wise mean/std without materializing a full-size
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
# STATISTIC — annualized Sharpe per trial column, vectorized over bootstrap replicas
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

    n_replicas = starts_last.shape[0]
    batch_size = x64.shape[1]

    boot_deviations = np.empty((n_replicas, batch_size), dtype=np.float64)

    for chunk_start in range(0, n_replicas, replica_chunk):
        chunk_end = min(chunk_start + replica_chunk, n_replicas)
        chunk_size = chunk_end - chunk_start

        starts_full_chunk = starts_full[chunk_start:chunk_end]
        starts_last_chunk = starts_last[chunk_start:chunk_end]

        if starts_full_chunk.shape[1] > 0:
            end_full_chunk   = starts_full_chunk + block_size
            sum_full_chunk   = (ps[end_full_chunk]  - ps[starts_full_chunk]).sum(axis=1)
            sumsq_full_chunk = (ps2[end_full_chunk] - ps2[starts_full_chunk]).sum(axis=1)
        else:
            sum_full_chunk   = np.zeros((chunk_size, batch_size), dtype=np.float64)
            sumsq_full_chunk = np.zeros((chunk_size, batch_size), dtype=np.float64)

        end_last_chunk   = starts_last_chunk + len_last
        sum_last_chunk   = ps[end_last_chunk]  - ps[starts_last_chunk]
        sumsq_last_chunk = ps2[end_last_chunk] - ps2[starts_last_chunk]

        total_sum_chunk   = sum_full_chunk + sum_last_chunk
        total_sumsq_chunk = sumsq_full_chunk + sumsq_last_chunk

        means_chunk = total_sum_chunk / n_obs
        var_chunk = (total_sumsq_chunk - n_obs * means_chunk * means_chunk) / (n_obs - 1)
        np.maximum(var_chunk, 0.0, out=var_chunk)  # guard tiny negative fp error before sqrt
        stds_chunk = np.sqrt(var_chunk)

        with np.errstate(divide="ignore", invalid="ignore"):
            boot_sharpe_chunk = (means_chunk / stds_chunk) * np.sqrt(SHARPE_PERIODS_YEAR)
        boot_sharpe_chunk = np.where(stds_chunk > 0, boot_sharpe_chunk, -np.inf)

        # NOTE: computation stays in float64 for numerical stability; only
        boot_deviations[chunk_start:chunk_end] = boot_sharpe_chunk - real_sharpe_batch[None, :]

    return boot_deviations


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
        matrix       = base_arrays["stepm"]["matrix"]
        starts_full  = base_arrays["stepm"]["starts_full"]
        starts_last  = base_arrays["stepm"]["starts_last"]
        deviations   = base_arrays["stepm"]["deviations"]
        batch_values = matrix[:, start:end]
        deviations[:, start:end] = _bootstrap_deviations_batch_prefix(
            batch_values, starts_full, starts_last, block_size, len_last, n_obs, real_sharpe_batch,
        )
    finally:
        for shm in shm_handles:
            shm.close()
            
def apply_spa_recentering(studentized_deviations: np.ndarray, z_stat: np.ndarray, n_obs: int) -> tuple:
    """Hansen (2005) SPA_c: recenter clearly-losing columns (z_stat below a
    slowly-growing threshold) to 0 instead of their own negative mean, before
    computing the bootstrap null. 
    """
    threshold = -np.sqrt(2.0 * np.log(np.log(max(n_obs, 3))))
    bad_mask  = z_stat < threshold
    if bad_mask.any():
        studentized_deviations[:, bad_mask] += z_stat[bad_mask][None, :]
    return studentized_deviations, bad_mask, threshold

def compute_deviation_matrix(
    matrix_arr: np.ndarray,
    col_names: list,
    n_bootstrap: int = WHITE_N_BOOTSTRAP,
    block_size: int = WHITE_BLOCK_SIZE,
    seed: int = RANDOM_SEED,
    n_jobs: int = None,
    progress_label: str = "",
) -> dict:

    n_jobs = n_jobs if n_jobs is not None else STEPM_N_JOBS

    n_cols_built  = matrix_arr.shape[1]
    col_names_arr = np.asarray(col_names)

    real_sharpe = _sharpe_per_column(matrix_arr)

    if logger.isEnabledFor(logging.DEBUG):
        # NOTE: day offsets on the global calendar grid, not real calendar
        # dates — stepM does not receive global_start_day from the caller.
        day_offsets = np.arange(matrix_arr.shape[0])
        print_stepm_matrix_debug(col_names, matrix_arr, matrix_arr.shape[0], day_offsets)

    finite_mask = np.isfinite(real_sharpe)
    if finite_mask.all():
        kept_columns = col_names_arr
    else:
        n_keep       = compact_columns_inplace(finite_mask, matrix_arr, real_sharpe, col_names_arr, chunk_size=COLUMN_CHUNK_SIZE)
        matrix_arr   = matrix_arr[:, :n_keep]
        real_sharpe  = real_sharpe[:n_keep]
        kept_columns = col_names_arr[:n_keep]

    if logger.isEnabledFor(logging.DEBUG):
        print_stepm_real_variance_filter_debug(progress_label, n_cols_built, matrix_arr.shape[1])

    n_obs  = matrix_arr.shape[0]
    n_cols = matrix_arr.shape[1]

    rng = np.random.default_rng(seed)
    starts_full, starts_last, len_last, n_blocks_needed = _generate_block_starts(
        n_obs, block_size, n_bootstrap, rng,
    )

    if logger.isEnabledFor(logging.DEBUG):
        print_stepm_block_starts_debug(progress_label, n_blocks_needed, block_size, len_last, n_obs, n_cols)

    matrix_arr32 = matrix_arr  # already float32 — no extra copy needed

    n_batches = int(np.ceil(n_cols / BOOTSTRAP_BATCH_SIZE))
    batch_bounds = [
        (i * BOOTSTRAP_BATCH_SIZE, min((i + 1) * BOOTSTRAP_BATCH_SIZE, n_cols))
        for i in range(n_batches)
    ]

    # MEMORY OPTIMIZATION: deviations is the dominant buffer in this pipeline
    deviations_dtype  = np.float32
    deviations_nbytes = n_bootstrap * n_cols * np.dtype(deviations_dtype).itemsize

    shm_list, shm_metadata = arrays_to_shared_memory({
        "stepm": {"matrix": matrix_arr32, "starts_full": starts_full, "starts_last": starts_last},
    })
    deviations_shm = SharedMemory(create=True, size=max(deviations_nbytes, 1))
    shm_list.append(deviations_shm)
    shm_metadata["stepm"]["deviations"] = {
        "name":  deviations_shm.name,
        "shape": (n_bootstrap, n_cols),
        "dtype": str(np.dtype(deviations_dtype)),
    }
    deviations_shared = np.ndarray((n_bootstrap, n_cols), dtype=deviations_dtype, buffer=deviations_shm.buf)

    try:
        desc = f"STEPM BOOTSTRAP {progress_label} ({BOOTSTRAP_BATCH_SIZE} cols/batch)".strip()
        with tqdm_joblib(tqdm(desc=desc, total=n_batches, dynamic_ncols=True)):
            Parallel(n_jobs=n_jobs)(
                delayed(_bootstrap_deviations_batch_prefix_shm)(
                    shm_metadata, start, end, block_size, len_last, n_obs, real_sharpe[start:end],
                )
                for start, end in batch_bounds
            )
        # Own copy, decoupled from the shared-memory segment released below.
        deviations = deviations_shared.copy()
    finally:
        for shm in shm_list:
            shm.close()
            shm.unlink()

    if logger.isEnabledFor(logging.DEBUG):
        print_stepm_bootstrap_replicas_debug(progress_label, deviations, n_cols, n_bootstrap)

    sigma_hat = _std_by_column_chunks(deviations, ddof=1)

    valid_se = sigma_hat > 0
    if not valid_se.all():
        n_keep       = compact_columns_inplace(valid_se, deviations, real_sharpe, sigma_hat, kept_columns, chunk_size=COLUMN_CHUNK_SIZE)
        deviations   = deviations[:, :n_keep]
        real_sharpe  = real_sharpe[:n_keep]
        sigma_hat    = sigma_hat[:n_keep]
        kept_columns = kept_columns[:n_keep]

    if logger.isEnabledFor(logging.DEBUG):
        print_stepm_se_filter_debug(progress_label, n_cols, kept_columns.shape[0], sigma_hat)

    deviations /= sigma_hat[None, :]
    studentized_deviations = deviations
    z_stat = real_sharpe / sigma_hat

    if logger.isEnabledFor(logging.DEBUG):
        print_stepm_studentization_debug(
            progress_label, studentized_deviations, z_stat, n_cols_built, n_cols, kept_columns.shape[0],
        )

    if STEPM_USE_SPA:
        studentized_deviations, spa_mask, spa_threshold = apply_spa_recentering(studentized_deviations, z_stat, n_obs)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"SPA RECENTERING {progress_label} ── threshold={spa_threshold:.4f} ── "
                f"{int(spa_mask.sum())}/{spa_mask.shape[0]} columns recentered to 0"
            )

    return {
        "real_sharpe":            real_sharpe,
        "sigma_hat":              sigma_hat,
        "studentized_deviations": studentized_deviations,
        "z_stat":                 z_stat,
        "kept_columns":           kept_columns,
    }

# =============================================================================
# GLOBAL P-VALUE — single number per timeframe, the original White (2000) test.
# =============================================================================
def compute_global_pvalue(deviations: np.ndarray, statistic: np.ndarray) -> dict:

    max_deviation  = np.max(deviations, axis=1)          # (n_bootstrap,)
    best_col_idx   = int(np.argmax(statistic))
    best_statistic = float(statistic[best_col_idx])

    global_p = float(np.mean(max_deviation >= best_statistic))

    return {
        "global_p":       global_p,
        "best_col_idx":    best_col_idx,
        "best_statistic":  best_statistic,
    }

# =============================================================================
# ROW-CHUNKED K-TH LARGEST — same np.partition(...)[:, part_idx] result, but
# =============================================================================
def _kth_largest_by_row_chunks(values: np.ndarray, k_eff: int, chunk_size: int = PARTITION_ROW_CHUNK) -> np.ndarray:
    n_rows, n_cols = values.shape
    part_idx = n_cols - k_eff
    result = np.empty(n_rows, dtype=values.dtype)
    for start in range(0, n_rows, chunk_size):
        end = min(start + chunk_size, n_rows)
        result[start:end] = np.partition(values[start:end], part_idx, axis=1)[:, part_idx]
    return result

# =============================================================================
# STEPM (ROMANO & WOLF, 2005) — stepdown per-rule p-values controlling FWER
# =============================================================================
def stepwise_reality_check_pvalues(
    deviations: np.ndarray,
    statistic: np.ndarray,
    alpha: float = STEPM_ALPHA,
    max_iterations: int = STEPM_MAX_ITERATIONS,
    k: int = STEPM_K_FWE,
) -> np.ndarray:

    if k < 1:
        raise ValueError(f"k (k-FWE level) must be >= 1, got {k}.")

    n_bootstrap, n_cols = deviations.shape

    order       = np.argsort(-statistic)
    dev_sorted  = deviations[:, order]
    stat_sorted = statistic[order]

    raw_pval_sorted = np.full(n_cols, np.nan, dtype=np.float64)
    active_start = 0

    for _iteration in range(max_iterations):
        n_active = n_cols - active_start
        if n_active <= 0:
            break

        buffer_size    = min(active_start, k - 1)
        extended_start = active_start - buffer_size

        extended_view = dev_sorted[:, extended_start:]
        active_stat   = stat_sorted[active_start:]
        n_extended    = extended_view.shape[1]
        k_eff         = min(k, n_extended)

        if k_eff == n_extended and n_extended != k:
            logger.debug(
                f"STEPDOWN iter={_iteration} ── requested k={k} exceeds extended set "
                f"size={n_extended} (active={n_active} + buffer={buffer_size}) ── "
                f"clamping to k_eff={k_eff} "
                f"(k-FWE degenerates toward the global minimum in this iteration)"
            )

        if k_eff == 1:
            kth_dev_extended = extended_view.max(axis=1)
        else:
            kth_dev_extended = _kth_largest_by_row_chunks(extended_view, k_eff)

        sorted_dev  = np.sort(kth_dev_extended)
        insert_pos  = np.searchsorted(sorted_dev, active_stat, side="left")
        candidate_p = (n_bootstrap - insert_pos) / n_bootstrap

        reject_local = candidate_p <= alpha
        n_reject     = int(reject_local.sum())

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                f"STEPDOWN iter={_iteration} ── k={k} ── active={n_active} ── "
                f"rejected_this_iter={n_reject} ── "
                f"candidate_p range=[{candidate_p.min():.4f}, {candidate_p.max():.4f}]"
            )
            if _iteration == 0:
                print_stepm_pvalue_quantile_equivalence_debug(k, kth_dev_extended, alpha, active_stat, reject_local, n_active)

        if n_reject == 0:
            raw_pval_sorted[active_start:] = candidate_p
            break

        raw_pval_sorted[active_start:active_start + n_reject] = candidate_p[:n_reject]
        active_start += n_reject
    else:
        unresolved_sorted = np.flatnonzero(np.isnan(raw_pval_sorted))
        if unresolved_sorted.size > 0:
            unresolved = order[unresolved_sorted]
            raise RuntimeError(
                f"StepM stepdown did not converge within {max_iterations} "
                f"iterations; unresolved column indices: {unresolved.tolist()}"
            )

    non_finite_sorted = ~np.isfinite(raw_pval_sorted)
    if non_finite_sorted.any():
        bad = order[non_finite_sorted]
        raise RuntimeError(
            f"StepM stepdown produced an undefined raw p-value for column "
            f"index(es) {bad.tolist()}; refusing to silently treat it as zero."
        )

    adjusted_pval_sorted = np.maximum.accumulate(raw_pval_sorted)

    adjusted_pval = np.empty(n_cols, dtype=np.float64)
    adjusted_pval[order] = adjusted_pval_sorted

    if logger.isEnabledFor(logging.DEBUG):
        print_stepm_monotonicity_debug(k, adjusted_pval_sorted)

    return adjusted_pval

# =============================================================================
# PIPE STEPM — orchestration layer, mirroring dsr.py's pipe_dsr exactly.
# =============================================================================
def empty_stepm_fields() -> dict:
    """Placeholder StepM fields for rules that were never evaluated (pipe skipped)."""
    return {
        "passed_stepm": True,
        "passed_mbias": True,
        "stepm_p":      None,
    }
def pipe_stepm(
    raw_results: list,
    matrix_arr: np.ndarray,
    col_names: list,
    stepm_alpha: float = None,
    stepm_pvalue_th: float = None,
    n_bootstrap: int = None,
    block_size: int = None,
    n_jobs: int = None,
    stepm_k_percentile: float = None,
    seed: int = None,
    timeframe: str = "",
) -> list:

    stepm_alpha     = stepm_alpha     if stepm_alpha     is not None else STEPM_ALPHA
    stepm_pvalue_th  = stepm_pvalue_th if stepm_pvalue_th is not None else stepm_alpha
    n_bootstrap      = n_bootstrap     if n_bootstrap     is not None else WHITE_N_BOOTSTRAP
    block_size       = block_size      if block_size      is not None else WHITE_BLOCK_SIZE
    n_jobs           = n_jobs          if n_jobs          is not None else STEPM_N_JOBS
    seed             = seed            if seed            is not None else RANDOM_SEED

    if STEPM_K_MODE == "percentile" and stepm_k_percentile is None:
        raise ValueError(
            "stepm_k_percentile is required when STEPM_K_MODE == 'percentile' — "
            "it has no module-level default; pass it explicitly from the caller."
        )

    if not np.isclose(stepm_pvalue_th, stepm_alpha):
        raise ValueError(
            f"stepm_pvalue_th ({stepm_pvalue_th}) must equal stepm_alpha "
            f"({stepm_alpha}). The FWE guarantee of Algorithm 4.1 only holds "
            "when the pass/fail threshold matches the alpha used to build "
            "the stepdown active sets — decoupling them invalidates the FWE "
            "control for both values."
        )

    start = time.time()

    if matrix_arr is None:
        logger.warning(f"STEPM ── {timeframe} ── insufficient data — skipping, passing all rules through untouched")
        return [{**r, **empty_stepm_fields()} for r in raw_results]

    if matrix_arr.shape[1] < 2:
        logger.warning(f"STEPM ── {timeframe} ── insufficient columns — skipping, passing all rules through untouched")
        return [{**r, **empty_stepm_fields()} for r in raw_results]
    bootstrap_result = compute_deviation_matrix(
        matrix_arr, col_names, n_bootstrap=n_bootstrap, block_size=block_size,
        seed=seed, n_jobs=n_jobs, progress_label=timeframe,
    )
    kept_columns            = bootstrap_result["kept_columns"]
    real_sharpe             = bootstrap_result["real_sharpe"]
    sigma_hat               = bootstrap_result["sigma_hat"]
    studentized_deviations  = bootstrap_result["studentized_deviations"]
    z_stat                  = bootstrap_result["z_stat"]

    logger.info(
        f"STEPM ── {timeframe} ── {matrix_arr.shape[1] - len(kept_columns)} degenerate "
        f"columns dropped ── {len(kept_columns)} columns remain"
    )

    global_result = compute_global_pvalue(studentized_deviations, z_stat)
    best_col_idx  = global_result["best_col_idx"]
    best_col_name = str(kept_columns[best_col_idx])

    logger.info(f"\n{'─' * 70}")
    logger.info(f"  GLOBAL WHITE p-value (studentized) ── {timeframe}")
    logger.info(f"{'─' * 70}")
    logger.info(f"  best column       : {best_col_name}")
    logger.info(f"  best real Sharpe  : {real_sharpe[best_col_idx]:.4f}")
    logger.info(f"  best z-statistic  : {global_result['best_statistic']:.4f}  (sigma_hat={sigma_hat[best_col_idx]:.4f})")
    logger.info(f"  global p-value    : {global_result['global_p']:.4f}")
    logger.info(f"{'─' * 70}\n")

    if STEPM_K_MODE == "absolute":
        k_fwe = STEPM_K_FWE
    elif STEPM_K_MODE == "percentile":
        n_cols_for_k = len(kept_columns)
        k_fwe = max(1, int(np.ceil(stepm_k_percentile * n_cols_for_k)))
        logger.info(
            f"STEPM ── {timeframe} ── STEPM_K_MODE=percentile ── resolved k={k_fwe} "
            f"from {stepm_k_percentile:.4%} of {n_cols_for_k} surviving columns"
        )
    else:
        raise ValueError(f"Unknown STEPM_K_MODE={STEPM_K_MODE!r}; expected 'absolute' or 'percentile'.")

    logger.info(f"STEPM ── {timeframe} ── k-FWE level k={k_fwe}" + (" (strict FWE)" if k_fwe == 1 else " (relaxed control — reasoned extension, see module docstring)"))

    stepm_pvals    = stepwise_reality_check_pvalues(studentized_deviations, z_stat, alpha=stepm_alpha, k=k_fwe)
    stepm_p_by_col = dict(zip(kept_columns, stepm_pvals))

    if logger.isEnabledFor(logging.DEBUG):
        print_stepm_brc_equivalence_debug(timeframe, k_fwe, global_result["global_p"], stepm_p_by_col, best_col_name)

    n_passed = 0
    results  = []
    for r in raw_results:
        best_combo_id = r.get("best_combo_id")
        col_name      = f"{r['rule_id']}__{best_combo_id}" if best_combo_id else None
        stepm_p       = stepm_p_by_col.get(col_name, float("nan"))
        passed        = bool(np.isfinite(stepm_p) and stepm_p <= stepm_pvalue_th)
        n_passed     += int(passed)

        results.append({
            **r,
            "passed_stepm": passed,
            "passed_mbias": passed,
            "stepm_p":      float(stepm_p) if np.isfinite(stepm_p) else None,
        })

    logger.info(f"STEPM ── {timeframe} ── k={k_fwe} ── {n_passed}/{len(raw_results)} rules pass")

    elapsed = int(time.time() - start)
    logger.info(f"STEPM ── {timeframe} ── elapsed {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")

    return results