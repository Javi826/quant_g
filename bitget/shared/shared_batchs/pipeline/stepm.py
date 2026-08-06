#shared_batchs/pipeline/stepm.py
"""
StepM (Romano & Wolf, 2005) — bootstrap machinery and orchestration, in one
module.

k-FWE EXTENSION — IMPORTANT CAVEAT:
Romano & Wolf (2005) only mention k-FWE control in passing (Section 6),
crediting Lehmann & Romano (2005) for a method based on individual p-values
under worst-case dependence (a generalization of Holm's method) — NOT a
bootstrap-based method that exploits the joint dependence structure the way
Algorithm 4.1 does. STEPM_K_FWE below is a reasoned extension of Algorithm 4.1
(replacing the max of the active bootstrap deviations with the k-th largest),
consistent with how the rest of the algorithm is built, but it is NOT a
transcription of either paper's proof. With STEPM_K_FWE=1 every code path
collapses exactly onto the strict-FWE Algorithm 4.1 already validated
(bit-identical to the pre-k-FWE version of this file).
"""
import time
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

logger = logging.getLogger("BOT_batch.pipeline.stepm")

# =============================================================================
# STEPM CONFIG (moving block bootstrap, same style as montecarlo.py)
# =============================================================================
STEPM_ALPHA           = 0.1    # significance level used inside the Romano-Wolf stepdown search
# The pass/fail threshold MUST equal STEPM_ALPHA. The FWE guarantee of
# Algorithm 4.1 (Romano & Wolf, 2005) only holds when the active set at each
# stepdown iteration is built with the same alpha used to decide pass/fail.
# Decoupling them silently invalidates the FWE control, so this is derived
# from STEPM_ALPHA rather than set independently.
WHITE_PVALUE_TH       = STEPM_ALPHA
WHITE_N_BOOTSTRAP     = 500
WHITE_BLOCK_SIZE      = 20     # fixed block length — mirrors montecarlo.py BLOCK_SIZE
SHARPE_PERIODS_YEAR   = 365.0  # must match batch_metrics.py's annualization factor
RANDOM_SEED           = 42
STEPM_MAX_ITERATIONS  = 500    # safety cap on stepdown iterations
BOOTSTRAP_BATCH_SIZE  = 100

# k-FWE control level. k=1 is strict FWE control (Algorithm 4.1 as published):
# reject a column only if its statistic beats the MAX of the active bootstrap
# deviations. k>1 relaxes this to k-FWE: reject if the statistic beats the
# K-TH LARGEST active bootstrap deviation, i.e. tolerate up to (k-1) false
# rejections with non-negligible probability in exchange for power. See the
# module docstring above for the caveat on how far this is from the papers.
# Override from main.py the same way BACKTEST_N_JOBS is overridden, e.g.:
#     import shared_batchs.pipeline.stepm as stepm_module
#     stepm_module.STEPM_K_FWE = 5
STEPM_K_FWE = 200

# Toggle for the correctness-verification and distribution-descriptive debug
# logging below. All of it is logger.debug, so it is inert unless the module
# logger is raised to DEBUG — this flag additionally gates the extra O(B*S)
# computations those checks need, so it can be turned off even at DEBUG level
# once the calculation has been validated. Override from main.py the same way
# BACKTEST_N_JOBS is overridden on backtest_runner.py, e.g.:
#     import shared_batchs.pipeline.stepm as stepm_module
#     stepm_module.STEPM_VERIFY = True
STEPM_VERIFY = True


# =============================================================================
# DAILY MATRIX CONSTRUCTION — T days x M trials (rule x combo)
# =============================================================================
def _column_dates(column: tuple) -> np.ndarray:

    day_offsets, _values, start_day = column
    return start_day + day_offsets.astype("timedelta64[D]")


def build_flat_daily_matrix(rules: list) -> pd.DataFrame | None:

    series_by_col = {}
    for r in rules:
        combo_profit = r.get("combo_daily_profit") or {}
        for combo_id, s in combo_profit.items():
            series_by_col[f"{r['rule_id']}__{combo_id}"] = s

    if len(series_by_col) < 2:
        logger.debug(f"MATRIX ── built {len(series_by_col)} columns — below minimum of 2")
        return None

    col_names = list(series_by_col.keys())

    all_dates = np.unique(
        np.concatenate([_column_dates(col) for col in series_by_col.values()])
    )

    matrix_arr = np.zeros((all_dates.shape[0], len(col_names)), dtype=np.float64)
    for col_idx, col_name in enumerate(col_names):
        day_offsets, values, _start_day = series_by_col[col_name]
        row_idx = np.searchsorted(all_dates, _column_dates(series_by_col[col_name]))
        matrix_arr[row_idx, col_idx] = values.astype(np.float64)

    logger.debug(
        f"MATRIX ── built {len(col_names)} columns (rule__combo) over "
        f"{all_dates.shape[0]} distinct days ── "
        f"range [{all_dates.min()} .. {all_dates.max()}]"
    )

    if STEPM_VERIFY:
        # DESCRIBE[zero_fill] — how much of each column is the zero-fill from
        # days where that rule__combo had no trade. High zero fractions on
        # infrequent rules mechanically shrink both mean and std toward zero
        # at different rates (mean ~k, std ~sqrt(k)), which biases the Sharpe
        # of sparse rules downward relative to frequently-trading ones.
        zero_frac = (matrix_arr == 0).mean(axis=0)
        pct = np.percentile(zero_frac, [0, 50, 90, 99, 100])
        logger.debug(
            f"DESCRIBE[zero_fill] ── fraction of zero-filled days per column, "
            f"percentiles [min,p50,p90,p99,max] = "
            f"[{pct[0]:.3f}, {pct[1]:.3f}, {pct[2]:.3f}, {pct[3]:.3f}, {pct[4]:.3f}]"
        )

    return pd.DataFrame(matrix_arr, index=pd.DatetimeIndex(all_dates), columns=col_names)


# =============================================================================
# STATISTIC — annualized Sharpe per trial column, vectorized over bootstrap replicas
# =============================================================================
def _sharpe_per_column(matrix_arr: np.ndarray) -> np.ndarray:

    means = matrix_arr.mean(axis=0)
    stds  = matrix_arr.std(axis=0, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = (means / stds) * np.sqrt(SHARPE_PERIODS_YEAR)

    return np.where(stds > 0, sharpe, -np.inf)


# =============================================================================
# MOVING BLOCK BOOTSTRAP INDEX GENERATION — fixed block length, with replacement
# (same technique as montecarlo.py's _make_overlapping_blocks, vectorized here
# across all bootstrap replicas and applied to the shared day axis)
# =============================================================================
def _moving_block_bootstrap_indices_batch(
    n_obs: int, block_size: int, n_replicas: int, rng: np.random.Generator
) -> np.ndarray:

    n_blocks_needed = int(np.ceil(n_obs / block_size))
    n_block_starts   = n_obs - block_size + 1

    chosen_starts = rng.integers(0, n_block_starts, size=(n_replicas, n_blocks_needed))

    block_offsets = np.arange(block_size)[None, None, :]
    indices = chosen_starts[:, :, None] + block_offsets           # (n_replicas, n_blocks_needed, block_size)
    indices = indices.reshape(n_replicas, n_blocks_needed * block_size)[:, :n_obs]

    return indices


def _bootstrap_deviations_batch(
    batch_values: np.ndarray, boot_idx: np.ndarray, real_sharpe_batch: np.ndarray
) -> np.ndarray:

    boot_samples = batch_values[boot_idx]  # (n_bootstrap, n_obs, batch_size), float32

    means = boot_samples.mean(axis=1)
    stds  = boot_samples.std(axis=1, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        boot_sharpe = (means / stds) * np.sqrt(SHARPE_PERIODS_YEAR)
    boot_sharpe = np.where(stds > 0, boot_sharpe, -np.inf).astype(np.float64)

    return boot_sharpe - real_sharpe_batch[None, :]


def compute_deviation_matrix(
    matrix: pd.DataFrame,
    n_bootstrap: int = WHITE_N_BOOTSTRAP,
    block_size: int = WHITE_BLOCK_SIZE,
    seed: int = RANDOM_SEED,
    n_jobs: int = -1,
    progress_label: str = "",
) -> dict:

    n_cols_built = matrix.shape[1]

    matrix_arr  = matrix.to_numpy(dtype=np.float64)
    real_sharpe = _sharpe_per_column(matrix_arr)

    # Degenerate-column filtering only (non-finite Sharpe from a zero-variance
    # series). This is NOT outlier filtering — no column is dropped for having
    # a large or small statistic, only for being mathematically undefined.
    finite_mask  = np.isfinite(real_sharpe)
    matrix_arr   = matrix_arr[:, finite_mask]
    real_sharpe  = real_sharpe[finite_mask]
    kept_columns = matrix.columns[finite_mask]

    n_dropped_real_variance = n_cols_built - int(finite_mask.sum())
    logger.debug(
        f"MATRIX FILTER (real variance) {progress_label} ── "
        f"{n_dropped_real_variance}/{n_cols_built} columns dropped "
        f"(zero-variance original series) ── {matrix_arr.shape[1]} remain"
    )

    n_obs  = matrix_arr.shape[0]
    n_cols = matrix_arr.shape[1]

    rng      = np.random.default_rng(seed)
    boot_idx = _moving_block_bootstrap_indices_batch(n_obs, block_size, n_bootstrap, rng)

    if STEPM_VERIFY:
        # VERIFY[shared_index] — boot_idx is built once and passed unchanged
        # into every _bootstrap_deviations_batch call below. This is what
        # preserves the joint dependence structure across columns; if each
        # batch instead drew its own indices, StepM would collapse to
        # Bonferroni-with-extra-steps and lose its whole advantage.
        idx_hash = hash(boot_idx.tobytes())
        logger.debug(
            f"VERIFY[shared_index] {progress_label} ── boot_idx shape={boot_idx.shape} "
            f"hash={idx_hash} ── one array reused across every column batch by construction"
        )

        # VERIFY[blocks] — confirm boot_idx actually contains contiguous runs
        # of length block_size, not i.i.d. single-day draws. Expected fraction
        # of consecutive (i, i+1) column pairs that are +1 apart is
        # (block_size - 1) / block_size.
        consecutive_diffs = np.diff(boot_idx, axis=1)
        frac_consecutive  = float(np.mean(consecutive_diffs == 1))
        expected_frac     = (block_size - 1) / block_size
        blocks_ok = abs(frac_consecutive - expected_frac) < 0.02
        logger.debug(
            f"VERIFY[blocks] {progress_label} ── fraction of consecutive-index pairs = "
            f"{frac_consecutive:.4f} (expected ≈ {expected_frac:.4f} for block_size={block_size}) "
            f"── {'✅' if blocks_ok else '❌'}"
        )

    matrix_arr32 = matrix_arr.astype(np.float32)
    del matrix_arr

    n_batches = int(np.ceil(n_cols / BOOTSTRAP_BATCH_SIZE))
    batch_bounds = [
        (i * BOOTSTRAP_BATCH_SIZE, min((i + 1) * BOOTSTRAP_BATCH_SIZE, n_cols))
        for i in range(n_batches)
    ]

    desc = f"STEPM BOOTSTRAP {progress_label} ({BOOTSTRAP_BATCH_SIZE} cols/batch)".strip()
    with tqdm_joblib(tqdm(desc=desc, total=n_batches, dynamic_ncols=True)):
        deviations_per_batch = Parallel(n_jobs=n_jobs)(
            delayed(_bootstrap_deviations_batch)(
                matrix_arr32[:, start:end], boot_idx, real_sharpe[start:end]
            )
            for start, end in batch_bounds
        )

    deviations = np.concatenate(deviations_per_batch, axis=1)  # shape (n_bootstrap, n_cols)
    del deviations_per_batch, matrix_arr32

    # Diagnostic only: count columns whose bootstrap deviations contain at
    # least one -inf value (a bootstrap replica hit a zero-variance block for
    # that column). These columns are NOT necessarily dropped here — they
    # only get dropped below if sigma_hat ends up exactly zero or non-finite.
    # This distinguishes "genuinely degenerate column" from "column that
    # merely got unlucky in some bootstrap replicas."
    inf_mask               = ~np.isfinite(deviations)
    n_inf_per_col          = inf_mask.sum(axis=0)
    cols_with_inf_replica  = int((n_inf_per_col > 0).sum())
    logger.debug(
        f"BOOTSTRAP REPLICAS {progress_label} ── "
        f"{cols_with_inf_replica}/{n_cols} columns hit a non-finite Sharpe "
        f"in at least one bootstrap replica (zero-variance block)"
    )

    if STEPM_VERIFY:
        affected = n_inf_per_col[n_inf_per_col > 0]
        if affected.size:
            pct = np.percentile(affected, [0, 50, 90, 100])
            logger.debug(
                f"DESCRIBE[inf_replicas] {progress_label} ── among affected columns, "
                f"non-finite replica count per column percentiles "
                f"[min,p50,p90,max] out of {n_bootstrap} = "
                f"[{pct[0]:.0f}, {pct[1]:.0f}, {pct[2]:.0f}, {pct[3]:.0f}]"
            )

    sigma_hat = deviations.std(axis=0, ddof=1)

    # Same degenerate-column rule as above, applied post-bootstrap: drop
    # columns whose bootstrap standard error is exactly zero (undefined
    # studentization), not columns whose statistic is large.
    valid_se     = sigma_hat > 0
    deviations   = deviations[:, valid_se]      # copy — boolean mask indexing cannot be a view
    real_sharpe  = real_sharpe[valid_se]
    sigma_hat    = sigma_hat[valid_se]
    kept_columns = kept_columns[valid_se]

    n_dropped_bootstrap_se = n_cols - int(valid_se.sum())
    logger.debug(
        f"MATRIX FILTER (bootstrap SE) {progress_label} ── "
        f"{n_dropped_bootstrap_se}/{n_cols} columns dropped "
        f"(sigma_hat == 0 or non-finite after bootstrap) ── "
        f"{kept_columns.shape[0]} remain"
    )

    if STEPM_VERIFY:
        pct_sigma = np.percentile(sigma_hat, [0, 50, 90, 99, 100])
        ratio_max_min = float(pct_sigma[-1] / max(pct_sigma[0], 1e-12))
        logger.debug(
            f"DESCRIBE[sigma_hat] {progress_label} ── bootstrap SE percentiles "
            f"[min,p50,p90,p99,max] = "
            f"[{pct_sigma[0]:.4f}, {pct_sigma[1]:.4f}, {pct_sigma[2]:.4f}, "
            f"{pct_sigma[3]:.4f}, {pct_sigma[4]:.4f}] ── ratio max/min = {ratio_max_min:.2f} "
            f"(White 2000 Sec.9 flagged a ratio of 22.2 as enough to break the basic method)"
        )

    deviations /= sigma_hat[None, :]
    studentized_deviations = deviations
    z_stat                 = real_sharpe / sigma_hat

    if STEPM_VERIFY:
        # VERIFY[studentization] — confirm this is Hansen-style studentization
        # (sigma_hat* == sigma_hat, a single constant per column applied to
        # every replica) rather than the paper's preferred per-replica
        # sigma_hat*,m (Algorithm 4.2 step 4a, Remark in Sec.4.1 footnote 21).
        # With a single constant divisor, post-division std MUST be exactly
        # 1.0 by construction — that exactness is itself the signature of
        # which variant is running, not evidence that studentization is doing
        # its full per-replica job.
        post_std = studentized_deviations.std(axis=0, ddof=1)
        studentization_ok = bool(np.allclose(post_std, 1.0, atol=1e-3))
        logger.debug(
            f"VERIFY[studentization] {progress_label} ── post-division std per column: "
            f"min={post_std.min():.6f} max={post_std.max():.6f} (expected ≡ 1.0 exactly "
            f"under Hansen-style constant sigma_hat*, NOT under the paper's per-replica "
            f"sigma_hat*,m) ── {'✅' if studentization_ok else '❌'}"
        )

        pct_z = np.percentile(z_stat, [0, 50, 90, 99, 100])
        logger.debug(
            f"DESCRIBE[z_stat] {progress_label} ── studentized statistic percentiles "
            f"[min,p50,p90,p99,max] = "
            f"[{pct_z[0]:.4f}, {pct_z[1]:.4f}, {pct_z[2]:.4f}, {pct_z[3]:.4f}, {pct_z[4]:.4f}]"
        )

    logger.debug(
        f"FUNNEL {progress_label} ── built={n_cols_built} → "
        f"after_real_variance_filter={n_cols} → "
        f"after_bootstrap_se_filter={kept_columns.shape[0]} "
        f"(survival rate={kept_columns.shape[0] / n_cols_built:.2%})"
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
# Always k=1 (max) by definition — this is White's original BRC, unrelated to
# the STEPM_K_FWE setting used inside the stepdown below.
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
# k-th LARGEST ACROSS ACTIVE COLUMNS, PER BOOTSTRAP REPLICA
# k=1 reduces to np.max exactly (same values, same dtype) — this is what
# guarantees STEPM_K_FWE=1 reproduces the original strict-FWE numbers.
# =============================================================================
def _kth_largest_active(deviations_active: np.ndarray, k: int) -> np.ndarray:

    n_active = deviations_active.shape[1]
    k_eff = min(k, n_active)

    if k_eff == 1:
        return np.max(deviations_active, axis=1)

    # np.partition(-x, k_eff-1)[:, k_eff-1] gives the (k_eff-1)-th smallest of
    # -x, i.e. the k_eff-th largest of x, without a full sort.
    neg_partitioned = np.partition(-deviations_active, k_eff - 1, axis=1)
    return -neg_partitioned[:, k_eff - 1]


# =============================================================================
# STEPM (ROMANO & WOLF, 2005) — stepdown per-rule p-values controlling FWER
# (k=1) or k-FWE (k>1, reasoned extension — see module docstring).
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

    n_cols   = statistic.shape[0]
    active   = np.ones(n_cols, dtype=bool)
    raw_pval = np.full(n_cols, np.nan, dtype=np.float64)

    for _iteration in range(max_iterations):
        active_idx = np.flatnonzero(active)
        if active_idx.size == 0:
            break

        if k > active_idx.size:
            logger.debug(
                f"STEPDOWN iter={_iteration} ── requested k={k} exceeds active set "
                f"size={active_idx.size} ── clamping to k_eff={active_idx.size} "
                f"(k-FWE degenerates toward the global minimum in this iteration)"
            )

        kth_dev_active = _kth_largest_active(deviations[:, active_idx], k)  # (n_bootstrap,)

        candidate_p = (
            kth_dev_active[:, None] >= statistic[active_idx][None, :]
        ).mean(axis=0)

        # Rejection is decided directly from the bootstrap p-value against
        # alpha (reject iff candidate_p <= alpha). With k=1 this is the same
        # criterion as comparing the statistic to the (1-alpha) quantile of
        # max_dev_active (Algorithm 4.1 step 3), without the extra step of
        # picking a quantile interpolation rule, and it ties the active-set
        # construction to the same alpha used for the final pass/fail
        # decision (see WHITE_PVALUE_TH). With k>1 the same mechanics apply
        # to the k-th largest active deviation instead of the max.
        reject_local = candidate_p <= alpha

        logger.debug(
            f"STEPDOWN iter={_iteration} ── k={k} ── active={active_idx.size} ── "
            f"rejected_this_iter={int(reject_local.sum())} ── "
            f"candidate_p range=[{candidate_p.min():.4f}, {candidate_p.max():.4f}]"
        )

        if STEPM_VERIFY and _iteration == 0:
            # DESCRIBE[kth_dev_active] — the actual bar every column has to
            # clear in the first step, for the configured k. Comparing this
            # to DESCRIBE[z_stat] tells you directly how much k relaxes the
            # bar relative to k=1 (strict FWE).
            pct_dev = np.percentile(kth_dev_active, [0, 50, 90, 99, 100])
            logger.debug(
                f"DESCRIBE[kth_dev_active] iter0 (k={k}) ── percentiles "
                f"[min,p50,p90,p99,max] = "
                f"[{pct_dev[0]:.4f}, {pct_dev[1]:.4f}, {pct_dev[2]:.4f}, "
                f"{pct_dev[3]:.4f}, {pct_dev[4]:.4f}]"
            )

            # VERIFY[pvalue_quantile_equivalence] — the p-value rule
            # (candidate_p <= alpha) must agree with inverting the (1-alpha)
            # quantile of kth_dev_active against the statistic (statistic >
            # quantile). This holds by construction for any k, since both
            # sides are built from the same kth_dev_active vector — it is not
            # specific to k=1.
            quantile_val      = np.quantile(kth_dev_active, 1.0 - alpha)
            predicted_reject  = statistic[active_idx] > quantile_val
            mismatches        = int(np.sum(predicted_reject != reject_local))
            mismatch_rate     = mismatches / max(active_idx.size, 1)
            logger.debug(
                f"VERIFY[pvalue_quantile_equivalence] iter0 (k={k}) ── mismatches between "
                f"p-value rule and quantile-inversion rule = {mismatches}/{active_idx.size} "
                f"({mismatch_rate:.4%}) ── {'✅' if mismatch_rate < 0.01 else '❌'}"
            )

        if not reject_local.any():
            raw_pval[active_idx] = candidate_p
            break

        rejected_idx = active_idx[reject_local]
        raw_pval[rejected_idx] = candidate_p[reject_local]
        active[rejected_idx] = False
    else:
        # max_iterations exhausted without the active set reaching a fixed
        # point. Any column left as NaN here must not be allowed to silently
        # become p=0.0 in the monotonization step below, so we fail loudly.
        unresolved = np.flatnonzero(np.isnan(raw_pval))
        if unresolved.size > 0:
            raise RuntimeError(
                f"StepM stepdown did not converge within {max_iterations} "
                f"iterations; unresolved column indices: {unresolved.tolist()}"
            )

    order = np.argsort(-statistic)
    running_max = 0.0
    adjusted_pval = np.empty(n_cols, dtype=np.float64)
    for idx in order:
        val = raw_pval[idx]
        if not np.isfinite(val):
            raise RuntimeError(
                f"StepM stepdown produced an undefined raw p-value for column "
                f"index {idx}; refusing to silently treat it as zero."
            )
        running_max = max(running_max, val)
        adjusted_pval[idx] = running_max

    if STEPM_VERIFY:
        # VERIFY[monotonicity] — adjusted p-values must be non-decreasing when
        # read in descending-statistic order. This is the defining property
        # of the running-max monotonization step and holds regardless of k;
        # if it fails, the adjusted p-values are not valid stepdown p-values.
        ordered_vals = adjusted_pval[order]
        diffs = np.diff(ordered_vals)
        monotonic_ok = bool(np.all(diffs >= -1e-9))
        min_diff = float(diffs.min()) if diffs.size else float("nan")
        logger.debug(
            f"VERIFY[monotonicity] (k={k}) ── adjusted p-values non-decreasing along "
            f"descending-statistic order ── {'✅' if monotonic_ok else '❌'} "
            f"(min diff={min_diff:.2e})"
        )

    return adjusted_pval


# =============================================================================
# PIPE STEPM — orchestration layer, mirroring dsr.py's pipe_dsr exactly.
# =============================================================================
def empty_stepm_fields() -> dict:
    """Placeholder StepM fields for rules that were never evaluated (pipe skipped)."""
    return {
        "passed_stepm": True,
        "stepm_p":      None,
    }


def pipe_stepm(
    raw_results: list,
    stepm_alpha: float = None,
    stepm_pvalue_th: float = None,
    n_bootstrap: int = None,
    block_size: int = None,
    n_jobs: int = -1,
    timeframe: str = "",
) -> list:

    stepm_alpha     = stepm_alpha     if stepm_alpha     is not None else STEPM_ALPHA
    stepm_pvalue_th = stepm_pvalue_th if stepm_pvalue_th is not None else stepm_alpha
    n_bootstrap     = n_bootstrap     if n_bootstrap     is not None else WHITE_N_BOOTSTRAP
    block_size      = block_size      if block_size      is not None else WHITE_BLOCK_SIZE
    # k-FWE level is intentionally NOT a pipe_stepm parameter — it is read
    # directly from the module constant below, so changing it only ever
    # requires editing STEPM_K_FWE at the top of this file, never main.py.
    k_fwe           = STEPM_K_FWE

    if not np.isclose(stepm_pvalue_th, stepm_alpha):
        raise ValueError(
            f"stepm_pvalue_th ({stepm_pvalue_th}) must equal stepm_alpha "
            f"({stepm_alpha}). The FWE guarantee of Algorithm 4.1 only holds "
            "when the pass/fail threshold matches the alpha used to build "
            "the stepdown active sets — decoupling them invalidates the FWE "
            "control for both values."
        )

    start = time.time()

    matrix = build_flat_daily_matrix(raw_results)
    if matrix is None:
        logger.warning(f"STEPM ── {timeframe} ── insufficient data — skipping, passing all rules through untouched")
        return [{**r, **empty_stepm_fields()} for r in raw_results]

    if matrix.shape[1] < 2:
        logger.warning(f"STEPM ── {timeframe} ── insufficient columns — skipping, passing all rules through untouched")
        return [{**r, **empty_stepm_fields()} for r in raw_results]

    bootstrap_result = compute_deviation_matrix(
        matrix, n_bootstrap=n_bootstrap, block_size=block_size,
        n_jobs=n_jobs, progress_label=timeframe,
    )
    kept_columns            = bootstrap_result["kept_columns"]
    real_sharpe             = bootstrap_result["real_sharpe"]
    sigma_hat               = bootstrap_result["sigma_hat"]
    studentized_deviations  = bootstrap_result["studentized_deviations"]
    z_stat                  = bootstrap_result["z_stat"]

    logger.info(
        f"STEPM ── {timeframe} ── {matrix.shape[1] - len(kept_columns)} degenerate "
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

    logger.info(f"STEPM ── {timeframe} ── k-FWE level k={k_fwe}" + (" (strict FWE)" if k_fwe == 1 else " (relaxed control — reasoned extension, see module docstring)"))

    stepm_pvals    = stepwise_reality_check_pvalues(studentized_deviations, z_stat, alpha=stepm_alpha, k=k_fwe)
    stepm_p_by_col = dict(zip(kept_columns, stepm_pvals))

    if STEPM_VERIFY:
        if k_fwe == 1:
            # VERIFY[BRC_equivalence] — only holds under strict FWE (k=1): the
            # BRC (White 2000) is exactly the first step of StepM restricted
            # to the single best statistic (Romano & Wolf 2005, Sec.3). Under
            # k-FWE (k>1) this equivalence does not hold by construction,
            # since the reference statistic is no longer the max.
            p_from_stepm = float(stepm_p_by_col.get(best_col_name, float("nan")))
            brc_match = bool(np.isclose(p_from_stepm, global_result["global_p"], atol=1e-9))
            logger.debug(
                f"VERIFY[BRC_equivalence] {timeframe} (k={k_fwe}) ── global White p-value = "
                f"{global_result['global_p']:.6f} vs StepM p-value of the same best column = "
                f"{p_from_stepm:.6f} ── {'✅' if brc_match else '❌'}"
            )
        else:
            logger.debug(
                f"VERIFY[BRC_equivalence] {timeframe} ── skipped: not applicable under "
                f"k-FWE (k={k_fwe} > 1) by construction"
            )

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
            "stepm_p":      float(stepm_p) if np.isfinite(stepm_p) else None,
        })

    logger.info(f"STEPM ── {timeframe} ── k={k_fwe} ── {n_passed}/{len(raw_results)} rules pass")

    elapsed = int(time.time() - start)
    logger.info(f"STEPM ── {timeframe} ── elapsed {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")

    return results