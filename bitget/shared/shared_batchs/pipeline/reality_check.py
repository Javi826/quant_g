#shared_batchs/pipeline/reality_check.py
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import pearsonr, spearmanr
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

logger = logging.getLogger("BOT_batch.pipeline.reality_check")

# =============================================================================
# WHITE REALITY CHECK / STEPM CONFIG (moving block bootstrap, same style as montecarlo.py)
# =============================================================================
WHITE_PVALUE_TH      = 0.1    # threshold used only for reporting / comparison against DSR
STEPM_ALPHA          = 0.1    # significance level used inside the Romano-Wolf stepdown search
WHITE_N_BOOTSTRAP     = 1000
WHITE_BLOCK_SIZE      = 20     # fixed block length — mirrors montecarlo.py BLOCK_SIZE
SHARPE_PERIODS_YEAR   = 365.0  # must match compute_metrics annualization factor
RANDOM_SEED           = 42
TOP_N_TABLE           = 10
STEPM_MAX_ITERATIONS  = 500    # safety cap on stepdown iterations


BOOTSTRAP_BATCH_SIZE = 100

WHITE_MAX_SHARPE = 10.0  # matches dsr.py DSR_MAX_SHARPE_ANN


# =============================================================================
# DAILY MATRIX CONSTRUCTION — T days x M trials (rule x combo)
# =============================================================================
def _column_dates(column: tuple) -> np.ndarray:

    day_offsets, _values, start_day = column
    return start_day + day_offsets.astype("timedelta64[D]")


def _build_flat_daily_matrix(rules: list) -> pd.DataFrame | None:

    series_by_col = {}
    for r in rules:
        combo_profit = r.get("combo_daily_profit") or {}
        for combo_id, s in combo_profit.items():
            series_by_col[f"{r['rule_id']}__{combo_id}"] = s

    if len(series_by_col) < 2:
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
# SHARPE-OUTLIER COLUMN FILTERING — before running the bootstrap
# =============================================================================
def _filter_matrix_columns(
    matrix: pd.DataFrame,
    max_sharpe: float = WHITE_MAX_SHARPE,
) -> tuple:

    matrix_arr = matrix.to_numpy(dtype=np.float64)
    sharpe     = _sharpe_per_column(matrix_arr)

    keep_mask  = np.isfinite(sharpe) & (sharpe <= max_sharpe)
    n_excluded = int((~keep_mask).sum())

    return matrix.loc[:, keep_mask], n_excluded


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


def _compute_deviation_matrix(
    matrix: pd.DataFrame,
    n_bootstrap: int = WHITE_N_BOOTSTRAP,
    block_size: int = WHITE_BLOCK_SIZE,
    seed: int = RANDOM_SEED,
    n_jobs: int = -1,
    progress_label: str = "",
) -> dict:

    matrix_arr  = matrix.to_numpy(dtype=np.float64)
    real_sharpe = _sharpe_per_column(matrix_arr)

    finite_mask  = np.isfinite(real_sharpe)
    matrix_arr   = matrix_arr[:, finite_mask]
    real_sharpe  = real_sharpe[finite_mask]
    kept_columns = matrix.columns[finite_mask]

    n_obs  = matrix_arr.shape[0]
    n_cols = matrix_arr.shape[1]

    rng      = np.random.default_rng(seed)
    boot_idx = _moving_block_bootstrap_indices_batch(n_obs, block_size, n_bootstrap, rng)


    matrix_arr32 = matrix_arr.astype(np.float32)

    n_batches = int(np.ceil(n_cols / BOOTSTRAP_BATCH_SIZE))
    batch_bounds = [
        (i * BOOTSTRAP_BATCH_SIZE, min((i + 1) * BOOTSTRAP_BATCH_SIZE, n_cols))
        for i in range(n_batches)
    ]

    desc = f"REALITY CHECK BOOTSTRAP {progress_label} ({BOOTSTRAP_BATCH_SIZE} cols/batch)".strip()
    with tqdm_joblib(tqdm(desc=desc, total=n_batches, dynamic_ncols=True)):
        deviations_per_batch = Parallel(n_jobs=n_jobs)(
            delayed(_bootstrap_deviations_batch)(
                matrix_arr32[:, start:end], boot_idx, real_sharpe[start:end]
            )
            for start, end in batch_bounds
        )

    deviations = np.concatenate(deviations_per_batch, axis=1)  # shape (n_bootstrap, n_cols)

    # ---- Studentization (Romano & Wolf, 2005, Sec. 4 / Hansen, 2004 simplification) ----
    sigma_hat = deviations.std(axis=0, ddof=1)  # bootstrap SE of the Sharpe statistic per column

    valid_se     = sigma_hat > 0
    deviations   = deviations[:, valid_se]
    real_sharpe  = real_sharpe[valid_se]
    sigma_hat    = sigma_hat[valid_se]
    kept_columns = kept_columns[valid_se]

    studentized_deviations = deviations / sigma_hat[None, :]
    z_stat                 = real_sharpe / sigma_hat

    return {
        "deviations":             deviations,
        "real_sharpe":            real_sharpe,
        "sigma_hat":              sigma_hat,
        "studentized_deviations": studentized_deviations,
        "z_stat":                 z_stat,
        "kept_columns":           kept_columns,
    }


# =============================================================================
# GLOBAL P-VALUE — single number per timeframe, the original White (2000) test:
# =============================================================================
def _compute_global_pvalue(deviations: np.ndarray, statistic: np.ndarray) -> dict:

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
# STEPM (ROMANO & WOLF, 2005) — stepdown per-rule p-values controlling FWER.
# =============================================================================
def _stepwise_reality_check_pvalues(
    deviations: np.ndarray,
    statistic: np.ndarray,
    alpha: float = STEPM_ALPHA,
    max_iterations: int = STEPM_MAX_ITERATIONS,
) -> np.ndarray:

    n_cols   = statistic.shape[0]
    active   = np.ones(n_cols, dtype=bool)
    raw_pval = np.full(n_cols, np.nan, dtype=np.float64)

    for _iteration in range(max_iterations):
        active_idx = np.flatnonzero(active)
        if active_idx.size == 0:
            break

        max_dev_active = np.max(deviations[:, active_idx], axis=1)  # (n_bootstrap,)

        # Vectorized one-sided empirical p-value for every active column
        # against the shrinking max distribution: P(max_dev_active >= statistic_i)
        candidate_p = (
            max_dev_active[:, None] >= statistic[active_idx][None, :]
        ).mean(axis=0)

        crit_value = np.quantile(max_dev_active, 1.0 - alpha)
        reject_local = statistic[active_idx] > crit_value

        if not reject_local.any():
            # No further rejections possible: remaining active columns get
            # the p-value implied by this final (largest remaining) active set.
            raw_pval[active_idx] = candidate_p
            break

        rejected_idx = active_idx[reject_local]
        raw_pval[rejected_idx] = candidate_p[reject_local]
        active[rejected_idx] = False

    # Enforce monotonicity along the rejection order (descending statistic).
    order = np.argsort(-statistic)
    running_max = 0.0
    adjusted_pval = np.empty(n_cols, dtype=np.float64)
    for idx in order:
        running_max = max(running_max, float(raw_pval[idx]))
        adjusted_pval[idx] = running_max

    return adjusted_pval


# =============================================================================
# TOP-N TABLE — rules with best (lowest) StepM p-value, per timeframe
# =============================================================================
def _print_top_stepm_table(
    rules_tf: list,
    stepm_p_by_col: dict,
    dsr_th: float,
    stepm_pvalue_th: float,
    timeframe: str,
    top_n: int = TOP_N_TABLE,
) -> None:

    rows = []
    for r in rules_tf:
        best_combo_id = r.get("best_combo_id")
        col_name      = f"{r['rule_id']}__{best_combo_id}" if best_combo_id else None

        stepm_p = stepm_p_by_col.get(col_name, float("nan"))
        if not np.isfinite(stepm_p):
            continue

        dsr_val = r.get("dsr", 0.0)
        rows.append((r["rule_id"], stepm_p, dsr_val))

    if not rows:
        return

    rows.sort(key=lambda row: row[1])
    rows = rows[:top_n]

    id_width = max((len(row[0]) for row in rows), default=8) + 2

    logger.info(f"\n{'─' * 70}")
    logger.info(f"  STEPM REALITY CHECK ── {timeframe} ── top {len(rows)} by p-value")
    logger.info(f"{'─' * 70}")
    logger.info(f"{'RULE_ID':<{id_width}}{'STEPM_p':<10}{'STEPM_OK':<10}{'DSR':<10}{'DSR_OK':<10}")
    logger.info(f"{'─' * 70}")

    for rule_id, stepm_p, dsr_val in rows:
        stepm_ok   = stepm_p <= stepm_pvalue_th
        dsr_ok     = dsr_val >= dsr_th
        stepm_mark = "✅" if stepm_ok else "❌"
        dsr_mark   = "✅" if dsr_ok else "❌"
        logger.info(
            f"{rule_id:<{id_width}}{stepm_p:<10.4f}{stepm_mark:<10}{dsr_val:<10.4f}{dsr_mark:<10}"
        )

    logger.info(f"{'─' * 70}\n")


# =============================================================================
# CORRELATION — DSR vs STEPM p-value, all rules AND excluding STEPM_p=1.0
# =============================================================================
def _correlation_stats(dsr_vals: list, stepm_vals: list) -> tuple:
    dsr_arr   = np.asarray(dsr_vals, dtype=np.float64)
    stepm_arr = np.asarray(stepm_vals, dtype=np.float64)

    pearson_r, pearson_p   = pearsonr(dsr_arr, stepm_arr)
    spearman_r, spearman_p = spearmanr(dsr_arr, stepm_arr)

    return pearson_r, pearson_p, spearman_r, spearman_p, len(dsr_arr)


def _print_dsr_stepm_correlation(
    rules_tf: list,
    stepm_p_by_col: dict,
    timeframe: str,
) -> None:

    dsr_all,     stepm_all     = [], []
    dsr_no_ones, stepm_no_ones = [], []

    for r in rules_tf:
        best_combo_id = r.get("best_combo_id")
        col_name      = f"{r['rule_id']}__{best_combo_id}" if best_combo_id else None

        stepm_p = stepm_p_by_col.get(col_name, float("nan"))
        if not np.isfinite(stepm_p):
            continue

        dsr_val = r.get("dsr", 0.0)
        dsr_all.append(dsr_val)
        stepm_all.append(stepm_p)

        if stepm_p < 1.0:
            dsr_no_ones.append(dsr_val)
            stepm_no_ones.append(stepm_p)

    if len(dsr_all) < 3:
        logger.warning(f"CORRELATION ── {timeframe} ── not enough rules to compute correlation")
        return

    logger.info(f"\n{'─' * 70}")
    logger.info(f"  DSR vs STEPM_p CORRELATION ── {timeframe}")
    logger.info(f"{'─' * 70}")

    pearson_r, pearson_p, spearman_r, spearman_p, n = _correlation_stats(dsr_all, stepm_all)
    logger.info(f"  [ALL RULES]              n={n}")
    logger.info(f"    Pearson  r = {pearson_r:.4f}  (p={pearson_p:.4g})")
    logger.info(f"    Spearman r = {spearman_r:.4f}  (p={spearman_p:.4g})")

    if len(dsr_no_ones) >= 3:
        pearson_r, pearson_p, spearman_r, spearman_p, n = _correlation_stats(dsr_no_ones, stepm_no_ones)
        logger.info(f"  [EXCLUDING STEPM_p=1.0]  n={n}")
        logger.info(f"    Pearson  r = {pearson_r:.4f}  (p={pearson_p:.4g})")
        logger.info(f"    Spearman r = {spearman_r:.4f}  (p={spearman_p:.4g})")
    else:
        logger.info(f"  [EXCLUDING STEPM_p=1.0]  not enough rules (n={len(dsr_no_ones)})")

    logger.info(f"{'─' * 70}\n")


# =============================================================================
# PER-TIMEFRAME SUMMARY — global p-value, DSR vs STEPM pass rates + agreement
# =============================================================================
def _summarize_timeframe(
    rules_tf: list,
    dsr_th: float,
    stepm_pvalue_th: float,
    n_bootstrap: int,
    block_size: int,
    stepm_alpha: float,
    timeframe: str,
) -> None:

    matrix = _build_flat_daily_matrix(rules_tf)
    if matrix is None:
        logger.warning(f"REALITY CHECK ── {timeframe} ── insufficient data — skipping")
        return

    matrix, n_filtered = _filter_matrix_columns(matrix)
    logger.info(
        f"REALITY CHECK ── {timeframe} ── filtered {n_filtered} outlier "
        f"columns (sharpe>{WHITE_MAX_SHARPE} or degenerate) ── {matrix.shape[1]} columns remain"
    )
    if matrix.shape[1] < 2:
        logger.warning(f"REALITY CHECK ── {timeframe} ── insufficient columns after filtering — skipping")
        return

    bootstrap_result = _compute_deviation_matrix(
        matrix, n_bootstrap=n_bootstrap, block_size=block_size, progress_label=timeframe,
    )
    kept_columns           = bootstrap_result["kept_columns"]
    real_sharpe            = bootstrap_result["real_sharpe"]
    sigma_hat              = bootstrap_result["sigma_hat"]
    studentized_deviations = bootstrap_result["studentized_deviations"]
    z_stat                 = bootstrap_result["z_stat"]

    # ---- Global p-value (studentized): is the single best candidate in the
    # universe significant, on a scale comparable across columns with very
    # different trade counts? (Romano & Wolf, 2005, Sec. 4) ----
    global_result = _compute_global_pvalue(studentized_deviations, z_stat)
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

    # ---- StepM per-rule p-values (studentized) ----
    stepm_pvals = _stepwise_reality_check_pvalues(studentized_deviations, z_stat, alpha=stepm_alpha)
    stepm_p_by_col = dict(zip(kept_columns, stepm_pvals))

    _print_top_stepm_table(rules_tf, stepm_p_by_col, dsr_th, stepm_pvalue_th, timeframe)
    _print_dsr_stepm_correlation(rules_tf, stepm_p_by_col, timeframe)

    n_total        = len(rules_tf)
    n_dsr_passed   = 0
    n_stepm_passed = 0
    n_agreement    = 0
    n_both_passed  = 0

    for r in rules_tf:
        best_combo_id = r.get("best_combo_id")
        col_name      = f"{r['rule_id']}__{best_combo_id}" if best_combo_id else None

        dsr_ok   = r.get("dsr", 0.0) >= dsr_th
        stepm_p  = stepm_p_by_col.get(col_name, float("nan"))
        stepm_ok = stepm_p <= stepm_pvalue_th if np.isfinite(stepm_p) else False

        n_dsr_passed   += int(dsr_ok)
        n_stepm_passed += int(stepm_ok)
        n_agreement    += int(dsr_ok == stepm_ok)
        n_both_passed  += int(dsr_ok and stepm_ok)

    pct_dsr       = n_dsr_passed   / n_total * 100.0 if n_total else 0.0
    pct_stepm     = n_stepm_passed / n_total * 100.0 if n_total else 0.0
    pct_agreement = n_agreement    / n_total * 100.0 if n_total else 0.0
    pct_both      = n_both_passed  / n_total * 100.0 if n_total else 0.0

    logger.info(f"\n{'─' * 70}")
    logger.info(f"  DSR vs STEPM REALITY CHECK ── {timeframe} ── summary")
    logger.info(f"{'─' * 70}")
    logger.info(f"  DSR        ── {n_dsr_passed}/{n_total} passed ({pct_dsr:.2f}%)")
    logger.info(f"  STEPM      ── {n_stepm_passed}/{n_total} passed ({pct_stepm:.2f}%)")
    logger.info(f"  AGREEMENT  ── {n_agreement}/{n_total} rules match ({pct_agreement:.2f}%)")
    logger.info(f"  OK-OK      ── {n_both_passed}/{n_total} both pass ({pct_both:.2f}%)")
    logger.info(f"{'─' * 70}\n")


def print_comparison_table(
    all_raw_results: list,
    dsr_th: float,
    stepm_pvalue_th: float = WHITE_PVALUE_TH,
    n_bootstrap: int = WHITE_N_BOOTSTRAP,
    block_size: int = WHITE_BLOCK_SIZE,
    stepm_alpha: float = STEPM_ALPHA,
) -> None:

    timeframes = sorted({r["timeframe"] for r in all_raw_results})

    for timeframe in timeframes:
        rules_tf = [r for r in all_raw_results if r["timeframe"] == timeframe]
        _summarize_timeframe(
            rules_tf         = rules_tf,
            dsr_th           = dsr_th,
            stepm_pvalue_th  = stepm_pvalue_th,
            n_bootstrap      = n_bootstrap,
            block_size       = block_size,
            stepm_alpha      = stepm_alpha,
            timeframe        = timeframe,
        )