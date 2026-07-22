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
# WHITE REALITY CHECK CONFIG (moving block bootstrap, same style as montecarlo.py)
# =============================================================================
WHITE_PVALUE_TH      = 0.05
WHITE_N_BOOTSTRAP    = 1000
WHITE_BLOCK_SIZE     = 20      # fixed block length — mirrors montecarlo.py BLOCK_SIZE
SHARPE_PERIODS_YEAR  = 365.0   # must match compute_metrics annualization factor
RANDOM_SEED          = 42
TOP_N_TABLE          = 100

# Sharpe-outlier column filtering — mirrors dsr.py's DSR_MAX_SHARPE_ANN filter.
# The DSR_MIN_TRADES filter is NOT duplicated here: combos with too few trades
# are already dropped upstream in dsr.py's _evaluate_combo_sharpe (their
# daily_profit is set to None there), so they never reach combo_daily_profit
# and never appear as columns in this matrix in the first place.
WHITE_MAX_SHARPE = 10.0  # matches dsr.py DSR_MAX_SHARPE_ANN


# =============================================================================
# DAILY MATRIX CONSTRUCTION — T days x M trials (rule x combo)
# =============================================================================
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
        np.concatenate([s.index.to_numpy() for s in series_by_col.values()])
    )

    matrix_arr = np.zeros((all_dates.shape[0], len(col_names)), dtype=np.float64)
    for col_idx, col_name in enumerate(col_names):
        s = series_by_col[col_name]
        row_idx = np.searchsorted(all_dates, s.index.to_numpy())
        matrix_arr[row_idx, col_idx] = s.to_numpy(dtype=np.float64)

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

    keep_mask  = sharpe <= max_sharpe
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
    """Vectorized moving block bootstrap index generation.

    Returns an (n_replicas, n_obs) array of resampled day indices. Blocks are
    fixed-length, drawn with replacement from all overlapping windows of the
    day axis, then concatenated and truncated to n_obs — mirroring the
    sliding_window_view + random block selection used in montecarlo.py.
    """
    n_blocks_needed = int(np.ceil(n_obs / block_size))
    n_block_starts   = n_obs - block_size + 1

    chosen_starts = rng.integers(0, n_block_starts, size=(n_replicas, n_blocks_needed))

    block_offsets = np.arange(block_size)[None, None, :]
    indices = chosen_starts[:, :, None] + block_offsets           # (n_replicas, n_blocks_needed, block_size)
    indices = indices.reshape(n_replicas, n_blocks_needed * block_size)[:, :n_obs]

    return indices


def _max_deviation_for_column(
    col_values: np.ndarray, boot_idx: np.ndarray, real_sharpe_col: float
) -> np.ndarray:
    """Sharpe of one column across all bootstrap replicas at once -> (n_replicas,)."""
    boot_samples = col_values[boot_idx]  # shape (n_replicas, n_obs)

    means = boot_samples.mean(axis=1)
    stds  = boot_samples.std(axis=1, ddof=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        boot_sharpe = (means / stds) * np.sqrt(SHARPE_PERIODS_YEAR)
    boot_sharpe = np.where(stds > 0, boot_sharpe, -np.inf)

    return boot_sharpe - real_sharpe_col


# =============================================================================
# CORE — WHITE (2000) BOOTSTRAP REALITY CHECK, single-step FWER-adjusted p-values
# =============================================================================
def _compute_white_p_values(
    matrix: pd.DataFrame,
    n_bootstrap: int = WHITE_N_BOOTSTRAP,
    block_size: int = WHITE_BLOCK_SIZE,
    seed: int = RANDOM_SEED,
    n_jobs: int = -1,
    progress_label: str = "",
) -> dict:

    matrix_arr = matrix.to_numpy(dtype=np.float64)
    n_obs      = matrix_arr.shape[0]
    n_cols     = matrix_arr.shape[1]

    real_sharpe = _sharpe_per_column(matrix_arr)

    rng      = np.random.default_rng(seed)
    boot_idx = _moving_block_bootstrap_indices_batch(n_obs, block_size, n_bootstrap, rng)

    desc = f"WHITE REALITY CHECK {progress_label}".strip()
    with tqdm_joblib(tqdm(desc=desc, total=n_cols, dynamic_ncols=True)):
        deviations_per_col = Parallel(n_jobs=n_jobs)(
            delayed(_max_deviation_for_column)(matrix_arr[:, col], boot_idx, real_sharpe[col])
            for col in range(n_cols)
        )

    deviations    = np.stack(deviations_per_col, axis=1)  # shape (n_bootstrap, n_cols)
    max_deviation = np.max(np.where(np.isfinite(deviations), deviations, -np.inf), axis=1)

    p_values = {
        col: float(np.mean(max_deviation >= sharpe_k))
        for col, sharpe_k in zip(matrix.columns, real_sharpe)
    }
    return p_values


# =============================================================================
# TOP-N TABLE — rules with best (lowest) White p-value, per timeframe
# =============================================================================
def _print_top_white_table(
    rules_tf: list,
    white_p_by_col: dict,
    dsr_th: float,
    white_pvalue_th: float,
    timeframe: str,
    top_n: int = TOP_N_TABLE,
) -> None:

    rows = []
    for r in rules_tf:
        best_combo_id = r.get("best_combo_id")
        col_name      = f"{r['rule_id']}__{best_combo_id}" if best_combo_id else None

        white_p = white_p_by_col.get(col_name, float("nan"))
        if not np.isfinite(white_p):
            continue

        dsr_val = r.get("dsr", 0.0)
        rows.append((r["rule_id"], white_p, dsr_val))

    if not rows:
        return

    rows.sort(key=lambda row: row[1])
    rows = rows[:top_n]

    id_width = max((len(row[0]) for row in rows), default=8) + 2

    logger.info(f"\n{'─' * 70}")
    logger.info(f"  WHITE REALITY CHECK ── {timeframe} ── top {len(rows)} by p-value")
    logger.info(f"{'─' * 70}")
    logger.info(f"{'RULE_ID':<{id_width}}{'WHITE_p':<10}{'WHITE_OK':<10}{'DSR':<10}{'DSR_OK':<10}")
    logger.info(f"{'─' * 70}")

    for rule_id, white_p, dsr_val in rows:
        white_ok   = white_p <= white_pvalue_th
        dsr_ok     = dsr_val >= dsr_th
        white_mark = "✅" if white_ok else "❌"
        dsr_mark   = "✅" if dsr_ok else "❌"
        logger.info(
            f"{rule_id:<{id_width}}{white_p:<10.4f}{white_mark:<10}{dsr_val:<10.4f}{dsr_mark:<10}"
        )

    logger.info(f"{'─' * 70}\n")


# =============================================================================
# CORRELATION — DSR vs WHITE p-value, all rules AND excluding WHITE_p=1.0
# =============================================================================
def _correlation_stats(dsr_vals: list, white_vals: list) -> tuple:
    dsr_arr   = np.asarray(dsr_vals, dtype=np.float64)
    white_arr = np.asarray(white_vals, dtype=np.float64)

    pearson_r, pearson_p   = pearsonr(dsr_arr, white_arr)
    spearman_r, spearman_p = spearmanr(dsr_arr, white_arr)

    return pearson_r, pearson_p, spearman_r, spearman_p, len(dsr_arr)


def _print_dsr_white_correlation(
    rules_tf: list,
    white_p_by_col: dict,
    timeframe: str,
) -> None:

    dsr_all,     white_all     = [], []
    dsr_no_ones, white_no_ones = [], []

    for r in rules_tf:
        best_combo_id = r.get("best_combo_id")
        col_name      = f"{r['rule_id']}__{best_combo_id}" if best_combo_id else None

        white_p = white_p_by_col.get(col_name, float("nan"))
        if not np.isfinite(white_p):
            continue

        dsr_val = r.get("dsr", 0.0)
        dsr_all.append(dsr_val)
        white_all.append(white_p)

        if white_p < 1.0:
            dsr_no_ones.append(dsr_val)
            white_no_ones.append(white_p)

    if len(dsr_all) < 3:
        logger.warning(f"CORRELATION ── {timeframe} ── not enough rules to compute correlation")
        return

    logger.info(f"\n{'─' * 70}")
    logger.info(f"  DSR vs WHITE_p CORRELATION ── {timeframe}")
    logger.info(f"{'─' * 70}")

    pearson_r, pearson_p, spearman_r, spearman_p, n = _correlation_stats(dsr_all, white_all)
    logger.info(f"  [ALL RULES]              n={n}")
    logger.info(f"    Pearson  r = {pearson_r:.4f}  (p={pearson_p:.4g})")
    logger.info(f"    Spearman r = {spearman_r:.4f}  (p={spearman_p:.4g})")

    if len(dsr_no_ones) >= 3:
        pearson_r, pearson_p, spearman_r, spearman_p, n = _correlation_stats(dsr_no_ones, white_no_ones)
        logger.info(f"  [EXCLUDING WHITE_p=1.0]  n={n}")
        logger.info(f"    Pearson  r = {pearson_r:.4f}  (p={pearson_p:.4g})")
        logger.info(f"    Spearman r = {spearman_r:.4f}  (p={spearman_p:.4g})")
    else:
        logger.info(f"  [EXCLUDING WHITE_p=1.0]  not enough rules (n={len(dsr_no_ones)})")

    logger.info(f"{'─' * 70}\n")


# =============================================================================
# PER-TIMEFRAME SUMMARY — DSR vs WHITE, pass rates + agreement + both-pass
# =============================================================================
def _summarize_timeframe(
    rules_tf: list,
    dsr_th: float,
    white_pvalue_th: float,
    n_bootstrap: int,
    block_size: int,
    timeframe: str,
) -> None:

    matrix = _build_flat_daily_matrix(rules_tf)
    if matrix is None:
        logger.warning(f"WHITE REALITY CHECK ── {timeframe} ── insufficient data — skipping")
        return

    matrix, n_filtered = _filter_matrix_columns(matrix)
    logger.info(
        f"WHITE REALITY CHECK ── {timeframe} ── filtered {n_filtered} outlier "
        f"columns (sharpe>{WHITE_MAX_SHARPE}) ── {matrix.shape[1]} columns remain"
    )
    if matrix.shape[1] < 2:
        logger.warning(f"WHITE REALITY CHECK ── {timeframe} ── insufficient columns after filtering — skipping")
        return

    white_p_by_col = _compute_white_p_values(
        matrix, n_bootstrap=n_bootstrap, block_size=block_size, progress_label=timeframe,
    )

    _print_top_white_table(rules_tf, white_p_by_col, dsr_th, white_pvalue_th, timeframe)
    _print_dsr_white_correlation(rules_tf, white_p_by_col, timeframe)

    n_total        = len(rules_tf)
    n_dsr_passed   = 0
    n_white_passed = 0
    n_agreement    = 0
    n_both_passed  = 0

    for r in rules_tf:
        best_combo_id = r.get("best_combo_id")
        col_name      = f"{r['rule_id']}__{best_combo_id}" if best_combo_id else None

        dsr_ok   = r.get("dsr", 0.0) >= dsr_th
        white_p  = white_p_by_col.get(col_name, float("nan"))
        white_ok = white_p <= white_pvalue_th if np.isfinite(white_p) else False

        n_dsr_passed   += int(dsr_ok)
        n_white_passed += int(white_ok)
        n_agreement    += int(dsr_ok == white_ok)
        n_both_passed  += int(dsr_ok and white_ok)

    pct_dsr       = n_dsr_passed   / n_total * 100.0 if n_total else 0.0
    pct_white     = n_white_passed / n_total * 100.0 if n_total else 0.0
    pct_agreement = n_agreement    / n_total * 100.0 if n_total else 0.0
    pct_both      = n_both_passed  / n_total * 100.0 if n_total else 0.0

    logger.info(f"\n{'─' * 70}")
    logger.info(f"  DSR vs WHITE REALITY CHECK ── {timeframe} ── summary")
    logger.info(f"{'─' * 70}")
    logger.info(f"  DSR        ── {n_dsr_passed}/{n_total} passed ({pct_dsr:.2f}%)")
    logger.info(f"  WHITE      ── {n_white_passed}/{n_total} passed ({pct_white:.2f}%)")
    logger.info(f"  AGREEMENT  ── {n_agreement}/{n_total} rules match ({pct_agreement:.2f}%)")
    logger.info(f"  OK-OK      ── {n_both_passed}/{n_total} both pass ({pct_both:.2f}%)")
    logger.info(f"{'─' * 70}\n")


def print_comparison_table(
    all_raw_results: list,
    dsr_th: float,
    white_pvalue_th: float = WHITE_PVALUE_TH,
    n_bootstrap: int = WHITE_N_BOOTSTRAP,
    block_size: int = WHITE_BLOCK_SIZE,
) -> None:

    timeframes = sorted({r["timeframe"] for r in all_raw_results})

    for timeframe in timeframes:
        rules_tf = [r for r in all_raw_results if r["timeframe"] == timeframe]
        _summarize_timeframe(
            rules_tf         = rules_tf,
            dsr_th           = dsr_th,
            white_pvalue_th  = white_pvalue_th,
            n_bootstrap      = n_bootstrap,
            block_size       = block_size,
            timeframe        = timeframe,
        )