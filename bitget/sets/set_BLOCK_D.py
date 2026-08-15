#BOT_batch/set_BLOCK_SPARSITY_DIAG.py
"""
Calendar-time vs trade-time sensitivity check for WHITE_BLOCK_SIZE.

For a subsample of columns, computes the Politis-White b_opt twice:
  (A) CALENDAR — the actual series used by stepM.py's bootstrap, zero-P&L
      days included.
  (B) COMPRESSED — the same column with zero-P&L days removed, keeping only
      the ordered sequence of realized-trade P&L ("trade time").

This is a SENSITIVITY CHECK ONLY. The compressed series is not a valid
substitute for WHITE_BLOCK_SIZE — the production bootstrap always resamples
in calendar time. If (A) and (B) largely agree, zero-inflation is not
distorting the calendar-time estimate. If they diverge a lot, treat the
calendar-time b_opt with more caution (prefer the higher end of its
distribution, or set_BLOCK_DIAG.py's stability read).
"""
import os
import sys
import time
import logging
import numpy as np
from scipy.stats import pearsonr, spearmanr

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_batch")))

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_batch.block_sparsity_diag")
logger.setLevel(logging.INFO)
logging.getLogger("BOT_batch.pipeline.backtest_runner").setLevel(logging.INFO)
logging.getLogger("joblib").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

import set_BLOCK  # reuses universe loading, filtering, and the PW building blocks — __main__ block never runs on import

# =============================================================================
# DIAGNOSTIC CONFIGURATION
# =============================================================================
TIMEFRAMES = set_BLOCK.TIMEFRAMES
N_SYMBOLS  = set_BLOCK.N_SYMBOLS
PARAM_GRID = set_BLOCK.PARAM_GRID

SPARSITY_DIAG_MAX_COLUMNS = 500   # per-column looped FFT — much smaller than the vectorized diagnostic
MIN_ACTIVE_OBS_COMPRESSED = 30    # below this, the compressed series is too short for a stable ACF read


def _politis_white_b_opt_single_column(values: np.ndarray) -> float:
    """Same math as set_BLOCK.method_politis_white, applied to one 1-D series."""

    n_obs = values.shape[0]
    if n_obs < MIN_ACTIVE_OBS_COMPRESSED:
        return float("nan")

    psi, valid = set_BLOCK._sharpe_influence_function(values.reshape(-1, 1))
    if not valid[0]:
        return float("nan")
    psi = np.ascontiguousarray(psi)

    k_n     = int(max(5, np.ceil(np.sqrt(np.log10(n_obs)))))
    m_max   = int(np.ceil(np.sqrt(n_obs)) + k_n)
    b_max   = int(np.ceil(min(3.0 * np.sqrt(n_obs), n_obs / 3.0)))
    lag_max = min(m_max + k_n, n_obs - 1)
    if lag_max < 1:
        return float("nan")

    acov = set_BLOCK._autocovariance_by_column(psi, max_lag=lag_max)
    with np.errstate(divide="ignore", invalid="ignore"):
        acf = acov / acov[0][None, :]
    acf = np.nan_to_num(acf, nan=0.0)

    band       = set_BLOCK.PW_C_SIGNIF * np.sqrt(np.log10(n_obs) / n_obs)
    signif     = (np.abs(acf[1:]) >= band).astype(np.int32)
    cumulative = np.vstack([np.zeros((1, 1), dtype=np.int32), np.cumsum(signif, axis=0)])

    n_windows = signif.shape[0] - k_n + 1
    m_hat     = lag_max
    for window_start in range(max(n_windows, 0)):
        clean = (cumulative[window_start + k_n, 0] - cumulative[window_start, 0]) == 0
        if clean:
            m_hat = window_start + 1
            break

    bandwidth = np.clip(2 * m_hat, 1, m_max)
    lags      = np.arange(1, lag_max + 1, dtype=np.float64)
    weights   = set_BLOCK._flat_top_lag_window(lags, np.array([float(bandwidth)]))

    g_hat = acov[0, 0] + 2.0 * (weights[:, 0] * acov[1:, 0]).sum()
    g_big = 2.0 * (weights[:, 0] * lags * acov[1:, 0]).sum()

    if g_hat <= 0 or not np.isfinite(g_big):
        return float("nan")

    d_mbb = (4.0 / 3.0) * g_hat ** 2
    b_opt = float(np.cbrt(2.0 * g_big ** 2 / d_mbb) * np.cbrt(n_obs))
    return float(np.clip(b_opt, 1.0, b_max))


def compare_calendar_vs_compressed(matrix_arr: np.ndarray, max_columns: int, seed: int) -> dict:

    n_cols = matrix_arr.shape[1]
    n_sample = min(max_columns, n_cols)
    rng = np.random.default_rng(seed)
    sampled_idx = np.sort(rng.choice(n_cols, size=n_sample, replace=False))

    b_calendar   = np.full(n_sample, np.nan, dtype=np.float64)
    b_compressed = np.full(n_sample, np.nan, dtype=np.float64)
    active_fraction = np.full(n_sample, np.nan, dtype=np.float64)

    for i, col_idx in enumerate(sampled_idx):
        column = matrix_arr[:, col_idx]
        active_mask = column != 0.0
        active_fraction[i] = active_mask.mean()

        b_calendar[i]   = _politis_white_b_opt_single_column(column)
        b_compressed[i] = _politis_white_b_opt_single_column(column[active_mask])

    return {
        "active_fraction": active_fraction,
        "b_calendar":      b_calendar,
        "b_compressed":    b_compressed,
    }


def print_sparsity_comparison(result: dict, timeframe: str) -> None:

    active_fraction = result["active_fraction"]
    b_calendar      = result["b_calendar"]
    b_compressed    = result["b_compressed"]

    both_valid = np.isfinite(b_calendar) & np.isfinite(b_compressed)
    n_total    = active_fraction.shape[0]
    n_valid    = int(both_valid.sum())

    logger.info(f"\n{'=' * 90}")
    logger.info(f"  CALENDAR vs COMPRESSED (trade-time) b_opt ── {timeframe}")
    logger.info(f"{'=' * 90}")
    logger.info(f"  sampled columns : {n_total}")
    logger.info(f"  both estimable  : {n_valid}/{n_total} "
                f"(rest skipped — fewer than {MIN_ACTIVE_OBS_COMPRESSED} active days)")

    if n_valid < 3:
        logger.warning("  not enough columns with both estimates to compare — widen the sample or lower the threshold")
        logger.info(f"{'=' * 90}\n")
        return

    cal  = b_calendar[both_valid]
    comp = b_compressed[both_valid]
    ratio = comp / cal

    pearson_r, pearson_p   = pearsonr(cal, comp)
    spearman_r, spearman_p = spearmanr(cal, comp)

    logger.info(f"\n  {'QUANTILE':<12}{'b_calendar':<14}{'b_compressed':<14}{'ratio (comp/cal)':<18}")
    logger.info(f"  {'-' * 58}")
    for q in set_BLOCK.PW_QUANTILES:
        logger.info(
            f"  p{q:<11}{np.percentile(cal, q):<14.2f}"
            f"{np.percentile(comp, q):<14.2f}{np.percentile(ratio, q):<18.3f}"
        )
    logger.info(f"  {'mean':<12}{cal.mean():<14.2f}{comp.mean():<14.2f}{ratio.mean():<18.3f}")

    logger.info(f"\n  correlation (b_calendar vs b_compressed):")
    logger.info(f"    Pearson  r = {pearson_r:.4f}  (p={pearson_p:.4g})")
    logger.info(f"    Spearman r = {spearman_r:.4f}  (p={spearman_p:.4g})")

    logger.info(f"\n  Reading this: ratio near 1.0 and high Spearman r means zero-inflation")
    logger.info(f"  is NOT distorting the calendar-time estimate — trust set_BLOCK.py as is.")
    logger.info(f"  ratio systematically > 1 means calendar-time UNDERESTIMATES b_opt (be more")
    logger.info(f"  conservative); ratio systematically < 1 means it OVERESTIMATES it.")
    logger.info(f"{'=' * 90}\n")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    start = time.time()

    logger.info(f"\n{'=' * 90}")
    logger.info(f"  CALENDAR vs COMPRESSED (trade-time) b_opt SENSITIVITY CHECK")
    logger.info(f"{'=' * 90}")
    logger.info(f"  TIMEFRAMES                 : {TIMEFRAMES}")
    logger.info(f"  N_SYMBOLS                  : {N_SYMBOLS}")
    logger.info(f"  PARAM_GRID                 : {PARAM_GRID}")
    logger.info(f"  SPARSITY_DIAG_MAX_COLUMNS  : {SPARSITY_DIAG_MAX_COLUMNS}")
    logger.info(f"  MIN_ACTIVE_OBS_COMPRESSED  : {MIN_ACTIVE_OBS_COMPRESSED}")
    logger.info(f"{'=' * 90}")

    for timeframe in TIMEFRAMES:
        ohlcv_is  = set_BLOCK.select_universe(
            data_folder_is    = set_BLOCK.DATA_FOLDER_IS,
            timeframe         = timeframe,
            min_price         = set_BLOCK.MIN_PRICE,
            filter_symbols_fn = set_BLOCK.filter_symbols,
        )
        ohlcv_is  = set_BLOCK.select_top_n_by_volume(ohlcv_is, N_SYMBOLS)
        ohlcv_arr = set_BLOCK.prepare_ohlcv_arrays(ohlcv_is)
        rules     = set_BLOCK._build_rule_dicts(ohlcv_is, timeframe, set_BLOCK.RULE_MAX_DEPTH)

        original_n_jobs = set_BLOCK.backtest_module.BACKTEST_N_JOBS
        set_BLOCK.backtest_module.BACKTEST_N_JOBS = set_BLOCK.N_JOBS
        try:
            _raw_results, _n_combos, matrix_arr, _col_names = set_BLOCK.backtest_module.pipe_backtesting(
                rules        = rules,
                ohlcv_arr    = ohlcv_arr,
                param_grid   = PARAM_GRID,
                order_amount = set_BLOCK.ORDER_AMOUNT,
                timeframe    = timeframe,
            )
        finally:
            set_BLOCK.backtest_module.BACKTEST_N_JOBS = original_n_jobs

        if matrix_arr is None or matrix_arr.shape[1] < 2:
            logger.warning(f"  {timeframe} — matrix has fewer than 2 usable columns, skipping")
            continue

        active_matrix, _active_idx, _ = set_BLOCK._filter_active_columns(matrix_arr, set_BLOCK.MIN_ACTIVE_DAYS_FRACTION)
        if active_matrix.shape[1] < 2:
            logger.warning(f"  {timeframe} — fewer than 2 columns survive the activity filter, skipping")
            continue

        logger.info(f"\n  MATRIX ── {timeframe} ── {active_matrix.shape[0]} days x {active_matrix.shape[1]} columns")

        result = compare_calendar_vs_compressed(active_matrix, SPARSITY_DIAG_MAX_COLUMNS, set_BLOCK.RANDOM_SEED)
        print_sparsity_comparison(result, timeframe)

    elapsed = int(time.time() - start)
    logger.info(f"🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")