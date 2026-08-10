#BOT_batch/main_BLOCKSIZE.py
"""
Block-size selection for the StepM moving-block bootstrap (WHITE_BLOCK_SIZE).

Two independent estimators are computed and compared:

  1. POLITIS & WHITE (2004) -- closed-form automatic block-length selection for
     the dependent bootstrap, applied per column and aggregated across columns.
     Cheap. Optimal for variance estimation of the sample mean; blind to the
     multiple-testing geometry that StepM actually depends on.

  2. ROMANO & WOLF (2005), Algorithm 7.1 -- calibration of the realized joint
     coverage probability of the first-step studentized StepM confidence
     region. Expensive, but targets exactly the quantity the FWE guarantee
     rests on: g(b) = P{theta in JCR(b)} should equal 1 - alpha.

DEVIATION FROM ALGORITHM 7.1 AS PUBLISHED
The paper fits a low-order VAR to the observed panel. With thousands of highly
collinear trial columns and only a few hundred daily observations a full VAR(1)
is not identifiable, so the semiparametric model P_tilde used here is a
DIAGONAL VAR(1) (per-column AR(1)) combined with a stationary bootstrap of the
residual VECTORS:
  - marginal serial dependence            -> AR(1) coefficient per column
  - contemporaneous cross-sectional dep.  -> resampling whole residual rows
  - leftover serial dependence            -> geometric block lengths
This mirrors footnote 27 of the paper, which bootstraps VAR residuals with a
stationary bootstrap for exactly this reason.

The backtest is re-run on every invocation; no matrix cache is kept.
"""
import os
import sys
import time
import logging
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_batch")))

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL = logging.INFO
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_batch.main_blocksize")
logger.setLevel(LOG_LEVEL)

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
from shared_batchs.pipeline.stepM import STEPM_ALPHA, WHITE_BLOCK_SIZE, SHARPE_PERIODS_YEAR

# =============================================================================
# UNIVERSE / SEARCH SPACE CONFIGURATION -- keep aligned with main_COMP.py
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
# CALIBRATION CONFIG
# =============================================================================
BLOCK_SIZE_CANDIDATES = [1,5,10,25,50,100,150]

CALIB_N_COLUMNS   = 200    # columns drawn per subsample (random, NOT top-Sharpe)
CALIB_N_SUBSAMPLE = 3      # independent column draws; final answer is their median
CALIB_N_DATASETS  = 100    # M synthetic datasets per subsample (paper suggests many more)
CALIB_N_BOOTSTRAP = 250    # bootstrap replicas inside each JCR construction
CALIB_SEED        = 12345

# Semiparametric model P_tilde
AR1_PHI_CAP        = 0.95  # keeps the fitted AR(1) comfortably stationary
RESID_BLOCK_MEAN   = 5.0   # mean geometric block length for residual resampling
BURN_IN            = 250   # discarded steps before retaining the synthetic path

REPLICA_CHUNK = 100        # bootstrap replicas per gather chunk (memory bound)
MIN_NONZERO_DAYS = 30      # columns with fewer active days are excluded

# =============================================================================
# STATISTIC -- annualized Sharpe per column, matching stepm.py exactly
# =============================================================================
def _annualized_sharpe(values: np.ndarray) -> np.ndarray:
    means = values.mean(axis=0, dtype=np.float64)
    stds  = values.std(axis=0, ddof=1, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        sharpe = (means / stds) * np.sqrt(SHARPE_PERIODS_YEAR)
    return np.where(stds > 0, sharpe, -np.inf)


# =============================================================================
# EMPIRICAL AUTOCORRELATION DIAGNOSTICS
# =============================================================================
def _autocovariance(series: np.ndarray, max_lag: int) -> np.ndarray:
    centered = series - series.mean()
    n_obs    = centered.size
    acov     = np.empty(max_lag + 1, dtype=np.float64)
    for lag in range(max_lag + 1):
        acov[lag] = np.dot(centered[lag:], centered[: n_obs - lag]) / n_obs
    return acov


def empirical_acf_profile(matrix: np.ndarray, max_lag: int = 30) -> dict:
    """Cross-column median absolute autocorrelation, per lag."""
    n_obs   = matrix.shape[0]
    max_lag = min(max_lag, n_obs - 2)
    acf_by_column = []

    for col in range(matrix.shape[1]):
        acov = _autocovariance(matrix[:, col].astype(np.float64), max_lag)
        if acov[0] <= 0:
            continue
        acf_by_column.append(acov[1:] / acov[0])

    if not acf_by_column:
        return {"lags": np.empty(0), "median_abs_acf": np.empty(0), "significance_band": np.nan}

    acf_arr = np.abs(np.vstack(acf_by_column))
    return {
        "lags":              np.arange(1, max_lag + 1),
        "median_abs_acf":    np.median(acf_arr, axis=0),
        "p90_abs_acf":       np.percentile(acf_arr, 90, axis=0),
        "significance_band": 2.0 / np.sqrt(n_obs),
    }


# =============================================================================
# POLITIS & WHITE (2004) -- AUTOMATIC BLOCK LENGTH
# =============================================================================
def _flat_top_kernel(scaled_lags: np.ndarray) -> np.ndarray:
    """Trapezoidal (flat-top) kernel of Politis & Romano (1995)."""
    abs_s  = np.abs(scaled_lags)
    kernel = np.zeros_like(abs_s, dtype=np.float64)
    kernel[abs_s <= 0.5] = 1.0
    taper = (abs_s > 0.5) & (abs_s <= 1.0)
    kernel[taper] = 2.0 * (1.0 - abs_s[taper])
    return kernel


def _correlogram_cutoff(acf: np.ndarray, n_obs: int) -> int:
    """
    Smallest m such that |rho(k)| is negligible for k = m+1, ..., m+K_N.
    Falls back to the largest significant lag when no such m exists.
    """
    threshold = 2.0 * np.sqrt(np.log10(n_obs) / n_obs)
    band      = max(5, int(np.sqrt(np.log10(n_obs))))
    n_lags    = acf.size

    for m in range(n_lags - band):
        if np.all(np.abs(acf[m : m + band]) < threshold):
            return m

    significant = np.flatnonzero(np.abs(acf) >= threshold)
    return int(significant[-1]) + 1 if significant.size else 1


def politis_white_block_size(series: np.ndarray) -> float:
    """
    Closed-form block length minimizing MSE of the block-bootstrap variance
    estimator of the sample mean, for the circular/moving block bootstrap.
    Returns np.nan when the series is degenerate.
    """
    series = np.asarray(series, dtype=np.float64)
    n_obs  = series.size
    if n_obs < 20 or series.std() <= 0:
        return np.nan

    lag_cap = int(np.ceil(np.sqrt(n_obs))) + max(5, int(np.sqrt(np.log10(n_obs))))
    lag_cap = min(lag_cap, n_obs - 2)

    acov = _autocovariance(series, lag_cap)
    if acov[0] <= 0:
        return np.nan
    acf = acov[1:] / acov[0]

    m_hat     = _correlogram_cutoff(acf, n_obs)
    bandwidth = max(1, min(2 * m_hat, int(np.ceil(np.sqrt(n_obs)))))

    lags        = np.arange(-bandwidth, bandwidth + 1)
    abs_lags    = np.abs(lags)
    kernel      = _flat_top_kernel(lags / bandwidth)
    acov_padded = acov[abs_lags]

    spectral_at_zero  = float(np.sum(kernel * acov_padded))
    bias_term         = float(np.sum(kernel * abs_lags * acov_padded))
    variance_constant = (4.0 / 3.0) * spectral_at_zero ** 2

    if variance_constant <= 0 or bias_term == 0:
        return np.nan

    block_size = ((2.0 * bias_term ** 2) / variance_constant) ** (1.0 / 3.0) * n_obs ** (1.0 / 3.0)
    block_cap  = np.ceil(min(3.0 * np.sqrt(n_obs), n_obs / 3.0))
    return float(np.clip(block_size, 1.0, block_cap))


def politis_white_panel(matrix: np.ndarray) -> dict:
    estimates = np.array([politis_white_block_size(matrix[:, c]) for c in range(matrix.shape[1])])
    finite    = estimates[np.isfinite(estimates)]
    if finite.size == 0:
        return {"n_valid": 0, "median": np.nan, "mean": np.nan, "p25": np.nan, "p75": np.nan, "max": np.nan}
    return {
        "n_valid": int(finite.size),
        "median":  float(np.median(finite)),
        "mean":    float(finite.mean()),
        "p25":     float(np.percentile(finite, 25)),
        "p75":     float(np.percentile(finite, 75)),
        "max":     float(finite.max()),
    }


# =============================================================================
# SEMIPARAMETRIC MODEL P_tilde -- DIAGONAL VAR(1) + STATIONARY RESIDUAL BOOTSTRAP
# =============================================================================
def fit_diagonal_var1(matrix: np.ndarray) -> dict:
    """
    Per-column AR(1) by OLS. Returns coefficients, residual rows (which retain
    the contemporaneous cross-sectional dependence) and the analytic stationary
    Sharpe of the fitted model -- the 'true' theta of P_tilde.
    """
    values = matrix.astype(np.float64, copy=False)
    lagged, current = values[:-1], values[1:]

    mean_lagged, mean_current = lagged.mean(axis=0), current.mean(axis=0)
    centered_lagged  = lagged - mean_lagged
    centered_current = current - mean_current

    cross_moment  = (centered_lagged * centered_current).sum(axis=0)
    lagged_moment = (centered_lagged ** 2).sum(axis=0)

    with np.errstate(divide="ignore", invalid="ignore"):
        phi = np.where(lagged_moment > 0, cross_moment / lagged_moment, 0.0)
    phi = np.clip(np.nan_to_num(phi), -AR1_PHI_CAP, AR1_PHI_CAP)

    intercept = mean_current - phi * mean_lagged
    residuals = current - intercept - phi * lagged

    stationary_mean = intercept / (1.0 - phi)
    residual_var    = residuals.var(axis=0, ddof=1)
    stationary_var  = residual_var / (1.0 - phi ** 2)

    with np.errstate(divide="ignore", invalid="ignore"):
        theta = (stationary_mean / np.sqrt(stationary_var)) * np.sqrt(SHARPE_PERIODS_YEAR)

    return {
        "phi":             phi,
        "intercept":       intercept,
        "residuals":       residuals,
        "stationary_mean": stationary_mean,
        "theta":           np.where(stationary_var > 0, theta, np.nan),
    }


def _stationary_bootstrap_row_indices(n_source: int, n_draw: int, mean_block: float,
                                      rng: np.random.Generator) -> np.ndarray:
    """Circular stationary bootstrap (Politis & Romano 1994) over residual rows."""
    restart_prob = 1.0 / mean_block
    indices      = np.empty(n_draw, dtype=np.int64)
    restart      = rng.random(n_draw) < restart_prob
    restart[0]   = True
    fresh_starts = rng.integers(0, n_source, size=n_draw)

    current = 0
    for step in range(n_draw):
        current = fresh_starts[step] if restart[step] else (current + 1) % n_source
        indices[step] = current
    return indices


def simulate_from_model(model: dict, n_obs: int, rng: np.random.Generator) -> np.ndarray:
    """One synthetic dataset of length n_obs drawn from P_tilde."""
    phi, intercept, residuals = model["phi"], model["intercept"], model["residuals"]
    n_cols  = phi.size
    n_steps = n_obs + BURN_IN

    row_idx    = _stationary_bootstrap_row_indices(residuals.shape[0], n_steps, RESID_BLOCK_MEAN, rng)
    innovation = residuals[row_idx]

    path  = np.empty((n_steps, n_cols), dtype=np.float64)
    state = model["stationary_mean"].copy()
    for step in range(n_steps):
        state = intercept + phi * state + innovation[step]
        path[step] = state

    return path[BURN_IN:]


# =============================================================================
# STUDENTIZED MOVING-BLOCK BOOTSTRAP -- PREFIX-SUM FORMULATION
# =============================================================================
def studentized_bootstrap_quantile(values: np.ndarray, block_size: int, n_bootstrap: int,
                                   alpha: float, rng: np.random.Generator) -> dict:
    """
    First-step studentized StepM machinery of Algorithm 4.2: bootstrap the
    centered Sharpe, studentize by its own bootstrap standard error, and return
    the 1 - alpha quantile of the column-wise maximum.
    """
    matrix = values.astype(np.float64, copy=False)
    n_obs, n_cols = matrix.shape

    prefix_sum = np.zeros((n_obs + 1, n_cols), dtype=np.float64)
    np.cumsum(matrix, axis=0, out=prefix_sum[1:])
    prefix_sq = np.zeros((n_obs + 1, n_cols), dtype=np.float64)
    np.cumsum(matrix * matrix, axis=0, out=prefix_sq[1:])

    real_sharpe = _annualized_sharpe(matrix)

    n_blocks     = int(np.ceil(n_obs / block_size))
    n_starts     = n_obs - block_size + 1
    length_last  = n_obs - (n_blocks - 1) * block_size
    starts       = rng.integers(0, n_starts, size=(n_bootstrap, n_blocks), dtype=np.int32)
    starts_full  = starts[:, :-1] if n_blocks > 1 else starts[:, :0]
    starts_last  = starts[:, -1]

    deviations = np.empty((n_bootstrap, n_cols), dtype=np.float64)

    for chunk_start in range(0, n_bootstrap, REPLICA_CHUNK):
        chunk_end  = min(chunk_start + REPLICA_CHUNK, n_bootstrap)
        chunk_size = chunk_end - chunk_start

        full_chunk = starts_full[chunk_start:chunk_end]
        last_chunk = starts_last[chunk_start:chunk_end]

        if full_chunk.shape[1] > 0:
            sum_full   = (prefix_sum[full_chunk + block_size] - prefix_sum[full_chunk]).sum(axis=1)
            sumsq_full = (prefix_sq[full_chunk + block_size]  - prefix_sq[full_chunk]).sum(axis=1)
        else:
            sum_full   = np.zeros((chunk_size, n_cols), dtype=np.float64)
            sumsq_full = np.zeros((chunk_size, n_cols), dtype=np.float64)

        total_sum   = sum_full   + (prefix_sum[last_chunk + length_last] - prefix_sum[last_chunk])
        total_sumsq = sumsq_full + (prefix_sq[last_chunk + length_last]  - prefix_sq[last_chunk])

        boot_mean = total_sum / n_obs
        boot_var  = (total_sumsq - n_obs * boot_mean * boot_mean) / (n_obs - 1)
        np.maximum(boot_var, 0.0, out=boot_var)
        boot_std = np.sqrt(boot_var)

        with np.errstate(divide="ignore", invalid="ignore"):
            boot_sharpe = (boot_mean / boot_std) * np.sqrt(SHARPE_PERIODS_YEAR)
        boot_sharpe = np.where(boot_std > 0, boot_sharpe, -np.inf)

        deviations[chunk_start:chunk_end] = boot_sharpe - real_sharpe[None, :]

    sigma_hat = deviations.std(axis=0, ddof=1)
    usable    = np.isfinite(sigma_hat) & (sigma_hat > 0) & np.isfinite(real_sharpe)
    if usable.sum() < 2:
        return {"critical_value": np.nan, "real_sharpe": real_sharpe, "sigma_hat": sigma_hat, "usable": usable}

    studentized = deviations[:, usable] / sigma_hat[None, usable]
    replica_max = np.max(np.where(np.isfinite(studentized), studentized, -np.inf), axis=1)
    replica_max = replica_max[np.isfinite(replica_max)]
    if replica_max.size == 0:
        return {"critical_value": np.nan, "real_sharpe": real_sharpe, "sigma_hat": sigma_hat, "usable": usable}

    return {
        "critical_value": float(np.quantile(replica_max, 1.0 - alpha)),
        "real_sharpe":    real_sharpe,
        "sigma_hat":      sigma_hat,
        "usable":         usable,
    }


# =============================================================================
# ROMANO & WOLF (2005) ALGORITHM 7.1 -- COVERAGE CALIBRATION
# =============================================================================
def _coverage_for_dataset(model: dict, n_obs: int, block_sizes: list, alpha: float, seed: int) -> np.ndarray:
    """
    One synthetic dataset, evaluated against every candidate block size.
    Returns a boolean vector: was theta(P_tilde) inside the JCR?
    """
    rng       = np.random.default_rng(seed)
    synthetic = simulate_from_model(model, n_obs, rng)
    theta     = model["theta"]
    covered   = np.zeros(len(block_sizes), dtype=bool)

    for idx, block_size in enumerate(block_sizes):
        result = studentized_bootstrap_quantile(synthetic, block_size, CALIB_N_BOOTSTRAP, alpha, rng)
        critical_value = result["critical_value"]
        if not np.isfinite(critical_value):
            continue

        usable = result["usable"] & np.isfinite(theta)
        if usable.sum() < 2:
            continue

        studentized_gap = (result["real_sharpe"][usable] - theta[usable]) / result["sigma_hat"][usable]
        covered[idx] = bool(np.max(studentized_gap) <= critical_value)

    return covered


def calibrate_block_size(matrix: np.ndarray, block_sizes: list, alpha: float,
                         n_datasets: int, seed: int, n_jobs: int, progress_label: str = "") -> dict:
    """
    Algorithm 7.1: estimate g(b) = P{theta in JCR(b)} by simulation from
    P_tilde, then pick b minimizing |g(b) - (1 - alpha)|.
    """
    model = fit_diagonal_var1(matrix)
    n_obs = matrix.shape[0]

    desc = f"CALIBRATION {progress_label}".strip()
    with tqdm_joblib(tqdm(desc=desc, total=n_datasets, dynamic_ncols=True)):
        coverage_rows = Parallel(n_jobs=n_jobs)(
            delayed(_coverage_for_dataset)(model, n_obs, block_sizes, alpha, seed + dataset_idx)
            for dataset_idx in range(n_datasets)
        )

    coverage = np.vstack(coverage_rows).mean(axis=0)
    target   = 1.0 - alpha
    best_idx = int(np.argmin(np.abs(coverage - target)))

    return {
        "block_sizes":     np.asarray(block_sizes),
        "coverage":        coverage,
        "target":          target,
        "selected":        int(block_sizes[best_idx]),
        "phi_median":      float(np.median(model["phi"])),
        "phi_p90":         float(np.percentile(model["phi"], 90)),
        "n_theta_finite":  int(np.isfinite(model["theta"]).sum()),
    }


# =============================================================================
# COLUMN SELECTION
# =============================================================================
def draw_column_subsample(matrix: np.ndarray, n_columns: int, rng: np.random.Generator) -> np.ndarray:
    """
    Random draw restricted to columns with enough active days and nonzero
    variance. Deliberately NOT the top-Sharpe columns: those are a selected
    subset whose dependence structure is exactly what StepM must not assume.
    """
    active     = (matrix != 0).sum(axis=0) >= MIN_NONZERO_DAYS
    non_degen  = matrix.std(axis=0, ddof=1) > 0
    candidates = np.flatnonzero(active & non_degen)

    if candidates.size == 0:
        raise ValueError("No usable columns: every column is degenerate or too sparse.")

    take = min(n_columns, candidates.size)
    return rng.choice(candidates, size=take, replace=False)


# =============================================================================
# REPORTING
# =============================================================================
def _report_acf(profile: dict, timeframe: str) -> None:
    logger.info(f"\n{'─' * 70}")
    logger.info(f"  EMPIRICAL AUTOCORRELATION OF DAILY P&L ── {timeframe}")
    logger.info(f"{'─' * 70}")
    if profile["lags"].size == 0:
        logger.info("  (no usable columns)")
        return

    band = profile["significance_band"]
    logger.info(f"  white-noise band ±{band:.4f}   (median |acf| across columns)")
    logger.info(f"{'LAG':<8}{'MEDIAN|ACF|':<16}{'P90|ACF|':<16}{'SIGNIFICANT':<12}")
    logger.info(f"{'─' * 70}")
    for lag, med, p90 in zip(profile["lags"], profile["median_abs_acf"], profile["p90_abs_acf"]):
        mark = "✅" if med > band else "❌"
        logger.info(f"{lag:<8}{med:<16.4f}{p90:<16.4f}{mark:<12}")

    beyond_band = profile["lags"][profile["median_abs_acf"] > band]
    last_lag    = int(beyond_band.max()) if beyond_band.size else 0
    logger.info(f"{'─' * 70}")
    logger.info(f"  last lag with median |acf| above band : {last_lag}")


def _report_politis_white(stats: dict, timeframe: str) -> None:
    logger.info(f"\n{'─' * 70}")
    logger.info(f"  POLITIS & WHITE (2004) AUTOMATIC BLOCK LENGTH ── {timeframe}")
    logger.info(f"{'─' * 70}")
    if stats["n_valid"] == 0:
        logger.info("  (no usable columns)")
        return
    logger.info(f"  columns with a valid estimate : {stats['n_valid']}")
    logger.info(f"  median                        : {stats['median']:.2f}")
    logger.info(f"  mean                          : {stats['mean']:.2f}")
    logger.info(f"  IQR                           : [{stats['p25']:.2f}, {stats['p75']:.2f}]")
    logger.info(f"  max                           : {stats['max']:.2f}")


def _report_calibration(results: list, timeframe: str, alpha: float) -> int:
    logger.info(f"\n{'─' * 70}")
    logger.info(f"  ROMANO & WOLF ALGORITHM 7.1 CALIBRATION ── {timeframe}")
    logger.info(f"{'─' * 70}")
    logger.info(f"  target joint coverage : {1.0 - alpha:.4f}   (alpha={alpha})")

    block_sizes = results[0]["block_sizes"]
    header      = f"{'BLOCK':<10}" + "".join(f"{'g(b)#' + str(i + 1):<12}" for i in range(len(results)))
    logger.info(f"\n{header}{'MEAN g(b)':<12}{'|GAP|':<10}")
    logger.info(f"{'─' * 70}")

    coverage_stack = np.vstack([r["coverage"] for r in results])
    mean_coverage  = coverage_stack.mean(axis=0)

    for idx, block_size in enumerate(block_sizes):
        per_subsample = "".join(f"{coverage_stack[j, idx]:<12.4f}" for j in range(len(results)))
        gap = abs(mean_coverage[idx] - (1.0 - alpha))
        logger.info(f"{block_size:<10}{per_subsample}{mean_coverage[idx]:<12.4f}{gap:<10.4f}")

    per_subsample_choice = [r["selected"] for r in results]
    median_choice        = int(np.median(per_subsample_choice))
    mean_choice          = int(block_sizes[int(np.argmin(np.abs(mean_coverage - (1.0 - alpha))))])

    logger.info(f"{'─' * 70}")
    logger.info(f"  fitted AR(1) phi ── median={results[0]['phi_median']:.4f}  p90={results[0]['phi_p90']:.4f}")
    logger.info(f"  per-subsample selections : {per_subsample_choice}")
    logger.info(f"  median of selections     : {median_choice}")
    logger.info(f"  argmin of mean coverage  : {mean_choice}")
    return median_choice


def _report_verdict(timeframe: str, politis_white: dict, calibrated: int) -> None:
    logger.info(f"\n{'─' * 70}")
    logger.info(f"  VERDICT ── {timeframe}")
    logger.info(f"{'─' * 70}")
    logger.info(f"  WHITE_BLOCK_SIZE currently in stepm.py : {WHITE_BLOCK_SIZE}")
    logger.info(f"  Politis & White (median)               : {politis_white['median']:.2f}")
    logger.info(f"  Algorithm 7.1 (coverage-calibrated)    : {calibrated}")
    logger.info(f"{'─' * 70}")
    logger.info("  Algorithm 7.1 is the one that targets StepM's FWE guarantee;")
    logger.info("  Politis & White is a sanity check on the order of magnitude.")
    logger.info(f"{'─' * 70}\n")


# =============================================================================
# PER-TIMEFRAME DRIVER
# =============================================================================
def analyze_timeframe(matrix_arr: np.ndarray, timeframe: str, alpha: float, n_jobs: int) -> dict:
    if matrix_arr is None or matrix_arr.size == 0 or matrix_arr.shape[1] < 2:
        logger.warning(f"BLOCKSIZE ── {timeframe} ── insufficient data, skipping")
        return {}

    logger.info(f"\nBLOCKSIZE ── {timeframe} ── matrix {matrix_arr.shape[0]} days x {matrix_arr.shape[1]} columns")

    block_sizes = [b for b in BLOCK_SIZE_CANDIDATES if b < matrix_arr.shape[0] // 3]
    if not block_sizes:
        logger.warning(f"BLOCKSIZE ── {timeframe} ── too few observations for the candidate grid, skipping")
        return {}

    rng = np.random.default_rng(CALIB_SEED)
    calibration_results = []
    politis_white_stats = None
    acf_profile         = None

    for subsample_idx in range(CALIB_N_SUBSAMPLE):
        columns = draw_column_subsample(matrix_arr, CALIB_N_COLUMNS, rng)
        sample  = matrix_arr[:, columns].astype(np.float64)

        if subsample_idx == 0:
            acf_profile         = empirical_acf_profile(sample)
            politis_white_stats = politis_white_panel(sample)
            _report_acf(acf_profile, timeframe)
            _report_politis_white(politis_white_stats, timeframe)

        calibration_results.append(
            calibrate_block_size(
                matrix         = sample,
                block_sizes    = block_sizes,
                alpha          = alpha,
                n_datasets     = CALIB_N_DATASETS,
                seed           = CALIB_SEED + 10_000 * (subsample_idx + 1),
                n_jobs         = n_jobs,
                progress_label = f"{timeframe} subsample {subsample_idx + 1}/{CALIB_N_SUBSAMPLE}",
            )
        )

    calibrated = _report_calibration(calibration_results, timeframe, alpha)
    _report_verdict(timeframe, politis_white_stats, calibrated)

    return {
        "politis_white":       politis_white_stats,
        "calibration":         calibration_results,
        "calibrated_selected": calibrated,
        "acf":                 acf_profile,
    }


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    start = time.time()

    logger.info(f"\n{'─' * 115}")
    logger.info(f"  WHITE_BLOCK_SIZE SELECTION — POLITIS-WHITE (2004) vs ROMANO-WOLF ALGORITHM 7.1")
    logger.info(f"{'─' * 115}")
    logger.info(f"  TIMEFRAMES        : {TIMEFRAMES}")
    logger.info(f"  N_SYMBOLS         : {N_SYMBOLS}")
    logger.info(f"  PARAM_GRID        : {PARAM_GRID}")
    logger.info(f"  CURRENT BLOCK     : {WHITE_BLOCK_SIZE}")
    logger.info(f"  CANDIDATES        : {BLOCK_SIZE_CANDIDATES}")
    logger.info(f"  STEPM_ALPHA       : {STEPM_ALPHA}")
    logger.info(f"  CALIB_N_COLUMNS   : {CALIB_N_COLUMNS}")
    logger.info(f"  CALIB_N_SUBSAMPLE : {CALIB_N_SUBSAMPLE}")
    logger.info(f"  CALIB_N_DATASETS  : {CALIB_N_DATASETS}")
    logger.info(f"  CALIB_N_BOOTSTRAP : {CALIB_N_BOOTSTRAP}")
    logger.info(f"{'─' * 115}\n")

    # -------------------------------------------------------------------
    # DATA LOADING
    # -------------------------------------------------------------------
    ohlcv_data_by_timeframe = {}
    ohlcv_arr_by_timeframe  = {}

    for timeframe in TIMEFRAMES:
        ohlcv_is = select_universe(
            data_folder_is    = DATA_FOLDER_IS,
            timeframe         = timeframe,
            min_price         = MIN_PRICE,
            filter_symbols_fn = filter_symbols,
        )
        ohlcv_data_by_timeframe[timeframe] = ohlcv_is
        ohlcv_arr_by_timeframe[timeframe]  = prepare_ohlcv_arrays(ohlcv_is)

    # -------------------------------------------------------------------
    # BACKTEST + BLOCK SIZE ANALYSIS -- one full run per timeframe
    # -------------------------------------------------------------------
    results_by_timeframe = {}

    for timeframe in TIMEFRAMES:
        rules_for_timeframe = _build_rule_dicts(
            ohlcv_data_by_timeframe[timeframe], timeframe, RULE_MAX_DEPTH,
        )

        original_n_jobs = backtest_module.BACKTEST_N_JOBS
        backtest_module.BACKTEST_N_JOBS = N_JOBS
        try:
            _, _, matrix_arr, _ = backtest_module.pipe_backtesting(
                rules        = rules_for_timeframe,
                ohlcv_arr    = ohlcv_arr_by_timeframe[timeframe],
                param_grid   = PARAM_GRID,
                order_amount = ORDER_AMOUNT,
                dtype        = DTYPE,
                timeframe    = timeframe,
            )
        finally:
            backtest_module.BACKTEST_N_JOBS = original_n_jobs

        results_by_timeframe[timeframe] = analyze_timeframe(
            matrix_arr = matrix_arr,
            timeframe  = timeframe,
            alpha      = STEPM_ALPHA,
            n_jobs     = N_JOBS,
        )

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")