#BOT_batch/main_NULL.py
"""
StepM null calibration — two-level Monte Carlo.

The null population is the real PnL matrix with every column demeaned, so that
theta_s = 0 exactly for all trials while the temporal and cross-sectional
dependence structure of the real data is left untouched.

Each run draws a pseudo-sample from that population with a circular block
bootstrap (outer level) and hands it to StepM, which builds its own critical
values with a moving block bootstrap (inner level). Under the null every
rejection is a false rejection, so the fraction of runs reaching k rejections
estimates the k-FWER directly.

Note: this design calibrates StepM only. DSR is not evaluated here because its
per-rule inputs in raw_results belong to the original sample, not to the
resampled pseudo-samples.
"""
import os
import sys
import time
import logging
import numpy as np
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_batch")))

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================
LOG_LEVEL = logging.INFO
logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout, force=True)
logger = logging.getLogger("BOT_batch.main_null")
logger.setLevel(LOG_LEVEL)

logging.getLogger("BOT_batch.pipeline.stepm").setLevel(logging.WARNING)
logging.getLogger("joblib").setLevel(logging.WARNING)
logging.getLogger("matplotlib").setLevel(logging.WARNING)

from shared_batchs.symbols.universe import filter_symbols, select_universe
from shared_batchs.setup.config_paths import DATA_FOLDER_IS
from shared_batchs.rule_mining.rule_generator import MAX_DEPTH as RULE_MAX_DEPTH
from shared_batchs.rule_mining.rule_runner import _build_rule_dicts
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_batchs.setup.config_backtest import MIN_PRICE, ORDER_AMOUNT
from shared_batchs.pipeline import backtest_runner as backtest_module
from shared_batchs.pipeline import stepm as stepm_module
from shared_batchs.pipeline.stepm import pipe_stepm, WHITE_BLOCK_SIZE, RANDOM_SEED

# =============================================================================
# UNIVERSE / SEARCH SPACE CONFIGURATION — mirrors main_COMP.py
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
# NULL CALIBRATION CONFIG
# =============================================================================
N_NULL_RUNS = 30  # a 5% violation rate cannot be estimated from a handful of runs
NULL_SEED   = 12345

# Outer draw — generates the pseudo-sample StepM sees. Matching the inner block
# size keeps the exercise focused on the critical values rather than on a
# block-length mismatch between the two levels.
NULL_OUTER_BLOCK_SIZE = WHITE_BLOCK_SIZE

# Optional column cap to bound peak RAM. None = use the full universe. Capping
# changes the multiplicity the test faces, so the resulting rate applies to the
# reduced universe only.
NULL_MAX_COLUMNS = None

STEPM_K_MODE_NULL       = "percentile"  # "absolute" (proven FWE guarantee) or "percentile" (production config, unproven for multi-step)
STEPM_K_FWE_NULL        = 1           # used only when STEPM_K_MODE_NULL == "absolute"
STEPM_K_PERCENTILE_NULL = 0.0001       # used only when STEPM_K_MODE_NULL == "percentile" — mirrors production STEPM_K_PERCENTILE

# =============================================================================
# CALIBRATION CRITERION — k-FWER is the probability of making k or more false
# rejections (Romano & Wolf 2007, eq. 1), NOT a per-comparison error rate.
# =============================================================================
STEPM_ALPHA_NULL = 0.05

# =============================================================================
# NULL POPULATION — demean every column so theta_s = 0 by construction while
# leaving the dependence structure of the real data intact.
# =============================================================================
def _build_null_population(matrix_arr: np.ndarray, col_names: list, chunk_size: int = 5000) -> tuple:
    if NULL_MAX_COLUMNS is not None and matrix_arr.shape[1] > NULL_MAX_COLUMNS:
        rng = np.random.default_rng(NULL_SEED)
        keep = np.sort(rng.choice(matrix_arr.shape[1], size=NULL_MAX_COLUMNS, replace=False))
        matrix_arr = np.ascontiguousarray(matrix_arr[:, keep])
        col_names = [col_names[i] for i in keep]

    n_cols = matrix_arr.shape[1]
    for start in range(0, n_cols, chunk_size):
        end = min(start + chunk_size, n_cols)
        chunk = matrix_arr[:, start:end]
        chunk -= chunk.mean(axis=0, dtype=np.float64).astype(matrix_arr.dtype)[None, :]

    return matrix_arr, col_names

# =============================================================================
# OUTER DRAW — circular block bootstrap over rows. Circular rather than moving
# to avoid the edge effects discussed in Romano & Wolf (2005), Appendix B.
# =============================================================================
def _circular_block_resample(population: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    n_obs = population.shape[0]
    n_blocks = int(np.ceil(n_obs / block_size))

    starts = rng.integers(0, n_obs, size=n_blocks, dtype=np.int64)
    row_idx = (starts[:, None] + np.arange(block_size, dtype=np.int64)[None, :]).ravel()[:n_obs] % n_obs

    return np.ascontiguousarray(population[row_idx])

# =============================================================================
# ONE NULL REALIZATION — outer draw, then StepM with its own inner bootstrap
# =============================================================================
def _run_one_null_iteration(
    population: np.ndarray,
    col_names: list,
    raw_results: list,
    timeframe: str,
    run_idx: int,
) -> dict:

    rng = np.random.default_rng(NULL_SEED + run_idx)
    sample_arr = _circular_block_resample(population, NULL_OUTER_BLOCK_SIZE, rng)

    original_k_mode = stepm_module.STEPM_K_MODE
    original_k_fwe  = stepm_module.STEPM_K_FWE
    stepm_module.STEPM_K_MODE = STEPM_K_MODE_NULL
    if STEPM_K_MODE_NULL == "absolute":
        stepm_module.STEPM_K_FWE = STEPM_K_FWE_NULL
    try:
        stepm_results = pipe_stepm(
            raw_results        = raw_results,
            matrix_arr         = sample_arr,
            col_names          = col_names,
            stepm_alpha        = STEPM_ALPHA_NULL,
            stepm_k_percentile = STEPM_K_PERCENTILE_NULL if STEPM_K_MODE_NULL == "percentile" else None,
            seed               = RANDOM_SEED + run_idx,
            timeframe          = timeframe,
        )
    finally:
        stepm_module.STEPM_K_MODE = original_k_mode
        stepm_module.STEPM_K_FWE  = original_k_fwe

    if STEPM_K_MODE_NULL == "absolute":
        k_effective = STEPM_K_FWE_NULL
    else:
        k_effective = max(1, int(np.ceil(STEPM_K_PERCENTILE_NULL * sample_arr.shape[1])))

    n_pass = sum(1 for r in stepm_results if r["passed_stepm"])

    return {
        "n_total":      len(raw_results),
        "n_cols":       sample_arr.shape[1],
        "k_effective":  k_effective,
        "n_stepm_pass": n_pass,
    }

# =============================================================================
# REPORTING
# =============================================================================
def _wilson_interval(n_success: int, n_trials: int, z: float = 1.96) -> tuple:
    """Wilson score interval — well behaved near 0, unlike the normal approximation."""
    if n_trials == 0:
        return 0.0, 1.0
    p_hat  = n_success / n_trials
    denom  = 1.0 + z ** 2 / n_trials
    center = (p_hat + z ** 2 / (2 * n_trials)) / denom
    margin = z * np.sqrt(p_hat * (1 - p_hat) / n_trials + z ** 2 / (4 * n_trials ** 2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def _print_null_summary(timeframe: str, iterations: list) -> None:
    n_runs  = len(iterations)
    n_total = iterations[0]["n_total"]
    n_cols  = iterations[0]["n_cols"]

    stepm_pass = np.asarray([it["n_stepm_pass"] for it in iterations])
    k_eff_arr  = np.asarray([it["k_effective"]  for it in iterations])

    violations = int((stepm_pass >= k_eff_arr).sum())
    rate       = violations / n_runs
    rate_lo, rate_hi = _wilson_interval(violations, n_runs)

    logger.info(f"\n{'─' * 70}")
    logger.info(f"  STEPM NULL CALIBRATION ── {timeframe} ── n_runs={n_runs}")
    logger.info(f"{'─' * 70}")
    logger.info(f"  rules / columns     : {n_total} / {n_cols}")
    logger.info(f"  null construction   : per-column demean + circular block resample (b={NULL_OUTER_BLOCK_SIZE})")
    logger.info(f"  k judged against    : {k_eff_arr.min()}–{k_eff_arr.max()} ({STEPM_K_MODE_NULL})")
    logger.info(f"{'─' * 70}")
    logger.info(f"  runs with >= k rejections : {violations}/{n_runs}")
    logger.info(f"    empirical k-FWER  : {rate:.4f}  (95% CI [{rate_lo:.4f}, {rate_hi:.4f}])")
    logger.info(f"    nominal alpha     : {STEPM_ALPHA_NULL:.4f}")
    if rate_lo > STEPM_ALPHA_NULL:
        verdict = "FAIL — rate significantly above nominal alpha"
    elif rate_hi < STEPM_ALPHA_NULL:
        verdict = "CONSERVATIVE — rate significantly below nominal alpha (valid, but power is being left on the table)"
    else:
        verdict = "PASS — rate consistent with nominal alpha"
    logger.info(f"    verdict           : {verdict}")
    logger.info(f"{'─' * 70}")
    logger.info(f"  rejections per run ── min/median/mean/max: "
                f"{stepm_pass.min()} / {np.median(stepm_pass):.1f} / {stepm_pass.mean():.2f} / {stepm_pass.max()}")
    logger.info(f"  runs with 0 rejections: {int((stepm_pass == 0).sum())}/{n_runs}")
    logger.info(f"{'─' * 70}\n")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    start = time.time()

    logger.info(f"\n{'─' * 115}")
    logger.info(f"  STEPM NULL CALIBRATION — demeaned population, resampled pseudo-samples")
    logger.info(f"{'─' * 115}")
    logger.info(f"  TIMEFRAMES            : {TIMEFRAMES}")
    logger.info(f"  N_SYMBOLS             : {N_SYMBOLS}")
    logger.info(f"  PARAM_GRID            : {PARAM_GRID}")
    logger.info(f"  N_NULL_RUNS           : {N_NULL_RUNS}")
    logger.info(f"  NULL_OUTER_BLOCK_SIZE : {NULL_OUTER_BLOCK_SIZE}")
    logger.info(f"  NULL_MAX_COLUMNS      : {NULL_MAX_COLUMNS}")
    logger.info(f"  STEPM_ALPHA_NULL      : {STEPM_ALPHA_NULL}")
    if STEPM_K_MODE_NULL == "absolute":
        logger.info(f"  STEPM_K_MODE_NULL     : {STEPM_K_MODE_NULL} (k={STEPM_K_FWE_NULL})")
    else:
        logger.info(f"  STEPM_K_MODE_NULL     : {STEPM_K_MODE_NULL} (k_percentile={STEPM_K_PERCENTILE_NULL})")
    logger.info(f"{'─' * 115}\n")

    for timeframe in TIMEFRAMES:
        # -------------------------------------------------------------------
        # SINGLE BACKTEST — the real PnL matrix is the basis of the null
        # population, so the expensive stages run once per timeframe.
        # -------------------------------------------------------------------
        ohlcv_is = select_universe(
            data_folder_is    = DATA_FOLDER_IS,
            timeframe         = timeframe,
            min_price         = MIN_PRICE,
            filter_symbols_fn = filter_symbols,
        )
        ohlcv_arr = prepare_ohlcv_arrays(ohlcv_is)
        rules = _build_rule_dicts(ohlcv_is, timeframe, RULE_MAX_DEPTH)

        original_n_jobs = backtest_module.BACKTEST_N_JOBS
        backtest_module.BACKTEST_N_JOBS = N_JOBS
        try:
            raw_results, n_combos, matrix_arr, col_names = backtest_module.pipe_backtesting(
                rules        = rules,
                ohlcv_arr    = ohlcv_arr,
                param_grid   = PARAM_GRID,
                order_amount = ORDER_AMOUNT,
                dtype        = DTYPE,
                timeframe    = timeframe,
            )
        finally:
            backtest_module.BACKTEST_N_JOBS = original_n_jobs

        population, col_names = _build_null_population(matrix_arr, col_names)
        logger.info(
            f"NULL ── {timeframe} ── population built ── {population.shape[0]} obs x "
            f"{population.shape[1]} columns ── max |column mean| = "
            f"{np.abs(population.mean(axis=0, dtype=np.float64)).max():.3e}"
        )

        # -------------------------------------------------------------------
        # OUTER MONTE CARLO — one pseudo-sample per run, StepM on each.
        # -------------------------------------------------------------------
        iterations = []
        for run_idx in range(N_NULL_RUNS):
            iterations.append(
                _run_one_null_iteration(population, col_names, raw_results, timeframe, run_idx)
            )
            last = iterations[-1]
            logger.info(
                f"NULL ── {timeframe} ── run {run_idx + 1}/{N_NULL_RUNS} ── "
                f"rejections={last['n_stepm_pass']} (k={last['k_effective']})"
            )

        _print_null_summary(timeframe, iterations)

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")