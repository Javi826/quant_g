#BOT_batch/main_POWER.py
"""
StepM power measurement.

main_NULL.py established that StepM does not approve noise. That leaves the
complementary question open: does it approve genuine signal? A test that never
rejects anything passes a null calibration perfectly and is useless.

The population is the real PnL matrix with every column demeaned, so theta_s = 0
everywhere by construction. A known drift is then injected into a subset of
columns, sized so that each injected column carries a target annualized Sharpe.
Power is the fraction of injected rules StepM recovers; rejections among the
untouched columns are false positives and should stay near the null rate.

Signal is injected only into columns that pipe_stepm actually evaluates, i.e.
those named "{rule_id}__{best_combo_id}". Injecting elsewhere would be invisible
to the per-rule pass/fail contract and would understate power.
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
logger = logging.getLogger("BOT_batch.main_power")
logger.setLevel(LOG_LEVEL)

logging.getLogger("BOT_batch.pipeline.stepM").setLevel(logging.WARNING)
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
from shared_batchs.pipeline.stepM import pipe_stepm, WHITE_BLOCK_SIZE, RANDOM_SEED, SHARPE_PERIODS_YEAR

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
# POWER SWEEP CONFIG
# =============================================================================
# Full sweep: three Sharpe targets, three multiplicity regimes, full universe.
# ~90 pipe_stepm calls on the full ~295k-column universe — several hours.
#
# --- fast sanity-check config, restore if you need a quick re-check ---------
# SHARPE_TARGETS    = [3.0]
# N_SIGNAL_RULES    = [10, 200]
# N_RUNS_PER_CELL   = 5
# POWER_MAX_COLUMNS = 30000
# ------------------------------------------------------------------------------

# Target annualized Sharpe of each injected column. Spans the range that would
# be worth trading: 1.0 is marginal, 3.0 is a strong strategy.
SHARPE_TARGETS = [1.0, 2.0, 3.0]

# How many rules carry signal. Power depends strongly on this: a lone winner
# must beat the maximum of the whole null universe, while many winners raise
# each other's chances of clearing the stepdown.
N_SIGNAL_RULES = [10, 100, 1000]

N_RUNS_PER_CELL = 10
POWER_SEED      = 4242

# Outer draw block size. Matched to the inner bootstrap, which the robustness
# grid showed to be the reliable choice.
OUTER_BLOCK_SIZE = WHITE_BLOCK_SIZE

# None = full universe, which keeps the multiplicity StepM faces in production.
# Capping speeds the sweep up but makes the reported power optimistic.
POWER_MAX_COLUMNS = None

# k = 1 is the regime with a proven guarantee and a validated null rate, so the
# power figure it yields is the one that can be trusted.
STEPM_ALPHA_POWER = 0.05
STEPM_K_FWE_POWER = 1

# =============================================================================
# NULL POPULATION — demean every column so theta_s = 0 by construction while
# leaving the dependence structure of the real data intact.
# =============================================================================
def _build_null_population(matrix_arr: np.ndarray, col_names: list, chunk_size: int = 5000) -> tuple:
    if POWER_MAX_COLUMNS is not None and matrix_arr.shape[1] > POWER_MAX_COLUMNS:
        rng = np.random.default_rng(POWER_SEED)
        keep = np.sort(rng.choice(matrix_arr.shape[1], size=POWER_MAX_COLUMNS, replace=False))
        matrix_arr = np.ascontiguousarray(matrix_arr[:, keep])
        col_names = [col_names[i] for i in keep]

    n_cols = matrix_arr.shape[1]
    for start in range(0, n_cols, chunk_size):
        end = min(start + chunk_size, n_cols)
        chunk = matrix_arr[:, start:end]
        chunk -= chunk.mean(axis=0, dtype=np.float64).astype(matrix_arr.dtype)[None, :]

    return matrix_arr, col_names

# =============================================================================
# EVALUABLE COLUMNS — pipe_stepm only ever looks up "{rule_id}__{best_combo_id}",
# so signal outside that set could never be reported as a pass.
# =============================================================================
def _evaluable_columns(raw_results: list, col_names: list) -> tuple:
    col_index = {name: idx for idx, name in enumerate(col_names)}

    rule_ids, col_idx = [], []
    for r in raw_results:
        best_combo_id = r.get("best_combo_id")
        if not best_combo_id:
            continue
        name = f"{r['rule_id']}__{best_combo_id}"
        idx = col_index.get(name)
        if idx is not None:
            rule_ids.append(r["rule_id"])
            col_idx.append(idx)

    return np.asarray(rule_ids, dtype=object), np.asarray(col_idx, dtype=np.int64)

# =============================================================================
# SIGNAL SIZING — the drift that turns a demeaned column into one with a given
# annualized Sharpe: mu = target * sigma / sqrt(periods_per_year).
# =============================================================================
def _drift_for_target_sharpe(population: np.ndarray, col_idx: np.ndarray, target: float) -> np.ndarray:
    stds = population[:, col_idx].std(axis=0, ddof=1, dtype=np.float64)
    return (target * stds / np.sqrt(SHARPE_PERIODS_YEAR)).astype(np.float64)

# =============================================================================
# OUTER DRAW — circular block bootstrap over rows.
# =============================================================================
def _circular_block_resample(population: np.ndarray, block_size: int, rng: np.random.Generator) -> np.ndarray:
    n_obs = population.shape[0]
    n_blocks = int(np.ceil(n_obs / block_size))

    starts = rng.integers(0, n_obs, size=n_blocks, dtype=np.int64)
    row_idx = (starts[:, None] + np.arange(block_size, dtype=np.int64)[None, :]).ravel()[:n_obs] % n_obs

    return np.ascontiguousarray(population[row_idx])

# =============================================================================
# ONE SWEEP CELL — N runs at a fixed (n_signal, sharpe_target) pair
# =============================================================================
def _run_cell(
    population: np.ndarray,
    col_names: list,
    raw_results: list,
    eval_rule_ids: np.ndarray,
    eval_col_idx: np.ndarray,
    n_signal: int,
    sharpe_target: float,
    timeframe: str,
) -> dict:

    original_k_mode = stepm_module.STEPM_K_MODE
    original_k_fwe  = stepm_module.STEPM_K_FWE
    stepm_module.STEPM_K_MODE = "absolute"
    stepm_module.STEPM_K_FWE  = STEPM_K_FWE_POWER

    n_true_pos   = np.empty(N_RUNS_PER_CELL, dtype=np.int64)
    n_false_pos  = np.empty(N_RUNS_PER_CELL, dtype=np.int64)
    realized_shp = np.empty(N_RUNS_PER_CELL, dtype=np.float64)

    try:
        for run_idx in range(N_RUNS_PER_CELL):
            rng = np.random.default_rng(POWER_SEED + 100 * run_idx + int(sharpe_target * 10))

            pick          = rng.choice(eval_col_idx.shape[0], size=n_signal, replace=False)
            signal_cols   = eval_col_idx[pick]
            signal_rules  = set(eval_rule_ids[pick])

            # Adding a constant commutes with row resampling, so the drift is
            # applied to the drawn sample rather than to a copy of the population.
            sample_arr = _circular_block_resample(population, OUTER_BLOCK_SIZE, rng)
            drift = _drift_for_target_sharpe(population, signal_cols, sharpe_target)
            sample_arr[:, signal_cols] += drift.astype(sample_arr.dtype)[None, :]

            means = sample_arr[:, signal_cols].mean(axis=0, dtype=np.float64)
            stds  = sample_arr[:, signal_cols].std(axis=0, ddof=1, dtype=np.float64)
            with np.errstate(divide="ignore", invalid="ignore"):
                realized_shp[run_idx] = float(np.nanmedian(means / stds * np.sqrt(SHARPE_PERIODS_YEAR)))

            stepm_results = pipe_stepm(
                raw_results = raw_results,
                matrix_arr  = sample_arr,
                col_names   = col_names,
                stepm_alpha = STEPM_ALPHA_POWER,
                block_size  = WHITE_BLOCK_SIZE,
                seed        = RANDOM_SEED + run_idx,
                timeframe   = timeframe,
            )

            passed_rules = {r["rule_id"] for r in stepm_results if r["passed_stepm"]}
            n_true_pos[run_idx]  = len(passed_rules & signal_rules)
            n_false_pos[run_idx] = len(passed_rules - signal_rules)
    finally:
        stepm_module.STEPM_K_MODE = original_k_mode
        stepm_module.STEPM_K_FWE  = original_k_fwe

    return {
        "n_signal":        n_signal,
        "sharpe_target":   sharpe_target,
        "realized_sharpe": float(np.mean(realized_shp)),
        "power":           float(n_true_pos.mean() / n_signal),
        "mean_true_pos":   float(n_true_pos.mean()),
        "max_true_pos":    int(n_true_pos.max()),
        "mean_false_pos":  float(n_false_pos.mean()),
        "any_detect_rate": float((n_true_pos > 0).mean()),
    }

# =============================================================================
# REPORTING
# =============================================================================
def _print_sweep(timeframe: str, cells: dict, n_cols: int, n_eval: int) -> None:
    cell_width = 16

    logger.info(f"\n{'─' * 100}")
    logger.info(f"  STEPM POWER SWEEP ── {timeframe}")
    logger.info(f"{'─' * 100}")
    logger.info(f"  columns / evaluable : {n_cols} / {n_eval}")
    logger.info(f"  runs per cell       : {N_RUNS_PER_CELL}")
    logger.info(f"  block size          : {WHITE_BLOCK_SIZE} (inner and outer)")
    logger.info(f"  alpha / k           : {STEPM_ALPHA_POWER} / {STEPM_K_FWE_POWER}")
    logger.info(f"{'─' * 100}")

    logger.info(f"  POWER — fraction of injected rules recovered")
    logger.info(f"{'─' * 100}")
    logger.info(f"  {'n_signal':<12}" + "".join(f"{f'SR={t}':<{cell_width}}" for t in SHARPE_TARGETS))
    logger.info(f"{'─' * 100}")
    for n_signal in N_SIGNAL_RULES:
        row = f"  {n_signal:<12}"
        for target in SHARPE_TARGETS:
            cell = cells[(n_signal, target)]
            label = f"{cell['power']:.3f}"
            row += f"{label:<{cell_width}}"
        logger.info(row)

    logger.info(f"{'─' * 100}")
    logger.info(f"  DETECTION RATE — runs recovering at least one injected rule")
    logger.info(f"{'─' * 100}")
    logger.info(f"  {'n_signal':<12}" + "".join(f"{f'SR={t}':<{cell_width}}" for t in SHARPE_TARGETS))
    logger.info(f"{'─' * 100}")
    for n_signal in N_SIGNAL_RULES:
        row = f"  {n_signal:<12}"
        for target in SHARPE_TARGETS:
            cell = cells[(n_signal, target)]
            label = f"{cell['any_detect_rate']:.2f}"
            row += f"{label:<{cell_width}}"
        logger.info(row)

    logger.info(f"{'─' * 100}")
    logger.info(f"  FALSE POSITIVES — mean rejections among untouched rules (should stay near 0)")
    logger.info(f"{'─' * 100}")
    logger.info(f"  {'n_signal':<12}" + "".join(f"{f'SR={t}':<{cell_width}}" for t in SHARPE_TARGETS))
    logger.info(f"{'─' * 100}")
    for n_signal in N_SIGNAL_RULES:
        row = f"  {n_signal:<12}"
        for target in SHARPE_TARGETS:
            cell = cells[(n_signal, target)]
            label = f"{cell['mean_false_pos']:.2f}"
            row += f"{label:<{cell_width}}"
        logger.info(row)

    logger.info(f"{'─' * 100}")
    logger.info(f"  realized median Sharpe of injected columns (sanity check vs target)")
    for target in SHARPE_TARGETS:
        realized = [cells[(n, target)]["realized_sharpe"] for n in N_SIGNAL_RULES]
        logger.info(f"    target SR={target:<5} realized={np.mean(realized):.3f}")
    logger.info(f"{'─' * 100}\n")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    start = time.time()

    n_cells = len(N_SIGNAL_RULES) * len(SHARPE_TARGETS)

    logger.info(f"\n{'─' * 115}")
    logger.info(f"  STEPM POWER — known signal injected into a demeaned population")
    logger.info(f"{'─' * 115}")
    logger.info(f"  TIMEFRAMES         : {TIMEFRAMES}")
    logger.info(f"  PARAM_GRID         : {PARAM_GRID}")
    logger.info(f"  SHARPE_TARGETS     : {SHARPE_TARGETS}")
    logger.info(f"  N_SIGNAL_RULES     : {N_SIGNAL_RULES}")
    logger.info(f"  N_RUNS_PER_CELL    : {N_RUNS_PER_CELL}")
    logger.info(f"  TOTAL STEPM CALLS  : {n_cells * N_RUNS_PER_CELL}")
    logger.info(f"  BLOCK SIZE         : {WHITE_BLOCK_SIZE}")
    logger.info(f"  POWER_MAX_COLUMNS  : {POWER_MAX_COLUMNS}")
    logger.info(f"  ALPHA / K          : {STEPM_ALPHA_POWER} / {STEPM_K_FWE_POWER}")
    logger.info(f"{'─' * 115}\n")

    for timeframe in TIMEFRAMES:
        # -------------------------------------------------------------------
        # SINGLE BACKTEST — the real PnL matrix is the basis of the population.
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
        eval_rule_ids, eval_col_idx = _evaluable_columns(raw_results, col_names)

        logger.info(
            f"POWER ── {timeframe} ── population {population.shape[0]} obs x {population.shape[1]} cols "
            f"── evaluable columns: {eval_col_idx.shape[0]} "
            f"── max |column mean| = {np.abs(population.mean(axis=0, dtype=np.float64)).max():.3e}\n"
        )

        max_signal = max(N_SIGNAL_RULES)
        if eval_col_idx.shape[0] < max_signal:
            raise ValueError(
                f"only {eval_col_idx.shape[0]} evaluable columns available but "
                f"N_SIGNAL_RULES requests up to {max_signal}."
            )

        # -------------------------------------------------------------------
        # SWEEP
        # -------------------------------------------------------------------
        cells = {}
        cell_idx = 0
        for n_signal in N_SIGNAL_RULES:
            for target in SHARPE_TARGETS:
                cell_idx += 1
                cell_start = time.time()
                cells[(n_signal, target)] = _run_cell(
                    population, col_names, raw_results, eval_rule_ids, eval_col_idx,
                    n_signal, target, timeframe,
                )
                cell = cells[(n_signal, target)]
                logger.info(
                    f"POWER ── cell {cell_idx}/{n_cells} ── n_signal={n_signal} SR={target} ── "
                    f"power={cell['power']:.3f} ── mean TP/FP={cell['mean_true_pos']:.1f}/{cell['mean_false_pos']:.1f} ── "
                    f"{int(time.time() - cell_start)}s"
                )

        _print_sweep(timeframe, cells, population.shape[1], eval_col_idx.shape[0])

    elapsed = int(time.time() - start)
    logger.info(f"\n🏁 TOTAL — {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")