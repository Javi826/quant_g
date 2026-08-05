#shared_batchs/pipeline/n_eff_calibration.py
import time
import logging
import itertools
import numpy as np
from scipy.optimize import brentq
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from shared_config import VOLUME_COL
from shared_batchs.rule_mining.rule_runner import _build_rule_dicts
from shared_batchs.rule_mining.rule_generator import MAX_DEPTH
from shared_batchs.backtesters.ZX_compute_BT import prepare_backtest_data, prepare_signal_arrays
from shared_batchs.utils.paralelization import arrays_to_shared_memory, arrays_from_shared_memory
from shared_batchs.pipeline.dsr import (
    _expected_max_sharpe,
    _unannualize_sharpe,
    _build_full_period_ohlcv,
    _evaluate_combo_sharpe,
    _get_static_bundle,
    DSR_N_JOBS,
)
from shared_batchs.pipeline.multiverse import (
    _generate_mcpt_paths_all_symbols,
    _synthetic_ohlcv_arr,
    BLOCK_SIZE,
)

logger = logging.getLogger("BOT_batch.pipeline.n_eff_calibration")

# =============================================================================
# N_EFF CALIBRATION CONFIG
# =============================================================================
N_NULL_PATHS_DEFAULT = 20
BASE_SEED_DEFAULT    = 42
N_IMPLIED_LOWER       = 1.001
N_IMPLIED_UPPER       = 1.0e9


# =============================================================================
# PER-COMBO GRANULAR SEARCH — every combo's Sharpe, not just the per-rule winner
# (mirrors dsr.py's run_full_period_search / _run_full_period_for_rule, but
# keeps the full combo-level distribution instead of collapsing to the best)
# =============================================================================
def _all_combo_sharpes_for_rule(
    ohlcv_arr: dict,
    signal_fn: callable,
    param_grid: dict,
    order_amount: int,
    dtype,
    static_bundle: dict | None = None,
) -> list:

    keys   = list(param_grid.keys())
    combos = [dict(zip(keys, c)) for c in itertools.product(*[param_grid[k] for k in keys])]

    ohlcv_arrays = _build_full_period_ohlcv(ohlcv_arr, signal_fn, dtype)

    if static_bundle is not None:
        prepared_data = prepare_signal_arrays(static_bundle, ohlcv_arrays)
    else:
        prepared_data = prepare_backtest_data(ohlcv_arrays)

    prepared_arrays = prepared_data[7]

    sharpes = []
    for params in combos:
        sharpe_rank, _params, _bundle, _daily = _evaluate_combo_sharpe(params, prepared_arrays, order_amount)
        if np.isfinite(sharpe_rank):
            sharpes.append(sharpe_rank)

    return sharpes


def _all_combo_sharpes_for_rule_shm(
    rule_id: str,
    shm_metadata: dict,
    signal_fn: callable,
    param_grid: dict,
    order_amount: int,
    dtype,
) -> tuple:

    ohlcv_arr, shm_handles = arrays_from_shared_memory(shm_metadata)
    try:
        static_bundle = _get_static_bundle(shm_metadata, ohlcv_arr)
        sharpes = _all_combo_sharpes_for_rule(
            ohlcv_arr, signal_fn, param_grid, order_amount, dtype, static_bundle,
        )
        return rule_id, sharpes
    finally:
        for shm in shm_handles:
            shm.close()


def _all_combo_sharpes_search(
    rules: list,
    param_grid: dict,
    order_amount: int,
    dtype,
    progress_label: str = "",
) -> dict:
    """Same shared-memory / parallel orchestration as dsr.py's run_full_period_search,
    but returns {rule_id: [combo_sharpe, ...]} instead of the per-rule winner."""

    if not rules:
        return {}

    desc = f"N_EFF NULL SEARCH {progress_label}".strip()

    shm_list, shm_metadata = arrays_to_shared_memory(rules[0]["ohlcv_arr"])
    try:
        with tqdm_joblib(tqdm(desc=desc, total=len(rules), dynamic_ncols=True)):
            results = Parallel(n_jobs=DSR_N_JOBS, batch_size=1, pre_dispatch='all')(
                delayed(_all_combo_sharpes_for_rule_shm)(
                    r["rule_id"], shm_metadata, r["signal_fn"], param_grid, order_amount, dtype,
                )
                for r in rules
            )
    finally:
        for shm in shm_list:
            shm.close()
            shm.unlink()

    return dict(results)


# =============================================================================
# NULL-UNIVERSE MAXIMUM SHARPE — one full per-combo search per null path
# =============================================================================
def _max_sharpe_for_null_path(
    rules: list,
    synthetic_arr: dict,
    param_grid: dict,
    order_amount: int,
    dtype,
    progress_label: str,
) -> tuple:
    """Run the per-combo granular search on one synthetic (null) universe.

    Returns (max_sharpe, all_unannualized_combo_sharpes) for that path, or
    (None, []) if no combo produced a finite Sharpe (e.g. insufficient trades
    everywhere).
    """
    rules_for_search = [
        {"rule_id": r["rule_id"], "ohlcv_arr": synthetic_arr, "signal_fn": r["signal_fn"]}
        for r in rules
    ]
    sharpes_by_rule = _all_combo_sharpes_search(
        rules          = rules_for_search,
        param_grid     = param_grid,
        order_amount   = order_amount,
        dtype          = dtype,
        progress_label = progress_label,
    )

    sharpe_values = [
        _unannualize_sharpe(s)
        for sharpes in sharpes_by_rule.values()
        for s in sharpes
    ]
    sharpe_values = [s for s in sharpe_values if np.isfinite(s)]
    if not sharpe_values:
        return None, []

    return float(max(sharpe_values)), sharpe_values


# =============================================================================
# EQ.1 INVERSION — solve for N given an empirical E[max{SR}] and V[{SR}]
# =============================================================================
def _invert_expected_max_sharpe(mean_max_sr_null: float, var_sr_null: float) -> float | None:
    """Find N such that _expected_max_sharpe(var_sr_null, N) == mean_max_sr_null."""
    if var_sr_null <= 0 or not np.isfinite(mean_max_sr_null):
        return None

    def _objective(n_trials: float) -> float:
        return _expected_max_sharpe(var_sr_null, n_trials) - mean_max_sr_null

    lo, hi = N_IMPLIED_LOWER, N_IMPLIED_UPPER
    f_lo, f_hi = _objective(lo), _objective(hi)

    if f_lo > 0:
        # even N->1 overshoots the empirical max — cannot bracket a root
        return None
    if f_hi < 0:
        # even N->1e9 undershoots — the empirical max is implausibly high
        return None

    return float(brentq(_objective, lo, hi, xtol=1e-6, rtol=1e-8, maxiter=200))


# =============================================================================
# PER-TIMEFRAME CALIBRATION
# =============================================================================
def _calibrate_timeframe(
    timeframe: str,
    ohlcv_data: dict,
    param_grid: dict,
    order_amount: int,
    dtype,
    n_null_paths: int,
    block_size: int,
    max_depth: int,
    base_seed: int,
) -> dict | None:

    rules = _build_rule_dicts(ohlcv_data, timeframe, max_depth)
    if not rules:
        logger.warning(f"N_EFF CALIBRATION ── {timeframe} ── no candidate rules — skipping")
        return None

    ref_sym  = max(ohlcv_data.keys(), key=lambda sym: len(ohlcv_data[sym]))
    n_obs    = len(ohlcv_data[ref_sym])
    ts_index = ohlcv_data[ref_sym].index[:n_obs].to_numpy()
    n_symbols_expected = len(ohlcv_data)

    paths = _generate_mcpt_paths_all_symbols(
        ohlcv_data, n_paths=n_null_paths, raw_columns=[VOLUME_COL],
        dtype=dtype, base_seed=base_seed, block_size=block_size,
    )

    max_sr_per_path = []
    pooled_sharpes  = []

    for path_idx in range(n_null_paths):
        synthetic_arr = _synthetic_ohlcv_arr(paths, path_idx, ts_index, dtype)
        if len(synthetic_arr) < n_symbols_expected:
            logger.debug(f"N_EFF CALIBRATION ── {timeframe} ── null path {path_idx + 1} incomplete — skipping")
            continue

        max_sr, sharpe_values = _max_sharpe_for_null_path(
            rules          = rules,
            synthetic_arr  = synthetic_arr,
            param_grid     = param_grid,
            order_amount   = order_amount,
            dtype          = dtype,
            progress_label = f"{timeframe} null {path_idx + 1}/{n_null_paths}",
        )
        if max_sr is None:
            continue

        max_sr_per_path.append(max_sr)
        pooled_sharpes.extend(sharpe_values)

    if not max_sr_per_path:
        logger.warning(f"N_EFF CALIBRATION ── {timeframe} ── no valid null paths — skipping")
        return None

    max_sr_arr        = np.asarray(max_sr_per_path, dtype=np.float64)
    mean_max_sr_null  = float(max_sr_arr.mean())
    std_max_sr_null   = float(max_sr_arr.std(ddof=1)) if max_sr_arr.size > 1 else 0.0
    var_sr_null       = float(np.var(np.asarray(pooled_sharpes, dtype=np.float64), ddof=1))

    n_implied = _invert_expected_max_sharpe(mean_max_sr_null, var_sr_null)

    return {
        "timeframe":         timeframe,
        "n_implied":         n_implied,
        "mean_max_sr_null":  mean_max_sr_null,
        "std_max_sr_null":   std_max_sr_null,
        "var_sr_null":       var_sr_null,
        "n_null_paths_used": len(max_sr_per_path),
        "max_sr_per_path":   max_sr_per_path,
    }


# =============================================================================
# LOGGING
# =============================================================================
def _print_calibration_result(result: dict) -> None:
    timeframe = result["timeframe"]
    n_implied = result["n_implied"]
    n_implied_str = f"{n_implied:,.1f}".replace(",", ".") if n_implied is not None else "n/a (root not bracketed)"

    logger.info(f"\n{'─' * 70}")
    logger.info(f"  N_EFF CALIBRATION ── {timeframe}")
    logger.info(f"{'─' * 70}")
    logger.info(f"  null paths used     : {result['n_null_paths_used']}")
    logger.info(f"  E[max SR] (null)    : {result['mean_max_sr_null']:.4f}  (std={result['std_max_sr_null']:.4f})")
    logger.info(f"  V[{{SR}}] (null)      : {result['var_sr_null']:.6f}")
    logger.info(f"  N_implied (Eq.1)    : {n_implied_str}")
    logger.info(f"{'─' * 70}\n")


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================
def calibrate_n_eff(
    ohlcv_data_by_timeframe: dict,
    timeframes: list,
    param_grid: dict,
    order_amount: int,
    dtype,
    n_null_paths: int = N_NULL_PATHS_DEFAULT,
    block_size: int = BLOCK_SIZE,
    max_depth: int = MAX_DEPTH,
    base_seed: int = BASE_SEED_DEFAULT,
    enabled: bool = True,
) -> dict:
    """Empirically calibrate N (independent trials) per timeframe.

    For each timeframe: generates `n_null_paths` synthetic (null-hypothesis)
    universes via moving-block-bootstrapped MCPT paths, runs the full DSR
    rule x combo search on each, and records max(sharpe_train) per path.
    Inverts Eq.1 (Bailey & Lopez de Prado, 2014) to back out the N that
    would make the analytical E[max{SR}] match the empirical one.

    Returns {timeframe: {...}} — see _calibrate_timeframe for fields.
    Empty dict if disabled or if no timeframe yields a valid calibration.
    """
    if not enabled:
        logger.info("N_EFF CALIBRATION ── disabled — skipping")
        return {}

    start = time.time()
    results_by_timeframe = {}

    for timeframe in timeframes:
        result = _calibrate_timeframe(
            timeframe      = timeframe,
            ohlcv_data     = ohlcv_data_by_timeframe[timeframe],
            param_grid     = param_grid,
            order_amount   = order_amount,
            dtype          = dtype,
            n_null_paths   = n_null_paths,
            block_size     = block_size,
            max_depth      = max_depth,
            base_seed      = base_seed,
        )
        if result is None:
            continue

        results_by_timeframe[timeframe] = result
        _print_calibration_result(result)

    elapsed = int(time.time() - start)
    logger.info(f"N_EFF CALIBRATION ── elapsed {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")

    return results_by_timeframe