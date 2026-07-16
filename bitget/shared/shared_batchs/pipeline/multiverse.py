#shared_batchs/pipeline/multiverse.py
import logging
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from shared_config import VOLUME_COL
from shared_batchs.pipeline.wfo import run_wfo_is
from shared_batchs.engines.optimize_MC import generate_paths_for_all_symbols_functional
logger = logging.getLogger("BOT_batch.pipeline.multiverse")

# =============================================================================
# MULTIVERSE EXECUTION CONFIG
# =============================================================================

N_PATHS    = 500 
N_JOBS     = -1   
BLOCK_SIZE = 20     
DEBUG_MAX_PATHS = 3

# =============================================================================
# PRIVATE HELPERS
# =============================================================================
def _synthetic_ohlcv_data(paths_per_symbol: dict, path_idx: int, ts_index: np.ndarray, dtype) -> dict:

    ohlcv_data = {}
    for sym, arr_paths in paths_per_symbol.items():
        if path_idx >= arr_paths.shape[0]:
            continue
        arr = arr_paths[path_idx]  # (n_obs, n_features)
        ohlcv_data[sym] = pd.DataFrame(
            {
                "open":       arr[:, 0].astype(dtype),
                "low":        arr[:, 1].astype(dtype),
                "high":       arr[:, 2].astype(dtype),
                "close":      arr[:, 3].astype(dtype),
                "low_time":   np.array(arr[:, 4], dtype="datetime64[ns]"),
                "high_time":  np.array(arr[:, 5], dtype="datetime64[ns]"),
                VOLUME_COL:   arr[:, 7].astype(dtype),
            },
            index=pd.DatetimeIndex(ts_index),
        )
    return ohlcv_data


def _evaluate_universe(
    path_idx: int,
    paths: dict,
    ts_index: np.ndarray,
    n_symbols_expected: int,
    param_names: list,
    lists_for_grid: list,
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    timeframe: str,
    net_gain_th: float,
    dd_th: float,
    r2_th: float,
    wfr_th: float,
    dtype,
    n_symbols: int,
) -> tuple:

    synthetic_ohlcv = _synthetic_ohlcv_data(paths, path_idx, ts_index, dtype)
    if len(synthetic_ohlcv) < n_symbols_expected:
        return None, None

    (
        _best_params, _approved_wfo, _net_gain, _max_dd, _train_trades, wfo_test_trades,
        _df_results, _wfr, _window_best_params, _window_test_arrays, _window_test_start_ts,
    ) = run_wfo_is(
        ohlcv_data          = synthetic_ohlcv,
        param_names         = param_names,
        lists_for_grid       = lists_for_grid,
        signal_fn            = signal_fn,
        signal_params_keys   = signal_params_keys,
        order_amount         = order_amount,
        timeframe            = timeframe,
        net_gain_th          = net_gain_th,
        dd_th                = dd_th,
        r2_th                = r2_th,
        wfr_th               = wfr_th,
        dtype                = dtype,
        n_jobs               = 1,
        show_progress        = False,
        n_symbols            = n_symbols,
    )

    if wfo_test_trades is None or wfo_test_trades.empty:
        if path_idx < DEBUG_MAX_PATHS:
            logger.debug(f"MULTIVERSE path={path_idx} ── no test trades ── result=False profit_sum=0.0")
        return False, 0.0

    profit_sum = float(wfo_test_trades["profit"].sum())
    approved   = profit_sum > 0

    if path_idx < DEBUG_MAX_PATHS:
        per_window = wfo_test_trades.groupby("wfo_window")["profit"].sum()
        window_breakdown = " | ".join(f"w{w}={p:.2f}" for w, p in per_window.items())
        logger.debug(
            f"MULTIVERSE path={path_idx} ── {len(per_window)} windows with trades ── "
            f"{window_breakdown} ── TOTAL={profit_sum:.2f} -> {'PASS' if approved else 'FAIL'}"
        )

    return approved, profit_sum


# =============================================================================
# APPROVAL CRITERION
# =============================================================================
def _evaluate_multiverse_approval(pct_profitable: float, pct_profitable_th: float) -> bool:
    return pct_profitable >= pct_profitable_th

# =============================================================================
# RUN MULTIVERSE
# =============================================================================
def pipe_multiverse(
    ohlcv_data: dict,
    timeframe: str,
    param_grid: dict,
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    net_gain_th: float,
    dd_th: float,
    r2_th: float,
    wfr_th: float,
    dtype,
    n_symbols: int,
    pct_profitable_th: float,
    n_paths: int = N_PATHS,
    n_jobs: int = N_JOBS,
    block_size: int = BLOCK_SIZE,
) -> tuple:

    if not ohlcv_data:
        return False, 0.0

    ref_sym  = max(ohlcv_data.keys(), key=lambda sym: len(ohlcv_data[sym]))
    n_obs    = len(ohlcv_data[ref_sym])
    ts_index = ohlcv_data[ref_sym].index[:n_obs].to_numpy()

    paths = generate_paths_for_all_symbols_functional(
        ohlcv_data, n_paths=n_paths, n_obs=n_obs, raw_columns=[VOLUME_COL], block_size=block_size,
    )

    param_names    = list(param_grid.keys())
    lists_for_grid = [param_grid[k] for k in param_names]
    n_symbols_expected = len(ohlcv_data)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_evaluate_universe)(
            path_idx, paths, ts_index, n_symbols_expected, param_names, lists_for_grid,
            signal_fn, signal_params_keys, order_amount, timeframe,
            net_gain_th, dd_th, r2_th, wfr_th, dtype, n_symbols,
        )
        for path_idx in range(n_paths)
    )

    valid_flags   = [r[0] for r in results if r[0] is not None]
    valid_profits = [r[1] for r in results if r[0] is not None]
    n_valid       = len(valid_flags)
    if n_valid == 0:
        return False, 0.0

    n_profitable   = sum(valid_flags)
    pct_profitable = float(n_profitable) / n_valid * 100.0
    approved       = _evaluate_multiverse_approval(pct_profitable, pct_profitable_th)

    logger.debug(
        f"MULTIVERSE ── n_paths={n_paths} valid_universes={n_valid} block_size={block_size} "
        f"pct_profitable={pct_profitable:.1f}% -> {'PASS' if approved else 'FAIL'}"
    )
    return approved, pct_profitable