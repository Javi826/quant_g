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

N_PATHS    = 2 
N_JOBS     = 1   
BLOCK_SIZE = 1     
DEBUG_MAX_PATHS = 1
# === DEBUG BLOCK: DRIFT ANALYSIS (remove entire block to disable) ===
DEBUG_DRIFT_ANALYSIS = True

def _log_drift_analysis(ohlcv_data: dict, paths: dict, block_size: int) -> None:
    rows = []
    for sym, df_hist in ohlcv_data.items():
        arr_paths = paths.get(sym)
        if arr_paths is None or arr_paths.shape[0] == 0:
            continue

        hist_close = df_hist["close"].to_numpy(dtype=np.float64)
        hist_open  = df_hist["open"].to_numpy(dtype=np.float64)
        hist_mean_ret       = float(np.mean((hist_close - hist_open) / hist_open))
        hist_total_ret_pct  = float((hist_close[-1] / hist_close[0] - 1.0) * 100.0)

        synth_open  = arr_paths[:, :, 0].astype(np.float64)
        synth_close = arr_paths[:, :, 3].astype(np.float64)
        synth_ret_per_bar   = (synth_close - synth_open) / synth_open
        synth_mean_ret      = float(np.mean(synth_ret_per_bar))
        synth_total_ret_pct = (synth_close[:, -1] / synth_close[:, 0] - 1.0) * 100.0
        synth_mean_total_ret_pct = float(np.mean(synth_total_ret_pct))
        synth_pct_paths_bullish  = float(np.mean(synth_total_ret_pct > 0) * 100.0)

        rows.append({
            "symbol":                   sym,
            "hist_mean_ret":            hist_mean_ret,
            "hist_total_ret_pct":       hist_total_ret_pct,
            "synth_mean_ret":           synth_mean_ret,
            "synth_mean_total_ret_pct": synth_mean_total_ret_pct,
            "synth_pct_paths_bullish":  synth_pct_paths_bullish,
            "block_size":               block_size,
        })

    if not rows:
        logger.warning("DRIFT ANALYSIS ── no valid symbols to analyze")
        return

    df_drift = pd.DataFrame(rows)
    summary  = df_drift.drop(columns=["symbol", "block_size"]).mean()
    df_drift = pd.concat(
        [df_drift, pd.DataFrame([{
            "symbol": "MEAN",
            **summary.to_dict(),
            "block_size": block_size,
        }])],
        ignore_index=True,
    )

    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    logger.info(f"\n{'─' * 115}")
    logger.info("  DRIFT ANALYSIS ── historical vs synthetic paths")
    logger.info(f"{'─' * 115}")
    logger.info(f"\n{df_drift.to_string(index=False)}")
    logger.info(f"{'─' * 115}\n")
# === END DEBUG BLOCK ===
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
    # === DEBUG BLOCK: DRIFT ANALYSIS (remove entire block to disable) ===
    if DEBUG_DRIFT_ANALYSIS:
        _log_drift_analysis(ohlcv_data, paths, block_size)
    # === END DEBUG BLOCK ===

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