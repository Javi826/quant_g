#shared_batchs/runs/run_deploy.py
import importlib.util
import itertools
import logging
import os
import numpy as np
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed

from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest
from shared_batchs.pipeline.wfo import WFO_WINDOW_CONFIG, _build_ohlcv_with_signal, _compute_metric
from shared_batchs.engines.wfo_WF import WARMUP_BARS, _select_window_symbols
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays, get_bars_per_year
from shared_config import VOLUME_COL

logger = logging.getLogger("BOT_batch.runs.run_deploy")


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _slice_deploy_window(
    ohlcv_is: dict,
    timeframe: str,
) -> tuple:

    _wfo_cfg = WFO_WINDOW_CONFIG.get(timeframe)
    if _wfo_cfg is None:
        raise ValueError(f"No WFO window config for timeframe: {timeframe}")

    bars_per_month   = get_bars_per_year(timeframe) / 12
    length_train_set = int(_wfo_cfg["train_months"] * bars_per_month)

    ohlcv_arrays = prepare_ohlcv_arrays(ohlcv_is)
    ref_sym      = max(ohlcv_arrays, key=lambda k: len(ohlcv_arrays[k]["ts"]))
    ref_ts       = ohlcv_arrays[ref_sym]["ts"]
    start_idx    = max(0, len(ref_ts) - length_train_set)
    train_start_ts = ref_ts[start_idx]
    train_end_ts   = ref_ts[-1]

    ohlcv_window = {}
    for sym, arr in ohlcv_arrays.items():
        sym_ts     = arr["ts"]
        t0         = int(np.searchsorted(sym_ts, train_start_ts, side="left"))
        warm_start = max(0, t0 - WARMUP_BARS)
        if t0 >= len(sym_ts):
            continue
        ohlcv_window[sym] = {
            "ts":        sym_ts[warm_start:],
            "open":      arr["open"][warm_start:],
            "high":      arr["high"][warm_start:],
            "low":       arr["low"][warm_start:],
            "close":     arr["close"][warm_start:],
            VOLUME_COL:  arr.get(VOLUME_COL, arr["close"] * 0)[warm_start:],
            "low_time":  arr["low_time"][warm_start:],
            "high_time": arr["high_time"][warm_start:],
        }

    return ohlcv_window, train_start_ts, train_end_ts


def _select_deploy_symbols(
    ohlcv_window: dict,
    n_symbols: int,
    train_start_ts,
) -> dict:
    """Select top n_symbols by average volume within the deploy train window."""
    candidate_indices = {}
    for sym, arr in ohlcv_window.items():
        sym_ts = arr["ts"]
        t0     = int(np.searchsorted(sym_ts, train_start_ts, side="left"))
        t1     = len(sym_ts)
        if t1 > t0:
            candidate_indices[sym] = (t0, t1, t0, t1)

    selected = _select_window_symbols(candidate_indices, ohlcv_window, n_symbols)
    return {sym: ohlcv_window[sym] for sym in selected}


def _run_deploy_grid(
    ohlcv_selected: dict,
    param_names: list,
    lists_for_grid: list,
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    dtype,
    n_jobs: int,
) -> dict:
    """Run parallel grid search on deploy window and return best params."""
    dict_combinations = [
        dict(zip(param_names, comb))
        for comb in itertools.product(*lists_for_grid)
    ]

    def _evaluate(params):
        arrays  = _build_ohlcv_with_signal(ohlcv_selected, signal_fn, signal_params_keys, params, dtype)
        results = run_grid_backtest(
            arrays,
            sell_after   = params["SELL_AFTER"],
            tp_pct       = params["TP_PCT"],
            sl_pct       = params["SL_PCT"],
            order_amount = order_amount,
        )
        return _compute_metric(results), params

    results        = Parallel(n_jobs=n_jobs)(delayed(_evaluate)(p) for p in dict_combinations)
    _, best_params = max(results, key=lambda x: x[0])
    return best_params


def _save_deploy_symbols(
    strategy_id: str,
    deploy_symbols: list,
    timeframe: str,
    symbols_live_folder: str,
) -> bool:

    os.makedirs(symbols_live_folder, exist_ok=True)
    path = os.path.join(symbols_live_folder, f"symbols_live_{strategy_id}_{timeframe}.csv")

    if os.path.exists(path):
        prev_symbols    = pd.read_csv(path, header=None)[0].tolist()
        symbols_changed = prev_symbols != list(deploy_symbols)
    else:
        symbols_changed = True

    pd.DataFrame(deploy_symbols).to_csv(path, index=False, header=False)
    logger.debug(f"symbols_live saved → {path}")

    return symbols_changed