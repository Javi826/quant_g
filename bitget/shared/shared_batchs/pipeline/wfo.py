#shared_batchs/pipeline/wfo.py
import logging
from functools import partial

import numpy as np
import pandas as pd

from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest, INITIAL_BALANCE
from shared_batchs.tools.wfo_ST import walk_forward_optimization
from shared_batchs.tools.wfo_MC import walk_forward_optimization_mc
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays, get_bars_per_year, get_n_obs, extract_ohlcv_from_path
from shared_batchs.regime import regime_module
from shared_batch_regime.regime_core import lookup_indicator_batch, classify_market_regime

logger = logging.getLogger("BOT_batch.pipeline.wfo")

# =============================================================================
# WFO EXECUTION CONFIG
# =============================================================================
TRAIN_MONTHS         = 6
TEST_MONTHS          = 2
ANCHORED             = False
METRIC_MODE          = "NET_GAIN_PCT"   # "NET_GAIN_PCT" or "CALMAR"
PARAM_SELECTION_MODE = "MODE"           # "MODE", "MEAN" or "EMA"
WFO_MC_N_PATHS       = 100              # MC paths per train window, WFO_MC mode only


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _build_ohlcv_with_signal(
    base_arrays: dict,
    signal_fn: callable,
    signal_params_keys: list,
    param_dict: dict,
    dtype,
) -> dict:
    """Attach generated signals to each symbol's OHLCV arrays for one WFO window."""
    ohlcv_arrays = {}
    for sym, arr in base_arrays.items():
        sig_kwargs = {k: param_dict[k.upper()] for k in signal_params_keys if k.upper() in param_dict}
        signals    = signal_fn(arr, **sig_kwargs, live_trading=False)
        ohlcv_arrays[sym] = {**arr, "signal": np.asarray(signals, dtype=dtype)}
    return ohlcv_arrays


def _compute_metric(results: dict) -> float:
    """Selection metric for WFO window optimization: Net Gain % or Calmar ratio."""
    port         = results.get("__PORTFOLIO__", {})
    trades       = port.get("trades", [])
    net_gain     = float(np.sum(trades)) if trades else 0.0
    net_gain_pct = (net_gain / INITIAL_BALANCE) * 100.0

    if METRIC_MODE == "NET_GAIN_PCT":
        return net_gain_pct

    if METRIC_MODE == "CALMAR":
        max_dd_pct = float(port.get("max_dd", 0.0)) * 100.0
        return net_gain_pct / max_dd_pct if max_dd_pct > 0 else net_gain_pct

    raise ValueError(f"Unknown METRIC_MODE: {METRIC_MODE}")


def _evaluate_fn(
    params: dict,
    base_arrays: dict,
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    dtype,
) -> tuple:
    """Single param combination evaluation for one WFO train window."""
    ohlcv_arrays = _build_ohlcv_with_signal(base_arrays, signal_fn, signal_params_keys, params, dtype)
    results = run_grid_backtest(
        ohlcv_arrays,
        sell_after   = params["SELL_AFTER"],
        tp_pct       = params["TP_PCT"],
        sl_pct       = params["SL_PCT"],
        order_amount = order_amount,
    )
    return _compute_metric(results), params


def _evaluate_fn_mc_paths(
    params: dict,
    paths_per_symbol: dict,
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    dtype,
) -> tuple:
    """Single param combination evaluated across all MC paths of one WFO-MC train window."""
    n_paths = next(iter(paths_per_symbol.values())).shape[0] if paths_per_symbol else 0
    metrics = []

    for path_idx in range(n_paths):
        ohlcv_arrays = extract_ohlcv_from_path(paths_per_symbol, path_idx, dtype=dtype)
        ohlcv_arrays = _build_ohlcv_with_signal(ohlcv_arrays, signal_fn, signal_params_keys, params, dtype)
        results = run_grid_backtest(
            ohlcv_arrays,
            sell_after   = params["SELL_AFTER"],
            tp_pct       = params["TP_PCT"],
            sl_pct       = params["SL_SEC"],
            order_amount = order_amount,
        )
        metrics.append(_compute_metric(results))

    avg_metric = float(np.mean(metrics)) if metrics else -np.inf
    return avg_metric, params


def _evaluate_fn_with_regime(
    params: dict,
    base_arrays: dict,
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    dtype,
    bins_to_filter,
    regime_enabled: bool = False,
    indicator_cache: dict = None,
) -> tuple:
    """Single param combination evaluation with regime signal filtering for WFO test windows."""
    _bins = [bins_to_filter] if isinstance(bins_to_filter, str) else list(bins_to_filter)

    ohlcv_arrays = {}
    for sym, arr in base_arrays.items():
        sig_kwargs = {k: params[k.upper()] for k in signal_params_keys if k.upper() in params}
        signals    = signal_fn(arr, **sig_kwargs, live_trading=False)

        if regime_enabled and _bins and _bins != ["neutral"] and indicator_cache:
            sym_cache = indicator_cache.get(sym)
            if sym_cache is not None:
                signal_idxs = np.nonzero(signals)[0]
                if signal_idxs.size > 0:
                    signal_ts = arr["ts"][signal_idxs]
                    lookups   = {
                        key: lookup_indicator_batch(sym_cache["ts"], sym_cache[key], signal_ts)
                        for key in sym_cache if key != "ts"
                    }
                    for i, idx in enumerate(signal_idxs):
                        close_idx = idx - 1 if idx > 0 else idx
                        context = {"close": float(arr["close"][close_idx])}
                        for key, values in lookups.items():
                            context[key] = float(values[i]) if not np.isnan(values[i]) else None
                        if classify_market_regime(context, cfg=regime_module.INDICATOR_CFG) not in _bins:
                            signals[idx] = 0

        ohlcv_arrays[sym] = {**arr, "signal": np.asarray(signals, dtype=dtype)}

    results = run_grid_backtest(
        ohlcv_arrays,
        sell_after   = params["SELL_AFTER"],
        tp_pct       = params["TP_PCT"],
        sl_pct       = params["SL_PCT"],
        order_amount = order_amount,
    )
    return _compute_metric(results), params


def _collect_trades_fn(
    params: dict,
    base_arrays: dict,
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    dtype,
) -> pd.DataFrame:
    """Run backtest with best_params on a window and return the trade log."""
    ohlcv_arrays = _build_ohlcv_with_signal(base_arrays, signal_fn, signal_params_keys, params, dtype)
    results      = run_grid_backtest(
        ohlcv_arrays,
        sell_after   = params["SELL_AFTER"],
        tp_pct       = params["TP_PCT"],
        sl_pct       = params["SL_PCT"],
        order_amount = order_amount,
    )
    trades             = results["__PORTFOLIO__"]["trade_log"].copy()
    trades.columns     = trades.columns.str.lower().str.strip()
    trades["buy_time"] = pd.to_datetime(trades["buy_time"])
    return trades


def _collect_trades_fn_with_regime(
    params: dict,
    base_arrays: dict,
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    dtype,
    bins_to_filter,
    regime_enabled: bool,
    indicator_cache: dict,
) -> pd.DataFrame:
    """Run backtest with regime filtering on a window and return the trade log."""
    _bins = [bins_to_filter] if isinstance(bins_to_filter, str) else list(bins_to_filter)

    ohlcv_arrays = {}
    for sym, arr in base_arrays.items():
        sig_kwargs = {k: params[k.upper()] for k in signal_params_keys if k.upper() in params}
        signals    = signal_fn(arr, **sig_kwargs, live_trading=False)

        if regime_enabled and _bins and _bins != ["neutral"] and indicator_cache:
            sym_cache = indicator_cache.get(sym)
            if sym_cache is not None:
                signal_idxs = np.nonzero(signals)[0]
                if signal_idxs.size > 0:
                    signal_ts = arr["ts"][signal_idxs]
                    lookups   = {
                        key: lookup_indicator_batch(sym_cache["ts"], sym_cache[key], signal_ts)
                        for key in sym_cache if key != "ts"
                    }
                    for i, idx in enumerate(signal_idxs):
                        close_idx = idx - 1 if idx > 0 else idx
                        context = {"close": float(arr["close"][close_idx])}
                        for key, values in lookups.items():
                            context[key] = float(values[i]) if not np.isnan(values[i]) else None
                        if classify_market_regime(context, cfg=regime_module.INDICATOR_CFG) not in _bins:
                            signals[idx] = 0

        ohlcv_arrays[sym] = {**arr, "signal": np.asarray(signals, dtype=dtype)}

    results            = run_grid_backtest(
        ohlcv_arrays,
        sell_after   = params["SELL_AFTER"],
        tp_pct       = params["TP_PCT"],
        sl_pct       = params["SL_PCT"],
        order_amount = order_amount,
    )
    trades             = results["__PORTFOLIO__"]["trade_log"].copy()
    trades.columns     = trades.columns.str.lower().str.strip()
    trades["buy_time"] = pd.to_datetime(trades["buy_time"])
    return trades


# =============================================================================
# APPROVAL CRITERION
# =============================================================================

def _evaluate_wfo_approval(
    df_results: pd.DataFrame,
    th_rate: float,
    win_rate_th: float,
) -> tuple:
    """
    Approval criterion based on WFO per-window out-of-sample (test) performance.
    Excludes the aggregated summary row (last row of df_results).

    Returns:
        tuple: (approved, win_rate)
    """
    criteria = df_results.iloc[:-1]["best_crite"]
    win_rate = float((criteria > th_rate).mean())
    approved = win_rate >= win_rate_th

    return approved, win_rate


# =============================================================================
# RUN WFO IS
# =============================================================================

def run_wfo_is(
    ohlcv_data: dict,
    param_names: list,
    lists_for_grid: list,
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    timeframe: str,
    th_rate: float,
    win_rate_th: float,
    dtype,
    n_jobs: int = -1,
    show_progress: bool = False,
    n_symbols: int = None,
    bins_to_filter=None,
    regime_enabled: bool = False,
) -> tuple:
    """
    Run Walk-Forward Optimization on IS data and evaluate the window-based approval criterion.
    Collects trade logs for each train and test window.

    Returns:
        tuple: (best_params, approved_wfo, win_rate, wfo_train_trades, wfo_test_trades, df_results)
    """
    ohlcv_arr    = prepare_ohlcv_arrays(ohlcv_data)
    param_ranges = dict(zip(param_names, lists_for_grid))

    bars_per_month   = get_bars_per_year(timeframe) / 12
    length_train_set = int(TRAIN_MONTHS * bars_per_month)
    pct_train_set    = TRAIN_MONTHS / (TRAIN_MONTHS + TEST_MONTHS)

    evaluate_fn = partial(
        _evaluate_fn,
        signal_fn          = signal_fn,
        signal_params_keys = signal_params_keys,
        order_amount       = order_amount,
        dtype              = dtype,
    )

    collect_train_fn = partial(
        _collect_trades_fn,
        signal_fn          = signal_fn,
        signal_params_keys = signal_params_keys,
        order_amount       = order_amount,
        dtype              = dtype,
    )

    test_evaluate_fn = None
    collect_test_fn  = None

    if regime_enabled and bins_to_filter:
        indicator_cache = {}
        for sym in ohlcv_data:
            cache = regime_module._get_indicator_cache(sym)
            if cache is not None:
                indicator_cache[sym] = cache

        test_evaluate_fn = partial(
            _evaluate_fn_with_regime,
            signal_fn          = signal_fn,
            signal_params_keys = signal_params_keys,
            order_amount       = order_amount,
            dtype              = dtype,
            bins_to_filter     = bins_to_filter,
            regime_enabled     = regime_enabled,
            indicator_cache    = indicator_cache,
        )
        collect_test_fn = partial(
            _collect_trades_fn_with_regime,
            signal_fn          = signal_fn,
            signal_params_keys = signal_params_keys,
            order_amount       = order_amount,
            dtype              = dtype,
            bins_to_filter     = bins_to_filter,
            regime_enabled     = regime_enabled,
            indicator_cache    = indicator_cache,
        )
    else:
        collect_test_fn = collect_train_fn

    best_params, df_results, wfo_train_trades, wfo_test_trades = walk_forward_optimization(
        ohlcv_arr               = ohlcv_arr,
        param_ranges            = param_ranges,
        length_train_set        = length_train_set,
        pct_train_set           = pct_train_set,
        anchored                = ANCHORED,
        evaluate_fn             = evaluate_fn,
        param_selection_mode    = PARAM_SELECTION_MODE,
        n_jobs                  = n_jobs,
        show_progress           = show_progress,
        n_symbols               = n_symbols,
        test_evaluate_fn        = test_evaluate_fn,
        collect_train_trades_fn = collect_train_fn,
        collect_test_trades_fn  = collect_test_fn,
    )

    approved_wfo, win_rate = _evaluate_wfo_approval(
        df_results  = df_results,
        th_rate     = th_rate,
        win_rate_th = win_rate_th,
    )

    return best_params, approved_wfo, win_rate, wfo_train_trades, wfo_test_trades, df_results


# =============================================================================
# RUN WFO-MC IS
# =============================================================================

def run_wfo_mc_is(
    ohlcv_data: dict,
    param_names: list,
    lists_for_grid: list,
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    timeframe: str,
    th_rate: float,
    win_rate_th: float,
    dtype,
    n_jobs: int = -1,
    show_progress: bool = False,
) -> tuple:

    param_ranges = dict(zip(param_names, lists_for_grid))

    bars_per_month   = get_bars_per_year(timeframe) / 12
    length_train_set = int(TRAIN_MONTHS * bars_per_month)
    pct_train_set    = TRAIN_MONTHS / (TRAIN_MONTHS + TEST_MONTHS)
    n_obs            = get_n_obs(timeframe)

    train_evaluate_fn = partial(
        _evaluate_fn_mc_paths,
        signal_fn          = signal_fn,
        signal_params_keys = signal_params_keys,
        order_amount       = order_amount,
        dtype              = dtype,
    )
    test_evaluate_fn = partial(
        _evaluate_fn,
        signal_fn          = signal_fn,
        signal_params_keys = signal_params_keys,
        order_amount       = order_amount,
        dtype              = dtype,
    )

    best_params, df_results = walk_forward_optimization_mc(
        ohlcv_data           = ohlcv_data,
        param_ranges         = param_ranges,
        length_train_set     = length_train_set,
        pct_train_set        = pct_train_set,
        anchored             = ANCHORED,
        train_evaluate_fn    = train_evaluate_fn,
        test_evaluate_fn     = test_evaluate_fn,
        n_paths              = WFO_MC_N_PATHS,
        n_obs                = n_obs,
        param_selection_mode = PARAM_SELECTION_MODE,
        n_jobs               = n_jobs,
        show_progress        = show_progress,
    )

    approved_wfo, win_rate = _evaluate_wfo_approval(
        df_results  = df_results,
        th_rate     = th_rate,
        win_rate_th = win_rate_th,
    )

    verdict    = "🟢 PASS" if approved_wfo else "🔴 FAIL"
    params_str = " | ".join(f"{k}={v}" for k, v in best_params.items() if k not in ("SELL_AFTER",))
    logger.info(f"STAGE 1 ── WFO-MC results ── {verdict} WinRate={win_rate*100:.1f}%")
    logger.info(f"STAGE 1 ── WFO-MC params  ── {params_str}")

    return best_params, approved_wfo, win_rate