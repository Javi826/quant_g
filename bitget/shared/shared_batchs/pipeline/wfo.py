#shared_batchs/pipeline/wfo.py
import logging
from functools import partial

import numpy as np
import pandas as pd

from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest, INITIAL_BALANCE
from shared_batchs.engines.wfo_WF import walk_forward_optimization
from shared_batchs.engines.wfo_MC import walk_forward_optimization_mc
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays, get_bars_per_year, get_n_obs, extract_ohlcv_from_path
from shared_batchs.utils.batch_metrics import compute_metrics
from shared_batchs.regime import regime_module
from shared_batch_regime.config_paths import DATA_FOLDER_IS
from shared_batch_regime.regime_core import apply_regime_filter
logger = logging.getLogger("BOT_batch.pipeline.wfo")

# =============================================================================
# WFO EXECUTION CONFIG
# =============================================================================
WFO_WINDOW_CONFIG = {
    "15m":   {"train_months": 9, "test_months": 2},
    "30m":   {"train_months": 9, "test_months": 2},
    "1H":    {"train_months": 9, "test_months": 2},
    "4H":    {"train_months": 9, "test_months": 2},
    "6Hutc": {"train_months": 9, "test_months": 2},
}

# =============================================================================
# WFO_WINDOW_CONFIG = {
#     "15m":   {"train_months": 12, "test_months": 3},
#     "30m":   {"train_months": 12, "test_months": 3},
#     "1H":    {"train_months": 12, "test_months": 3},
#     "4H":    {"train_months": 12, "test_months":3},
#     "6Hutc": {"train_months": 12, "test_months":3},
# }
# =============================================================================

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
    """Single param combination evaluation with regime signal filtering for WFO train windows."""
    _bins = [bins_to_filter] if isinstance(bins_to_filter, str) else list(bins_to_filter)

    ohlcv_arrays = {}
    for sym, arr in base_arrays.items():
        sig_kwargs = {k: params[k.upper()] for k in signal_params_keys if k.upper() in params}
        signals    = signal_fn(arr, **sig_kwargs, live_trading=False)

        if regime_enabled and _bins and _bins != ["neutral"] and indicator_cache:
            signals = apply_regime_filter(
                signals        = signals,
                arr            = arr,
                sym_cache      = indicator_cache.get(sym),
                cfg            = regime_module.INDICATOR_CFG,
                bins_to_filter = _bins,
            )

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
            signals = apply_regime_filter(
                signals        = signals,
                arr            = arr,
                sym_cache      = indicator_cache.get(sym),
                cfg            = regime_module.INDICATOR_CFG,
                bins_to_filter = _bins,
            )

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
    wfo_test_trades: pd.DataFrame,
    net_gain_th: float,
    dd_th: float,
) -> tuple:

    if wfo_test_trades.empty:
        return False, 0.0, 0.0

    m            = compute_metrics(wfo_test_trades, capital=INITIAL_BALANCE, name="")
    net_gain_pct = m["Net_Gain_pct"]
    max_dd_pct   = m["Max_DD_pct"]
    approved     = net_gain_pct > net_gain_th and abs(max_dd_pct) < dd_th

    return approved, net_gain_pct, max_dd_pct

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
    net_gain_th: float,
    dd_th: float,
    dtype,
    n_jobs: int = -1,
    show_progress: bool = False,
    n_symbols: int = None,
    bins_to_filter=None,
    regime_enabled: bool = False,
    collect_test_fn_override: callable = None,
) -> tuple:

    ohlcv_arr    = prepare_ohlcv_arrays(ohlcv_data)
    param_ranges = dict(zip(param_names, lists_for_grid))

    _wfo_cfg = WFO_WINDOW_CONFIG.get(timeframe)
    if _wfo_cfg is None:
        raise ValueError(f"No WFO window config for timeframe: {timeframe}")
    bars_per_month   = get_bars_per_year(timeframe) / 12
    length_train_set = int(_wfo_cfg["train_months"] * bars_per_month)
    pct_train_set    = _wfo_cfg["train_months"] / (_wfo_cfg["train_months"] + _wfo_cfg["test_months"])

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

    collect_test_fn = None

    if collect_test_fn_override is not None:
        collect_test_fn = collect_test_fn_override
    elif regime_enabled and bins_to_filter:
        indicator_cache = {}
        for sym in ohlcv_data:
            cache = regime_module._get_indicator_cache(sym, DATA_FOLDER_IS)
            if cache is not None:
                indicator_cache[sym] = cache

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

    best_params, df_results, wfo_train_trades, wfo_test_trades, n_windows = walk_forward_optimization(
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
        collect_train_trades_fn = collect_train_fn,
        collect_test_trades_fn  = collect_test_fn,
    )
    
    logger.info(
    f"STAGE 1 ── WFO completed  ── {n_windows} windows | "
    f"train={_wfo_cfg['train_months']}m  test={_wfo_cfg['test_months']}m"
    )

    approved_wfo, wfo_net_gain, wfo_max_dd = _evaluate_wfo_approval(
        wfo_test_trades = wfo_test_trades,
        net_gain_th     = net_gain_th,
        dd_th           = dd_th,
    )

    return best_params, approved_wfo, wfo_net_gain, wfo_max_dd, wfo_train_trades, wfo_test_trades, df_results


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
    net_gain_th: float,
    dd_th: float,
    dtype,
    n_jobs: int = -1,
    show_progress: bool = False,
    n_symbols: int = None,
    bins_to_filter=None,
    regime_enabled: bool = False,
) -> tuple:

    ohlcv_arr    = prepare_ohlcv_arrays(ohlcv_data)
    param_ranges = dict(zip(param_names, lists_for_grid))

    _wfo_cfg = WFO_WINDOW_CONFIG.get(timeframe)
    if _wfo_cfg is None:
        raise ValueError(f"No WFO window config for timeframe: {timeframe}")
    bars_per_month   = get_bars_per_year(timeframe) / 12
    length_train_set = int(_wfo_cfg["train_months"] * bars_per_month)
    pct_train_set    = _wfo_cfg["train_months"] / (_wfo_cfg["train_months"] + _wfo_cfg["test_months"])
    n_obs            = get_n_obs(timeframe)

    indicator_cache = None
    if regime_enabled and bins_to_filter:
        indicator_cache = {}
        for sym in ohlcv_data:
            cache = regime_module._get_indicator_cache(sym, DATA_FOLDER_IS)
            if cache is not None:
                indicator_cache[sym] = cache

    best_params, df_results, wfo_train_trades, wfo_test_trades, n_windows = walk_forward_optimization_mc(
        ohlcv_arr             = ohlcv_arr,
        param_ranges          = param_ranges,
        length_train_set      = length_train_set,
        pct_train_set         = pct_train_set,
        anchored              = ANCHORED,
        signal_fn             = signal_fn,
        signal_params_keys    = signal_params_keys,
        order_amount          = order_amount,
        dtype                 = dtype,
        n_paths               = WFO_MC_N_PATHS,
        n_obs                 = n_obs,
        param_selection_mode  = PARAM_SELECTION_MODE,
        n_jobs                = n_jobs,
        show_progress         = show_progress,
        n_symbols             = n_symbols,
        bins_to_filter        = bins_to_filter,
        regime_enabled        = regime_enabled,
        indicator_cache       = indicator_cache,
    )

    logger.info(
        f"STAGE 1 ── WFO-MC completed ── {n_windows} windows | "
        f"train={_wfo_cfg['train_months']}m  test={_wfo_cfg['test_months']}m"
    )

    approved_wfo, wfo_net_gain, wfo_max_dd = _evaluate_wfo_approval(
        wfo_test_trades = wfo_test_trades,
        net_gain_th     = net_gain_th,
        dd_th           = dd_th,
    )

    return best_params, approved_wfo, wfo_net_gain, wfo_max_dd, wfo_train_trades, wfo_test_trades, df_results