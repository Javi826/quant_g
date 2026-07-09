#shared_batchs/pipeline/wfo.py
import logging
from functools import partial

import numpy as np
import pandas as pd

from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest, INITIAL_BALANCE
from shared_batchs.engines.wfo_WF import walk_forward_optimization
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays, get_bars_per_year
from shared_batchs.utils.batch_metrics import compute_metrics
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
#     "15m":   {"train_months": 12, "test_months": 4},
#     "30m":   {"train_months": 12, "test_months": 4},
#     "1H":    {"train_months": 12, "test_months": 4},
#     "4H":    {"train_months": 12, "test_months": 4},
#     "6Hutc": {"train_months": 12, "test_months": 4},
# }
# =============================================================================

ANCHORED             = False
METRIC_MODE          = "NET_GAIN_PCT"   # "NET_GAIN_PCT" or "CALMAR"
PARAM_SELECTION_MODE = "MODE"           # "MODE", "MEAN" or "EMA"

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

    collect_test_fn = collect_test_fn_override if collect_test_fn_override is not None else collect_train_fn

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
    
    logger.debug(
        f"STAGE 1 ── WFO completed  ── {n_windows} windows | "
        f"train={_wfo_cfg['train_months']}m  test={_wfo_cfg['test_months']}m"
    )

    approved_wfo, wfo_net_gain, wfo_max_dd = _evaluate_wfo_approval(
        wfo_test_trades = wfo_test_trades,
        net_gain_th     = net_gain_th,
        dd_th           = dd_th,
    )

    return best_params, approved_wfo, wfo_net_gain, wfo_max_dd, wfo_train_trades, wfo_test_trades, df_results