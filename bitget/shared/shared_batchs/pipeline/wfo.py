#shared_batchs/pipeline/wfo.py
import logging
from functools import partial

import numpy as np

from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest, INITIAL_BALANCE
from shared_batchs.tools.wfo import walk_forward_optimization
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays, get_bars_per_year

logger = logging.getLogger("BOT_batch.pipeline.wfo")

# =============================================================================
# WFO EXECUTION CONFIG
# =============================================================================
TRAIN_MONTHS = 12
TEST_MONTHS  = 3
ANCHORED     = True


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
    """Selection metric for WFO window optimization: Net Gain %."""
    port     = results.get("__PORTFOLIO__", {})
    trades   = port.get("trades", [])
    net_gain = float(np.sum(trades)) if trades else 0.0
    return (net_gain / INITIAL_BALANCE) * 100.0


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
    dtype,
    n_jobs: int = -1,
) -> tuple:
    """
    Run Walk-Forward Optimization on IS data.

    Returns:
        tuple: (best_params, None)
    """
    ohlcv_arr        = prepare_ohlcv_arrays(ohlcv_data)
    param_ranges     = dict(zip(param_names, lists_for_grid))

    bars_per_month    = get_bars_per_year(timeframe) / 12
    length_train_set  = int(TRAIN_MONTHS * bars_per_month)
    pct_train_set     = TRAIN_MONTHS / (TRAIN_MONTHS + TEST_MONTHS)

    evaluate_fn = partial(
        _evaluate_fn,
        signal_fn           = signal_fn,
        signal_params_keys  = signal_params_keys,
        order_amount        = order_amount,
        dtype               = dtype,
    )

    best_params = walk_forward_optimization(
        ohlcv_arr        = ohlcv_arr,
        param_ranges     = param_ranges,
        length_train_set = length_train_set,
        pct_train_set    = pct_train_set,
        anchored         = ANCHORED,
        evaluate_fn      = evaluate_fn,
        n_jobs           = n_jobs,
    )

    params_str = " | ".join(f"{k}={v}" for k, v in best_params.items() if k not in ("SELL_AFTER",))
    logger.info(f"STAGE 1 ── WFO Best params        ── {params_str}")

    return best_params, None