#shared_batchs/pipeline/backtest_runner.py
import itertools
import logging
import numpy as np
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from shared_batchs.setup.config_backtest import INITIAL_BALANCE, COMISION
from shared_batchs.backtesters.ZX_compute_BT import backtest_core
from shared_batchs.backtesters.ZX_compute_BT import prepare_backtest_data
from shared_batchs.backtesters.ZX_compute_BT import prepare_static_arrays
from shared_batchs.backtesters.ZX_compute_BT import prepare_signal_arrays
from shared_batchs.utils.batch_metrics import sharpe_from_daily_values, skew_kurtosis_from_daily_values, daily_values_from_sell_days
from shared_batchs.utils.paralelization import arrays_to_shared_memory, arrays_from_shared_memory

logger = logging.getLogger("BOT_batch.pipeline.backtest_runner")

# =============================================================================
# BACKTEST EXECUTION CONFIG
# =============================================================================
BACKTEST_N_JOBS     = -1
BACKTEST_MIN_TRADES = 100

# =============================================================================
# FULL-PERIOD GRID SEARCH — selection-bias metrics (single pass, no WFO windows)
# =============================================================================

def _combo_id(params: dict) -> str:
    return "_".join(f"{k}{v}" for k, v in sorted(params.items()))

def _sparse_daily_profit(daily_values: np.ndarray, start_day: np.datetime64) -> tuple | None:

    nonzero_idx = np.flatnonzero(daily_values)
    if nonzero_idx.size == 0:
        return None
    return (
        nonzero_idx.astype(np.int32),
        daily_values[nonzero_idx].astype(np.float32),
        start_day,
    )

def _build_full_period_ohlcv(ohlcv_arr: dict, signal_fn: callable, dtype) -> dict:
    ohlcv_arrays = {}
    for sym, arr in ohlcv_arr.items():
        signals = signal_fn(arr, live_trading=False)
        ohlcv_arrays[sym] = {**arr, "signal": np.asarray(signals, dtype=dtype)}
    return ohlcv_arrays

def prepare_full_period_data(ohlcv_arr: dict, signal_fn: callable, dtype):

    ohlcv_arrays = _build_full_period_ohlcv(ohlcv_arr, signal_fn, dtype)
    return prepare_backtest_data(ohlcv_arrays)

# =============================================================================
# PRIVATE HELPERS — metrics derived in-place from the backtest core output
# =============================================================================
def _run_backtest_core_light(prepared_arrays, sell_after, tp_pct, sl_pct, order_amount) -> tuple:

    (open_2d, close_2d, high_2d, low_2d,
     high_time_2d, low_time_2d, ts_int_2d, signal_2d, sym_len,
     signal_events, all_timestamps_int, ev_col0) = prepared_arrays

    core_output = backtest_core(
        open_2d, close_2d, high_2d, low_2d,
        high_time_2d, low_time_2d, ts_int_2d, signal_2d, sym_len,
        signal_events, all_timestamps_int, ev_col0,
        float(INITIAL_BALANCE), float(COMISION) / 100.0, float(order_amount),
        int(sell_after), float(tp_pct), float(sl_pct),
    )
    n_trades      = core_output[0]
    sell_time_int = core_output[4]
    profits       = core_output[7]
    return n_trades, sell_time_int, profits

def _daily_values_from_trades(sell_time_int: np.ndarray, profits: np.ndarray) -> tuple:

    sell_days_ns = sell_time_int.view("datetime64[ns]")
    return daily_values_from_sell_days(sell_days_ns, profits)

def _winner_metrics_from_daily_values(daily_values: np.ndarray, n_days: int, sharpe: float) -> dict:
    eq       = INITIAL_BALANCE + np.cumsum(daily_values)
    cm       = np.maximum.accumulate(eq)
    max_dd   = ((eq - cm) / cm * 100).min()
    net_gain = (eq[-1] - INITIAL_BALANCE) / INITIAL_BALANCE * 100

    if n_days > 2:
        skew_val, kurt_val = skew_kurtosis_from_daily_values(daily_values)
    else:
        skew_val, kurt_val = np.nan, np.nan

    return {
        "sharpe_train":   sharpe,
        "skew_train":     skew_val,
        "kurtosis_train": kurt_val,
        "n_days_train":   n_days,
        "net_gain_train": round(float(net_gain), 2),
        "max_dd_train":   round(float(max_dd), 2),
    }

def _empty_winner_metrics() -> dict:
    return {
        "sharpe_train":   np.nan,
        "skew_train":     np.nan,
        "kurtosis_train": np.nan,
        "n_days_train":   0,
        "net_gain_train": np.nan,
        "max_dd_train":   np.nan,
    }

def _evaluate_combo_sharpe(params: dict, prepared_arrays, order_amount: int) -> tuple:
    n_trades, sell_time_int, profits = _run_backtest_core_light(
        prepared_arrays,
        sell_after   = params["SELL_AFTER"],
        tp_pct       = params["TP_PCT"],
        sl_pct       = params["SL_PCT"],
        order_amount = order_amount,
    )
    if n_trades == 0 or n_trades < BACKTEST_MIN_TRADES:
        return -np.inf, params, None, None
 
    daily_values, n_days, start_day = _daily_values_from_trades(sell_time_int, profits)

    sharpe_metric = sharpe_from_daily_values(daily_values)
    sharpe_rank   = sharpe_metric if np.isfinite(sharpe_metric) else -np.inf

    daily_profit = _sparse_daily_profit(daily_values, start_day)
    return sharpe_rank, params, (daily_values, n_days, sharpe_metric), daily_profit

def _run_full_period_for_rule(
    rule_id: str,
    ohlcv_arr: dict,
    signal_fn: callable,
    param_grid: dict,
    order_amount: int,
    dtype,
    static_bundle: dict | None = None,
) -> tuple:

    keys   = list(param_grid.keys())
    combos = [dict(zip(keys, c)) for c in itertools.product(*[param_grid[k] for k in keys])]

    ohlcv_arrays        = _build_full_period_ohlcv(ohlcv_arr, signal_fn, dtype)
    max_possible_trades = sum(int(np.count_nonzero(arr["signal"])) for arr in ohlcv_arrays.values())

    if max_possible_trades < BACKTEST_MIN_TRADES:
        return rule_id, {**_empty_winner_metrics(), "combo_daily_profit": {}, "best_combo_id": _combo_id(combos[0])}

    if static_bundle is not None:
        prepared_data = prepare_signal_arrays(static_bundle, ohlcv_arrays)
    else:
        prepared_data = prepare_backtest_data(ohlcv_arrays)

    prepared_arrays = prepared_data[7]

    rows = [_evaluate_combo_sharpe(params, prepared_arrays, order_amount) for params in combos]

    combo_daily_profit = {
        _combo_id(params): daily_profit
        for _sharpe, params, _bundle, daily_profit in rows
        if daily_profit is not None and daily_profit[0].size > 1
    }
    best_sharpe, best_params, best_bundle, _best_daily = max(rows, key=lambda x: x[0])
    best_combo_id = _combo_id(best_params)

    if best_bundle is None:
        winner_metrics = _empty_winner_metrics()
    else:
        best_daily_values, best_n_days, best_sharpe_metric = best_bundle
        winner_metrics = _winner_metrics_from_daily_values(best_daily_values, best_n_days, best_sharpe_metric)

    return rule_id, {**winner_metrics, "combo_daily_profit": combo_daily_profit, "best_combo_id": best_combo_id}

_STATIC_BUNDLE_CACHE: dict = {}

def _static_bundle_cache_key(shm_metadata: dict):
    try:
        return tuple(
            (sym, key, info.get("name", info.get("value")))
            for sym in sorted(shm_metadata)
            for key, info in sorted(shm_metadata[sym].items())
        )
    except TypeError:
        return None


def _get_static_bundle(shm_metadata: dict, ohlcv_arr: dict):
    cache_key = _static_bundle_cache_key(shm_metadata)
    if cache_key is None:
        return prepare_static_arrays(ohlcv_arr)

    bundle = _STATIC_BUNDLE_CACHE.get(cache_key)
    if bundle is None:
        _STATIC_BUNDLE_CACHE.clear()
        bundle = prepare_static_arrays(ohlcv_arr)
        _STATIC_BUNDLE_CACHE[cache_key] = bundle
    return bundle

def _run_full_period_for_rule_shm(
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
        return _run_full_period_for_rule(rule_id, ohlcv_arr, signal_fn, param_grid, order_amount, dtype, static_bundle)
    finally:
        for shm in shm_handles:
            shm.close()
            
def run_full_period_search(rules: list, param_grid: dict, order_amount: int, dtype, progress_label: str = "") -> dict:

    desc = f"BACKTEST FULL {progress_label}".strip()

    if not rules:
        return {}

    shm_list, shm_metadata = arrays_to_shared_memory(rules[0]["ohlcv_arr"])
    try:
        with tqdm_joblib(tqdm(desc=desc, total=len(rules), dynamic_ncols=True)):
            results = Parallel(n_jobs=BACKTEST_N_JOBS, batch_size=1, pre_dispatch='all')(
                delayed(_run_full_period_for_rule_shm)(
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
# PIPE BACKTESTING — the single stage that runs the search and returns the
# =============================================================================
def pipe_backtesting(
    rules: list,
    ohlcv_arr: dict,
    param_grid: dict,
    order_amount: int,
    dtype,
    timeframe: str = "",
) -> tuple:

    n_combos = 1
    for _values in param_grid.values():
        n_combos *= len(_values)


    rules_for_search = [
        {"rule_id": r["rule_id"], "ohlcv_arr": ohlcv_arr, "signal_fn": r["signal_fn"]}
        for r in rules
    ]
    full_period_by_rule = run_full_period_search(
        rules          = rules_for_search,
        param_grid     = param_grid,
        order_amount   = order_amount,
        dtype          = dtype,
        progress_label = timeframe,
    )

    raw_results = [
        {**r, **full_period_by_rule[r["rule_id"]]}
        for r in rules
    ]
    return raw_results, n_combos