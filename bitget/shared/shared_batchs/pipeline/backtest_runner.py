#shared_batchs/pipeline/backtest_runner.py
import itertools
import logging
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from shared_batchs.setup.config_backtest import INITIAL_BALANCE, COMISION
from shared_batchs.backtesters.ZX_compute_BT import backtest_core
from shared_batchs.backtesters.ZX_compute_BT import prepare_backtest_data
from shared_batchs.backtesters.ZX_compute_BT import prepare_static_arrays
from shared_batchs.backtesters.ZX_compute_BT import prepare_signal_arrays
from multiprocessing.shared_memory import SharedMemory
from shared_batchs.utils.batch_metrics import sharpe_from_daily_values, skew_kurtosis_from_daily_values, daily_values_from_sell_days
from shared_batchs.utils.paralelization import arrays_to_shared_memory, arrays_from_shared_memory, compact_columns_inplace
from signals.condition_bank import ConditionBank
logger = logging.getLogger("BOT_batch.pipeline.backtest_runner")
DTYPE                = np.float32
# =============================================================================
# BACKTEST EXECUTION CONFIG
# =============================================================================
BACKTEST_N_JOBS     = -1
BACKTEST_MIN_TRADES = 100
COLUMN_CHUNK_SIZE   = 5000  # columns processed per chunk during final matrix compaction

# =============================================================================
# FULL-PERIOD GRID SEARCH — selection-bias metrics (single pass, no WFO windows)
# =============================================================================

def _combo_id(params: dict) -> str:
    return "_".join(f"{k}{v}" for k, v in sorted(params.items()))

def _global_day_grid(ohlcv_arr: dict) -> tuple:

    all_ts   = np.concatenate([arr["ts"] for arr in ohlcv_arr.values()])
    all_days = all_ts.astype("datetime64[D]")
    global_start_day = all_days.min()
    global_end_day   = all_days.max()
    n_days_range = int((global_end_day - global_start_day) / np.timedelta64(1, "D")) + 1
    return global_start_day, n_days_range


def _compact_matrix(matrix_arr: np.ndarray, col_names: np.ndarray, valid_mask: np.ndarray) -> tuple:

    n_keep_cols = compact_columns_inplace(valid_mask, matrix_arr, col_names, chunk_size=COLUMN_CHUNK_SIZE, axis=1)
    matrix_arr  = matrix_arr[:, :n_keep_cols]
    col_names   = col_names[:n_keep_cols]

    if matrix_arr.shape[1] == 0:
        return matrix_arr, col_names.tolist()

    row_mask = np.zeros(matrix_arr.shape[0], dtype=bool)
    for start in range(0, matrix_arr.shape[1], COLUMN_CHUNK_SIZE):
        end = min(start + COLUMN_CHUNK_SIZE, matrix_arr.shape[1])
        row_mask |= np.any(matrix_arr[:, start:end] != 0, axis=1)

    n_keep_rows = compact_columns_inplace(row_mask, matrix_arr, chunk_size=COLUMN_CHUNK_SIZE, axis=0)
    matrix_arr  = matrix_arr[:n_keep_rows, :]

    return matrix_arr, col_names.tolist()

def _build_full_period_ohlcv(ohlcv_arr: dict, signal_fn: callable, condition_banks: dict | None = None) -> dict:
    ohlcv_arrays = {}
    for sym, arr in ohlcv_arr.items():
        bank    = condition_banks.get(sym) if condition_banks else None
        signals = signal_fn(arr, live_trading=False, bank=bank)
        ohlcv_arrays[sym] = {**arr, "signal": np.asarray(signals, dtype=DTYPE)}
    return ohlcv_arrays

def prepare_full_period_data(ohlcv_arr: dict, signal_fn: callable, condition_banks: dict | None = None):
    ohlcv_arrays = _build_full_period_ohlcv(ohlcv_arr, signal_fn, condition_banks)
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

def _evaluate_combo_sharpe(
    params: dict,
    prepared_arrays,
    order_amount: int,
    matrix_view: np.ndarray,
    valid_view: np.ndarray,
    col_idx: int,
    global_start_day: np.datetime64,
) -> tuple:
    n_trades, sell_time_int, profits = _run_backtest_core_light(
        prepared_arrays,
        sell_after   = params["SELL_AFTER"],
        tp_pct       = params["TP_PCT"],
        sl_pct       = params["SL_PCT"],
        order_amount = order_amount,
    )
    if n_trades == 0 or n_trades < BACKTEST_MIN_TRADES:
        return -np.inf, params, None

    daily_values, n_days, start_day = _daily_values_from_trades(sell_time_int, profits)

    sharpe_metric = sharpe_from_daily_values(daily_values)
    sharpe_rank   = sharpe_metric if np.isfinite(sharpe_metric) else -np.inf

    if np.count_nonzero(daily_values) > 1:
        row_offset = int((start_day - global_start_day) / np.timedelta64(1, "D"))
        matrix_view[row_offset:row_offset + n_days, col_idx] = daily_values.astype(np.float32)
        valid_view[col_idx] = 1

    return sharpe_rank, params, (daily_values, n_days, sharpe_metric)

def _run_full_period_for_rule(
    rule_id: str,
    rule_idx: int,
    ohlcv_arr: dict,
    signal_fn: callable,
    param_grid: dict,
    order_amount: int,
    matrix_view: np.ndarray,
    valid_view: np.ndarray,
    global_start_day: np.datetime64,
    n_combos: int,
    static_bundle: dict | None = None,
    condition_banks: dict | None = None,
) -> tuple:

    keys   = list(param_grid.keys())
    combos = [dict(zip(keys, c)) for c in itertools.product(*[param_grid[k] for k in keys])]

    ohlcv_arrays        = _build_full_period_ohlcv(ohlcv_arr, signal_fn, condition_banks)
    max_possible_trades = sum(int(np.count_nonzero(arr["signal"])) for arr in ohlcv_arrays.values())

    if max_possible_trades < BACKTEST_MIN_TRADES:
        return rule_id, {**_empty_winner_metrics(), "best_combo_id": _combo_id(combos[0])}

    col_base = rule_idx * n_combos

    if static_bundle is not None:
        prepared_data = prepare_signal_arrays(static_bundle, ohlcv_arrays)
    else:
        prepared_data = prepare_backtest_data(ohlcv_arrays)

    prepared_arrays = prepared_data[7]

    rows = [
        _evaluate_combo_sharpe(
            params, prepared_arrays, order_amount,
            matrix_view, valid_view, col_base + combo_idx, global_start_day,
        )
        for combo_idx, params in enumerate(combos)
    ]

    best_sharpe, best_params, best_bundle = max(rows, key=lambda x: x[0])
    best_combo_id = _combo_id(best_params)

    if best_bundle is None:
        winner_metrics = _empty_winner_metrics()
    else:
        best_daily_values, best_n_days, best_sharpe_metric = best_bundle
        winner_metrics = _winner_metrics_from_daily_values(best_daily_values, best_n_days, best_sharpe_metric)

    return rule_id, {**winner_metrics, "best_combo_id": best_combo_id}

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

_CONDITION_BANK_CACHE: dict = {}

def _get_condition_banks(shm_metadata: dict, ohlcv_arr: dict) -> dict:
    """
    One ConditionBank per symbol, built once per worker process and reused
    across every rule/combo this worker evaluates. Same eviction policy as
    _get_static_bundle: only one timeframe's banks live in cache at a time.
    """
    cache_key = _static_bundle_cache_key(shm_metadata)
    if cache_key is None:
        return {sym: ConditionBank(arr) for sym, arr in ohlcv_arr.items()}

    banks = _CONDITION_BANK_CACHE.get(cache_key)
    if banks is None:
        _CONDITION_BANK_CACHE.clear()
        banks = {sym: ConditionBank(arr) for sym, arr in ohlcv_arr.items()}
        _CONDITION_BANK_CACHE[cache_key] = banks
    return banks

def _run_full_period_for_rule_shm(
    rule_id: str,
    rule_idx: int,
    shm_metadata: dict,
    signal_fn: callable,
    param_grid: dict,
    order_amount: int,
    matrix_metadata: dict,
    valid_metadata: dict,
    global_start_day: np.datetime64,
    n_combos: int,
) -> tuple:

    ohlcv_arr, shm_handles = arrays_from_shared_memory(shm_metadata)
    matrix_shm = SharedMemory(name=matrix_metadata["name"], create=False)
    valid_shm  = SharedMemory(name=valid_metadata["name"], create=False)
    try:
        matrix_view = np.ndarray(matrix_metadata["shape"], dtype=np.dtype(matrix_metadata["dtype"]), buffer=matrix_shm.buf)
        valid_view  = np.ndarray(valid_metadata["shape"], dtype=np.dtype(valid_metadata["dtype"]), buffer=valid_shm.buf)

        static_bundle   = _get_static_bundle(shm_metadata, ohlcv_arr)
        condition_banks = _get_condition_banks(shm_metadata, ohlcv_arr)
        return _run_full_period_for_rule(
            rule_id, rule_idx, ohlcv_arr, signal_fn, param_grid, order_amount,
            matrix_view, valid_view, global_start_day, n_combos, static_bundle, condition_banks,
        )
    finally:
        matrix_shm.close()
        valid_shm.close()
        for shm in shm_handles:
            shm.close()
            
def run_full_period_search(
    rules: list,
    param_grid: dict,
    order_amount: int,
    global_start_day: np.datetime64,
    n_days_range: int,
    n_combos: int,
    progress_label: str = "",
) -> tuple:

    desc = f"BACKTEST FULL   {progress_label}".strip()

    if not rules:
        return {}, np.empty((0, 0), dtype=np.float32), np.empty((0,), dtype=bool)

    n_rules = len(rules)
    n_cols  = n_rules * n_combos

    matrix_shm = SharedMemory(create=True, size=max(n_days_range * n_cols * 4, 1))
    valid_shm  = SharedMemory(create=True, size=max(n_cols, 1))
    matrix_metadata = {"name": matrix_shm.name, "shape": (n_days_range, n_cols), "dtype": "float32"}
    valid_metadata  = {"name": valid_shm.name,  "shape": (n_cols,),              "dtype": "int8"}

    # Zero-init: SharedMemory does not guarantee zeroed pages.
    np.ndarray(matrix_metadata["shape"], dtype=np.float32, buffer=matrix_shm.buf)[:] = 0.0
    np.ndarray(valid_metadata["shape"],  dtype=np.int8,    buffer=valid_shm.buf)[:]  = 0

    shm_list, ohlcv_metadata = arrays_to_shared_memory(rules[0]["ohlcv_arr"])
    try:
        results = list(tqdm(
        Parallel(n_jobs=BACKTEST_N_JOBS, batch_size=1, pre_dispatch='all', return_as="generator")(
            delayed(_run_full_period_for_rule_shm)(
                r["rule_id"], r["rule_idx"], ohlcv_metadata, r["signal_fn"], param_grid, order_amount,
                matrix_metadata, valid_metadata, global_start_day, n_combos,
            )
            for r in rules
            ),
            desc=desc,
            total=len(rules),
            dynamic_ncols=True,
        ))
    finally:
        for shm in shm_list:
            shm.close()
            shm.unlink()

    try:
        matrix_shared = np.ndarray(matrix_metadata["shape"], dtype=np.float32, buffer=matrix_shm.buf)
        valid_shared  = np.ndarray(valid_metadata["shape"], dtype=np.int8, buffer=valid_shm.buf)
        matrix_arr = matrix_shared.copy()
        valid_mask = valid_shared.astype(bool)
    finally:
        matrix_shm.close()
        matrix_shm.unlink()
        valid_shm.close()
        valid_shm.unlink()

    return dict(results), matrix_arr, valid_mask


# =============================================================================
# PIPE BACKTESTING — the single stage that runs the search and returns the
# =============================================================================
def pipe_backtesting(
    rules: list,
    ohlcv_arr: dict,
    param_grid: dict,
    order_amount: int,
    timeframe: str = "",
) -> tuple:

    n_combos = 1
    for _values in param_grid.values():
        n_combos *= len(_values)

    global_start_day, n_days_range = _global_day_grid(ohlcv_arr)

    rules_for_search = [
        {"rule_id": r["rule_id"], "rule_idx": i, "ohlcv_arr": ohlcv_arr, "signal_fn": r["signal_fn"]}
        for i, r in enumerate(rules)
    ]
    full_period_by_rule, matrix_arr, valid_mask = run_full_period_search(
        rules            = rules_for_search,
        param_grid       = param_grid,
        order_amount     = order_amount,
        global_start_day = global_start_day,
        n_days_range     = n_days_range,
        n_combos         = n_combos,
        progress_label   = timeframe,
    )

    raw_results = [
        {**r, **full_period_by_rule[r["rule_id"]]}
        for r in rules
    ]

    keys      = list(param_grid.keys())
    combo_ids = [_combo_id(dict(zip(keys, c))) for c in itertools.product(*[param_grid[k] for k in keys])]
    col_names = np.array([f"{r['rule_id']}__{cid}" for r in rules for cid in combo_ids], dtype=object)

    matrix_arr, col_names = _compact_matrix(matrix_arr, col_names, valid_mask)

    return raw_results, n_combos, matrix_arr, col_names