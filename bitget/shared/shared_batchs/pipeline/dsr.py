#shared_batchs/pipeline/dsr.py
import time
import itertools
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from scipy.stats import norm
from shared_batchs.setup.config_backtest import INITIAL_BALANCE, COMISION
from shared_batchs.backtesters.ZX_compute_BT import backtest_core
from shared_batchs.backtesters.ZX_compute_BT import prepare_backtest_data
from shared_batchs.backtesters.ZX_compute_BT import prepare_static_arrays
from shared_batchs.backtesters.ZX_compute_BT import prepare_signal_arrays
from shared_batchs.utils.batch_metrics import sharpe_from_daily_values, skew_kurtosis_from_daily_values
from shared_batchs.utils.paralelization import arrays_to_shared_memory, arrays_from_shared_memory
from shared_batchs.utils.reporting import print_dsr_train_metrics
logger = logging.getLogger("BOT_batch.pipeline.dsr")

# =============================================================================
# DSR EXECUTION CONFIG
# =============================================================================
EULER_GAMMA         = 0.5772156649015328606
SHARPE_PERIODS_YEAR = 365.0               
DSR_N_JOBS          = -1                
DSR_MIN_TRADES      = 100   
DSR_MAX_SHARPE_ANN  = 10.0              
M_TO_T_WARN_RATIO   = 2.0   
        
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

    sell_days    = sell_time_int.view("datetime64[ns]").astype("datetime64[D]")
    start_day    = sell_days.min()
    end_day      = sell_days.max()
    n_days       = int((end_day - start_day).astype("int64")) + 1
    day_offset   = (sell_days - start_day).astype("int64")
    daily_values = np.bincount(day_offset, weights=profits, minlength=n_days)
    return daily_values, n_days, start_day

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
    if n_trades == 0 or n_trades < DSR_MIN_TRADES:
        return -np.inf, params, None, None
 
    daily_values, n_days, start_day = _daily_values_from_trades(sell_time_int, profits)
 
    sharpe_metric = sharpe_from_daily_values(daily_values)
    sharpe_rank   = sharpe_metric if np.isfinite(sharpe_metric) else -np.inf
    if sharpe_rank > DSR_MAX_SHARPE_ANN:
        return -np.inf, params, None, None
 
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

    if max_possible_trades < DSR_MIN_TRADES:
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

    desc = f"DSR FULL-PERIOD SEARCH {progress_label}".strip()

    if not rules:
        return {}

    shm_list, shm_metadata = arrays_to_shared_memory(rules[0]["ohlcv_arr"])
    try:
        with tqdm_joblib(tqdm(desc=desc, total=len(rules), dynamic_ncols=True)):
            results = Parallel(n_jobs=DSR_N_JOBS, batch_size=1, pre_dispatch='all')(
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
# PRIVATE HELPERS — N_eff estimation (streaming Gram accumulation, eigenvalue method)
# =============================================================================

def _iter_daily_profit_columns(all_raw_results: list):

    for r in all_raw_results:
        combo_profit = r.get("combo_daily_profit") or {}
        for combo_id, s in combo_profit.items():
            yield f"{r['rule_id']}__{combo_id}", s

def _column_dates(column: tuple) -> np.ndarray:
    """Rebuild the datetime64[D] index of a sparse column from its raw payload."""
    day_offsets, _values, start_day = column
    return start_day + day_offsets.astype("timedelta64[D]")


def _common_date_axis(all_raw_results: list) -> np.ndarray | None:

    date_arrays = [_column_dates(col) for _col_name, col in _iter_daily_profit_columns(all_raw_results)]
    if len(date_arrays) < 2:
        return None
    return np.unique(np.concatenate(date_arrays))
def _participation_ratio_from_gram(gram: np.ndarray, n_const: int) -> float:

    sum_eig    = float(np.einsum("ii->", gram)) + n_const
    sum_eig_sq = float(np.einsum("ij,ij->", gram, gram)) + n_const
    if sum_eig_sq <= 0:
        return 1.0
    return float((sum_eig ** 2) / sum_eig_sq)

BATCH_SIZE_N_EFF = 2000  # standardized columns accumulated before each BLAS matmul —
                         # bounds RAM to T x BATCH_SIZE_N_EFF while keeping each matmul
def _estimate_n_eff_streaming(all_raw_results: list, all_dates: np.ndarray, batch_size: int = BATCH_SIZE_N_EFF) -> float:

    t_days     = all_dates.shape[0]
    gram       = np.zeros((t_days, t_days), dtype=np.float64)
    n_const    = 0
    n_valid    = 0
    batch_cols = []

    for _col_name, column in _iter_daily_profit_columns(all_raw_results):
        col = np.zeros(t_days, dtype=np.float32)
        row_idx = np.searchsorted(all_dates, _column_dates(column))
        col[row_idx] = column[1].astype(np.float32)

        std = col.std(ddof=1)
        if std <= 0:
            n_const += 1
            continue

        batch_cols.append((col - col.mean()) / std)
        n_valid += 1

        if len(batch_cols) >= batch_size:
            x_batch = np.column_stack(batch_cols)
            gram   += x_batch @ x_batch.T
            batch_cols = []

    if batch_cols:
        x_batch = np.column_stack(batch_cols)
        gram   += x_batch @ x_batch.T

    if n_valid == 0:
        return float(n_const) if n_const > 0 else 1.0

    gram /= (t_days - 1)
    return _participation_ratio_from_gram(gram, n_const)
def estimate_n_eff_flat(all_raw_results: list) -> float | None:

    all_dates = _common_date_axis(all_raw_results)
    if all_dates is None:
        return None
    return _estimate_n_eff_streaming(all_raw_results, all_dates)

# =============================================================================
# PRIVATE HELPERS — DSR formula (paper Eq. 1-2)
# =============================================================================
def _unannualize_sharpe(sharpe_annualized: float, periods_per_year: float = SHARPE_PERIODS_YEAR) -> float:
    if sharpe_annualized is None or not np.isfinite(sharpe_annualized):
        return np.nan
    return float(sharpe_annualized / np.sqrt(periods_per_year))

def _expected_max_sharpe(var_sr: float, n_trials: float) -> float:
    """Eq. 1 — expected maximum Sharpe ratio under N independent trials, assuming null skill."""
    if n_trials <= 1 or var_sr <= 0:
        return 0.0
    z_n  = norm.ppf(1.0 - 1.0 / n_trials)
    z_ne = norm.ppf(1.0 - 1.0 / (n_trials * np.e))
    term = (1.0 - EULER_GAMMA) * z_n + EULER_GAMMA * z_ne
    return float(np.sqrt(var_sr) * term)


def _deflated_sharpe_ratio(sr: float, sr0: float, t_obs: int, skew_r: float, kurt_r: float) -> float:
    """Eq. 2. sr and sr0 must both be UNANNUALIZED. kurt_r is raw kurtosis (fisher=False)."""
    if t_obs <= 1 or not np.isfinite(sr):
        return 0.0
    moment_term = 1.0 - skew_r * sr + ((kurt_r - 1.0) / 4.0) * (sr ** 2)
    if moment_term <= 0:
        return 0.0
    numerator = (sr - sr0) * np.sqrt(t_obs - 1)
    return float(norm.cdf(numerator / np.sqrt(moment_term)))

# =============================================================================
# APPROVAL CRITERION
# =============================================================================
def _evaluate_dsr_approval(dsr_value: float, dsr_th: float) -> bool:
    return dsr_value >= dsr_th

# =============================================================================
# CORE DSR CALCULATION (across a set of candidate trials — typically one timeframe)
# =============================================================================
def _compute_dsr(all_raw_results: list, dsr_th: float, n_combos: int) -> dict:

    total_candidates = len(all_raw_results)
    n_bruto           = total_candidates * max(n_combos, 1)

    n_eff = estimate_n_eff_flat(all_raw_results)

    n_bruto_str    = f"{n_bruto:,}".replace(",", ".")
    m_str          = f"{total_candidates:,}".replace(",", ".")
    n_eff_str      = f"{n_eff:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".") if n_eff is not None else "n/a (insufficient data)"

    logger.info(
        f"DSR ── N_bruto={n_bruto_str} (M={m_str} x n_combos={n_combos})  "
        f"N_eff={n_eff_str}"
    )

    raw_by_id = {r["rule_id"]: r for r in all_raw_results}

    if n_eff is None:
        logger.debug("DSR ── N_eff unavailable — setting DSR=0.0 for all rules (no rules pass).")
        dsr_by_id = {rule_id: 0.0 for rule_id in raw_by_id}
        return {
            "passed_dsr_ids": [],
            "dsr_by_rule_id": dsr_by_id,
            "n_eff":          None,
            "n_bruto":        n_bruto,
            "sr0":            np.nan,
        }

    sr_by_id = {
        rule_id: _unannualize_sharpe(r.get("sharpe_train", np.nan))
        for rule_id, r in raw_by_id.items()
    }
    sr_array = np.array(list(sr_by_id.values()), dtype=np.float64)
    sr_array = sr_array[np.isfinite(sr_array)]
    var_sr   = float(np.var(sr_array, ddof=1)) if sr_array.size > 1 else 0.0

    sr0 = _expected_max_sharpe(var_sr, n_eff)

    logger.debug(
        f"DSR ── SR0 terms ── total_candidates={total_candidates} n_combos={n_combos} "
        f"n_eff={n_eff:.4f} n_sr={sr_array.size} var_sr={var_sr:.6f} -> SR0={sr0:.4f}"
    )

    dsr_by_id = {}
    for rule_id, r in raw_by_id.items():
        t_days = int(r.get("n_days_train", 0))
        skew_r = float(r.get("skew_train", np.nan))
        kurt_r = float(r.get("kurtosis_train", np.nan))

        if not (np.isfinite(skew_r) and np.isfinite(kurt_r)):
            dsr_by_id[rule_id] = 0.0
            continue

        dsr_by_id[rule_id] = _deflated_sharpe_ratio(sr_by_id[rule_id], sr0, t_days, skew_r, kurt_r)

    passed_dsr_ids = [rid for rid, dsr_val in dsr_by_id.items() if _evaluate_dsr_approval(dsr_val, dsr_th)]

    if logger.isEnabledFor(logging.DEBUG):
        print_dsr_train_metrics(raw_by_id, dsr_by_id, sr_by_id, set(passed_dsr_ids), set(passed_dsr_ids), sr0)

    logger.debug(
        f"DSR ── M={total_candidates} n_combos={n_combos} N_bruto={n_bruto} N_eff={n_eff:.4f} SR0={sr0:.3f} "
        f"-> {len(passed_dsr_ids)}/{total_candidates} significant at th={dsr_th}"
    )

    return {
        "passed_dsr_ids": passed_dsr_ids,
        "dsr_by_rule_id": dsr_by_id,
        "n_eff":          n_eff,
        "n_bruto":        n_bruto,
        "sr0":            sr0,
    }

# =============================================================================
# PIPE DSR — one timeframe at a time
# =============================================================================
def _empty_dsr_fields() -> dict:
    """Placeholder DSR fields for rules that were never evaluated (pipe disabled)."""
    return {
        "passed_dsr":         True,
        "dsr":                0.0,
        "sharpe_train":       None,
        "skew_train":         None,
        "kurtosis_train":     None,
        "n_days_train":       None,
        "net_gain_train":     None,
        "max_dd_train":       None,
        "combo_daily_profit": None,
        "best_combo_id":      None,
    }

def pipe_dsr(
    rules: list,
    ohlcv_arr: dict,
    param_grid: dict,
    order_amount: int,
    dtype,
    dsr_th: float,
    enabled: bool = True,
    timeframe: str = "",
) -> list:

    start = time.time()

    if not enabled:
        logger.info(f"DSR ── {timeframe} ── disabled — passing all {len(rules)} rules through untouched")
        return [{**r, **_empty_dsr_fields()} for r in rules]

    n_combos = 1
    for _values in param_grid.values():
        n_combos *= len(_values)

    _check_m_vs_t_ratio(ohlcv_arr, n_rules=len(rules), n_combos=n_combos, timeframe=timeframe)

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

    raw_for_dsr = [
        {**r, **full_period_by_rule[r["rule_id"]]}
        for r in rules
    ]
    dsr_result     = _compute_dsr(raw_for_dsr, dsr_th=dsr_th, n_combos=n_combos)
    passed_dsr_ids = set(dsr_result["passed_dsr_ids"])
    dsr_by_id      = dsr_result["dsr_by_rule_id"]

    logger.info(f"DSR ── {timeframe} ── {len(passed_dsr_ids)}/{len(rules)} rules pass")
 
    results = []
    for r in rules:
        rid    = r["rule_id"]
        fp     = full_period_by_rule[rid]
        passed = rid in passed_dsr_ids
        results.append({
            **r,
            "passed_dsr":         passed,
            "dsr":                dsr_by_id.get(rid, 0.0),
            "sharpe_train":       fp["sharpe_train"],
            "skew_train":         fp["skew_train"],
            "kurtosis_train":     fp["kurtosis_train"],
            "n_days_train":       fp["n_days_train"],
            "net_gain_train":     fp["net_gain_train"],
            "max_dd_train":       fp["max_dd_train"],
            "combo_daily_profit": fp["combo_daily_profit"] if passed else None,
            "best_combo_id":      fp["best_combo_id"] if passed else None,
        })

    elapsed = int(time.time() - start)
    logger.info(f"DSR ── {timeframe} ── elapsed {elapsed // 3600} h {(elapsed % 3600) // 60} min {elapsed % 60} s")

    return results

def _check_m_vs_t_ratio(ohlcv_arr: dict, n_rules: int, n_combos: int, timeframe: str) -> None:

    any_sym  = next(iter(ohlcv_arr.values()))
    ts_range = pd.to_datetime(any_sym["ts"])
    t_days   = max((ts_range.max() - ts_range.min()).days, 1)
    m_bruto  = n_rules * max(n_combos, 1)

    if m_bruto > M_TO_T_WARN_RATIO * t_days:
        logger.debug(
            f"DSR ── {timeframe} ── M/T check ⚠️ ── M_bruto={m_bruto} (rules={n_rules} x combos={n_combos}) "
            f"vs T={t_days} days (ratio={m_bruto / t_days:.2f}x, warn_th={M_TO_T_WARN_RATIO}x) "
        )
    else:
        logger.debug(
            f"DSR ── {timeframe} ── M/T check ✅ ── M_bruto={m_bruto} (rules={n_rules} x combos={n_combos}) "
            f"vs T={t_days} days (ratio={m_bruto / t_days:.2f}x, warn_th={M_TO_T_WARN_RATIO}x)"
        )