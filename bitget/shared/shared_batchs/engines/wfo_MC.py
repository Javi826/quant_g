#shared/shared_batchs/engines/wfo_MC.py
import logging
import contextlib

import numpy as np
import pandas as pd
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from shared_batchs.engines.wfo_WF import _AGGREGATORS, _find_window_indices, _select_window_symbols, WARMUP_BARS
from shared_batchs.engines.optimize_MC import generate_paths_for_all_symbols_functional
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_config import VOLUME_COL
from shared_batchs.backtesters.ZX_compute_BT import INITIAL_BALANCE
from shared_batch_regime.regime_core import apply_regime_filter
from shared_batchs.regime import regime_module

logger = logging.getLogger("BOT_batch.engines.wfo_MC")


# =============================================================================
# TRAIN EVALUATION — synthetic MC paths, no regime filter
# =============================================================================

def _evaluate_train_mc(
    params: dict,
    paths_per_symbol: dict,
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    dtype,
) -> tuple:
    """Evaluate one param combination across all MC paths of the train window."""
    from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest

    n_paths = next(iter(paths_per_symbol.values())).shape[0] if paths_per_symbol else 0
    metrics = []

    for path_idx in range(n_paths):
        ohlcv_arrays = _extract_ohlcv_from_path(paths_per_symbol, path_idx, dtype=dtype)
        for sym, arr in ohlcv_arrays.items():
            sig_kwargs      = {k: params[k.upper()] for k in signal_params_keys if k.upper() in params}
            signals         = signal_fn(arr, **sig_kwargs, live_trading=False)
            ohlcv_arrays[sym] = {**arr, "signal": np.asarray(signals, dtype=dtype)}

        results  = run_grid_backtest(
            ohlcv_arrays,
            sell_after   = params["SELL_AFTER"],
            tp_pct       = params["TP_PCT"],
            sl_pct       = params["SL_PCT"],
            order_amount = order_amount,
        )
        port     = results.get("__PORTFOLIO__", {})
        trades   = port.get("trades", [])
        net_gain = float(np.sum(trades)) if trades else 0.0
        metrics.append((net_gain / INITIAL_BALANCE) * 100.0)

    avg_metric = float(np.mean(metrics)) if metrics else -np.inf
    return avg_metric, params


def _extract_ohlcv_from_path(paths_per_symbol: dict, path_idx: int, dtype) -> dict:
    """Extract one MC path (all symbols) as OHLCV arrays for backtesting."""
    from shared_batchs.utils.ohlcv_utils import extract_ohlcv_from_path
    return extract_ohlcv_from_path(paths_per_symbol, path_idx, dtype=dtype)


# =============================================================================
# TEST EVALUATION — real data, optional regime filtering
# =============================================================================

def _collect_test_trades_mc(
    params: dict,
    base_arrays_test: dict,
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    dtype,
    bins_to_filter,
    regime_enabled: bool,
    indicator_cache: dict,
) -> pd.DataFrame:
    """Run backtest with best_params on real test data, with optional regime filtering."""
    from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest

    _bins = [bins_to_filter] if isinstance(bins_to_filter, str) else list(bins_to_filter) if bins_to_filter else []

    ohlcv_arrays = {}
    for sym, arr in base_arrays_test.items():
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
    trades             = results["__PORTFOLIO__"]["trade_log"].copy()
    if not trades.empty:
        trades.columns     = trades.columns.str.lower().str.strip()
        trades["buy_time"] = pd.to_datetime(trades["buy_time"])
    return trades


# =============================================================================
# WALK FORWARD OPTIMIZATION — MONTE CARLO
# =============================================================================

def walk_forward_optimization_mc(
    ohlcv_arr: dict,
    param_ranges: dict,
    length_train_set: int,
    pct_train_set: float,
    anchored: bool,
    signal_fn: callable,
    signal_params_keys: list,
    order_amount: int,
    dtype,
    n_paths: int,
    n_obs: int,
    param_selection_mode: str = "MODE",
    n_jobs: int = -1,
    show_progress: bool = False,
    n_symbols: int | None = None,
    bins_to_filter=None,
    regime_enabled: bool = False,
    indicator_cache: dict | None = None,
) -> tuple:

    if param_selection_mode not in _AGGREGATORS:
        raise ValueError(f"Unknown param_selection_mode: {param_selection_mode}")

    keys              = list(param_ranges.keys())
    all_combinations  = list(itertools.product(*[param_ranges[k] for k in keys]))
    dict_combinations = [dict(zip(keys, comb)) for comb in all_combinations]

    length_test         = int(length_train_set / pct_train_set - length_train_set)
    best_params_list    = []
    best_criteria_list  = []
    window_idx          = 1
    last_test_end_ref   = length_train_set + length_test

    train_start_dates   = []
    train_end_dates     = []
    test_start_dates    = []
    test_end_dates      = []
    train_symbols_list  = []
    test_symbols_list   = []

    train_trades_list   = []
    test_trades_list    = []
    test_n_trades_list  = []

    start = 0
    end   = length_train_set

    ref_sym    = max(ohlcv_arr.keys(), key=lambda k: len(ohlcv_arr[k]["ts"]))
    ref_ts     = ohlcv_arr[ref_sym]["ts"]
    max_length = len(ref_ts)

    while start < max_length:
        remaining_data = max_length - (end if anchored else start)
        is_last_window = remaining_data < (length_train_set + length_test)

        # -----------------------------------------------------------------
        # Define window date boundaries from ref_sym
        # -----------------------------------------------------------------
        if is_last_window:
            remaining_from_last = max_length - last_test_end_ref
            if remaining_from_last < int(length_test * 0.5):
                break
            t0_ref    = 0 if anchored else start
            t1_ref    = last_test_end_ref
            test0_ref = t1_ref
            test1_ref = max_length
        else:
            t0_ref    = 0 if anchored else start
            t1_ref    = end if anchored else start + length_train_set
            test0_ref = t1_ref
            test1_ref = min(t1_ref + length_test, max_length)
            last_test_end_ref = test1_ref

        train_start_ts = ref_ts[t0_ref]
        test_start_ts  = ref_ts[t1_ref] if t1_ref < max_length else ref_ts[-1]
        test_end_ts    = ref_ts[test1_ref - 1]

        # -----------------------------------------------------------------
        # Collect all valid candidates using date-aligned indices
        # -----------------------------------------------------------------
        candidate_indices = {}
        for sym, arr_dict in ohlcv_arr.items():
            indices = _find_window_indices(arr_dict["ts"], train_start_ts, test_start_ts, test_end_ts)
            if indices is not None:
                candidate_indices[sym] = indices

        # -----------------------------------------------------------------
        # Select exactly n_symbols by train-window volume (same as WF)
        # -----------------------------------------------------------------
        selected_indices = _select_window_symbols(candidate_indices, ohlcv_arr, n_symbols)

        train_indices = {sym: (t0, t1)       for sym, (t0, t1, _,     _    ) in selected_indices.items()}
        test_indices  = {sym: (test0, test1) for sym, (_,  _,  test0, test1) in selected_indices.items()}

        train_symbols_list.append(sorted(train_indices.keys()))
        test_symbols_list.append(sorted(test_indices.keys()))

        if not train_indices:
            break

        if ref_sym in selected_indices:
            t0, t1, test0, test1 = selected_indices[ref_sym]
        else:
            t0, t1, test0, test1 = t0_ref, t1_ref, test0_ref, test1_ref

        # -----------------------------------------------------------
        # Slice real OHLCV for train window (source for MC paths) and test window
        # -----------------------------------------------------------
        base_arrays = {}
        for sym, (t0_sym, t1_sym) in train_indices.items():
            arr_dict   = ohlcv_arr[sym]
            warm_start = max(0, t0_sym - WARMUP_BARS)
            base_arrays[sym] = {
                "ts":        arr_dict["ts"][warm_start:t1_sym],
                "open":      arr_dict["open"][warm_start:t1_sym],
                "high":      arr_dict["high"][warm_start:t1_sym],
                "low":       arr_dict["low"][warm_start:t1_sym],
                "close":     arr_dict["close"][warm_start:t1_sym],
                VOLUME_COL:  arr_dict.get(VOLUME_COL, arr_dict["close"] * 0)[warm_start:t1_sym],
                "low_time":  arr_dict["low_time"][warm_start:t1_sym],
                "high_time": arr_dict["high_time"][warm_start:t1_sym],
            }

        base_arrays_test = {}
        for sym, (t0_sym, t1_sym) in test_indices.items():
            arr_dict   = ohlcv_arr[sym]
            warm_start = max(0, t0_sym - WARMUP_BARS)
            base_arrays_test[sym] = {
                "ts":        arr_dict["ts"][warm_start:t1_sym],
                "open":      arr_dict["open"][warm_start:t1_sym],
                "high":      arr_dict["high"][warm_start:t1_sym],
                "low":       arr_dict["low"][warm_start:t1_sym],
                "close":     arr_dict["close"][warm_start:t1_sym],
                VOLUME_COL:  arr_dict.get(VOLUME_COL, arr_dict["close"] * 0)[warm_start:t1_sym],
                "low_time":  arr_dict["low_time"][warm_start:t1_sym],
                "high_time": arr_dict["high_time"][warm_start:t1_sym],
            }

        # -----------------------------------------------------------
        # Generate MC paths from the real train window, evaluate grid
        # -----------------------------------------------------------
        ohlcv_train_df = {
            sym: pd.DataFrame({
                "open":      arr["open"],
                "high":      arr["high"],
                "low":       arr["low"],
                "close":     arr["close"],
                VOLUME_COL:  arr[VOLUME_COL],
                "low_time":  arr["low_time"],
                "high_time": arr["high_time"],
            }, index=pd.DatetimeIndex(arr["ts"], name="ts"))
            for sym, arr in base_arrays.items()
        }
        paths_per_symbol = generate_paths_for_all_symbols_functional(
            ohlcv_train_df, n_paths=n_paths, n_obs=n_obs, raw_columns=[],
        )

        with (tqdm_joblib(
            tqdm(desc=f"🔁 WFO-MC Window {window_idx}", total=len(dict_combinations), dynamic_ncols=True)
        ) if show_progress else contextlib.nullcontext()):
            results = Parallel(n_jobs=n_jobs)(
                delayed(_evaluate_train_mc)(
                    params, paths_per_symbol, signal_fn, signal_params_keys, order_amount, dtype
                )
                for params in dict_combinations
            )

        # -----------------------------------------------------------
        # Select best result (on MC train), validate on real test with regime filter
        # -----------------------------------------------------------
        _, best_params = max(results, key=lambda x: x[0])
        best_params_list.append(best_params)

        window_test_n_trades = 0
        test_criterion        = np.nan

        # NOTE: train trades are not computed for WFO-MC (train uses synthetic
        # MC paths, not real data). wfo_train_trades is returned empty below,
        # kept only for return-signature compatibility with run_wfo_is.

        if base_arrays_test:
            df_test = _collect_test_trades_mc(
                params              = best_params,
                base_arrays_test    = base_arrays_test,
                signal_fn           = signal_fn,
                signal_params_keys  = signal_params_keys,
                order_amount        = order_amount,
                dtype               = dtype,
                bins_to_filter      = bins_to_filter,
                regime_enabled      = regime_enabled,
                indicator_cache     = indicator_cache,
            )
            if df_test is not None and not df_test.empty:
                df_test = df_test[df_test["buy_time"] >= pd.Timestamp(test_start_ts)].copy()
                df_test["wfo_window"] = window_idx
                test_trades_list.append(df_test)
                window_test_n_trades = len(df_test)
                test_criterion       = float(df_test["profit"].sum()) / INITIAL_BALANCE * 100

        best_criteria_list.append(test_criterion)
        test_n_trades_list.append(window_test_n_trades)

        train_start_dates.append(ref_ts[t0]     if t0     < len(ref_ts) else None)
        train_end_dates.append(ref_ts[t1 - 1]   if t1 - 1 < len(ref_ts) else None)
        test_start_dates.append(ref_ts[test0]   if test0  < len(ref_ts) else None)
        test_end_dates.append(ref_ts[test1 - 1] if test1 - 1 < len(ref_ts) else None)

        window_idx += 1
        if is_last_window:
            break

        if anchored:
            end += length_test
        else:
            start += length_test
            end = start + length_train_set

    # -----------------------------------------------------------
    # Final parameter summary (aggregated via param_selection_mode)
    # -----------------------------------------------------------
    df_params    = pd.DataFrame(best_params_list)
    final_params = _AGGREGATORS[param_selection_mode](df_params, param_ranges)

    # -----------------------------------------------------------
    # Final DataFrame with train/test dates, params, and criterion
    # -----------------------------------------------------------
    train_start_ts_raw = list(train_start_dates)
    test_start_ts_raw  = list(test_start_dates)
    test_end_ts_raw    = list(test_end_dates)

    train_start_dates = [pd.to_datetime(d).date() if d is not None else None for d in train_start_dates]
    train_end_dates   = [pd.to_datetime(d).date() if d is not None else None for d in train_end_dates]
    test_start_dates  = [pd.to_datetime(d).date() if d is not None else None for d in test_start_dates]
    test_end_dates    = [pd.to_datetime(d).date() if d is not None else None for d in test_end_dates]

    df_results = pd.DataFrame(best_params_list)
    df_results.insert(0, "train_start", train_start_dates)
    df_results.insert(1, "train_end",   train_end_dates)
    df_results.insert(2, "test_start",  test_start_dates)
    df_results.insert(3, "test_end",    test_end_dates)
    df_results["best_crite"]      = best_criteria_list
    df_results["tn_trades"]       = test_n_trades_list
    df_results["tr_symbols"]      = [len(s) for s in train_symbols_list]
    df_results["ts_symbols"]      = [len(s) for s in test_symbols_list]
    df_results["tr_syms"]         = train_symbols_list
    df_results["ts_syms"]         = test_symbols_list
    df_results["_train_start_ts"] = train_start_ts_raw
    df_results["_test_start_ts"]  = test_start_ts_raw
    df_results["_test_end_ts"]    = test_end_ts_raw

    summary_row = dict(final_params)
    summary_row["train_start"] = param_selection_mode
    summary_row["train_end"]   = ""
    summary_row["test_start"]  = ""
    summary_row["test_end"]    = ""
    summary_row["best_crite"]  = df_results["best_crite"].mean() if "best_crite" in df_results else None

    df_results = pd.concat([df_results, pd.DataFrame([summary_row])], ignore_index=True)

    sep_row = {col: "·" * min(8, len(str(col))) for col in df_results.columns}
    sep_row["train_start"] = "·" * 10
    df_display   = pd.concat([df_results.iloc[:-1], pd.DataFrame([sep_row]), df_results.iloc[[-1]]], ignore_index=True)
    display_cols = [c for c in df_display.columns if not c.startswith("_") and c not in ("tr_syms", "ts_syms")]
    logger.debug(f"WFO-MC Final summary — parameters, criterion, and train/test dates per window:\n{df_display[display_cols].to_string()}\n{'─'*115}")

    # -----------------------------------------------------------
    # Concatenate per-window trade logs
    # -----------------------------------------------------------
    wfo_train_trades = pd.concat(train_trades_list, ignore_index=True) if train_trades_list else pd.DataFrame()
    wfo_test_trades  = pd.concat(test_trades_list,  ignore_index=True) if test_trades_list  else pd.DataFrame()

    return final_params, df_results, wfo_train_trades, wfo_test_trades, window_idx