#shared/shared_batchs/tools/wfo_ST.py
import logging
import contextlib
import numpy as np
import pandas as pd
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from collections import Counter
from shared_config import VOLUME_COL

logger = logging.getLogger("BOT_batch.tools.wfo_ST")

EMA_ALPHA = 0.3


# =============================================================================
# PARAM AGGREGATION HELPERS
# =============================================================================

def _decimals_for_values(values) -> int:
    """Max number of decimal places present in a list of numeric param values."""
    max_decimals = 0
    for v in values:
        s = f"{float(v):.10f}".rstrip("0")
        if "." in s:
            max_decimals = max(max_decimals, len(s.split(".")[1]))
    return max_decimals


def _round_param(value, decimals: int):
    """Round a value to match the precision of its original param grid."""
    return int(round(value)) if decimals == 0 else round(float(value), decimals)


def _aggregate_mode(df_params: pd.DataFrame, param_ranges: dict) -> dict:
    final_params = {}
    for col in df_params.columns:
        most_common_val, _ = Counter(df_params[col]).most_common(1)[0]
        decimals = _decimals_for_values(param_ranges[col])
        final_params[col] = _round_param(most_common_val, decimals)
    return final_params


def _aggregate_mean(df_params: pd.DataFrame, param_ranges: dict) -> dict:
    final_params = {}
    for col in df_params.columns:
        decimals = _decimals_for_values(param_ranges[col])
        final_params[col] = _round_param(df_params[col].mean(), decimals)
    return final_params


def _aggregate_ema(df_params: pd.DataFrame, param_ranges: dict) -> dict:
    final_params = {}
    for col in df_params.columns:
        decimals = _decimals_for_values(param_ranges[col])
        ema_val  = df_params[col].ewm(alpha=EMA_ALPHA).mean().iloc[-1]
        final_params[col] = _round_param(ema_val, decimals)
    return final_params


_AGGREGATORS = {
    "MODE": _aggregate_mode,
    "MEAN": _aggregate_mean,
    "EMA":  _aggregate_ema,
}


# =============================================================================
# WINDOW SYMBOL SELECTION
# =============================================================================

def _find_window_indices(
    sym_ts: np.ndarray,
    train_start_ts,
    test_start_ts,
    test_end_ts,
) -> tuple | None:
    """
    Find date-aligned train/test indices for a symbol using timestamp search.
    Returns (t0, t1, test0, test1) or None if insufficient data in either period.
    """
    t0    = int(np.searchsorted(sym_ts, train_start_ts, side="left"))
    t1    = int(np.searchsorted(sym_ts, test_start_ts,  side="left"))
    test0 = t1
    test1 = int(np.searchsorted(sym_ts, test_end_ts,    side="right"))

    if t1 <= t0 or test1 <= test0:
        return None
    return t0, t1, test0, test1


def _select_window_symbols(
    candidate_indices: dict,
    ohlcv_arr: dict,
    n_symbols: int | None,
) -> dict:
    """
    Select exactly n_symbols for a WFO window, ranked by avg train-window volume (descending).
    If n_symbols is None, returns all candidates (backward-compatible).
    """
    if n_symbols is None:
        return candidate_indices

    def _avg_train_vol(sym: str) -> float:
        t0, t1, _, _ = candidate_indices[sym]
        arr = ohlcv_arr[sym]
        vol = arr.get(VOLUME_COL, arr["close"] * 0)
        return float(np.mean(vol[t0:t1])) if t1 > t0 else 0.0

    selected_syms = sorted(candidate_indices, key=_avg_train_vol, reverse=True)[:n_symbols]
    return {sym: candidate_indices[sym] for sym in selected_syms}


# =============================================================================
# WALK FORWARD OPTIMIZATION
# =============================================================================

def walk_forward_optimization(ohlcv_arr, param_ranges,
                              length_train_set, pct_train_set,
                              anchored,
                              evaluate_fn,
                              param_selection_mode="MODE",
                              n_jobs=-1,
                              show_progress=False,
                              n_symbols=None,
                              test_evaluate_fn=None):

    if evaluate_fn is None:
        raise ValueError("You must pass an evaluate_fn(params, base_arrays) function")

    if param_selection_mode not in _AGGREGATORS:
        raise ValueError(f"Unknown param_selection_mode: {param_selection_mode}")

    keys               = list(param_ranges.keys())
    all_combinations   = list(itertools.product(*[param_ranges[k] for k in keys]))
    dict_combinations  = [dict(zip(keys, comb)) for comb in all_combinations]

    length_test        = int(length_train_set / pct_train_set - length_train_set)
    best_params_list   = []
    best_criteria_list = []
    window_idx         = 1

    train_start_dates  = []
    train_end_dates    = []
    test_start_dates   = []
    test_end_dates     = []
    train_symbols_list = []
    test_symbols_list  = []

    start = 0
    end   = length_train_set

    ref_sym    = max(ohlcv_arr.keys(), key=lambda k: len(ohlcv_arr[k]['ts']))
    ref_ts     = ohlcv_arr[ref_sym]['ts']
    max_length = len(ref_ts)

    while start < max_length:
        remaining_data = max_length - start
        is_last_window = remaining_data < (length_train_set + length_test)

        # -----------------------------------------------------------------
        # Define window date boundaries from ref_sym
        # -----------------------------------------------------------------
        if is_last_window:
            train_size_ref = int(remaining_data * pct_train_set)
            if train_size_ref < int(length_train_set * 0.8):
                break
            t0_ref, t1_ref       = start, start + train_size_ref
            test0_ref, test1_ref = t1_ref, max_length
        else:
            t0_ref    = 0 if anchored else start
            t1_ref    = end if anchored else start + length_train_set
            test0_ref = t1_ref
            test1_ref = min(t1_ref + length_test, max_length)

        train_start_ts = ref_ts[t0_ref]
        test_start_ts  = ref_ts[t1_ref] if t1_ref < max_length else ref_ts[-1]
        test_end_ts    = ref_ts[test1_ref - 1]

        # -----------------------------------------------------------------
        # Collect all valid candidates using date-aligned indices
        # -----------------------------------------------------------------
        candidate_indices = {}
        for sym, arr_dict in ohlcv_arr.items():
            indices = _find_window_indices(
                arr_dict["ts"], train_start_ts, test_start_ts, test_end_ts
            )
            if indices is not None:
                candidate_indices[sym] = indices

        # -----------------------------------------------------------------
        # Select exactly n_symbols (OOS1 priority + fill by volume)
        # -----------------------------------------------------------------
        selected_indices = _select_window_symbols(
            candidate_indices, ohlcv_arr, n_symbols
        )

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
        # Prepare base arrays (train + test)
        # -----------------------------------------------------------
        base_arrays = {}
        for sym, (t0_sym, t1_sym) in train_indices.items():
            arr_dict = ohlcv_arr[sym]
            base_arrays[sym] = {
                'ts':        arr_dict['ts'][t0_sym:t1_sym],
                'open':      arr_dict['open'][t0_sym:t1_sym],
                'high':      arr_dict['high'][t0_sym:t1_sym],
                'low':       arr_dict['low'][t0_sym:t1_sym],
                'close':     arr_dict['close'][t0_sym:t1_sym],
                VOLUME_COL:  arr_dict.get(VOLUME_COL, arr_dict['close'] * 0)[t0_sym:t1_sym],
                'low_time':  arr_dict['low_time'][t0_sym:t1_sym],
                'high_time': arr_dict['high_time'][t0_sym:t1_sym],
            }

        base_arrays_test = {}
        for sym, (t0_sym, t1_sym) in test_indices.items():
            arr_dict = ohlcv_arr[sym]
            base_arrays_test[sym] = {
                'ts':        arr_dict['ts'][t0_sym:t1_sym],
                'open':      arr_dict['open'][t0_sym:t1_sym],
                'high':      arr_dict['high'][t0_sym:t1_sym],
                'low':       arr_dict['low'][t0_sym:t1_sym],
                'close':     arr_dict['close'][t0_sym:t1_sym],
                VOLUME_COL:  arr_dict.get(VOLUME_COL, arr_dict['close'] * 0)[t0_sym:t1_sym],
                'low_time':  arr_dict['low_time'][t0_sym:t1_sym],
                'high_time': arr_dict['high_time'][t0_sym:t1_sym],
            }

        if window_idx == 1:
            for sym, arr in base_arrays_test.items():
                logger.debug(f"[DEBUG WFO test] {sym}: {len(arr['ts'])} bars | {arr['ts'][0]} → {arr['ts'][-1]}")
                break

        # -----------------------------------------------------------
        # Parallel evaluation
        # -----------------------------------------------------------
        with (tqdm_joblib(
            tqdm(desc=f"🔁 WFO Window {window_idx}", total=len(dict_combinations), dynamic_ncols=True)
        ) if show_progress else contextlib.nullcontext()):
            results = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_fn)(params, base_arrays) for params in dict_combinations
            )

        # -----------------------------------------------------------
        # Select best result (on train) and validate on test
        # -----------------------------------------------------------
        _, best_params = max(results, key=lambda x: x[0])

        _test_fn = test_evaluate_fn if test_evaluate_fn is not None else evaluate_fn
        if base_arrays_test:
            test_criterion, _ = _test_fn(best_params, base_arrays_test)
        else:
            test_criterion = np.nan

        best_params_list.append(best_params)
        best_criteria_list.append(test_criterion)

        train_start_dates.append(ref_ts[t0]       if t0       < len(ref_ts) else None)
        train_end_dates.append(ref_ts[t1 - 1]     if t1 - 1   < len(ref_ts) else None)
        test_start_dates.append(ref_ts[test0]     if test0    < len(ref_ts) else None)
        test_end_dates.append(ref_ts[test1 - 1]   if test1 - 1 < len(ref_ts) else None)

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
    train_start_dates = [pd.to_datetime(d).date() if d is not None else None for d in train_start_dates]
    train_end_dates   = [pd.to_datetime(d).date() if d is not None else None for d in train_end_dates]
    test_start_dates  = [pd.to_datetime(d).date() if d is not None else None for d in test_start_dates]
    test_end_dates    = [pd.to_datetime(d).date() if d is not None else None for d in test_end_dates]

    df_results = pd.DataFrame(best_params_list)
    df_results.insert(0, 'train_start', train_start_dates)
    df_results.insert(1, 'train_end',   train_end_dates)
    df_results.insert(2, 'test_start',  test_start_dates)
    df_results.insert(3, 'test_end',    test_end_dates)
    df_results['best_criterion'] = best_criteria_list
    df_results['train_symbols']  = [len(s) for s in train_symbols_list]
    df_results['test_symbols']   = [len(s) for s in test_symbols_list]

    summary_row = dict(final_params)
    summary_row['train_start']    = param_selection_mode
    summary_row['train_end']      = ''
    summary_row['test_start']     = ''
    summary_row['test_end']       = ''
    summary_row['best_criterion'] = df_results['best_criterion'].mean() if 'best_criterion' in df_results else None

    df_results = pd.concat([df_results, pd.DataFrame([summary_row])], ignore_index=True)

    logger.info(f"WFO completed: {window_idx} windows processed (parallelized with {n_jobs} threads)")
    logger.info(f"WFO Final summary — parameters, criterion, and train/test dates per window:\n{df_results}")

    return final_params, df_results