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


def walk_forward_optimization(ohlcv_arr, param_ranges,
                              length_train_set, pct_train_set,
                              anchored,
                              evaluate_fn,
                              param_selection_mode="MODE",
                              n_jobs=-1,
                              show_progress=False):

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

    # New lists ONLY for train/test dates
    train_start_dates  = []
    train_end_dates    = []
    test_start_dates   = []
    test_end_dates     = []

    start = 0
    end   = length_train_set

    ref_sym    = max(ohlcv_arr.keys(), key=lambda k: len(ohlcv_arr[k]['ts']))
    ref_ts     = ohlcv_arr[ref_sym]['ts']
    max_length = len(ref_ts)

    while start < max_length:
        remaining_data = max_length - start
        is_last_window = remaining_data < (length_train_set + length_test)

        train_indices = {}
        test_indices  = {}

        for sym, arr_dict in ohlcv_arr.items():
            sym_length = len(arr_dict['ts'])
            if start >= sym_length:
                continue

            if is_last_window:
                remaining  = sym_length - start
                train_size = int(remaining * pct_train_set)
                test_size  = remaining - train_size
                if train_size < (length_train_set * 0.8):
                    continue
                t0, t1 = start, start + train_size
                test0, test1 = t1, sym_length
            else:
                t0    = 0 if anchored else start
                t1    = min(end, sym_length) if anchored else min(start + length_train_set, sym_length)
                test0 = t1
                test1 = min(t1 + length_test, sym_length)

            if t1 > t0 and test1 > test0:
                train_indices[sym] = (t0, t1)
                test_indices[sym]  = (test0, test1)

        if not train_indices:
            break

        if ref_sym in train_indices:
            t0, t1 = train_indices[ref_sym]
            test0, test1 = test_indices[ref_sym]

        # -----------------------------------------------------------
        # Prepare base arrays (train + test)
        # -----------------------------------------------------------
        base_arrays = {}
        for sym, (t0_sym, t1_sym) in train_indices.items():
            arr_dict = ohlcv_arr[sym]

            base_arrays[sym] = {
                'ts': arr_dict['ts'][t0_sym:t1_sym],
                'open': arr_dict['open'][t0_sym:t1_sym],
                'high': arr_dict['high'][t0_sym:t1_sym],
                'low': arr_dict['low'][t0_sym:t1_sym],
                'close': arr_dict['close'][t0_sym:t1_sym],
                VOLUME_COL: arr_dict.get(VOLUME_COL, arr_dict['close']*0)[t0_sym:t1_sym],
                'low_time': arr_dict['low_time'][t0_sym:t1_sym],
                'high_time': arr_dict['high_time'][t0_sym:t1_sym],
            }

        base_arrays_test = {}
        for sym, (t0_sym, t1_sym) in test_indices.items():
            arr_dict = ohlcv_arr[sym]

            base_arrays_test[sym] = {
                'ts': arr_dict['ts'][t0_sym:t1_sym],
                'open': arr_dict['open'][t0_sym:t1_sym],
                'high': arr_dict['high'][t0_sym:t1_sym],
                'low': arr_dict['low'][t0_sym:t1_sym],
                'close': arr_dict['close'][t0_sym:t1_sym],
                VOLUME_COL: arr_dict.get(VOLUME_COL, arr_dict['close']*0)[t0_sym:t1_sym],
                'low_time': arr_dict['low_time'][t0_sym:t1_sym],
                'high_time': arr_dict['high_time'][t0_sym:t1_sym],
            }

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
        # Select the best result (on train) and validate it on test
        # -----------------------------------------------------------
        _, best_params = max(results, key=lambda x: x[0])

        if base_arrays_test:
            test_criterion, _ = evaluate_fn(best_params, base_arrays_test)
        else:
            test_criterion = np.nan

        best_params_list.append(best_params)
        best_criteria_list.append(test_criterion)

        # Store train/test dates
        train_start_dates.append(ref_ts[t0] if t0 < len(ref_ts) else None)
        train_end_dates.append(ref_ts[t1 - 1] if t1 - 1 < len(ref_ts) else None)
        test_start_dates.append(ref_ts[test0] if test0 < len(ref_ts) else None)
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
    train_start_dates = [pd.to_datetime(d).date() if d is not None else None for d in train_start_dates]
    train_end_dates   = [pd.to_datetime(d).date() if d is not None else None for d in train_end_dates]
    test_start_dates  = [pd.to_datetime(d).date() if d is not None else None for d in test_start_dates]
    test_end_dates    = [pd.to_datetime(d).date() if d is not None else None for d in test_end_dates]

    df_results = pd.DataFrame(best_params_list)
    df_results.insert(0, 'train_start', train_start_dates)
    df_results.insert(1, 'train_end', train_end_dates)
    df_results.insert(2, 'test_start', test_start_dates)
    df_results.insert(3, 'test_end', test_end_dates)
    df_results['best_criterion'] = best_criteria_list

    # -----------------------------------------------------------
    # Add a summary row with the aggregated final_params
    # -----------------------------------------------------------
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