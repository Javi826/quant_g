#shared/shared_batchs/tools/wfo_MC.py
import logging
import contextlib

import numpy as np
import pandas as pd
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from shared_batchs.tools.wfo_ST import _AGGREGATORS
from shared_batchs.tools.optimize_MC import generate_paths_for_all_symbols_functional
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays

logger = logging.getLogger("BOT_batch.tools.wfo_MC")


def walk_forward_optimization_mc(
    ohlcv_data: dict,
    param_ranges: dict,
    length_train_set: int,
    pct_train_set: float,
    anchored: bool,
    train_evaluate_fn,
    test_evaluate_fn,
    n_paths: int,
    n_obs: int,
    param_selection_mode: str = "MODE",
    n_jobs: int = -1,
    show_progress: bool = False,
) -> tuple:
    """
    Walk-Forward Optimization with Monte Carlo evaluation on train windows
    and real-data evaluation on test windows.

    train_evaluate_fn(params, paths_per_symbol) -> (metric, params)
    test_evaluate_fn(params, base_arrays)       -> (metric, params)

    Returns:
        tuple: (final_params, df_results)
    """
    if train_evaluate_fn is None or test_evaluate_fn is None:
        raise ValueError("You must pass both train_evaluate_fn and test_evaluate_fn")

    if param_selection_mode not in _AGGREGATORS:
        raise ValueError(f"Unknown param_selection_mode: {param_selection_mode}")

    keys              = list(param_ranges.keys())
    all_combinations  = list(itertools.product(*[param_ranges[k] for k in keys]))
    dict_combinations = [dict(zip(keys, comb)) for comb in all_combinations]

    length_test        = int(length_train_set / pct_train_set - length_train_set)
    best_params_list   = []
    best_criteria_list = []
    window_idx         = 1

    train_start_dates  = []
    train_end_dates    = []
    test_start_dates   = []
    test_end_dates     = []

    start = 0
    end   = length_train_set

    ref_sym    = max(ohlcv_data.keys(), key=lambda k: len(ohlcv_data[k]))
    ref_index  = ohlcv_data[ref_sym].index
    max_length = len(ref_index)

    while start < max_length:
        remaining_data = max_length - start
        is_last_window = remaining_data < (length_train_set + length_test)

        train_indices = {}
        test_indices  = {}

        for sym, df in ohlcv_data.items():
            sym_length = len(df)
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
        # Slice DataFrames for train (MC source) and test (real)
        # -----------------------------------------------------------
        ohlcv_train_df = {sym: ohlcv_data[sym].iloc[t0_sym:t1_sym] for sym, (t0_sym, t1_sym) in train_indices.items()}
        ohlcv_test_df  = {sym: ohlcv_data[sym].iloc[t0_sym:t1_sym] for sym, (t0_sym, t1_sym) in test_indices.items()}

        # -----------------------------------------------------------
        # Generate MC paths once per window, evaluate grid in parallel
        # -----------------------------------------------------------
        paths_per_symbol = generate_paths_for_all_symbols_functional(
            ohlcv_train_df, n_paths=n_paths, n_obs=n_obs, raw_columns=[],
        )

        with (tqdm_joblib(
            tqdm(desc=f"🔁 WFO-MC Window {window_idx}", total=len(dict_combinations), dynamic_ncols=True)
        ) if show_progress else contextlib.nullcontext()):
            results = Parallel(n_jobs=n_jobs)(
                delayed(train_evaluate_fn)(params, paths_per_symbol) for params in dict_combinations
            )

        # -----------------------------------------------------------
        # Select the best result (on MC train) and validate it on real test
        # -----------------------------------------------------------
        _, best_params = max(results, key=lambda x: x[0])

        base_arrays_test = prepare_ohlcv_arrays(ohlcv_test_df) if ohlcv_test_df else {}
        if base_arrays_test:
            test_criterion, _ = test_evaluate_fn(best_params, base_arrays_test)
        else:
            test_criterion = np.nan

        best_params_list.append(best_params)
        best_criteria_list.append(test_criterion)

        train_start_dates.append(ref_index[t0] if t0 < len(ref_index) else None)
        train_end_dates.append(ref_index[t1 - 1] if t1 - 1 < len(ref_index) else None)
        test_start_dates.append(ref_index[test0] if test0 < len(ref_index) else None)
        test_end_dates.append(ref_index[test1 - 1] if test1 - 1 < len(ref_index) else None)

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

    summary_row = dict(final_params)
    summary_row['train_start']    = param_selection_mode
    summary_row['train_end']      = ''
    summary_row['test_start']     = ''
    summary_row['test_end']       = ''
    summary_row['best_criterion'] = df_results['best_criterion'].mean() if 'best_criterion' in df_results else None

    df_results = pd.concat([df_results, pd.DataFrame([summary_row])], ignore_index=True)

    logger.info(f"WFO-MC completed: {window_idx} windows processed (parallelized with {n_jobs} threads)")
    logger.info(f"WFO-MC Final summary — parameters, criterion, and train/test dates per window:\n{df_results}")

    return final_params, df_results