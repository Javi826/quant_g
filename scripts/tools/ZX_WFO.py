import numpy as np
import pandas as pd
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib 
from collections import Counter 


def walk_forward_optimization(ohlcv_arr, param_ranges,
                              length_train_set=2000, pct_train_set=0.8,
                              anchored=True, 
                              evaluate_fn=None,
                              n_jobs=-1):

    if evaluate_fn is None:
        raise ValueError("You must pass an evaluate_fn(params, base_arrays) function")

    keys              = list(param_ranges.keys())
    all_combinations  = list(itertools.product(*[param_ranges[k] for k in keys]))
    dict_combinations = [dict(zip(keys, comb)) for comb in all_combinations]

    length_test       = int(length_train_set / pct_train_set - length_train_set)
    best_params_list  = []
    best_criteria_list = []  # ✅ list to store best criterion per window
    window_idx        = 1

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
        # Prepare base arrays
        # -----------------------------------------------------------
        base_arrays = {}
        for sym, (t0, t1) in train_indices.items():
            arr_dict = ohlcv_arr[sym]
        
            base_arrays[sym] = {
                'ts': arr_dict['ts'][t0:t1],
                'open': arr_dict['open'][t0:t1],
                'high': arr_dict['high'][t0:t1],
                'low': arr_dict['low'][t0:t1],
                'close': arr_dict['close'][t0:t1],
                'volume_quote': arr_dict.get('volume_quote', arr_dict['close']*0)[t0:t1],
                'low_time': arr_dict['low_time'][t0:t1],
                'high_time': arr_dict['high_time'][t0:t1],
            }

        # -----------------------------------------------------------
        # Parallel evaluation
        # -----------------------------------------------------------
        print(f"\n🧠 Window {window_idx}: evaluating {len(dict_combinations)} combinations...\n")

        with tqdm_joblib(
            tqdm(desc=f"🔁 WFO Window {window_idx}", total=len(dict_combinations), dynamic_ncols=True)
        ) as progress:
            results = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_fn)(params, base_arrays) for params in dict_combinations
            )

        # -----------------------------------------------------------
        # Select the best result
        # -----------------------------------------------------------
        best_criterion, best_params = max(results, key=lambda x: x[0])
        best_params_list.append(best_params)
        best_criteria_list.append(best_criterion)  # ✅ store best criterion

        window_idx += 1
        if is_last_window:
            break

        if anchored:
            end += length_test
        else:
            start += length_train_set + length_test
            end = start + length_train_set
    
    # DataFrame de ejemplo
    df_params = pd.DataFrame(best_params_list)
    
    # Calculamos la "moda" de cada columna manualmente
    final_params = {}
    for col in df_params.columns:
        # Contamos ocurrencias de cada valor en la columna
        counts = Counter(df_params[col])
        # Tomamos el valor con más ocurrencias
        most_common_val, _ = counts.most_common(1)[0]
        
        # Convertimos a int si es numérico y no termina en "_MAX"
        if isinstance(most_common_val, (int, float)) and not str(col).endswith("_MAX"):
            final_params[col] = int(round(most_common_val))
        else:
            final_params[col] = most_common_val
    

    print(f"\n✅ WFO completed: {window_idx} windows processed (parallelized with {n_jobs} threads)\n")
    return final_params
