import numpy as np
import pandas as pd
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib 
from collections import Counter 

# -----------------------------------------------------------------------------
# ADAPTADOR PARA WFO: necesitamos pasar ambos timeframes
# -----------------------------------------------------------------------------
def walk_forward_optimization_tf(
    ohlcv_arr_minor, ohlcv_arr_major, param_ranges,
    length_train_set, pct_train_set, anchored, 
    evaluate_fn, n_jobs=-1
):

    
    if evaluate_fn is None:
        raise ValueError("You must pass an evaluate_fn(params, base_arrays) function")

    keys = list(param_ranges.keys())
    all_combinations = list(itertools.product(*[param_ranges[k] for k in keys]))
    dict_combinations = [dict(zip(keys, comb)) for comb in all_combinations]

    length_test = int(length_train_set / pct_train_set - length_train_set)
    best_params_list = []
    best_criteria_list = []
    window_idx = 1

    train_start_dates = []
    train_end_dates = []
    test_start_dates = []
    test_end_dates = []

    start = 0
    end = length_train_set

    # Usamos el timeframe menor como referencia
    ref_sym = max(ohlcv_arr_minor.keys(), key=lambda k: len(ohlcv_arr_minor[k]['ts']))
    ref_ts = ohlcv_arr_minor[ref_sym]['ts']
    max_length = len(ref_ts)

    while start < max_length:
        remaining_data = max_length - start
        is_last_window = remaining_data < (length_train_set + length_test)

        train_indices_minor = {}
        test_indices_minor = {}
        train_indices_major = {}
        test_indices_major = {}

        # Calcular índices para timeframe menor
        for sym, arr_dict in ohlcv_arr_minor.items():
            sym_length = len(arr_dict['ts'])
            if start >= sym_length:
                continue

            if is_last_window:
                remaining = sym_length - start
                train_size = int(remaining * pct_train_set)
                test_size = remaining - train_size
                if train_size < (length_train_set * 0.8):
                    continue
                t0, t1 = start, start + train_size
                test0, test1 = t1, sym_length
            else:
                t0 = 0 if anchored else start
                t1 = min(end, sym_length) if anchored else min(start + length_train_set, sym_length)
                test0 = t1
                test1 = min(t1 + length_test, sym_length)

            if t1 > t0 and test1 > test0:
                train_indices_minor[sym] = (t0, t1)
                test_indices_minor[sym] = (test0, test1)

        # Calcular índices correspondientes para timeframe mayor
        for sym in train_indices_minor.keys():
            if sym not in ohlcv_arr_major:
                continue
                
            t0_minor, t1_minor = train_indices_minor[sym]
            ts_minor = ohlcv_arr_minor[sym]['ts']
            ts_major = ohlcv_arr_major[sym]['ts']
            
            # Convertir índices de tiempo menor a mayor
            t0_major = np.searchsorted(ts_major, ts_minor[t0_minor])
            t1_major = np.searchsorted(ts_major, ts_minor[t1_minor - 1]) + 1
            
            test0_minor, test1_minor = test_indices_minor[sym]
            test0_major = np.searchsorted(ts_major, ts_minor[test0_minor])
            test1_major = np.searchsorted(ts_major, ts_minor[test1_minor - 1]) + 1
            
            train_indices_major[sym] = (t0_major, min(t1_major, len(ts_major)))
            test_indices_major[sym] = (test0_major, min(test1_major, len(ts_major)))

        if not train_indices_minor:
            break

        # Preparar base_arrays para ambos timeframes
        base_arrays_minor = {}
        base_arrays_major = {}
        
        for sym, (t0_sym, t1_sym) in train_indices_minor.items():
            if sym not in train_indices_major:
                continue
                
            arr_dict_minor = ohlcv_arr_minor[sym]
            base_arrays_minor[sym] = {
                'ts': arr_dict_minor['ts'][t0_sym:t1_sym],
                'open': arr_dict_minor['open'][t0_sym:t1_sym],
                'high': arr_dict_minor['high'][t0_sym:t1_sym],
                'low': arr_dict_minor['low'][t0_sym:t1_sym],
                'close': arr_dict_minor['close'][t0_sym:t1_sym],
                'volume_quote': arr_dict_minor.get('volume_quote', arr_dict_minor['close']*0)[t0_sym:t1_sym],
                'low_time': arr_dict_minor['low_time'][t0_sym:t1_sym],
                'high_time': arr_dict_minor['high_time'][t0_sym:t1_sym],
            }
            
            t0_major, t1_major = train_indices_major[sym]
            arr_dict_major = ohlcv_arr_major[sym]
            base_arrays_major[sym] = {
                'ts': arr_dict_major['ts'][t0_major:t1_major],
                'open': arr_dict_major['open'][t0_major:t1_major],
                'high': arr_dict_major['high'][t0_major:t1_major],
                'low': arr_dict_major['low'][t0_major:t1_major],
                'close': arr_dict_major['close'][t0_major:t1_major],
                'volume_quote': arr_dict_major.get('volume_quote', arr_dict_major['close']*0)[t0_major:t1_major],
                'low_time': arr_dict_major['low_time'][t0_major:t1_major],
                'high_time': arr_dict_major['high_time'][t0_major:t1_major],
            }

        # Evaluación paralela
        with tqdm_joblib(
            tqdm(desc=f"🔁 WFO Window {window_idx}", total=len(dict_combinations), dynamic_ncols=True)
        ) as progress:
            results = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_fn)(params, (base_arrays_minor, base_arrays_major)) 
                for params in dict_combinations
            )

        # Seleccionar el mejor resultado
        best_criterion, best_params = max(results, key=lambda x: x[0])
        best_params_list.append(best_params)
        best_criteria_list.append(best_criterion)

        # Guardar fechas
        if ref_sym in train_indices_minor:
            t0, t1 = train_indices_minor[ref_sym]
            test0, test1 = test_indices_minor[ref_sym]
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

    # Resumen de parámetros finales (moda)
    df_params = pd.DataFrame(best_params_list)
    final_params = {}
    for col in df_params.columns:
        counts = Counter(df_params[col])
        most_common_val, _ = counts.most_common(1)[0]
        
        if isinstance(most_common_val, (int, float)) and not str(col).endswith("_MAX"):
            final_params[col] = int(round(most_common_val))
        else:
            final_params[col] = most_common_val

    # Convertir fechas
    train_start_dates = [pd.to_datetime(d).date() if d is not None else None for d in train_start_dates]
    train_end_dates = [pd.to_datetime(d).date() if d is not None else None for d in train_end_dates]
    test_start_dates = [pd.to_datetime(d).date() if d is not None else None for d in test_start_dates]
    test_end_dates = [pd.to_datetime(d).date() if d is not None else None for d in test_end_dates]

    df_results = pd.DataFrame(best_params_list)
    df_results.insert(0, 'train_start', train_start_dates)
    df_results.insert(1, 'train_end', train_end_dates)
    df_results['best_criterion'] = best_criteria_list

    print("\n📊 Resumen final de parámetros, criterio y fechas de train por ventana:")
    print(df_results)
    print(f"\n✅ WFO completed: {window_idx} windows processed (parallelized with {n_jobs} threads)\n")

    return final_params

