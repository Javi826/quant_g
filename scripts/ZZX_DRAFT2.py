import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from ZX_compute_BT import run_grid_backtest, INITIAL_BALANCE, ORDER_AMOUNT
from Z_add_signals_03 import add_indicators_03, explosive_signal_03


def walk_forward_optimization(data_dict, param_ranges,
                              length_train_set=2000, pct_train_set=0.8,
                              anchored=True, com_ewm=1.5,
                              n_jobs=-1):

    keys = list(param_ranges.keys())
    all_combinations = list(itertools.product(*[param_ranges[k] for k in keys]))
    dict_combinations = [dict(zip(keys, comb)) for comb in all_combinations]
    length_test = int(length_train_set / pct_train_set - length_train_set)
    best_params_list = []
    window_idx = 0

    start = 0
    end = length_train_set

    ref_sym = max(data_dict.keys(), key=lambda k: len(data_dict[k]['ts']))
    ref_ts = data_dict[ref_sym]['ts']
    max_length = len(ref_ts)

    def evaluate_WFO(params, base_arrays):
        ohlcv_arrays = {}
        for sym, arrs in base_arrays.items():
            entropia, accel = add_indicators_03(arrs['close'], m_accel=params.get('ACCEL_SPAN', 5))
            signal = explosive_signal_03(entropia, accel,
                                         entropia_max=params.get('ENTROPY_MAX', 1.0), live=False)
            ohlcv_arrays[sym] = {**arrs, 'signal': signal}

        results = run_grid_backtest(
            ohlcv_arrays,
            sell_after=params.get('SELL_AFTER', 10),
            tp_pct=params.get('TP_PCT', 0),
            sl_pct=params.get('SL_PCT', 0),
            initial_balance=INITIAL_BALANCE,
            order_amount=ORDER_AMOUNT,
            comi_pct=0.05
        )

        port = results.get("__PORTFOLIO__", {})
        net_gain = np.sum(port.get('trades', [])) if len(port.get('trades', [])) > 0 else 0.0
        net_gain_pct = (net_gain / INITIAL_BALANCE) * 100.0 if INITIAL_BALANCE != 0 else 0.0
        criterion = net_gain_pct
        return criterion, params

    while start < max_length:
        remaining_data = max_length - start
        is_last_window = remaining_data < (length_train_set + length_test)

        train_indices = {}
        test_indices = {}

        for sym, arr_dict in data_dict.items():
            sym_length = len(arr_dict['ts'])
            if start >= sym_length:
                continue

            if is_last_window:
                remaining = sym_length - start
                train_size = int(remaining * pct_train_set)
                test_size = remaining - train_size
                if train_size < 100 or test_size < 50:
                    continue
                t0, t1 = start, start + train_size
                test0, test1 = t1, sym_length
            else:
                t0 = 0 if anchored else start
                t1 = min(end, sym_length) if anchored else min(start + length_train_set, sym_length)
                test0 = t1
                test1 = min(t1 + length_test, sym_length)

            if t1 > t0 and test1 > test0:
                train_indices[sym] = (t0, t1)
                test_indices[sym] = (test0, test1)

        if not train_indices:
            break

        ventana_start_train = ref_ts[start]
        train_end_idx = min(start + length_train_set - 1, len(ref_ts) - 1)
        ventana_end_train = ref_ts[train_end_idx]
        test_start_idx = min(start + length_train_set, len(ref_ts) - 1)
        ventana_start_test = ref_ts[test_start_idx]
        test_end_idx = min(start + length_train_set + length_test - 1, len(ref_ts) - 1)
        ventana_end_test = ref_ts[test_end_idx]

        print(f"\nVentana {window_idx}{' (ÚLTIMA - ADAPTATIVA)' if is_last_window else ''}:")
        print(f"  Train: {ventana_start_train} -> {ventana_end_train}")
        print(f"  Test:  {ventana_start_test} -> {ventana_end_test}")
        print(f"  Símbolos en ventana: {len(train_indices)}")

        base_arrays = {}
        for sym, (t0, t1) in train_indices.items():
            arr_dict = data_dict[sym]
            ts_slice = arr_dict['ts'][t0:t1].astype('datetime64[ns]')
            open_slice = np.asarray(arr_dict['open'][t0:t1], dtype=np.float64)
            high_slice = np.asarray(arr_dict['high'][t0:t1], dtype=np.float64)
            low_slice = np.asarray(arr_dict['low'][t0:t1], dtype=np.float64)
            close_slice = np.asarray(arr_dict['close'][t0:t1], dtype=np.float64)
            volume_slice = np.asarray(arr_dict.get('volume_quote', np.zeros_like(close_slice))[t0:t1], dtype=np.float64)
            base_arrays[sym] = {
                'ts': ts_slice, 'open': open_slice, 'high': high_slice,
                'low': low_slice, 'close': close_slice, 'volume_quote': volume_slice
            }

        with tqdm_joblib(tqdm(desc=f"🔁 WFO Ventana {window_idx}...", total=len(dict_combinations))):
            results = Parallel(n_jobs=n_jobs)(
                delayed(evaluate_WFO)(params, base_arrays) for params in dict_combinations
            )

        best_criterion, best_params = max(results, key=lambda x: x[0])
        best_params_list.append(best_params)

        print(f"  Mejor criterio: {best_criterion:.4f}")
        print(f"  Mejores parámetros: {best_params}")

        window_idx += 1
        if is_last_window:
            break

        if anchored:
            end += length_test
        else:
            start += length_train_set + length_test
            end = start + length_train_set

    df_params = pd.DataFrame(best_params_list)
    df_num = df_params.select_dtypes(include=[np.number])
    df_cat = df_params.select_dtypes(exclude=[np.number])
    df_num_ewm = df_num.ewm(com=com_ewm, ignore_na=True).mean()

    if not df_cat.empty:
        df_cat_mode = df_cat.mode().iloc[-1:] if len(df_cat) > 0 else df_cat.iloc[-1:]
        df_final = pd.concat([df_num_ewm, df_cat_mode], axis=1)
    else:
        df_final = df_num_ewm

    final_params = df_final.iloc[-1].to_dict()
    print(f"\n✅ WFO completado: {window_idx} ventanas procesadas")
    return final_params
