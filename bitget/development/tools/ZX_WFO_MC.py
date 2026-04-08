import numpy as np
import pandas as pd
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from collections import Counter

from tools.ZX_st_tools import extract_ohlcv_from_path, compile_MC_results, prepare_ohlcv_arrays
from tools.ZX_optimize_MCf_tf import generate_paths_for_all_symbols_functional


def _slice_ohlcv_data(ohlcv_data: dict, t0: int, t1: int) -> dict:
    return {
        sym: df.iloc[t0:t1]
        for sym, df in ohlcv_data.items()
        if t1 > t0 and len(df) > t0
    }


def _evaluate_combo_on_paths(
    param_dict: dict,
    paths_per_symbol: dict,
    n_paths: int,
    signal_fn,
    run_grid_backtest,
    order_amount: float,
    initial_balance: float,
    dtype=np.float32
) -> float:
    records = []

    for path_idx in range(n_paths):
        ohlcv_arrays = extract_ohlcv_from_path(paths_per_symbol, path_idx, dtype=dtype)

        for sym, arr in ohlcv_arrays.items():
            signals = signal_fn(
                arr,
                lookback=param_dict.get('LOOKBACK'),
                tolerance=param_dict.get('TOLERANCE'),
                impulse=param_dict.get('IMPULSE'),
                live_trading=False
            )
            arr['signal'] = np.asarray(signals, dtype=dtype)

        result = run_grid_backtest(
            ohlcv_arrays,
            sell_after=param_dict.get('SELL_AFTER'),
            tp_pct=param_dict.get('TP_PCT'),
            sl_pct=param_dict.get('SL_PCT'),
            order_amount=order_amount
        )

        record = compile_MC_results(result, param_dict, path_idx, initial_balance, dtype=dtype)
        records.append(record)

    if not records:
        return -np.inf

    net_gains = [r['Net_Gain_pct'] for r in records if r['Net_Gain_pct'] is not None]
    return float(np.mean(net_gains)) if net_gains else -np.inf


def _evaluate_combo_on_real(
    param_dict: dict,
    ohlcv_arr: dict,
    signal_fn,
    run_grid_backtest,
    order_amount: float,
    initial_balance: float,
    dtype=np.float32
) -> dict:
    for sym, arr in ohlcv_arr.items():
        signals = signal_fn(
            arr,
            lookback=param_dict.get('LOOKBACK'),
            tolerance=param_dict.get('TOLERANCE'),
            impulse=param_dict.get('IMPULSE'),
            live_trading=False
        )
        arr['signal'] = np.asarray(signals, dtype=dtype)

    result = run_grid_backtest(
        ohlcv_arr,
        sell_after=param_dict.get('SELL_AFTER'),
        tp_pct=param_dict.get('TP_PCT'),
        sl_pct=param_dict.get('SL_PCT'),
        order_amount=order_amount
    )

    port = result.get('__PORTFOLIO__', {})
    trades = np.asarray(port.get('trades', []), dtype=np.float64)
    net_gain = float(np.sum(trades)) if trades.size > 0 else 0.0
    net_gain_pct = (net_gain / initial_balance) * 100.0

    return {
        'OOS_Net_Gain': net_gain,
        'OOS_Net_Gain_pct': net_gain_pct,
        'OOS_Win_Ratio': float(port.get('proportion_winners', np.nan)),
        'OOS_DD_pct': float(port.get('max_dd', 0.0)) * 100.0,
        'OOS_Sharpe': float(port.get('sharpe', np.nan)),
        'OOS_Num_Signals': int(port.get('num_signals', 0)),
    }


def walk_forward_optimization_mc(
    ohlcv_data: dict,
    param_ranges: dict,
    signal_fn,
    run_grid_backtest,
    length_train_set: int,
    pct_train_set: float,
    anchored: bool,
    n_paths: int,
    n_obs: int,
    order_amount: float,
    initial_balance: float,
    n_jobs: int = -1,
    dtype=np.float32
) -> tuple[dict, pd.DataFrame]:

    keys = list(param_ranges.keys())
    all_combinations = [dict(zip(keys, comb)) for comb in itertools.product(*[param_ranges[k] for k in keys])]

    length_test = int(length_train_set / pct_train_set - length_train_set)

    ref_sym = max(ohlcv_data.keys(), key=lambda k: len(ohlcv_data[k]))
    ref_df = ohlcv_data[ref_sym]
    max_length = len(ref_df)

    window_idx = 1
    start = 0
    end = length_train_set

    window_records = []

    while start < max_length:
        remaining_data = max_length - start
        is_last_window = remaining_data < (length_train_set + length_test)

        if is_last_window:
            remaining = max_length - start
            train_size = int(remaining * pct_train_set)
            test_size = remaining - train_size
            if train_size < int(length_train_set * 0.8):
                break
            t0 = start
            t1 = start + train_size
            test0 = t1
            test1 = max_length
        else:
            t0 = 0 if anchored else start
            t1 = end if anchored else start + length_train_set
            test0 = t1
            test1 = min(t1 + length_test, max_length)

        if t1 <= t0 or test1 <= test0:
            break

        # ------------------------------------------------------------------
        # IS: slice DataFrames → generate MC paths → evaluate grid in parallel
        # ------------------------------------------------------------------
        ohlcv_is = _slice_ohlcv_data(ohlcv_data, t0, t1)
        if not ohlcv_is:
            break

        paths_per_symbol = generate_paths_for_all_symbols_functional(
            ohlcv_is, n_paths=n_paths, n_obs=n_obs, raw_columns=[]
        )

        with tqdm_joblib(tqdm(desc=f"🔁 WFO-MC Window {window_idx} [IS]", total=len(all_combinations), dynamic_ncols=True)):
            is_scores = Parallel(n_jobs=n_jobs)(
                delayed(_evaluate_combo_on_paths)(
                    param_dict, paths_per_symbol, n_paths,
                    signal_fn, run_grid_backtest,
                    order_amount, initial_balance, dtype
                )
                for param_dict in all_combinations
            )

        best_idx = int(np.argmax(is_scores))
        best_params = all_combinations[best_idx]
        best_is_score = is_scores[best_idx]

        # ------------------------------------------------------------------
        # OOS: slice DataFrames → convert to arrays → evaluate best combo
        # ------------------------------------------------------------------
        ohlcv_oos_df = _slice_ohlcv_data(ohlcv_data, test0, test1)
        ohlcv_oos_arr = prepare_ohlcv_arrays(ohlcv_oos_df)

        oos_metrics = _evaluate_combo_on_real(
            best_params, ohlcv_oos_arr,
            signal_fn, run_grid_backtest,
            order_amount, initial_balance, dtype
        )

        # ------------------------------------------------------------------
        # Store window result
        # ------------------------------------------------------------------
        def _fmt(d): return pd.to_datetime(d).strftime('%Y-%m')

        is_period  = f"{_fmt(ref_df.index[t0])} / {_fmt(ref_df.index[t1 - 1])}"
        oos_period = f"{_fmt(ref_df.index[test0])} / {_fmt(ref_df.index[test1 - 1])}"

        oos_metrics.pop('OOS_Net_Gain', None)

        record = {
            'window':     window_idx,
            'IS_period':  is_period,
            'OOS_period': oos_period,
            **best_params,
            **oos_metrics
        }
        window_records.append(record)

        print(
            f"\n✅ Window {window_idx} | IS: {is_period} | "
            f"OOS: {oos_period} | "
            f"IS_score: {best_is_score:.2f}% | "
            f"OOS_Net_Gain_pct: {oos_metrics['OOS_Net_Gain_pct']:.2f}%"
        )
        for k, v in best_params.items():
            print(f"   {k}: {v}")

        window_idx += 1
        if is_last_window:
            break

        if anchored:
            end += length_test
        else:
            start += length_test
            end = start + length_train_set

# ------------------------------------------------------------------
    # Summary DataFrame
    # ------------------------------------------------------------------
    df_results = pd.DataFrame(window_records)

    mean_row = df_results.select_dtypes(include=[np.number]).mean().to_dict()
    mean_row['window'] = 'MEAN'
    mean_row['IS_period'] = ''
    mean_row['OOS_period'] = ''

    mode_row = {}
    for col in df_results.columns:
        if col in list(param_ranges.keys()):
            counts = Counter(df_results[col].dropna().tolist())
            mode_row[col] = counts.most_common(1)[0][0] if counts else np.nan
        else:
            mode_row[col] = np.nan
    mode_row['window'] = 'MODE'
    mode_row['IS_period'] = ''
    mode_row['OOS_period'] = ''

    df_results = pd.concat([df_results, pd.DataFrame([mean_row]), pd.DataFrame([mode_row])], ignore_index=True)

    numeric_cols = df_results.select_dtypes(include=[np.number]).columns
    df_results[numeric_cols] = df_results[numeric_cols].round(2)

    print("\n📊 WFO-MC Summary:")
    print(df_results.to_string(index=False))

    # ------------------------------------------------------------------
    # Final params: mode of best_params across windows
    # ------------------------------------------------------------------
    param_rows = df_results[df_results['window'] != 'MEAN'][list(param_ranges.keys())]
    final_params = {}
    for col in param_rows.columns:
        counts = Counter(param_rows[col].dropna().tolist())
        most_common_val, _ = counts.most_common(1)[0]
        if isinstance(most_common_val, (int, float)):
            final_params[col] = int(round(most_common_val))
        else:
            final_params[col] = most_common_val

    print(f"\n✅ WFO-MC completed: {window_idx - 1} windows processed")
    print(f"🏆 Final params (mode): {final_params}")

    return final_params, df_results