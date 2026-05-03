import numpy as np
import pandas as pd
import itertools
from joblib import Parallel, delayed
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib
from collections import Counter
from sklearn.linear_model import LinearRegression

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.torque import extract_ohlcv_from_path, compile_MC_results, prepare_ohlcv_arrays
from tools.optimize_MCf_tf import generate_paths_for_all_symbols_functional


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
    _BACKTEST_PARAMS = {'sell_after', 'tp_pct', 'sl_pct'}
    for path_idx in range(n_paths):
        ohlcv_arrays = extract_ohlcv_from_path(paths_per_symbol, path_idx, dtype=dtype)

        for sym, arr in ohlcv_arrays.items():
            signals = signal_fn(arr, **{k.lower(): v for k, v in param_dict.items() if k.lower() not in _BACKTEST_PARAMS}, live_trading=False)
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
    _BACKTEST_PARAMS = {'sell_after', 'tp_pct', 'sl_pct'}
    for sym, arr in ohlcv_arr.items():
        signals = signal_fn(arr, **{k.lower(): v for k, v in param_dict.items() if k.lower() not in _BACKTEST_PARAMS}, live_trading=False)
        arr['signal'] = np.asarray(signals, dtype=dtype)

    result = run_grid_backtest(
        ohlcv_arr,
        sell_after=param_dict.get('SELL_AFTER'),
        tp_pct=param_dict.get('TP_PCT'),
        sl_pct=param_dict.get('SL_PCT'),
        order_amount=order_amount
    )

    port   = result.get('__PORTFOLIO__', {})
    trades = np.asarray(port.get('trades', []), dtype=np.float64)
    net_gain     = float(np.sum(trades)) if trades.size > 0 else 0.0
    net_gain_pct = (net_gain / initial_balance) * 100.0

    # R² equity curve
    equity_hist = port.get('sim_balance_history', {})
    balances    = equity_hist.get('balance', [])
    if len(balances) >= 2:
        y  = np.array(balances).reshape(-1, 1)
        X  = np.arange(len(y)).reshape(-1, 1)
        r2 = round(LinearRegression().fit(X, y).score(X, y), 2)
    else:
        r2 = np.nan

    return {
        'OOS_Net_Gain_pct': net_gain_pct,
        'OOS_Win_Ratio':    float(port.get('proportion_winners', np.nan)),
        'OOS_DD_pct':       float(port.get('max_dd', 0.0)) * 100.0,
        'OOS_Sharpe':       float(port.get('sharpe', np.nan)),
        'OOS_R2':           r2,
        'OOS_Num_Signals':  int(port.get('num_signals', 0)),
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
) -> pd.DataFrame:

    keys = list(param_ranges.keys())
    all_combinations = [dict(zip(keys, comb)) for comb in itertools.product(*[param_ranges[k] for k in keys])]

    length_test = int(length_train_set / pct_train_set - length_train_set)

    ref_sym    = max(ohlcv_data.keys(), key=lambda k: len(ohlcv_data[k]))
    ref_df     = ohlcv_data[ref_sym]
    max_length = len(ref_df)

    window_idx     = 1
    start          = 0
    end            = length_train_set
    window_records = []

    while start < max_length:
        remaining_data = max_length - start
        is_last_window = remaining_data < (length_train_set + length_test)

        if is_last_window:
            remaining  = max_length - start
            train_size = int(remaining * pct_train_set)
            if train_size < int(length_train_set * 0.8):
                break
            t0    = start
            t1    = start + train_size
            test0 = t1
            test1 = max_length
        else:
            t0    = 0 if anchored else start
            t1    = end if anchored else start + length_train_set
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

        best_idx      = int(np.argmax(is_scores))
        best_params   = all_combinations[best_idx]
        best_is_score = is_scores[best_idx]

        # ------------------------------------------------------------------
        # OOS: slice DataFrames → convert to arrays → evaluate best combo
        # ------------------------------------------------------------------
        ohlcv_oos_df  = _slice_ohlcv_data(ohlcv_data, test0, test1)
        ohlcv_oos_arr = prepare_ohlcv_arrays(ohlcv_oos_df)

        # Skip if OOS too short for indicators
        if any(len(arr['close']) < 50 for arr in ohlcv_oos_arr.values()):
            print(f"\n⚠️  Window {window_idx} OOS too short ({min(len(arr['close']) for arr in ohlcv_oos_arr.values())} bars), skipping.")
            oos_metrics = {
                'OOS_Net_Gain_pct': np.nan,
                'OOS_Win_Ratio':    np.nan,
                'OOS_DD_pct':       np.nan,
                'OOS_Sharpe':       np.nan,
                'OOS_R2':           np.nan,
            }
        else:
            oos_metrics = _evaluate_combo_on_real(
                best_params, ohlcv_oos_arr,
                signal_fn, run_grid_backtest,
                order_amount, initial_balance, dtype
            )

        # ------------------------------------------------------------------
        # Store window result
        # ------------------------------------------------------------------
        def _fmt(d): return pd.to_datetime(d).strftime('%Y-%m')

        is_period  = f"{_fmt(ref_df.index[t0])}-{_fmt(ref_df.index[t1 - 1])}"
        oos_period = f"{_fmt(ref_df.index[test0])}-{_fmt(ref_df.index[test1 - 1])}"

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
    if not window_records:
        print("\n⚠️  No windows were processed — insufficient data.")
        return pd.DataFrame()

    df_results   = pd.DataFrame(window_records)
    param_cols   = list(param_ranges.keys())
    numeric_cols = df_results.select_dtypes(include=[np.number]).columns

    # MEAN
    mean_row = df_results.select_dtypes(include=[np.number]).mean().to_dict()
    mean_row['window']     = 'MEAN'
    mean_row['IS_period']  = ''
    mean_row['OOS_period'] = ''

    # MODE
    mode_row = {}
    for col in df_results.columns:
        if col in param_cols:
            counts = Counter(df_results[col].dropna().tolist())
            mode_row[col] = counts.most_common(1)[0][0] if counts else np.nan
        else:
            mode_row[col] = np.nan
    mode_row['window']     = 'MODE'
    mode_row['IS_period']  = ''
    mode_row['OOS_period'] = ''

    # EWM
    ewm_row = {}
    for col in df_results.columns:
        if col in param_cols:
            ewm_row[col] = round(df_results[col].ewm(span=len(df_results), adjust=True).mean().iloc[-1], 2)
        else:
            ewm_row[col] = np.nan
    ewm_row['window']     = 'EWM'
    ewm_row['IS_period']  = ''
    ewm_row['OOS_period'] = ''

    df_results = pd.concat(
        [df_results, pd.DataFrame([mean_row]), pd.DataFrame([mode_row]), pd.DataFrame([ewm_row])],
        ignore_index=True
    )
    df_results[numeric_cols] = df_results[numeric_cols].round(2)

    # ------------------------------------------------------------------
    # Print: data rows + separator + summary rows
    # ------------------------------------------------------------------
    df_data    = df_results[df_results['window'].apply(lambda x: str(x).isdigit())]
    df_summary = df_results[df_results['window'].isin(['MEAN', 'MODE', 'EWM'])]

    full_table = pd.concat([df_data, df_summary], ignore_index=True)
    full_lines = full_table.to_string(index=False).split('\n')
    n_data_rows = len(df_data)
    separator   = '-' * len(full_lines[0])

    print("\n📊 WFO-MC Summary:")
    print(full_lines[0])
    for i, line in enumerate(full_lines[1:]):
        print(line)
        if i == n_data_rows - 1:
            print(separator)

    oos_std     = df_data['OOS_Net_Gain_pct'].std()
    oos_pos_pct = (df_data['OOS_Net_Gain_pct'] > 0).mean() * 100
    print(f"\nStd OOS_Net_Gain_pct     : {oos_std:.2f}%")
    print(f"Positive OOS windows     : {oos_pos_pct:.1f}%")
    print(f"\n✅ WFO-MC completed: {window_idx - 1} windows processed")

    return df_results