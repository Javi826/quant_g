
import pandas as pd
import numpy as np

def prepare_ohlcv_arrays(ohlcv_data):
    ohlcv_arr = {}
    for sym, df in ohlcv_data.items():
        ohlcv_arr[sym] = {
            'ts': df.index.values.astype('datetime64[ns]'),
            'open': df['open'].to_numpy(dtype=np.float64),
            'high': df['high'].to_numpy(dtype=np.float64),
            'low': df['low'].to_numpy(dtype=np.float64),
            'close': df['close'].to_numpy(dtype=np.float64),
            'volume_quote': df['volume_quote'].to_numpy(dtype=np.float64),
            'low_time': (pd.to_datetime(df['low_time']).to_numpy(dtype='datetime64[ns]')),
            'high_time': (pd.to_datetime(df['high_time']).to_numpy(dtype='datetime64[ns]'))
        }
    return ohlcv_arr

def extract_ohlcv_from_path(paths_per_symbol, path_idx, ts_index=None, dtype=np.float32):
    ohlcv_arrays = {}

    for sym, arr_paths in paths_per_symbol.items():
        if path_idx >= arr_paths.shape[0]:
            continue

        arr = arr_paths[path_idx]  # (n_obs, n_features)
        ohlcv_arrays[sym] = {
            'ts': ts_index if ts_index is not None else np.arange(arr.shape[0]),
            'open': arr[:, 0].astype(dtype),
            'low':  arr[:, 1].astype(dtype),
            'high': arr[:, 2].astype(dtype),
            'close': arr[:, 3].astype(dtype),
            'low_time': np.array(arr[:, 4], dtype='datetime64[ns]'),
            'high_time': np.array(arr[:, 5], dtype='datetime64[ns]'),
        }

    return ohlcv_arrays


def compile_grid_results(grid_results_list, param_names, initial_balance):

    records = []

    for comb, results in grid_results_list:
        port = results.get("__PORTFOLIO__", None)
        if port is None:
            continue

        net_gain      = np.sum(port['trades']) if len(port.get('trades', [])) > 0 else 0.0
        net_gain_pct  = (net_gain / initial_balance) * 100.0 if initial_balance != 0 else np.nan
        num_signals   = int(port.get('num_signals', 0))
        num_trades    = len(port.get('trades', []))
        win_ratio     = port.get('proportion_winners', np.nan)
        dd_pct        = port.get('max_dd', 0.0) * 100.0
        final_balance = float(port.get('final_balance', initial_balance))
        avg_trade     = np.nan if num_trades == 0 else np.mean(port['trades'])
        median_trade  = np.nan if num_trades == 0 else np.median(port['trades'])
        sharpe_ratio  = float(port.get('sharpe', np.nan))

        row = {param: value for param, value in zip(param_names, comb)}
        row.update({
            "symbol": "__PORTFOLIO__",
            "Net_Gain": float(net_gain),
            "Net_Gain_pct": float(net_gain_pct),
            "Final_Balance": final_balance,
            "Num_Signals": num_signals,
            "Num_Trades": num_trades,
            "Win_Ratio": float(win_ratio) if not pd.isna(win_ratio) else np.nan,
            "Avg_Trade": float(avg_trade) if not pd.isna(avg_trade) else np.nan,
            "Median_Trade": float(median_trade) if not pd.isna(median_trade) else np.nan,
            "DD_pct": float(dd_pct),
            "Sharpe": sharpe_ratio,
            "sim_balance_history": port.get("sim_balance_history", [])
        })
        records.append(row)
        
        

    return records


def compile_MC_results(result, param_dict, path_idx, initial_balance, dtype=np.float64):

    portfolio     = result.get("__PORTFOLIO__", {})
    trades        = np.asarray(portfolio.get('trades', []), dtype=dtype) if portfolio.get('trades') else np.array([], dtype=dtype)
    final_balance = np.float64(portfolio.get('final_balance', initial_balance))
    num_signals   = portfolio.get('num_signals', 0)
    win_ratio     = portfolio.get('proportion_winners', np.nan)
    max_dd        = portfolio.get('max_dd', 0.0)
    sharpe        = float(portfolio.get('sharpe', np.nan))

    return {
        **param_dict,
        "path_index": path_idx,
        "symbol": "__PORTFOLIO__",
        "Net_Gain": np.sum(trades) if trades.size > 0 else 0.0,
        "Net_Gain_pct": (np.sum(trades)/initial_balance*100.0) if trades.size > 0 else 0.0,
        "Num_Signals": num_signals,
        "Win_Ratio": win_ratio,
        "DD": max_dd*100 if isinstance(max_dd,(int,float)) else np.nan,
        "Portfolio_Final_Balance": final_balance,
        "Portfolio_Num_Signals": num_signals,
        "Sharpe": sharpe,
        "error": None
    }



