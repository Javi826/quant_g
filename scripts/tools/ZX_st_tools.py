import os
from tqdm.auto import tqdm
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
        

        final_balance = float(port.get('final_balance', initial_balance))
        num_signals   = int(port.get('num_signals', 0))
        win_ratio     = float(port.get('proportion_winners', np.nan))
        dd_pct        = float(port.get('max_dd', 0.0)) * 100.0
        sharpe_ratio  = float(port.get('sharpe', np.nan))
        

        trades = port.get('trades', [])
        num_trades = len(trades)
        
        if num_trades > 0:
            trades_arr = np.array(trades, dtype=np.float64)
            net_gain = float(np.sum(trades_arr))
            avg_trade = float(np.mean(trades_arr))
            median_trade = float(np.median(trades_arr))
        else:
            net_gain = 0.0
            avg_trade = np.nan
            median_trade = np.nan
        
        net_gain_pct = (net_gain / initial_balance) * 100.0 if initial_balance != 0 else np.nan
        

        duration_days = _calculate_duration_optimized(port.get('trade_log'))

        row = {param: value for param, value in zip(param_names, comb)}
        row.update({
            "symbol": "__PORTFOLIO__",
            "Net_Gain": net_gain,
            "Net_Gain_pct": float(net_gain_pct),
            "Final_Balance": final_balance,
            "Num_Signals": num_signals,
            "Num_Trades": num_trades,
            "Win_Ratio": win_ratio,
            "Avg_Trade": avg_trade,
            "Median_Trade": median_trade,
            "DD_pct": dd_pct,
            "Sharpe": sharpe_ratio,
            "sim_balance_history": port.get("sim_balance_history", {}),
            "duration_m": duration_days  # En días, no minutos (nombre legacy)
        })
        records.append(row)
    
    return records


def _calculate_duration_optimized(trade_log):

    if trade_log is None or not isinstance(trade_log, pd.DataFrame) or trade_log.empty:
        return np.nan
    
    if 'buy_time' not in trade_log.columns or 'sell_time' not in trade_log.columns:
        return np.nan
    
    try:
        # Ya son datetime64 desde run_grid_backtest, no necesitan conversión
        durations_sec = (trade_log['sell_time'] - trade_log['buy_time']).dt.total_seconds()
        
        # Filtrar valores válidos (positivos)
        valid_durations = durations_sec[durations_sec > 0]
        
        if len(valid_durations) == 0:
            return np.nan
        
        # Convertir segundos a días
        return float(valid_durations.mean() / 86400.0)
    
    except Exception:
        return np.nan


def save_all_trades_to_excel(grid_results_list, param_names, filename, save=True):

    if not save:
        print("⚠️ save=False, no se guardará el archivo de trades")
        return
    
    all_trades_records = []
    
    for comb, results in tqdm(grid_results_list, desc="📊 Procesando trades"):
        port = results.get("__PORTFOLIO__", None)
        if port is None:
            continue
        
        trade_log = port.get('trade_log', None)
        if trade_log is None or (isinstance(trade_log, pd.DataFrame) and trade_log.empty):
            continue
        
        # Convertir a DataFrame si es necesario
        if isinstance(trade_log, pd.DataFrame):
            tl_df = trade_log.copy()
        elif isinstance(trade_log, dict):
            tl_df = pd.DataFrame(trade_log)
        else:
            continue
        
        # Añadir columnas de parámetros
        for param_name, param_value in zip(param_names, comb):
            tl_df[param_name] = param_value
        
        all_trades_records.append(tl_df)
    
    if all_trades_records:
        # Concatenar todos los trades
        all_trades_df = pd.concat(all_trades_records, ignore_index=True)
        
        # Reordenar columnas (parámetros primero)
        param_cols = param_names
        trade_cols = [col for col in all_trades_df.columns if col not in param_names]
        all_trades_df = all_trades_df[param_cols + trade_cols]
        
        # Crear carpeta si no existe (igual que save_results)
        folder = os.path.dirname(filename)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        # Guardar
        all_trades_df.to_excel(filename, index=False, engine='openpyxl')
        
        file_size_mb = os.path.getsize(filename) / 1024 / 1024
        print(f"✅ Saved {len(all_trades_df):,} trades en: {filename}")
        print(f"   📦 Size: {file_size_mb:.2f} MB")
    else:
        print("⚠️ No hay trades para guardar")

        
def save_results(grid_results, grid_results_df, filename="grid_backtest.xlsx",save=False):
    
    if save:
        folder = os.path.dirname(filename)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        grid_results_df.to_excel(filename, index=False)
        print(f"📂 File saved successfully as: {filename}")

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



