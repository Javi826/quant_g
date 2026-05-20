#shared/shared_batch/backtest_compiler.py
import pandas as pd
import numpy as np

import logging
logger = logging.getLogger("shared.utils.backtest_compiler")

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
            #"trade_log": port.get("trade_log", pd.DataFrame()),
            "duration_m": duration_days 
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



