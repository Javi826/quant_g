"""
quality_control/analyzer.py
Quality Control Analyzer - Drift detection and execution quality metrics.

FIXED: Changed to use CLOSE execution data (order_price_close, order_ts_close, exec_ts_close)
       in lowercase to match PostgreSQL column names.

UPDATED: Adaptive window size - calculates with available trades (no minimum required).

Calculates:
1. Drift status per strategy (HEALTHY, WARNING, DANGER)
2. Execution quality per strategy (slippage, latency)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

from config.settings import DRIFT_WINDOW_SIZE, DRIFT_CHECK_INTERVAL
from config.settings import EXECUTION_WINDOW_SIZE
from config.settings import SLIPPAGE_WARNING_PCT, SLIPPAGE_CRITICAL_PCT, LATENCY_WARNING_SEC, LATENCY_CRITICAL_SEC


DRIFT_REFERENCE = {}  # overridden at runtime by configure_account()

def configure_account(account_number: str) -> None:
    global DRIFT_REFERENCE
    try:
        module = __import__(
            f'quality_control.drift_montecarlo_{account_number}',
            fromlist=['DRIFT_REFERENCE']
        )
        DRIFT_REFERENCE = module.DRIFT_REFERENCE
    except ModuleNotFoundError:
        pass  # keep default

logger = logging.getLogger('BOT_trading.quality_control.analyzer')


def analyze_drift_status(df_trades: pd.DataFrame, strategies_config: List[Dict]) -> Dict[str, Any]:
    """
    Analyze drift status for all strategies.
    
    UPDATED: Now uses adaptive window size - calculates with available trades (no minimum).
    
    Args:
        df_trades: DataFrame with all closed trades (columns: STRATEGY, PROFIT, CLOSE_AT, etc.)
        strategies_config: List of strategy configurations
    
    Returns:
        Dict with drift analysis per strategy:
        {
            'strategy_id': {
                'status': 'HEALTHY' | 'WARNING' | 'DANGER' | 'NO_DATA' | 'NO_REFERENCE',
                'winrate_100': 62.5,
                'winrate_100_l20': 58.3,
                'p5_reference': 52.0,
                'p50_reference': 60.0,
                'avg_profit_100': 12.5,
                'total_trades': 145,
                'counter': 0
            }
        }
    """
    results = {}
    
    for strat_config in strategies_config:
        strategy_id = strat_config['id']
        
        # Get strategy trades
        df_strat = df_trades[df_trades['STRATEGY'] == strategy_id].copy()
        
        if len(df_strat) == 0:
            results[strategy_id] = {
                'status': 'NO_DATA',
                'winrate_100': None,
                'winrate_100_l20': None,
                'p5_reference': None,
                'p50_reference': None,
                'avg_profit_100': None,
                'total_trades': 0,
                'counter': 0
            }
            continue
        
        total_trades = len(df_strat)
        
        # Get reference values
        reference = DRIFT_REFERENCE.get(strategy_id, {})
        p5_wr = reference.get('p5_winrate')
        p50_wr = reference.get('p50_winrate')
        
        if p5_wr is None or p50_wr is None:
            logger.warning(f"[DRIFT] No reference values for {strategy_id}")
            results[strategy_id] = {
                'status': 'NO_REFERENCE',
                'winrate_100': None,
                'winrate_100_l20': None,
                'p5_reference': None,
                'p50_reference': None,
                'avg_profit_100': None,
                'total_trades': total_trades,
                'counter': 0
            }
            continue
        
        # UPDATED: Use adaptive window size (max DRIFT_WINDOW_SIZE, or all available if less)
        window_size = min(total_trades, DRIFT_WINDOW_SIZE)
        df_last_N = df_strat.tail(window_size)
        
        # WinRate_100 (current window - using available trades)
        winning_trades = len(df_last_N[df_last_N['PROFIT'] > 0])
        winrate_100 = (winning_trades / window_size) * 100
        
        # WinRate_100_L20 (previous window - 20 trades ago)
        winrate_100_l20 = None
        if total_trades >= window_size + DRIFT_CHECK_INTERVAL:
            start_idx = total_trades - window_size - DRIFT_CHECK_INTERVAL
            end_idx = total_trades - DRIFT_CHECK_INTERVAL
            df_prev_N = df_strat.iloc[start_idx:end_idx]
            
            if len(df_prev_N) == window_size:
                prev_winning = len(df_prev_N[df_prev_N['PROFIT'] > 0])
                winrate_100_l20 = (prev_winning / window_size) * 100
        
        # Avg Profit per trade (simpler than Avg_R)
        avg_profit_100 = df_last_N['PROFIT'].mean()
        
        # Determine status
        status = 'HEALTHY'
        counter = 0
        
        if winrate_100 < p50_wr:
            status = 'WARNING'
        
        if winrate_100 < p5_wr and avg_profit_100 < 0:
            counter = 1
            
            # Check if this is 2nd consecutive failure (on-the-fly calculation)
            if total_trades >= window_size + DRIFT_CHECK_INTERVAL:
                # Get previous window
                start_idx = total_trades - window_size - DRIFT_CHECK_INTERVAL
                end_idx = total_trades - DRIFT_CHECK_INTERVAL
                df_prev = df_strat.iloc[start_idx:end_idx]
                
                if len(df_prev) == window_size:
                    prev_winning = len(df_prev[df_prev['PROFIT'] > 0])
                    prev_winrate = (prev_winning / window_size) * 100
                    prev_avg_profit = df_prev['PROFIT'].mean()
                    
                    if prev_winrate < p5_wr and prev_avg_profit < 0:
                        status = 'DANGER'
                        counter = 2
        
        results[strategy_id] = {
            'status': status,
            'winrate_100': round(winrate_100, 1),
            'winrate_100_l20': round(winrate_100_l20, 1) if winrate_100_l20 is not None else None,
            'p5_reference': round(p5_wr, 1),
            'p50_reference': round(p50_wr, 1),
            'avg_profit_100': round(avg_profit_100, 2),
            'total_trades': int(total_trades),
            'counter': int(counter)
        }
    
    return results


def analyze_execution_quality(df_trades: pd.DataFrame, strategies_config: List[Dict]) -> Dict[str, Any]:
    """
    Analyze execution quality (slippage, latency) for all strategies.
    
    Calculates:
    - Close slippage: difference between order price and actual execution price
    - TP slippage: difference between TP target and actual close price (TP trades only)
    - SL slippage: difference between SL target and actual close price (SL trades only)
    - Latency: time between order submission and execution
    
    Args:
        df_trades: DataFrame with trades (columns: STRATEGY, order_price_close, PRICE_CLOSE, 
                   order_ts_close, exec_ts_close, TP_TARGET, SL_TARGET, REASON_OUT)
        strategies_config: List of strategy configurations
    
    Returns:
        Dict with execution quality per strategy:
        {
            'strategy_id': {
                'total_trades': 145,
                'avg_close_slippage_pct': 0.02,
                'close_slippage_status': 'HEALTHY' | 'WARNING' | 'DANGER' | 'NO_DATA',
                'avg_tp_slippage_pct': -0.15,
                'tp_slippage_status': 'HEALTHY' | 'WARNING' | 'DANGER' | 'NO_DATA',
                'avg_sl_slippage_pct': 0.25,
                'sl_slippage_status': 'HEALTHY' | 'WARNING' | 'DANGER' | 'NO_DATA',
                'avg_latency_sec': 0.8,
                'latency_status': 'HEALTHY' | 'WARNING' | 'DANGER' | 'NO_DATA'
            }
        }
    """
    results = {}
    
    for strat_config in strategies_config:
        strategy_id = strat_config['id']
        
        # Get strategy trades
        df_strat = df_trades[df_trades['STRATEGY'] == strategy_id].copy()
        
        if len(df_strat) == 0:
            results[strategy_id] = {
                'total_trades': 0,
                'avg_close_slippage_pct': None,
                'close_slippage_status': 'NO_DATA',
                'avg_tp_slippage_pct': None,
                'tp_slippage_status': 'NO_DATA',
                'avg_sl_slippage_pct': None,
                'sl_slippage_status': 'NO_DATA',
                'avg_latency_sec': None,
                'latency_status': 'NO_DATA'
            }
            continue
        
        total_trades = len(df_strat)
        
        # Get last EXECUTION_WINDOW_SIZE trades
        df_last = df_strat.tail(EXECUTION_WINDOW_SIZE)
        
        # =====================================================================
        # CLOSE SLIPPAGE (execution slippage)
        # =====================================================================
        avg_close_slippage_pct = None
        close_slippage_status = 'NO_DATA'
        
        if 'order_price_close' in df_last.columns and 'PRICE_CLOSE' in df_last.columns:
            df_with_slippage = df_last[
                df_last['order_price_close'].notna() & 
                df_last['PRICE_CLOSE'].notna()
            ].copy()
            
            if len(df_with_slippage) > 0:
                df_with_slippage['close_slippage_pct'] = (
                    (df_with_slippage['PRICE_CLOSE'] - df_with_slippage['order_price_close']) 
                    / df_with_slippage['order_price_close'] 
                    * 100
                )
                avg_close_slippage_pct = df_with_slippage['close_slippage_pct'].mean()
                
                # Determine status
                abs_slippage = abs(avg_close_slippage_pct)
                if abs_slippage < SLIPPAGE_WARNING_PCT:
                    close_slippage_status = 'HEALTHY'
                elif abs_slippage < SLIPPAGE_CRITICAL_PCT:
                    close_slippage_status = 'WARNING'
                else:
                    close_slippage_status = 'DANGER'
        
        # =====================================================================
        # TP SLIPPAGE (target vs actual)
        # =====================================================================
        avg_tp_slippage_pct = None
        tp_slippage_status = 'NO_DATA'
        
        if 'TP_TARGET' in df_last.columns and 'PRICE_CLOSE' in df_last.columns and 'REASON_OUT' in df_last.columns:
            df_tp = df_last[
                (df_last['REASON_OUT'] == 'TP') &
                df_last['TP_TARGET'].notna() & 
                df_last['PRICE_CLOSE'].notna()
            ].copy()
            
            if len(df_tp) > 0:
                df_tp['tp_slippage_pct'] = (
                    (df_tp['PRICE_CLOSE'] - df_tp['TP_TARGET']) 
                    / df_tp['TP_TARGET'] 
                    * 100
                )
                avg_tp_slippage_pct = df_tp['tp_slippage_pct'].mean()
                
                # Determine status (same thresholds as close slippage)
                abs_slippage = abs(avg_tp_slippage_pct)
                if abs_slippage < SLIPPAGE_WARNING_PCT:
                    tp_slippage_status = 'HEALTHY'
                elif abs_slippage < SLIPPAGE_CRITICAL_PCT:
                    tp_slippage_status = 'WARNING'
                else:
                    tp_slippage_status = 'DANGER'
        
        # =====================================================================
        # SL SLIPPAGE (target vs actual)
        # =====================================================================
        avg_sl_slippage_pct = None
        sl_slippage_status = 'NO_DATA'
        
        if 'SL_TARGET' in df_last.columns and 'PRICE_CLOSE' in df_last.columns and 'REASON_OUT' in df_last.columns:
            df_sl = df_last[
                (df_last['REASON_OUT'] == 'SL') &
                df_last['SL_TARGET'].notna() & 
                df_last['PRICE_CLOSE'].notna()
            ].copy()
            
            if len(df_sl) > 0:
                df_sl['sl_slippage_pct'] = (
                    (df_sl['PRICE_CLOSE'] - df_sl['SL_TARGET']) 
                    / df_sl['SL_TARGET'] 
                    * 100
                )
                avg_sl_slippage_pct = df_sl['sl_slippage_pct'].mean()
                
                # Determine status (same thresholds as close slippage)
                abs_slippage = abs(avg_sl_slippage_pct)
                if abs_slippage < SLIPPAGE_WARNING_PCT:
                    sl_slippage_status = 'HEALTHY'
                elif abs_slippage < SLIPPAGE_CRITICAL_PCT:
                    sl_slippage_status = 'WARNING'
                else:
                    sl_slippage_status = 'DANGER'
        
        # =====================================================================
        # LATENCY
        # =====================================================================
        avg_latency_sec = None
        latency_status = 'NO_DATA'
        
        if 'exec_ts_close' in df_last.columns and 'order_ts_close' in df_last.columns:
            df_with_latency = df_last[
                df_last['exec_ts_close'].notna() & 
                df_last['order_ts_close'].notna()
            ].copy()
            
            if len(df_with_latency) > 0:
                # Latency in seconds (timestamps already in seconds with decimals)
                df_with_latency['latency_sec'] = (
                    df_with_latency['exec_ts_close'] - df_with_latency['order_ts_close']
                )
                avg_latency_sec = df_with_latency['latency_sec'].mean()
                
                # Determine status
                if avg_latency_sec < LATENCY_WARNING_SEC:
                    latency_status = 'HEALTHY'
                elif avg_latency_sec < LATENCY_CRITICAL_SEC:
                    latency_status = 'WARNING'
                else:
                    latency_status = 'DANGER'
        
        results[strategy_id] = {
            'total_trades': int(total_trades),
            'avg_close_slippage_pct': round(avg_close_slippage_pct, 4) if avg_close_slippage_pct is not None else None,
            'close_slippage_status': close_slippage_status,
            'avg_tp_slippage_pct': round(avg_tp_slippage_pct, 4) if avg_tp_slippage_pct is not None else None,
            'tp_slippage_status': tp_slippage_status,
            'avg_sl_slippage_pct': round(avg_sl_slippage_pct, 4) if avg_sl_slippage_pct is not None else None,
            'sl_slippage_status': sl_slippage_status,
            'avg_latency_sec': round(avg_latency_sec, 3) if avg_latency_sec is not None else None,
            'latency_status': latency_status
        }
    
    return results

def analyze_target_deviation(df_trades: pd.DataFrame, strategies_config: List[Dict]) -> Dict[str, Any]:
    """
    Analyze target deviation (TP/SL real vs configured) for all strategies.
    
    Args:
        df_trades: DataFrame with trades (columns: STRATEGY, PROFIT_PCT, REASON_OUT)
        strategies_config: List of strategy configurations with tp_pct and sl_pct
    
    Returns:
        Dict with target deviation per strategy:
        {
            'strategy_id': {
                'tp_trades': 45,
                'tp_real_pct': 2.8,
                'tp_target_pct': 3.0,
                'tp_deviation': -0.2,
                'sl_trades': 12,
                'sl_real_pct': -9.5,
                'sl_target_pct': -10.0,
                'sl_deviation': 0.5
            }
        }
    """
    results = {}
    
    for strat_config in strategies_config:
        strategy_id = strat_config['id']
        
        # Get configured TP/SL from strategy config
        tp_target_pct = strat_config.get('tp_pct')
        sl_target_pct_raw = strat_config.get('sl_pct')
        
        # Convert SL to negative (config stores positive values like 10.0)
        sl_target_pct = -abs(sl_target_pct_raw) if sl_target_pct_raw is not None else None
        
        # Get strategy trades
        df_strat = df_trades[df_trades['STRATEGY'] == strategy_id].copy()
        
        if len(df_strat) == 0:
            results[strategy_id] = {
                'tp_trades': 0,
                'tp_real_pct': None,
                'tp_target_pct': tp_target_pct,
                'tp_deviation': None,
                'sl_trades': 0,
                'sl_real_pct': None,
                'sl_target_pct': sl_target_pct,
                'sl_deviation': None
            }
            continue
        
        # TP analysis
        df_tp = df_strat[df_strat['REASON_OUT'] == 'TP']
        tp_trades = len(df_tp)
        tp_real_pct = None
        tp_deviation = None
        
        if tp_trades > 0 and 'PROFIT_PCT' in df_tp.columns:
            tp_real_pct = df_tp['PROFIT_PCT'].mean()
            if tp_target_pct is not None:
                tp_deviation = tp_real_pct - tp_target_pct
        
        # SL analysis
        df_sl = df_strat[df_strat['REASON_OUT'] == 'SL']
        sl_trades = len(df_sl)
        sl_real_pct = None
        sl_deviation = None
        
        if sl_trades > 0 and 'PROFIT_PCT' in df_sl.columns:
            sl_real_pct = df_sl['PROFIT_PCT'].mean()
            if sl_target_pct is not None:
                sl_deviation = sl_real_pct - sl_target_pct
        
        results[strategy_id] = {
            'tp_trades': int(tp_trades),
            'tp_real_pct': round(tp_real_pct, 2) if tp_real_pct is not None else None,
            'tp_target_pct': round(tp_target_pct, 2) if tp_target_pct is not None else None,
            'tp_deviation': round(tp_deviation, 2) if tp_deviation is not None else None,
            'sl_trades': int(sl_trades),
            'sl_real_pct': round(sl_real_pct, 2) if sl_real_pct is not None else None,
            'sl_target_pct': round(sl_target_pct, 2) if sl_target_pct is not None else None,
            'sl_deviation': round(sl_deviation, 2) if sl_deviation is not None else None
        }
    
    return results

def analyze_drift_binomial(df_trades: pd.DataFrame, strategies_config: List[Dict]) -> Dict[str, Any]:
    """
    Analyze drift status using binomial distribution with double confirmation.
    
    Uses statistical thresholds (P_target ± σ) instead of fixed Montecarlo percentiles.
    Requires both current and lagged windows to be in DANGER zone for confirmation.
    
    Args:
        df_trades: DataFrame with all closed trades (columns: STRATEGY, PROFIT, etc.)
        strategies_config: List of strategy configurations
    
    Returns:
        Dict with binomial drift analysis per strategy:
        {
            'strategy_id': {
                'status': 'HEALTHY' | 'WARNING' | 'DANGER' | 'INSUFFICIENT_DATA',
                'trades_count': 203,
                'winrate_current': 69.0,
                'winrate_l30': 65.0,
                'p_target': 82.0,
                'limit_warning': 73.6,
                'limit_danger': 69.1,
                'sigma': 3.84,
                'z_score': -3.39,
                'z_score_l30': -4.43
            }
        }
    """
    from config.settings import DRIFT_BINOMIAL_WINDOW, DRIFT_BINOMIAL_DEFAULT_P50, DRIFT_CHECK_INTERVAL
    import math
    
    results = {}
    
    for strategy in strategies_config:
        strategy_id = strategy['id']
        
        # Get P_target from DRIFT_REFERENCE
        if strategy_id in DRIFT_REFERENCE:
            p_target = DRIFT_REFERENCE[strategy_id]['p50_winrate'] / 100
        else:
            p_target = DRIFT_BINOMIAL_DEFAULT_P50
        
        # Filter trades for this strategy
        strategy_trades = df_trades[df_trades['STRATEGY'] == strategy_id].copy()
        total_trades = len(strategy_trades)
        
        if total_trades < DRIFT_BINOMIAL_WINDOW:
            results[strategy_id] = {
                'status': 'INSUFFICIENT_DATA',
                'trades_count': total_trades,
                'required': DRIFT_BINOMIAL_WINDOW,
                'winrate_current': None,
                'winrate_l30': None,
                'p_target': round(p_target * 100, 2),
                'limit_warning': None,
                'limit_danger': None,
                'sigma': None,
                'z_score': None,
                'z_score_l30': None
            }
            continue
        
        # Calculate binomial standard deviation
        sigma = math.sqrt(p_target * (1 - p_target) / DRIFT_BINOMIAL_WINDOW)
        
        # Calculate thresholds
        limit_warning = p_target - 2 * sigma  # -2σ
        limit_danger = p_target - 3 * sigma   # -3σ (0.13% probability)
        
        # =====================================================================
        # CURRENT WINDOW (last N trades)
        # =====================================================================
        last_trades = strategy_trades.tail(DRIFT_BINOMIAL_WINDOW)
        wins_current = (last_trades['PROFIT'] > 0).sum()
        winrate_current = wins_current / DRIFT_BINOMIAL_WINDOW
        z_score = (winrate_current - p_target) / sigma if sigma > 0 else 0
        
        # =====================================================================
        # LAGGED WINDOW (L30 - shifted 30 trades back)
        # =====================================================================
        winrate_l30 = None
        z_score_l30 = None
        
        if total_trades >= DRIFT_BINOMIAL_WINDOW + DRIFT_CHECK_INTERVAL:
            start_idx = total_trades - DRIFT_BINOMIAL_WINDOW - DRIFT_CHECK_INTERVAL
            end_idx = total_trades - DRIFT_CHECK_INTERVAL
            lagged_trades = strategy_trades.iloc[start_idx:end_idx]
            
            if len(lagged_trades) == DRIFT_BINOMIAL_WINDOW:
                wins_l30 = (lagged_trades['PROFIT'] > 0).sum()
                winrate_l30 = wins_l30 / DRIFT_BINOMIAL_WINDOW
                z_score_l30 = (winrate_l30 - p_target) / sigma if sigma > 0 else 0
        
        # =====================================================================
        # DETERMINE STATUS (with double confirmation)
        # =====================================================================
        status = 'HEALTHY'
        
        # Check current window
        if winrate_current < limit_warning:
            status = 'WARNING'
        
        # DANGER requires BOTH windows to be below danger limit
        if winrate_current < limit_danger:
            if winrate_l30 is not None and winrate_l30 < limit_danger:
                status = 'DANGER'  # Double confirmation
            elif winrate_l30 is None:
                status = 'WARNING'  # Can't confirm, keep as WARNING
        
        results[strategy_id] = {
            'status': status,
            'trades_count': total_trades,
            'winrate_current': round(winrate_current * 100, 2),
            'winrate_l30': round(winrate_l30 * 100, 2) if winrate_l30 is not None else None,
            'p_target': round(p_target * 100, 2),
            'limit_warning': round(limit_warning * 100, 2),
            'limit_danger': round(limit_danger * 100, 2),
            'sigma': round(sigma * 100, 2),
            'z_score': round(z_score, 2),
            'z_score_l30': round(z_score_l30, 2) if z_score_l30 is not None else None
        }
    
    return results