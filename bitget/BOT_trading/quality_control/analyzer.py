"""
Quality Control Analyzer - Drift detection and execution quality metrics.

FIXED: Changed to use CLOSE execution data (order_price_close, order_ts_close, exec_ts_close)
       in lowercase to match PostgreSQL column names.

Calculates:
1. Drift status per strategy (HEALTHY, WARNING, DANGER)
2. Execution quality per strategy (slippage, latency)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any
import logging

from config.settings import (
    DRIFT_WINDOW_SIZE,
    DRIFT_CHECK_INTERVAL,
    EXECUTION_WINDOW_SIZE,
    SLIPPAGE_WARNING_PCT,
    SLIPPAGE_CRITICAL_PCT,
    LATENCY_WARNING_SEC,
    LATENCY_CRITICAL_SEC
)
from .drift_montecarlo import DRIFT_REFERENCE

logger = logging.getLogger('BOT_trading.quality_control.analyzer')


def analyze_drift_status(df_trades: pd.DataFrame, strategies_config: List[Dict]) -> Dict[str, Any]:
    """
    Analyze drift status for all strategies.
    
    Args:
        df_trades: DataFrame with all closed trades (columns: STRATEGY, PROFIT, CLOSE_AT, etc.)
        strategies_config: List of strategy configurations
    
    Returns:
        Dict with drift analysis per strategy:
        {
            'strategy_id': {
                'status': 'HEALTHY' | 'WARNING' | 'DANGER' | 'NO_DATA' | 'INSUFFICIENT_DATA' | 'NO_REFERENCE',
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
        
        # Need at least DRIFT_WINDOW_SIZE trades to evaluate
        if total_trades < DRIFT_WINDOW_SIZE:
            results[strategy_id] = {
                'status': 'INSUFFICIENT_DATA',
                'winrate_100': None,
                'winrate_100_l20': None,
                'p5_reference': None,
                'p50_reference': None,
                'avg_profit_100': None,
                'total_trades': total_trades,
                'counter': 0
            }
            continue
        
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
        
        # Calculate metrics for last DRIFT_WINDOW_SIZE trades (current window)
        df_last_100 = df_strat.tail(DRIFT_WINDOW_SIZE)
        
        # WinRate_100 (current window)
        winning_trades = len(df_last_100[df_last_100['PROFIT'] > 0])
        winrate_100 = (winning_trades / len(df_last_100)) * 100
        
        # WinRate_100_L20 (previous window - 20 trades ago)
        winrate_100_l20 = None
        if total_trades >= DRIFT_WINDOW_SIZE + DRIFT_CHECK_INTERVAL:
            start_idx = total_trades - DRIFT_WINDOW_SIZE - DRIFT_CHECK_INTERVAL
            end_idx = total_trades - DRIFT_CHECK_INTERVAL
            df_prev_100 = df_strat.iloc[start_idx:end_idx]
            
            if len(df_prev_100) == DRIFT_WINDOW_SIZE:
                prev_winning = len(df_prev_100[df_prev_100['PROFIT'] > 0])
                winrate_100_l20 = (prev_winning / DRIFT_WINDOW_SIZE) * 100
        
        # Avg Profit per trade (simpler than Avg_R)
        avg_profit_100 = df_last_100['PROFIT'].mean()
        
        # Determine status
        status = 'HEALTHY'
        counter = 0
        
        if winrate_100 < p50_wr:
            status = 'WARNING'
        
        if winrate_100 < p5_wr and avg_profit_100 < 0:
            counter = 1
            
            # Check if this is 2nd consecutive failure (on-the-fly calculation)
            if total_trades >= DRIFT_WINDOW_SIZE + DRIFT_CHECK_INTERVAL:
                # Get previous window
                start_idx = total_trades - DRIFT_WINDOW_SIZE - DRIFT_CHECK_INTERVAL
                end_idx = total_trades - DRIFT_CHECK_INTERVAL
                df_prev = df_strat.iloc[start_idx:end_idx]
                
                if len(df_prev) == DRIFT_WINDOW_SIZE:
                    prev_winning = len(df_prev[df_prev['PROFIT'] > 0])
                    prev_winrate = (prev_winning / DRIFT_WINDOW_SIZE) * 100
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
    
    FIXED: Now uses CLOSE execution data (order_price_close, order_ts_close, exec_ts_close)
           instead of OPEN data (which is lost on bot crashes).
    
    Args:
        df_trades: DataFrame with trades (columns: STRATEGY, order_price_close, PRICE_CLOSE, 
                   order_ts_close, exec_ts_close)
        strategies_config: List of strategy configurations
    
    Returns:
        Dict with execution quality per strategy:
        {
            'strategy_id': {
                'total_trades': 145,
                'avg_slippage_pct': 0.02,
                'slippage_status': 'OK' | 'WARNING' | 'CRITICAL' | 'NO_DATA',
                'avg_latency_sec': 0.8,
                'latency_status': 'OK' | 'WARNING' | 'CRITICAL' | 'NO_DATA'
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
                'avg_slippage_pct': None,
                'slippage_status': 'NO_DATA',
                'avg_latency_sec': None,
                'latency_status': 'NO_DATA'
            }
            continue
        
        total_trades = len(df_strat)
        
        # Get last EXECUTION_WINDOW_SIZE trades
        df_last = df_strat.tail(EXECUTION_WINDOW_SIZE)
        
        # Calculate slippage from CLOSE data (lowercase columns)
        avg_slippage_pct = None
        slippage_status = 'NO_DATA'
        
        if 'order_price_close' in df_last.columns and 'PRICE_CLOSE' in df_last.columns:
            df_with_slippage = df_last[
                df_last['order_price_close'].notna() & 
                df_last['PRICE_CLOSE'].notna()
            ].copy()
            
            if len(df_with_slippage) > 0:
                df_with_slippage['slippage_pct'] = (
                    (df_with_slippage['PRICE_CLOSE'] - df_with_slippage['order_price_close']) 
                    / df_with_slippage['order_price_close'] 
                    * 100
                )
                avg_slippage_pct = df_with_slippage['slippage_pct'].mean()
                
                # Determine status
                abs_slippage = abs(avg_slippage_pct)
                if abs_slippage < SLIPPAGE_WARNING_PCT:
                    slippage_status = 'HEALTHY'
                elif abs_slippage < SLIPPAGE_CRITICAL_PCT:
                    slippage_status = 'WARNING'
                else:
                    slippage_status = 'DANGER'
        
        # Calculate latency from CLOSE data (lowercase columns, timestamps in seconds with decimals)
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
            'avg_slippage_pct': round(avg_slippage_pct, 2) if avg_slippage_pct is not None else None,
            'slippage_status': slippage_status,
            'avg_latency_sec': round(avg_latency_sec, 3) if avg_latency_sec is not None else None,
            'latency_status': latency_status
        }
    
    return results