"""
Signal Detector - Detects real trading signals for strategies.

This module contains the logic for detecting signals from actual market data
using the strategy-specific signal generation functions.
"""
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))  
bot_dir     = os.path.dirname(current_dir)                    
scripts_dir = os.path.dirname(bot_dir)                    

if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)
    
import logging
logger = logging.getLogger('BOT_trading.execution.signal_detector')

from market_data import fetch_ohlcv_data, normalize_live_ohlcv, df_to_arrays_live


def detect_signals_for_strategy(
    strat: dict,
    final_symbols: list,
    exchange,
    use_hardcoded: bool = False
):
    """
    Generic signal detector - works with any strategy using registry.
    """
    from .registry import get_strategy_function
    
    strategy_name = strat['name']
    timeframe = strat['timeframe']
    
    logger.info(f"Processing strategy: {strat['id']}")
    logger.info("-" * 48)
    
    # Get strategy function from registry
    try:
        strategy_func = get_strategy_function(strategy_name)
    except KeyError as e:
        logger.error(str(e))
        return []
    
    # Fetch OHLCV data
    ohlcv_data = fetch_ohlcv_data(final_symbols, timeframe)
    
    all_signals = []
    
    for symbol, df in ohlcv_data.items():
        if df is None or df.empty:
            continue
        
        # Normalize data
        df_norm = normalize_live_ohlcv(df)
        arr = df_to_arrays_live(df_norm)
        
        # Extract parameters for this strategy
        params = extract_strategy_params(strat)
        
        # Call strategy function with extracted params
        try:
            signals = strategy_func(arr, **params)
        except TypeError as e:
            logger.error(f"Error calling {strategy_name} for {symbol}: {e}")
            logger.debug(f"Params passed: {params}")
            continue
        
        # ✅ FIX: Verificar señal en última vela
        if signals is not None and len(signals) > 0 and signals[-1] != 0:
            all_signals.append({
                'symbol': symbol,
                'close': float(arr['close'][-1])  # Precio de cierre necesario
            })
    
    logger.info(f"Signals detected  {strat['id']}: {len(all_signals)}")
    
    return all_signals


def extract_strategy_params(strat: dict) -> dict:
    """
    Extract relevant parameters for strategy function from config dict.
    
    Args:
        strat: Strategy configuration dict
    
    Returns:
        Dict of parameters to pass to strategy function
    """
    params = {}
    
    # List of parameter names that strategies use
    param_names = [
        'lookback',      # Usado por la mayoría
        'tolerance',     # Usado por todas
        'trend_th',      # Usado por double_top
        'ma_period',     # Usado por reversal/parity
        'impulse',       # Usado por orderblocks
    ]
    
    # Extract all parameters that exist in config
    for param_name in param_names:
        if param_name in strat:
            params[param_name] = strat[param_name]
    
    return params
