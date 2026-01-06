"""
Strategy Registry - Explicit strategy implementation with elif structure.

This module provides:
- detect_signals_for_strategy(): Main signal detection function
- get_implemented_strategies(): Returns set of implemented strategies
- IMPLEMENTED_STRATEGIES: Set of all implemented strategy names

This is the SINGLE SOURCE OF TRUTH for strategy implementations.

To add a new strategy:
1. Add entry in strategies.yaml
2. Add elif in detect_signals_for_strategy() below
3. Add strategy name to get_implemented_strategies()
"""

import sys
import os
import logging

# Setup logger
logger = logging.getLogger('BOT_trading.strategies.registry')

#===========================================================================
# PATH SETUP (same as old registry.py)
#===========================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))  
bot_dir     = os.path.dirname(current_dir)                    
scripts_dir = os.path.dirname(bot_dir)

if scripts_dir not in sys.path:
    sys.path.insert(0, scripts_dir)

#===========================================================================
# IMPORTS - Signal generation functions from Z_add_signals_*.py
#===========================================================================

from Z_add_signals_double_top import double_top_long
from Z_add_signals_reversal import reversal_long, reversal_short
from Z_add_signals_parity import parity_long, parity_short
from Z_add_signals_orderblocks import orderblocks_long, orderblocks_short

# Import market data utilities
from market_data import fetch_ohlcv_data, normalize_live_ohlcv, df_to_arrays_live


#===========================================================================
# MAIN SIGNAL DETECTION FUNCTION
#===========================================================================

def detect_signals_for_strategy(
    strat: dict,
    final_symbols: list,
    exchange,
    use_hardcoded: bool = False
) -> list:
    """
    Detect trading signals for a strategy across multiple symbols.
    
    This function maintains EXACT SAME LOGIC as the old signal_detector.py
    but with explicit elif structure instead of registry lookup.
    
    Args:
        strat: Strategy configuration dictionary from YAML containing:
            - id: Strategy identifier
            - name: Strategy name (e.g., 'reversal_long_4H')
            - timeframe: Timeframe (e.g., '4H', '1H', '6Hutc')
            - lookback: Lookback period
            - tolerance: Price tolerance
            - ... (strategy-specific parameters)
        final_symbols: List of symbols to analyze
        exchange: Exchange connection (not used, kept for compatibility)
        use_hardcoded: Whether to use hardcoded signals (not used here)
    
    Returns:
        List of detected signals:
        [
            {'symbol': 'BTCUSDT', 'close': 91167.7, 'timestamp': ...},
            {'symbol': 'ETHUSDT', 'close': 3245.2, 'timestamp': ...},
            ...
        ]
    
    Example:
        >>> strat = {
        ...     'id': '02_reversal_long_4H',
        ...     'name': 'reversal_long_4H',
        ...     'timeframe': '4H',
        ...     'lookback': 4,
        ...     'tolerance': 20,
        ...     'ma_period': 50
        ... }
        >>> signals = detect_signals_for_strategy(strat, ['BTCUSDT', 'ETHUSDT'], None)
    """
    strategy_name = strat['name']
    timeframe = strat['timeframe']
    
    logger.info(f"Processing strategy: {strat['id']}")
    logger.info("-" * 48)
    
    # Validate symbols
    if not final_symbols:
        logger.warning(f"No symbols to process for {strat['id']}")
        return []
    
    # Fetch OHLCV data for all symbols
    ohlcv_data  = fetch_ohlcv_data(final_symbols, timeframe)    
    all_signals = []
    
    # Process each symbol
    for symbol, df in ohlcv_data.items():
        if df is None or df.empty:
            continue
        
        # Normalize data
        df_norm = normalize_live_ohlcv(df)
        arr     = df_to_arrays_live(df_norm)
        
        try:
            signals = None
            
            # ==============================================================
            # STRATEGY IMPLEMENTATIONS - Add elif for each new strategy
            # ==============================================================
            
            if strategy_name == 'double_top_long_4H':
                signals = double_top_long(
                    arr,
                    lookback_minor=strat['lookback'],
                    price_tolerance=strat['tolerance'],
                    trend_th=strat['trend_th'],
                    live_trading=True
                )
            
            elif strategy_name == 'reversal_long_4H':
                signals = reversal_long(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )
            
            elif strategy_name == 'reversal_long_1H':
                signals = reversal_long(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )
            
            elif strategy_name == 'reversal_long_6Hutc':
                signals = reversal_long(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )
            
            elif strategy_name == 'reversal_short_4H':
                signals = reversal_short(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )
            
            elif strategy_name == 'reversal_short_1H':
                signals = reversal_short(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )
            
            elif strategy_name == 'reversal_short_6Hutc':
                signals = reversal_short(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )
            
            elif strategy_name == 'parity_long_4H':
                signals = parity_long(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )
            
            elif strategy_name == 'parity_long_1H':
                signals = parity_long(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )
            
            elif strategy_name == 'parity_long_6Hutc':
                signals = parity_long(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )
            
            elif strategy_name == 'parity_short_4H':
                signals = parity_short(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )
            
            elif strategy_name == 'parity_short_1H':
                signals = parity_short(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )
            
            elif strategy_name == 'orderblocks_long_4H':
                signals = orderblocks_long(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    impulse=strat['impulse'],
                    live_trading=True
                )
            
            elif strategy_name == 'orderblocks_short_4H':
                signals = orderblocks_short(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    impulse=strat['impulse'],
                    live_trading=True
                )
            
            # ==============================================================
            # STRATEGY NOT IMPLEMENTED
            # ==============================================================
            else:
                logger.warning(
                    f"WAR-Strategy '{strategy_name}' not implemented in registry. "
                    f"WAR-Add elif in strategies/strategy_registry.py"
                )
                continue
            
            # ==============================================================
            # VERIFY SIGNAL
            # ==============================================================
            if signals is None or len(signals) == 0:
                continue
            
            # Check if last candle has signal
            if signals[-1] != 0:
                last_row = df_norm.iloc[-1]
                all_signals.append({
                    'symbol': symbol,
                    'timestamp': last_row.name if 'timestamp' not in df_norm.columns else last_row['timestamp'],
                    'close': float(arr['close'][-1])
                })
        
        except Exception as e:
            logger.error(
                f"Error detecting signals for {symbol} ({strategy_name}): {e}"
            )
            continue
    
    logger.debug(f"Signals detected {strat['id']}: {len(all_signals)}")
    
    return all_signals


# ==============================================================
# IMPLEMENTED STRATEGIES - For validation
# ==============================================================

def get_implemented_strategies() -> set:
    """
    Returns set of all implemented strategy names.
    
    This is used by validation system to ensure strategies in YAML
    are actually implemented.
    
    IMPORTANT: When adding a new strategy, add its name here!
    """
    strategies = {
        'double_top_long_4H',
        'reversal_long_4H',
        'reversal_short_4H',
        'reversal_long_1H',
        'reversal_short_1H',
        'reversal_long_6Hutc',
        'reversal_short_6Hutc',
        'parity_long_4H',
        'parity_short_4H',
        'parity_long_1H',
        'parity_short_1H',
        'parity_long_6Hutc',
        'orderblocks_long_4H',
        'orderblocks_short_4H',
    }
    return strategies


# Create constant for backward compatibility
IMPLEMENTED_STRATEGIES = get_implemented_strategies()

