#BOT_trading/strategies/strategy_registry.py
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
# PATH SETUP 
#===========================================================================

current_dir = os.path.dirname(os.path.abspath(__file__))  # strategies/
bot_dir     = os.path.dirname(current_dir)                  # BOT_trading/
bitget_dir  = os.path.dirname(bot_dir)                      # bitget/

if bitget_dir not in sys.path:
    sys.path.insert(0, bitget_dir)

#===========================================================================
# IMPORTS - Signal generation functions 
#===========================================================================

from signals.add_signals_reversal import reversal_long, reversal_short
from signals.add_signals_parity import parity_long, parity_short
from signals.add_signals_orderblocks import orderblocks_long, orderblocks_short
from signals.add_signals_flag import flag_long, flag_short


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

        >>> signals = detect_signals_for_strategy(strat, ['BTCUSDT', 'ETHUSDT'], None)
    """
    strategy_id = strat['id']
    timeframe   = strat['timeframe']
    
    logger.info(f"Processing strategy: {strategy_id}")
    logger.info("-" * 48)
    
    # Validate symbols
    if not final_symbols:
        logger.error(f"No symbols to process for {strategy_id}")
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
        #DEBUG
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"[CANDLE_CHECK] {symbol} {timeframe} — last 50 candles:\n{df_norm[['open','high','low','close','volume_quote']].tail(50).to_string()}")
            logger.debug(df_norm[['open','high','low','close','volume_quote']].tail(50).to_string())
        
        try:
            signals = None
            
            # ==============================================================
            # STRATEGY IMPLEMENTATIONS - Add elif for each new strategy
            # ==============================================================

            if strategy_id == '01_reversal_long_15m':
                signals = reversal_long(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '02_reversal_short_15m':
                signals = reversal_short(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '03_reversal_long_30m':
                signals = reversal_long(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '04_reversal_short_30m':
                signals = reversal_short(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '05_reversal_long_1H':
                signals = reversal_long(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '06_reversal_short_1H':
                signals = reversal_short(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '07_reversal_long_4H':
                signals = reversal_long(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '08_reversal_short_4H':
                signals = reversal_short(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '09_reversal_long_6Hutc':
                signals = reversal_long(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '10_reversal_short_6Hutc':
                signals = reversal_short(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '11_parity_long_15m':
                signals = parity_long(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '12_parity_short_15m':
                signals = parity_short(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '13_parity_long_30m':
                signals = parity_long(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '14_parity_short_30m':
                signals = parity_short(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '15_parity_long_1H':
                signals = parity_long(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '16_parity_short_1H':
                signals = parity_short(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '17_parity_long_4H':
                signals = parity_long(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '18_parity_short_4H':
                signals = parity_short(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '19_parity_long_6Hutc':
                signals = parity_long(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '20_parity_short_6Hutc':
                signals = parity_short(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '21_flag_long_15m':
                signals = flag_long(
                    arr,
                    lookback=strat['lookback'],
                    impulse=strat['impulse'],
                    flag=strat['flag'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '22_flag_short_15m':
                signals = flag_short(
                    arr,
                    lookback=strat['lookback'],
                    impulse=strat['impulse'],
                    flag=strat['flag'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '23_flag_long_30m':
                signals = flag_long(
                    arr,
                    lookback=strat['lookback'],
                    impulse=strat['impulse'],
                    flag=strat['flag'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '24_flag_short_30m':
                signals = flag_short(
                    arr,
                    lookback=strat['lookback'],
                    impulse=strat['impulse'],
                    flag=strat['flag'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '25_flag_long_1H':
                signals = flag_long(
                    arr,
                    lookback=strat['lookback'],
                    impulse=strat['impulse'],
                    flag=strat['flag'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '26_flag_short_1H':
                signals = flag_short(
                    arr,
                    lookback=strat['lookback'],
                    impulse=strat['impulse'],
                    flag=strat['flag'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '27_flag_long_4H':
                signals = flag_long(
                    arr,
                    lookback=strat['lookback'],
                    impulse=strat['impulse'],
                    flag=strat['flag'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '28_flag_short_4H':
                signals = flag_short(
                    arr,
                    lookback=strat['lookback'],
                    impulse=strat['impulse'],
                    flag=strat['flag'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '29_flag_long_6Hutc':
                signals = flag_long(
                    arr,
                    lookback=strat['lookback'],
                    impulse=strat['impulse'],
                    flag=strat['flag'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '30_flag_short_6Hutc':
                signals = flag_short(
                    arr,
                    lookback=strat['lookback'],
                    impulse=strat['impulse'],
                    flag=strat['flag'],
                    ma_period=strat['ma_period'],
                    live_trading=True
                )

            elif strategy_id == '31_orderblocks_long_15m':
                signals = orderblocks_long(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    impulse=strat['impulse'],
                    live_trading=True
                )

            elif strategy_id == '32_orderblocks_short_15m':
                signals = orderblocks_short(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    impulse=strat['impulse'],
                    live_trading=True
                )

            elif strategy_id == '33_orderblocks_long_30m':
                signals = orderblocks_long(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    impulse=strat['impulse'],
                    live_trading=True
                )

            elif strategy_id == '34_orderblocks_short_30m':
                signals = orderblocks_short(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    impulse=strat['impulse'],
                    live_trading=True
                )

            elif strategy_id == '35_orderblocks_long_1H':
                signals = orderblocks_long(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    impulse=strat['impulse'],
                    live_trading=True
                )

            elif strategy_id == '36_orderblocks_short_1H':
                signals = orderblocks_short(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    impulse=strat['impulse'],
                    live_trading=True
                )

            elif strategy_id == '37_orderblocks_long_4H':
                signals = orderblocks_long(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    impulse=strat['impulse'],
                    live_trading=True
                )

            elif strategy_id == '38_orderblocks_short_4H':
                signals = orderblocks_short(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    impulse=strat['impulse'],
                    live_trading=True
                )

            elif strategy_id == '39_orderblocks_long_6Hutc':
                signals = orderblocks_long(
                    arr,
                    lookback=strat['lookback'],
                    tolerance=strat['tolerance'],
                    impulse=strat['impulse'],
                    live_trading=True
                )

            elif strategy_id == '40_orderblocks_short_6Hutc':
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
                    f"WAR-Strategy '{strategy_id}' not implemented in registry. "
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
                f"Error detecting signals for {symbol} ({strategy_id}): {e}"
            )
            continue
    
    logger.debug(f"Signals detected {strategy_id}: {len(all_signals)}")
    
    return all_signals


# ==============================================================
# IMPLEMENTED STRATEGIES - For validation
# ==============================================================

def get_implemented_strategies() -> set:
    """
    Returns set of all implemented strategy IDs.
    
    This is used by validation system to ensure strategies in YAML
    are actually implemented.
    
    IMPORTANT: When adding a new strategy, add its ID here!
    """
    strategies = {
       #'01_reversal_long_15m',
       #'02_reversal_short_15m',
       #'03_reversal_long_30m',
       #'04_reversal_short_30m',
       #'05_reversal_long_1H',
       #'06_reversal_short_1H',
       #'07_reversal_long_4H',
       #'08_reversal_short_4H',
       #'09_reversal_long_6Hutc',
       #'10_reversal_short_6Hutc',
       #'11_parity_long_15m',
       #'12_parity_short_15m',
       #'13_parity_long_30m',
       #'14_parity_short_30m',
       #'15_parity_long_1H',
       #'16_parity_short_1H',
       #'17_parity_long_4H',
       #'18_parity_short_4H',
       #'19_parity_long_6Hutc',
       #'20_parity_short_6Hutc',
       #'21_flag_long_15m',
       #'22_flag_short_15m',
       #'23_flag_long_30m',
       #'24_flag_short_30m',
       #'25_flag_long_1H',
       #'26_flag_short_1H',
       #'27_flag_long_4H',
       #'28_flag_short_4H',
       #'29_flag_long_6Hutc',
       #'30_flag_short_6Hutc',
       #'31_orderblocks_long_15m',
       #'32_orderblocks_short_15m',
       #'33_orderblocks_long_30m',
       #'34_orderblocks_short_30m',
       #'35_orderblocks_long_1H',
       #'36_orderblocks_short_1H',
       #'37_orderblocks_long_4H',
       #'38_orderblocks_short_4H',
       #'39_orderblocks_long_6Hutc',
       #'40_orderblocks_short_6Hutc',
    }
    return strategies

# Create constant for backward compatibility
IMPLEMENTED_STRATEGIES = get_implemented_strategies()