"""
Timeframe utilities - Functions for timeframe calculations.

This module provides utilities for:
- Calculating next candle close times
- Grouping strategies by timeframe
- Getting unique timeframes from strategy list

This module has no dependencies on other bot modules.
"""

from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict


# ==========================================================================
# TIMEFRAME CALCULATIONS
# ==========================================================================
def calculate_next_candle_time(timeframe: str = '4H', hour_zone=None) -> datetime:
    """
    Calculate the next candle close time for a given timeframe.
    
    This function supports standard timeframes (m, H) and UTC-based
    timeframes (Hutc, Dutc).
    
    Args:
        timeframe: Timeframe string (e.g., '15m', '4H', '6Hutc', '1Dutc')
        hour_zone: Timezone object (uses UTC if None)
    
    Returns:
        Datetime of next candle close (with 45 second buffer)
    
    Raises:
        ValueError: If timeframe format is invalid
    
    Examples:
        >>> from zoneinfo import ZoneInfo
        >>> calculate_next_candle_time('4H', ZoneInfo('UTC'))
        datetime.datetime(2026, 1, 3, 16, 0, 45, tzinfo=...)
        
        >>> calculate_next_candle_time('15m')
        # Returns next 15-minute candle close time
        
        >>> calculate_next_candle_time('6Hutc')
        # Returns next 6-hour UTC-aligned candle close
    """
    now = datetime.now(hour_zone)
    
    # Parse timeframe
    if timeframe.endswith('Hutc'):
        hours = int(timeframe[:-4])
        minutes = hours * 60
    elif timeframe.endswith('H'):
        hours = int(timeframe[:-1])
        minutes = hours * 60
    elif timeframe.endswith('m'):
        minutes = int(timeframe[:-1])
    elif timeframe.endswith('Dutc'):
        days = int(timeframe[:-4])
        minutes = days * 24 * 60
    else:
        raise ValueError(
            "Invalid timeframe. Use 'm', 'H', 'Hutc', or 'Dutc'. "
            "Examples: '15m', '4H', '6Hutc', '1Dutc'"
        )
    
    # Calculate next candle time
    total_minutes = now.hour * 60 + now.minute
    next_total_minutes = ((total_minutes // minutes) + 1) * minutes
    delta_minutes = next_total_minutes - total_minutes
    
    next_candle = now + timedelta(
        minutes=delta_minutes,
        seconds=-now.second,
        microseconds=-now.microsecond
    )
    
    # Add 45 second buffer to ensure candle is fully closed
    next_candle = next_candle + timedelta(seconds=45)
    
    return next_candle


# ==========================================================================
# STRATEGY GROUPING
# ==========================================================================
def group_strategies_by_timeframe(strategies: List[Dict]) -> Dict[str, List]:
    """
    Group strategies by their timeframe.
    
    Args:
        strategies: List of strategy configuration dictionaries
    
    Returns:
        Dictionary mapping timeframe -> list of strategies
    
    Example:
        >>> strategies = [
        ...     {'id': 'strat1', 'timeframe': '4H'},
        ...     {'id': 'strat2', 'timeframe': '4H'},
        ...     {'id': 'strat3', 'timeframe': '1H'}
        ... ]
        >>> grouped = group_strategies_by_timeframe(strategies)
        >>> print(grouped)
        {
            '4H': [
                {'id': 'strat1', 'timeframe': '4H'},
                {'id': 'strat2', 'timeframe': '4H'}
            ],
            '1H': [
                {'id': 'strat3', 'timeframe': '1H'}
            ]
        }
    """
    grouped = defaultdict(list)
    for strat in strategies:
        grouped[strat['timeframe']].append(strat)
    return grouped


def get_unique_timeframes(strategies: List[Dict]) -> List[str]:
    """
    Get list of unique timeframes from strategies.
    
    Args:
        strategies: List of strategy configuration dictionaries
    
    Returns:
        Sorted list of unique timeframe strings
    
    Example:
        >>> strategies = [
        ...     {'timeframe': '4H'},
        ...     {'timeframe': '1H'},
        ...     {'timeframe': '4H'}
        ... ]
        >>> timeframes = get_unique_timeframes(strategies)
        >>> print(timeframes)
        ['1H', '4H']
    """
    return sorted(list(set(s['timeframe'] for s in strategies)))
