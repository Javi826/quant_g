"""
Hardcoded Signals - Generate test signals for testing and debugging.

This module provides hardcoded signal generation for testing the bot
without relying on real market signal detection.
"""

from typing import Callable, List, Dict
from datetime import datetime

import logging
logger = logging.getLogger('BOT_trading.execution.hardcoded_signals')


def get_hardcoded_signals(
    strat_id: str,
    send_request_func: Callable,
    hour_zone,
    product_type: str = 'USDT-FUTURES'
) -> List[Dict]:
    """
    Generate hardcoded test signals for testing.
    
    This function fetches real prices from the API but returns them as
    hardcoded signals for all specified test symbols.
    
    Args:
        strat_id: Strategy identifier (unused but kept for compatibility)
        send_request_func: Function to send REST API requests
        hour_zone: Timezone object for timestamps
        product_type: Product type for the API request
    
    Returns:
        List of signal dictionaries, each containing:
            - symbol: Trading symbol
            - close: Current price from API
            - timestamp: ISO format timestamp
    
    Example:
        >>> signals = get_hardcoded_signals('01_double_top_long_2m', send_func, utc_zone)
        >>> print(signals)
        [
            {'symbol': 'BTCUSDT', 'close': 96543.2, 'timestamp': '2026-01-03T17:30:45'},
            {'symbol': 'BNBUSDT', 'close': 645.8, 'timestamp': '2026-01-03T17:30:45'}
        ]
    """
    # Symbols to generate signals for
    symbols = ['BTCUSDT', 'BNBUSDT']
    signals = []
    
    for symbol in symbols:
        # Fetch current price from API
        code, resp = send_request_func(
            "GET",
            "/api/v2/mix/market/ticker",
            params={"productType": product_type, "symbol": symbol}
        )
        
        # Default fallback price
        current_price = 50000.0
        
        # Extract price from API response
        if code == 200 and isinstance(resp, dict) and resp.get("code") == "00000":
            try:
                current_price = float(resp['data'][0]['lastPr'])
            except Exception:
                pass
        
        # Create signal dictionary
        signals.append({
            'symbol': symbol,
            'close': current_price,
            'timestamp': datetime.now(hour_zone).isoformat()
        })
    
    return signals
