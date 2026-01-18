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

    # Symbols to generate signals for
    symbols = ['BTCUSDT', 'BNBUSDT']
    signals = []
    
    for symbol in symbols:
        # Fetch current price from API
        code, resp = send_request_func("GET","/api/v2/mix/market/ticker",params={"productType": product_type, "symbol": symbol})
        
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
