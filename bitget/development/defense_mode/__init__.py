"""
flip_control - Flip detection and partial closing simulation

Detects regime flips in BTC and simulates partial position closing
to improve portfolio performance during regime transitions.

Usage:
    python -m flip_control.flip_simulator
    
    Or:
    from flip_control.flip_simulator import run_simulation
    run_simulation()
"""

from .flip_detector import detect_flips, load_btc_ohlc, get_regime_at_time
from .flip_simulator import run_simulation

__all__ = [
    'detect_flips',
    'load_btc_ohlc',
    'get_regime_at_time',
    'run_simulation'
]