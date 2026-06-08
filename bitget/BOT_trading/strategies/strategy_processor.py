#BOT_trading/strategies/strategy_processor.py
"""
Handles strategy execution and order placement.

This module contains the StrategyProcessor class which coordinates:
- Signal detection (hardcoded or real)
- Order placement
- Position tracking
- Balance management
"""

import time
import logging
from typing import Dict, List, Callable, Any, Optional

logger = logging.getLogger('BOT_trading.strategies.processor')

from state import reset_strategy_candles
from .strategy_registry import detect_signals_for_strategy
from .hardcoded_signals import get_hardcoded_signals


class StrategyProcessor:


    def __init__(
        self,
        send_request_func: Callable,
        get_balance_func:  Callable,
        hour_zone,
        account_number:    str,
        state_file:        str,
        use_hardcoded:     bool = False,
        regime_enabled:    bool = True,
    ):
        self.send_request   = send_request_func
        self.get_balance    = get_balance_func
        self.hour_zone      = hour_zone
        self.account_number = account_number
        self.state_file     = state_file
        self.use_hardcoded  = use_hardcoded
        self.operative      = None
        self.regime_enabled = regime_enabled

        self._detect_real_signals   = detect_signals_for_strategy
        self._get_hardcoded_signals = get_hardcoded_signals

        logger.info(f"StrategyProcessor initialized (hardcoded mode: {use_hardcoded})")

    def detect_signals(
        self,
        strat:        Dict[str, Any],
        final_symbols: List[str],
        exchange,
    ) -> List[Dict]:
        """
        Detect signals for a strategy across all symbols.

        Returns:
            List of signal dicts: [{'symbol': ..., 'close': ..., 'regime': ..., 'timestamp': ...}]
        """
        strat_id = strat['id']

        if self.use_hardcoded:
            signals = self._get_hardcoded_signals(strat_id, self.send_request, self.hour_zone)
        else:
            signals = self._detect_real_signals(strat, final_symbols, None, regime_enabled=self.regime_enabled)

        logger.info(f"Signals detected {strat_id}: {len(signals)}")
        return signals

    def execute_signals(
        self,
        strat:            Dict[str, Any],
        signals:          List[Dict],
        open_positions:   Dict,
        strategy_candles: Dict,
        order_amount:     float,
    ) -> None:

        strat_id = strat['id']

        if not signals:
            return

        reset_strategy_candles(
            strat_id,
            strategy_candles,
            open_positions,
            self.account_number,
            self.state_file
        )

        for sig in signals:
            self.operative.place_order(
                symbol       = sig['symbol'],
                direction    = strat['direction'],
                usdt_amount  = order_amount,
                tp_pct       = strat['tp_pct'],
                sl_pct       = strat['sl_pct'],
                strategy_id  = strat_id,
                signal_close = sig.get('close', 0),
                regime       = sig.get('regime', 'unknown')
            )
            time.sleep(0.05)