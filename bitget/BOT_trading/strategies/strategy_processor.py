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
    """
    Processes trading strategies: detects signals and executes orders.

    Attributes:
        send_request   : Function to send REST API requests
        get_balance    : Function to get USDT balance
        hour_zone      : Timezone object for timestamps
        account_number : Account identifier
        state_file     : Path to state file
        use_hardcoded  : Whether to use hardcoded signals for testing
        operative      : Operative mode instance
    """

    def __init__(
        self,
        send_request_func: Callable,
        get_balance_func:  Callable,
        hour_zone,
        account_number:    str,
        state_file:        str,
        use_hardcoded:     bool = False,
    ):
        self.send_request   = send_request_func
        self.get_balance    = get_balance_func
        self.hour_zone      = hour_zone
        self.account_number = account_number
        self.state_file     = state_file
        self.use_hardcoded  = use_hardcoded
        self.operative      = None

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
            signals = self._detect_real_signals(strat, final_symbols, None)

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
        """
        Execute orders for a list of approved signals.

        Args:
            strat            : Strategy config dict
            signals          : Approved signals from orchestrator (already regime-filtered)
            open_positions   : Current open positions
            strategy_candles : Candle counters
            order_amount     : Final order amount to use
        """
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
                signal_close = sig.get('close', 0)
            )
            time.sleep(0.05)