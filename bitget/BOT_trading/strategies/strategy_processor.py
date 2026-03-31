"""
strategies/strategy_processor.py - Handles strategy execution and order placement.

This module contains the StrategyProcessor class which coordinates:
- Signal detection (hardcoded or real)
- Order placement
- Position tracking
- Balance management

MODIFICATION: Added support for regime-based position sizing via adjusted_order_amount parameter
"""

import time
import logging
from typing import Dict, List, Callable, Any, Optional

# Setup module logger
logger = logging.getLogger('BOT_trading.strategies.processor')

# Import from state module
from state import reset_strategy_candles

# Import signal detection functions
from .strategy_registry import detect_signals_for_strategy
from .hardcoded_signals import get_hardcoded_signals


class StrategyProcessor:
    """
    Processes trading strategies: detects signals and executes orders.
    
    Attributes:
        send_request (Callable): Function to send REST API requests
        get_balance (Callable): Function to get USDT balance
        hour_zone: Timezone object for timestamps
        state_file (str): Path to state file
        use_hardcoded (bool): Whether to use hardcoded signals for testing
        operative: Operative mode instance (DemoOperative or ProductionOperative)
    """
    
    def __init__(
        self,
        send_request_func: Callable,
        get_balance_func: Callable,
        hour_zone,
        account_number: str,
        state_file: str,
        use_hardcoded: bool = False
    ):
        self.send_request  = send_request_func
        self.get_balance   = get_balance_func
        self.hour_zone     = hour_zone
        self.account_number = account_number
        self.state_file    = state_file
        self.use_hardcoded = use_hardcoded
        self.operative     = None

        self._detect_real_signals  = detect_signals_for_strategy
        self._get_hardcoded_signals = get_hardcoded_signals

        logger.info(f"StrategyProcessor initialized (hardcoded mode: {use_hardcoded})")

    def process(
        self,
        strat: Dict[str, Any],
        final_symbols: List[str],
        exchange,
        open_positions: Dict,
        strategy_candles: Dict,
        adjusted_order_amount: Optional[float] = None,
        regime_family: str = 'unknown',
        regime_multiplier: float = 1.0,
        direction: str = 'unknown',
        direction_multiplier: float = 1.0
    ) -> None:

        strat_id = strat['id']

        # ====================================================================
        # REGIME-BASED POSITION SIZING
        # ====================================================================
        order_amount = adjusted_order_amount if adjusted_order_amount is not None else strat['order_amount']

        if adjusted_order_amount is not None and adjusted_order_amount != strat['order_amount']:
            logger.debug(
                f"[REGIME] {strat_id}: Base=${strat['order_amount']:.2f} → "
                f"Adjusted=${order_amount:.2f} (multiplier={order_amount/strat['order_amount']:.2f}x)"
            )

        # ====================================================================
        # SIGNAL DETECTION
        # ====================================================================
        if self.use_hardcoded:
            logger.debug(f"Using hardcoded signals for {strat_id}")
            signals = self._get_hardcoded_signals(strat_id, self.send_request, self.hour_zone)
        else:
            logger.debug(f"Using real signal detection for {strat_id}")
            signals = self._detect_real_signals(strat, final_symbols, None)

        logger.info(f"Signals detected {strat_id}: {len(signals)}")

        if not signals:
            logger.info(f"No signals for {strat_id}, returning")
            return

        reset_strategy_candles(
            strat_id,
            strategy_candles,
            open_positions,
            self.account_number,
            self.state_file
        )

        # ====================================================================
        # OPERATIVE: Delegate order placement
        # ====================================================================
        for sig in signals:
            self.operative.place_order(
                symbol=sig['symbol'],
                direction=strat['direction'],
                usdt_amount=order_amount,
                tp_pct=strat['tp_pct'],
                sl_pct=strat['sl_pct'],
                strategy_id=strat_id,
                regime_family=regime_family,
                regime_multiplier=regime_multiplier,
                market_direction=direction,
                direction_multiplier=direction_multiplier,
                signal_close=sig.get('close', 0)
            )
            time.sleep(0.05)