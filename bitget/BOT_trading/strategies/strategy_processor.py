"""
Strategy Processor - Handles strategy execution and order placement.

This module contains the StrategyProcessor class which coordinates:
- Signal detection (hardcoded or real)
- Order placement
- Position tracking
- Balance management

MODIFICATION: Added support for regime-based position sizing via adjusted_order_amount parameter
"""

import time
import logging
from decimal import Decimal
from typing import Dict, List, Callable, Any, Optional

# Setup module logger
logger = logging.getLogger('BOT_trading.strategies.processor')

# Import from execution module
from execution import place_order,add_position,get_fills_for_order

# Import from state module
from state import reset_strategy_candles

# Import signal detection functions
from .strategy_registry import detect_signals_for_strategy
from .hardcoded_signals import get_hardcoded_signals


class StrategyProcessor:
    """
    Processes trading strategies: detects signals and executes orders.
    
    This class maintains the exact same logic as the original process_strategy
    function but with cleaner parameter handling through initialization.
    
    Attributes:
        send_request (Callable): Function to send REST API requests
        get_balance (Callable): Function to get USDT balance
        hour_zone: Timezone object for timestamps
        state_file (str): Path to state file
        use_hardcoded (bool): Whether to use hardcoded signals for testing
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
        """
        Initialize the StrategyProcessor.
        
        Args:
            send_request_func: Function to send REST API requests
            get_balance_func: Function to get USDT balance
            hour_zone: Timezone object for timestamps
            state_file: Path to state file for persistence
            use_hardcoded: Whether to use hardcoded signals (testing mode)
        """
        self.send_request = send_request_func
        self.get_balance = get_balance_func
        self.hour_zone = hour_zone
        self.account_number = account_number
        self.state_file = state_file
        self.use_hardcoded = use_hardcoded
        self.demo_operative = None
        
        # Store references to signal detection functions
        self._detect_real_signals = detect_signals_for_strategy
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
        direction: str = 'unknown',                    # ← NUEVO
        direction_multiplier: float = 1.0              # ← NUEVO
    ) -> None:
        """
        Process a strategy: detect signals and place orders if needed.
        
        This method maintains CLONED logic from original process_strategy function.
        
        Args:
            strat: Strategy configuration dictionary containing:
                - id: Strategy identifier
                - name: Strategy name
                - direction: 'long' or 'short'
                - order_amount: USDT amount per order
                - tp_pct: Take profit percentage
                - sl_pct: Stop loss percentage
            final_symbols: List of symbols to process
            exchange: Exchange connection (for balance check)
            open_positions: Dictionary of open positions by strategy
            strategy_candles: Dictionary of candle counters by strategy
            adjusted_order_amount: Optional regime-adjusted order amount.
                                  If None, uses strat['order_amount']
        
        Returns:
            None
        """
        strat_id = strat['id']
        
        # ====================================================================
        # REGIME-BASED POSITION SIZING (NEW)
        # ====================================================================
        # Use adjusted amount if provided, otherwise use config amount
        order_amount = adjusted_order_amount if adjusted_order_amount is not None else strat['order_amount']
        
        # Log if regime adjustment is active
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
            signals = self._get_hardcoded_signals(
                strat_id,
                self.send_request,
                self.hour_zone
            )
        else:
            logger.debug(f"Using real signal detection for {strat_id}")
            signals = self._detect_real_signals(strat, final_symbols, None)
        
        logger.info(f"Signals detected {strat_id}: {len(signals)}")
        
        if not signals:
            logger.info(f"No signals for {strat_id}, returning")
            return
        
        # Reset candle counter when opening new positions 
        reset_strategy_candles(
            strat_id,
            strategy_candles,
            open_positions,
            self.account_number,  # ← NECESITA PASAR account_number
            self.state_file
        )
        
        # ====================================================================
        # DEMO MODE: Skip balance checks and use simulated orders
        # ====================================================================
        if hasattr(self, 'demo_operative') and self.demo_operative:
            for sig in signals:
                self.demo_operative.place_simulated_order(
                    symbol=sig['symbol'],
                    direction=strat['direction'],
                    usdt_amount=order_amount,
                    tp_pct=strat['tp_pct'],
                    sl_pct=strat['sl_pct'],
                    strategy_id=strat_id
                )
                time.sleep(0.05)
            return  # Exit early, don't process real orders
        
        # ====================================================================
        # PROCESS ALL SIGNALS 
        # ====================================================================
        for sig in signals:
            # Check balance before each order - USE ADJUSTED AMOUNT
            usdt_balance = self.get_balance(exchange)
            logger.debug(f"Current balance: {usdt_balance:.2f} USDT")
            
            if usdt_balance < order_amount:
                logger.warning(
                    f"WAR-Insufficient balance ({usdt_balance:.2f} USDT) for {sig['symbol']} "
                    f"(needs {order_amount:.2f} USDT)"
                )
                continue
            
            # Place order - USE ADJUSTED AMOUNT
            logger.debug(f"Placing order for {sig['symbol']} with ${order_amount:.2f}")
            order_result = place_order(
                symbol=sig['symbol'],
                direction=strat['direction'],
                usdt_amount=order_amount,  
                send_request_func=self.send_request
            )
            
            # =========================================================================
            # BLOQUE 2: NUEVO - EXTRAER DATOS DEL DICT RETORNADO
            # =========================================================================
            if order_result is None:
                logger.error(f"Error-Failed to place order for {sig['symbol']}")
                continue
            
            # Extraer datos de ejecución
            resp_order = order_result.get('resp_order')  # Dict original del broker

            
            # Validar que resp_order existe
            if resp_order is None:
                logger.error(f"Error-Invalid order result for {sig['symbol']}")
                continue
            
            # Extract order data - CLONED LOGIC
            data     = resp_order.get('data', {}) if isinstance(resp_order, dict) else {}
            order_id = data.get('orderId')
            
            if order_id:
                logger.debug(f"Order placed successfully: {order_id}")
                
                # Get fills for accurate entry price 
                filled_size, entry_price_from_fills, _, _ = get_fills_for_order(
                    order_id=order_id,
                    symbol=sig['symbol'],
                    send_request_func=self.send_request
                )
                time.sleep(0.05)
                
                # Determine size and entry price 
                if filled_size is None or filled_size == 0:
                    size        = Decimal(str(data.get('size', data.get('filledQty', data.get('baseVolume', 0)))))
                    entry_price = Decimal(str(data.get('price', sig.get('close', 0))))
                    logger.debug(f"Using order data: size={size}, price={entry_price}")
                else:
                    size = filled_size
                    entry_price = entry_price_from_fills if entry_price_from_fills is not None else Decimal(str(sig.get('close', 0)))
                    logger.debug(f"Using fills data: size={size}, price={entry_price}")
                
                # Add position to tracking - USE ADJUSTED AMOUNT
                add_position(
                    strat_id=strat_id,
                    symbol=sig['symbol'],
                    size=size,
                    entry_price=entry_price,
                    direction=strat['direction'],
                    tp_pct=strat['tp_pct'],
                    sl_pct=strat['sl_pct'],
                    order_id=order_id,
                    open_positions=open_positions,
                    strategy_candles=strategy_candles,
                    account_number=self.account_number,  # ← NECESITA PASAR account_number
                    state_file=self.state_file,
                    hour_zone=self.hour_zone,
                    usdt_amount=order_amount,
                    regime_family=regime_family,
                    regime_multiplier=regime_multiplier,
                    market_direction=direction,
                    direction_multiplier=direction_multiplier
                )
                logger.debug(f"Position added to tracking: {sig['symbol']}")
            else:
                logger.warning("Order executed but no orderId in response")
            
            time.sleep(0.05)