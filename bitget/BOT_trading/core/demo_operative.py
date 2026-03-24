"""
Demo Operative Module - Simulation Trading Mode

Handles simulated trading for LAB validation:
- Intercepts order placement (NO real broker orders)
- Tracks simulated positions in memory
- Monitors TP/SL/TIMEOUT using WebSocket prices
- Logs ONLY to Excel (bypasses PostgreSQL)
- Operates WITHOUT REGIME 0 + REGIME 1 layers

This module runs in parallel with live accounts for validation purposes.
"""

import os
import pandas as pd
from datetime import datetime
from decimal import Decimal
from typing import Dict, List, Optional, Any
import logging

from config.settings import HOUR_ZONE
from execution.order_manager import get_current_price

logger = logging.getLogger('BOT_trading.demo_operative')


class DemoOperative:
    """
    Simulated trading handler for validation against LAB results.
    
    Attributes:
        account_number: Account identifier (typically '01' for demo)
        ws_manager: WebSocket manager for real-time prices
        excel_path: Path to Excel file for trade logging
        simulated_positions: In-memory list of open simulated positions
        strategy_configs: Strategy configurations for timeout tracking
    """
    
    def __init__(self, account_number: str, ws_manager, excel_path: str, 
                 strategy_configs: List[Dict]):
        """
        Initialize demo operative module.
        
        Args:
            account_number: Account number (e.g., '01')
            ws_manager: WebSocket manager instance
            excel_path: Path to Excel file for logging
            strategy_configs: List of strategy configurations
        """
        self.account_number = account_number
        self.ws_manager = ws_manager
        self.excel_path = excel_path
        self.strategy_configs = {s['id']: s for s in strategy_configs}
        
        # Simulated positions (in-memory only)
        self.simulated_positions: List[Dict] = []
        
        logger.info(f"[DEMO] Demo operative initialized for account {account_number}")
        logger.info(f"[DEMO] Excel logging path: {excel_path}")
        logger.info(f"[DEMO] NO real orders will be placed")
        logger.info(f"[DEMO] REGIME 0 + REGIME 1 layers DISABLED")
    
    
    def place_simulated_order(self, symbol: str, direction: str, 
                             usdt_amount: float, tp_pct: float, 
                             sl_pct: float, strategy_id: str) -> Optional[Dict]:
        """
        Simulate order placement without sending to broker.
        
        Uses current WebSocket price as entry price.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            direction: 'long' or 'short'
            usdt_amount: USDT amount to invest
            tp_pct: Take profit percentage
            sl_pct: Stop loss percentage
            strategy_id: Strategy identifier
        
        Returns:
            Simulated position dict or None if failed
        """
        try:
            # Get current market price from WebSocket
            current_price = get_current_price(symbol, max_cache_age=0.5)
            
            if not current_price:
                logger.warning(f"[DEMO] No price available for {symbol}, skipping entry")
                return None
            
            current_price_float = float(current_price)
            
            # Calculate TP/SL prices
            if direction.lower() == 'long':
                tp_price = current_price_float * (1 + tp_pct / 100)
                sl_price = current_price_float * (1 - sl_pct / 100)
            else:  # short
                tp_price = current_price_float * (1 - tp_pct / 100)
                sl_price = current_price_float * (1 + sl_pct / 100)
            
            # Create simulated position
            position = {
                'strategy': strategy_id,
                'symbol': symbol,
                'direction': direction.lower(),
                'entry_price': current_price_float,
                'usdt_amount': usdt_amount,
                'tp': tp_price,
                'sl': sl_price,
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
                'entry_time': datetime.now(HOUR_ZONE),
                'candles': 0,
                'simulated': True
            }
            
            self.simulated_positions.append(position)
            
            logger.info(f"[DEMO] ENTRY {direction.upper()} {symbol} @ ${current_price_float:.4f} | "
                       f"Amount: ${usdt_amount:.2f} | TP: ${tp_price:.4f} | SL: ${sl_price:.4f}")
            
            return position
            
        except Exception as e:
            logger.error(f"[DEMO] Error placing simulated order for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def increment_candles(self, strategy_id: str) -> None:
        """
        Increment candle counter for all positions of a strategy.
        
        Called when a new candle closes for the strategy's timeframe.
        
        Args:
            strategy_id: Strategy identifier
        """
        for position in self.simulated_positions:
            if position['strategy'] == strategy_id:
                position['candles'] += 1
    
    def monitor_exits(self, strategy_candles: Dict[str, int]) -> None:
        """
        Monitor simulated positions for TP/SL/TIMEOUT exits.
        
        Should be called periodically (every few seconds) to check exit conditions.
        """
        if not self.simulated_positions:
            return
        
        positions_to_close = []
        
        for position in self.simulated_positions:
            symbol = position['symbol']
            strategy_id = position['strategy']
            
            try:
                # Get current price from WebSocket
                current_price = get_current_price(symbol, max_cache_age=0.5)
                
                if not current_price:
                    continue
                
                current_price_float = float(current_price)
                direction = position['direction']
                
                # Check TP hit
                tp_hit = (direction == 'long' and current_price_float >= position['tp']) or \
                         (direction == 'short' and current_price_float <= position['tp'])
                
                if tp_hit:
                    positions_to_close.append((position, 'TP', current_price_float))
                    continue
                
                # Check SL hit
                sl_hit = (direction == 'long' and current_price_float <= position['sl']) or \
                         (direction == 'short' and current_price_float >= position['sl'])
                
                if sl_hit:
                    positions_to_close.append((position, 'SL', current_price_float))
                    continue
                
                # Check TIMEOUT
                if strategy_id in self.strategy_configs:
                    max_candles = self.strategy_configs[strategy_id].get('sell_after_ncandles', 50)
                    if strategy_candles.get(strategy_id, 0) >= max_candles:
                        positions_to_close.append((position, 'TIMEOUT', current_price_float))
            
            except Exception as e:
                logger.error(f"[DEMO] Error monitoring {symbol}: {e}")
        
        # Close positions that hit exit conditions
        for position, reason, close_price in positions_to_close:
            self._close_simulated_position(position, reason, close_price)
    
    def _close_simulated_position(self, position: Dict, reason: str, 
                                  close_price: float) -> None:
        """
        Close simulated position and log to Excel ONLY.
        
        Args:
            position: Position dictionary
            reason: Exit reason ('TP', 'SL', 'TIMEOUT')
            close_price: Close price
        """
        try:
            entry_price = position['entry_price']
            direction = position['direction']
            usdt_amount = position['usdt_amount']
            symbol = position['symbol']
            strategy_id = position['strategy']
            
            # Calculate profit
            if direction == 'long':
                profit_pct = ((close_price - entry_price) / entry_price) * 100
            else:  # short
                profit_pct = ((entry_price - close_price) / entry_price) * 100
            
            profit_usd = usdt_amount * (profit_pct / 100)
            
            # Calculate size (for compatibility with Excel format)
            size = usdt_amount / entry_price
            
            # Log to Excel ONLY (NO PostgreSQL)
            self._log_to_excel(
                strategy=strategy_id,
                symbol=symbol,
                direction=direction.upper(),
                entry_price=entry_price,
                close_price=close_price,
                profit=profit_usd,
                profit_pct=profit_pct,
                reason=reason,
                entry_time=position['entry_time'],
                close_time=datetime.now(HOUR_ZONE),
                usdt_amount=usdt_amount,
                size=size,
                tp_target=position['tp'],
                sl_target=position['sl']
            )
            
            logger.info(f"[DEMO] EXIT {symbol} | {reason} | "
                       f"Profit: ${profit_usd:.2f} ({profit_pct:+.2f}%) | "
                       f"Strategy: {strategy_id}")
            
            # Remove from simulated positions list
            self.simulated_positions.remove(position)
            
        except Exception as e:
            logger.error(f"[DEMO] Error closing simulated position {position.get('symbol')}: {e}")
            import traceback
            traceback.print_exc()
    
    def _log_to_excel(self, strategy: str, symbol: str, direction: str,
                     entry_price: float, close_price: float, profit: float,
                     profit_pct: float, reason: str, entry_time: datetime,
                     close_time: datetime, usdt_amount: float, size: float,
                     tp_target: float, sl_target: float) -> None:
        """
        Write trade to Excel ONLY (bypass PostgreSQL).
        
        Creates Excel file if it doesn't exist.
        Appends trade to existing Excel file.
        
        Args:
            All trade data fields for Excel logging
        """
        try:
            duration_days = (close_time - entry_time).total_seconds() / (3600 * 24)
            
            # Build trade record (matching PostgreSQL schema for compatibility)
            trade_data = {
                'OPEN_AT': entry_time.strftime('%Y-%m-%d %H:%M:%S'),
                'CLOSE_AT': close_time.strftime('%Y-%m-%d %H:%M:%S'),
                'DURATION_DAYS': round(duration_days, 4),
                'STRATEGY': strategy,
                'SYMBOL': symbol,
                'DIRECTION': direction,
                'USDT_AMOUNT': round(usdt_amount, 2),
                'SIZE': round(size, 6),
                'PRICE_ENTRY': round(entry_price, 6),
                'PRICE_CLOSE': round(close_price, 6),
                'PROFIT': round(profit, 2),
                'FEE': 0.0,  # No fees in simulation
                'PROFIT_PCT': round(profit_pct, 2),
                'REASON_OUT': reason,
                'REGIME_FAMILY': 'no_regime',  # No regime layers in demo
                'REGIME_MULTIPLIER': 1.0,
                'MARKET_DIRECTION': 'no_direction',
                'DIRECTION_MULTIPLIER': 1.0,
                'ORDER_PRICE_CLOSE': None,
                'ORDER_TS_CLOSE': None,
                'EXEC_TS_CLOSE': None,
                'TP_TARGET': round(tp_target, 6),
                'SL_TARGET': round(sl_target, 6),
                'SIMULATED': 'YES'  # Flag to identify demo trades
            }
            
            # Append to Excel
            if os.path.exists(self.excel_path):
                df = pd.read_excel(self.excel_path, engine='openpyxl')
                df = pd.concat([df, pd.DataFrame([trade_data])], ignore_index=True)
            else:
                df = pd.DataFrame([trade_data])
            
            df.to_excel(self.excel_path, index=False, engine='openpyxl')
            
            logger.debug(f"[DEMO] Trade logged to Excel: {symbol} | {reason}")
            
        except Exception as e:
            logger.error(f"[DEMO] Error logging to Excel: {e}")
            import traceback
            traceback.print_exc()
    
    def get_open_positions_count(self) -> int:
        """
        Get count of currently open simulated positions.
        
        Returns:
            Number of open positions
        """
        return len(self.simulated_positions)
    
    def get_positions_by_strategy(self, strategy_id: str) -> List[Dict]:
        """
        Get all open positions for a specific strategy.
        
        Args:
            strategy_id: Strategy identifier
        
        Returns:
            List of position dictionaries
        """
        return [p for p in self.simulated_positions if p['strategy'] == strategy_id]