"""
core/demo_operative.py Module - Simulation Trading Mode

PERSISTENCE MODEL:
- ✅ JSON for state (open positions)
- ✅ Excel for closed trades
- ❌ NO PostgreSQL
- ❌ NO broker API calls

REGIME BYPASS:
- Skips REGIME 0 (BTC 1D filter)
- Skips REGIME 1 (market regime sizing)
- Uses base order_amount from config

For LAB validation without touching PostgreSQL or broker.
"""

import os
import json
import time
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional
import logging

from config.settings import HOUR_ZONE
from execution.order_manager import get_current_price

logger = logging.getLogger('BOT_trading.demo_operative')


class DemoOperative:
    """
    Simulated trading with JSON + Excel persistence.
    
    Attributes:
        account_number: Account identifier
        ws_manager: WebSocket manager for prices
        excel_path: Path to Excel file for trades
        json_path: Path to JSON file for state
        open_positions: Simulated positions (shared with orchestrator)
        strategy_candles: Candle counters (shared with orchestrator)
        strategy_configs: Strategy configurations
    """
    
    def __init__(self, account_number: str, ws_manager, excel_path: str, 
                 strategy_configs: List[Dict]):
        """
        Initialize demo operative.
        
        Args:
            account_number: Account number
            ws_manager: WebSocket manager
            excel_path: Path to Excel file
            strategy_configs: Strategy configurations
        """
        self.account_number = account_number
        self.ws_manager = ws_manager
        self.excel_path = excel_path
        self.strategy_configs = {s['id']: s for s in strategy_configs}
        
        # JSON state file path
        state_dir = os.path.dirname(excel_path)
        self.json_path = os.path.join(state_dir, f'demo_state_{account_number}.json')
        
        # References (injected by orchestrator)
        self.open_positions: Optional[Dict] = None
        self.strategy_candles: Optional[Dict] = None
        self.state_file: Optional[str] = None  # Ignored in demo mode
        
        logger.info(f"[DEMO] Demo mode initialized for account {account_number}")
        logger.info(f"[DEMO] Persistence: JSON (state) + Excel (trades)")
        logger.info(f"[DEMO] JSON path: {self.json_path}")
        logger.info(f"[DEMO] Excel path: {excel_path}")
        logger.info(f"[DEMO] NO PostgreSQL writes")
        logger.info(f"[DEMO] NO broker API calls")
        logger.info(f"[DEMO] REGIME layers DISABLED")
    
    
    def place_simulated_order(self, symbol: str, direction: str, 
                             usdt_amount: float, tp_pct: float, 
                             sl_pct: float, strategy_id: str) -> Optional[Dict]:
        """
        Simulate order placement and save to JSON.
        
        Args:
            symbol: Trading symbol
            direction: 'long' or 'short'
            usdt_amount: USDT amount
            tp_pct: TP percentage
            sl_pct: SL percentage
            strategy_id: Strategy ID
        
        Returns:
            Position dict or None
        """
        try:
            # Get current price from WebSocket
            current_price = get_current_price(symbol, max_cache_age=0.5)
            if not current_price:
                logger.warning(f"[DEMO] No price for {symbol}")
                return None
            
            entry_price = float(current_price)
            
            # Calculate TP/SL prices
            if direction.lower() == 'long':
                tp_price = entry_price * (1 + tp_pct / 100)
                sl_price = entry_price * (1 - sl_pct / 100)
            else:
                tp_price = entry_price * (1 - tp_pct / 100)
                sl_price = entry_price * (1 + sl_pct / 100)
            
            size = usdt_amount / entry_price
            
            # Create position dict
            position = {
                'symbol': symbol,
                'size': size,
                'entry_price': entry_price,
                'direction': direction.lower(),
                'tp': tp_price,
                'sl': sl_price,
                'order_id': f"demo_{int(time.time() * 1000000)}",
                'opened_at': datetime.now(HOUR_ZONE),
                'usdt_amount': usdt_amount,
                'regime_family': 'no_regime',
                'regime_multiplier': 1.0,
                'market_direction': 'no_direction',
                'direction_multiplier': 1.0,
                'tp_pct': tp_pct,
                'sl_pct': sl_pct,
                'simulated': True
            }
            
            # Add to open_positions (shared reference)
            if strategy_id not in self.open_positions:
                self.open_positions[strategy_id] = []
            
            self.open_positions[strategy_id].append(position)
            
            # Save to JSON only
            self._save_state_json()
            
            logger.info(
                f"[DEMO] ENTRY {direction.upper()} {symbol} @ ${entry_price:.4f} | "
                f"${usdt_amount:.2f} | TP: ${tp_price:.4f} | SL: ${sl_price:.4f}"
            )
            
            return position
            
        except Exception as e:
            logger.error(f"[DEMO] Error placing simulated order {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    
    def increment_candles(self, strategy_id: str) -> None:
        """No-op - orchestrator manages candles."""
        pass
    
    
    def monitor_exits(self, strategy_candles: Dict[str, int]) -> None:
        """
        Monitor simulated positions for TP/SL/TIMEOUT exits.
        
        Args:
            strategy_candles: Candle counters from orchestrator
        """
        if not self.open_positions:
            return
        
        positions_to_close = []
        
        for strategy_id, positions in self.open_positions.items():
            if not positions:
                continue
            
            for position in positions:
                # Skip non-simulated positions
                if not position.get('simulated', False):
                    continue
                
                symbol = position['symbol']
                
                try:
                    # Get current price
                    current_price = get_current_price(symbol, max_cache_age=0.5)
                except (TimeoutError, RuntimeError):
                    continue
                except Exception as e:
                    logger.error(f"[DEMO] Error getting price {symbol}: {e}")
                    continue
                
                try:
                    current_price_float = float(current_price)
                    direction = position['direction']
                    
                    # Check TP hit
                    tp_hit = (direction == 'long' and current_price_float >= position['tp']) or \
                             (direction == 'short' and current_price_float <= position['tp'])
                    
                    if tp_hit:
                        positions_to_close.append((strategy_id, position, 'TP', current_price_float))
                        continue
                    
                    # Check SL hit
                    sl_hit = (direction == 'long' and current_price_float <= position['sl']) or \
                             (direction == 'short' and current_price_float >= position['sl'])
                    
                    if sl_hit:
                        positions_to_close.append((strategy_id, position, 'SL', current_price_float))
                        continue
                    
                    # Check TIMEOUT
                    if strategy_id in self.strategy_configs:
                        max_candles = self.strategy_configs[strategy_id].get('sell_after_ncandles', 50)
                        if strategy_candles.get(strategy_id, 0) >= max_candles:
                            positions_to_close.append((strategy_id, position, 'TIMEOUT', current_price_float))
                
                except Exception as e:
                    logger.error(f"[DEMO] Error processing {symbol}: {e}")
        
        # ============================================================
        # CLOSE POSITIONS WITH PRODUCTION-STYLE LOGS
        # ============================================================
        for strategy_id, position, reason, close_price in positions_to_close:
            # Production-style log (for consistency with live mode)
            symbol = position['symbol']
            now_time = datetime.now(HOUR_ZONE).strftime('%H:%M')
            logger.info(f"{reason} for {symbol} ({strategy_id}) at {now_time}")
            
            self._close_simulated_position(strategy_id, position, reason, close_price)
    
    
    def _close_simulated_position(self, strategy_id: str, position: Dict, 
                                  reason: str, close_price: float) -> None:
        """
        Close simulated position: Excel logging + JSON state update.
        
        NO PostgreSQL writes, NO broker calls.
        
        Args:
            strategy_id: Strategy ID
            position: Position dict
            reason: Exit reason ('TP', 'SL', 'TIMEOUT')
            close_price: Close price
        """
        try:
            entry_price = position['entry_price']
            direction = position['direction']
            usdt_amount = position['usdt_amount']
            symbol = position['symbol']  # ← EXTRACT FIRST
            
            # Calculate profit
            if direction == 'long':
                profit_pct = ((close_price - entry_price) / entry_price) * 100
            else:
                profit_pct = ((entry_price - close_price) / entry_price) * 100
            
            profit_usd = usdt_amount * (profit_pct / 100)
            size = position.get('size', usdt_amount / entry_price)
            
            # Parse entry time
            entry_time = position['opened_at']
            if isinstance(entry_time, str):
                entry_time = datetime.strptime(entry_time, '%Y-%m-%d %H:%M:%S')
                entry_time = entry_time.replace(tzinfo=HOUR_ZONE)
            
            close_time = datetime.now(HOUR_ZONE)
            
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
                entry_time=entry_time,
                close_time=close_time,
                usdt_amount=usdt_amount,
                size=size,
                tp_target=position['tp'],
                sl_target=position['sl']
            )
            
            logger.info(
                f"[DEMO] EXIT {symbol} | {reason} | "
                f"${profit_usd:.2f} ({profit_pct:+.2f}%) | "
                f"Strategy: {strategy_id}"
            )
            
            # ============================================================
            # CRITICAL: Remove position by symbol (not by object reference)
            # ============================================================
            if strategy_id in self.open_positions:
                # Count positions before removal
                count_before = len(self.open_positions[strategy_id])
                
                # Filter out by symbol (NOT by object reference)
                self.open_positions[strategy_id] = [
                    p for p in self.open_positions[strategy_id] 
                    if p['symbol'] != symbol
                ]
                
                # Count positions after removal
                count_after = len(self.open_positions[strategy_id])
                
                # Verify removal succeeded
                if count_after == count_before:
                    logger.error(
                        f"[DEMO] CRITICAL: Failed to remove {symbol} from {strategy_id}! "
                        f"Positions before: {count_before}, after: {count_after}"
                    )
                    raise RuntimeError(f"Failed to remove position {symbol} from state")
                else:
                    logger.debug(
                        f"[DEMO] Removed {symbol} from {strategy_id} "
                        f"({count_before} → {count_after} positions)"
                    )
            
            # Save to JSON only
            self._save_state_json()
            
        except Exception as e:
            logger.error(f"[DEMO] Error closing position {position.get('symbol')}: {e}")
            import traceback
            traceback.print_exc()
            raise  # ← Re-raise to stop bot if position removal fails
    
    
    def _log_to_excel(self, strategy: str, symbol: str, direction: str,
                     entry_price: float, close_price: float, profit: float,
                     profit_pct: float, reason: str, entry_time: datetime,
                     close_time: datetime, usdt_amount: float, size: float,
                     tp_target: float, sl_target: float) -> None:
        """
        Write trade to Excel ONLY (bypass PostgreSQL).
        
        Args:
            All trade data fields for Excel logging
        """
        try:
            duration_days = (close_time - entry_time).total_seconds() / (3600 * 24)
            
            # Build trade record
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
                'FEE': 0.0,
                'PROFIT_PCT': round(profit_pct, 2),
                'REASON_OUT': reason,
                'REGIME_FAMILY': 'no_regime',
                'REGIME_MULTIPLIER': 1.0,
                'MARKET_DIRECTION': 'no_direction',
                'DIRECTION_MULTIPLIER': 1.0,
                'ORDER_PRICE_CLOSE': None,
                'ORDER_TS_CLOSE': None,
                'EXEC_TS_CLOSE': None,
                'TP_TARGET': round(tp_target, 6),
                'SL_TARGET': round(sl_target, 6),
                'SIMULATED': 'YES'
            }
            
            # Append to Excel
            if os.path.exists(self.excel_path):
                df = pd.read_excel(self.excel_path, engine='openpyxl')
                df = pd.concat([df, pd.DataFrame([trade_data])], ignore_index=True)
            else:
                df = pd.DataFrame([trade_data])
            
            df.to_excel(self.excel_path, index=False, engine='openpyxl')
            
        except Exception as e:
            logger.error(f"[DEMO] Error logging to Excel: {e}")
            import traceback
            traceback.print_exc()
    
    
    def _save_state_json(self) -> None:
        """
        Save state to JSON file ONLY (bypass PostgreSQL).
        
        Serializes self.open_positions and self.strategy_candles to JSON.
        """
        try:
            if self.open_positions is None or self.strategy_candles is None:
                return
            
            # Serialize datetime objects to ISO format
            serializable_positions = {}
            for strategy_id, positions in self.open_positions.items():
                serializable_positions[strategy_id] = []
                for pos in positions:
                    pos_copy = pos.copy()
                    if isinstance(pos_copy.get('opened_at'), datetime):
                        pos_copy['opened_at'] = pos_copy['opened_at'].isoformat()
                    serializable_positions[strategy_id].append(pos_copy)
            
            state_data = {
                'open_positions': serializable_positions,  # ← CAMBIAR AQUÍ
                'strategy_candles': self.strategy_candles,
                'account': self.account_number,
                'last_updated': datetime.now(HOUR_ZONE).isoformat()
            }
            
            # Write to JSON
            with open(self.json_path, 'w') as f:
                json.dump(state_data, f, indent=2)
            
        except Exception as e:
            logger.error(f"[DEMO] Error saving JSON state: {e}")
            import traceback
            traceback.print_exc()
    
    
    def get_open_positions_count(self) -> int:
        """Get count of open simulated positions."""
        if not self.open_positions:
            return 0
        
        count = 0
        for positions in self.open_positions.values():
            count += len([p for p in positions if p.get('simulated', False)])
        
        return count
    
    
    def get_positions_by_strategy(self, strategy_id: str) -> List[Dict]:
        """Get positions for strategy."""
        if not self.open_positions or strategy_id not in self.open_positions:
            return []
        
        return [p for p in self.open_positions[strategy_id] if p.get('simulated', False)]
