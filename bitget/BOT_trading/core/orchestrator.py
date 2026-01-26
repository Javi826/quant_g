"""
Bot Orchestrator

Main orchestration class for the trading bot.
Encapsulates all bot state and execution logic.
"""

import os
import sys
import time
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

# Ensure BOT_trading is in path
current_dir = os.path.dirname(os.path.abspath(__file__))  # core/
parent_dir  = os.path.dirname(current_dir)                # BOT_trading/

if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Now do imports
from market_data.api_client import get_futures_symbols_from_api
from market_data import load_final_symbols, init_websocket
from market_regime import get_current_regime, get_current_direction, PositionSizer

from validation import validate_strategy_configuration,validate_settings,validate_postgresql_connection

from analytics import BotState
from api.backend import DashboardServer, create_dashboard_template

from execution import configure_paths, get_current_price, check_tp_sl_for_strategy
from execution import check_all_tp_sl, get_usdt_balance_ws, BitgetClient

from state import load_state, save_state_local, sync_broker
from state import increment_strategy_candles, check_candles_timeout_for_strategy

from bot_utils import calculate_next_candle_time, group_strategies_by_timeframe
from bot_utils import get_unique_timeframes

from strategies import StrategyProcessor, IMPLEMENTED_STRATEGIES,load_strategies

from config.utils import get_account_config
from config.settings import PRODUCT_TYPE, CHECK_INTERVAL, USE_HARDCODED_SIGNALS,HOUR_ZONE
from config.settings import REGIME_GENERAL, DIRECTION_MATRIX, DIRECTION_GENERAL

class BotOrchestrator:
    """
    Main orchestrator for the trading bot.
    
    Encapsulates all bot state and execution logic, making it
    testable and allowing multiple instances to run in parallel.
    
    Attributes:
        account_number: Account identifier (e.g., "01", "E1", "00")
        config: Account-specific configuration
        open_positions: Dictionary of open positions by strategy
        strategy_candles: Candle counters by strategy
        strategies: List of active strategies
        
    Example:
        >>> bot = BotOrchestrator("01")
        >>> bot.run()
    """
    
    def __init__(
        self,
        account_number: str,
        bitget_client: BitgetClient,
        connect_bitget_func: callable,
        active_strategy_ids: Optional[List[str]] = None
    ):
        """
        Initialize the bot orchestrator.
        
        Args:
            account_number: Account number (e.g., "01", "E1", "00")
            bitget_client: Authenticated Bitget API client
            connect_bitget_func: Function to connect CCXT exchange
            active_strategy_ids: Optional list of strategy IDs to set as active
                               (overrides YAML configuration)
        """
        self.account_number = account_number
        self.logger = logging.getLogger(f'BOT_trading.core.orchestrator.{account_number}')
        
        # Account configuration
        self.config          = get_account_config(account_number)
        self.initial_capital = self.config['initial_capital']
        self.dashboard_port  = self.config['dashboard_port']
        self.base_dir        = self.config['paths']['base_dir']
        self.state_file      = self.config['paths']['state_file']
        self.trades_log_path = self.config['paths']['trades_file']
        self.log_file_path   = self.config['paths']['log_file']
        
        # API clients
        self.bitget_client = bitget_client
        self.connect_bitget_func = connect_bitget_func
        
        # Bot state (encapsulated)
        self.open_positions: Dict[str, List[Dict]] = {}
        self.strategy_candles: Dict[str, int] = {}
        self.bot_state: Optional[BotState] = None
        
        # Strategies
        self.strategies: List[Dict] = []
        self.active_strategy_ids = active_strategy_ids
        
        # Market data & execution
        self.exchange = None
        self.ws_manager = None
        self.dashboard = None
        self.strategy_processor = None
        
        # Runtime state
        self.all_symbols: List[str]                  = []
        self.final_by_strat: Dict[str, List[str]]    = {}
        self.strategies_by_tf: Dict[str, List[Dict]] = {}
        self.unique_timeframes: List[str]            = []
        self.next_candle_times: Dict[str, datetime] = {}
        self.last_tpsl_check: float = 0
        
        # Control flags
        self._running = False
        self._initialized = False
        
        #Marke regime
        self.regime_cache: Dict[str, str] = {}
        self.direction_cache: Dict[str, str] = {}
        self.position_sizer: Optional[PositionSizer] = None
        
    # ======================================================================
    # PUBLIC API
    # ======================================================================
    
    def run(self) -> None:
        """
        Main entry point to run the bot.
        
        Initializes all components and starts the main trading loop.
        """
        if not self._initialized:
            self.initialize()
        
        self._running = True
        self._log_startup()
        
        try:
            self._main_loop()
        except KeyboardInterrupt:
            self.shutdown()
    
    def initialize(self) -> None:
        """
        Initialize all bot components.
        
        This method sets up:
        - Directory structure
        - logger
        - State loading
        - Strategy loading and validation
        - Market data connections
        - Dashboard
        - WebSocket connections
        """
        if self._initialized:
            self.logger.warning("WAR-Bot already initialized, skipping...")
            return
        
        self._setup_directories()
        self._load_bot_state()
        self._load_and_validate_strategies()
        self._load_market_symbols()
        self._initialize_connections()
        self._start_dashboard()
        self._initialize_websocket()
        self._calculate_next_candles()
        
        self._initialized = True
        self.logger.info("BOT Initialization completed\n")
        
    def shutdown(self) -> None:
        """
        Gracefully shutdown the bot.
        
        Saves state and closes all connections.
        """
        self._running = False
        try:
           save_state_local(self.open_positions, self.strategy_candles, self.account_number, self.state_file)
        except Exception as e:
            self.logger.error(f"CRITICAL ERROR saving state during shutdown")
            self.logger.error(f"Account: {self.account_number}, Positions: {sum(len(p) for p in self.open_positions.values())}")
            self.logger.error(f"Error: {e}")
        self.logger.info("⛔ BOT Stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current bot status for monitoring.
        
        Returns:
            Dictionary with current bot status including:
            - account_number
            - total_positions
            - active_strategies
            - websocket_status
            - uptime
            - etc.
        """
        total_positions   = sum(len(positions) for positions in self.open_positions.values())
        active_strategies = sum(1 for s in self.strategies if s.get('active', True))
        
        return {
            'account_number': self.account_number,
            'running': self._running,
            'initialized': self._initialized,
            'total_positions': total_positions,
            'active_strategies': active_strategies,
            'total_strategies': len(self.strategies),
            'websocket_connected': self.ws_manager.is_connected() if self.ws_manager else False,
            'dashboard_port': self.dashboard_port,
            'total_profit': self.bot_state.closed_total_profit if self.bot_state else 0
        }
    
    def reload_strategies(self) -> None:
        """
        Reload strategies from YAML without restarting the bot.
        
        Useful for applying strategy parameter changes without downtime.
        """
        self.logger.info("Reloading strategies...")
        self.strategies = load_strategies(self.account_number)
        
        if self.active_strategy_ids:
            from strategies.strategy_loader import apply_set_active_argument
            apply_set_active_argument(self.strategies, self.active_strategy_ids)
        
        self.logger.info(f"Reloaded {len(self.strategies)} strategies")
    
    # ======================================================================
    # INITIALIZATION METHODS (Private)
    # ======================================================================
    
    def _setup_directories(self) -> None:
        """Setup necessary directories and paths."""
        os.makedirs(self.base_dir, exist_ok=True)
        configure_paths(
            self.trades_log_path,
            initial_capital=self.initial_capital
        )
    
    def _load_bot_state(self) -> None:
        """Load bot state from disk and initialize BotState."""
        self.open_positions, self.strategy_candles = load_state(
            self.account_number,
            self.state_file
        )
        
        self.bot_state = BotState()
        if os.path.exists(self.trades_log_path):
            import pandas as pd
            df = pd.read_excel(self.trades_log_path)
            if not df.empty:
                self.bot_state.closed_total_profit = df['PROFIT'].sum()
                
    
    def _load_and_validate_strategies(self) -> None:
        # Load strategies
        self.strategies = load_strategies(self.account_number)
        
        # Apply --set-active
        if self.active_strategy_ids:
            from strategies.strategy_loader import apply_set_active_argument
            apply_set_active_argument(self.strategies, self.active_strategy_ids)
        
        # Validate
        self.logger.info(f"Validating configuration...")
        self.logger.info(f"{'-' * 48}")
        
        validate_postgresql_connection()
        
        all_errors = []
        all_warnings = []
              
        # 1. Strategies
        strat_errors, strat_warnings = validate_strategy_configuration(self.strategies, IMPLEMENTED_STRATEGIES)
        all_errors.extend(strat_errors)
        all_warnings.extend(strat_warnings)
                
        # 3. Settings
        settings_errors, settings_warnings = validate_settings()
        all_errors.extend(settings_errors)
        all_warnings.extend(settings_warnings)
        
        
        # Check errors
        if all_errors:
            self.logger.error(f"{'=' * 48}")
            self.logger.error(f"CONFIGURATION ERRORS FOUND:\n")
            for err in all_errors:
                self.logger.error(f"  {err}")
            self.logger.error(f"\n⛔ BOT STOPPED - Fix configuration")
            self.logger.error(f"{'=' * 48}\n")
            
            # FLUSH antes de raise
            for handler in logging.getLogger('BOT_trading').handlers:
                handler.flush()
            
            raise ValueError("Invalid configuration")
        
        if all_warnings:
            self.logger.warning(f"CONFIGURATION WARNINGS:")
            for warn in all_warnings:
                self.logger.warning(f"{warn}")
                
        self.position_sizer = PositionSizer(self.logger)
    
    def _load_market_symbols(self) -> None:
        """Load market symbols for each strategy."""
        self.all_symbols = get_futures_symbols_from_api(PRODUCT_TYPE)
        
        self.logger.info(f"Operative Strategies: {len(self.strategies)}")
        self.logger.info(f"{'-' * 48}")
        
        for strat in self.strategies:
            if not strat.get('active', True):
                self.logger.info(f"{strat['id']:<24}: DEPRECATED")
                continue
            
            self.final_by_strat[strat['id']] = load_final_symbols(
                self.all_symbols,
                strategy=strat['id'],
                timeframe=strat['timeframe']
            )
            self.logger.info(
                f"{strat['id']:<24} : {len(self.final_by_strat[strat['id']]):>2} symbols"
            )
        
        # Group strategies by timeframe
        self.strategies_by_tf  = group_strategies_by_timeframe(self.strategies)
        self.unique_timeframes = get_unique_timeframes(self.strategies)
        
        self.logger.info(f"Detected timeframes: {', '.join(self.unique_timeframes)}")
    
    def _initialize_connections(self) -> None:
        """Initialize exchange and strategy processor."""
        self.exchange = self.connect_bitget_func()
        
        self.strategy_processor = StrategyProcessor(
            send_request_func=self._send_request_wrapper,
            get_balance_func=get_usdt_balance_ws,
            hour_zone=HOUR_ZONE,
            account_number=self.account_number,
            state_file=self.state_file,
            use_hardcoded=USE_HARDCODED_SIGNALS
        )
    
    def _start_dashboard(self) -> None:
        """Start the web dashboard."""
        self.logger.info(f"Starting Web...")
        self.logger.info(f"{'-' * 48}")
        
        create_dashboard_template(self.base_dir)
        
        self.dashboard = DashboardServer(
            account_number=self.account_number,
            base_dir=self.base_dir,
            get_current_price_func=get_current_price,
            get_balance_func=get_usdt_balance_ws,
            strategies_config=self.strategies,
            initial_capital=self.initial_capital,
            implemented_strategies=IMPLEMENTED_STRATEGIES,
            symbols_by_strategy=self.final_by_strat
        )
        
        self.dashboard.start(port=self.dashboard_port)
        self.logger.info(f"Bot monitoring at http://localhost:{self.dashboard_port}")
    
    def _initialize_websocket(self) -> None:
        """Initialize WebSocket connections."""
        self.logger.info(f"Init WebSocket...")
        self.logger.info(f"{'-' * 48}")
        
        self.ws_manager = init_websocket(
            api_key=self.bitget_client.api_key,
            api_secret=self.bitget_client.api_secret,
            api_passphrase=self.bitget_client.api_passphrase
        )
        
        # Pre-load contracts
        if self.ws_manager:
            all_strategy_symbols = set()
            for strat_id, symbols in self.final_by_strat.items():
                all_strategy_symbols.update(symbols)
            
            if all_strategy_symbols:
                self.ws_manager.preload_contracts(
                    list(all_strategy_symbols),
                    product_type=PRODUCT_TYPE
                )
    
    def _calculate_next_candles(self) -> None:
        """Calculate next candle close times."""
        self.logger.info(f"Candles incoming:")
        self.logger.info(f"{'-' * 48}")
        
        for tf in self.unique_timeframes:
            self.next_candle_times[tf] = calculate_next_candle_time(tf, hour_zone=HOUR_ZONE)
            self.logger.info(
                f"Next for {tf:<5} : "
                f"{self.next_candle_times[tf].strftime('%Y-%m-%d %H:%M:%S'):<18} UTC"
            )
        
        self.last_tpsl_check = time.time()
    
    # ======================================================================
    # MAIN LOOP (Private)
    # ======================================================================
    
    def _main_loop(self) -> None:
        """
        Main trading loop.
        
        Continuously checks for:
        - New closed candles → process signals
        - TP/SL hits → close positions
        """
        while self._running:
            current_time      = time.time()
            now_datetime      = datetime.now(HOUR_ZONE)
            closed_timeframes = self._get_closed_timeframes(now_datetime)
            
            if closed_timeframes:
                self._process_new_candles(closed_timeframes, now_datetime)
            else:
                self._periodic_tpsl_check(current_time)
            
            time.sleep(0.05)
    
    def _get_closed_timeframes(self, now_datetime: datetime) -> List[str]:
        """Check which timeframes have closed candles."""
        closed_timeframes = []
        for tf in self.unique_timeframes:
            if now_datetime >= self.next_candle_times[tf]:
                closed_timeframes.append(tf)
        return closed_timeframes
    
    def _process_new_candles(
        self,
        closed_timeframes: List[str],
        now_datetime: datetime
    ) -> None:
        """
        Process new closed candles.
        
        Args:
            closed_timeframes: List of timeframes that just closed
            now_datetime: Current datetime
        """
        self.logger.info(f"{'=' * 48}")
        self.logger.info(f"New candles {now_datetime.strftime('%Y-%m-%d %H:%M:%S')} UTC")
        self.logger.info(f"Timeframes: {', '.join(closed_timeframes)}")
        
        # Sync with broker
        sync_broker(self.open_positions, self.strategy_candles, self.account_number, self.state_file)
        
        # ========================================================================
        # NEW: Update regime for closed timeframes
        # ========================================================================
        self._update_regime_for_timeframes(closed_timeframes)
        
        now = datetime.now(HOUR_ZONE).strftime('%Y-%m-%d %H:%M:%S')
        self.logger.info(f"Searching Signals... - {now}")
        self.logger.info(f"{'-' * 48}")
        
        # Get strategies to process
        strategies_to_process = []
        for tf in closed_timeframes:
            strategies_to_process.extend(self.strategies_by_tf[tf])
                
        # Process candle timeouts
        self._process_candle_timeouts(strategies_to_process)
        
        # Search for new signals (will use regime cache)
        self._search_signals(strategies_to_process)
        
        self.logger.info("Signal cycle completed")
        self.logger.info(f"{'=' * 48}\n")
        
        # Recalculate next candle times
        self._update_next_candle_times(closed_timeframes)
        
        self.last_tpsl_check = time.time()

    
    def _process_candle_timeouts(self, strategies_to_process: List[Dict]) -> None:
        """
        Process strategies with open positions (candle timeout logic).
        
        Args:
            strategies_to_process: Strategies to check for timeouts
        """
        for strat in strategies_to_process:
            strat_id = strat['id']
            has_positions = (
                strat_id in self.open_positions and
                len(self.open_positions[strat_id]) > 0
            )
            
            if has_positions:
                increment_strategy_candles(
                    strat_id,
                    self.strategy_candles,
                    self.open_positions,
                    self.account_number,
                    self.state_file
                )
                candles       = self.strategy_candles.get(strat_id, 0)
                num_positions = len(self.open_positions.get(strat_id, []))
                
                self.logger.info(
                    f"Skip {strat_id:<23} {candles:>2}/"
                    f"{strat['sell_after_ncandles']:<2} | {num_positions:>2} pos."
                )
                
                check_candles_timeout_for_strategy(
                    strat_id,
                    strat['sell_after_ncandles'],
                    self.open_positions,
                    self.strategy_candles,
                    self.account_number,
                    self.state_file,
                    self._send_request_wrapper,
                    bot_state=self.bot_state
                )
    
    def _search_signals(self, strategies_to_process: List[Dict]) -> None:
            """
            Search for new trading signals.
            
            Args:
                strategies_to_process: Strategies to check for signals
            """
            for strat in strategies_to_process:
                strat_id = strat['id']
                
                # Skip deprecated strategies
                if not strat.get('active', True):
                    continue
                
                # Skip if already has positions
                num_positions = len(self.open_positions.get(strat_id, []))
                if num_positions > 0:
                    continue
                
                # Get market state from cache
                timeframe = strat['timeframe']
                market_regime = self.regime_cache.get(timeframe, 'ranging')
                market_direction = self.direction_cache.get(timeframe, 'uptrend')
                
                # Calculate adjusted amount using PositionSizer
                # Calculate adjusted amount using PositionSizer
                adjusted_amount, metadata = self.position_sizer.calculate_adjusted_amount(
                    base_amount=strat['order_amount'],
                    regime_trending=strat.get('regime_trending', 1.0),
                    regime_ranging=strat.get('regime_ranging', 1.0),
                    regime_volatile=strat.get('regime_volatile', 1.0),
                    direction_mode=strat.get('direction_mode', 'general'),
                    market_regime=market_regime,
                    market_direction=market_direction
                )
                
                # Check if blocked
                if metadata['blocked']:
                    log_msg = self.position_sizer.format_log_message(strat_id, metadata)
                    self.logger.info(log_msg)
                    continue
                
                # Log sizing decision
                log_msg = self.position_sizer.format_log_message(strat_id, metadata)
                self.logger.info(log_msg)
                
                # Process strategy with retry logic
                try:
                    self.strategy_processor.process(
                        strat=strat,
                        final_symbols=self.final_by_strat.get(strat['id'], []),
                        exchange=self.exchange,
                        open_positions=self.open_positions,
                        strategy_candles=self.strategy_candles,
                        adjusted_order_amount=adjusted_amount,
                        regime_family=metadata['market_regime'],
                        regime_multiplier=metadata['regime_multiplier'],
                        direction=metadata['market_direction'],
                        direction_multiplier=metadata['direction_multiplier']
                    )
                except Exception as e:
                    self.logger.warning(f"WAR-first try processing {strat_id}: {e}")
                    
                    # Retry once
                    self.logger.info(f"Retrying {strat_id} after 3 seconds...")
                    time.sleep(3)
                    try:
                        self.strategy_processor.process(
                            strat=strat,
                            final_symbols=self.final_by_strat.get(strat['id'], []),
                            exchange=self.exchange,
                            open_positions=self.open_positions,
                            strategy_candles=self.strategy_candles,
                            adjusted_order_amount=adjusted_amount,
                            regime_family=metadata['market_regime'],
                            regime_multiplier=metadata['regime_multiplier'],
                            direction=metadata['market_direction'],
                            direction_multiplier=metadata['direction_multiplier']
                        )
                        self.logger.info(f"Retry successful for {strat_id}")
                    except Exception as e2:
                        self.logger.error(f"Error-Retry failed for {strat_id}: {e2}")
    
    def _periodic_tpsl_check(self, current_time: float) -> None:
        """
        Periodic TP/SL check (when no new candles).
        
        Args:
            current_time: Current timestamp
        """
        if current_time - self.last_tpsl_check >= CHECK_INTERVAL:
            check_all_tp_sl(
                self.strategies,
                self.open_positions,
                self.strategy_candles,
                self.account_number,  # ← AÑADIR
                self.state_file,
                self._send_request_wrapper,
                check_tp_sl_for_strategy,
                bot_state=self.bot_state
            )
            self.last_tpsl_check = current_time
    
    def _update_next_candle_times(self, closed_timeframes: List[str]) -> None:
        """
        Update next candle close times after processing.
        
        Args:
            closed_timeframes: Timeframes that just closed
        """
        for tf in closed_timeframes:
            self.next_candle_times[tf] = calculate_next_candle_time(tf, hour_zone=HOUR_ZONE)
            self.logger.info(
                f"Next for {tf}: "
                f"{self.next_candle_times[tf].strftime('%Y-%m-%d %H:%M:%S')} UTC"
            )
    
    # ======================================================================
    # HELPER METHODS (Private)
    # ======================================================================
    
    # ----------------------------------------------------------------------------
    # CHANGE 3: Add new method (after _calculate_next_candles, around line 350)
    # ----------------------------------------------------------------------------
    
    def _update_regime_for_timeframes(self, closed_timeframes: List[str]) -> None:
            """
            Update regime AND direction for timeframes that just closed.
            
            Args:
                closed_timeframes: List of timeframes that just closed (e.g., ['4H', '1H'])
            """
            self.logger.info(f"[REGIME] Updating regime & direction for: {closed_timeframes}")
            
            for tf in closed_timeframes:
                try:
                    # 1. Calculate REGIME
                    family, metrics = get_current_regime(tf)
                    if family == 'default':
                        family = 'ranging'
                    self.regime_cache[tf] = family
                    
                    # 2. Calculate DIRECTION (now returns price and MA50)
                    direction, btc_price, btc_ma50 = get_current_direction(tf)
                    self.direction_cache[tf] = direction
                    
                    # Format price/MA50 strings for logging
                    price_str = f"${btc_price:.2f}" if btc_price else "N/A"
                    ma50_str = f"${btc_ma50:.2f}" if btc_ma50 else "N/A"
                    
                    self.logger.info(
                        f"[REGIME] {tf}: REGIME={family.upper()}, DIRECTION={direction.upper()} "
                        f"(BTC={price_str}, MA50={ma50_str}, "
                        f"hurst={metrics.get('hurst', 0):.2f}, "
                        f"er={metrics.get('efficiency_ratio', 0):.2f})"
                    )
                    
                except Exception as e:
                    self.logger.error(f"[REGIME] Error for {tf}: {e}")
                    self.regime_cache[tf] = 'ranging'
                    self.direction_cache[tf] = 'uptrend'
    
    def _send_request_wrapper(
        self,
        method: str,
        path: str,
        params: Optional[Dict] = None,
        body: Optional[Dict] = None
    ) -> Any:
        """
        Wrapper for Bitget API requests.
        
        Args:
            method: HTTP method
            path: API path
            params: Query parameters
            body: Request body
        
        Returns:
            API response
        """
        return self.bitget_client.send_request(method, path, params, body)
    
    def _log_startup(self) -> None:
        """Log bot startup information."""
        self.logger.info(f"{'=' * 48}")
        self.logger.info(f"STARTING BOT IN ACCOUNT: {self.account_number}")
        self.logger.info(f"{'=' * 48}")
