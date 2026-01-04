"""
Bot multi-estrategy & timeframe with WebSocket support and API operations.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import argparse
from datetime import datetime
from parquet_process.Z_parquet_A0_extraction import get_futures_symbols_from_api
from market_data import load_final_symbols

import logging
logger = logging.getLogger('BOT_trading.execution.BOT_orchestator_WS')

# BOT auxiliars
from validation import validate_strategy_configuration
from analytics import BotState

# Execution module
from execution import (
    configure_paths,
    place_order,
    close_position,
    get_current_price,
    add_position,
    check_tp_sl_for_strategy,
    check_all_tp_sl,
    calculate_tp_sl_prices,
    log_closed_position,
    get_usdt_balance_ws,
    get_fills_for_order,
    BitgetClient  
)

# State module
from state import (
    load_state,
    save_state_local,
    sync_broker,
    increment_strategy_candles,
    reset_strategy_candles,
    check_candles_timeout_for_strategy
)

# Utils module
from bot_utils import (
    calculate_next_candle_time,
    group_strategies_by_timeframe,
    get_unique_timeframes
)

# Market data
from market_data import init_websocket

# Strategies module
from strategies import StrategyProcessor, IMPLEMENTED_STRATEGIES
from strategies import load_strategies

# Logger
from bot_utils.logger import setup_logger

from config.settings import get_account_strategies
from config.settings import (
    get_account_config,
    HOUR_ZONE,
    PRODUCT_TYPE,
    CHECK_INTERVAL,
    USE_HARDCODED_SIGNALS,
    DISPLAY_MODE,
    COLOR_RESET
)

# Credentials & CCXT connections
from utils.ZZ_connect import (
    BITGET_API_KEY_00, BITGET_API_SECRET_00, BITGET_API_PASS_00,
    BITGET_API_KEY_01, BITGET_API_SECRET_01, BITGET_API_PASS_01,
    BITGET_API_KEY_E1, BITGET_API_SECRET_E1, BITGET_API_PASS_E1,
    connect_bitget_00, connect_bitget_01, connect_bitget_E1 
)

#==========================================================================
# CREDENTIALS & ACCOUNT LAUNCHING
#==========================================================================
# Create Bitget clients
BITGET_CLIENTS = {
    "00": BitgetClient(BITGET_API_KEY_00, BITGET_API_SECRET_00, BITGET_API_PASS_00),
    "01": BitgetClient(BITGET_API_KEY_01, BITGET_API_SECRET_01, BITGET_API_PASS_01),
    "E1": BitgetClient(BITGET_API_KEY_E1, BITGET_API_SECRET_E1, BITGET_API_PASS_E1)
}

# Keep CCXT connections (still needed for balance)
CCXT_CONNECTIONS = {
    "00": connect_bitget_00,
    "01": connect_bitget_01,
    "E1": connect_bitget_E1
}
parser = argparse.ArgumentParser()
parser.add_argument('--account', type=str, default='E1', help='Account number (01, E1, etc)')
parser.add_argument('--set-active', type=str, default=None, help='Comma-separated list of strategy IDs to set as active')
args = parser.parse_args()
ACCOUNT_NUMBER = args.account

#==========================================================================
# ACCOUNT CONFIGURATION
#==========================================================================
account_config  = get_account_config(ACCOUNT_NUMBER)
COLOR           = account_config['color']
INITIAL_CAPITAL = account_config['initial_capital']
dashboard_port  = account_config['dashboard_port']
BASE_DIR        = account_config['paths']['base_dir']
STATE_FILE      = account_config['paths']['state_file']
TRADES_LOG_PATH = account_config['paths']['trades_file']
LOG_FILE_PATH   = account_config['paths']['log_file']

#==========================================================================
# PATHS FILES & CREDENTIALS
#==========================================================================
# Creat directory 
os.makedirs(BASE_DIR, exist_ok=True)
# Configuration paths
configure_paths(TRADES_LOG_PATH, display_color=COLOR, initial_capital=INITIAL_CAPITAL)
# Configuration logger
setup_logger(BASE_DIR, logfile_name=os.path.basename(LOG_FILE_PATH))
# Configuration Credentials
bitget_client = BITGET_CLIENTS[ACCOUNT_NUMBER]
connect_bitget = CCXT_CONNECTIONS[ACCOUNT_NUMBER]

# Create send_request function that uses the client
def send_request_func(method, path, params=None, body=None):
    """Wrapper to maintain compatibility with existing code"""
    return bitget_client.send_request(method, path, params, body)

# For logger/debugging
BITGET_API_KEY = bitget_client.api_key
BITGET_API_SECRET = bitget_client.api_secret
BITGET_API_PASS = bitget_client.api_passphrase


#==========================================================================
# POSITIONS & STATES
#==========================================================================
OPEN_POSITIONS   = {}
STRATEGY_CANDLES = {}

# ==========================================================================
# CONNECTIONS
# ==========================================================================
send_request_common = send_request_func 
get_balance_common  = get_usdt_balance_ws
exchange            = connect_bitget() 

# ==========================================================================
# STRATEGY PROCESSOR INITIALIZATION
# ==========================================================================
strategy_processor = StrategyProcessor(
    send_request_func=send_request_common,
    get_balance_func=get_balance_common,
    hour_zone=HOUR_ZONE,
    state_file=STATE_FILE,
    use_hardcoded=USE_HARDCODED_SIGNALS
) 

# ==========================================================================
# STRATEGY SELECTION
# ==========================================================================
strategy_ids = get_account_strategies(ACCOUNT_NUMBER)
STRATEGIES = load_strategies(strategy_ids)

# ==========================================================================
# APPLY --set-active ARGUMENT
# ==========================================================================
if args.set_active:
    from strategies.strategy_loader import apply_set_active_argument
    active_ids = [s.strip() for s in args.set_active.split(',')]
    logger.info(f"Setting active strategies from command line: {active_ids}")
    apply_set_active_argument(STRATEGIES, active_ids)  

# ==========================================================================
# MAIN LOOP
# ==========================================================================
def main_loop():
    global OPEN_POSITIONS, STRATEGY_CANDLES
    RESET = COLOR_RESET
    
    logger.info(f"{COLOR}{'=' * 48}{RESET}")
    logger.info(f"{COLOR}STARTING BOT IN ACCOUNT: {ACCOUNT_NUMBER}{RESET}")
    logger.info(f"{COLOR}{'=' * 48}{RESET}")
    
    # --------------------------------------------------------------------
    # LOAD STATE & SYMBOLS 
    # --------------------------------------------------------------------
        
    OPEN_POSITIONS, STRATEGY_CANDLES = load_state(STATE_FILE, display_color=COLOR)
    bot_state = BotState()
    if os.path.exists(TRADES_LOG_PATH):
        import pandas as pd
        df = pd.read_excel(TRADES_LOG_PATH)
        if not df.empty:
            bot_state.closed_total_profit = df['PROFIT'].sum()
    all_symbols = get_futures_symbols_from_api(PRODUCT_TYPE)
    
    # --------------------------------------------------------------------
    # STRATEGY VALIDATION
    # --------------------------------------------------------------------
    logger.info(f"{COLOR}Validating strategy configuration...{RESET}")
    logger.info(f"{COLOR}{'-' * 48}{RESET}")
        
    errors, warnings = validate_strategy_configuration(STRATEGIES, IMPLEMENTED_STRATEGIES)
    
    if errors:
        logger.info(f"{COLOR}{'=' * 48}{RESET}")
        logger.info(f"{COLOR}CONFIGURATION ERRORS FOUND:{RESET}\n")
        for err in errors:
            logger.info(f"  {err}")
        logger.info(f"\n{COLOR}⛔ BOT STOPPED - Fix configuration before running{RESET}")
        logger.info(f"{COLOR}{'=' * 48}{RESET}\n")
        return  

    if warnings:
        logger.info(f"{COLOR}CONFIGURATION WARNINGS:{RESET}")
        for warn in warnings:
            logger.info(f"{warn}")
    else:
        logger.info(f"All strategies validated successfully\n")
    
    # --------------------------------------------------------------------
    # LOAD SYMBOLS PER STRATEGY & TIMEFRAMES
    # --------------------------------------------------------------------
    logger.info(f"{COLOR}Operative Strategies: {len(STRATEGIES)} {RESET}")
    logger.info(f"{COLOR}{'-' * 48}{RESET}")
    final_by_strat = {}
    for strat in STRATEGIES:
        #DEPRECATED strategies
        if not strat.get('active', True):
            logger.info(f"{strat['id']:<24}: DEPRECATING")
            continue
        
        final_by_strat[strat['id']] = load_final_symbols(all_symbols,strategy=strat['name'],timeframe=strat['timeframe'])
        logger.info(f"{strat['id']:<24} : {len(final_by_strat[strat['id']]):>2} symbols")

    
    # Group strategies by timeframe
    strategies_by_tf  = group_strategies_by_timeframe(STRATEGIES)
    unique_timeframes = get_unique_timeframes(STRATEGIES)
    
    logger.info(f"Detected timeframes: {', '.join(unique_timeframes)}")
    for tf in unique_timeframes:
        strat_names = [s['id'] for s in strategies_by_tf[tf]]

    
    logger.info("BOT Initialization completed\n")
 
    # Después de cargar símbolos y antes del WebSocket
    logger.info(f"{COLOR}Starting Web...{RESET}")
    logger.info(f"{COLOR}{'-' * 48}{RESET}")
    
    # Crear template si no existe
    create_dashboard_template(BASE_DIR)
    
    # Iniciar dashboard
    dashboard = DashboardServer(
        account_number=ACCOUNT_NUMBER,
        base_dir=BASE_DIR,
        get_current_price_func=get_current_price,
        get_balance_func=get_usdt_balance_ws,
        strategies_config=STRATEGIES,
        color_code=COLOR,
        initial_capital=INITIAL_CAPITAL,
        implemented_strategies=IMPLEMENTED_STRATEGIES,
        symbols_by_strategy=final_by_strat  
    )
    
    dashboard.start(port=dashboard_port)
    
    logger.info(f"Bot monitoring at http://localhost:{dashboard_port}")

    # --------------------------------------------------------------------
    # INIT WEBSOCKET
    # --------------------------------------------------------------------
    logger.info(f"{COLOR}Init  WebSocket...{RESET}")
    logger.info(f"{COLOR}{'-' * 48}{RESET}")
    ws_manager = init_websocket(api_key=BITGET_API_KEY,api_secret=BITGET_API_SECRET,api_passphrase=BITGET_API_PASS)
    
    # --------------------------------------------------------------------
    # PRE-LOAD CONTRACTS INFO
    # --------------------------------------------------------------------
    if ws_manager:
        all_strategy_symbols = set()
        for strat_id, symbols in final_by_strat.items():
            all_strategy_symbols.update(symbols)
        
        if all_strategy_symbols:
            ws_manager.preload_contracts(list(all_strategy_symbols),product_type=PRODUCT_TYPE)
    
    # --------------------------------------------------------------------
    # NEXT CANDLES
    # --------------------------------------------------------------------
    logger.info(f"{COLOR}Candles incoming:  {RESET}")
    logger.info(f"{COLOR}{'-' * 48}{RESET}")
    next_candle_times = {}
    for tf in unique_timeframes:
        next_candle_times[tf] = calculate_next_candle_time(tf, hour_zone=HOUR_ZONE)
        logger.info(f"Next for {tf:<{5}} : {next_candle_times[tf].strftime('%Y-%m-%d %H:%M:%S'):<{18}} UTC")
    #bot_metrics(excel_file=TRADES_LOG_PATH, color_code=COLOR, initial_capital=INITIAL_CAPITAL)

    last_tpsl_check = time.time()
  
    try:
        while True:
            current_time = time.time()
            now_datetime = datetime.now(HOUR_ZONE)
            
            # Check which timeframes closed
            closed_timeframes = []
            for tf in unique_timeframes:
                if now_datetime >= next_candle_times[tf]:
                    closed_timeframes.append(tf)
            
            # Process closed timeframes
            if closed_timeframes:

                
                logger.info(f"{'=' * 48}")
                logger.info(f"New candles {now_datetime.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                logger.info(f"Timeframes: {', '.join(closed_timeframes)}")

                # Sync with broker
                sync_broker(OPEN_POSITIONS, STRATEGY_CANDLES, STATE_FILE)
                
                now = datetime.now(HOUR_ZONE).strftime('%Y-%m-%d %H:%M:%S')
                logger.info(f"Searching Signals... - {now}")
                logger.info(f"{'-' * 48}")
                
                # Process strategies for closed timeframes
                strategies_to_process = []
                for tf in closed_timeframes:
                    strategies_to_process.extend(strategies_by_tf[tf])
                
                # Increment candle counters and check timeouts
                for strat in strategies_to_process:
                    strat_id = strat['id']
                    has_positions = strat_id in OPEN_POSITIONS and len(OPEN_POSITIONS[strat_id]) > 0
                    
                    if has_positions:
                        increment_strategy_candles(strat_id, STRATEGY_CANDLES, OPEN_POSITIONS, STATE_FILE)
                        candles       = STRATEGY_CANDLES.get(strat_id, 0)
                        num_positions = len(OPEN_POSITIONS.get(strat_id, []))
                        
                        logger.info(f"Skip {strat_id:<23} {candles:>2}/{strat['sell_after_ncandles']:<2} | {num_positions:>2} pos.")
                      
                        check_candles_timeout_for_strategy(
                            strat_id,
                            strat['sell_after_ncandles'],
                            OPEN_POSITIONS,
                            STRATEGY_CANDLES,
                            STATE_FILE,
                            send_request_common,
                            bot_state=bot_state
                        )
                
                # Signal search (only if no positions)
                for strat in strategies_to_process:
                    strat_id = strat['id']
                    #Stratgies deprecated
                    if not strat.get('active', True):
                        continue
                    num_positions = len(OPEN_POSITIONS.get(strat_id, []))
                    
                    if num_positions > 0:
                        continue
                    try:
                        strategy_processor.process(
                            strat=strat,
                            final_symbols=final_by_strat.get(strat['id'], []),
                            exchange=exchange,
                            open_positions=OPEN_POSITIONS,
                            strategy_candles=STRATEGY_CANDLES
                        )
                    except Exception as e:
                        logger.warning(f"WAR- first try processing {strat_id}: {e}")
                        
                        # Retry once
                        logger.info(f"Retrying {strat_id} after 3 seconds...")
                        time.sleep(3)
                        try:
                            strategy_processor.process(
                                strat=strat,
                                final_symbols=final_by_strat.get(strat['id'], []),
                                exchange=exchange,
                                open_positions=OPEN_POSITIONS,
                                strategy_candles=STRATEGY_CANDLES
                            )
                            logger.info(f"Retry successful for {strat_id}")
                        except Exception as e2:
                            logger.error(f"Error Retry failed for {strat_id}: {e2}")
                                
                logger.info("Signal cycle completed")
                logger.info(f"{'=' * 48}\n")
                
                # Recalculate next candle times
                for tf in closed_timeframes:
                    next_candle_times[tf] = calculate_next_candle_time(tf, hour_zone=HOUR_ZONE)
                    logger.info(f"Next for {tf}: {next_candle_times[tf].strftime('%Y-%m-%d %H:%M:%S')} UTC")
                
                last_tpsl_check = time.time()
            
            # Periodic TP/SL check
            else:
                if current_time - last_tpsl_check >= CHECK_INTERVAL:
                    check_all_tp_sl(
                        STRATEGIES,
                        OPEN_POSITIONS,
                        STRATEGY_CANDLES,
                        STATE_FILE,
                        send_request_common,
                        HOUR_ZONE,
                        check_tp_sl_for_strategy,  
                        get_current_price,
                        DISPLAY_MODE,
                        account_number=ACCOUNT_NUMBER,
                        display_color=COLOR,
                        bot_state=bot_state
                    )
                    last_tpsl_check = current_time
            
            time.sleep(0.05)
            
    except KeyboardInterrupt:
        save_state_local(OPEN_POSITIONS, STRATEGY_CANDLES, STATE_FILE)
        logger.info("⛔ BOT Stopped")

if __name__ == '__main__':
    main_loop()