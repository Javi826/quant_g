#BOT_orchestator.py
"""
Bot multi-estrategy & timeframe with WebSocket support and  API operations.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import argparse
from datetime import datetime
from zoneinfo import ZoneInfo
from parquet_process.Z_parquet_A0_extraction import get_futures_symbols_from_api
from ZX_utils_live import load_final_symbols, fetch_ohlcv_data, normalize_live_ohlcv, df_to_arrays_live
from ZX_connect_live import send_request_01, send_request_E1,send_request_00

# BOT auxiliars
from ZX_BOT_validations import validate_strategy_configuration
from ZX_BOT_websocket import init_websocket
from ZX_BOT_metrics import BotState

from ZX_BOT_backend import DashboardServer, create_dashboard_template
from ZX_BOT_operative import check_tp_sl_for_strategy, get_current_price,configure_paths,process_strategy, setup_print_logger,check_all_tp_sl
from ZX_BOT_operative import sync_broker, load_state, save_state_local, calculate_next_candle_time,increment_strategy_candles
from ZX_BOT_operative import check_candles_timeout_for_strategy, group_strategies_by_timeframe, get_unique_timeframes, get_usdt_balance_ws

# Signals generators
from Z_add_signals_double_top import double_top_long
from Z_add_signals_reversal import reversal_long, reversal_short
from Z_add_signals_parity import parity_long, parity_short
from Z_add_signals_orderblocks import orderblocks_long, orderblocks_short

# Dinamic import account passwords
from utils.ZZ_connect import BITGET_API_KEY_00, BITGET_API_SECRET_00, BITGET_API_PASS_00
from utils.ZZ_connect import BITGET_API_KEY_01, BITGET_API_SECRET_01, BITGET_API_PASS_01
from utils.ZZ_connect import BITGET_API_KEY_E1, BITGET_API_SECRET_E1, BITGET_API_PASS_E1
from utils.ZZ_connect import connect_bitget_01, connect_bitget_E1, connect_bitget_00

#==========================================================================
# CREDENTIALS & ACCOUNT LAUNCHING
#==========================================================================
CREDENTIALS = {
"00": (BITGET_API_KEY_00, BITGET_API_SECRET_00, BITGET_API_PASS_00, connect_bitget_00, send_request_00),
"01": (BITGET_API_KEY_01, BITGET_API_SECRET_01, BITGET_API_PASS_01, connect_bitget_01, send_request_01),
"E1": (BITGET_API_KEY_E1, BITGET_API_SECRET_E1, BITGET_API_PASS_E1, connect_bitget_E1, send_request_E1)
}
parser = argparse.ArgumentParser()
parser.add_argument('--account', type=str, default='E1', help='Account number (01, E1, etc)')
parser.add_argument('--set-active', type=str, default=None, help='Comma-separated list of strategy IDs to set as active')
args = parser.parse_args()
ACCOUNT_NUMBER = args.account
#==========================================================================
# PATHS
#==========================================================================
BASE_DIR        = os.path.expanduser(f'~/projects/quant/quant_g/scripts/live_trading/bot_files_{ACCOUNT_NUMBER}')
TRADES_LOG_PATH = os.path.join(BASE_DIR, f'bot_trades_{ACCOUNT_NUMBER}.xlsx')
LOG_FILE_PATH   = os.path.join(BASE_DIR, f'BOT_orchestator_{ACCOUNT_NUMBER}.log')

#==========================================================================
# DISPLAY: None | summary | detailed
#==========================================================================
BLUE         = "\033[1;94m"   
CYAN         = "\033[1;96m" 
WHITE        = "\033[1;97m" 
RESET        = "\033[0m"
DISPLAY_MODE = "summary"

if ACCOUNT_NUMBER == "00":
    COLOR = BLUE
    INITIAL_CAPITAL = 3671
    dashboard_port = 5000 
    
elif ACCOUNT_NUMBER == "E1":
    COLOR = CYAN
    INITIAL_CAPITAL = 1761
    dashboard_port=5001

elif ACCOUNT_NUMBER == "01":
    COLOR = WHITE
    INITIAL_CAPITAL =9999
    dashboard_port=5099
#==========================================================================
# PATHS FILES & CREDENTIALS
#==========================================================================
# Creat directory 
os.makedirs(BASE_DIR, exist_ok=True)
# Configuration paths
configure_paths(TRADES_LOG_PATH, display_color=COLOR, initial_capital=INITIAL_CAPITAL)
# Configuration logger
setup_print_logger(BASE_DIR, logfile_name=os.path.basename(LOG_FILE_PATH))
# Configuration Credentials
BITGET_API_KEY, BITGET_API_SECRET, BITGET_API_PASS, connect_bitget, send_request_func = CREDENTIALS[ACCOUNT_NUMBER]

#==========================================================================
# GENERAL CONFIG
#==========================================================================
HOUR_ZONE             = ZoneInfo('UTC')
PRODUCT_TYPE          = 'USDT-FUTURES'
CHECK_INTERVAL        = 10
USE_HARDCODED_SIGNALS = False

#==========================================================================
# POSITIONS & STATES
#==========================================================================
STATE_FILE       = os.path.join(BASE_DIR, f'bot_state_{ACCOUNT_NUMBER}.json') 
OPEN_POSITIONS   = {}
STRATEGY_CANDLES = {}

#==========================================================================
# STRATEGY CONFIGURATION
#==========================================================================
STRAT_A = {
    'id': '01_double_top_long_4H',
    'name': 'double_top_long_4H',
    'timeframe': '4H',
    'active': True,
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'lookback': 2,
    'tolerance': 15,
    'trend_th': 5,
    'tp_pct': 4,
    'sl_pct': 10,
    'direction': 'long'
}

STRAT_B = {
    'id': '02_reversal_long_4H',
    'name': 'reversal_long_4H',
    'timeframe': '4H',
    'active': True,
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'lookback': 4,
    'tolerance': 20,
    'ma_period':50,
    'tp_pct': 3,
    'sl_pct': 10,
    'direction': 'long'
}

STRAT_C = {
    'id': '03_parity_long_4H',
    'name': 'parity_long_4H',
    'timeframe': '4H', 
    'active': True,
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'lookback': 150,
    'tolerance': 40,
    'ma_period':50,
    'tp_pct': 3,  
    'sl_pct': 10,  
    'direction': 'long'
}

STRAT_D = {
    'id': '04_reversal_short_4H',
    'name': 'reversal_short_4H',
    'timeframe': '4H',  
    'active': True,
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'lookback': 4,
    'tolerance': 25,
    'ma_period':50,
    'tp_pct': 3,
    'sl_pct': 9,
    'direction': 'short'
}

STRAT_E = {
    'id': '05_parity_short_4H',
    'name': 'parity_short_4H',
    'timeframe': '4H',
    'active': False,
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'lookback': 100,
    'tolerance': 30,
    'ma_period':50,
    'tp_pct': 3,  
    'sl_pct': 9,  
    'direction': 'short'
}

STRAT_F = {
    'id': '06_reversal_long_1H',
    'name': 'reversal_long_1H',
    'timeframe': '1H',
    'active': True,
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'lookback': 7,
    'tolerance': 40,
    'ma_period':25,
    'tp_pct': 2,
    'sl_pct': 10,
    'direction': 'long'
}

STRAT_G = {
    'id': '07_reversal_short_1H',
    'name': 'reversal_short_1H',
    'timeframe': '1H', 
    'active': True,
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'lookback': 5,
    'tolerance': 30,
    'ma_period':50,
    'tp_pct': 1.9,
    'sl_pct': 5,
    'direction': 'short'
}

STRAT_H = {
    'id': '08_reversal_long_6Hutc',
    'name': 'reversal_long_6Hutc',
    'timeframe': '6Hutc', 
    'active': True,
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'lookback': 3,
    'tolerance': 20,
    'ma_period':50,
    'tp_pct': 4,
    'sl_pct': 10,
    'direction': 'long'
}

STRAT_I = {
    'id': '09_reversal_short_6Hutc',
    'name': 'reversal_short_6Hutc',
    'timeframe': '6Hutc', 
    'active': True,
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'lookback': 6,
    'tolerance': 30,
    'ma_period':25,
    'tp_pct': 4,
    'sl_pct': 7.5,
    'direction': 'short'
}

STRAT_J = {
    'id': '10_parity_long_1H',
    'name': 'parity_long_1H',
    'timeframe': '1H', 
    'active': True,
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'lookback': 150,
    'tolerance': 15,
    'ma_period':25,
    'tp_pct': 2,  
    'sl_pct': 10,  
    'direction': 'long'
}

STRAT_K = {
    'id': '11_parity_short_1H',
    'name': 'parity_short_1H',
    'timeframe': '1H',
    'active': True,
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'lookback': 150,
    'tolerance': 20,
    'ma_period':50,
    'tp_pct': 2,  
    'sl_pct': 7.5,  
    'direction': 'short'
}

STRAT_L = {
    'id': '12_parity_long_6Hutc',
    'name': 'parity_long_6Hutc',
    'timeframe': '6Hutc',
    'active': True,
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'lookback': 50,
    'tolerance': 40,
    'ma_period':25,
    'tp_pct': 3.5,  
    'sl_pct': 10,  
    'direction': 'long'
}

STRAT_M = {
    'id': '13_orderblocks_short_4H',
    'name': 'orderblocks_short_4H',
    'timeframe': '4H',
    'active': True,
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'lookback': 150,
    'tolerance': 30,
    'impulse':1.0,
    'tp_pct': 5,  
    'sl_pct': 10,  
    'direction': 'short'
}

STRAT_N = {
    'id': '14_orderblocks_long_4H',
    'name': 'orderblocks_long_4H',
    'timeframe': '4H',
    'active': True,
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'lookback': 50,
    'tolerance': 40,
    'impulse':0.01,
    'tp_pct': 5,  
    'sl_pct': 10,  
    'direction': 'long'
}

implemented_strategies = {
    'double_top_long_4H',
    'reversal_long_4H',
    'parity_long_4H',
    'reversal_short_4H',
    'parity_short_4H',
    'reversal_long_1H',
    'reversal_short_1H',
    'reversal_long_6Hutc',
    'reversal_short_6Hutc',
    'parity_long_1H',
    'parity_short_1H',
    'parity_long_6Hutc',
    'orderblocks_long_4H',
    'orderblocks_short_4H'
}

# ==========================================================================
# CONNECTIONS
# ==========================================================================
send_request_common = send_request_func 
get_balance_common  = get_usdt_balance_ws
exchange            = connect_bitget()  

# ==========================================================================
# APPLY --set-active ARGUMENT
# ==========================================================================
if args.set_active:
    active_ids = [s.strip() for s in args.set_active.split(',')]
    print(f"Setting active strategies from command line: {active_ids}")
    for strat in [STRAT_A, STRAT_B, STRAT_C, STRAT_D, STRAT_E, STRAT_F, 
    
                  STRAT_G, STRAT_H, STRAT_I, STRAT_J, STRAT_K, STRAT_L, 
                  STRAT_M, STRAT_N]:
        if strat['id'] in active_ids:
            strat['active'] = True
        else:
            strat['active'] = False

# ==========================================================================
# STRATEGY SELECTION
# ==========================================================================
if ACCOUNT_NUMBER == "00":
    STRATEGIES = [STRAT_A, STRAT_B, STRAT_C, STRAT_D, STRAT_E, STRAT_F, STRAT_G, STRAT_H, STRAT_I, STRAT_J, STRAT_K, STRAT_L, STRAT_M, STRAT_N]
elif ACCOUNT_NUMBER == "E1":
    STRATEGIES = [STRAT_A, STRAT_B, STRAT_C, STRAT_D, STRAT_E, STRAT_F, STRAT_G, STRAT_H, STRAT_I, STRAT_J, STRAT_K,STRAT_M]
elif ACCOUNT_NUMBER == "01":
    STRATEGIES = [STRAT_A, STRAT_B, STRAT_C]
  
# ==========================================================================
# SIGNAL DETECTION
# ==========================================================================
def detect_signal_for_strategy(strategy, final_symbols):
    """Detecta señales para una estrategia"""
    detected = []
    if not final_symbols:
        return detected
    
    ohlcv = fetch_ohlcv_data(final_symbols, strategy['timeframe'])
    for sym, df in ohlcv.items():
        if df is None or df.empty:
            continue
        
        df_norm = normalize_live_ohlcv(df)
        arr     = df_to_arrays_live(df_norm)
        
        try:
            if strategy['name'] == 'double_top_long_4H':
                signals = double_top_long(
                    arr,
                    lookback_minor=strategy['lookback'],
                    price_tolerance=strategy['tolerance'],
                    trend_th=strategy['trend_th'],
                    live_trading=True
                )
            elif strategy['name'] == 'reversal_long_4H':
                signals = reversal_long(
                    arr,
                    lookback=strategy['lookback'],
                    tolerance=strategy['tolerance'],
                    ma_period=strategy['ma_period'],
                    live_trading=True
                )
            elif strategy['name'] == 'parity_long_4H':
                signals = parity_long(
                    arr,
                    lookback=strategy['lookback'],
                    tolerance=strategy['tolerance'],
                    ma_period=strategy['ma_period'],
                    live_trading=True
                )
            elif strategy['name'] == 'reversal_short_4H':
                signals = reversal_short(
                    arr,
                    lookback=strategy['lookback'],
                    tolerance=strategy['tolerance'],
                    ma_period=strategy['ma_period'],
                    live_trading=True
                )
            elif strategy['name'] == 'parity_short_4H':
                signals = parity_short(
                    arr,
                    lookback=strategy['lookback'],
                    tolerance=strategy['tolerance'],
                    ma_period=strategy['ma_period'],
                    live_trading=True
                )
            elif strategy['name'] == 'reversal_long_1H':
                signals = reversal_long(
                    arr,
                    lookback=strategy['lookback'],
                    tolerance=strategy['tolerance'],
                    ma_period=strategy['ma_period'],
                    live_trading=True
                )
            elif strategy['name'] == 'reversal_short_1H':
                signals = reversal_short(
                    arr,
                    lookback=strategy['lookback'],
                    tolerance=strategy['tolerance'],
                    ma_period=strategy['ma_period'],
                    live_trading=True
                )
            elif strategy['name'] == 'reversal_long_6Hutc':
                signals = reversal_long(
                    arr,
                    lookback=strategy['lookback'],
                    tolerance=strategy['tolerance'],
                    ma_period=strategy['ma_period'],
                    live_trading=True
                )
            elif strategy['name'] == 'reversal_short_6Hutc':
                signals = reversal_short(
                    arr,
                    lookback=strategy['lookback'],
                    tolerance=strategy['tolerance'],
                    ma_period=strategy['ma_period'],
                    live_trading=True
                )
            elif strategy['name'] == 'parity_long_1H':
                signals = parity_long(
                    arr,
                    lookback=strategy['lookback'],
                    tolerance=strategy['tolerance'],
                    ma_period=strategy['ma_period'],
                    live_trading=True
                )
            elif strategy['name'] == 'parity_short_1H':
                signals = parity_short(
                    arr,
                    lookback=strategy['lookback'],
                    tolerance=strategy['tolerance'],
                    ma_period=strategy['ma_period'],
                    live_trading=True
                )
            elif strategy['name'] == 'parity_long_6Hutc':
                signals = parity_long(
                    arr,
                    lookback=strategy['lookback'],
                    tolerance=strategy['tolerance'],
                    ma_period=strategy['ma_period'],
                    live_trading=True
                )
            elif strategy['name'] == 'orderblocks_long_4H':
                signals = orderblocks_long(
                    arr,
                    lookback=strategy['lookback'],
                    tolerance=strategy['tolerance'],
                    impulse=strategy['impulse'],
                    live_trading=True
                )
            elif strategy['name'] == 'orderblocks_short_4H':
                signals = orderblocks_short(
                    arr,
                    lookback=strategy['lookback'],
                    tolerance=strategy['tolerance'],
                    impulse=strategy['impulse'],
                    live_trading=True
                )
            else:
                continue
        except Exception as e:
            print(f"❌ Error looking for signals {sym} ({strategy['name']}): {e}")
            continue
        
        if signals is None or len(signals) == 0:
            continue
        
        if signals[-1] != 0:
            last_row = df_norm.iloc[-1]
            detected.append({
                'symbol': sym,
                'timestamp': last_row.name if 'timestamp' not in df_norm.columns else last_row['timestamp'],
                'close': float(last_row['close'])
            })
    
    return detected

# ==========================================================================
# MAIN LOOP
# ==========================================================================
def main_loop():
    global OPEN_POSITIONS, STRATEGY_CANDLES
    
    print(f"{COLOR}{'=' * 48}{RESET}")
    print(f"{COLOR}STARTING BOT IN ACCOUNT: {ACCOUNT_NUMBER}{RESET}")
    print(f"{COLOR}{'=' * 48}{RESET}")
    
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
    print(f"{COLOR}Validating strategy configuration...{RESET}")
    print(f"{COLOR}{'-' * 48}{RESET}")
        
    errors, warnings = validate_strategy_configuration(STRATEGIES, implemented_strategies)
    
    if errors:
        print(f"{COLOR}{'=' * 48}{RESET}")
        print(f"{COLOR}CONFIGURATION ERRORS FOUND:{RESET}\n")
        for err in errors:
            print(f"  {err}")
        print(f"\n{COLOR}⛔ BOT STOPPED - Fix configuration before running{RESET}")
        print(f"{COLOR}{'=' * 48}{RESET}\n")
        return  

    if warnings:
        print(f"{COLOR}CONFIGURATION WARNINGS:{RESET}")
        for warn in warnings:
            print(f"{warn}")
        print()
    else:
        print(f"All strategies validated successfully\n")
    
    # --------------------------------------------------------------------
    # LOAD SYMBOLS PER STRATEGY & TIMEFRAMES
    # --------------------------------------------------------------------
    print(f"{COLOR}Operative Strategies: {len(STRATEGIES)} {RESET}")
    print(f"{COLOR}{'-' * 48}{RESET}")
    final_by_strat = {}
    for strat in STRATEGIES:
        #DEPRECATED strategies
        if not strat.get('active', True):
            print(f"{strat['id']:<24}: DEPRECATING")
            continue
        
        final_by_strat[strat['id']] = load_final_symbols(all_symbols,strategy=strat['name'],timeframe=strat['timeframe'])
        print(f"{strat['id']:<24} : {len(final_by_strat[strat['id']]):>2} symbols")

    
    # Group strategies by timeframe
    strategies_by_tf  = group_strategies_by_timeframe(STRATEGIES)
    unique_timeframes = get_unique_timeframes(STRATEGIES)
    
    print(f"Detected timeframes: {', '.join(unique_timeframes)}")
    for tf in unique_timeframes:
        strat_names = [s['id'] for s in strategies_by_tf[tf]]

    
    print("BOT Initialization completed\n")
 
    # Después de cargar símbolos y antes del WebSocket
    print(f"{COLOR}Starting Web...{RESET}")
    print(f"{COLOR}{'-' * 48}{RESET}")
    
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
        implemented_strategies=implemented_strategies,
        symbols_by_strategy=final_by_strat  
    )
    
    dashboard.start(port=dashboard_port)
    
    print(f"Bot monitoring at http://localhost:{dashboard_port}")


    # --------------------------------------------------------------------
    # INIT WEBSOCKET
    # --------------------------------------------------------------------
    print(f"{COLOR}Init  WebSocket...{RESET}")
    print(f"{COLOR}{'-' * 48}{RESET}")
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
    print()
    
    # --------------------------------------------------------------------
    # NEXT CANDLES
    # --------------------------------------------------------------------
    print(f"{COLOR}Candles incoming:  {RESET}")
    print(f"{COLOR}{'-' * 48}{RESET}")
    next_candle_times = {}
    for tf in unique_timeframes:
        next_candle_times[tf] = calculate_next_candle_time(tf, hour_zone=HOUR_ZONE)
        print(f"Next for {tf:<{5}} : {next_candle_times[tf].strftime('%Y-%m-%d %H:%M:%S'):<{18}} UTC")
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

                
                print(f"{'=' * 48}")
                print(f"New candles {now_datetime.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                #print(f"Timeframes: {', '.join(closed_timeframes)}")

                # Sync with broker
                sync_broker(OPEN_POSITIONS, STRATEGY_CANDLES, STATE_FILE)
                
                now = datetime.now(HOUR_ZONE).strftime('%Y-%m-%d %H:%M:%S')
                print(f"Searching Signals... - {now}")
                print(f"{'-' * 48}")
                
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
                        
                        print(f"Skip {strat_id:<23} {candles:>2}/{strat['sell_after_ncandles']:<2} | {num_positions:>2} pos.")
                      
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
                        process_strategy(
                            strat=strat,
                            final_symbols=final_by_strat.get(strat['id'], []),
                            exchange=exchange,
                            open_positions=OPEN_POSITIONS,
                            strategy_candles=STRATEGY_CANDLES,
                            state_file=STATE_FILE,
                            send_request_func=send_request_common,
                            get_balance_func=get_balance_common,
                            hour_zone=HOUR_ZONE,
                            use_hardcoded=USE_HARDCODED_SIGNALS,
                            detect_signal_func=detect_signal_for_strategy
                        )
                    except Exception as e:
                        print(f"⚠️ WAR- first try processing {strat_id}: {e}")
                        
                        # Retry once
                        print(f"Retrying {strat_id} after 3 seconds...")
                        time.sleep(3)
                        try:
                            process_strategy(
                                strat=strat,
                                final_symbols=final_by_strat.get(strat['id'], []),
                                exchange=exchange,
                                open_positions=OPEN_POSITIONS,
                                strategy_candles=STRATEGY_CANDLES,
                                state_file=STATE_FILE,
                                send_request_func=send_request_common,
                                get_balance_func=get_balance_common,
                                hour_zone=HOUR_ZONE,
                                use_hardcoded=USE_HARDCODED_SIGNALS,
                                detect_signal_func=detect_signal_for_strategy
                            )
                            print(f"Retry successful for {strat_id}")
                        except Exception as e2:
                            print(f"❌ Error Retry failed for {strat_id}: {e2}")
                                
                print("Signal cycle completed")
                print(f"{'=' * 48}\n")
                
                # Recalculate next candle times
                for tf in closed_timeframes:
                    next_candle_times[tf] = calculate_next_candle_time(tf, hour_zone=HOUR_ZONE)
                    #print(f"Next for {tf}: {next_candle_times[tf].strftime('%Y-%m-%d %H:%M:%S')} UTC")
                
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
        print("🔚 Interrupted by user.")
        save_state_local(OPEN_POSITIONS, STRATEGY_CANDLES, STATE_FILE)
        print("⛔ BOT Stopped")

if __name__ == '__main__':
    main_loop()