#BOT_orchestator.py
"""
Bot multi-estrategy with WebSocket support and complet for API operations.
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
from ZX_connect_live import send_request_01, send_request_E1

# BOT auxiliars
from ZX_BOT_ws_manager import init_websocket
from ZX_BOT_metrics import bot_metrics,BotState
from ZX_BOT_display import check_all_tp_sl  
from ZX_BOT_operative import check_tp_sl_for_strategy, get_current_price,configure_paths,process_strategy, setup_print_logger
from ZX_BOT_operative import sync_broker, load_state, save_state_local, calculate_next_candle_time,increment_strategy_candles
from ZX_BOT_operative import check_candles_timeout_for_strategy, group_strategies_by_timeframe, get_unique_timeframes, get_usdt_balance_ws

# Signals generators
from Z_add_signals_double_top import double_top_long
from Z_add_signals_reversal import reversal_long, reversal_short
from Z_add_signals_parity import parity_long, parity_short
from Z_add_signals_orderblocks import orderblocks_long, orderblocks_short

# Dinamic import account passwords
from utils.ZZ_connect import BITGET_API_KEY_01, BITGET_API_SECRET_01, BITGET_API_PASS_01
from utils.ZZ_connect import BITGET_API_KEY_E1, BITGET_API_SECRET_E1, BITGET_API_PASS_E1
from utils.ZZ_connect import connect_bitget_01, connect_bitget_E1

#Credentials according account number
CREDENTIALS = {
"01": (BITGET_API_KEY_01, BITGET_API_SECRET_01, BITGET_API_PASS_01, connect_bitget_01, send_request_01),
"E1": (BITGET_API_KEY_E1, BITGET_API_SECRET_E1, BITGET_API_PASS_E1, connect_bitget_E1, send_request_E1)
}

#==========================================================================
# ACCOUNTS
#==========================================================================
parser = argparse.ArgumentParser()
parser.add_argument('--account', type=str, default='E1', help='Account number (01, E1, etc)')
args = parser.parse_args()
ACCOUNT_NUMBER = args.account
#ACCOUNT_NUMBER  = "01"  

#==========================================================================
# PATHS
#==========================================================================
BASE_DIR        = os.path.expanduser(f'~/projects/quant/quant_g/scripts/live_trading/bot_files_{ACCOUNT_NUMBER}')
TRADES_LOG_PATH = os.path.join(BASE_DIR, f'bot_trades_{ACCOUNT_NUMBER}.xlsx')
LOG_FILE_PATH   = os.path.join(BASE_DIR, f'BOT_all_strategies_{ACCOUNT_NUMBER}.log')

#==========================================================================
# DISPLAY: None | summary | detailed
#==========================================================================
BLUE         = "\033[1;94m"   
CYAN         = "\033[1;96m"  
RESET        = "\033[0m"
DISPLAY_MODE = "summary"

if ACCOUNT_NUMBER == "01":
    COLOR = BLUE
    INITIAL_CAPITAL = 3671
elif ACCOUNT_NUMBER == "E1":
    COLOR = CYAN
    INITIAL_CAPITAL = 1761
    
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
    'id': 'double_top_long_4H',
    'name': 'double_top_long_4H',
    'timeframe': '4H',
    'active': True,
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'lookback': 2,
    'tolerance': 10,
    'trend_th': 10,
    'tp_pct': 5,
    'sl_pct': 10,
    'direction': 'long'
}

STRAT_B = {
    'id': 'revers_long_4H',
    'name': 'reversal_long_4H',
    'timeframe': '4H',
    'active': True,
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'left_lookback': 5,
    'tolerance': 30,
    'ma_period':50,
    'tp_pct': 3,
    'sl_pct': 10,
    'direction': 'long'
}

STRAT_C = {
    'id': 'parity_long_4H',
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
    'id': 'revers_short_4H',
    'name': 'reversal_short_4H',
    'timeframe': '4H',  
    'active': True,
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'left_lookback': 8,
    'tolerance': 30,
    'ma_period':50,
    'tp_pct': 5,
    'sl_pct': 10,
    'direction': 'short'
}

STRAT_E = {
    'id': 'parity_short_4H',
    'name': 'parity_short_4H',
    'timeframe': '4H',
    'active': True,
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'lookback': 150,
    'tolerance': 20,
    'ma_period':50,
    'tp_pct': 5,  
    'sl_pct': 10,  
    'direction': 'short'
}

STRAT_F = {
    'id': 'revers_long_1H',
    'name': 'reversal_long_1H',
    'timeframe': '1H',
    'active': True,
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'left_lookback': 7,
    'tolerance': 40,
    'ma_period':25,
    'tp_pct': 2,
    'sl_pct': 10,
    'direction': 'long'
}

STRAT_G = {
    'id': 'revers_short_1H',
    'name': 'reversal_short_1H',
    'timeframe': '1H', 
    'active': True,
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'left_lookback': 5,
    'tolerance': 30,
    'ma_period':50,
    'tp_pct': 1.9,
    'sl_pct': 5,
    'direction': 'short'
}

STRAT_H = {
    'id': 'revers_long_6Hutc',
    'name': 'reversal_long_6Hutc',
    'timeframe': '6Hutc', 
    'active': True,
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'left_lookback': 3,
    'tolerance': 20,
    'ma_period':50,
    'tp_pct': 4,
    'sl_pct': 10,
    'direction': 'long'
}

STRAT_I = {
    'id': 'revers_short_6Hutc',
    'name': 'reversal_short_6Hutc',
    'timeframe': '6Hutc', 
    'active': True,
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'left_lookback': 6,
    'tolerance': 30,
    'ma_period':25,
    'tp_pct': 4,
    'sl_pct': 7.5,
    'direction': 'short'
}

STRAT_J = {
    'id': 'parity_long_1H',
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
    'id': 'parity_short_1H',
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
    'id': 'parity_long_6Hutc',
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
    'id': 'orderblocks_short_4H',
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
    'id': 'orderblocks_long_4H',
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

# ==========================================================================
# CONNECTIONS
# ==========================================================================
send_request_common = send_request_func 
get_balance_common  = get_usdt_balance_ws
exchange            = connect_bitget()  

# ==========================================================================
# STRATEGY SELECTION
# ==========================================================================
if ACCOUNT_NUMBER == "01":
    STRATEGIES = [STRAT_A, STRAT_B, STRAT_C, STRAT_D, STRAT_E, STRAT_F, STRAT_G, STRAT_H, STRAT_I, STRAT_J, STRAT_K, STRAT_L, STRAT_M, STRAT_N]
elif ACCOUNT_NUMBER == "E1":
    STRATEGIES = [STRAT_A, STRAT_B, STRAT_C, STRAT_D, STRAT_E, STRAT_F, STRAT_G, STRAT_H, STRAT_I, STRAT_J, STRAT_K]
  
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
                    left_lookback=strategy['left_lookback'],
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
                    left_lookback=strategy['left_lookback'],
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
                    left_lookback=strategy['left_lookback'],
                    tolerance=strategy['tolerance'],
                    ma_period=strategy['ma_period'],
                    live_trading=True
                )
            elif strategy['name'] == 'reversal_short_1H':
                signals = reversal_short(
                    arr,
                    left_lookback=strategy['left_lookback'],
                    tolerance=strategy['tolerance'],
                    ma_period=strategy['ma_period'],
                    live_trading=True
                )
            elif strategy['name'] == 'reversal_long_6Hutc':
                signals = reversal_long(
                    arr,
                    left_lookback=strategy['left_lookback'],
                    tolerance=strategy['tolerance'],
                    ma_period=strategy['ma_period'],
                    live_trading=True
                )
            elif strategy['name'] == 'reversal_short_6Hutc':
                signals = reversal_short(
                    arr,
                    left_lookback=strategy['left_lookback'],
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
# VALIDATION
# ==========================================================================
def validate_strategy_configuration(strategies):

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
    declared_strategies    = {s['name'] for s in strategies}    
    missing_implementation = declared_strategies - implemented_strategies
    unused_implementation  = implemented_strategies - declared_strategies
    errors   = []
    warnings = []
    
    # --------------------------------------------------------------------
    # VALIDATION 1: Names
    # --------------------------------------------------------------------
    if missing_implementation:
        errors.append(f"❌ Strategies WITHOUT implementation: {missing_implementation}")
    
    if unused_implementation:
        warnings.append(f"⚠️  Implemented but NOT declared: {unused_implementation}")
    
    if not missing_implementation:
        print("   🆗 Validation 1: All strategy names implemented")
    
    # --------------------------------------------------------------------
    # VALIDATION 2: Coherence direction
    # --------------------------------------------------------------------
    validation_2_errors = 0
    for strat in strategies:
        name      = strat.get('name', '')
        direction = strat.get('direction', '')
        strat_id  = strat.get('id', 'UNKNOWN')

        name_indicates_long  = '_long_' in name.lower()
        name_indicates_short = '_short_' in name.lower()
        
        if name_indicates_long and direction != 'long':
            errors.append(
                f"❌ Strategy '{strat_id}' has name='{name}' (indicates LONG) "
                f"but direction='{direction}'"
            )
            validation_2_errors += 1
        
        if name_indicates_short and direction != 'short':
            errors.append(
                f"❌ Strategy '{strat_id}' has name='{name}' (indicates SHORT) "
                f"but direction='{direction}'"
            )
            validation_2_errors += 1
        
        if direction not in ['long', 'short']:
            errors.append(
                f"❌ Strategy '{strat_id}' has invalid direction='{direction}' "
                f"(must be 'long' or 'short')"
            )
            validation_2_errors += 1
    
    if validation_2_errors == 0:
        print("   🆗 Validation 2: All directions coherent with names")
            
    # ====================================================================
    # VALIDATION 3: Timeframe coherence
    # ====================================================================
    validation_3_errors = 0
    for strat in strategies:
        name      = strat.get('name', '')
        timeframe = strat.get('timeframe', '')
        strat_id  = strat.get('id', 'UNKNOWN')
        
        if '_4H' in name:
            if timeframe != '4H':
                errors.append(
                    f"❌ Strategy '{strat_id}' has name='{name}' (indicates 4H) "
                    f"but timeframe='{timeframe}'"
                )
                validation_3_errors += 1
        
        elif '_1H' in name:
            if timeframe != '1H':
                errors.append(
                    f"❌ Strategy '{strat_id}' has name='{name}' (indicates 1H) "
                    f"but timeframe='{timeframe}'"
                )
                validation_3_errors += 1
        
        elif '_6Hutc' in name:
            if timeframe != '6Hutc':
                errors.append(
                    f"❌ Strategy '{strat_id}' has name='{name}' (indicates 6Hutc) "
                    f"but timeframe='{timeframe}'"
                )
                validation_3_errors += 1
    
    if validation_3_errors == 0:
        print("   🆗 Validation 3: All timeframes coherent with names")
    
    return errors, warnings
# ==========================================================================
# MAIN LOOP
# ==========================================================================
def main_loop():
    global OPEN_POSITIONS, STRATEGY_CANDLES
    
    print(f"{COLOR}{'=' * 120}{RESET}")
    print(f"{COLOR}🤖 === STARTING MULTI-STRATEGY BOT OPERATING IN ACCOUNT: {ACCOUNT_NUMBER} 🤖 ==={RESET}")
    print(f"{COLOR}{'=' * 120}{RESET}")
    
    # --------------------------------------------------------------------
    # LOAD STATE & SYMBOLS 
    # --------------------------------------------------------------------
        
    OPEN_POSITIONS, STRATEGY_CANDLES = load_state(STATE_FILE)
    bot_state = BotState()
    if os.path.exists(TRADES_LOG_PATH):
        summary = bot_metrics(excel_file=TRADES_LOG_PATH, show_table=False, return_data=True,initial_capital=INITIAL_CAPITAL)
        if summary:
            bot_state.closed_total_profit = summary[0].get('total_profit', 0)
    all_symbols = get_futures_symbols_from_api(PRODUCT_TYPE)
    
    # --------------------------------------------------------------------
    # STRATEGY VALIDATION
    # --------------------------------------------------------------------
    print(f"\n{COLOR}🪪 Validating strategy configuration...{RESET}")
    errors, warnings = validate_strategy_configuration(STRATEGIES)
    
    if errors:
        print(f"\n{COLOR}{'=' * 120}{RESET}")
        print(f"{COLOR}❗ CONFIGURATION ERRORS FOUND:{RESET}\n")
        for err in errors:
            print(f"  {err}")
        print(f"\n{COLOR}⛔ BOT STOPPED - Fix configuration before running{RESET}")
        print(f"{COLOR}{'=' * 120}{RESET}\n")
        return  # NO arrancar el bot

    if warnings:
        print(f"\n{COLOR}⚠️  CONFIGURATION WARNINGS:{RESET}")
        for warn in warnings:
            print(f"  {warn}")
        print()
    else:
        print(f"✅ All strategies validated successfully\n")
    
    # --------------------------------------------------------------------
    # LOAD SYMBOLS PER STRATEGY & TIMEFRAMES
    # --------------------------------------------------------------------
    print(f"\n{COLOR} OPERATIVE STRATEGIES: {len(STRATEGIES)} {RESET}")
    print(f"{COLOR}{'-' * 120}{RESET}")
    final_by_strat = {}
    for strat in STRATEGIES:
        #DEPRECATED
        if not strat.get('active', True):
            print(f"⏸️  Strategy {strat['id']:<18} ({strat['timeframe']:<2}): DEPRECATING (monitoring only)")
            continue
        
        final_by_strat[strat['id']] = load_final_symbols(all_symbols,strategy=strat['name'],timeframe=strat['timeframe'])
        print(f"🔹 Strategy {strat['id']:<18} ({strat['timeframe']:<2}): {len(final_by_strat[strat['id']]):>2} symbols")
    
    # Group strategies by timeframe
    strategies_by_tf  = group_strategies_by_timeframe(STRATEGIES)
    unique_timeframes = get_unique_timeframes(STRATEGIES)
    
    print(f"\n➡️  Detected timeframes: {', '.join(unique_timeframes)}")
    for tf in unique_timeframes:
        strat_names = [s['id'] for s in strategies_by_tf[tf]]
        print(f"   🔹 {tf}: {', '.join(strat_names)}")
    
    print("\n✅ BOT Initialization completed\n")
    bot_metrics(excel_file=TRADES_LOG_PATH, color_code=COLOR, initial_capital=INITIAL_CAPITAL)
    
    # --------------------------------------------------------------------
    # INIT WEBSOCKET
    # --------------------------------------------------------------------
    print(f"\n{COLOR}Initializing WebSocket connections...{RESET}")
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
    next_candle_times = {}
    for tf in unique_timeframes:
        next_candle_times[tf] = calculate_next_candle_time(tf, hour_zone=HOUR_ZONE)
        print(f"⏰ Next candle for {tf:<{5}} : {next_candle_times[tf].strftime('%Y-%m-%d %H:%M:%S'):<{18}} UTC")

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
                print(f"\n{'=' * 120}")
                print(f"🔀 New candle(s) detected {now_datetime.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                print(f"🔹 Timeframes: {', '.join(closed_timeframes)}")

                # Sync with broker
                sync_broker(OPEN_POSITIONS, STRATEGY_CANDLES, STATE_FILE)
                
                now = datetime.now(HOUR_ZONE).strftime('%Y-%m-%d %H:%M:%S')
                print(f"📡 Signal search - {now}")
                print(f"{'-' * 120}")
                
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
                        candles = STRATEGY_CANDLES.get(strat_id, 0)
                        num_positions = len(OPEN_POSITIONS.get(strat_id, []))
                        #print(f"➡️  Status   : {strat_id:<18} ({strat['timeframe']:<2}): {candles}/{strat['sell_after_ncandles']:<2} candles")
                        print(f"🚫 Skipping {strat_id:<18} ({strat['timeframe']:<2}) ➡️  Status: {candles}/{strat['sell_after_ncandles']:<2} candles | {num_positions} open positions.")
                        
                        check_candles_timeout_for_strategy(
                            strat_id,
                            strat['sell_after_ncandles'],
                            OPEN_POSITIONS,
                            STRATEGY_CANDLES,
                            STATE_FILE,
                            send_request_common
                        )
                
                # Signal search (only if no positions)
                for strat in strategies_to_process:
                    strat_id = strat['id']
                    #DEPRECATED
                    if not strat.get('active', True):
                        continue
                    num_positions = len(OPEN_POSITIONS.get(strat_id, []))
                    
                    if num_positions > 0:
                        #print(f"🚫 Skipping : {strat_id:<18} ({strat['timeframe']:<2}): {num_positions} open positions")
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
                        print(f"❌ Error processing {strat_id}: {e}")
                        
                        # Retry once
                        print(f"⏳ Retrying {strat_id} after 3 seconds...")
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
                            print(f"✅ Retry successful for {strat_id}")
                        except Exception as e2:
                            print(f"❌ Retry failed for {strat_id}: {e2}")
                                
                print("🔂 Signal cycle completed")
                print(f"{'=' * 120}\n")
                
                # Recalculate next candle times
                for tf in closed_timeframes:
                    next_candle_times[tf] = calculate_next_candle_time(tf, hour_zone=HOUR_ZONE)
                    print(f"⏰ Next candle for {tf}: {next_candle_times[tf].strftime('%Y-%m-%d %H:%M:%S')} UTC")
                
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
        print("\n🔚 Interrupted by user.")
        save_state_local(OPEN_POSITIONS, STRATEGY_CANDLES, STATE_FILE)
        print("⛔ BOT Stopped")

if __name__ == '__main__':
    main_loop()