#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bot multi-estrategia con soporte WebSocket completo para todas las operaciones API.
"""

# =============================================================================
# ⭐ CONFIGURACIÓN DE CUENTA - CAMBIAR SOLO ESTE NÚMERO
# =============================================================================
ACCOUNT_NUMBER = "02"  # Opciones: "00", "01", "02", "03", "04", "05"
# =============================================================================

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from BOT_metrics import bot_metrics

# --- Imports de módulos ---
from parquet_process.Z_parquet_A0_extraction import get_futures_symbols_from_api
from ZX_utils_live import load_final_symbols, fetch_ohlcv_data, normalize_live_ohlcv, df_to_arrays_live

# Import WebSocket-enabled utilities
from ZX_utils_bot_u_websocket import (
    increment_strategy_candles, process_strategy, check_all_tp_sl,
    setup_print_logger, sync_broker, load_state, save_state_local,
    calculate_next_candle_time, check_candles_timeout_for_strategy,
    group_strategies_by_timeframe, get_unique_timeframes,
    init_websocket, get_usdt_balance_ws
)

# Set global WebSocket mode
import ZX_utils_bot_u_websocket
ZX_utils_bot_u_websocket.USE_WEBSOCKET_FOR_TRADING = False  # ⭐ True = órdenes via WS (requiere permisos), False = REST API (recomendado)

from Z_add_signals_double_top import double_top_long
from Z_add_signals_reversal import reversal_long, reversal_short
from Z_add_signals_parity import parity_long, parity_short

# ⭐ Import dinámico basado en ACCOUNT_NUMBER
from utils.ZZ_connect import (
    BITGET_API_KEY_00, BITGET_API_SECRET_00, BITGET_API_PASS_00,
    BITGET_API_KEY_01, BITGET_API_SECRET_01, BITGET_API_PASS_01,
    BITGET_API_KEY_02, BITGET_API_SECRET_02, BITGET_API_PASS_02,
    BITGET_API_KEY_03, BITGET_API_SECRET_03, BITGET_API_PASS_03,
    BITGET_API_KEY_04, BITGET_API_SECRET_04, BITGET_API_PASS_04,
    BITGET_API_KEY_05, BITGET_API_SECRET_05, BITGET_API_PASS_05,
    connect_bitget_00, connect_bitget_01, connect_bitget_02,
    connect_bitget_03, connect_bitget_04, connect_bitget_05
)

from ZX_connect_live import (
    send_request_00, send_request_01, send_request_02,
    send_request_03, send_request_04, send_request_05
)

# ⭐ Seleccionar credenciales y funciones según ACCOUNT_NUMBER
CREDENTIALS = {
    "00": (BITGET_API_KEY_00, BITGET_API_SECRET_00, BITGET_API_PASS_00, connect_bitget_00, send_request_00),
    "01": (BITGET_API_KEY_01, BITGET_API_SECRET_01, BITGET_API_PASS_01, connect_bitget_01, send_request_01),
    "02": (BITGET_API_KEY_02, BITGET_API_SECRET_02, BITGET_API_PASS_02, connect_bitget_02, send_request_02),
    "03": (BITGET_API_KEY_03, BITGET_API_SECRET_03, BITGET_API_PASS_03, connect_bitget_03, send_request_03),
    "04": (BITGET_API_KEY_04, BITGET_API_SECRET_04, BITGET_API_PASS_04, connect_bitget_04, send_request_04),
    "05": (BITGET_API_KEY_05, BITGET_API_SECRET_05, BITGET_API_PASS_05, connect_bitget_05, send_request_05),
}

if ACCOUNT_NUMBER not in CREDENTIALS:
    raise ValueError(f"ACCOUNT_NUMBER inválido: {ACCOUNT_NUMBER}. Debe ser '00', '01', '02', '03', '04', o '05'")

BITGET_API_KEY, BITGET_API_SECRET, BITGET_API_PASS, connect_bitget, send_request_func = CREDENTIALS[ACCOUNT_NUMBER]

BLUE_BOLD      = "\033[1;94m"
YELLOW_BOLD    = "\033[0;93m"
RESET          = "\033[0m"
HOUR_ZONE      = ZoneInfo('UTC')
PRODUCT_TYPE   = 'USDT-FUTURES'
CHECK_INTERVAL = 5
USE_HARDCODED_SIGNALS = True

print(f"{BLUE_BOLD}{'='*120}")
print(f"🤖 BOT OPERATING IN ACCOUNT: {ACCOUNT_NUMBER}")
print(f"{'='*120}{RESET}\n")

# Setup logging
logdir = os.path.expanduser('~/projects/quant/quant_g/scripts/live_trading/bot_files')
setup_print_logger(logdir)

# State file
STATE_FILE = 'bot_state_u.json'

# Global state
OPEN_POSITIONS   = {}
STRATEGY_CANDLES = {}

# ==========================================================================
# STRATEGY CONFIGURATION
# ==========================================================================
STRAT_A = {
    'id': 'double_top_long_4H',
    'name': 'double_top_long_4H',
    'timeframe': '1m',
    'sell_after_ncandles': 2,
    'order_amount': 10,
    'lookback': 2,
    'tolerance': 20,
    'trend_th': 10,
    'tp_pct': 5,
    'sl_pct': 10,
    'direction': 'long'
}

STRAT_B = {
    'id': 'revers_long_4H',
    'name': 'reversal_long_4H',
    'timeframe': '2m',
    'sell_after_ncandles': 50,
    'order_amount': 10,
    'left_lookback': 5,
    'tolerance': 30,
    'ma_period': 50,
    'tp_pct': 3,
    'sl_pct': 10,
    'direction': 'long'
}

STRAT_C = {
    'id': 'parity_long_4H',
    'name': 'parity_long_4H',
    'timeframe': '4H',
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'lookback': 150,
    'tolerance': 40,
    'tp_pct': 3,
    'sl_pct': 10,
    'direction': 'long'
}

STRAT_D = {
    'id': 'revers_short_4H',
    'name': 'reversal_short_4H',
    'timeframe': '4H',
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'left_lookback': 8,
    'tolerance': 30,
    'ma_period': 50,
    'tp_pct': 5,
    'sl_pct': 10,
    'direction': 'short'
}

STRAT_E = {
    'id': 'parity_short_4H',
    'name': 'parity_short_4H',
    'timeframe': '4H',
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'lookback': 150,
    'tolerance': 20,
    'tp_pct': 5,
    'sl_pct': 10,
    'direction': 'short'
}

STRAT_F = {
    'id': 'revers_long_1H',
    'name': 'reversal_long_1H',
    'timeframe': '1H',
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'left_lookback': 7,
    'tolerance': 40,
    'ma_period': 25,
    'tp_pct': 2,
    'sl_pct': 10,
    'direction': 'long'
}

STRAT_G = {
    'id': 'revers_short_1H',
    'name': 'reversal_short_1H',
    'timeframe': '1H',
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'left_lookback': 5,
    'tolerance': 30,
    'ma_period': 50,
    'tp_pct': 1.9,
    'sl_pct': 5,
    'direction': 'short'
}

STRAT_H = {
    'id': 'revers_long_6Hutc',
    'name': 'reversal_long_6Hutc',
    'timeframe': '6Hutc',
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'left_lookback': 3,
    'tolerance': 20,
    'ma_period': 50,
    'tp_pct': 4,
    'sl_pct': 10,
    'direction': 'long'
}

STRAT_I = {
    'id': 'revers_short_6Hutc',
    'name': 'reversal_short_6Hutc',
    'timeframe': '2m',
    'sell_after_ncandles': 50,
    'order_amount': 40,
    'left_lookback': 6,
    'tolerance': 30,
    'ma_period': 25,
    'tp_pct': 4,
    'sl_pct': 7.5,
    'direction': 'short'
}

STRATEGIES = [STRAT_A, STRAT_B, STRAT_C, STRAT_D, STRAT_E, STRAT_F, STRAT_G, STRAT_H, STRAT_I]

send_request_common = send_request_func 
get_balance_common  = get_usdt_balance_ws
exchange            = connect_bitget()  

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
        arr = df_to_arrays_live(df_norm)
        
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
    
    print(f"{BLUE_BOLD}{'=' * 120}{RESET}")
    print(f"{BLUE_BOLD}🤖 === STARTING MULTI-STRATEGY BOT WITH WEBSOCKET 🤖 ==={RESET}")
    print(f"{BLUE_BOLD}{'=' * 120}{RESET}")
    
    # Load state
    OPEN_POSITIONS, STRATEGY_CANDLES = load_state(STATE_FILE)
    
    # Símbolos disponibles
    all_symbols = get_futures_symbols_from_api(PRODUCT_TYPE)
    
    # Load symbols for each strategy
    final_by_strat = {}
    for strat in STRATEGIES:
        final_by_strat[strat['id']] = load_final_symbols(
            all_symbols,
            strategy=strat['name'],
            timeframe=strat['timeframe']
        )
        print(f"🔹 Strategy {strat['id']} ({strat['timeframe']}): {len(final_by_strat[strat['id']])} symbols")
    
    # Group strategies by timeframe
    strategies_by_tf = group_strategies_by_timeframe(STRATEGIES)
    unique_timeframes = get_unique_timeframes(STRATEGIES)
    
    print(f"\n➡️  Detected timeframes: {', '.join(unique_timeframes)}")
    for tf in unique_timeframes:
        strat_names = [s['id'] for s in strategies_by_tf[tf]]
        print(f"   🔹 {tf}: {', '.join(strat_names)}")
    
    print("\n✅ BOT Initialization completed\n")
    bot_metrics()
    
    # ⭐ INITIALIZE WEBSOCKET WITH CREDENTIALS
    print(f"\n{BLUE_BOLD}🔌 Initializing WebSocket connections...{RESET}")
    ws_manager = init_websocket(api_key=BITGET_API_KEY,api_secret=BITGET_API_SECRET,api_passphrase=BITGET_API_PASS)
    print(f"✅ WebSocket initialized (Trading via {'WS' if ZX_utils_bot_u_websocket.USE_WEBSOCKET_FOR_TRADING else 'REST API'})")
    
    # ⭐ PRE-LOAD CONTRACTS for common symbols (única operación API al inicio)
    if ws_manager:
        all_strategy_symbols = set()
        for strat_id, symbols in final_by_strat.items():
            all_strategy_symbols.update(symbols)
        
        if all_strategy_symbols:
            ws_manager.preload_contracts(
                list(all_strategy_symbols),
                product_type=PRODUCT_TYPE
            )
    print()
    
    # Calculate next candle times
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
                        print(f"➡️  {strat_id:<18} ({strat['timeframe']:<2}): {candles}/{strat['sell_after_ncandles']:<2} candles")

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
                    num_positions = len(OPEN_POSITIONS.get(strat_id, []))
                    
                    if num_positions > 0:
                        print(f"🚫 Skipping {strat_id:<18} ({strat['timeframe']:<2}) - {num_positions} open positions")
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
                        print(f"⏳ Retrying {strat_id} after 5 seconds...")
                        time.sleep(5)
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
                        HOUR_ZONE
                    )
                    last_tpsl_check = current_time
            
            time.sleep(0.1)
            
    except KeyboardInterrupt:
        print("\n🔚 Interrupted by user.")
        save_state_local(OPEN_POSITIONS, STRATEGY_CANDLES, STATE_FILE)
        print("⛔ BOT Stopped")

if __name__ == '__main__':
    main_loop()