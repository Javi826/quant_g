#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bot multi-estrategia con soporte para múltiples timeframes simultáneos.
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from BOT_metrics import bot_metrics

# --- Imports de tus módulos ---
from parquet_process.Z_parquet_A0_extraction import get_futures_symbols_from_api
from ZX_utils_live import load_final_symbols, fetch_ohlcv_data, normalize_live_ohlcv, df_to_arrays_live
from ZX_utils_bot import increment_strategy_candles, process_strategy, check_all_tp_sl,setup_print_logger, sync_broker, load_state, save_state_local,calculate_next_candle_time, check_candles_timeout_for_strategy
from ZX_utils_bot import group_strategies_by_timeframe,get_unique_timeframes
from Z_add_signals_double_top import double_top_long
from Z_add_signals_reversal import reversal_long, reversal_short
from Z_add_signals_parity import parity_long, parity_short

logdir = os.path.expanduser('~/projects/quant/quant_g/scripts/live_trading/bot_files')
setup_print_logger(logdir)

from utils.ZZ_connect import connect_bitget_00
from ZX_connect_live import get_usdt_balance_00, send_request_00

HOUR_ZONE                 = ZoneInfo('UTC')
PRODUCT_TYPE              = 'USDT-FUTURES'
CHECK_INTERVAL            = 10  
USE_HARDCODED_SIGNALS     = False

# Archivo de estado
STATE_FILE = 'bot_state.json'

# Registro de posiciones abiertas por estrategia
OPEN_POSITIONS   = {}
STRATEGY_CANDLES = {}

# ----------------------
# Configuración de Estrategias
# ----------------------
STRAT_A = {
    'id': 'double_top_long',
    'name': 'double_top_long',
    'timeframe': '4H',
    'sell_after_ncandles': 50,
    'order_amount': 80,
    'lookback': 2,
    'tolerance': 20,
    'trend_th': 10,
    'tp_pct': 5,
    'sl_pct': 10,
    'direction': 'long'
}

STRAT_B = {
    'id': 'revers_long',
    'name': 'reversal_long',
    'timeframe': '4H',
    'sell_after_ncandles': 50,
    'order_amount': 80,
    'left_lookback': 5,
    'tolerance': 30,
    'tp_pct': 3,
    'sl_pct': 10,
    'direction': 'long'
}

STRAT_C = {
    'id': 'parity_long',
    'name': 'parity_long',
    'timeframe': '4H',  
    'sell_after_ncandles': 50,
    'order_amount': 80,
    'lookback': 150,
    'tolerance': 40,
    'tp_pct': 3,  
    'sl_pct': 10,  
    'direction': 'long'
}

STRAT_D = {
    'id': 'revers_short',
    'name': 'reversal_short',
    'timeframe': '4H',  
    'sell_after_ncandles': 50,
    'order_amount': 80,
    'left_lookback': 8,
    'tolerance': 30,
    'tp_pct': 5,
    'sl_pct': 10,
    'direction': 'short'
}

STRAT_E = {
    'id': 'parity_short',
    'name': 'parity_short',
    'timeframe': '4H',
    'sell_after_ncandles': 50,
    'order_amount': 10,
    'lookback': 150,
    'tolerance': 20,
    'tp_pct': 5,  
    'sl_pct': 10,  
    'direction': 'short'
}

STRATEGIES = [STRAT_A, STRAT_B, STRAT_C, STRAT_D, STRAT_E]

# Funciones comunes
connect_common      = connect_bitget_00
send_request_common = send_request_00
get_balance_common  = get_usdt_balance_00

# ==========================================================================
# SIGNALS & STRATEGIES
# ==========================================================================        
def detect_signal_for_strategy(strategy, final_symbols):
    """
    Detecta señales de trading para una estrategia específica.
    Devuelve lista de dicts {'symbol', 'timestamp', 'close'}.
    """
    detected = []
    if not final_symbols:
        return detected
    
    ohlcv = fetch_ohlcv_data(final_symbols, strategy['timeframe'])
    for sym, df in ohlcv.items():
        if df is None or df.empty:
            continue
        
        df_norm = normalize_live_ohlcv(df)
        arr = df_to_arrays_live(df_norm)
        
        # Obtener señales según estrategia
        try:
            if strategy['name'] == 'double_top_long':
                signals = double_top_long(
                    arr,
                    lookback_minor=strategy['lookback'],
                    price_tolerance=strategy['tolerance'],
                    trend_th=strategy['trend_th'],
                    live_trading=True
                )
            elif strategy['name'] == 'reversal_long':
                signals = reversal_long(
                    arr,
                    left_lookback=strategy['left_lookback'],
                    tolerance=strategy['tolerance'],
                    live_trading=True
                )
            elif strategy['name'] == 'parity_long':
                signals = parity_long(
                    arr,
                    lookback=strategy['lookback'],
                    tolerance=strategy['tolerance'],
                    live_trading=True
                )
            elif strategy['name'] == 'reversal_short':
                signals = reversal_short(
                    arr,
                    left_lookback=strategy['left_lookback'],
                    tolerance=strategy['tolerance'],
                    live_trading=True
                )
            elif strategy['name'] == 'parity_short':
                signals = parity_short(
                    arr,
                    lookback=strategy['lookback'],
                    tolerance=strategy['tolerance'],
                    live_trading=True
                )
            else:
                continue
        except Exception as e:
            print(f"❌ Error looking for signals {sym} ({strategy['name']}): {e}")
            continue
        
        # Verificar si hay señal en la última vela
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
# LOOP PRINCIPAL
# ==========================================================================  

def main_loop():
    global OPEN_POSITIONS, STRATEGY_CANDLES
    
    print(f"\n{'=' * 115}")
    print("🤖 === Starting multi-strategy multi-timeframe bot... ===")
    print(f"{'=' * 115}")
    OPEN_POSITIONS, STRATEGY_CANDLES = load_state(STATE_FILE)
    exchange    = connect_common()
    all_symbols = get_futures_symbols_from_api(PRODUCT_TYPE)
    
    final_by_strat = {}
    for strat in STRATEGIES:
        final_by_strat[strat['id']] = load_final_symbols(all_symbols,strategy=strat['name'],timeframe=strat['timeframe'])
        print(f"🔹 Strategy {strat['id']} ({strat['timeframe']}): {len(final_by_strat[strat['id']])} symbols")
    
    strategies_by_tf  = group_strategies_by_timeframe(STRATEGIES)
    unique_timeframes = get_unique_timeframes(STRATEGIES)
    
    print(f"\n➡️  Detected timeframes: {', '.join(unique_timeframes)}")
    for tf in unique_timeframes:
        strat_names = [s['id'] for s in strategies_by_tf[tf]]
        print(f"   🔹 {tf}: {', '.join(strat_names)}")
    
    print("\n✅ BOT Initialization completed\n")
    bot_metrics()
    
    # ⭐ Calcular next_candle_time para cada timeframe
    next_candle_times = {}
    for tf in unique_timeframes:
        next_candle_times[tf] = calculate_next_candle_time(tf, hour_zone=HOUR_ZONE)
        print(f"⏰ Next candle for {tf}: {next_candle_times[tf].strftime('%Y-%m-%d %H:%M:%S')} UTC")

    last_tpsl_check = time.time()
    
    try:
        while True:
            current_time = time.time()
            now_datetime = datetime.now(HOUR_ZONE)
            
            # ⭐ Verificar qué timeframes cerraron vela
            closed_timeframes = []
            for tf in unique_timeframes:
                if now_datetime >= next_candle_times[tf]:
                    closed_timeframes.append(tf)
            
            # Si al menos un timeframe cerró vela, procesar
            if closed_timeframes:
                print(f"\n{'=' * 115}")
                print(f"🔀 New candle(s) detected {now_datetime.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                print(f"🔹 Timeframes: {', '.join(closed_timeframes)}")

                # Sincronizar con broker una sola vez
                sync_broker(OPEN_POSITIONS, STRATEGY_CANDLES, STATE_FILE, send_request_common)
                
                now = datetime.now(HOUR_ZONE).strftime('%Y-%m-%d %H:%M:%S')
                print(f"📡 Signal search - {now}")
                print(f"{'-' * 115}")
                
                # ⭐ Procesar solo las estrategias de los timeframes que cerraron
                strategies_to_process = []
                for tf in closed_timeframes:
                    strategies_to_process.extend(strategies_by_tf[tf])
                
                # Incrementar contador de velas y chequear timeouts
                for strat in strategies_to_process:
                    strat_id = strat['id']
                    has_positions = strat_id in OPEN_POSITIONS and len(OPEN_POSITIONS[strat_id]) > 0
                    
                    if has_positions:
                        increment_strategy_candles(strat_id, STRATEGY_CANDLES, OPEN_POSITIONS, STATE_FILE)
                        candles = STRATEGY_CANDLES.get(strat_id, 0)
                        print(f"➡️  {strat_id} ({strat['timeframe']}): {candles}/{strat['sell_after_ncandles']} candles")
                        check_candles_timeout_for_strategy(strat_id, strat['sell_after_ncandles'],OPEN_POSITIONS, STRATEGY_CANDLES, STATE_FILE, send_request_common)
                
                # Búsqueda de señales (solo si no hay posiciones)
                for strat in strategies_to_process:
                    strat_id = strat['id']
                    num_positions = len(OPEN_POSITIONS.get(strat_id, []))
                    
                    if num_positions > 0:
                        print(f"🚫 Skipping {strat_id} ({strat['timeframe']}) - {num_positions} open positions")
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
                        import traceback
                        traceback.print_exc()
                        
                        # ⭐ SECOND TRY
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
                print(f"{'=' * 115}\n")
                
                # ⭐ Recalcular next_candle_time para los timeframes que cerraron
                for tf in closed_timeframes:
                    next_candle_times[tf] = calculate_next_candle_time(tf, hour_zone=HOUR_ZONE)
                    print(f"⏰ Next candle for {tf}: {next_candle_times[tf].strftime('%Y-%m-%d %H:%M:%S')} UTC")
                
                # Resetear el tiempo del último chequeo TP/SL
                last_tpsl_check = time.time()
            
            # Chequeo periódico de TP/SL cada CHECK_INTERVAL segundos
            else:
                if current_time - last_tpsl_check >= CHECK_INTERVAL:
                    check_all_tp_sl(STRATEGIES, OPEN_POSITIONS, STRATEGY_CANDLES,STATE_FILE, send_request_common, HOUR_ZONE)
                    last_tpsl_check = current_time
            
            # Pequeña pausa para no saturar el CPU
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🔚 Interrupted by user.")
        save_state_local(OPEN_POSITIONS, STRATEGY_CANDLES, STATE_FILE)
        print("⛔ BOT Stopped")

if __name__ == '__main__':
    main_loop()