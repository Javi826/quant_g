#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bot multi-estrategia con monitoreo activo de TP/SL y persistencia de estado.
Versión con un solo bucle principal (sin threads).
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from datetime import datetime
import numpy as np
from zoneinfo import ZoneInfo

# --- Imports de tus módulos ---
from parquet_process.Z_parquet_A0_extraction import get_futures_symbols_from_api
from ZX_utils_live import load_final_symbols,fetch_ohlcv_data, normalize_live_ohlcv, df_to_arrays_live
from ZX_utils_bot import increment_strategy_candles,process_strategy,check_all_tp_sl
from ZX_utils_bot import load_state,save_state_local,calculate_next_candle_time,check_candles_timeout_for_strategy
from Z_add_signals_double_top import detect_double_top_long
from Z_add_signals_reversal import trend_reversal_entry_long
from Z_add_signals_parity import detect_parity_long
from Z_add_signals_reversal import trend_reversal_entry_short
from Z_add_signals_parity import detect_parity_short

from utils.ZZ_connect import connect_bitget_00
from ZX_connect_live import get_usdt_balance_00, send_request_00

HOUR_ZONE                 = ZoneInfo('UTC')
PRODUCT_TYPE              = 'USDT-FUTURES'
MIN_TIMEFRAME             = '4H'
CHECK_INTERVAL            = 30   
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
    'sell_after_ncandles': 45,
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
    'sell_after_ncandles': 45,
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
    'sell_after_ncandles': 45,
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
    'sell_after_ncandles': 45,
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
    'sell_after_ncandles': 45,
    'order_amount': 80,
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
    Normaliza la salida de las funciones de señal y evita evaluar arrays directamente.
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

        # obtener señales según estrategia
        try:
            if strategy['name'] == 'double_top_long':
                signals = detect_double_top_long(
                    arr,
                    lookback_minor=strategy['lookback'],
                    price_tolerance=strategy['tolerance'],
                    trend_th=strategy['trend_th'],
                    live_trading=True
                )
            elif strategy['name'] == 'reversal_long':
                signals = trend_reversal_entry_long(
                    arr,
                    left_lookback=strategy['left_lookback'],
                    tolerance=strategy['tolerance'],
                    live_trading=True
                )
            elif strategy['name'] == 'parity_long':
                signals = detect_parity_long(
                    arr,
                    lookback=strategy['lookback'],
                    tolerance=strategy['tolerance'],
                    live_trading=True
                )
            elif strategy['name'] == 'reversal_short':
                signals = trend_reversal_entry_short(
                    arr,
                    left_lookback=strategy['left_lookback'],
                    tolerance=strategy['tolerance'],
                    live_trading=True
                )
            elif strategy['name'] == 'parity_short':
                signals = detect_parity_short(
                    arr,
                    lookback=strategy['lookback'],
                    tolerance=strategy['tolerance'],
                    live_trading=True
                )
            else:
                signals = None
        except Exception as e:
            print(f"⚠️ Error ejecutando la función de señales para {sym} ({strategy['name']}): {e}")
            signals = None


        # Normalizar signals para evitar truthiness ambiguo
        if signals is None:
            continue

        # convertir a array numpy para inspección segura
        try:
            signals_arr = np.asarray(signals)
        except Exception:
            try:
                signals_arr = np.array(list(signals))
            except Exception:
                continue

        if signals_arr.size == 0:
            continue
        last     = signals_arr.flat[-1]
        last_arr = np.asarray(last)

        # si cualquier elemento del último valor es distinto de 0, consideramos señal
        try:
            has_signal = np.any(last_arr != 0)
        except Exception:
            try:
                has_signal = (float(last_arr) != 0.0)
            except Exception:
                has_signal = False

        if has_signal:
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
    """Loop principal del bot - versión sin threads"""
    global OPEN_POSITIONS, STRATEGY_CANDLES
    
    print("🚀 Iniciando bot multi-estrategia con persistencia de estado...")
    
    # Cargar estado previo
    OPEN_POSITIONS, STRATEGY_CANDLES = load_state(STATE_FILE)
    exchange = connect_common()
    all_symbols = get_futures_symbols_from_api(PRODUCT_TYPE)
    
    # Cargar símbolos finales por estrategia
    final_by_strat = {}
    for strat in STRATEGIES:
        final_by_strat[strat['id']] = load_final_symbols(all_symbols,strategy=strat['name'],timeframe=strat['timeframe'])
        print(f"▶️ Estrategia {strat['id']}: {len(final_by_strat[strat['id']])} símbolos")
    
    print("▶️ Inicialización completada\n")
    print("=" * 60)
    
    # Calcular la próxima vela
    next_candle_time = calculate_next_candle_time(MIN_TIMEFRAME, hour_zone=HOUR_ZONE)

    last_tpsl_check  = time.time()
    
    try:
        while True:
            current_time = time.time()
            now_datetime = datetime.now(HOUR_ZONE)
            
            # Verificar si llegó el momento de buscar señales (nueva vela)
            if now_datetime >= next_candle_time:
                print('\n')
                print(f"🔷 === Nueva vela detectada ===: {now_datetime.strftime('%Y-%m-%d %H:%M:%S')} UTC")
                print('\n')
                
                now = datetime.now(HOUR_ZONE).strftime('%Y-%m-%d %H:%M:%S')
                print(f"\n{'=' * 60}")
                print(f"📡 Búsqueda de señales - {now}")
                print(f"{'=' * 60}")
                
                # Incrementar contador de velas y chequear timeouts
                for strat in STRATEGIES:
                    strat_id = strat['id']
                    has_positions = strat_id in OPEN_POSITIONS and len(OPEN_POSITIONS[strat_id]) > 0
                    
                    if has_positions:
                        increment_strategy_candles(strat_id, STRATEGY_CANDLES, OPEN_POSITIONS, STATE_FILE)

                        candles = STRATEGY_CANDLES.get(strat_id, 0)
                        print(f"▶️ {strat_id}: {candles}/{strat['sell_after_ncandles']} velas")
                        check_candles_timeout_for_strategy(strat_id, strat['sell_after_ncandles'], OPEN_POSITIONS, STRATEGY_CANDLES, STATE_FILE, send_request_common)

                
                # Búsqueda de señales (solo si no hay posiciones)
                for strat in STRATEGIES:
                    strat_id = strat['id']
                    num_positions = len(OPEN_POSITIONS.get(strat_id, []))
                    
                    if num_positions > 0:
                        print(f"🚫 Saltando búsqueda de señales para {strat_id} (tiene {num_positions} posiciones abiertas)")
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
                        print(f"🔶 Error procesando {strat_id}: {e}")
                        import traceback
                        traceback.print_exc()
                
                print(f"\n{'=' * 60}")
                print("🔷 Ciclo de señales completado")
                print(f"{'=' * 60}\n")
                
                # Calcular la siguiente vela & Resetear el tiempo del último chequeo TP/SL
                next_candle_time = calculate_next_candle_time(MIN_TIMEFRAME)               
                last_tpsl_check  = time.time()
            
            # Chequeo periódico de TP/SL cada CHECK_INTERVAL segundos
            else:
                if current_time - last_tpsl_check >= CHECK_INTERVAL:
                    check_all_tp_sl(STRATEGIES, OPEN_POSITIONS, STRATEGY_CANDLES, STATE_FILE, send_request_common, HOUR_ZONE)
                    last_tpsl_check = current_time
            
            # Pequeña pausa para no saturar el CPU
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🚨 Interrumpido por usuario. Guardando estado...")
        save_state_local(OPEN_POSITIONS, STRATEGY_CANDLES, STATE_FILE)
        print("▶️ Bot detenido correctamente")


if __name__ == '__main__':
    main_loop()
