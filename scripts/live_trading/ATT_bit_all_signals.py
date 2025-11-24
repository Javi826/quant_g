#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bot multi-estrategia con monitoreo activo de TP/SL y persistencia de estado.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import threading
import json
import numpy as np
from datetime import datetime
from zoneinfo import ZoneInfo
from decimal import Decimal

# --- Imports de tus módulos ---
from parquet_process.Z_parquet_A0_extraction import get_futures_symbols_from_api
from ZX_utils_live import wait_for_next_candle,load_final_symbols
from ZX_utils_bot import  get_fills_for_order, place_order,calculate_tp_sl_prices
from ZX_utils_bot import  get_current_price,close_position
from ZX_utils_bot import  save_state,load_state
from ZX_utils_live import fetch_ohlcv_data,normalize_live_ohlcv,df_to_arrays_live
from Z_add_signals_reversal import trend_reversal_entry_short
from Z_add_signals_parity import detect_parity_short

from utils.ZZ_connect import connect_bitget_TT
from ZX_connect_live import get_usdt_balance_TT, send_request_TT

HOUR_ZONE                 = ZoneInfo('UTC')
PRODUCT_TYPE              = 'USDT-FUTURES'
MIN_TIMEFRAME             = '5m'
CHECK_INTERVAL            = 30  
PAUSE_INTERVAL            = 10
PAUSE_TPSL_AROUND_SIGNALS = False   
USE_HARDCODED_SIGNALS     = False

# Archivo de estado
STATE_FILE = 'bot_state.json'

# Control de pausa para TP/SL
TPSL_PAUSED     = False
TPSL_PAUSE_LOCK = threading.Lock()


def get_hardcoded_signals(strat_id):
    """Genera señales de prueba para testing"""
    symbols = ['BTCUSDT', 'BNBUSDT']
    signals = []
    for symbol in symbols:
        code, resp = send_request_common("GET", "/api/v2/mix/market/ticker",params={"productType": PRODUCT_TYPE, "symbol": symbol})
        current_price = 50000.0
        if code == 200 and isinstance(resp, dict) and resp.get("code") == "00000":
            try:
                current_price = float(resp['data'][0]['lastPr'])
            except:
                pass
        signals.append({
            'symbol': symbol,
            'close': current_price,
            'timestamp': datetime.now(HOUR_ZONE).isoformat()
        })
    return signals

# ----------------------
# Configuración de Estrategias
# ----------------------
STRAT_A = {
    'id': 'revers_short',
    'name': 'reversal_short',
    'timeframe': '5m',
    'sell_after_ncandles': 3,
    'order_amount': 10,
    'left_lookback': 8,
    'tolerance': 30,
    'tp_pct': 5.0,
    'sl_pct': 5.0,
    'direction': 'short'
}

STRAT_B = {
    'id': 'parity_short',
    'name': 'parity_short',
    'timeframe': '5m',
    'sell_after_ncandles': 2,
    'order_amount': 20,
    'lookback': 150,
    'tolerance': 20,
    'tp_pct': 5.0,  
    'sl_pct': 5.0,  
    'direction': 'short'
}

STRATEGIES = [STRAT_A, STRAT_B]

# Funciones comunes
connect_common      = connect_bitget_TT
send_request_common = send_request_TT
get_balance_common  = get_usdt_balance_TT

# ----------------------
# Registro de posiciones abiertas por estrategia
# ----------------------
OPEN_POSITIONS   = {}
POSITIONS_LOCK   = threading.Lock()
STOP_THREADS     = False
STRATEGY_CANDLES = {}

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
            if strategy['name'] == 'reversal_short':
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
            # fallback: intentar convertir a lista
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
            # si comparación falla, intentar comparación escalar
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

def process_strategy(strat, final_symbols, exchange, use_hardcoded=False):
    """Procesa una estrategia: busca señales y abre posiciones"""
    strat_id = strat['id']
    
    print(f"\n{'─' * 40}")
    print(f"🔄 Procesando estrategia: {strat_id}")
    print(f"{'─' * 40}")
    
    # Detectar señales
    if use_hardcoded:
        signals = get_hardcoded_signals(strat_id)
    else:
        signals = detect_signal_for_strategy(strat, final_symbols)
    
    print(f"✨ Señales detectadas para {strat_id}: {len(signals)}")
    
    # Si no hay señales, salir
    if not signals:
        return
    
    # NUEVO: Resetear contador de velas al abrir nuevas posiciones
    reset_strategy_candles(strat_id)
    
    # Procesar cada señal
    for sig in signals:
        # Verificar saldo disponible
        usdt_balance = get_balance_common(exchange)
        if usdt_balance < strat['order_amount']:
            print(f"🔶 Saldo insuficiente ({usdt_balance:.2f} USDT) para {sig['symbol']}")
            continue
        
        print(f"\n▶️ Abriendo {strat['direction']} en {sig['symbol']} para {strat_id}...")
        
        # Colocar orden
        resp_order = place_order(
            symbol=sig['symbol'],
            direction=strat['direction'],
            usdt_amount=strat['order_amount'],
            send_request_func=send_request_common
        )
        
        if resp_order is None:
            print(f"❌ Error al colocar orden para {sig['symbol']}")
            continue
        
        # Extraer datos de la orden ejecutada
        data = resp_order.get('data', {}) if isinstance(resp_order, dict) else {}
        order_id = data.get('orderId')
        
        if order_id:
            # Obtener size real y precio real
            filled_size, entry_price_from_fills = get_fills_for_order(order_id=order_id, symbol=sig['symbol'], send_request_func=send_request_common)
            time.sleep(0.5)
            
            if filled_size is None or filled_size == 0:
                size        = Decimal(str(data.get('size', data.get('filledQty', data.get('baseVolume', 0)))))
                entry_price = Decimal(str(data.get('price', sig.get('close', 0))))
            else:
                size = filled_size
                entry_price = entry_price_from_fills if entry_price_from_fills is not None else Decimal(str(sig.get('close', 0)))
            
            print(f"▶️ Orden ejecutada - ID: {order_id}")
            
            # Registrar posición (esto también guarda el estado)
            add_position(strat_id=strat_id,symbol=sig['symbol'],size=size,entry_price=entry_price,direction=strat['direction'],tp_pct=strat['tp_pct'],sl_pct=strat['sl_pct'],order_id=order_id)
        else:
            print(f"🔶 Orden ejecutada pero sin orderId en respuesta")
        
        time.sleep(0.5)
      
# ==========================================================================
# INCREMENTAL & POSITONS
# ==========================================================================   
def increment_strategy_candles(strat_id):
    """Incrementa el contador de velas de una estrategia"""
    if strat_id not in STRATEGY_CANDLES:
        STRATEGY_CANDLES[strat_id] = 0
    
    STRATEGY_CANDLES[strat_id] += 1
    
    save_state(OPEN_POSITIONS, STRATEGY_CANDLES, STATE_FILE, POSITIONS_LOCK)

def reset_strategy_candles(strat_id):
    """Resetea el contador de velas de una estrategia (cuando abre nuevas posiciones)"""
    STRATEGY_CANDLES[strat_id] = 0
    save_state(OPEN_POSITIONS, STRATEGY_CANDLES, STATE_FILE, POSITIONS_LOCK)
    
def add_position(strat_id, symbol, size, entry_price, direction, tp_pct, sl_pct, order_id):
    """Registra una nueva posición abierta"""
    with POSITIONS_LOCK:
        if strat_id not in OPEN_POSITIONS:
            OPEN_POSITIONS[strat_id] = []
        
        tp_price, sl_price = calculate_tp_sl_prices(entry_price, direction, tp_pct, sl_pct)
        
        position = {'symbol': symbol,'size': size,'entry_price': entry_price,'direction': direction,'tp': tp_price,'sl': sl_price,'order_id': order_id,'opened_at': datetime.now(HOUR_ZONE)}       
        OPEN_POSITIONS[strat_id].append(position)
        
        print(f"▶️ Posición registrada:")
        print(f"  Symbol: {symbol} | Size: {size} | Entry: {entry_price}")
        print(f"  TP: {tp_price} | SL: {sl_price}")
    
    # Guardar estado actualizado
    save_state(OPEN_POSITIONS, STRATEGY_CANDLES, STATE_FILE, POSITIONS_LOCK)
    
# ==========================================================================
# CHECKINGS & MONITORING
# ==========================================================================   
def check_tp_sl_for_strategy(strat_id):
    """Comprueba TP/SL para todas las posiciones de una estrategia"""
    with POSITIONS_LOCK:
        if strat_id not in OPEN_POSITIONS or not OPEN_POSITIONS[strat_id]:
            return
        
        positions = OPEN_POSITIONS[strat_id][:]
    
    positions_to_remove = []
    
    for i, pos in enumerate(positions):
        symbol = pos['symbol']
        current_price = get_current_price(symbol, send_request_func=send_request_common)
        
        if current_price is None:
            print(f"🔶 No se pudo obtener precio para {symbol}")
            continue
        
        direction = pos['direction']
        tp_price = pos['tp']
        sl_price = pos['sl']
        entry_price = pos['entry_price']
        
        current_price = Decimal(str(current_price))
        
        # Calcular distancias al TP y SL
        if direction.lower() == 'short':
            dist_to_tp = float(current_price - tp_price)
            dist_to_sl = float(sl_price - current_price)
            tp_pct_away = (dist_to_tp / float(entry_price)) * 100
            sl_pct_away = (dist_to_sl / float(entry_price)) * 100
        else:  # long
            dist_to_tp = float(tp_price - current_price)
            dist_to_sl = float(current_price - sl_price)
            tp_pct_away = (dist_to_tp / float(entry_price)) * 100
            sl_pct_away = (dist_to_sl / float(entry_price)) * 100
        
        print(f"  [{symbol}] {direction.upper()}")
        print(f"    Current: {current_price} | Entry: {entry_price}")
        print(f"    TP: {tp_price} (Δ {tp_pct_away:+.3f}%) | SL: {sl_price} (Δ {sl_pct_away:+.3f}%)")
        
        # Verificar si se alcanzó TP o SL
        hit_tp = False
        hit_sl = False
        
        if direction.lower() == 'long':
            hit_tp = current_price >= tp_price
            hit_sl = current_price <= sl_price
        else:  # short
            hit_tp = current_price <= tp_price
            hit_sl = current_price >= sl_price
        
        if hit_tp:
            print(f"\n🌟 TP ALCANZADO para {symbol} ({strat_id})")
            print(f"   Entry: {entry_price} | Current: {current_price} | TP: {tp_price}")
            if close_position(symbol, pos['size'], direction, send_request_common,reason="TP"):
                positions_to_remove.append(i)
        
        elif hit_sl:
            print(f"\n🔻 SL ALCANZADO para {symbol} ({strat_id})")
            print(f"   Entry: {entry_price} | Current: {current_price} | SL: {sl_price}")
            if close_position(symbol, pos['size'], direction, send_request_common,reason="SL"):
                positions_to_remove.append(i)
    
    # Eliminar posiciones cerradas y guardar estado
    if positions_to_remove:
        with POSITIONS_LOCK:
            for i in reversed(positions_to_remove):
                if i < len(OPEN_POSITIONS[strat_id]):
                    OPEN_POSITIONS[strat_id].pop(i)
        
        # Guardar estado actualizado
        save_state(OPEN_POSITIONS, STRATEGY_CANDLES, STATE_FILE, POSITIONS_LOCK)

def check_candles_timeout_for_strategy(strat_id, sell_after_ncandles):
    """Cierra todas las posiciones de una estrategia si superan el límite de velas"""
    candles_elapsed = STRATEGY_CANDLES.get(strat_id, 0)
    
    if candles_elapsed < sell_after_ncandles:
        return
    
    with POSITIONS_LOCK:
        if strat_id not in OPEN_POSITIONS or not OPEN_POSITIONS[strat_id]:
            return
        
        positions = OPEN_POSITIONS[strat_id][:]
    
    if not positions:
        return
    
    print(f"\n▶️ TIMEOUT ALCANZADO para estrategia {strat_id}")
    print(f"▶️ Velas transcurridas: {candles_elapsed}/{sell_after_ncandles}")
    print(f"▶️ Cerrando {len(positions)} posiciones...")
    
    # Cerrar todas las posiciones de la estrategia
    all_closed = True
    for pos in positions:
        if not close_position(pos['symbol'], pos['size'], pos['direction'],send_request_common, reason="TIMEOUT"):
            all_closed = False
    
    # Si se cerraron todas, limpiar la estrategia
    if all_closed:
        with POSITIONS_LOCK:
            OPEN_POSITIONS[strat_id] = []
            STRATEGY_CANDLES[strat_id] = 0
        save_state(OPEN_POSITIONS, STRATEGY_CANDLES, STATE_FILE, POSITIONS_LOCK)

# ==========================================================================
# LOOPS
# ==========================================================================  
 
def tpsl_monitor_thread():
    """Thread que chequea TP/SL cada CHECK_INTERVAL segundos"""
    print(f"▶️ Thread de monitoreo TP/SL iniciado (cada {CHECK_INTERVAL}s)")
    
    while not STOP_THREADS:
        try:
            # Verificar si está pausado
            with TPSL_PAUSE_LOCK:
                if TPSL_PAUSED:
                    time.sleep(0.5)
                    continue
            
            now = datetime.now(HOUR_ZONE).strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n{'─' * 60}")
            print(f"🔎 Chequeo TP/SL - {now}")
            print(f"{'─' * 60}")
            
            for strat in STRATEGIES:
                strat_id = strat['id']
                with POSITIONS_LOCK:
                    num_positions = len(OPEN_POSITIONS.get(strat_id, []))
                
                if num_positions > 0:
                    print(f"\n🔹Estrategia {strat_id}: {num_positions} posiciones abiertas")
                    check_tp_sl_for_strategy(strat_id)
            
            time.sleep(CHECK_INTERVAL)
            
        except Exception as e:
            print(f"🔶 Error en thread de monitoreo TP/SL: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(0.5)
            
def signals_loop(final_by_strat, exchange):
    """Loop que busca señales en cada nueva vela"""
    global TPSL_PAUSED
    print("▶️ Thread de búsqueda de señales iniciado")
    
    while not STOP_THREADS:
        try:
            # Pausar TP/SL antes de buscar señales
            if PAUSE_TPSL_AROUND_SIGNALS:
                with TPSL_PAUSE_LOCK:
                    TPSL_PAUSED = True
                print("⏸️ TP/SL pausado para búsqueda de señales")
                time.sleep(PAUSE_INTERVAL)  # Espera equivalente a -1 ciclo
            
            wait_for_next_candle(MIN_TIMEFRAME)
            
            now = datetime.now(HOUR_ZONE).strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n{'=' * 60}")
            print(f"📡 Búsqueda de señales - {now}")
            print(f"{'=' * 60}")
            
            # NUEVO: Incrementar contador e chequear timeouts
            for strat in STRATEGIES:
                strat_id = strat['id']
                
                with POSITIONS_LOCK:
                    has_positions = strat_id in OPEN_POSITIONS and len(OPEN_POSITIONS[strat_id]) > 0
                
                if has_positions:
                    increment_strategy_candles(strat_id)
                    candles = STRATEGY_CANDLES.get(strat_id, 0)
                    print(f"▶️ {strat_id}: {candles}/{strat['sell_after_ncandles']} velas")
                    check_candles_timeout_for_strategy(strat_id, strat['sell_after_ncandles'])
            
            # Búsqueda de señales (solo si no hay posiciones)
            for strat in STRATEGIES:
                strat_id = strat['id']
                
                with POSITIONS_LOCK:
                    num_positions = len(OPEN_POSITIONS.get(strat_id, []))
                
                if num_positions > 0:
                    print(f"🚫 Saltando búsqueda de señales para {strat_id} (tiene {num_positions} posiciones abiertas)")
                    continue
                
                try:
                    process_strategy(
                        strat=strat,
                        final_symbols=final_by_strat.get(strat_id, []),
                        exchange=exchange,
                        use_hardcoded=USE_HARDCODED_SIGNALS
                    )
                except Exception as e:
                    print(f"🔶 Error procesando {strat_id}: {e}")
                    import traceback
                    traceback.print_exc()
            
            print(f"\n{'=' * 60}")
            print("🔷 Ciclo de señales completado")
            print(f"{'=' * 60}\n")
            
            # Mantener pausado después de buscar señales
            if PAUSE_TPSL_AROUND_SIGNALS:
                print("⏸️ TP/SL pausado post-señales")
                time.sleep(PAUSE_INTERVAL)  # Espera equivalente a +1 ciclo
                with TPSL_PAUSE_LOCK:
                    TPSL_PAUSED = False
                print("▶️ TP/SL reanudado")
            
        except Exception as e:
            print(f"🔶 Error en loop de señales: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)

def main_loop():
    """Loop principal del bot"""

    global STOP_THREADS, OPEN_POSITIONS, STRATEGY_CANDLES
    
    print("🚀 Iniciando bot multi-estrategia con persistencia de estado...")
    
    OPEN_POSITIONS, STRATEGY_CANDLES = load_state(STATE_FILE) 
    exchange                         = connect_common()    
    all_symbols                      = get_futures_symbols_from_api(PRODUCT_TYPE)
    
    # Cargar símbolos finales por estrategia
    final_by_strat = {}
    for strat in STRATEGIES:
        final_by_strat[strat['id']] = load_final_symbols(all_symbols,strategy=strat['name'],timeframe=strat['timeframe'])
        print(f"▶️ Estrategia {strat['id']}: {len(final_by_strat[strat['id']])} símbolos")
    
    print("▶️ Inicialización completada\n")
    print("=" * 60)
    
    # Iniciar thread de monitoreo TP/SL
    tpsl_thread = threading.Thread(target=tpsl_monitor_thread, daemon=True)
    tpsl_thread.start()
    
    # Iniciar thread de búsqueda de señales
    signals_thread = threading.Thread(
        target=signals_loop, 
        args=(final_by_strat, exchange), 
        daemon=True
    )
    signals_thread.start()
    
    try:
        while True:
            time.sleep(0.5)
            
    except KeyboardInterrupt:
        print("\n🚨 Interrumpido por usuario. Deteniendo threads...")
        STOP_THREADS = True
        
        # Guardar estado final antes de cerrar
        print("💾 Guardando estado final...")
        save_state(OPEN_POSITIONS, STRATEGY_CANDLES, STATE_FILE, POSITIONS_LOCK)
        
        tpsl_thread.join(timeout=5)
        signals_thread.join(timeout=5)
        print("▶️ Bot detenido correctamente")


if __name__ == '__main__':
    main_loop()