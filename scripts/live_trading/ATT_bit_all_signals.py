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
from datetime import datetime
from zoneinfo import ZoneInfo
from decimal import Decimal

# --- Imports de tus módulos ---
from parquet_process.Z_parquet_A0_extraction import get_futures_symbols_from_api
from ZX_utils_live import wait_for_next_candle, get_fills_for_order, load_final_symbols, detect_signal_for_strategy

from utils.ZZ_connect import connect_bitget_TT
from ZX_connect_live import get_usdt_balance_TT, send_request_TT
from ZX_place_orders import place_order

MADRID_TZ = ZoneInfo('Europe/Madrid')
PRODUCT_TYPE = 'USDT-FUTURES'
MIN_TIMEFRAME = '5m'
CHECK_INTERVAL = 60  # segundos
PAUSE_TPSL_AROUND_SIGNALS = False  # 

# Archivo de estado
STATE_FILE = 'bot_state.json'

# Control de pausa para TP/SL
TPSL_PAUSED = False
TPSL_PAUSE_LOCK = threading.Lock()

# ----------------------
# TESTING: Señales Hardcodeadas
# ----------------------
USE_HARDCODED_SIGNALS = True

def get_hardcoded_signals(strat_id):
    """Genera señales de prueba para testing"""
    symbols = ['BTCUSDT', 'BNBUSDT']
    signals = []
    for symbol in symbols:
        code, resp = send_request_common("GET", "/api/v2/mix/market/ticker",
                                         params={"productType": PRODUCT_TYPE, "symbol": symbol})
        current_price = 50000.0
        if code == 200 and isinstance(resp, dict) and resp.get("code") == "00000":
            try:
                current_price = float(resp['data'][0]['lastPr'])
            except:
                pass
        signals.append({
            'symbol': symbol,
            'close': current_price,
            'timestamp': datetime.now(MADRID_TZ).isoformat()
        })
    return signals

# ----------------------
# Configuración de Estrategias
# ----------------------
STRAT_A = {
    'id': 'revers_short',
    'name': 'reversal_short',
    'timeframe': '5m',
    'sell_after_ncandles': 5,
    'order_amount': 10,
    'left_lookback': 8,
    'tolerance': 30,
    'tp_pct': 10,
    'sl_pct': 10,
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
    'tp_pct': 0.1,  
    'sl_pct': 0.1,  
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
OPEN_POSITIONS = {}
POSITIONS_LOCK = threading.Lock()
STOP_THREADS   = False


# ==============================
# PERSISTENCIA DE ESTADO
# ==============================

def load_state():
    """Carga el estado desde el archivo JSON"""
    global OPEN_POSITIONS
    
    if not os.path.exists(STATE_FILE):
        print("📂 No se encontró archivo de estado previo")
        return
    
    try:
        with open(STATE_FILE, 'r') as f:
            data = json.load(f)
        
        # Convertir strings a Decimal donde sea necesario
        for strat_id, positions in data.items():
            OPEN_POSITIONS[strat_id] = []
            for pos in positions:
                OPEN_POSITIONS[strat_id].append({
                    'symbol': pos['symbol'],
                    'size': Decimal(pos['size']),
                    'entry_price': Decimal(pos['entry_price']),
                    'direction': pos['direction'],
                    'tp': Decimal(pos['tp']),
                    'sl': Decimal(pos['sl']),
                    'order_id': pos['order_id'],
                    'opened_at': datetime.fromisoformat(pos['opened_at'])
                })
        
        total_positions = sum(len(positions) for positions in OPEN_POSITIONS.values())
        print(f"🔹Estado cargado: {total_positions} posiciones recuperadas")
        
        # Mostrar resumen
        for strat_id, positions in OPEN_POSITIONS.items():
            if positions:
                print(f"   ▶️ {strat_id}: {len(positions)} posiciones")
                for pos in positions:
                    print(f"      - {pos['symbol']} | Size: {pos['size']} | Entry: {pos['entry_price']}")
        
    except Exception as e:
        print(f"🔶 Error cargando estado: {e}")
        import traceback
        traceback.print_exc()


def save_state():
    """Guarda el estado actual en el archivo JSON"""
    try:
        with POSITIONS_LOCK:
            # Convertir Decimal y datetime a formato serializable
            serializable_data = {}
            for strat_id, positions in OPEN_POSITIONS.items():
                serializable_data[strat_id] = []
                for pos in positions:
                    serializable_data[strat_id].append({
                        'symbol': pos['symbol'],
                        'size': str(pos['size']),
                        'entry_price': str(pos['entry_price']),
                        'direction': pos['direction'],
                        'tp': str(pos['tp']),
                        'sl': str(pos['sl']),
                        'order_id': pos['order_id'],
                        'opened_at': pos['opened_at'].isoformat()
                    })
        
        # Guardar con formato legible
        with open(STATE_FILE, 'w') as f:
            json.dump(serializable_data, f, indent=2)
        
    except Exception as e:
        print(f"🔶 Error guardando estado: {e}")
        import traceback
        traceback.print_exc()


# ==============================
# FUNCIONES PRINCIPALES
# ==============================

def calculate_tp_sl_prices(entry_price, direction, tp_pct, sl_pct):
    """Calcula los precios de TP y SL basados en el precio de entrada"""
    entry = Decimal(str(entry_price))
    tp_decimal = Decimal(str(tp_pct)) / Decimal('100')
    sl_decimal = Decimal(str(sl_pct)) / Decimal('100')
    
    if direction.lower() == 'long':
        tp_price = entry * (Decimal('1') + tp_decimal)
        sl_price = entry * (Decimal('1') - sl_decimal)
    else:  # short
        tp_price = entry * (Decimal('1') - tp_decimal)
        sl_price = entry * (Decimal('1') + sl_decimal)
    
    return tp_price, sl_price


def get_current_price(symbol, send_request_func):
    """Obtiene el precio actual del mercado"""
    try:
        code, resp = send_request_func("GET", "/api/v2/mix/market/ticker",params={"productType": "USDT-FUTURES", "symbol": symbol})
        if code == 200 and resp.get("code") == "00000":
            return Decimal(str(resp['data'][0]['lastPr']))
    except Exception as e:
        print(f"🔶 Error obteniendo precio de {symbol}: {e}")
    return None


def close_position(symbol, size, direction, reason="TP/SL"):
    """Cierra una posición con orden market en HEDGE MODE"""
    try:
        close_side = "sell" if direction.lower() == "short" else "buy"
        
        body = {
            "symbol": symbol,
            "productType": PRODUCT_TYPE,
            "marginMode": "isolated",
            "marginCoin": "USDT",
            "size": format(size, "f"),
            "side": close_side,
            "tradeSide": "close",
            "orderType": "market"
        }
        
        print(f"🔄 Cerrando posición {direction} en {symbol}:")        
        code, resp = send_request_common("POST", "/api/v2/mix/order/place-order", body=body)
        
        if code == 200 and resp.get("code") == "00000":
            print(f"▶️ Posición cerrada por {reason}: {symbol} | Size: {size}")
            return True
        else:
            print(f"🔶 Error cerrando posición {symbol}: {resp}")
            if resp.get("code") == "22002":
                print(f"   → Removiendo del registro local (posición inexistente)")
                return True
            return False
            
    except Exception as e:
        print(f"❌ Error al cerrar posición {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return False


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
            if close_position(symbol, pos['size'], direction, reason="TP"):
                positions_to_remove.append(i)
        
        elif hit_sl:
            print(f"\n🔻 SL ALCANZADO para {symbol} ({strat_id})")
            print(f"   Entry: {entry_price} | Current: {current_price} | SL: {sl_price}")
            if close_position(symbol, pos['size'], direction, reason="SL"):
                positions_to_remove.append(i)
    
    # Eliminar posiciones cerradas y guardar estado
    if positions_to_remove:
        with POSITIONS_LOCK:
            for i in reversed(positions_to_remove):
                if i < len(OPEN_POSITIONS[strat_id]):
                    OPEN_POSITIONS[strat_id].pop(i)
        
        # Guardar estado actualizado
        save_state()


def add_position(strat_id, symbol, size, entry_price, direction, tp_pct, sl_pct, order_id):
    """Registra una nueva posición abierta"""
    with POSITIONS_LOCK:
        if strat_id not in OPEN_POSITIONS:
            OPEN_POSITIONS[strat_id] = []
        
        tp_price, sl_price = calculate_tp_sl_prices(entry_price, direction, tp_pct, sl_pct)
        
        position = {
            'symbol': symbol,
            'size': size,
            'entry_price': entry_price,
            'direction': direction,
            'tp': tp_price,
            'sl': sl_price,
            'order_id': order_id,
            'opened_at': datetime.now(MADRID_TZ)
        }
        
        OPEN_POSITIONS[strat_id].append(position)
        
        print(f"📝 Posición registrada:")
        print(f"   Symbol: {symbol} | Size: {size} | Entry: {entry_price}")
        print(f"   TP: {tp_price} | SL: {sl_price}")
    
    # Guardar estado actualizado
    save_state()


def tpsl_monitor_thread():
    """Thread que chequea TP/SL cada CHECK_INTERVAL segundos"""
    print(f"▶️ Thread de monitoreo TP/SL iniciado (cada {CHECK_INTERVAL}s)")
    
    while not STOP_THREADS:
        try:
            # Verificar si está pausado
            with TPSL_PAUSE_LOCK:
                if TPSL_PAUSED:
                    time.sleep(1)
                    continue
            
            now = datetime.now(MADRID_TZ).strftime('%Y-%m-%d %H:%M:%S')
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
            time.sleep(5)


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
            filled_size, entry_price_from_fills = get_fills_for_order(
                order_id=order_id, 
                symbol=sig['symbol'], 
                send_request_func=send_request_common
            )
            
            if filled_size is None or filled_size == 0:
                size = Decimal(str(data.get('size', data.get('filledQty', data.get('baseVolume', 0)))))
                entry_price = Decimal(str(data.get('price', sig.get('close', 0))))
            else:
                size = filled_size
                entry_price = entry_price_from_fills if entry_price_from_fills is not None else Decimal(str(sig.get('close', 0)))
            
            print(f"▶️ Orden ejecutada - ID: {order_id}")
            
            # Registrar posición (esto también guarda el estado)
            add_position(
                strat_id=strat_id,
                symbol=sig['symbol'],
                size=size,
                entry_price=entry_price,
                direction=strat['direction'],
                tp_pct=strat['tp_pct'],
                sl_pct=strat['sl_pct'],
                order_id=order_id
            )
        else:
            print(f"🔶 Orden ejecutada pero sin orderId en respuesta")
        
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
                time.sleep(CHECK_INTERVAL)  # Espera equivalente a -1 ciclo
            
            wait_for_next_candle(MIN_TIMEFRAME)
            
            now = datetime.now(MADRID_TZ).strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n{'=' * 60}")
            print(f"📡 Búsqueda de señales - {now}")
            print(f"{'=' * 60}")
            
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
                time.sleep(CHECK_INTERVAL)  # Espera equivalente a +1 ciclo
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
    global STOP_THREADS
    
    print("🚀 Iniciando bot multi-estrategia con persistencia de estado...")
    
    # Cargar estado previo
    load_state()
    
    # Conectar al exchange
    exchange = connect_common()
    
    # Cargar símbolos disponibles
    all_symbols = get_futures_symbols_from_api(PRODUCT_TYPE)
    
    # Cargar símbolos finales por estrategia
    final_by_strat = {}
    for strat in STRATEGIES:
        final_by_strat[strat['id']] = load_final_symbols(
            all_symbols,
            strategy=strat['name'],
            timeframe=strat['timeframe']
        )
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
            time.sleep(1)
            
    except KeyboardInterrupt:
        print("\n🚨 Interrumpido por usuario. Deteniendo threads...")
        STOP_THREADS = True
        
        # Guardar estado final antes de cerrar
        print("💾 Guardando estado final...")
        save_state()
        
        tpsl_thread.join(timeout=5)
        signals_thread.join(timeout=5)
        print("▶️ Bot detenido correctamente")


if __name__ == '__main__':
    main_loop()