#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script que integra dos estrategias (reversal_short y parity_short) operando en la misma cuenta
pero desacopladas: cada estrategia rastrea sus propias órdenes/posiciones y cierra únicamente
la porción (size) que ella abrió cuando toque sell_after_n_candles o cuando quiera cerrar.

Genera y persiste un estado en tracked_orders_state.json para mapear order/client ids a cada
estrategia y permitir reintentos/reconciliación con el exchange.
"""

import os
import sys
import time
import json
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --- Imports de tus módulos ---
from parquet_process.Z_parquet_A0_extraction import get_futures_symbols_from_api
from Z_add_signals_reversal import trend_reversal_entry_short
from Z_add_signals_parity import detect_parity_short
from ZX_utils_live import (
    wait_for_next_candle,
    load_final_symbols,
    normalize_live_ohlcv,
    df_to_arrays_live,
    fetch_ohlcv_data,
)
from utils.ZZ_connect import connect_bitget_TT
from ZX_connect_live import get_usdt_balance_TT, send_request_TT, get_open_positions_TT
from ZX_place_orders import place_order

MADRID_TZ     = ZoneInfo('Europe/Madrid')
STATE_FILE    = os.path.join(os.path.dirname(__file__), 'tracked_orders_state.json')
PRODUCT_TYPE  = 'USDT-FUTURES'
MIN_TIMEFRAME = '5m'

# ----------------------
# Parámetros por estrategia
# ----------------------
STRAT_A = {
    'id': 'A',
    'name': 'reversal_short',
    'timeframe': '5m',
    'order_amount': 10,
    'sell_after_n_candles': 2,
    'left_lookback': 8,
    'tolerance': 30,
    'tp_pct': 5,
    'sl_pct': 10,
    'direction': 'short'
}

STRAT_B = {
    'id': 'B',
    'name': 'parity_short',
    'timeframe': '5m',
    'order_amount': 10,
    'sell_after_n_candles': 4,
    'lookback': 150,
    'tolerance': 20,
    'tp_pct': 10,
    'sl_pct': 20,
    'direction': 'short'
}

STRATEGIES = [STRAT_A, STRAT_B]

# Conexión y funciones comunes (misma cuenta)
connect_common = connect_bitget_TT
send_request_common = send_request_TT
get_balance_common = get_usdt_balance_TT
get_open_positions_common = get_open_positions_TT

# ----------------------
# Estado persistente
# ----------------------
def load_state(path=STATE_FILE):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error cargando estado: {e}")
    return {'strategies': {s['id']: [] for s in STRATEGIES}}

def save_state(state, path=STATE_FILE):
    try:
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        print(f"⚠️ Error guardando estado: {e}")

# ----------------------
# Helpers
# ----------------------
def make_client_oid(strategy_id):
    return f"{strategy_id}-{int(time.time())}-{uuid.uuid4().hex[:6]}"

def extract_filled_size_from_resp(resp_order):
    if not resp_order or 'data' not in resp_order:
        return 0.0
    data = resp_order.get('data') or {}
    for k in ('size', 'filledSize', 'filledQty', 'filled_amount', 'filled_size'):
        v = data.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                try:
                    return float(str(v))
                except Exception:
                    pass
    try:
        return float(data.get('size', 0) or 0)
    except Exception:
        return 0.0

def determine_side_for_open(direction):
    return 'buy' if direction.lower() == 'long' else 'sell'

def determine_side_for_close(direction):
    return 'sell' if direction.lower() == 'long' else 'buy'

def close_size_on_exchange_(symbol, size_requested, direction,
                           send_request_fn=send_request_common,
                           get_open_fn=get_open_positions_common,
                           product_type=PRODUCT_TYPE,
                           default_margin_mode='isolated',
                           default_margin_coin='USDT'):
    """
    Intenta cerrar `size_requested` unidades de `symbol`.
    - Consulta las posiciones abiertas en el exchange para saber cuánto hay disponible.
    - Si no hay nada disponible devuelve (False, {'code':'no_position', ...}).
    - Si hay menos que lo pedido, cierra la cantidad disponible (partial close).
    - Añade marginMode y marginCoin a la petición (tomados del exchange si están presentes).
    """
    try:
        # 1) Obtener posiciones abiertas actuales
        exchange_positions = []
        try:
            exchange_positions = get_open_fn(product_type=product_type)
        except Exception as e:
            # No podemos consultar: vamos a intentar cerrar de todas formas (riesgo)
            exchange_positions = []

        # determinar holdSide buscado (posible 'long'|'short')
        holdside = 'long' if direction.lower() == 'long' else 'short'
        available = 0.0
        chosen_margin_mode = default_margin_mode
        chosen_margin_coin = default_margin_coin

        for p in (exchange_positions or []):
            if p.get('symbol') != symbol:
                continue
            # p.get('holdSide') puede ser 'long' o 'short'
            if p.get('holdSide') == holdside:
                try:
                    available += float(p.get('total', 0) or 0)
                except Exception:
                    pass
                # preferir datos de marginMode/marginCoin si vienen en la primera coincidencia
                if p.get('marginMode'):
                    chosen_margin_mode = p.get('marginMode')
                if p.get('marginCoin'):
                    chosen_margin_coin = p.get('marginCoin')

        # 2) Si no hay nada para cerrar -> devolver no_position
        if available <= 0:
            return False, {'code': 22002, 'resp': {'code': '22002', 'msg': 'No position to close', 'data': None}}

        # 3) Determinar tamaño a cerrar: min(requested, available)
        size_to_close = float(size_requested)
        if size_to_close > available:
            size_to_close = available

        # 4) Preparar body con marginMode/marginCoin
        body = {
            'symbol': symbol,
            'productType': product_type,
            'size': format(size_to_close, 'f'),
            'side': ('sell' if direction.lower() == 'long' else 'buy'),  # cerrar opuesto
            'tradeSide': 'close',
            'orderType': 'market',
            'clientOid': f"close-{int(time.time())}-{uuid.uuid4().hex[:6]}",
            'marginMode': chosen_margin_mode,
            'marginCoin': chosen_margin_coin
        }

        # 5) Enviar orden de cierre
        code, resp = send_request_fn('POST', '/api/v2/mix/order/place-order', body=body)

        # 6) Interpretar respuesta
        if code == 200 and resp.get('code') == '00000':
            return True, {'code': code, 'resp': resp, 'closed_size': size_to_close}
        else:
            # Si el exchange responde que no hay posición (22002) o margin error, lo propagamos
            return False, {'code': code, 'resp': resp}

    except Exception as e:
        return False, {'error': 'exception', 'exc': str(e)}


def reconcile_with_exchange_(state, product_type=PRODUCT_TYPE, send_request_fn=send_request_common, get_open_fn=get_open_positions_common):
    try:
        exchange_positions = get_open_fn(product_type=product_type)
    except Exception as e:
        print(f"⚠️ Error obteniendo posiciones del exchange: {e}")
        return state

    exch_map = {}
    for p in exchange_positions or []:
        sym = p.get('symbol')
        hold = p.get('holdSide')
        try:
            total = float(p.get('total', 0) or 0)
        except Exception:
            total = 0.0
        if sym not in exch_map:
            exch_map[sym] = {'long': 0.0, 'short': 0.0}
        exch_map[sym][hold] = exch_map[sym].get(hold, 0.0) + total

    for strat_id, tracked in state['strategies'].items():
        to_remove = []
        for idx, t in enumerate(tracked):
            sym = t.get('symbol')
            dirn = t.get('direction')
            holdside = 'long' if dirn == 'long' else 'short'
            exch_size = exch_map.get(sym, {}).get(holdside, 0.0)
            if exch_size <= 0:
                print(f"🔁 {datetime.now(MADRID_TZ).strftime('%H:%M')} - Posición {sym} estrat {strat_id} ya no existe en exchange. Marcando cerrada localmente.")
                to_remove.append(idx)
            else:
                tracked_size = float(t.get('size', 0) or 0)
                if exch_size < tracked_size - 1e-12:
                    print(f"⚠️ Ajustando tamaño trackeado para {sym} (estrat {strat_id}): {tracked_size} -> {exch_size}")
                    t['size'] = exch_size
        for i in reversed(to_remove):
            try:
                del tracked[i]
            except Exception:
                pass

    return state

# ----------------------
# Lógica principal por estrategia
# ----------------------
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

        # tomar el último elemento
        last = signals_arr.flat[-1]

        # convertir last a array/numpy para comprobar si hay valores no nulos
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
    #FAKE
    now = datetime.now(MADRID_TZ).isoformat()
    return [
        {
            'symbol': 'BTCUSDT',
            'timestamp': now,
            'close': 50000.0
        },
        {
            'symbol': 'BNBUSDT',
            'timestamp': now,
            'close': 600.0
        }
    ]
       
    #return detected


def maybe_open_orders_for_strategy(state, strat, final_symbols, exchange):
    strat_id = strat['id']
    tracked = state['strategies'].get(strat_id, [])
    if tracked:
        print(f"⛔ Estrategia {strat_id} tiene posiciones activas; no busca nuevas señales.")
        return state

    print(f"🔎 Buscando señales para estrategia {strat_id} ({strat['name']})...")
    signals = detect_signal_for_strategy(strat, final_symbols)
    print(f"✨ {datetime.now(MADRID_TZ).strftime('%H:%M')} - Signals detected for {strat_id}: {len(signals)}")

    for sig in signals:
        usdt_balance = get_balance_common(exchange)
        if usdt_balance < strat['order_amount']:
            print(f"⚠️ Saldo USDT insuficiente ({usdt_balance:.2f}) para abrir {sig['symbol']} con estrategia {strat_id}")
            continue

        client_oid = make_client_oid(strat_id)
        resp_order, tpsl_info = place_order(
            sig['symbol'],
            direction=strat['direction'],
            usdt_amount=strat['order_amount'],
            tp_percent=strat['tp_pct'],
            sl_percent=strat['sl_pct'],
            send_request_func=send_request_common,
            client_oid=client_oid
        )

        if resp_order is None:
            print(f"⚠️ Orden no ejecutada para {sig['symbol']} (estrat {strat_id}).")
            continue

        filled = extract_filled_size_from_resp(resp_order)
        if (not filled or filled <= 0) and tpsl_info and 'size_tpsl' in tpsl_info:
            try:
                filled = float(tpsl_info['size_tpsl'])
            except Exception:
                pass

        if filled <= 0:
            print(f"⚠️ No se detectó tamaño ejecutado para la orden en {sig['symbol']}. Ignorando.")
            continue

        order_id = resp_order.get('data', {}).get('orderId')

        tracked_entry = {
            'symbol': sig['symbol'],
            'order_id': order_id,
            'client_oid': client_oid,
            'size': filled,
            'buy_price': sig['close'],
            'candles_to_sell': strat['sell_after_n_candles'],
            'direction': strat['direction'],
            'opened_at': datetime.now(MADRID_TZ).isoformat(),
            'just_bought': True
        }

        state['strategies'].setdefault(strat_id, []).append(tracked_entry)
        save_state(state)
        

        # sleep pequeño para respetar rate-limit
        time.sleep(0.3)

    return state

def manage_tracked_positions_(state, strat, exchange):
    strat_id = strat['id']
    tracked = state['strategies'].get(strat_id, [])
    if not tracked:
        return state

    to_remove = []
    for idx, t in enumerate(tracked):
        if t.get('just_bought', False):
            t['just_bought'] = False
            continue

        t['candles_to_sell'] = int(t.get('candles_to_sell', strat['sell_after_n_candles'])) - 1
        print(f"⏳ Estrat {strat_id} - {t['symbol']}      -> candles_left: {t['candles_to_sell']}")

        if t['candles_to_sell'] <= 0:
            try:
                sym = t['symbol']
                size = float(t['size'])
                dirn = t['direction']
                ok, resp = close_size_on_exchange(sym, size, dirn, send_request_fn=send_request_common, product_type=PRODUCT_TYPE)
                now = datetime.now(MADRID_TZ).replace(second=0, microsecond=0)
                if ok:
                    print(f"💥 {now.strftime('%Y-%m-%d %H:%M:%S')} - Estrat {strat_id} FLASH CLOSE OK {sym} | size={size}")
                else:
                    print(f"⚠️ {now} - Estrat {strat_id} fallo al cerrar {sym}: {resp}")
            except Exception as e:
                print(f"⚠️ Error cerrando posición trackeada: {e}")
            finally:
                to_remove.append(idx)

    for i in reversed(to_remove):
        try:
            del tracked[i]
        except Exception:
            pass

    save_state(state)
    return state

def close_size_on_exchange(symbol, size_requested, direction,
                           send_request_fn=send_request_common,
                           get_open_fn=get_open_positions_common,
                           product_type=PRODUCT_TYPE,
                           default_margin_mode='isolated',
                           default_margin_coin='USDT'):
    """
    Cierra una porción específica de una posición (puede ser compartida entre estrategias)
    """
    try:
        # 1) Obtener posiciones abiertas
        exchange_positions = []
        try:
            exchange_positions = get_open_fn(product_type=product_type)
        except Exception as e:
            print(f"⚠️ Error obteniendo posiciones: {e}")
            exchange_positions = []

        # DEBUG: Mostrar posiciones para este símbolo
        print(f"🔍 Posiciones en exchange para {symbol}:")
        for p in exchange_positions:
            if p.get('symbol') == symbol:
                print(f"  holdSide={p.get('holdSide')}, available={p.get('available')}, total={p.get('total')}")

        # Determinar holdSide
        holdside = 'long' if direction.lower() == 'long' else 'short'
        
        available = 0.0
        chosen_margin_mode = default_margin_mode
        chosen_margin_coin = default_margin_coin

        for p in (exchange_positions or []):
            if p.get('symbol') != symbol:
                continue
            
            exchange_holdside = p.get('holdSide', '').lower()
            
            if exchange_holdside == holdside.lower():
                try:
                    pos_available = float(p.get('available', 0) or 0)
                    available += pos_available
                except Exception as e:
                    print(f"⚠️ Error parseando available: {e}")
                
                if p.get('marginMode'):
                    chosen_margin_mode = p.get('marginMode')
                if p.get('marginCoin'):
                    chosen_margin_coin = p.get('marginCoin')

        # 2) Verificar si hay posición
        if available <= 1e-8:  # usar umbral más pequeño
            print(f"❌ No hay posición disponible para cerrar (available={available})")
            return False, {
                'code': 22002, 
                'resp': {'code': '22002', 'msg': 'No position to close', 'data': None}
            }

        # 3) Ajustar tamaño si es necesario
        size_to_close = min(float(size_requested), available)
        
        # IMPORTANTE: Redondear a la precisión adecuada del símbolo
        # Para la mayoría de símbolos, usar 4 decimales es seguro
        # Ajusta según las especificaciones del símbolo si es necesario
        if size_to_close < available * 0.99:  # Si es cierre parcial
            print(f"📉 Cierre PARCIAL: {size_to_close}/{available}")
        else:
            print(f"💥 Cierre TOTAL: {size_to_close}")

        # 4) Preparar orden para HEDGE MODE
        # En hedge mode: para cerrar se usa el MISMO side + tradeSide='close'
        # - Cerrar long: side='buy', tradeSide='close'
        # - Cerrar short: side='sell', tradeSide='close'
        body = {
            'symbol': symbol,
            'productType': product_type,
            'size': str(round(size_to_close, 6)),
            'side': 'sell' if direction.lower() == 'short' else 'buy',  # MISMO side que al abrir
            'tradeSide': 'close',  # Requerido en hedge mode
            'orderType': 'market',
            'clientOid': f"close-{int(time.time())}-{uuid.uuid4().hex[:6]}",
            'marginMode': chosen_margin_mode,
            'marginCoin': chosen_margin_coin
        }

        print(f"📤 Enviando orden de cierre: size={body['size']}, side={body['side']}")

        # 5) Enviar orden
        code, resp = send_request_fn('POST', '/api/v2/mix/order/place-order', body=body)

        # 6) Verificar respuesta
        if code == 200 and resp.get('code') == '00000':
            print(f"✅ Orden de cierre ejecutada: {size_to_close} unidades")
            return True, {'code': code, 'resp': resp, 'closed_size': size_to_close}
        else:
            error_msg = resp.get('msg', 'Unknown error')
            print(f"❌ Error al cerrar: {error_msg}")
            return False, {'code': code, 'resp': resp}

    except Exception as e:
        print(f"❌ Excepción: {e}")
        import traceback
        traceback.print_exc()
        return False, {'error': 'exception', 'exc': str(e)}


def reconcile_with_exchange(state, product_type=PRODUCT_TYPE, 
                           send_request_fn=send_request_common, 
                           get_open_fn=get_open_positions_common):
    """
    Reconcilia el estado local con las posiciones reales del exchange.
    Actualiza los tamaños trackeados proporcionalmente si hay diferencias.
    """
    try:
        exchange_positions = get_open_fn(product_type=product_type)
    except Exception as e:
        print(f"⚠️ Error obteniendo posiciones: {e}")
        return state

    # Mapear posiciones del exchange
    exch_map = {}
    for p in exchange_positions or []:
        sym = p.get('symbol')
        hold = p.get('holdSide', '').lower()
        try:
            available_size = float(p.get('available', 0) or 0)
        except Exception:
            available_size = 0.0
        
        if sym not in exch_map:
            exch_map[sym] = {'long': 0.0, 'short': 0.0}
        exch_map[sym][hold] = available_size

    # Para cada símbolo, calcular el total trackeado por todas las estrategias
    tracked_totals = {}
    for strat_id, tracked in state['strategies'].items():
        for t in tracked:
            sym = t.get('symbol')
            dirn = t.get('direction', '').lower()
            size = float(t.get('size', 0) or 0)
            
            key = (sym, dirn)
            if key not in tracked_totals:
                tracked_totals[key] = {'total': 0.0, 'entries': []}
            tracked_totals[key]['total'] += size
            tracked_totals[key]['entries'].append((strat_id, t))

    # Ajustar proporcionalmente si hay diferencias
    for (sym, dirn), info in tracked_totals.items():
        holdside = 'long' if dirn == 'long' else 'short'
        exch_size = exch_map.get(sym, {}).get(holdside, 0.0)
        tracked_total = info['total']
        
        if exch_size <= 1e-8:
            # No hay posición en el exchange, marcar todas como cerradas
            print(f"🔁 {sym} ({dirn}): No existe en exchange, removiendo {len(info['entries'])} entradas")
            for strat_id, entry in info['entries']:
                try:
                    state['strategies'][strat_id].remove(entry)
                except:
                    pass
        elif abs(exch_size - tracked_total) > 1e-6:
            # Hay diferencia, ajustar proporcionalmente
            ratio = exch_size / tracked_total if tracked_total > 0 else 0
            print(f"⚠️ Ajuste proporcional {sym} ({dirn}): exchange={exch_size:.6f}, tracked={tracked_total:.6f}, ratio={ratio:.4f}")
            
            for strat_id, entry in info['entries']:
                old_size = float(entry.get('size', 0))
                new_size = old_size * ratio
                entry['size'] = new_size
                print(f"   Estrat {strat_id}: {old_size:.6f} -> {new_size:.6f}")

    return state


def manage_tracked_positions(state, strat, exchange):
    """
    Versión mejorada que maneja cierres parciales correctamente
    """
    strat_id = strat['id']
    tracked = state['strategies'].get(strat_id, [])
    if not tracked:
        return state

    to_remove = []
    for idx, t in enumerate(tracked):
        if t.get('just_bought', False):
            t['just_bought'] = False
            continue

        t['candles_to_sell'] = int(t.get('candles_to_sell', strat['sell_after_n_candles'])) - 1
        print(f"⏳ Estrat {strat_id} - {t['symbol']}      -> candles_left: {t['candles_to_sell']}")

        if t['candles_to_sell'] <= 0:
            sym = t['symbol']
            size = float(t['size'])
            dirn = t['direction']
            
            print(f"🎯 Intentando cerrar {size:.6f} de {sym} (estrat {strat_id})")
            
            ok, resp = close_size_on_exchange(
                sym, size, dirn, 
                send_request_fn=send_request_common, 
                product_type=PRODUCT_TYPE
            )
            
            now = datetime.now(MADRID_TZ).strftime('%Y-%m-%d %H:%M:%S')
            
            if ok:
                closed = resp.get('closed_size', size)
                print(f"💥 {now} - Estrat {strat_id} cerró {closed:.6f} de {sym}")
                to_remove.append(idx)
            else:
                error_code = resp.get('resp', {}).get('code', 'unknown')
                error_msg = resp.get('resp', {}).get('msg', 'unknown')
                print(f"⚠️ {now} - Estrat {strat_id} error al cerrar {sym}: [{error_code}] {error_msg}")
                
                # Si el error es "no position", remover de tracking
                if error_code == '22002':
                    print(f"   Removiendo de tracking (posición ya no existe)")
                    to_remove.append(idx)

    for i in reversed(to_remove):
        try:
            del tracked[i]
        except Exception:
            pass

    save_state(state)
    return state

# ----------------------
# MAIN
# ----------------------
def main_loop():
    exchange = connect_common()
    all_symbols = get_futures_symbols_from_api(PRODUCT_TYPE)
    final_by_strat = {}
    for strat in STRATEGIES:
        final_by_strat[strat['id']] = load_final_symbols(all_symbols, strategy=strat['name'], timeframe=strat['timeframe'])

    state = load_state()
    print("▶️ Iniciando loop principal (ambas estrategias desacopladas)...")

    while True:
        try:
            wait_for_next_candle(MIN_TIMEFRAME)
            state = reconcile_with_exchange(state, product_type=PRODUCT_TYPE, send_request_fn=send_request_common, get_open_fn=get_open_positions_common)

            for strat in STRATEGIES:
                state = maybe_open_orders_for_strategy(state, strat, final_by_strat.get(strat['id'], []), exchange)

            for strat in STRATEGIES:
                state = manage_tracked_positions(state, strat, exchange)

            # Resumen rápido
            summary = {s['id']: len(state['strategies'].get(s['id'], [])) for s in STRATEGIES}
            print(f"📊 Posiciones activas por estrategia: {summary}")

            save_state(state)

        except KeyboardInterrupt:
            print("⏹️ Interrumpido por usuario. Guardando estado y saliendo...")
            save_state(state)
            break
        except Exception as e:
            print(f"⚠️ Error en el loop principal: {e}")
            time.sleep(2)

if __name__ == '__main__':
    main_loop()
