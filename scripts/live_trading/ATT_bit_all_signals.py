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
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time

from datetime import datetime
from zoneinfo import ZoneInfo
from decimal import Decimal, ROUND_DOWN, InvalidOperation, getcontext


# --- Imports de tus módulos ---
from parquet_process.Z_parquet_A0_extraction import get_futures_symbols_from_api
from ZX_utils_live import wait_for_next_candle, load_final_symbols, normalize_live_ohlcv, df_to_arrays_live, fetch_ohlcv_data
from ZX_utils_live import load_state,save_state,make_client_oid,extract_filled_size_from_resp,detect_signal_for_strategy,get_contract_info

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
    'id': 'revers_short',
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
    'id': 'parity_short',
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



def maybe_open_orders_for_strategy(state, strat, final_symbols, exchange):
    strat_id = strat['id']
    tracked = state['strategies'].get(strat_id, [])
    if tracked:
        print(f"🛑 Estrategia {strat_id} tiene posiciones activas; no busca nuevas señales.")
        return state

    print(f"▶️ Buscando señales para estrategia {strat_id} ({strat['name']})...")
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
def close_size_on_exchange(symbol, size_requested, direction,
                           send_request_fn=send_request_common,
                           get_open_fn=get_open_positions_common,
                           product_type=PRODUCT_TYPE,
                           default_margin_mode='isolated',
                           default_margin_coin='USDT',
                           contract_info=None):
    """
    Cierra un tamaño de posición ajustando al múltiplo/decimales/min/max del contrato.
    Si se pasa contract_info, se usa en vez de consultar la API.
    """

    import time, uuid
    getcontext().prec = 28

    if contract_info is None:
        contract_info = get_contract_info(symbol, product_type, send_request_fn)

    min_trade = contract_info.get('min_trade', Decimal('0'))
    size_mult = contract_info.get('size_mult', Decimal('0.00000001'))
    volume_place = contract_info.get('volume_place', 8)
    max_market_order_qty = contract_info.get('max_market_order_qty', None)

    # Obtener posiciones abiertas
    try:
        exchange_positions = get_open_fn(product_type=product_type)
    except Exception as e:
        print(f"⚠️ Error obteniendo posiciones: {e}")
        exchange_positions = []

    holdside = 'long' if direction.lower() == 'long' else 'short'
    available = Decimal('0')
    chosen_margin_mode = default_margin_mode
    chosen_margin_coin = default_margin_coin

    for p in (exchange_positions or []):
        if p.get('symbol') != symbol:
            continue
        if (p.get('holdSide') or '').lower() == holdside:
            try:
                available += Decimal(str(p.get('available', 0) or 0))
            except Exception:
                pass
            if p.get('marginMode'):
                chosen_margin_mode = p.get('marginMode')
            if p.get('marginCoin'):
                chosen_margin_coin = p.get('marginCoin')

    if available <= Decimal('0'):
        print(f"❌ No hay posición disponible para cerrar (available={available})")
        return False, {'code': 22002, 'resp': {'code': '22002', 'msg': 'No position to close'}, 'min_trade': float(min_trade)}

    # Ajuste de tamaño
    try:
        requested = Decimal(str(size_requested))
    except InvalidOperation:
        requested = Decimal('0')

    size_to_close = min(requested, available)

    if size_mult <= Decimal('0'):
        size_mult = Decimal('0.00000001')

    try:
        multiples = (size_to_close / size_mult).to_integral_value(rounding=ROUND_DOWN)
        adjusted = multiples * size_mult
    except Exception:
        adjusted = size_to_close.quantize(Decimal('1e-{}'.format(volume_place)), rounding=ROUND_DOWN)

    quant = Decimal('1e-{}'.format(max(0, volume_place)))
    try:
        adjusted = adjusted.quantize(quant, rounding=ROUND_DOWN)
    except Exception:
        adjusted = Decimal(str(float(adjusted))).quantize(quant, rounding=ROUND_DOWN)

    if max_market_order_qty is not None and adjusted > max_market_order_qty:
        multiples = (max_market_order_qty / size_mult).to_integral_value(rounding=ROUND_DOWN)
        adjusted = multiples * size_mult
        adjusted = adjusted.quantize(quant, rounding=ROUND_DOWN)

    if adjusted < min_trade:
        adjusted = min_trade

    size_str = format(adjusted, 'f')

    body = {
        'symbol': symbol,
        'productType': product_type,
        'size': size_str,
        'side': 'sell' if direction.lower() == 'short' else 'buy',
        'tradeSide': 'close',
        'orderType': 'market',
        'clientOid': f"close-{int(time.time())}-{uuid.uuid4().hex[:6]}",
        'marginMode': chosen_margin_mode,
        'marginCoin': chosen_margin_coin
    }

    print(f"▶️ Enviando orden de cierre: size={body['size']}, side={body['side']}")
    code, resp = send_request_fn('POST', '/api/v2/mix/order/place-order', body=body)

    if code == 200 and resp.get('code') == '00000':
        closed_size = float(adjusted)
        print(f"🎯 Orden de cierre ejecutada: {closed_size} unidades")
        return True, {'code': code, 'resp': resp, 'closed_size': closed_size}
    else:
        error_msg = resp.get('msg', 'Unknown error') if isinstance(resp, dict) else str(resp)
        print(f"❌ Error al cerrar: {error_msg}")
        return False, {'code': code, 'resp': resp, 'min_trade': float(min_trade)}



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
            print(f"▶️ Ajuste proporcional {sym} ({dirn}): exchange={exch_size:.6f}, tracked={tracked_total:.6f}, ratio={ratio:.4f}")
            
            for strat_id, entry in info['entries']:
                old_size = float(entry.get('size', 0))
                new_size = old_size * ratio
                entry['size'] = new_size
                print(f"   Estrat {strat_id}: {old_size:.6f} -> {new_size:.6f}")

    return state

def manage_tracked_positions(state, strat, exchange):
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

        if t['candles_to_sell'] <= 0:
            sym = t['symbol']
            size = float(t['size'])
            direction = t['direction']

            print(f"▶️ Intentando cerrar {size:.6f} de {sym} (estrat {strat_id})")

            # Obtener contract_info una sola vez
            contract_info = get_contract_info(sym)

            ok, resp = close_size_on_exchange(
                sym, size, direction,
                send_request_fn=send_request_common,
                product_type=PRODUCT_TYPE,
                contract_info=contract_info
            )

            now = datetime.now(MADRID_TZ).strftime('%Y-%m-%d %H:%M:%S')

            if ok:
                closed = resp.get('closed_size', 0)
                remaining = size - closed
                min_trade = float(contract_info.get('min_trade', 0))

                if remaining >= min_trade:
                    t['size'] = remaining
                    print(f"▶️ {now} - Estrat {strat_id} cerró parcialmente {closed:.6f} de {sym}, queda {remaining:.6f}")
                else:
                    to_remove.append(idx)
                    print(f"▶️ {now} - Estrat {strat_id} cerró totalmente {sym}, remaining {remaining:.6f} < min_trade {min_trade:.6f}")
            else:
                error_code = resp.get('resp', {}).get('code', 'unknown')
                error_msg = resp.get('resp', {}).get('msg', 'unknown')
                print(f"⚠️ {now} - Estrat {strat_id} error al cerrar {sym}: [{error_code}] {error_msg}")
                if error_code == '22002':
                    to_remove.append(idx)

    for i in reversed(to_remove):
        del tracked[i]

    save_state(state)
    return state



# ----------------------
# MAIN
# ----------------------
def main_loop():
    exchange       = connect_common()
    all_symbols    = get_futures_symbols_from_api(PRODUCT_TYPE)
    final_by_strat = {}
    for strat in STRATEGIES:
        final_by_strat[strat['id']] = load_final_symbols(all_symbols, strategy=strat['name'], timeframe=strat['timeframe'])

    state = load_state(STRATEGIES)
    print("▶️ Iniciando loop principal (ambas estrategias desacopladas)...")

    while True:
        try:
            wait_for_next_candle(MIN_TIMEFRAME)
            state = reconcile_with_exchange(state, product_type=PRODUCT_TYPE, send_request_fn=send_request_common, get_open_fn=get_open_positions_common)

            # Abrir nuevas posiciones si hay señales
            for strat in STRATEGIES:
                state = maybe_open_orders_for_strategy(state, strat, final_by_strat.get(strat['id'], []), exchange)

            # Cerrar posiciones según candles_to_sell
            for strat in STRATEGIES:
                state = manage_tracked_positions(state, strat, exchange)

            # Mostrar posiciones restantes POR ESTRATEGIA después de procesar los cierres
            for strat in STRATEGIES:
                strat_id = strat['id']
                tracked = state['strategies'].get(strat_id, [])
                for t in tracked:
                    print(f"▶️ Estrat {strat_id} - {t['symbol']} -> candles_left: {t.get('candles_to_sell')}")

            # Resumen rápido
            summary = {s['id']: len(state['strategies'].get(s['id'], [])) for s in STRATEGIES}
            print(f"▶️ Posiciones activas por estrategia: {summary}")

            save_state(state)

        except KeyboardInterrupt:
            print("🚨 Interrumpido por usuario. Guardando estado y saliendo...")
            save_state(state)
            break
        except Exception as e:
            print(f"⚠️ Error en el loop principal: {e}")
            time.sleep(2)

if __name__ == '__main__':
    main_loop()

