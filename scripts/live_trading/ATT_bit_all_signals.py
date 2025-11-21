#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script que integra dos estrategias (reversal_short y parity_short) operando en la misma cuenta
de forma TOTALMENTE DESACOPLADA usando plan orders independientes para cada estrategia.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time

from datetime import datetime
from zoneinfo import ZoneInfo
from decimal import Decimal, ROUND_DOWN, ROUND_UP, InvalidOperation, getcontext

# --- Imports de tus módulos ---
from parquet_process.Z_parquet_A0_extraction import get_futures_symbols_from_api
from ZX_utils_live import wait_for_next_candle, load_final_symbols
from ZX_utils_live import load_state, save_state, make_client_oid, detect_signal_for_strategy
from utils.ZZ_connect import connect_bitget_TT
from ZX_connect_live import get_usdt_balance_TT, send_request_TT
from ZX_place_orders import place_order

MADRID_TZ = ZoneInfo('Europe/Madrid')
STATE_FILE = os.path.join(os.path.dirname(__file__), 'tracked_orders_state.json')
PRODUCT_TYPE = 'USDT-FUTURES'
MIN_TIMEFRAME = '5m'

# ----------------------
# TESTING: Señales Hardcodeadas
# ----------------------
USE_HARDCODED_SIGNALS = True

def get_hardcoded_signals(strat_id):
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
        signals.append({'symbol': symbol, 'close': current_price,
                        'timestamp': datetime.now(MADRID_TZ).isoformat()})
    return signals

# ----------------------
# Parámetros por estrategia
# ----------------------
STRAT_A = {
    'id': 'revers_short',
    'name': 'reversal_short',
    'timeframe': '5m',
    'order_amount': 10,
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
    'lookback': 150,
    'tolerance': 20,
    'tp_pct': 1,
    'sl_pct': 2,
    'direction': 'short'
}

STRATEGIES = [STRAT_A, STRAT_B]

# Conexión y funciones comunes
connect_common = connect_bitget_TT
send_request_common = send_request_TT
get_balance_common = get_usdt_balance_TT

# ----------------------
# Helpers para plan orders
# ----------------------
def get_pending_plan_orders(send_request_func, symbol=None, product_type='USDT-FUTURES', limit=200, plan_type=None):
    """
    Llama al endpoint orders-plan-pending y devuelve la lista 'entrustedList'.
    plan_type: None (todas), 'normal_plan', 'track_plan', 'profit_loss'
    """
    endpoint = "/api/v2/mix/order/orders-plan-pending"
    params = {
        "productType": product_type,
        "limit": limit
    }
    if symbol:
        params["symbol"] = symbol
    if plan_type:
        params["planType"] = plan_type

    status, resp = send_request_func("GET", endpoint, params=params)

    if status != 200 or not isinstance(resp, dict):
        return []

    if resp.get("code") != "00000":
        return []

    data = resp.get("data", {}) or {}
    orders = data.get("entrustedList", []) or []
    return orders

def find_recent_plan_orders_for_position(send_request_func, symbol, hold_side, tp_str, sl_str):
    """
    Busca en las órdenes pendientes las órdenes cuyo trigger price coincida con tp_str o sl_str.
    """
    orders = get_pending_plan_orders(send_request_func, symbol=symbol, product_type=PRODUCT_TYPE, plan_type='profit_loss')
    matched_ids = []
    for o in orders:
        try:
            if o.get('posSide') and o.get('posSide').lower() != hold_side.lower():
                continue
        except:
            pass

        tp = o.get('stopSurplusTriggerPrice') or o.get('presetStopSurplusPrice')
        sl = o.get('stopLossTriggerPrice') or o.get('presetStopLossPrice')

        if tp and tp == str(tp_str):
            matched_ids.append(o.get('orderId'))
        if sl and sl == str(sl_str):
            matched_ids.append(o.get('orderId'))
    
    return list({mid for mid in matched_ids if mid})

# ----------------------
# Helpers para precios
# ----------------------
def get_contract_info(send_request_func, product_type, symbol):
    """Obtiene información del contrato: price_tick, price_scale"""
    try:
        code, resp = send_request_func(
            "GET",
            "/api/v2/mix/market/contracts",
            params={"productType": product_type, "symbol": symbol}
        )
        if code == 200 and isinstance(resp, dict) and resp.get("code") == "00000":
            data_list = resp.get("data", []) or []
            if data_list:
                c = data_list[0]
                
                price_tick = None
                price_scale = None
                if c.get("pricePlace") is not None:
                    try:
                        price_scale = int(c.get("pricePlace"))
                        price_tick = Decimal(f"1e-{price_scale}")
                    except Exception:
                        price_tick = None
                if price_tick is None and c.get("priceEndStep") is not None:
                    try:
                        price_tick = Decimal(str(c.get("priceEndStep")))
                        if price_tick == price_tick.to_integral():
                            price_scale = 0
                        else:
                            price_scale = max(0, -price_tick.as_tuple().exponent)
                    except Exception:
                        price_tick = None
                if price_tick is None:
                    price_tick = Decimal("0.01")
                    price_scale = 2
                
                return price_tick, price_scale
    except Exception as e:
        print(f"⚠️ Error consultando contracts: {e}")
    return Decimal("0.01"), 2

def quantize_price_for_tick(price: Decimal, price_tick: Decimal, direction_rounding):
    if direction_rounding == "down":
        rnd = ROUND_DOWN
    else:
        rnd = ROUND_UP
    scale = max(0, -price_tick.as_tuple().exponent)
    quant = Decimal(f"1e-{scale}")
    try:
        q = price.quantize(quant, rounding=rnd)
    except InvalidOperation:
        q = price.quantize(quant, rounding=ROUND_DOWN)
    return q

def place_independent_tpsl_orders(symbol, size, exec_price, direction, tp_pct, sl_pct, 
                                   send_request_func, product_type='USDT-FUTURES', 
                                   margin_coin='USDT', client_oid_prefix=''):
    """
    Coloca TP y SL como órdenes PLAN INDEPENDIENTES usando stopSurplusSize/stopLossSize.
    Retorna: (list_of_order_ids, response_data)
    """
    exec_price_d = Decimal(str(exec_price))

    # Calcular precios de TP y SL
    if direction.lower() == 'short':
        tp_price_d = exec_price_d * (Decimal('1') - Decimal(str(tp_pct)) / Decimal('100'))
        sl_price_d = exec_price_d * (Decimal('1') + Decimal(str(sl_pct)) / Decimal('100'))
        hold_side = 'short'
    else:
        tp_price_d = exec_price_d * (Decimal('1') + Decimal(str(tp_pct)) / Decimal('100'))
        sl_price_d = exec_price_d * (Decimal('1') - Decimal(str(sl_pct)) / Decimal('100'))
        hold_side = 'long'

    price_tick, price_scale = get_contract_info(send_request_func, product_type, symbol)
    print(f"   price_tick={price_tick} scale={price_scale}")

    # Cuantizar precios
    if direction.lower() == 'short':
        tp_q = quantize_price_for_tick(tp_price_d, price_tick, "down")
        sl_q = quantize_price_for_tick(sl_price_d, price_tick, "up")
    else:
        tp_q = quantize_price_for_tick(tp_price_d, price_tick, "up")
        sl_q = quantize_price_for_tick(sl_price_d, price_tick, "down")

    fmt = f"{{0:.{price_scale}f}}"
    tp_str = fmt.format(tp_q)
    sl_str = fmt.format(sl_q)

    # Formatear size
    size_str = str(size)

    print(f"▶️  Colocando TP/SL INDEPENDIENTES para {symbol}:")
    print(f"   Size: {size_str}, Exec: {exec_price_d}")
    print(f"   TP: {tp_str} ({tp_pct}%), SL: {sl_str} ({sl_pct}%)")

    # Cliente OIDs únicos
    tp_client_oid = f"{client_oid_prefix}_TP_{int(time.time() * 1000)}"
    sl_client_oid = f"{client_oid_prefix}_SL_{int(time.time() * 1000)}"

    # Usar place-pos-tpsl CON stopSurplusSize y stopLossSize
    body = {
        'symbol': symbol,
        'productType': product_type,
        'marginCoin': margin_coin,
        'stopSurplusTriggerPrice': tp_str,
        'stopSurplusTriggerType': 'mark_price',
        'stopSurplusExecutePrice': None,
        'stopSurplusSize': size_str,
        'stopLossTriggerPrice': sl_str,
        'stopLossTriggerType': 'mark_price',
        'stopLossExecutePrice': None,
        'stopLossSize': size_str,
        'holdSide': hold_side,
        'stopSurplusClientOid': tp_client_oid,
        'stopLossClientOid': sl_client_oid
    }

    print(f"   📤 Colocando TP/SL con size específico...")
    status, resp = send_request_func('POST', '/api/v2/mix/order/place-pos-tpsl', body=body)

    if status == 200 and isinstance(resp, dict) and resp.get('code') == '00000':
        data_list = resp.get('data', [])
        order_ids = []
        
        if data_list:
            # La API devuelve una lista con los plan orders creados
            # Cada item puede tener orderId (si es un solo plan order) o múltiples IDs
            for item in data_list:
                order_id = item.get('orderId')
                if order_id:
                    order_ids.append(order_id)
            
            print(f"   ✅ TP/SL colocados. Plan order IDs: {order_ids}")
            
            # Si la API no devolvió los IDs individuales, intentamos buscarlos
            if not order_ids:
                time.sleep(0.5)
                plan_ids = find_recent_plan_orders_for_position(send_request_func, symbol, hold_side, tp_str, sl_str)
                print(f"   🔍 Buscados manualmente: {plan_ids}")
                return plan_ids, resp
            
            return order_ids, resp
        else:
            print(f"   ⚠️ Respuesta exitosa pero sin data")
            return [], resp
    else:
        error_msg = resp.get('msg', 'Unknown error') if isinstance(resp, dict) else str(resp)
        print(f"   ❌ Error al colocar TP/SL: {error_msg}")
        return [], resp

# ----------------------
# Lógica principal
# ----------------------
def get_filled_size_from_fills(order_id, symbol, send_request_func, product_type='USDT-FUTURES', max_retries=3):
    for attempt in range(max_retries):
        params = {'orderId': order_id, 'symbol': symbol, 'productType': product_type}
        code, resp = send_request_func('GET', '/api/v2/mix/order/fills', params=params)
        if code == 200 and isinstance(resp, dict) and resp.get('code') == '00000':
            fill_list = resp.get('data', {}).get('fillList', [])
            if fill_list:
                total_filled = 0.0
                for fill in fill_list:
                    try:
                        total_filled += float(fill.get('baseVolume', 0))
                    except:
                        continue
                if total_filled > 0:
                    print(f"▶️  Tamaño ejecutado obtenido de fills: {total_filled} ({len(fill_list)} fills)")
                    return total_filled
            else:
                print(f"⏳ Intento {attempt + 1}/{max_retries}: Sin fills aún para orden {order_id}")
        else:
            err = resp.get('msg', 'Unknown') if isinstance(resp, dict) else str(resp)
            print(f"⚠️ Error consultando fills: {err}")
        if attempt < max_retries - 1:
            time.sleep(0.5)
    print(f"❌ No se pudieron obtener fills después de {max_retries} intentos")
    return 0.0

def maybe_open_orders_for_strategy(state, strat, final_symbols, exchange, use_hardcoded=False):
    """
    Abre nuevas posiciones para una estrategia específica.
    Solo busca señales si NO hay ningún TP/SL activo de esta estrategia.
    """
    strat_id = strat['id']
    tracked = state['strategies'].get(strat_id, [])

    # NUEVO: Verificar si hay TP/SL activos para esta estrategia
    if tracked:
        print(f"⏸️  {strat_id} tiene {len(tracked)} posiciones con TP/SL activos")
        print(f"   → Esperando cierre antes de buscar nuevas señales...")
        return state

    print(f"✅ {strat_id} sin posiciones activas. Buscando señales...")
    print(f"▶️ Buscando señales para estrategia {strat_id} ({strat['name']})...")
    if use_hardcoded:
        signals = get_hardcoded_signals(strat_id)
    else:
        signals = detect_signal_for_strategy(strat, final_symbols)

    print(f"✨ {datetime.now(MADRID_TZ).strftime('%H:%M')} - Señales detectadas para {strat_id}: {len(signals)}")

    for sig in signals:
        usdt_balance = get_balance_common(exchange)
        if usdt_balance < strat['order_amount']:
            print(f"⚠️ Saldo insuficiente ({usdt_balance:.2f} USDT) para {sig['symbol']}")
            continue

        client_oid = make_client_oid(strat_id)
        print(f"\n▶️  Abriendo posición {strat['direction']} en {sig['symbol']} para {strat_id}...")
        resp_order = place_order(
            sig['symbol'],
            direction=strat['direction'],
            usdt_amount=strat['order_amount'],
            send_request_func=send_request_common,
            client_oid=client_oid
        )

        if resp_order is None:
            print(f"⚠️ Orden no ejecutada para {sig['symbol']}")
            continue

        order_id = resp_order.get('data', {}).get('orderId')
        if not order_id:
            print(f"⚠️ No se obtuvo orderId de la respuesta")
            continue

        print(f"▶️  Order ID: {order_id}")
        time.sleep(1)

        filled_size = get_filled_size_from_fills(order_id, sig['symbol'], send_request_common, PRODUCT_TYPE)

        exec_price = sig['close']
        params = {'orderId': order_id, 'symbol': sig['symbol'], 'productType': PRODUCT_TYPE}
        code, resp = send_request_common('GET', '/api/v2/mix/order/fills', params=params)
        if code == 200 and isinstance(resp, dict) and resp.get('code') == '00000':
            fill_list = resp.get('data', {}).get('fillList', [])
            if fill_list:
                total_value = 0.0
                total_qty = 0.0
                for fill in fill_list:
                    try:
                        price = float(fill.get('price', 0))
                        qty = float(fill.get('baseVolume', 0))
                        total_value += price * qty
                        total_qty += qty
                    except:
                        continue
                if total_qty > 0:
                    exec_price = total_value / total_qty

        print(f"▶️  Orden ejecutada: {filled_size} @ {exec_price}")

        if filled_size <= 0:
            print(f"⚠️ No se detectó tamaño ejecutado para {sig['symbol']}")
            continue

        # Colocar TP/SL independientes
        tpsl_order_ids, tpsl_data = place_independent_tpsl_orders(
            sig['symbol'],
            filled_size,
            exec_price,
            strat['direction'],
            strat['tp_pct'],
            strat['sl_pct'],
            send_request_common,
            client_oid_prefix=f"{strat_id}_{sig['symbol']}"
        )

        if not tpsl_order_ids or len(tpsl_order_ids) < 2:
            print(f"🔶 No se pudieron crear ambos TP/SL para {sig['symbol']}")

        # Separar TP y SL order IDs (asumiendo que el primero es TP y el segundo SL)
        tp_order_id = tpsl_order_ids[0] if len(tpsl_order_ids) > 0 else None
        sl_order_id = tpsl_order_ids[1] if len(tpsl_order_ids) > 1 else None

        tracked_entry = {
            'symbol': sig['symbol'],
            'order_id': order_id,
            'client_oid': client_oid,
            'tp_order_id': tp_order_id,  # ID individual del TP
            'sl_order_id': sl_order_id,  # ID individual del SL
            'size': filled_size,
            'exec_price': float(exec_price),
            'direction': strat['direction'],
            'opened_at': datetime.now(MADRID_TZ).isoformat(),
            'tp_pct': strat['tp_pct'],
            'sl_pct': strat['sl_pct']
        }

        state['strategies'].setdefault(strat_id, []).append(tracked_entry)
        save_state(state)
        print(f"▶️ Posición guardada en estado para {strat_id}")

        time.sleep(0.5)

    return state

def cancel_plan_order(order_id, symbol, send_request_func, product_type='USDT-FUTURES', margin_coin='USDT'):
    """
    Cancela una plan order específica.
    """
    body = {
        'orderIdList': [
            {
                'orderId': order_id,
                'clientOid': ''
            }
        ],
        'symbol': symbol,
        'productType': product_type,
        'marginCoin': margin_coin
    }
    
    print(f"      📤 Cancelando order_id={order_id} en {symbol}...")
    status, resp = send_request_func('POST', '/api/v2/mix/order/cancel-plan-order', body=body)
    
    if status == 200 and isinstance(resp, dict) and resp.get('code') == '00000':
        data = resp.get('data', {})
        success_list = data.get('successList', [])
        failure_list = data.get('failureList', [])
        
        if success_list:
            print(f"      ✅ Plan order {order_id} cancelada exitosamente")
            return True
        elif failure_list:
            error_msg = failure_list[0].get('errorMsg', 'Unknown') if failure_list else 'Unknown'
            print(f"      ⚠️ Error al cancelar {order_id}: {error_msg}")
            return False
        else:
            # Listas vacías = orden ya no existe (probablemente ya ejecutada)
            # Consideramos esto como éxito ya que el objetivo (eliminar la orden) se cumplió
            print(f"      ✅ Orden {order_id} ha venido lilsta vacia")
            return True
    else:
        error_msg = resp.get('msg', 'Unknown error') if isinstance(resp, dict) else str(resp)
        print(f"      ❌ Error de API al cancelar {order_id}: {error_msg}")
        return False

def check_and_clean_executed_positions(state, strat):
    """
    Verifica las posiciones de una estrategia y elimina las que ya se cerraron.
    """
    strat_id = strat['id']
    tracked = state['strategies'].get(strat_id, [])

    if not tracked:
        return state

    print(f"\n🔍 Verificando {len(tracked)} posiciones de {strat_id}...")

    to_remove = []

    symbols = {pos['symbol'] for pos in tracked}
    pending_by_symbol = {}
    for s in symbols:
        pending_by_symbol[s] = get_pending_plan_orders(send_request_common, symbol=s, product_type=PRODUCT_TYPE, plan_type='profit_loss')

    for idx, pos in enumerate(tracked):
        symbol = pos['symbol']
        tp_order_id = pos.get('tp_order_id')
        sl_order_id = pos.get('sl_order_id')

        # Obtener IDs de órdenes pendientes para este símbolo
        pending_list_for_symbol = pending_by_symbol.get(symbol, []) or []
        pending_ids = {p.get('orderId') for p in pending_list_for_symbol if p.get('orderId')}

        # Verificar estado individual de TP y SL
        tp_active = tp_order_id in pending_ids if tp_order_id else False
        sl_active = sl_order_id in pending_ids if sl_order_id else False

        print(f"   {symbol} ({strat_id}): TP={'✅' if tp_active else '❌'} | SL={'✅' if sl_active else '❌'}")

        # Caso 1: Ambos activos → todo normal
        if tp_active and sl_active:
            print(f"      ✓ Ambas órdenes activas")
            continue

        # Caso 2: Uno ejecutado, el otro activo → cancelar el restante
        if tp_active and not sl_active:
            print(f"      🎯 SL ejecutado! Cancelando TP restante (ID: {tp_order_id})...")
            # Verificar que realmente esté en pending antes de cancelar
            time.sleep(0.3)  # Pequeño delay para sincronización
            success = cancel_plan_order(tp_order_id, symbol, send_request_common, PRODUCT_TYPE)
            if success:
                print(f"      ✓ TP cancelado exitosamente")
            to_remove.append(idx)
        elif sl_active and not tp_active:
            print(f"      🛑 TP ejecutado! Cancelando SL restante (ID: {sl_order_id})...")
            # Re-verificar que el SL aún existe antes de intentar cancelar
            print(f"      🔍 Verificando si SL {sl_order_id} aún está pendiente...")
            current_pending = get_pending_plan_orders(send_request_common, symbol=symbol, product_type=PRODUCT_TYPE, plan_type='profit_loss')
            current_ids = {p.get('orderId') for p in current_pending if p.get('orderId')}
            
            if sl_order_id in current_ids:
                print(f"      ✓ SL confirmado como pendiente, procediendo a cancelar...")
                success = cancel_plan_order(sl_order_id, symbol, send_request_common, PRODUCT_TYPE)
                if success:
                    print(f"      ✓ SL cancelado exitosamente")
                else:
                    print(f"      ⚠️ Fallo al cancelar, pero removiendo de tracking")
            else:
                print(f"      ℹ️ SL {sl_order_id} ya no está pendiente (ejecutado automáticamente)")
            
            to_remove.append(idx)
        
        # Caso 3: Ambos ejecutados o cancelados → remover posición
        elif not tp_active and not sl_active:
            print(f"      ✓ Ambas órdenes ejecutadas/canceladas")
            to_remove.append(idx)

    for i in reversed(to_remove):
        removed = tracked.pop(i)
        print(f"🗑️ Removida posición: {removed['symbol']} de {strat_id}")

    if to_remove:
        save_state(state)
        print(f"💾 Estado actualizado para {strat_id}")

    return state

# ----------------------
# MAIN LOOP
# ----------------------
def main_loop():
    print("🚀 Iniciando bot con TP/SL independientes por estrategia...")
    exchange = connect_common()
    all_symbols = get_futures_symbols_from_api(PRODUCT_TYPE)

    final_by_strat = {}
    for strat in STRATEGIES:
        final_by_strat[strat['id']] = load_final_symbols(
            all_symbols,
            strategy=strat['name'],
            timeframe=strat['timeframe']
        )
        print(f"📊 Estrategia {strat['id']}: {len(final_by_strat[strat['id']])} símbolos")

    state = load_state(STRATEGIES)
    print("✅ Estado cargado\n")
    print("=" * 60)

    while True:
        try:
            wait_for_next_candle(MIN_TIMEFRAME)
            now = datetime.now(MADRID_TZ).strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n{'=' * 60}")
            print(f"⏰ {now}")
            print(f"{'=' * 60}")

            for strat in STRATEGIES:
                strat_id = strat['id']
                print(f"\n{'─' * 40}")
                print(f"🔄 Procesando estrategia: {strat_id}")
                print(f"{'─' * 40}")

                state = check_and_clean_executed_positions(state, strat)
                state = maybe_open_orders_for_strategy(
                    state,
                    strat,
                    final_by_strat.get(strat_id, []),
                    exchange,
                    use_hardcoded=USE_HARDCODED_SIGNALS
                )

            all_pending = []
            try:
                all_pending = get_pending_plan_orders(send_request_common, product_type=PRODUCT_TYPE, plan_type='profit_loss')
            except:
                pass

            print(f"\n📌 Plan orders activas totales: {len(all_pending)}")

            print(f"\n{'=' * 60}")
            print("📊 RESUMEN DE POSICIONES ACTIVAS:")
            total_positions = 0
            for strat in STRATEGIES:
                strat_id = strat['id']
                tracked = state['strategies'].get(strat_id, [])
                total_positions += len(tracked)
                print(f"   {strat_id}: {len(tracked)} posiciones")
                for pos in tracked:
                    tp_id = pos.get('tp_order_id', 'N/A')
                    sl_id = pos.get('sl_order_id', 'N/A')
                    print(f"      └─ {pos['symbol']} ({pos['direction']}) @ {pos['exec_price']}")
                    print(f"         TP: {tp_id}")
                    print(f"         SL: {sl_id}")
            print(f"   TOTAL: {total_positions} posiciones activas")
            print(f"{'=' * 60}\n")

        except KeyboardInterrupt:
            print("\n🚨 Interrumpido por usuario. Guardando estado...")
            save_state(state)
            print("✅ Estado guardado. Saliendo...")
            break

        except Exception as e:
            print(f"\n⚠️ Error en el loop principal: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)

if __name__ == '__main__':
    main_loop()