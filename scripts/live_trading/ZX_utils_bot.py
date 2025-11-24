import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import json
import copy
import os
import traceback
from datetime import datetime
import numpy as np
from decimal import Decimal, ROUND_DOWN
from ZX_utils_live import fetch_ohlcv_data,normalize_live_ohlcv,df_to_arrays_live
from Z_add_signals_reversal import trend_reversal_entry_short
from Z_add_signals_parity import detect_parity_short
from zoneinfo import ZoneInfo


STATE_FILE   = os.path.join(os.path.dirname(__file__), 'tracked_orders_state.json')
#MADRID_TZ    = ZoneInfo("Europe/Madrid")
BASE_URL     = "https://api.bitget.com"
PRODUCT_TYPE = 'usdt-futures'



# ==========================================================================
# PLACE ORDER
# ==========================================================================   
def fetch_ticker(send_request_func, product_type, symbol):
    code, resp = send_request_func("GET", "/api/v2/mix/market/ticker",params={"productType": product_type, "symbol": symbol})
    if code != 200 or resp.get("code") != "00000":
        print("⚠️ Error ticker:", resp)
        return None, None
    last_price = Decimal(str(resp['data'][0]['lastPr']))
    time.sleep(0.5)
    return last_price, resp


def compute_size_base(usdt_amount, last_price):
    return Decimal(str(usdt_amount)) / last_price


def fetch_contracts(send_request_func, product_type, symbol):
    code_info, resp_info = send_request_func("GET", "/api/v2/mix/market/contracts",params={"productType": product_type, "symbol": symbol})
    if code_info == 200 and resp_info.get("code") == "00000":
        data_list = resp_info.get("data", [])
        if data_list:
            return data_list[0]
    return None


def extract_contract_params(c, last_price):
    """Extrae parámetros de configuración del contrato"""
    price_tick = None
    size_scale = None
    min_trade_num = None
    size_multiplier = None
    min_trade_usdt = None

    if c is None:
        return price_tick, size_scale, min_trade_num, size_multiplier, min_trade_usdt

    # Extraer pricePlace
    if "pricePlace" in c and c.get("pricePlace") is not None:
        try:
            price_tick = Decimal(f"1e-{int(c.get('pricePlace'))}")
        except Exception:
            pass
    if price_tick is None and "priceEndStep" in c and c.get("priceEndStep") is not None:
        try:
            price_tick = Decimal(str(c.get("priceEndStep")))
        except Exception:
            pass

    # Extraer volumePlace
    if "volumePlace" in c and c.get("volumePlace") is not None:
        try:
            size_scale = int(c.get("volumePlace"))
        except Exception:
            pass
    elif "sizeMultiplier" in c and c.get("sizeMultiplier") is not None:
        try:
            sm = Decimal(str(c.get("sizeMultiplier")))
            if sm == sm.to_integral():
                size_scale = 0
            else:
                size_scale = max(0, -sm.as_tuple().exponent)
        except Exception:
            pass

    # Extraer minTradeNum
    if "minTradeNum" in c and c.get("minTradeNum") is not None:
        try:
            min_trade_num = Decimal(str(c.get("minTradeNum")))
        except Exception:
            pass

    # Extraer sizeMultiplier
    if "sizeMultiplier" in c and c.get("sizeMultiplier") is not None:
        try:
            size_multiplier = Decimal(str(c.get("sizeMultiplier")))
        except Exception:
            pass

    # Extraer minTradeUSDT
    if "minTradeUSDT" in c and c.get("minTradeUSDT") is not None:
        try:
            min_trade_usdt = Decimal(str(c.get("minTradeUSDT")))
        except Exception:
            pass

    return price_tick, size_scale, min_trade_num, size_multiplier, min_trade_usdt


def fallback_params(price_tick, size_scale, last_price, min_trade_num=None, min_trade_usdt=None):
    """Aplica valores por defecto si faltan parámetros"""
    if price_tick is None:
        if last_price >= 1000:
            price_tick = Decimal("0.1")
        elif last_price >= 1:
            price_tick = Decimal("0.01")
        elif last_price >= 0.1:
            price_tick = Decimal("0.001")
        else:
            price_tick = Decimal("0.00001")
    
    if size_scale is None or size_scale < 0:
        size_scale = 6
    
    if min_trade_num is None:
        if last_price >= 100:
            min_trade_num = Decimal("0.01")
        elif last_price >= 10:
            min_trade_num = Decimal("0.1")
        else:
            min_trade_num = Decimal("1")
    
    return price_tick, size_scale, min_trade_num, min_trade_usdt


def quantize_size(size_base, size_scale):
    precision_size = Decimal(f"1e-{size_scale}")
    size_q = size_base.quantize(precision_size, rounding=ROUND_DOWN)
    if size_q == 0:
        size_q = size_base.quantize(Decimal("1e-6"), rounding=ROUND_DOWN)
    if size_q == 0:
        print("⚠️ Size = 0")
        return None, precision_size
    return size_q, precision_size


def build_order_body(symbol, product_type, margin_mode, margin_coin, size_q, side, client_oid):
    body = {
        "symbol": symbol,
        "productType": product_type,
        "marginMode": margin_mode,
        "marginCoin": margin_coin,
        "size": format(size_q, "f"),
        "side": side,
        "tradeSide": "open",
        "orderType": "market",
        "clientOid": client_oid if client_oid else f"script-{int(time.time())}"
    }
    return body


def place_market_order(send_request_func, body_order):
    code_order, resp_order = send_request_func("POST", "/api/v2/mix/order/place-order", body=body_order)
    if code_order != 200 or resp_order.get("code") != "00000":
        print("⚠️ Error order:", resp_order)
        return None, None
    return code_order, resp_order


def extract_filled_amount(resp_order, size_q):
    filled_amount = Decimal("0")
    data = resp_order.get("data") or {}
    for k in ("baseVolume", "filledQty", "size", "filledSize", "sz", "filled_amount"):
        if k in data and data[k] is not None:
            try:
                filled_amount = Decimal(str(data[k]))
                if filled_amount > 0:
                    break
            except Exception:
                continue
    if filled_amount == 0:
        filled_amount = size_q
    return filled_amount


def get_exec_price(resp_order, last_price):
    return Decimal(str(resp_order['data'].get('price', last_price)))


def place_order(symbol: str,
                direction: str,
                usdt_amount: float = 100,
                product_type: str = "USDT-FUTURES",
                margin_coin: str = "USDT",
                margin_mode: str = "isolated",
                send_request_func=None,
                client_oid: str = None):

    if send_request_func is None:
        raise ValueError("Send request error.")

    last_price, _ = fetch_ticker(send_request_func, product_type, symbol)
    if last_price is None:
        return None

    size_base = compute_size_base(usdt_amount, last_price)
    c = fetch_contracts(send_request_func, product_type, symbol)
    price_tick, size_scale, min_trade_num, size_multiplier, min_trade_usdt = extract_contract_params(c, last_price)
    price_tick, size_scale, min_trade_num, min_trade_usdt = fallback_params(price_tick, size_scale, last_price, min_trade_num, min_trade_usdt)

    size_q, _ = quantize_size(size_base, size_scale)
    if size_q is None:
        return None

    side       = "buy" if direction.lower() == "long" else "sell"
    body_order = build_order_body(symbol, product_type, margin_mode, margin_coin, size_q, side, client_oid)

    code_order, resp_order = place_market_order(send_request_func, body_order)
    if code_order is None:
        print(f"📊 Debug: last_price={last_price}, price_tick={price_tick},min_num: {min_trade_num}, min_usdt: {min_trade_usdt}")
        return None

    filled_amount = extract_filled_amount(resp_order, size_q)
    exec_price    = get_exec_price(resp_order, last_price)

    print(f"🎯 {('⬆️' if direction=='long' else '⬇️'):2} {direction.capitalize():<6} {symbol:<10} | Size: {filled_amount:<8} | Price: {exec_price:<10}")

    return resp_order


# ==========================================================================
# HELPERS
# ==========================================================================   
def get_fills_for_order(order_id, symbol, product_type='USDT-FUTURES', send_request_func=None, retries=5, delay=0.5):

    if send_request_func is None:
        raise ValueError("Se necesita send_request_func para hacer la consulta")
    
    for attempt in range(retries):
        try:
            code, resp = send_request_func("GET","/api/v2/mix/order/fills",params={"productType": product_type, "orderId": order_id, "symbol": symbol})
            if code == 200 and resp.get("code") == "00000":
                data = resp.get("data") or {}
                fill_list = data.get("fillList") or []
                if fill_list:
                    total_base = Decimal('0')
                    weighted = Decimal('0')
                    for f in fill_list:
                        bv = None
                        for k in ("baseVolume", "filledQty", "size", "filledSize", "sz", "filled_amount"):
                            if k in f and f[k] is not None:
                                bv = f[k]
                                break
                        price = f.get("price") or f.get("execPrice") or f.get("avgPrice") or None
                        try:
                            bv_d = Decimal(str(bv)) if bv is not None else Decimal('0')
                        except Exception:
                            bv_d = Decimal('0')
                        total_base += bv_d
                        if price is not None:
                            try:
                                p_d = Decimal(str(price))
                                weighted += p_d * bv_d
                            except Exception:
                                pass
                    entry_price = (weighted / total_base) if total_base > 0 and weighted > 0 else None
                    return total_base, entry_price
        except Exception as e:
            print(f"⚠️ Error consultando fills (attempt {attempt+1}): {e}")
        time.sleep(delay)
    return None, None

def get_current_price(symbol, send_request_func):
    """Obtiene el precio actual del mercado usando send_request_func"""
    try:
        code, resp = send_request_func("GET","/api/v2/mix/market/ticker",params={"productType": "USDT-FUTURES", "symbol": symbol})
        if code == 200 and resp.get("code") == "00000":
            return Decimal(str(resp['data'][0]['lastPr']))
    except Exception as e:
        print(f"⚠️ Error obteniendo precio de {symbol}: {e}")
    return None

def close_position(symbol, size, direction, send_request_func, reason="NO INFO"):
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
        code, resp = send_request_func("POST", "/api/v2/mix/order/place-order", body=body)
        time.sleep(1.0)

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

def load_state(state_file):
    """
    Carga el estado desde el archivo JSON y devuelve dos estructuras:
      (open_positions_dict, strategy_candles_dict)
    Mantiene la misma lógica de parsing que tenías en el script original.
    """
    OPEN_POSITIONS = {}
    STRATEGY_CANDLES = {}

    if not os.path.exists(state_file):
        print("📂 No se encontró archivo de estado previo")
        return OPEN_POSITIONS, STRATEGY_CANDLES

    try:
        with open(state_file, 'r') as f:
            data = json.load(f)

        # Cargar contador de velas por estrategia
        STRATEGY_CANDLES = data.get('strategy_candles', {})

        # Convertir strings a Decimal donde sea necesario
        positions_data = data.get('positions', {})
        for strat_id, positions in positions_data.items():
            OPEN_POSITIONS[strat_id] = []
            for pos in positions:
                OPEN_POSITIONS[strat_id].append({
                    'symbol': pos.get('symbol'),
                    'size': Decimal(pos.get('size')),
                    'entry_price': Decimal(pos.get('entry_price')),
                    'direction': pos.get('direction'),
                    'tp': Decimal(pos.get('tp')),
                    'sl': Decimal(pos.get('sl')),
                    'order_id': pos.get('order_id'),
                    'opened_at': datetime.fromisoformat(pos.get('opened_at'))
                })

        total_positions = sum(len(p) for p in OPEN_POSITIONS.values())
        print(f"🔹Estado cargado: {total_positions} posiciones recuperadas")

        # Resumen por estrategia
        for strat_id, positions in OPEN_POSITIONS.items():
            if positions:
                candles = STRATEGY_CANDLES.get(strat_id, 0)
                print(f"   ▶️ {strat_id}: {len(positions)} posiciones | Velas: {candles}")
                for pos in positions:
                    print(f"      - {pos['symbol']} | Size: {pos['size']} | Entry: {pos['entry_price']}")

        return OPEN_POSITIONS, STRATEGY_CANDLES

    except Exception as e:
        print(f"🔶 Error cargando estado: {e}")
        traceback.print_exc()
        return OPEN_POSITIONS, STRATEGY_CANDLES

def save_state(open_positions, strategy_candles, state_file, lock):
    try:
        with lock:
            positions_copy = copy.deepcopy(open_positions)
            strategy_candles_copy = copy.deepcopy(strategy_candles)

        serializable_positions = {}
        for strat_id, positions in positions_copy.items():
            serializable_positions[strat_id] = []
            for pos in positions:
                serializable_positions[strat_id].append({
                    'symbol': pos['symbol'],
                    'size': str(pos['size']),
                    'entry_price': str(pos['entry_price']),
                    'direction': pos['direction'],
                    'tp': str(pos['tp']),
                    'sl': str(pos['sl']),
                    'order_id': pos['order_id'],
                    'opened_at': pos['opened_at'].isoformat()
                })

        state_data = {
            'positions': serializable_positions,
            'strategy_candles': strategy_candles_copy
        }

        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2)

    except Exception as e:
        print(f"🔶 Error guardando estado: {e}")
        import traceback
        traceback.print_exc()