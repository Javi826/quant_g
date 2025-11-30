import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from ZX_utils_live import fetch_ohlcv_data
from datetime import datetime
from zoneinfo import ZoneInfo
import os
import json

MADRID_TZ    = ZoneInfo("Europe/Madrid")
BASE_URL     = "https://api.bitget.com"
PRODUCT_TYPE = 'usdt-futures'

#=================================================================
# PLACE ORDER
#================================================================


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
    code_info, resp_info = send_request_func("GET", "/api/v2/mix/market/contracts",
                                            params={"productType": product_type, "symbol": symbol})
    if code_info == 200 and resp_info.get("code") == "00000":
        data_list = resp_info.get("data", [])
        if data_list:
            return data_list[0]
    return None


def extract_contract_params(c, last_price):
    price_tick = None
    size_scale = None
    size_multiplier = None
    min_trade_num = None
    min_trade_usdt = None
    max_market_order_qty = None
    max_order_qty = None

    if c is None:
        return (price_tick, size_scale, size_multiplier,
                min_trade_num, min_trade_usdt, max_market_order_qty, max_order_qty)

    # price tick — preferimos pricePlace, fallback a priceEndStep
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

    # size scale — preferimos volumePlace; si no existe, inferir de sizeMultiplier
    if "volumePlace" in c and c.get("volumePlace") is not None:
        try:
            size_scale = int(c.get("volumePlace"))
        except Exception:
            pass
    elif "sizeMultiplier" in c and c.get("sizeMultiplier") is not None:
        try:
            sm = Decimal(str(c.get("sizeMultiplier")))
            size_multiplier = sm
            # inferimos cantidad de decimales de sizeMultiplier si es tipo 0.01 etc.
            if sm == sm.to_integral():  # entero
                size_scale = 0
            else:
                # número de decimales = -exponente de Decimal
                size_scale = max(0, -sm.as_tuple().exponent)
        except Exception:
            pass

    return (price_tick, size_scale, size_multiplier,
            min_trade_num, min_trade_usdt, max_market_order_qty, max_order_qty)


def fallback_params(price_tick, size_scale, last_price):
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
    return price_tick, size_scale


def quantize_size(size_base, size_scale):
    precision_size = Decimal(f"1e-{size_scale}")
    size_q = size_base.quantize(precision_size, rounding=ROUND_DOWN)
    if size_q == 0:
        size_q = size_base.quantize(Decimal("1e-6"), rounding=ROUND_DOWN)
    if size_q == 0:
        print("⚠️ Size = 0")
        return None, precision_size
    return size_q, precision_size


def calculate_tp_sl(last_price, tp_percent, sl_percent, direction, price_tick):
    if direction.lower() == "long":
        tp_price = (last_price * (1 + Decimal(str(tp_percent)) / 100)).quantize(price_tick, rounding=ROUND_DOWN)
        sl_price = (last_price * (1 - Decimal(str(sl_percent)) / 100)).quantize(price_tick, rounding=ROUND_DOWN)
        side = "buy"
    elif direction.lower() == "short":
        tp_price = (last_price * (1 - Decimal(str(tp_percent)) / 100)).quantize(price_tick, rounding=ROUND_DOWN)
        sl_price = (last_price * (1 + Decimal(str(sl_percent)) / 100)).quantize(price_tick, rounding=ROUND_UP)
        side = "sell"
    else:
        raise ValueError("direction should be 'long' o 'short'")
    return tp_price, sl_price, side


def build_order_body(symbol, product_type, margin_mode, margin_coin, size_q, side, client_oid, tp_price, sl_price):
    return {
        "symbol": symbol,
        "productType": product_type,
        "marginMode": margin_mode,
        "marginCoin": margin_coin,
        "size": format(size_q, "f"),
        "side": side,
        "tradeSide": "open",
        "orderType": "market",
        "clientOid": client_oid if client_oid else f"script-{int(time.time())}",
        "presetStopSurplusPrice": format(tp_price, "f"),
        "presetStopLossPrice": format(sl_price, "f")
    }


def place_market_order(send_request_func, body_order):
    code_order, resp_order = send_request_func("POST", "/api/v2/mix/order/place-order", body=body_order)
    if code_order != 200 or resp_order.get("code") != "00000":
        print("⚠️ Error order:", resp_order)
        return None, None
    return code_order, resp_order


def extract_filled_amount(resp_order, size_q):
    filled_amount = Decimal("0")
    data = resp_order.get("data") or {}
    for k in ("size", "filledSize", "filledQty", "filled_amount"):
        if k in data and data[k] is not None:
            filled_amount = Decimal(str(data[k]))
            break
    if filled_amount == 0:
        filled_amount = size_q
    return filled_amount


def compute_size_tpsl(filled_amount, precision_size):
    size_tpsl = filled_amount.quantize(precision_size, rounding=ROUND_DOWN)
    if size_tpsl == 0:
        size_tpsl = filled_amount.quantize(Decimal("1e-6"), rounding=ROUND_DOWN)
    if size_tpsl == 0:
        print("⚠️ After execution size_tpsl = 0. Aborting TP/SL.")
        return None
    return size_tpsl


def get_exec_price(resp_order, last_price):
    return Decimal(str(resp_order['data'].get('price', last_price)))


# Función principal (mantener firma y comportamiento)
def place_order_sub(symbol: str,
                direction: str,
                usdt_amount: float = 100,
                tp_percent: float = 0,
                sl_percent: float = 0,
                product_type: str = "USDT-FUTURES",
                margin_coin: str = "USDT",
                margin_mode: str = "isolated",
                send_request_func=None,
                client_oid: str = None):

    if send_request_func is None:
        raise ValueError("Send request error.")

    # 1) Último precio
    last_price, _ = fetch_ticker(send_request_func, product_type, symbol)
    if last_price is None:
        return None, None

    # 2) Tamaño base
    size_base = compute_size_base(usdt_amount, last_price)

    # 3) Obtener price tick y size scale desde la API (usar /contracts)
    c = fetch_contracts(send_request_func, product_type, symbol)
    (price_tick, size_scale, size_multiplier,
     min_trade_num, min_trade_usdt, max_market_order_qty, max_order_qty) = extract_contract_params(c, last_price)

    # 3b) Fallback dinámico según magnitud (se mantiene como respaldo)
    price_tick, size_scale = fallback_params(price_tick, size_scale, last_price)
    precision_size         = Decimal(f"1e-{size_scale}")

    # 4) Quantizar tamaño
    size_q, precision_size = quantize_size(size_base, size_scale)
    if size_q is None:
        return None, None

    # 5) Calcular TP/SL según dirección
    tp_price, sl_price, side = calculate_tp_sl(last_price, tp_percent, sl_percent, direction, price_tick)

    # 6) Preparar orden
    body_order = build_order_body(symbol, product_type, margin_mode, margin_coin, size_q, side, client_oid, tp_price, sl_price)

    code_order, resp_order = place_market_order(send_request_func, body_order)
    if code_order is None:
        print(f"   ➡ Debug: last_price={last_price}, price_tick={price_tick}, tp={tp_price}, sl={sl_price}")
        return None, None

    # 7) Cantidad ejecutada
    filled_amount = extract_filled_amount(resp_order, size_q)

    # 8) Tamaño para TP/SL
    size_tpsl = compute_size_tpsl(filled_amount, precision_size)
    if size_tpsl is None:
        return resp_order, None

    # Precio real de ejecución
    exec_price = get_exec_price(resp_order, last_price)

    print(f"✅ {('⬆️' if direction=='long' else '⬇️'):2} {direction.capitalize():<6} {symbol:<10} | Size: {filled_amount:<8} | Price: {exec_price:<10} | TP: {tp_price:<10} | SL: {sl_price:<10}")

    return resp_order, {"size_tpsl": format(size_tpsl, "f"), "tp_price": format(tp_price, "f"), "sl_price": format(sl_price, "f")}

#=================================================================
# LOOP FUNCTIONS
#================================================================
def has_open_positions_on_exchange(get_open_fn, product_type: str):

    try:
        pos_list = get_open_fn(product_type=product_type.upper())
        return bool(pos_list)
    except Exception as e:
        print(f"⚠️ Error checking positions: {e}")
        return True

def process_signals_and_buy(
    final_symbols,
    exchange,
    open_positions,
    order_amount,
    timeframe_minor,
    sell_after_n_candles,
    tp_pct,
    sl_pct,
    direction,
    send_request_fn,
    get_balance_fn,
    check_signal_fn 
):

    ohlcv_data = fetch_ohlcv_data(final_symbols, timeframe_minor)
    ohlcv_data = {sym: {"minor": df} for sym, df in ohlcv_data.items()}

    # ----------------------------------------
    # 2️⃣ Detectar señales
    # ----------------------------------------
    detected_signals = []
    for sym, dfs in ohlcv_data.items():
        signal = check_signal_fn(dfs["minor"], sym)
        if signal:
            detected_signals.append(signal)

    print(f"\n✨ {datetime.now(MADRID_TZ).strftime('%H:%M')} - Signals detected: {len(detected_signals)}")

    # ----------------------------------------
    # 3️⃣ Ejecutar órdenes
    # ----------------------------------------
    for signal in detected_signals:
        sym = signal["symbol"]
        usdt_balance = get_balance_fn(exchange)
        now = datetime.now(MADRID_TZ).replace(second=0, microsecond=0) 

        if usdt_balance < order_amount:
            print(f"⚠️ {now} - USDT balance too low to place order for {sym}")
            continue

        order, tpsl_info = place_order_sub(
            sym,
            direction=direction,
            usdt_amount=order_amount,
            tp_percent=tp_pct,
            sl_percent=sl_pct,
            send_request_func=send_request_fn
        )

        if order is not None:
            buy_price     = float(order['data'].get('price', signal['close']))
            filled_amount = float(order['data'].get('size', order_amount / buy_price))

            open_positions.append({
                'symbol': sym,
                'buy_price': buy_price,
                'amount': filled_amount,
                'candles_to_sell': sell_after_n_candles,
                'just_bought': True
            })

            usdt_after = get_balance_fn(exchange)
            print(f"➡ {now} - ORDER executed: {sym} | Remaining USDT: {usdt_after:.2f}\n")
            time.sleep(2)

        else:
            print(f"⚠️ {now} - Order for {sym} was not executed or returned None.")

    return open_positions

def manage_open_positions(open_positions, send_request_fn, product_type=PRODUCT_TYPE):

    for pos in open_positions[:]:
        if pos.get('just_bought', False):
            pos['just_bought'] = False
            continue

        pos['candles_to_sell'] -= 1

        if pos['candles_to_sell'] <= 0:
            try:
                body = {"symbol": pos['symbol'], "productType": product_type}
                code, resp = send_request_fn("POST", "/api/v2/mix/order/close-positions", body=body)
                now = datetime.now(MADRID_TZ).replace(second=0, microsecond=0)

                if code == 200 and resp.get("code") == "00000":
                    for success in resp['data']['successList']:
                        code_ticker, resp_ticker = send_request_fn("GET", "/api/v2/mix/market/ticker", params={"productType": product_type, "symbol": success['symbol']})
                        sell_price = None
                        if code_ticker == 200 and resp_ticker.get("code") == "00000":
                            sell_price = resp_ticker['data'][0]['lastPr']

                        print(f"💰 {now.strftime('%Y-%m-%d %H:%M:%S')} - FLASH CLOSE: {success['symbol']} | Sold at: {sell_price}")
                else:
                    print(f"⚠️ {now} - Failed Flash Close for {pos['symbol']}: {resp}")

            except Exception as e:
                now = datetime.now(MADRID_TZ).replace(second=0, microsecond=0)
                print(f"⚠️ {now} - Error closing position {pos['symbol']}: {e}")
            finally:
                try:
                    open_positions.remove(pos)
                except ValueError:
                    pass

            time.sleep(1.1)
            

# ============================================
# ZX_utils_sub.py - FUNCIONES A AÑADIR
# ============================================
import os
from zoneinfo import ZoneInfo

MADRID_TZ = ZoneInfo("Europe/Madrid")

# ----------------------
# PERSISTENCIA DE ESTADO
# ----------------------

def load_state(state_file):
    """Carga el estado guardado desde el archivo JSON"""
    if not os.path.exists(state_file):
        return []
    
    try:
        with open(state_file, "r") as f:
            state = json.load(f)
        print(f"✅ Estado cargado: {len(state)} posiciones")
        return state
    except Exception as e:
        print(f"⚠️ Error cargando estado: {e}")
        return []

def save_state(open_positions, state_file):
    """Guarda el estado actual en el archivo JSON"""
    try:
        with open(state_file, "w") as f:
            json.dump(open_positions, f, indent=4)
            
        #print(f"💾 Saving state...")
    except Exception as e:
        print(f"⚠️ Error guardando estado: {e}")

# ----------------------
# SINCRONIZACIÓN CON EXCHANGE
# ----------------------

def sync_positions_with_exchange(open_positions, get_open_fn, product_type: str):
    """Sincroniza el estado interno con las posiciones reales del exchange"""
    print(f"🌐 Syncronicing with broker...")
    try:
        exchange_positions = get_open_fn(product_type=product_type.upper())
        
        if not exchange_positions:
            if open_positions:
                print("🔄 No positions on exchange, clearing internal state (TP/SL hit)")
                open_positions.clear()
            return
        
        exchange_symbols = {pos['symbol'] for pos in exchange_positions}
        
        positions_to_remove = []
        for pos in open_positions:
            if pos['symbol'] not in exchange_symbols:
                positions_to_remove.append(pos)
                now = datetime.now(MADRID_TZ).replace(second=0, microsecond=0)
                print(f"➡  {now} - Position {pos['symbol']} closed on exchange (TP/SL hit)")
        
        for pos in positions_to_remove:
            open_positions.remove(pos)
        
        if positions_to_remove:
            print(f"✅ Removed {len(positions_to_remove)} closed positions from internal state")
            
    except Exception as e:
        print(f"⚠️ Error syncing positions: {e}")
