from decimal import Decimal, ROUND_DOWN, ROUND_UP
import time


def fetch_ticker(send_request_func, product_type, symbol):
    code, resp = send_request_func("GET", "/api/v2/mix/market/ticker",
                                   params={"productType": product_type, "symbol": symbol})
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
def place_order(symbol: str,
                direction: str,
                usdt_amount: float = 100,
                tp_percent: float = 5,
                sl_percent: float = 5,
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
    precision_size = Decimal(f"1e-{size_scale}")

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
        print(f"   📊 Debug: last_price={last_price}, price_tick={price_tick}, tp={tp_price}, sl={sl_price}")
        return None, None

    # 7) Cantidad ejecutada
    filled_amount = extract_filled_amount(resp_order, size_q)

    # 8) Tamaño para TP/SL
    size_tpsl = compute_size_tpsl(filled_amount, precision_size)
    if size_tpsl is None:
        return resp_order, None

    # Precio real de ejecución
    exec_price = get_exec_price(resp_order, last_price)

    print(f"🎯 {('⬆️' if direction=='long' else '⬇️'):2} {direction.capitalize():<6} {symbol:<10} | Size: {filled_amount:<8} | Price: {exec_price:<10} | TP: {tp_price:<10} | SL: {sl_price:<10}")

    return resp_order, {"size_tpsl": format(size_tpsl, "f"), "tp_price": format(tp_price, "f"), "sl_price": format(sl_price, "f")}