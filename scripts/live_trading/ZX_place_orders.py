import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import time
from decimal import Decimal, ROUND_DOWN,ROUND_UP

BASE_URL     = "https://api.bitget.com"
PRODUCT_TYPE = 'usdt-futures'


# =============================================================================
# TT
# =============================================================================
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
    code, resp = send_request_func("GET", "/api/v2/mix/market/ticker",
                                   params={"productType": product_type, "symbol": symbol})
    if code != 200 or resp.get("code") != "00000":
        print("⚠️ Error ticker:", resp)
        return None, None
    last_price = Decimal(str(resp['data'][0]['lastPr']))
    time.sleep(0.5)

    # 2) Tamaño base
    size_base = Decimal(str(usdt_amount)) / last_price

    # 3) Obtener price tick y size scale desde la API
    code_info, resp_info = send_request_func("GET", "/api/v2/mix/market/symbols")
    price_tick = None
    size_scale = None
    if code_info == 200 and resp_info.get("code") == "00000":
        for s in resp_info.get("data", []):
            if s.get("symbol") == symbol:
                # Price tick
                if "priceScale" in s and isinstance(s.get("priceScale"), int):
                    price_tick = Decimal(f"1e-{int(s.get('priceScale'))}")
                elif "tickSize" in s:
                    try:
                        price_tick = Decimal(str(s.get("tickSize")))
                    except:
                        pass
                elif "pricePrecision" in s:
                    price_tick = Decimal(f"1e-{int(s.get('pricePrecision'))}")
                # Size scale
                if "sizeScale" in s:
                    try:
                        size_scale = int(s.get("sizeScale"))
                    except:
                        pass
                elif "qtyScale" in s:
                    try:
                        size_scale = int(s.get("qtyScale"))
                    except:
                        pass
                break

    # 3b) Fallback dinámico según magnitud
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

    precision_size = Decimal(f"1e-{size_scale}")

    # 4) Quantizar tamaño
    size_q = size_base.quantize(precision_size, rounding=ROUND_DOWN)
    if size_q == 0:
        size_q = size_base.quantize(Decimal("1e-6"), rounding=ROUND_DOWN)
    if size_q == 0:
        print("⚠️ Size = 0")
        return None, None

    # 5) Calcular TP/SL según dirección
    if direction.lower() == "long":
        tp_price = (last_price * (1 + Decimal(str(tp_percent))/100)).quantize(price_tick, rounding=ROUND_DOWN)
        sl_price = (last_price * (1 - Decimal(str(sl_percent))/100)).quantize(price_tick, rounding=ROUND_DOWN)
        side = "buy"
    elif direction.lower() == "short":
        tp_price = (last_price * (1 - Decimal(str(tp_percent))/100)).quantize(price_tick, rounding=ROUND_DOWN)
        sl_price = (last_price * (1 + Decimal(str(sl_percent))/100)).quantize(price_tick, rounding=ROUND_UP)
        side = "sell"
    else:
        raise ValueError("direction should be 'long' o 'short'")

    # 6) Preparar orden
    body_order = {
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

    code_order, resp_order = send_request_func("POST", "/api/v2/mix/order/place-order", body=body_order)
    if code_order != 200 or resp_order.get("code") != "00000":
        print("⚠️ Error order:", resp_order)
        print(f"   📊 Debug: last_price={last_price}, price_tick={price_tick}, tp={tp_price}, sl={sl_price}")
        return None, None

    # 7) Cantidad ejecutada
    filled_amount = Decimal("0")
    data = resp_order.get("data") or {}
    for k in ("size", "filledSize", "filledQty", "filled_amount"):
        if k in data and data[k] is not None:
            filled_amount = Decimal(str(data[k]))
            break
    if filled_amount == 0:
        filled_amount = size_q

    # 8) Tamaño para TP/SL
    size_tpsl = filled_amount.quantize(precision_size, rounding=ROUND_DOWN)
    if size_tpsl == 0:
        size_tpsl = filled_amount.quantize(Decimal("1e-6"), rounding=ROUND_DOWN)
    if size_tpsl == 0:
        print("⚠️ After execution size_tpsl = 0. Aborting TP/SL.")
        return resp_order, None

    # Precio real de ejecución
    exec_price = Decimal(str(resp_order['data'].get('price', last_price)))

    print(f"🎯 {'⬆️' if direction=='long' else '⬇️'} {direction.capitalize()} Position {symbol}     | "
          f"Size: {filled_amount}     | Price: {exec_price}         | TP: {tp_price}     | SL: {sl_price}")

    return resp_order, {"size_tpsl": format(size_tpsl, "f"), "tp_price": format(tp_price, "f"), "sl_price": format(sl_price, "f")}
