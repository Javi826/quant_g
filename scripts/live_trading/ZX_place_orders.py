import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
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

    # 3) Obtener price tick y size scale desde la API (usar /contracts)
    price_tick = None
    size_scale = None
    size_multiplier = None
    min_trade_num = None
    min_trade_usdt = None
    max_market_order_qty = None
    max_order_qty = None

    code_info, resp_info = send_request_func("GET", "/api/v2/mix/market/contracts",params={"productType": product_type, "symbol": symbol})
    if code_info == 200 and resp_info.get("code") == "00000":
        data_list = resp_info.get("data", [])
        if data_list:
            c = data_list[0]  # la entrada del símbolo solicitado
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

            # otros campos útiles (no usados para preservar lógica, pero capturados)
            try:
                if c.get("minTradeNum") is not None:
                    min_trade_num = Decimal(str(c.get("minTradeNum")))
            except Exception:
                pass
            try:
                if c.get("minTradeUSDT") is not None:
                    min_trade_usdt = Decimal(str(c.get("minTradeUSDT")))
            except Exception:
                pass
            try:
                if c.get("maxMarketOrderQty") is not None:
                    max_market_order_qty = Decimal(str(c.get("maxMarketOrderQty")))
            except Exception:
                pass
            try:
                if c.get("maxOrderQty") is not None:
                    max_order_qty = Decimal(str(c.get("maxOrderQty")))
            except Exception:
                pass

    # 3b) Fallback dinámico según magnitud (se mantiene como respaldo)
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

    print(f"🎯 {('⬆️' if direction=='long' else '⬇️'):2} {direction.capitalize():<6} {symbol:<10} | Size: {filled_amount:<8} | Price: {exec_price:<10} | TP: {tp_price:<10} | SL: {sl_price:<10}")


    return resp_order, {"size_tpsl": format(size_tpsl, "f"), "tp_price": format(tp_price, "f"), "sl_price": format(sl_price, "f")}
