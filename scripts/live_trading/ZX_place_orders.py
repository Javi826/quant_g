
from decimal import Decimal, ROUND_DOWN
import time


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

    if c is None:
        return price_tick, size_scale

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

    return price_tick, size_scale


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
    price_tick, size_scale = extract_contract_params(c, last_price)
    price_tick, size_scale = fallback_params(price_tick, size_scale, last_price)

    size_q, _ = quantize_size(size_base, size_scale)
    if size_q is None:
        return None

    side       = "buy" if direction.lower() == "long" else "sell"
    body_order = build_order_body(symbol, product_type, margin_mode, margin_coin, size_q, side, client_oid)

    code_order, resp_order = place_market_order(send_request_func, body_order)
    if code_order is None:
        print(f"📊 Debug: last_price={last_price}, price_tick={price_tick}")
        return None

    filled_amount = extract_filled_amount(resp_order, size_q)
    exec_price    = get_exec_price(resp_order, last_price)

    print(f"🎯 {('⬆️' if direction=='long' else '⬇️'):2} {direction.capitalize():<6} {symbol:<10} | Size: {filled_amount:<8} | Price: {exec_price:<10}")

    return resp_order
