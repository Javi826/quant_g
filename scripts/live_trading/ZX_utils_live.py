import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import time
import pandas as pd
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from ZX_connect_live import send_request_01,send_request_03

BASE_URL     = "https://api.bitget.com"
PRODUCT_TYPE = 'usdt-futures'

# =============================================================================
# SYMBOLS & CANDLES
# =============================================================================
def normalize_live_ohlcv(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df.index = pd.to_datetime(df['timestamp'])
        else:
            df.index = pd.to_datetime(df.index)

    for col in ['open', 'high', 'low', 'close', 'volume_base', 'volume_quote']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

def load_final_symbols(all_symbols, strategy="_", timeframe="4H"):
    folder = os.path.join(os.path.dirname(__file__), "symbols_live")
    folder = os.path.abspath(folder)
    try:
        path_live = os.path.join(folder, f"symbols_live_{strategy}_{timeframe}.xlsx")
        df_live = pd.read_excel(path_live)
        live_symbols = set(df_live.iloc[:, 0].dropna().astype(str))
        final_symbols = set(all_symbols) & live_symbols

        print(f"🔹 Symbols for Live: {len(final_symbols)}")
        return sorted(final_symbols)

    except Exception as e:
        print(f"⚠️ Error loading symbols: {e}")
        return []

def wait_for_next_candle(timeframe='4H'):
    now = datetime.utcnow()
    
    if timeframe.endswith('H'):
        minutes = int(timeframe[:-1]) * 60
    elif timeframe.endswith('m'):
        minutes = int(timeframe[:-1])
    elif timeframe.endswith('D'):
        minutes = int(timeframe[:-1]) * 24 * 60
    else:
        raise ValueError("Invalid timeframe, use 'm', 'H', or 'D'.")

    total_minutes = now.hour * 60 + now.minute
    next_total_minutes = ((total_minutes // minutes) + 1) * minutes
    delta_minutes = next_total_minutes - total_minutes
    next_run = now + timedelta(minutes=delta_minutes, seconds=-now.second, microseconds=-now.microsecond)
    
    sleep_seconds = (next_run - now).total_seconds()
    print(f"🕒 Waiting for next candle: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    time.sleep(sleep_seconds)

def place_order_01(symbol: str, usdt_amount: float = 100, tp_percent: float = 5, sl_percent: float = 5,
                product_type: str = "USDT-FUTURES", margin_coin: str = "USDT", margin_mode: str = "isolated"):

    # 1) último precio
    code, resp = send_request_01("GET", "/api/v2/mix/market/ticker", params={"productType": product_type, "symbol": symbol})
    if code != 200 or resp.get("code") != "00000":
        print("⚠️ Error for ticker:", resp)
        return None, None
    last_price = Decimal(str(resp['data'][0]['lastPr']))
    time.sleep(0.5)

    # 2) tamaño estimado (base)
    size_base = (Decimal(str(usdt_amount)) / last_price)

    # 3) obtener metadata de símbolos para sizeScale y price tick (robusto)
    code_info, resp_info = send_request_01("GET", "/api/v2/mix/market/symbols")
    price_tick = None
    size_scale = None
    if code_info == 200 and resp_info.get("code") == "00000":
        for s in resp_info.get("data", []):
            if s.get("symbol") == symbol:
                # varios posibles campos según la API/versión
                if "priceScale" in s and isinstance(s.get("priceScale"), int):
                    price_tick = Decimal(f"1e-{int(s.get('priceScale'))}")
                elif "tickSize" in s:
                    try:
                        price_tick = Decimal(str(s.get("tickSize")))
                    except:
                        pass
                elif "pricePrecision" in s:
                    price_tick = Decimal(f"1e-{int(s.get('pricePrecision'))}")
                # size scale
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

    # fallbacks seguros
    if price_tick is None:
        # fallback razonable: determina tick por magnitud del precio
        if last_price >= 1:
            price_tick = Decimal("0.01")
        elif last_price >= 0.1:
            price_tick = Decimal("0.001")
        else:
            price_tick = Decimal("0.00001")
    if size_scale is None or size_scale < 0:
        size_scale = 6

    precision_size = Decimal(f"1e-{size_scale}")

    # 4) quantizar size al sizeScale y asegurarnos > 0
    size_q = size_base.quantize(precision_size, rounding=ROUND_DOWN)
    if size_q == 0:
        # intentar fallback con 1e-6
        size_q = size_base.quantize(Decimal("1e-6"), rounding=ROUND_DOWN)
    if size_q == 0:
        print("⚠️ Size obtained = 0. Increase usdt_amount o adjust precision.")
        return None, None

    # 5) calcular TP/SL y quantizar al tick del símbolo
    tp_price = (last_price * (Decimal("1") + Decimal(str(tp_percent)) / 100)).quantize(price_tick, rounding=ROUND_DOWN)
    sl_price = (last_price * (Decimal("1") - Decimal(str(sl_percent)) / 100)).quantize(price_tick, rounding=ROUND_DOWN)

    # 6) colocar orden market incluyendo preset TP/SL (pre-quantized)
    body_order = {
        "symbol": symbol,
        "productType": product_type,
        "marginMode": margin_mode,
        "marginCoin": margin_coin,
        "size": format(size_q, "f"),
        "side": "buy",
        "tradeSide": "open",
        "orderType": "market",
        "clientOid": f"script-{int(time.time())}",
        "presetStopSurplusPrice": format(tp_price, "f"),
        "presetStopLossPrice": format(sl_price, "f")
    }

    code_order, resp_order = send_request_01("POST", "/api/v2/mix/order/place-order", body=body_order)
    if code_order != 200 or resp_order.get("code") != "00000":
        # Si la API responde error por tick, imprimimos detalle adicional para depuración
        print("⚠️ Error in market order:", resp_order)
        return None, None
    
    # 7) obtener cantidad ejecutada (robusto)
    filled_amount = Decimal("0")
    data          = resp_order.get("data") or {}
    for key in ("size", "filledSize", "filledQty", "filled_amount"):
        if key in data and data[key] is not None:
            filled_amount = Decimal(str(data[key]))
            break
    if filled_amount == 0:
        filled_amount = size_q

    # 8) tamaño para TP/SL — no exceder lo fillado y quantizar
    size_tpsl = filled_amount.quantize(precision_size, rounding=ROUND_DOWN)
    if size_tpsl == 0:
        size_tpsl = filled_amount.quantize(Decimal("1e-6"), rounding=ROUND_DOWN)
    if size_tpsl == 0:
        print("⚠️ After execution size_tpsl = 0. Aborting TP/SL.")
        return resp_order, None
    
    # precio real de compra (long)
    buy_price = Decimal(str(resp_order['data'].get('price', last_price)))
    
    print(f"⬆️ & 🎯 Position for {symbol} | Size: {filled_amount} | Price: {buy_price} | TP: {tp_price} | SL: {sl_price}")


    return resp_order, {"size_tpsl": format(size_tpsl, "f"), "tp_price": format(tp_price, "f"), "sl_price": format(sl_price, "f")}


def place_order_03(symbol: str, usdt_amount: float = 100, tp_percent: float = 5, sl_percent: float = 5,
                product_type: str = "USDT-FUTURES", margin_coin: str = "USDT", margin_mode: str = "isolated"):

    # 1) último precio
    code, resp = send_request_03("GET", "/api/v2/mix/market/ticker", params={"productType": product_type, "symbol": symbol})
    if code != 200 or resp.get("code") != "00000":
        print("⚠️ Error for ticker:", resp)
        return None, None
    last_price = Decimal(str(resp['data'][0]['lastPr']))
    time.sleep(0.5)

    # 2) tamaño estimado (base)
    size_base = (Decimal(str(usdt_amount)) / last_price)

    # 3) obtener metadata de símbolos para sizeScale y price tick (robusto)
    code_info, resp_info = send_request_03("GET", "/api/v2/mix/market/symbols")
    price_tick = None
    size_scale = None
    if code_info == 200 and resp_info.get("code") == "00000":
        for s in resp_info.get("data", []):
            if s.get("symbol") == symbol:
                # varios posibles campos según la API/versión
                if "priceScale" in s and isinstance(s.get("priceScale"), int):
                    price_tick = Decimal(f"1e-{int(s.get('priceScale'))}")
                elif "tickSize" in s:
                    try:
                        price_tick = Decimal(str(s.get("tickSize")))
                    except:
                        pass
                elif "pricePrecision" in s:
                    price_tick = Decimal(f"1e-{int(s.get('pricePrecision'))}")
                # size scale
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

    # fallbacks seguros
    if price_tick is None:
        # fallback razonable: determina tick por magnitud del precio
        if last_price >= 1:
            price_tick = Decimal("0.01")
        elif last_price >= 0.1:
            price_tick = Decimal("0.001")
        else:
            price_tick = Decimal("0.00001")
    if size_scale is None or size_scale < 0:
        size_scale = 6

    precision_size = Decimal(f"1e-{size_scale}")

    # 4) quantizar size al sizeScale y asegurarnos > 0
    size_q = size_base.quantize(precision_size, rounding=ROUND_DOWN)
    if size_q == 0:
        # intentar fallback con 1e-6
        size_q = size_base.quantize(Decimal("1e-6"), rounding=ROUND_DOWN)
    if size_q == 0:
        print("⚠️ Size obtained = 0. Increase usdt_amount o adjust precision.")
        return None, None

    # 5) calcular TP/SL y quantizar al tick del símbolo
    tp_price = (last_price * (Decimal("1") + Decimal(str(tp_percent)) / 100)).quantize(price_tick, rounding=ROUND_DOWN)
    sl_price = (last_price * (Decimal("1") - Decimal(str(sl_percent)) / 100)).quantize(price_tick, rounding=ROUND_DOWN)

    # 6) colocar orden market incluyendo preset TP/SL (pre-quantized)
    body_order = {
        "symbol": symbol,
        "productType": product_type,
        "marginMode": margin_mode,
        "marginCoin": margin_coin,
        "size": format(size_q, "f"),
        "side": "buy",
        "tradeSide": "open",
        "orderType": "market",
        "clientOid": f"script-{int(time.time())}",
        "presetStopSurplusPrice": format(tp_price, "f"),
        "presetStopLossPrice": format(sl_price, "f")
    }

    code_order, resp_order = send_request_03("POST", "/api/v2/mix/order/place-order", body=body_order)
    if code_order != 200 or resp_order.get("code") != "00000":
        # Si la API responde error por tick, imprimimos detalle adicional para depuración
        print("⚠️ Error in market order:", resp_order)
        return None, None
    
    # 7) obtener cantidad ejecutada (robusto)
    filled_amount = Decimal("0")
    data          = resp_order.get("data") or {}
    for key in ("size", "filledSize", "filledQty", "filled_amount"):
        if key in data and data[key] is not None:
            filled_amount = Decimal(str(data[key]))
            break
    if filled_amount == 0:
        filled_amount = size_q

    # 8) tamaño para TP/SL — no exceder lo fillado y quantizar
    size_tpsl = filled_amount.quantize(precision_size, rounding=ROUND_DOWN)
    if size_tpsl == 0:
        size_tpsl = filled_amount.quantize(Decimal("1e-6"), rounding=ROUND_DOWN)
    if size_tpsl == 0:
        print("⚠️ After execution size_tpsl = 0. Aborting TP/SL.")
        return resp_order, None
    
    # precio real de compra (long)
    buy_price = Decimal(str(resp_order['data'].get('price', last_price)))
    
    print(f"⬆️ & 🎯 Position for {symbol} | Size: {filled_amount} | Price: {buy_price} | TP: {tp_price} | SL: {sl_price}")


    return resp_order, {"size_tpsl": format(size_tpsl, "f"), "tp_price": format(tp_price, "f"), "sl_price": format(sl_price, "f")}
