import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import os
import time
import json
import hashlib
import base64
import hmac
import pandas as pd
import requests
from urllib.parse import urlencode
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN, ROUND_UP
from typing import Dict, Any, List
from utils.ZZ_connect import BITGET_API_KEY_01, BITGET_API_SECRET_01, BITGET_API_PASS_01

BASE_URL = "https://api.bitget.com"
PRODUCT_TYPE = 'usdt-futures'

# =============================================================================
# BALANCE
# =============================================================================
def get_usdt_balance(exchange):
    balance = exchange.fetch_balance()
    return balance['free']['USDT']

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

# =============================================================================
# OPEN POSITIONS
# =============================================================================
def make_get(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    qs = "&".join(f"{k}={v}" for k, v in params.items() if v not in [None, ""])
    url = BASE_URL + endpoint + (f"?{qs}" if qs else "")
    timestamp = str(int(time.time() * 1000))
    sign = sign_request(timestamp, "GET", endpoint, qs, "")
    headers = {
        "ACCESS-KEY": BITGET_API_KEY_01,
        "ACCESS-SIGN": sign,
        "ACCESS-PASSPHRASE": BITGET_API_PASS_01,
        "ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()

def get_open_positions(product_type: str = "USDT-FUTURES") -> List[Dict[str, Any]]:
    endpoint = "/api/v2/mix/position/all-position"
    params = {"productType": product_type}
    response = make_get(endpoint, params)
    return response.get("data", [])

# =============================================================================
# PLACE ORDER
# =============================================================================
def _now_ms():
    return str(int(time.time() * 1000))

def _body_to_str(body):
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False) if body else ""

def sign_request(timestamp, method, path, query_string, body_str):
    to_sign = timestamp + method.upper() + path
    if query_string:
        to_sign += "?" + query_string
    to_sign += body_str
    digest = hmac.new(BITGET_API_SECRET_01.encode('utf-8'), to_sign.encode('utf-8'), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()

def send_request(method, path, params=None, body=None):
    ts = _now_ms()
    query_string = urlencode(params) if params else ""
    body_str = _body_to_str(body)
    sign = sign_request(ts, method, path, query_string, body_str)
    headers = {
        "ACCESS-KEY": BITGET_API_KEY_01,
        "ACCESS-SIGN": sign,
        "ACCESS-TIMESTAMP": ts,
        "ACCESS-PASSPHRASE": BITGET_API_PASS_01,
        "Content-Type": "application/json"
    }
    url = BASE_URL + path + (f"?{query_string}" if query_string else "")
    try:
        if method.upper() != "GET":
            r = requests.post(url, headers=headers, data=body_str.encode('utf-8'), timeout=15)
        else:
            r = requests.get(url, headers=headers, timeout=15)
        ct = r.headers.get("Content-Type", "")
        return r.status_code, r.json() if ct.startswith("application/json") else r.text
    except Exception as e:
        return 0, {"error": str(e)}

def place_order(symbol: str,
                usdt_amount: float = 100,
                tp_percent: float = 5,
                sl_percent: float = 5,
                product_type: str = "USDT-FUTURES",
                margin_coin: str = "USDT",
                margin_mode: str = "isolated"):
    
    # 1) Get current price
    code, resp = send_request("GET", "/api/v2/mix/market/ticker", 
                              params={"productType": product_type, "symbol": symbol})
    if code != 200 or resp.get("code") != "00000":
        print("⚠️ Error getting ticker:", resp)
        return None
    
    last_price = Decimal(str(resp['data'][0]['lastPr']))
    time.sleep(0.5)
    
    # 2) Get symbol metadata
    code_info, resp_info = send_request("GET", "/api/v2/mix/market/symbols")
    price_tick = Decimal("0.01")  # default
    size_scale = 6  # default
    
    if code_info == 200 and resp_info.get("code") == "00000":
        for s in resp_info.get("data", []):
            if s.get("symbol") == symbol:
                if "priceScale" in s:
                    price_tick = Decimal(f"1e-{int(s['priceScale'])}")
                elif "tickSize" in s:
                    price_tick = Decimal(str(s["tickSize"]))
                if "sizeScale" in s:
                    size_scale = int(s["sizeScale"])
                elif "qtyScale" in s:
                    size_scale = int(s["qtyScale"])
                break
    
    precision_size = Decimal(f"1e-{size_scale}")
    
    # 3) Calculate and quantize size
    size_base = Decimal(str(usdt_amount)) / last_price
    size_q = size_base.quantize(precision_size, rounding=ROUND_DOWN)
    
    if size_q == 0:
        print(f"⚠️ Size = 0 with {usdt_amount} USDT. Increase the amount.")
        return None
    
    # 4) Calculate TP/SL
    tp_price = (last_price * (Decimal("1") + Decimal(str(tp_percent)) / 100)).quantize(price_tick, rounding=ROUND_UP)
    sl_price = (last_price * (Decimal("1") - Decimal(str(sl_percent)) / 100)).quantize(price_tick, rounding=ROUND_DOWN)
    
    # 5) Place order with preset TP/SL
    body_order = {
        "symbol": symbol,
        "productType": product_type,
        "marginMode": margin_mode,
        "marginCoin": margin_coin,
        "size": format(size_q, "f"),
        "side": "buy",
        "tradeSide": "open",
        "orderType": "market",
        "clientOid": f"script-{int(time.time() * 1000)}",
        "presetTakeProfitPrice": format(tp_price, "f"),
        "presetStopLossPrice": format(sl_price, "f")
    }
    
    code_order, resp_order = send_request("POST", "/api/v2/mix/order/place-order", body=body_order)
    
    if code_order != 200 or resp_order.get("code") != "00000":
        print("⚠️ Error opening position:", resp_order)
        return None
    
    print(f"✅ Position opened: {symbol} | Size: {size_q} | Price: {last_price} | TP: {tp_price} | SL: {sl_price}")
    
    return resp_order
