import os
import sys
import time
import hmac
import hashlib
import base64
import json
import requests
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.ZZ_connect import connect_bitget,BITGET_API_KEY,BITGET_API_SECRET,BITGET_API_PASS
from ZZ_connect import connect_bitget,BITGET_API_KEY,BITGET_API_SECRET,BITGET_API_PASS

# -----------------------------
# BITGET ENTROPIA - CREDENCIALES
# -----------------------------

BASE_URL   = "https://api.bitget.com"

PRODUCT_TYPE = "USDT-FUTURES"  # Cambiar si es COIN-FUTURES o USDC-FUTURES

# -----------------------------
# FUNCIONES
# -----------------------------
def _now_ms():
    return str(int(time.time() * 1000))

def _body_to_str(body):
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False) if body else ""

def sign_request(timestamp, method, path, query_string, body_str):
    to_sign = timestamp + method.upper() + path
    if query_string:
        to_sign += "?" + query_string
    to_sign += body_str
    digest = hmac.new(BITGET_API_SECRET.encode('utf-8'), to_sign.encode('utf-8'), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()

def send_request(method, path, params=None, body=None):
    ts = _now_ms()
    query_string = "&".join(f"{k}={v}" for k, v in params.items()) if params else ""
    body_str = _body_to_str(body)
    sign = sign_request(ts, method, path, query_string, body_str)
    headers = {
        "ACCESS-KEY": BITGET_API_KEY,
        "ACCESS-SIGN": sign,
        "ACCESS-TIMESTAMP": ts,
        "ACCESS-PASSPHRASE": BITGET_API_PASS,
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

# -----------------------------
# CERRAR TODAS LAS POSICIONES ABIERTAS
# -----------------------------
def close_all_positions():
    code, resp = send_request(
        "GET",
        "/api/v2/mix/position/all-position",
        params={"productType": PRODUCT_TYPE, "marginCoin": "USDT"}
    )

    if code != 200 or resp.get("code") != "00000":
        print("⚠️ Error fetching positions:", resp)
        return

    positions = [p for p in resp['data'] if float(p['total']) > 0]

    if not positions:
        print("ℹ️ No open positions to close.")
        return

    for pos in positions:
        body = {
            "symbol": pos['symbol'],
            "productType": PRODUCT_TYPE
        }
        close_code, close_resp = send_request("POST", "/api/v2/mix/order/close-positions", body=body)
        if close_code == 200 and close_resp.get("code") == "00000":
            print(f"💰 FLASH CLOSE executed: {pos['symbol']}")
        else:
            print(f"⚠️ Failed to close {pos['symbol']}: {close_resp}")
        time.sleep(1.1)  # evitar limitación: 1 request/seg

if __name__ == "__main__":
    close_all_positions()
