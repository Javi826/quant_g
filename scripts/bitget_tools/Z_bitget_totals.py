import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import requests
import hashlib
import base64
import hmac
from typing import Dict, Any, List

from utils.ZZ_connect import BITGET_API_KEY_TT, BITGET_API_SECRET_TT, BITGET_API_PASS_TT

BASE_URL        = "https://api.bitget.com"
PRODUCT_TYPE    = "USDT-FUTURES"
INITIAL_CAPITAL = 4223  # Capital inicial

# -----------------------------
# FIRMA DE PETICIONES
# -----------------------------
def sign_request_TT(timestamp, method, path, query_string, body_str):
    to_sign = timestamp + method.upper() + path
    if query_string:
        to_sign += "?" + query_string
    to_sign += body_str
    digest = hmac.new(BITGET_API_SECRET_TT.encode("utf-8"), to_sign.encode("utf-8"), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()

# -----------------------------
# PETICIÓN GET FIRMADA
# -----------------------------
def make_get_TT(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    qs = "&".join(f"{k}={v}" for k, v in params.items() if v not in [None, ""])
    url = BASE_URL + endpoint + (f"?{qs}" if qs else "")
    timestamp = str(int(time.time() * 1000))
    sign = sign_request_TT(timestamp, "GET", endpoint, qs, "")
    headers = {
        "ACCESS-KEY": BITGET_API_KEY_TT,
        "ACCESS-SIGN": sign,
        "ACCESS-PASSPHRASE": BITGET_API_PASS_TT,
        "ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()

# -----------------------------
# OBTENER TOTAL USDT EN SUBCUENTAS FUTURES
# -----------------------------
def get_subaccounts_usdt_total(product_type: str = "USDT-FUTURES"):
    endpoint = "/api/v2/mix/account/sub-account-assets"
    params = {"productType": product_type}
    response = make_get_TT(endpoint, params)

    data = response.get("data", [])
    total_usdt = 0.0
    balances = []

    for sub in data:
        user_id = sub.get("userId")
        assets = sub.get("assetList", [])
        for asset in assets:
            if asset.get("marginCoin") == "USDT":
                usdt_equity = float(asset.get("usdtEquity", 0) or 0)
                total_usdt += usdt_equity
                balances.append({"userId": user_id, "usdtEquity": usdt_equity})

    return total_usdt, balances

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    total_usdt, balances = get_subaccounts_usdt_total()

    print("💰 Balances USDT (Futures):")
    for b in balances:
        print(f" - Subaccount {b['userId']}: {b['usdtEquity']:.4f} USDT")

    print(f"\n🏦 Total USDT : {total_usdt:.4f} USDT")

    # 🔹 Cálculo de delta y rentabilidad
    delta = total_usdt - INITIAL_CAPITAL
    profitability_pct = (delta / INITIAL_CAPITAL) * 100

    print("\n📈 Brief:")
    print(f"💵 INITIAL CAPITAL : {INITIAL_CAPITAL:.2f} USDT")
    print(f"💰 CURRENT CAPITAL : {total_usdt:.2f} USDT")
    print(f"📊 Delta           : {delta:+.2f} USDT")
    print(f"📈 Profit %        : {profitability_pct:+.2f}%")
