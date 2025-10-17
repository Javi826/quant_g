import os
import sys
import requests
import hmac
import hashlib
import base64
import time
from datetime import datetime
from typing import Dict, Any, List

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.ZZ_connect import BITGET_API_KEY_03, BITGET_API_SECRET_03, BITGET_API_PASS_03

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
BASE_URL = "https://api.bitget.com"

# -----------------------------
# Funciones auxiliares
# -----------------------------
def sign_request(timestamp: str, method: str, request_path: str, query_string: str = "", body: str = "") -> str:
    to_sign = timestamp + method.upper() + request_path
    if query_string:
        to_sign += "?" + query_string
    to_sign += body
    signature = hmac.new(BITGET_API_SECRET_03.encode("utf-8"), to_sign.encode("utf-8"), hashlib.sha256).digest()
    return base64.b64encode(signature).decode()

def make_get(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    qs = "&".join(f"{k}={v}" for k, v in params.items() if v not in [None, ""])
    url = BASE_URL + endpoint + (f"?{qs}" if qs else "")
    timestamp = str(int(time.time() * 1000))
    sign = sign_request(timestamp, "GET", endpoint, qs)
    headers = {
        "ACCESS-KEY": BITGET_API_KEY_03,
        "ACCESS-SIGN": sign,
        "ACCESS-PASSPHRASE": BITGET_API_PASS_03,
        "ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()

# -----------------------------
# Obtener posiciones abiertas
# -----------------------------
def get_open_positions(product_type: str = "USDT-FUTURES") -> List[Dict[str, Any]]:
    endpoint = "/api/v2/mix/position/all-position"
    params = {"productType": product_type}
    response = make_get(endpoint, params)
    return response.get("data", [])

# -----------------------------
# Mostrar resumen de posiciones
# -----------------------------
def summarize_positions(positions: List[Dict[str, Any]]):
    if not positions:
        print("\n🎯 No hay posiciones abiertas.")
        return
    
    print("\n🎰 Posiciones abiertas en Bitget:")
    summary = []
    for p in positions:
        symbol = p.get("symbol")
        side = p.get("holdSide", "?").upper()
        size = float(p.get("total", 0))
        entry = float(p.get("averageOpenPrice", 0))
        mark = float(p.get("marketPrice", 0))
        pnl = float(p.get("unrealizedPL", 0))
        leverage = p.get("leverage", "?")
        liq = p.get("liquidationPrice", "-")
        
        print(f" - {symbol:12} {side:>5} | Size: {size:<8} | Entry: {entry:<8.4f} | Mark: {mark:<8.4f} | "
              f"PnL: {pnl:<8.2f} | Lev: {leverage}x | Liq: {liq}")
        
        summary.append({
            "symbol": symbol,
            "side": side,
            "size": size,
            "entry_price": entry,
            "mark_price": mark,
            "unrealized_pnl": pnl,
            "leverage": leverage,
            "liq_price": liq
        })
    return summary

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    try:
        open_positions = get_open_positions(product_type="USDT-FUTURES")
        summarize_positions(open_positions)
    except Exception as e:
        print(f"\n⚠️ Error al obtener posiciones abiertas: {e}")
