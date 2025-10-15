import os
import sys
import requests
import hmac
import hashlib
import base64
import time
from typing import List, Dict, Any
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from utils.ZZ_connect import connect_bitget_03,BITGET_API_KEY_03,BITGET_API_SECRET_03,BITGET_API_PASS_03

# -----------------------------
# BITGET CONFIGURATION
# -----------------------------
BASE_URL        = "https://api.bitget.com"
INITIAL_CAPITAL = 511.0  

# -----------------------------
# Connect with CCXT
# -----------------------------
exchange = connect_bitget_03()

def get_usdt_balance_total(exchange):
    """Returns the total USDT balance including used in open positions"""
    balance = exchange.fetch_balance()
    return balance['total']['USDT']

# -----------------------------
# Convert date to timestamp in ms
# -----------------------------
def date_to_timestamp_ms(date_str: str) -> int:
    """Converts 'YYYY-MM-DD' to timestamp in milliseconds"""
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)

# -----------------------------
# Helpers for signing and HTTP request (history-position)
# -----------------------------
def sign_request(timestamp: str, method: str, request_path: str, query_string: str = "", body: str = "") -> str:
    to_sign = timestamp + method.upper() + request_path
    if query_string:
        to_sign += "?" + query_string
    to_sign += body
    signature = hmac.new(BITGET_API_SECRET_03.encode("utf-8"), to_sign.encode("utf-8"), hashlib.sha256).digest()
    return base64.b64encode(signature).decode()

def make_get(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    qs = "&".join(f"{k}={v}" for k, v in params.items() if v is not None and v != "")
    url = BASE_URL + endpoint
    if qs:
        url += "?" + qs
    timestamp = str(int(time.time() * 1000))
    sign = sign_request(timestamp, "GET", endpoint, qs, "")
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
# Fetch complete history with pagination
# -----------------------------
def fetch_all_history_positions(product_type: str = "USDT-FUTURES", symbol: str = None,
                                start_time: int = None, end_time: int = None) -> List[Dict[str, Any]]:
    endpoint = "/api/v2/mix/position/history-position"
    limit = 100
    all_items: List[Dict[str, Any]] = []
    id_less_than = None

    while True:
        params = {
            "productType": product_type if symbol is None else None,
            "symbol": symbol,
            "limit": limit,
            "idLessThan": id_less_than,
            "startTime": start_time,
            "endTime": end_time
        }
        response = make_get(endpoint, params)
        data = response.get("data", {})
        items = data.get("list") or []
        end_id = data.get("endId")

        if not items:
            break

        all_items.extend(items)

        if len(items) < limit or not end_id:
            break

        id_less_than = end_id
        time.sleep(0.05)

    return all_items

# -----------------------------
# Calculate winrate of closed positions
# -----------------------------
def calculate_winrate_from_history(history: List[Dict[str, Any]]):
    stats: Dict[str, Dict[str, int]] = {}
    total = 0
    winners = 0

    for pos in history:
        symbol = pos.get("symbol") or "UNKNOWN"
        try:
            net_profit = float(pos.get("netProfit", 0) or 0)
        except (ValueError, TypeError):
            net_profit = 0.0

        if symbol not in stats:
            stats[symbol] = {"positive": 0, "total": 0}

        stats[symbol]["total"] += 1
        total += 1

        if net_profit > 0:
            stats[symbol]["positive"] += 1
            winners += 1

    winrate_by_symbol = {sym: (info["positive"]/info["total"])*100 if info["total"]>0 else 0.0 for sym, info in stats.items()}
    total_winrate = (winners / total) * 100 if total > 0 else 0.0
    return winrate_by_symbol, total_winrate, stats

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    # Start date for filtering
    start_date = "2025-10-12"  # change as needed
    start_time = date_to_timestamp_ms(start_date)

    # End date always today
    end_time = int(datetime.now().timestamp() * 1000)

    print(f"Downloading closed positions history from {start_date} until today...")
    history = fetch_all_history_positions(product_type="USDT-FUTURES",
                                          start_time=start_time, end_time=end_time)
    print(f"Total positions downloaded: {len(history)}")

    # Winrate
    winrate_by_symbol, total_winrate, stats = calculate_winrate_from_history(history)
    #print("\n📊 Winrate by symbol:")
    #for s, pct in sorted(winrate_by_symbol.items(), key=lambda x: (-x[1], x[0])):
    #    info = stats.get(s, {})
        #print(f" - {s}: {pct:.2f}% ({info.get('positive',0)}/{info.get('total',0)})")

    # Total winrate with number of positions
    total_positions = sum(info["total"] for info in stats.values())
    total_winners   = sum(info["positive"] for info in stats.values())
    print(f"\n📊 Total winrate: {total_winrate:.2f}% ({total_winners}/{total_positions})")

    # Profitability using real USDT balance
    final_capital     = get_usdt_balance_total(exchange)
    profitability_pct = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100 if INITIAL_CAPITAL > 0 else 0

    print(f"\n💵 Initial capital: {INITIAL_CAPITAL:.2f} USDT")
    print(f"💰 Final capital (real in Bitget): {final_capital:.2f} USDT")
    print(f"📊 Total profitability: {profitability_pct:.2f}%")
