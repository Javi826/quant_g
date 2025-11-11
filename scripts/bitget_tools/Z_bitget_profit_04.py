import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import time
from typing import List, Dict, Any
from datetime import datetime
from live_trading.ZX_connect_live import make_get_04
from utils.ZZ_connect import connect_bitget_04

# -----------------------------
# BITGET CONFIGURATION
# -----------------------------
BASE_URL        = "https://api.bitget.com"
INITIAL_CAPITAL = 1037.69  

# -----------------------------
# Connect with CCXT
# -----------------------------
exchange = connect_bitget_04()

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
        response = make_get_04(endpoint, params)
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

    start_date = "2025-11-11" 
    start_time = date_to_timestamp_ms(start_date)
    end_time   = int(datetime.now().timestamp() * 1000)

    print(f"Downloading closed positions history from {start_date} until today...")
    history = fetch_all_history_positions(product_type="USDT-FUTURES",
                                          start_time=start_time, end_time=end_time)
    print(f"Total positions downloaded: {len(history)}")

    winrate_by_symbol, total_winrate, stats = calculate_winrate_from_history(history)

    total_positions = sum(info["total"] for info in stats.values())
    total_winners   = sum(info["positive"] for info in stats.values())
    print(f"\n📊 Total winrate: {total_winrate:.2f}% ({total_winners}/{total_positions})")

    final_capital     = get_usdt_balance_total(exchange)
    delta_capital     = final_capital - INITIAL_CAPITAL
    profitability_pct = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100 
    print('\n04')
    print(f"\n💵 Initial capital    : {INITIAL_CAPITAL:.2f} USDT")
    print(f"💰 Final capital      : {final_capital:.2f} USDT")
    print(f"📈 Delta (gain/loss)  : {delta_capital:+.2f} USDT")
    print(f"📊 Total profitability: {profitability_pct:.2f}%")
