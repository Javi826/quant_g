import os
import sys
import time
from typing import Dict, Any, List
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from live_trading.ZX_connect_live import get_open_positions_02, make_get_02
from utils.ZZ_connect import connect_bitget_02
from ZX_utils_tools import get_usdt_balance_total,date_to_timestamp_ms,summarize_positions,calculate_winrate_from_history

# -----------------------------
# CONFIG
# -----------------------------
BASE_URL              = "https://api.bitget.com"
INITIAL_CAPITAL       = 750.00  
STRATEGY              = "reversal_long"
TIMEFRAME_MINOR       = '4H'
SELL_AFTER_N_CANDLES  = 50

exchange = connect_bitget_02()

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
        response = make_get_02(endpoint, params)
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
# MAIN
# -----------------------------
if __name__ == "__main__":
    try:       
        # --- OPEN POSITIONS ---
        open_positions = get_open_positions_02(product_type="USDT-FUTURES")
        print(f"\n02_{STRATEGY}")
        summarize_positions(open_positions, SELL_AFTER_N_CANDLES, TIMEFRAME_MINOR)
    except Exception as e:
        print(f"\n⚠️ Fallen retrieving open positions: {e}")

    try:
        # --- HISTORIAL Y WINRATE ---  
        start_date = "2025-11-15"
        start_time = date_to_timestamp_ms(start_date)
        end_time   = int(datetime.now().timestamp() * 1000)

        history = fetch_all_history_positions(product_type="USDT-FUTURES",
                                              start_time=start_time, end_time=end_time)

        winrate_by_symbol, total_winrate, stats = calculate_winrate_from_history(history)

        total_positions = sum(info["total"] for info in stats.values())
        total_winners   = sum(info["positive"] for info in stats.values())
        print(f"\nTotal winrate    : {total_winrate:.2f}% ({total_winners}/{total_positions})")

        final_capital     = get_usdt_balance_total(exchange)
        delta_capital     = final_capital - INITIAL_CAPITAL
        profitability_pct = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100 
        print(f"Initial capital  : {INITIAL_CAPITAL:.2f} USDT")
        print(f"Final capital    : {final_capital:.2f} USDT")
        print(f"Delta (gain/loss): {delta_capital:+.2f} USDT")
        print(f"Profitability    : {profitability_pct:.2f}%")

    except Exception as e:
        print(f"\n⚠️ Fallen retrieving history or calculating stats: {e}")
