import os
import sys
import time
from typing import Dict, Any, List
from datetime import datetime, timedelta
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from live_trading.ZX_connect_live import get_open_positions_01, make_get_01
from utils.ZZ_connect import connect_bitget_01

# -----------------------------
# Connect with CCXT
# -----------------------------
exchange = connect_bitget_01()

def ms_to_date(ms):
    if not ms:
        return "Unknown"
    return datetime.fromtimestamp(int(ms)/1000).strftime("%Y-%m-%d")

def get_usdt_balance_total(exchange):
    """Returns the total USDT balance including used in open positions"""
    balance = exchange.fetch_balance()
    return balance['total']['USDT']

# -----------------------------
# Convert date to timestamp in ms
# -----------------------------
def date_to_timestamp_ms(date_str: str) -> int:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return int(dt.timestamp() * 1000)

# -----------------------------
# Open positions
# -----------------------------

def estimate_close_date(open_time_ms, candles_to_close, timeframe):

    open_dt = datetime.fromtimestamp(int(open_time_ms) / 1000)

    if timeframe.endswith("H"):
        delta = timedelta(hours=int(timeframe[:-1]))
    elif timeframe.endswith("D"):
        delta = timedelta(days=int(timeframe[:-1]))
    elif timeframe.endswith("M"):
        delta = timedelta(minutes=int(timeframe[:-1]))
    else:
        raise ValueError(f"Timeframe no reconocido: {timeframe}")

    estimated_close = open_dt + candles_to_close * delta
    return estimated_close.strftime("%Y-%m-%d")

def summarize_positions(positions: List[Dict[str, Any]], candles_to_close: int, timeframe: str):

    if not positions:
        print("\nNo positions.")
        return []

    print("\nOpen positions:")
    summary = []
    total_pnl = 0.0

    for p in positions:
        symbol = p.get("symbol")
        side = p.get("holdSide", "?").upper()
        margin_size = float(p.get("marginSize", 0))
        total_size = float(p.get("total", 0))
        entry = float(p.get("openPriceAvg", 0))
        pnl = float(p.get("unrealizedPL", 0))
        leverage = p.get("leverage", "?")
        open_time_ms = p.get("cTime")
        open_time = ms_to_date(open_time_ms)
        close_time_estimated = estimate_close_date(open_time_ms, candles_to_close, timeframe)

        total_pnl += pnl

        print(f" - {symbol:12} | {side:>5} | Open: {open_time} | F.Close: {close_time_estimated} | "
              f"Margin: {margin_size:<8.1f} | Entry: {entry:<8.2f} PnL: {pnl:<8.2f} | Lev: {leverage}x")

        summary.append({
            "symbol": symbol,
            "side": side,
            "margin_size": round(margin_size, 1),
            "total_size": total_size,
            "entry_price": entry,
            "unrealized_pnl": pnl,
            "leverage": leverage,
            "open_time": open_time,
            "estimated_close": close_time_estimated
        })

    print("\n------------------------------------------")
    print(f"Positions : {len(positions)}")
    print(f"PnL       : {total_pnl:.2f}")
    print("------------------------------------------")

    return summary

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

