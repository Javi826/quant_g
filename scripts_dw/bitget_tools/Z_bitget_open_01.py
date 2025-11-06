import os
import sys
from typing import Dict, Any, List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from live_trading.ZX_connect_live import get_open_positions_01


BASE_URL = "https://api.bitget.com"

# -----------------------------
# Mostrar resumen de posiciones
# -----------------------------
def summarize_positions(positions: List[Dict[str, Any]]):
    if not positions:
        print("\nNo positions.")
        return
    
    print("\nOpen positions:")
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
        open_positions = get_open_positions_01(product_type="USDT-FUTURES")
        print('\n01')
        summarize_positions(open_positions)
    except Exception as e:
        print(f"\n⚠️ Fallen: {e}")
