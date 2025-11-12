import os
import sys
from typing import Dict, Any, List
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from live_trading.ZX_connect_live import get_open_positions_05


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
    total_pnl = 0.0  # acumulador de PnL
    for p in positions:
        symbol = p.get("symbol")
        side = p.get("holdSide", "?").upper()
        margin_size = float(p.get("marginSize", 0))
        total_size = float(p.get("total", 0))
        entry = float(p.get("openPriceAvg", 0))
        pnl = float(p.get("unrealizedPL", 0))
        leverage = p.get("leverage", "?")

        total_pnl += pnl  # acumula el PnL total
        
        print(f" - {symbol:12} {side:>5} | Margin: {margin_size:<8.1f} "
              f"Size: {total_size:<8.1f} | Entry: {entry:<8.2f} "
              f"PnL: {pnl:<8.2f} | Lev: {leverage}x")

        summary.append({
            "symbol": symbol,
            "side": side,
            "margin_size": round(margin_size, 1),
            "total_size": total_size,
            "entry_price": entry,
            "unrealized_pnl": pnl,
            "leverage": leverage
        })
    
    # --- resumen final ---
    print("\n------------------------------------------")
    print(f"Positions: {len(positions)}")
    print(f"PnL      : {total_pnl:.2f}")
    print("------------------------------------------")

    return summary

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    try:
        open_positions = get_open_positions_05(product_type="USDT-FUTURES")
        print('\n05')
        summarize_positions(open_positions)
    except Exception as e:
        print(f"\n⚠️ Fallen: {e}")
