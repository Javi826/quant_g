import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from live_trading.ZX_connect_live import  make_get_TT

BASE_URL        = "https://api.bitget.com"
PRODUCT_TYPE    = "USDT-FUTURES"
INITIAL_CAPITAL = 4040.90  


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

    print("▶️  Balances USDT (Futures):")
    for b in balances:
        print(f" - Subaccount {b['userId']}: {b['usdtEquity']:.4f} USDT")

    print(f"\n💵 Total USDT : {total_usdt:.4f} USDT")

    # 🔹 Cálculo de delta y rentabilidad
    delta = total_usdt - INITIAL_CAPITAL
    profitability_pct = (delta / INITIAL_CAPITAL) * 100

    print("\n📈 Brief:")
    print(f"▶️ INITIAL CAPITAL : {INITIAL_CAPITAL:.2f} USDT")
    print(f"▶️ CURRENT CAPITAL : {total_usdt:.2f} USDT")
    print(f"💵 Delta          : {delta:+.2f} USDT")
    print(f"▶️ Profit %        : {profitability_pct:+.2f}%")
