import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
from live_trading.ZX_connect_live import send_request_02, PRODUCT_TYPE
from parquet_process.Z_parquet_01_extraction import get_futures_symbols_from_api  # or your symbol-fetching function


# =====================================
# CONFIGURATION
# =====================================
LEVERAGE_TARGET = 10
MARGIN_COIN     = "USDT"
MARGIN_MODE     = "isolated" 

# =====================================
# FUNCTIONS
# =====================================

def set_leverage(symbol: str, leverage: int = LEVERAGE_TARGET,
                 product_type: str = PRODUCT_TYPE,
                 margin_coin: str = MARGIN_COIN,
                 margin_mode: str = MARGIN_MODE):
    """
    Sets the leverage for the specified symbol on Bitget Futures.
    """
    body = {
        "symbol": symbol,
        "productType": product_type,
        "marginCoin": margin_coin,
    }

    if margin_mode.lower() == "isolated":
        body["longLeverage"] = str(leverage)
        body["shortLeverage"] = str(leverage)
    else:
        body["leverage"] = str(leverage)

    code, resp = send_request_02("POST", "/api/v2/mix/account/set-leverage", body=body)

    if code == 200 and resp.get("code") == "00000":
        print(f"✅ {symbol}: Leverage set to {leverage}x successfully.")
        return True
    else:
        print(f"⚠️ {symbol}: Error setting leverage → {resp}")
        return False


def get_leverage(symbol: str, product_type: str = PRODUCT_TYPE, margin_coin: str = MARGIN_COIN):
    """
    Retrieves the current leverage for the symbol.
    """
    params = {"symbol": symbol, "productType": product_type, "marginCoin": margin_coin}
    code, resp = send_request_02("GET", "/api/v2/mix/account/account", params=params)

    if code == 200 and resp.get("code") == "00000":
        data = resp.get("data", {})
        long_lev = data.get("longLeverage")
        short_lev = data.get("shortLeverage")
        print(f"🔍 {symbol}: Long={long_lev}x | Short={short_lev}x")
        return data
    else:
        print(f"⚠️ {symbol}: Error fetching leverage → {resp}")
        return None


# =====================================
# MAIN PROCESS
# =====================================

def main():
    print("\n📂 Fetching all available symbols...")
    all_symbols = get_futures_symbols_from_api(PRODUCT_TYPE)  # fetch all symbols from the API
    if not all_symbols:
        print("⚠️ No symbols found to process.")
        return

    print(f"\n🚀 Starting leverage update to {LEVERAGE_TARGET}x on {len(all_symbols)} symbols...\n")

    results = {}
    for sym in all_symbols:
        ok = set_leverage(sym)
        time.sleep(0.25)
        if ok:
            data = get_leverage(sym)
            if data:
                results[sym] = {
                    "long": data.get("longLeverage"),
                    "short": data.get("shortLeverage")
                }
        time.sleep(0.25)

    print("\n📊 FINAL SUMMARY:\n")
    for s, vals in results.items():
        print(f"{s}: Long={vals['long']}x | Short={vals['short']}x")

    print("\n✅ Process completed.\n")


if __name__ == "__main__":
    main()
