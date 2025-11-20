import sys
import os


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from live_trading.ZX_connect_live import PRODUCT_TYPE
from live_trading.ZX_connect_live import send_request_TT,send_request_01,send_request_02,send_request_03,send_request_04,send_request_05
from parquet_process.Z_parquet_01_extraction import get_futures_symbols_from_api

# ===========================
# CONFIG
# ===========================
TARGET_MODE = "hedge_mode"  

# ===========================
# FUNCTIONS
# ===========================
def set_position_mode(product_type: str = PRODUCT_TYPE, mode: str = TARGET_MODE):
    """
    Changes the position mode for all contracts of a product to 'hedge_mode' or 'one_way_mode'.
    """
    body = {
        "productType": product_type,
        "posMode": mode
    }

    code, resp = send_request_05("POST", "/api/v2/mix/account/set-position-mode", body=body)

    if code == 200 and resp.get("code") == "00000":
        print(f"✅ {product_type}: Position mode changed to {mode}.")
        return True
    else:
        print(f"⚠️ {product_type}: Error changing position mode → {resp}")
        return False

# ===========================
# MAIN
# ===========================
def main():
    print("\n📂 Fetching all available symbols...")
    all_symbols = get_futures_symbols_from_api(PRODUCT_TYPE)
    if not all_symbols:
        print("⚠️ No symbols found to process.")
        return

    # Bitget applies the mode to all contracts of a product, so only one POST request is needed
    print(f"\n🚀 Changing position mode to '{TARGET_MODE}' for all {PRODUCT_TYPE} contracts...\n")
    set_position_mode(PRODUCT_TYPE)

    print("\n✅ Process completed.\n")

if __name__ == "__main__":
    main()
