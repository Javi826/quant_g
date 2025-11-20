import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from live_trading.ZX_connect_live import send_request_05

BASE_URL     = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"  

def close_all_positions():
    code, resp = send_request_05(
        "GET",
        "/api/v2/mix/position/all-position",
        params={"productType": PRODUCT_TYPE, "marginCoin": "USDT"}
    )

    if code != 200 or resp.get("code") != "00000":
        print("⚠️ Error fetching positions:", resp)
        return

    positions = [p for p in resp['data'] if float(p['total']) > 0]

    if not positions:
        print("ℹ️ No open positions to close.")
        return

    for pos in positions:
        body = {
            "symbol": pos['symbol'],
            "productType": PRODUCT_TYPE
        }
        close_code, close_resp = send_request_05("POST", "/api/v2/mix/order/close-positions", body=body)
        if close_code == 200 and close_resp.get("code") == "00000":
            print(f"💰 FLASH CLOSE executed: {pos['symbol']}")
        else:
            print(f"⚠️ Failed to close {pos['symbol']}: {close_resp}")
        time.sleep(1.1)  # evitar limitación: 1 request/seg

if __name__ == "__main__":
    close_all_positions()
