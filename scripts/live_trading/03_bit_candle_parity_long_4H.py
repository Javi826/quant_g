import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from parquet_process.Z_parquet_01_extraction import get_futures_symbols_from_api, _call_history_candles, to_dataframe_from_api

from Z_add_signals_parity import detect_parity_reversal_long
from ZX_utils_live import wait_for_next_candle, load_final_symbols, normalize_live_ohlcv,df_to_arrays_live, PRODUCT_TYPE
from utils.ZZ_connect import connect_bitget_03
from ZX_place_orders import place_order_03
from ZX_connect_live import get_usdt_balance_03, send_request_03, get_open_positions_03

MADRID_TZ = ZoneInfo("Europe/Madrid")

# ----------------------
# CONFIGURATION
# ----------------------
STRATEGY             = "parity_candles_long"
TIMEFRAME_MINOR      = '4H'
ORDER_AMOUNT         = 500

SELL_AFTER_N_CANDLES  = 50

LOOKBACK              = 40
PRICE_TOLERANCE       = 70

TP_PCT                = 5
SL_PCT                = 5

# ----------------------
# FUNCTIONS
# ----------------------

def check_latest_signal(df_minor, symbol):
    
    df_minor  = normalize_live_ohlcv(df_minor)
    arr_minor = df_to_arrays_live(df_minor)

    signals = detect_parity_reversal_long(
        arr_minor,
        tolerance=PRICE_TOLERANCE,
        lookback=LOOKBACK,
        shift_for_execution=True
    )

    last_signal = signals[-1]


    if last_signal != 0:
        last = df_minor.iloc[-1]
        return {
            'symbol': symbol,
            'timestamp': last.name if 'timestamp' not in df_minor.columns else last['timestamp'],
            'close': last['close'],
        }

def has_open_positions_on_exchange(product_type: str = PRODUCT_TYPE) -> bool:   
    try:
        pos_list = get_open_positions_03(product_type=product_type.upper())
        return bool(pos_list)
    except Exception as e:
        print(f"⚠️ Mistake checking positions: {e}")
        return True

# ----------------------
# MAIN LOOP
# ----------------------
exchange       = connect_bitget_03()
all_symbols    = get_futures_symbols_from_api(PRODUCT_TYPE)
final_symbols  = load_final_symbols(all_symbols, strategy=STRATEGY, timeframe=TIMEFRAME_MINOR)
open_positions = []

while True:
    print(f'🧿 === {STRATEGY}_{TIMEFRAME_MINOR} strategy ===🧿')
    wait_for_next_candle(TIMEFRAME_MINOR)

    if not has_open_positions_on_exchange(PRODUCT_TYPE):
        if open_positions:
            print("🔁 All closed positions detected on the exchange. Resetting internal state.")
        open_positions = []

    # -------------------------------
    # SIGNALS & BUYs
    # -------------------------------
    if not open_positions:
        ohlcv_data = {}
        for sym in final_symbols:
            recent_minor    = _call_history_candles(symbol=sym, granularity=TIMEFRAME_MINOR, limit=100)
            df_minor        = to_dataframe_from_api(recent_minor)
            ohlcv_data[sym] = {"minor": df_minor}

        detected_signals = []
        for sym, dfs in ohlcv_data.items():
            signal = check_latest_signal(dfs['minor'], sym)
            if signal:
                detected_signals.append(signal)

        print(f"\n✨ {datetime.now(MADRID_TZ).strftime('%H:%M')} - Signals detected: {len(detected_signals)}")

        for signal in detected_signals:
            sym = signal['symbol']
            usdt_balance = get_usdt_balance_03(exchange)
            now = datetime.now(MADRID_TZ).replace(second=0, microsecond=0) + timedelta(minutes=1)  

            if usdt_balance < ORDER_AMOUNT:
                print(f"⚠️ {now} - USDT balance too low to place order for {sym}")
                continue

            order, tpsl_info = place_order_03(sym, usdt_amount=ORDER_AMOUNT, tp_percent=TP_PCT, sl_percent=SL_PCT)

            if order is not None:
                buy_price     = float(order['data']['price']) if 'price' in order.get('data', {}) else signal['close']
                filled_amount = float(order['data']['size']) if 'size' in order.get('data', {}) else ORDER_AMOUNT / buy_price

                open_positions.append({
                    'symbol': sym,
                    'buy_price': buy_price,
                    'amount': filled_amount,
                    'candles_to_sell': SELL_AFTER_N_CANDLES,
                    'just_bought': True
                })

                usdt_balance_after = get_usdt_balance_03(exchange)
                print(f"💵 {now} - BUY executed: {sym} | Remaining USDT: {usdt_balance_after:.2f}\n")
                time.sleep(2)
            else:
                print(f"⚠️ {now} - Buy order for {sym} was not executed or returned None.")

    else:
        print(f"⛔ {datetime.now(MADRID_TZ).strftime('%H:%M')} - Trades ongoing...")

    # -------------------------------
    # ORDERS MANAGEMENT
    # -------------------------------
    for pos in open_positions[:]:
        if pos.get('just_bought', False):
            pos['just_bought'] = False
            continue

        pos['candles_to_sell'] -= 1

        if pos['candles_to_sell'] <= 0:
            try:
                body = {
                    "symbol": pos['symbol'],
                    "productType": PRODUCT_TYPE
                }
                code, resp = send_request_03("POST", "/api/v2/mix/order/close-positions", body=body)
                now = datetime.now(MADRID_TZ).replace(second=0, microsecond=0)
                if code == 200 and resp.get("code") == "00000":
                    for success in resp['data']['successList']:
                        code_ticker, resp_ticker = send_request_03(
                            "GET",
                            "/api/v2/mix/market/ticker",
                            params={"productType": PRODUCT_TYPE, "symbol": success['symbol']}
                        )
                        sell_price = None
                        if code_ticker == 200 and resp_ticker.get("code") == "00000":
                            sell_price = resp_ticker['data'][0]['lastPr']

                        print(f"💰 {now.strftime('%Y-%m-%d %H:%M:%S')} - FLASH CLOSE: {success['symbol']} | Sold at: {sell_price}")

                else:
                    print(f"⚠️ {now} - Failed Flash Close for {pos['symbol']}: {resp}")
            except Exception as e:
                now = datetime.now(MADRID_TZ).replace(second=0, microsecond=0)
                print(f"⚠️ {now} - Error closing position {pos['symbol']}: {e}")
            finally:
                try:
                    open_positions.remove(pos)
                except ValueError:
                    pass

            time.sleep(1.1)
    
        if not has_open_positions_on_exchange(PRODUCT_TYPE):
            print("🔁 All positions have been closed on the exchange — returning to look for signals now.")
            open_positions = []
            break
