import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.ZZ_connect import connect_bitget_01
from parquet_process.Z_parquet_extraction import get_futures_symbols_from_api, _call_history_candles, to_dataframe_from_api
from Z_add_signals_01 import explosive_signal_01
from ZX_utils_live import wait_for_next_candle, place_order, load_final_symbols, normalize_live_ohlcv, PRODUCT_TYPE
from ZX_connect_live import get_usdt_balance_01,send_request_01,get_open_positions_01

MADRID_TZ = ZoneInfo("Europe/Madrid")

# ----------------------
# CONFIGURATION
# ----------------------
STRATEGY             = "entropy"
TIMEFRAME            = '4H'
ORDER_AMOUNT         = 100

SELL_AFTER_N_CANDLES = 100
ENTROPIA_MAX         = 0.4
ACCEL_SPAN           = 20

TP_PCT               = 100
SL_PCT               = 20

# ----------------------
# FUNCTIONS
# ----------------------

def check_latest_signal(df, symbol):
    df           = normalize_live_ohlcv(df)
    close_prices = df['close'].values
    signals      = explosive_signal_01(close_prices, m_accel=ACCEL_SPAN, entropia_max=ENTROPIA_MAX, live=True)

    last_signal = signals[-1]

    if last_signal:
        last = df.iloc[-1]
        return {
            'symbol': symbol,
            'timestamp': last['timestamp'],
            'close': last['close'],
        }

def has_open_positions_on_exchange(product_type: str = PRODUCT_TYPE) -> bool:
    
    try:
        pos_list = get_open_positions_01(product_type=product_type.upper())
        return bool(pos_list)
    except Exception as e:
        print(f"⚠️ Error comprobando posiciones en exchange: {e}")
        return True

# ----------------------
# MAIN LOOP (PARTE MODIFICADA)
# ----------------------
exchange       = connect_bitget_01()
all_symbols    = get_futures_symbols_from_api(PRODUCT_TYPE)
final_symbols  = load_final_symbols(all_symbols,strategy=STRATEGY,timeframe=TIMEFRAME)
open_positions = []

while True:
    print('🧿 === Entropy strategy ===🧿')
    wait_for_next_candle(TIMEFRAME)

    # --- sincronizar con el exchange: si no hay posiciones en el exchange, vaciamos open_positions ---
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
            recent_data = _call_history_candles(symbol=sym, granularity=TIMEFRAME, limit=50)
            if recent_data:
                df = to_dataframe_from_api(recent_data)
                ohlcv_data[sym] = df

        detected_signals = []
        for sym, df in ohlcv_data.items():
            signal = check_latest_signal(df, sym)
            if signal:
                detected_signals.append(signal)

        print(f"🔔 {datetime.now(MADRID_TZ).strftime('%H:%M')} - Signals detected: {len(detected_signals)}")

        for signal in detected_signals:
            sym = signal['symbol']
            usdt_balance = get_usdt_balance_01(exchange)
            now = datetime.now(MADRID_TZ).replace(second=0, microsecond=0)

            if usdt_balance < ORDER_AMOUNT:
                print(f"⚠️ {now} - USDT balance too low to place order for {sym}")
                continue

            order,tpsl_info = place_order(sym, usdt_amount=ORDER_AMOUNT, tp_percent=TP_PCT, sl_percent=SL_PCT)

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

                usdt_balance_after = get_usdt_balance_01(exchange)
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
                code, resp = send_request_01("POST", "/api/v2/mix/order/close-positions", body=body)
                now = datetime.now(MADRID_TZ).replace(second=0, microsecond=0)
                if code == 200 and resp.get("code") == "00000":
                    for success in resp['data']['successList']:
                        
                        code_ticker, resp_ticker = send_request_01(
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
