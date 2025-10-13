import time
from datetime import datetime
from zoneinfo import ZoneInfo
from utils.ZZ_connect_03 import connect_bitget
from parquet_process.ZZ_parquet_extraction import get_futures_symbols_from_api,_call_history_candles,to_dataframe_from_api
from Z_add_signals_03 import add_indicators, explosive_signal
from utils.ZX_utils import wait_for_next_candle, get_usdt_balance, place_order, load_final_symbols, normalize_live_ohlcv,send_request, PRODUCT_TYPE

MADRID_TZ = ZoneInfo("Europe/Madrid")

# ----------------------
# CONFIGURATION
# ----------------------
TIMEFRAME            = '4H'
ORDER_AMOUNT         = 100

SELL_AFTER_N_CANDLES = 25
ENTROPIA_MAX         = 0.6
ACCEL_SPAN           = 10

TP_PCT               = 0
SL_PCT               = 0

# ----------------------
# FUNCTIONS
# ----------------------
def check_latest_signal(df, symbol):

    df              = normalize_live_ohlcv(df)
    close_prices    = df['close'].values
    entropia, accel = add_indicators(close_prices, m_accel=ACCEL_SPAN)
    signals         = explosive_signal(entropia, accel, entropia_max=ENTROPIA_MAX, live=True)
    last_signal =    signals[-1]

    if last_signal:
        last = df.iloc[-1]  
        return {
            'symbol': symbol,
            'timestamp': last['timestamp'],
            'close': last['close'],
        }

# ----------------------
# MAIN LOOP
# ----------------------
exchange       = connect_bitget()
all_symbols    = get_futures_symbols_from_api(PRODUCT_TYPE)
final_symbols  = load_final_symbols(all_symbols,strategy="entropy",timeframe=TIMEFRAME)
open_positions = []

while True:
    print('🧿 === Entropy strategy ===🧿')
    wait_for_next_candle(TIMEFRAME)

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
            usdt_balance = get_usdt_balance(exchange)
            now = datetime.now(MADRID_TZ).replace(second=0, microsecond=0)

            if usdt_balance < ORDER_AMOUNT:
                print(f"⚠️ {now} - USDT balance too low to place order for {sym}")
                continue

            order, tpsl = place_order(sym, usdt_amount=ORDER_AMOUNT, tp_percent=TP_PCT, sl_percent=SL_PCT)

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

                usdt_balance_after = get_usdt_balance(exchange)
                print(f"💵 {now} - BUY executed: {sym} | Remaining USDT: {usdt_balance_after:.2f}\n")
                time.sleep(2)
            else:
                print(f"⚠️ {now} - Buy order for {sym} was not executed or returned None.")

    else:
        print(f"⛔ {datetime.now(MADRID_TZ).strftime('%H:%M')} - Trades ongoing...")

    # -------------------------------
    # ORDERS MANAGEMNTE
    # -------------------------------
    for pos in open_positions[:]:
        if pos.get('just_bought', False):
            pos['just_bought'] = False
            continue
    
        pos['candles_to_sell'] -= 1
    
        if pos['candles_to_sell'] <= 0:
            try:
                # Cerrar la posición con Flash Close vía API
                body = {
                    "symbol": pos['symbol'],
                    "productType": PRODUCT_TYPE
                }
                code, resp = send_request("POST", "/api/v2/mix/order/close-positions", body=body)
                now = datetime.now(MADRID_TZ).replace(second=0, microsecond=0)
                if code == 200 and resp.get("code") == "00000":
                    for success in resp['data']['successList']:
                        # Obtener el último precio del ticker para mostrar el precio de venta real
                        code_ticker, resp_ticker = send_request(
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
                open_positions.remove(pos)
            time.sleep(1.1)

