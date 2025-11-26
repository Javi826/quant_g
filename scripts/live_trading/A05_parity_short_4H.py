import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from parquet_process.Z_parquet_A0_extraction import get_futures_symbols_from_api

from Z_add_signals_parity import parity_short
from ZX_utils_live import wait_for_next_candle, load_final_symbols, normalize_live_ohlcv, df_to_arrays_live, PRODUCT_TYPE
from ZX_utils_sub import load_state,save_state,sync_positions_with_exchange,process_signals_and_buy,manage_open_positions

from utils.ZZ_connect import connect_bitget_05
from ZX_connect_live import get_usdt_balance_05, send_request_05, get_open_positions_05

MADRID_TZ = ZoneInfo("Europe/Madrid")

# ----------------------
# CONFIGURATION
# ----------------------
STRATEGY             = "parity_short"
TIMEFRAME_MINOR      = '4H'
ORDER_AMOUNT         = 80
STATE_FILE           = "robot_state_{STRATEGY}.json"

SELL_AFTER_N_CANDLES = 45

LOOKBACK             = 150
TOLERANCE            = 20

TP_PCT               = 5
SL_PCT               = 10

# ----------------------
# FUNCTIONS
# ----------------------

def check_latest_signal(df_minor, symbol):
    
    df_minor  = normalize_live_ohlcv(df_minor)
    arr_minor = df_to_arrays_live(df_minor)

    signals = parity_short(
        arr_minor,
        lookback=LOOKBACK,
        tolerance=TOLERANCE,
        live_trading=True
    )

    last_signal = signals[-1]

    if last_signal != 0:
        last = df_minor.iloc[-1]
        return {
            'symbol': symbol,
            'timestamp': last.name if 'timestamp' not in df_minor.columns else last['timestamp'],
            'close': last['close'],
        }

# ----------------------
# MAIN LOOP
# ----------------------
exchange       = connect_bitget_05()
all_symbols    = get_futures_symbols_from_api(PRODUCT_TYPE)
final_symbols  = load_final_symbols(all_symbols, strategy=STRATEGY, timeframe=TIMEFRAME_MINOR)
open_positions = []

while True:
    print(f'\n🔷 === 05_{STRATEGY}_{TIMEFRAME_MINOR} strategy === 🔷')
    wait_for_next_candle(TIMEFRAME_MINOR)

    # Si no hay posiciones activas en el exchange → limpiar estado interno
    if not has_open_positions_on_exchange(get_open_positions_05, PRODUCT_TYPE):
        if open_positions:
            print("🔄 All closed positions detected on the exchange. Resetting internal state.")
        open_positions = []

    # -------------------------------
    # SEÑALES Y COMPRAS
    # -------------------------------
    if not open_positions:
        open_positions = process_signals_and_buy(
            final_symbols=final_symbols,
            exchange=exchange,
            open_positions=open_positions,
            order_amount=ORDER_AMOUNT,
            timeframe_minor=TIMEFRAME_MINOR,
            sell_after_n_candles=SELL_AFTER_N_CANDLES,
            tp_pct=TP_PCT,
            sl_pct=SL_PCT,
            direction="short", 
            send_request_fn=send_request_05,
            get_balance_fn=get_usdt_balance_05,
            check_signal_fn=check_latest_signal
        )

    else:
        print(f"🚫 {datetime.now(MADRID_TZ).strftime('%H:%M')} - Trades ongoing...")
        print('\n')

    # -------------------------------
    # ORDERS MANAGEMENT
    # -------------------------------
    manage_open_positions(open_positions, send_request_fn=send_request_05)
    
    if not has_open_positions_on_exchange(get_open_positions_05, PRODUCT_TYPE):
        print("🔄 All positions have been closed on the exchange — returning to look for signals now.")
        open_positions.clear()
