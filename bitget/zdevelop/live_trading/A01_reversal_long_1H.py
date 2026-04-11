import os
import sys
from datetime import datetime
from zoneinfo import ZoneInfo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from parquet_process.Z_parquet_A0_extraction import get_futures_symbols_from_api
from Z_add_signals_reversal import reversal_long
from ZX_utils_live import load_final_symbols, normalize_live_ohlcv, df_to_arrays_live, PRODUCT_TYPE
from ZX_utils_sub import load_state,save_state,sync_positions_with_exchange,process_signals_and_buy,manage_open_positions,wait_for_next_candle
from utils.ZZ_connect import connect_bitget_01
from ZX_connect_live import get_usdt_balance_01, send_request_01, get_open_positions_01

MADRID_TZ = ZoneInfo("Europe/Madrid")
ROBOTS_JSON_DIR = os.path.join(os.path.dirname(__file__), "sub_states")
os.makedirs(ROBOTS_JSON_DIR, exist_ok=True)
# ----------------------
# CONFIGURATION
# ----------------------
STRATEGY              = "reversal_long_1H"
TIMEFRAME_MINOR       = '1H'
ORDER_AMOUNT          = 40
STATE_FILE            = os.path.join(ROBOTS_JSON_DIR, f"robot_state_{STRATEGY}.json")

SELL_AFTER_N_CANDLES  = 50  
LEFT_LOOKBACK         = 7 
TOLERANCE             = 40
MA_PERIOD             = 25

TP_PCT                = 2
SL_PCT                = 10
# ----------------------
# FUNCTIONS
# ----------------------
def check_latest_signal(df_minor, symbol):
    df_minor  = normalize_live_ohlcv(df_minor)
    arr_minor = df_to_arrays_live(df_minor)
    
    signals = reversal_long(
        arr_minor,
        lookback=LEFT_LOOKBACK,
        tolerance=TOLERANCE,
        ma_period=MA_PERIOD,
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
exchange       = connect_bitget_01()
all_symbols    = get_futures_symbols_from_api(PRODUCT_TYPE)
final_symbols  = load_final_symbols(all_symbols, strategy=STRATEGY, timeframe=TIMEFRAME_MINOR)

# 🔄 CARGAR ESTADO AL INICIAR
open_positions = load_state(STATE_FILE)
if open_positions:
    print(f"🔄 roBOT restarted with {len(open_positions)} active positions:")
    for pos in open_positions:
        print(f"   - {pos['symbol']}: {pos['candles_to_sell']} candles remaining")

try:
    while True:
        print(f'\n🔷 === 01_{STRATEGY}_{TIMEFRAME_MINOR} strategy === 🔷')
        wait_for_next_candle(TIMEFRAME_MINOR)
        # 🔍 SINCRONIZAR con el exchange (detecta cierres por TP/SL)
        sync_positions_with_exchange(open_positions, get_open_positions_01, PRODUCT_TYPE)
        
        # 💾 Guardar estado después de sincronizar
        save_state(open_positions, STATE_FILE)
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
                direction="long",
                send_request_fn=send_request_01,
                get_balance_fn=get_usdt_balance_01,
                check_signal_fn=check_latest_signal
            )
            
            # 💾 GUARDAR ESTADO después de comprar
            if open_positions:
                save_state(open_positions, STATE_FILE)
        else:
            print(f"🚫 {datetime.now(MADRID_TZ).strftime('%H:%M')} - Trades ongoing...")
            print(f"🔹 Currently open positions: {len(open_positions)}")
        # -------------------------------
        # ORDERS MANAGEMENT
        # -------------------------------
        manage_open_positions(open_positions, send_request_fn=send_request_01, product_type=PRODUCT_TYPE)
        save_state(open_positions, STATE_FILE)
        print("🔂 === Signal cycle completed ===")

except KeyboardInterrupt:
    print("\n🔚 Interrupted by user.")
    save_state(open_positions, STATE_FILE)
    print("⛔ roBOT Stopped")