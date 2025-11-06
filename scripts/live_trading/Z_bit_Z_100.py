import os
import sys
import time
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from parquet_process.Z_parquet_01_extraction import get_futures_symbols_from_api, _call_history_candles, to_dataframe_from_api

from parquet_process.Z_parquet_01_extraction import get_futures_symbols_from_api, _call_history_candles, to_dataframe_from_api
from ZX_utils_live import normalize_live_ohlcv, load_final_symbols, PRODUCT_TYPE
from Z_add_signals_parity import detect_parity_reversal_long
from Z_add_signals_parity import detect_parity_reversal_short
from datetime import datetime
from zoneinfo import ZoneInfo

MADRID_TZ = ZoneInfo("Europe/Madrid")

# ----------------------------
# CONFIG
# ----------------------------
STRATEGY        = "parity_candles_long"
TIMEFRAME_MINOR = "4H"
LOOKBACK        = 40
PRICE_TOLERANCE = 70
TEST_LIMIT      = 100  # número de velas a descargar

# ----------------------------
# FUNCIONES
# ----------------------------
def check_latest_signal(df_minor, symbol):
    df_minor  = normalize_live_ohlcv(df_minor)
    arr_minor = {
        'open': df_minor['open'].values,
        'high': df_minor['high'].values,
        'low': df_minor['low'].values,
        'close': df_minor['close'].values
    }
    
    signals = detect_parity_reversal_long(
        arr_minor,
        tolerance=PRICE_TOLERANCE,
        lookback=LOOKBACK,
        shift_for_execution=True
    )
    
    # Tomar solo la última señal
    last_signal = signals[-1]
    if last_signal != 0:
        last = df_minor.iloc[-1]
        return {
            'symbol': symbol,
            'timestamp': last['timestamp'],
            'close': last['close']
        }

# ----------------------------
# EJECUCIÓN
# ----------------------------
if __name__ == "__main__":
    # Obtener símbolos filtrados para la estrategia
    all_symbols = get_futures_symbols_from_api(PRODUCT_TYPE)
    final_symbols = load_final_symbols(all_symbols, strategy=STRATEGY, timeframe=TIMEFRAME_MINOR)
    print(f"🔹 Símbolos a revisar: {len(final_symbols)}")

    # Revisar cada símbolo y mostrar última señal
    for sym in final_symbols:
        candles = _call_history_candles(symbol=sym, granularity=TIMEFRAME_MINOR, limit=TEST_LIMIT)
        df = to_dataframe_from_api(candles)
        signal_info = check_latest_signal(df, sym)
        if signal_info:
            print(f"dd Última señal en {sym}: {signal_info}")
        else:
            print(f"No hay señales recientes en {sym}")
