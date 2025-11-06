import pandas as pd
from parquet_process.Z_parquet_01_extraction import _call_history_candles, to_dataframe_from_api
from ZX_utils_live import normalize_live_ohlcv, PRODUCT_TYPE
from Z_add_signals_pr import detect_parity_reversal_long

# CONFIG
TIMEFRAME = "4H"
LOOKBACK = 50
PRICE_TOLERANCE = 10
TEST_SYMBOLS = ["BTCUSDT_UMCBL", "ETHUSDT_UMCBL"]  # símbolos de prueba

def check_signals(symbols):
    for sym in symbols:
        # obtener datos
        recent_minor = _call_history_candles(symbol=sym, granularity=TIMEFRAME, limit=100)
        df_minor = to_dataframe_from_api(recent_minor)
        df_minor = normalize_live_ohlcv(df_minor)
        
        arr_minor = {
            'open': df_minor['open'].values,
            'high': df_minor['high'].values,
            'low': df_minor['low'].values,
            'close': df_minor['close'].values
        }
        
        signals = detect_parity_reversal_long(arr_minor, tolerance=PRICE_TOLERANCE, lookback=LOOKBACK, backtest=True)
        
        print(f"\n=== {sym} ===")
        print("Últimas 10 señales:", signals[-10:])
        print("Últimos 5 cierres:", df_minor['close'].tail(5).tolist())

if __name__ == "__main__":
    check_signals(TEST_SYMBOLS)
