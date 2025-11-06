import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from parquet_process.Z_parquet_01_extraction import get_futures_symbols_from_api, _call_history_candles, to_dataframe_from_api
from ZX_utils_live import normalize_live_ohlcv, load_final_symbols, PRODUCT_TYPE
from Z_add_signals_pt import detect_double_top

MADRID_TZ = ZoneInfo("Europe/Madrid")

# CONFIG
STRATEGY        = "double_top"
TIMEFRAME_MINOR = "4H"
LOOKBACK_MINOR  = 5
PRICE_TOLERANCE = 20
UPTREND_TH      = 5
CANDLES_LIMIT   = 50

def check_latest_signal(df_minor, symbol):
    df_minor = normalize_live_ohlcv(df_minor)
    arr_minor = {
        'high': df_minor['high'].values,
        'low': df_minor['low'].values,
        'close': df_minor['close'].values
    }
    signals = detect_double_top(
        arr_minor,
        lookback_minor=LOOKBACK_MINOR,
        price_tolerance=PRICE_TOLERANCE,
        uptrend_th=UPTREND_TH,
        backtest=False
    )
    last_signal = signals[-1]
    return last_signal if last_signal != 0 else 0

def main():
    all_symbols   = get_futures_symbols_from_api(PRODUCT_TYPE)
    final_symbols = load_final_symbols(all_symbols, strategy=STRATEGY, timeframe=TIMEFRAME_MINOR)

    for sym in final_symbols:
        recent_minor = _call_history_candles(symbol=sym, granularity=TIMEFRAME_MINOR, limit=CANDLES_LIMIT)
        if not recent_minor:
            print(f"{sym} ⚠️ no se pudieron obtener velas")
            continue

        df_minor = to_dataframe_from_api(recent_minor)
        df_minor = normalize_live_ohlcv(df_minor)

        # Señal completa detect_double_top (último valor)
        arr_minor = {
            'high': df_minor['high'].values,
            'low': df_minor['low'].values,
            'close': df_minor['close'].values
        }
        signals_direct = detect_double_top(
            arr_minor,
            lookback_minor=LOOKBACK_MINOR,
            price_tolerance=PRICE_TOLERANCE,
            uptrend_th=UPTREND_TH,
            backtest=False
        )
        last_signal_full = signals_direct[-1]

        # Última señal live
        signal_live = check_latest_signal(df_minor, sym)

        # Mostrar solo los últimos valores
        now = datetime.now(MADRID_TZ).strftime("%Y-%m-%d %H:%M:%S")
        print(f"{now} | {sym} - Última señal detect_double_top: {last_signal_full}, Última señal live: {signal_live}")

        # Comparación
        if last_signal_full == signal_live:
            print(f"{sym} ✅ señales coinciden")
        else:
            print(f"{sym} ❌ señales NO coinciden")

if __name__ == "__main__":
    main()
