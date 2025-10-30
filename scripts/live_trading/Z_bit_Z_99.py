import os
import sys
import time
import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# --- rutas y dependencias locales ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.ZZ_connect import connect_bitget_01
from parquet_process.Z_parquet_01_extraction import (
    _call_history_candles,
    to_dataframe_from_api,
)
from ZX_utils_live import wait_for_next_candle

# --- CONFIGURACIÓN ---
MADRID_TZ = ZoneInfo("Europe/Madrid")

SYMBOL             = "BTCUSDT"   # 🔹 símbolo a inspeccionar
TIMEFRAME_MAJOR    = "1Dutc"
TIMEFRAME_MINOR    = "1H"
OUTPUT_DIR         = "debug_excels"
REFRESH_EACH_CANDLE = True       # True = espera nueva vela 1H, False = solo una descarga

# --- FUNCIÓN AUXILIAR ---
def save_candles_to_excel_single(symbol, df_minor, df_major, output_dir="debug_excels", prefix="candles_debug"):
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Quitar zona horaria antes de guardar (para evitar error de Excel)
    for df in (df_minor, df_major):
        if df is not None and not df.empty and 'timestamp' in df.columns:
            if pd.api.types.is_datetime64tz_dtype(df['timestamp']):
                df['timestamp'] = df['timestamp'].dt.tz_convert(None)

    timestamp_now = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(output_dir, f"{prefix}_{symbol}_{timestamp_now}.xlsx")

    with pd.ExcelWriter(filename, engine='openpyxl') as writer:
        if df_minor is not None and not df_minor.empty:
            df_minor.to_excel(writer, sheet_name=f"{symbol}_minor", index=False)
        if df_major is not None and not df_major.empty:
            df_major.to_excel(writer, sheet_name=f"{symbol}_major", index=False)

    print(f"💾 Guardadas velas en: {filename}")
    print(f"🕒 Última minor: {df_minor['timestamp'].iloc[-1]}")
    print(f"🕒 Última major: {df_major['timestamp'].iloc[-1]}")
    print("-" * 60)



# --- MAIN LOOP ---
if __name__ == "__main__":
    print(f"🚀 Iniciando debug de velas para {SYMBOL} ({TIMEFRAME_MAJOR} + {TIMEFRAME_MINOR})")
    exchange = connect_bitget_01()

    while True:
        # 1️⃣ Descargamos velas
        recent_major = _call_history_candles(symbol=SYMBOL, granularity=TIMEFRAME_MAJOR, limit=50)
        recent_minor = _call_history_candles(symbol=SYMBOL, granularity=TIMEFRAME_MINOR, limit=50)

        if not recent_major or not recent_minor:
            print("⚠️ No se pudieron obtener velas.")
            time.sleep(30)
            continue

        df_major = to_dataframe_from_api(recent_major)
        df_minor = to_dataframe_from_api(recent_minor)

        # 2️⃣ Guardamos en Excel
        save_candles_to_excel_single(SYMBOL, df_minor, df_major, output_dir=OUTPUT_DIR)

        # 3️⃣ Salimos o esperamos próxima vela
        if not REFRESH_EACH_CANDLE:
            break
        print(f"⏳ Esperando siguiente cierre de {TIMEFRAME_MINOR}...\n")
        wait_for_next_candle(TIMEFRAME_MINOR)
