import os
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import pandas as pd

# Rutas y conexiones (ajusta si lo necesitas)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.ZZ_connect import connect_bitget_03
from parquet_process.Z_parquet_extraction import _call_history_candles, to_dataframe_from_api

# ===========================
# CONFIGURACIÓN
# ===========================
MADRID_TZ = ZoneInfo("Europe/Madrid")

SYMBOL = "BTCUSDT"
TIMEFRAME_MAJOR = "1D"
TIMEFRAME_MINOR = "4H"


# ===========================
# FUNCIÓN PRINCIPAL
# ===========================
def check_candle_sync(symbol, tf_major, tf_minor):
    exchange = connect_bitget_03()

    # Descarga las últimas velas
    recent_major = _call_history_candles(symbol=symbol, granularity=tf_major, limit=5)
    recent_minor = _call_history_candles(symbol=symbol, granularity=tf_minor, limit=5)

    df_major = to_dataframe_from_api(recent_major)
    df_minor = to_dataframe_from_api(recent_minor)

    # Últimos timestamps
    last_major_ts = pd.to_datetime(df_major.iloc[-1]["timestamp"]).astimezone(MADRID_TZ)
    last_minor_ts = pd.to_datetime(df_minor.iloc[-1]["timestamp"]).astimezone(MADRID_TZ)
    now = datetime.now(MADRID_TZ)

    print(f"\n=== {symbol} ===")
    print(f"🕒 Now:                {now.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🔹 Last {tf_major} candle: {last_major_ts.strftime('%Y-%m-%d %H:%M:%S')}  ({(now - last_major_ts).total_seconds()/3600:.1f}h ago)")
    print(f"🔹 Last {tf_minor} candle: {last_minor_ts.strftime('%Y-%m-%d %H:%M:%S')}  ({(now - last_minor_ts).total_seconds()/3600:.1f}h ago)")

    # Chequeo de si la vela mayor está cerrada
    if tf_major.endswith("D"):
        major_duration = timedelta(days=1)
    elif tf_major.endswith("H"):
        major_duration = timedelta(hours=int(tf_major[:-1]))
    elif tf_major.endswith("m"):
        major_duration = timedelta(minutes=int(tf_major[:-1]))
    else:
        major_duration = timedelta(hours=24)

    if now - last_major_ts < major_duration:
        print(f"⚠️ La vela {tf_major} todavía está ABIERTA (incompleta)")
    else:
        print(f"✅ La vela {tf_major} está CERRADA (completa)")

    if now - last_minor_ts < timedelta(hours=4):
        print(f"⚠️ La vela {tf_minor} probablemente sigue ABIERTA")
    else:
        print(f"✅ La vela {tf_minor} está CERRADA")


# ===========================
# EJECUCIÓN
# ===========================
if __name__ == "__main__":
    check_candle_sync(SYMBOL, TIMEFRAME_MAJOR, TIMEFRAME_MINOR)
