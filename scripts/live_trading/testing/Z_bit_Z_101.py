import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from zoneinfo import ZoneInfo

# --- importa los módulos de tu estructura ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from parquet_process.Z_parquet_01_extraction import _call_history_candles, to_dataframe_from_api

# ---------------------------
# CONFIGURACIÓN
# ---------------------------
SYMBOL       = "BTCUSDT"   # cámbialo si quieres probar otro
TIMEFRAME    = "4H"
LOOKBACK     = 100
TOLERANCE    = 30
MADRID_TZ    = ZoneInfo("Europe/Madrid")

print(f"🧭 Probando señal en {SYMBOL} [{TIMEFRAME}] ...")

# ---------------------------
# DESCARGA DE DATOS REALES
# ---------------------------
try:
    recent_minor = _call_history_candles(symbol=SYMBOL, granularity=TIMEFRAME, limit=300)
    if not recent_minor:
        raise ValueError("⚠️ _call_history_candles devolvió vacío o None.")
except Exception as e:
    print(f"❌ Error al obtener datos de {SYMBOL}: {e}")
    sys.exit(1)

# Muestra una parte del JSON crudo (máx. 2 elementos)
print(f"📦 Ejemplo de datos crudos:\n{recent_minor[:2] if isinstance(recent_minor, list) else recent_minor}")

try:
    df_minor = to_dataframe_from_api(recent_minor)
except Exception as e:
    print(f"❌ Error convirtiendo a DataFrame: {e}")
    sys.exit(1)

if df_minor is None or df_minor.empty:
    print(f"⚠️ DataFrame vacío tras conversión — revisa que el símbolo o timeframe existan ({SYMBOL}, {TIMEFRAME}).")
    sys.exit(1)

# Imprime datos básicos sin indexado directo
print(f"✅ Datos recibidos: {len(df_minor)} velas")
print(df_minor.head(3))
print(df_minor.tail(3))

# ---------------------------
# PREPARA LOS ARRAYS
# ---------------------------
arr = {
    'open':  df_minor['open'].to_numpy(),
    'close': df_minor['close'].to_numpy(),
}

# ---------------------------
# VERSIÓN DEBUG DE LA DETECCIÓN
# ---------------------------
def detect_parity_reversal_long_debug(arr, lookback, tolerance):
    opens  = arr['open']
    closes = arr['close']
    n      = len(closes)
    
    signals    = np.zeros(n, dtype=np.int8)
    body_sizes = np.abs(closes - opens)
    is_red     = closes < opens
    is_green   = closes > opens
    
    print(f"🔍 n={n}, lookback={lookback}, tolerance={tolerance}")
    
    for i in range(lookback, n):
        for j in range(1, lookback):
            if i - j - 1 < 0:
                break
            
            idx_red1 = i - j - 1
            idx_green = i - j
            
            if is_red[idx_red1] and is_green[idx_green]:
                size_red1 = body_sizes[idx_red1]
                size_green = body_sizes[idx_green]
                if size_red1 == 0:
                    continue
                
                diff_green_red1 = abs(size_green - size_red1) / size_red1 * 100
                if diff_green_red1 <= tolerance:
                    for k in range(idx_green + 1, i):
                        if is_red[k]:
                            size_red2 = body_sizes[k]
                            close_red1 = closes[idx_red1]
                            close_red2 = closes[k]
                            
                            diff_red2_red1 = abs(size_red2 - size_red1) / size_red1 * 100
                            diff_close = abs(close_red2 - close_red1) / abs(close_red1) * 100
                            
                            if diff_red2_red1 <= tolerance and diff_close <= tolerance:
                                print(f"\n✅ Señal detectada en vela {i}")
                                print(f"   Red1={idx_red1}, Green={idx_green}, Red2={k}")
                                print(f"   size_red1={size_red1:.4f}, size_green={size_green:.4f}, size_red2={size_red2:.4f}")
                                print(f"   diff_green_red1={diff_green_red1:.2f}%, diff_red2_red1={diff_red2_red1:.2f}%, diff_close={diff_close:.2f}%")
                                signals[i] = 1
                                break
                    
                    if signals[i] == 1:
                        break
    return signals


# ---------------------------
# EJECUCIÓN Y RESULTADOS
# ---------------------------
signals = detect_parity_reversal_long_debug(arr, LOOKBACK, TOLERANCE)

df_minor["signal"] = signals
print("\n📈 Últimas velas con señales:")
print(df_minor[df_minor["signal"] == 1][["open", "close"]].tail(10))

if df_minor["signal"].sum() == 0:
    print("\n⚠️ No se detectaron señales. Puede que la condición sea demasiado estricta.")
else:
    print(f"\n✅ Se detectaron {df_minor['signal'].sum()} señales.")
