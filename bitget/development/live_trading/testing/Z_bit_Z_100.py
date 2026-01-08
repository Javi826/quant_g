import os
import sys
from datetime import datetime, timezone

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from parquet_process.Z_parquet_01_extraction import _call_history_candles, to_dataframe_from_api

UTC_TZ = timezone.utc

# ----------------------
# CONFIGURACIÓN
# ----------------------
SYMBOL = 'BTCUSDT'
TIMEFRAME = '4H'

# ----------------------
# VER ÚLTIMA VELA
# ----------------------
print("=" * 80)
print("🕐 ÚLTIMA VELA RECIBIDA DE LA API")
print("=" * 80)

# Hora actual
now_utc = datetime.now(UTC_TZ)

print(f"\n⏰ Hora actual UTC: {now_utc.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📊 Símbolo: {SYMBOL}")
print(f"⏱️  Timeframe: {TIMEFRAME}")
print("\n" + "-" * 80 + "\n")

# Recibir datos
recent = _call_history_candles(symbol=SYMBOL, granularity=TIMEFRAME, limit=5)
df = to_dataframe_from_api(recent)


# Si hay columna timestamp, usarla
if 'timestamp' in df.columns:
    for idx, row in df.iterrows():
        # Convertir timestamp a datetime
        ts_val = row['timestamp']
        if hasattr(ts_val, 'timestamp'):
            # Ya es un Timestamp de pandas
            ts_utc = ts_val.tz_convert(UTC_TZ).strftime('%Y-%m-%d %H:%M') if ts_val.tz else ts_val.strftime('%Y-%m-%d %H:%M')
        else:
            # Es un número (milisegundos)
            dt = datetime.fromtimestamp(float(ts_val)/1000, tz=UTC_TZ)
            ts_utc = dt.strftime('%Y-%m-%d %H:%M')
        
        color = '🟢' if float(row['close']) > float(row['open']) else '🔴'
        print(f"   {ts_utc} | O: {float(row['open']):>9.2f} | H: {float(row['high']):>9.2f} | L: {float(row['low']):>9.2f} | C: {float(row['close']):>9.2f} {color}")
else:
    # Si el índice es DatetimeIndex
    if hasattr(df.index, 'strftime'):
        for idx, row in df.iterrows():
            ts = idx.strftime('%Y-%m-%d %H:%M')
            color = '🟢' if float(row['close']) > float(row['open']) else '🔴'
            print(f"   {ts} | O: {float(row['open']):>9.2f} | H: {float(row['high']):>9.2f} | L: {float(row['low']):>9.2f} | C: {float(row['close']):>9.2f} {color}")
    else:
        for idx, row in df.iterrows():
            color = '🟢' if float(row['close']) > float(row['open']) else '🔴'
            print(f"   Índice {idx} | O: {float(row['open']):>9.2f} | H: {float(row['high']):>9.2f} | L: {float(row['low']):>9.2f} | C: {float(row['close']):>9.2f} {color}")

print("\n" + "-" * 80 + "\n")


last = df.iloc[-1]
ts_val = last['timestamp'] if 'timestamp' in df.columns else df.index[-1]

if hasattr(ts_val, 'timestamp'):
    ts_utc = ts_val.tz_convert(UTC_TZ).strftime('%Y-%m-%d %H:%M:%S') if ts_val.tz else ts_val.strftime('%Y-%m-%d %H:%M:%S')
else:
    ts_utc = str(ts_val)



# ----------------------
# GENERAR SEÑALES PARA TODOS LOS SÍMBOLOS
# ----------------------
from Z_add_signals_double_top import detect_double_top_long
from ZX_utils_live import load_final_symbols, normalize_live_ohlcv, df_to_arrays_live, PRODUCT_TYPE
from parquet_process.Z_parquet_01_extraction import get_futures_symbols_from_api

STRATEGY = "double_top_long"
LOOKBACK_MINOR        = 2
PRICE_TOLERANCE       = 20
TREND_TH              = 10

print("\n🔍 GENERANDO SEÑALES PARA TODOS LOS SÍMBOLOS")
print("=" * 80)
print(f"\n⏰ Hora actual UTC: {datetime.now(UTC_TZ).strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📊 Estrategia: {STRATEGY}")
print(f"⏱️  Timeframe: {TIMEFRAME}")
print("\n" + "-" * 80 + "\n")

# Cargar símbolos
all_symbols   = get_futures_symbols_from_api(PRODUCT_TYPE)
final_symbols = load_final_symbols(all_symbols, strategy=STRATEGY, timeframe=TIMEFRAME)

print(f"📋 Total símbolos a analizar: {len(final_symbols)}\n")
print("-" * 80 + "\n")

# Recopilar datos
ohlcv_data = {}
for sym in final_symbols:
    try:
        recent = _call_history_candles(symbol=sym, granularity=TIMEFRAME, limit=100)
        df = to_dataframe_from_api(recent)
        ohlcv_data[sym] = df
    except Exception as e:
        print(f"⚠️  Error obteniendo datos para {sym}: {e}")

# Detectar señales
signals_summary = []

for sym, df_minor in ohlcv_data.items():
    try:
        # Normalizar y convertir
        df_normalized = normalize_live_ohlcv(df_minor)
        arr_minor = df_to_arrays_live(df_normalized)
        
        # Detectar señales con True y False
        signals_true = detect_double_top_long(
            arr_minor,
            lookback_minor=LOOKBACK_MINOR,
            price_tolerance=PRICE_TOLERANCE,
            trend_th=TREND_TH,
            live_trading=True
        )
        
        signals_false = detect_double_top_long(
            arr_minor,
            lookback_minor=LOOKBACK_MINOR,
            price_tolerance=PRICE_TOLERANCE,
            trend_th=TREND_TH,
            live_trading=False
        )
        
        last_signal_true = signals_true[-1]
        last_signal_false = signals_false[-1]
        
        # Obtener timestamp de última vela
        last = df_normalized.iloc[-1]
        ts_val = last.name if 'timestamp' not in df_normalized.columns else last['timestamp']
        
        if hasattr(ts_val, 'timestamp'):
            ts_utc = ts_val.tz_convert(UTC_TZ).strftime('%Y-%m-%d %H:%M') if ts_val.tz else ts_val.strftime('%Y-%m-%d %H:%M')
        else:
            ts_utc = str(ts_val)
        
        signals_summary.append({
            'symbol': sym,
            'timestamp': ts_utc,
            'signal_true': last_signal_true,
            'signal_false': last_signal_false,
            'close': float(last['close'])
        })
        
    except Exception as e:
        print(f"⚠️  Error procesando {sym}: {e}")

# Mostrar resultados
print("📊 RESUMEN DE SEÑALES (última vela de cada símbolo):\n")
print(f"{'Símbolo':<15} | {'Timestamp UTC':<17} | {'True':<6} | {'False':<6} | {'Close':<10} | {'Estado'}")
print("-" * 80)

for item in signals_summary:
    status = ""
    if item['signal_true'] == 1 and item['signal_false'] == 0:
        status = "⚠️  LOOK-AHEAD"
    elif item['signal_true'] == 0 and item['signal_false'] == 1:
        status = "✅ OPERABLE"
    elif item['signal_true'] == 1 and item['signal_false'] == 1:
        status = "🔵 AMBOS"
    else:
        status = ""
    
    print(f"{item['symbol']:<15} | {item['timestamp']:<17} | {item['signal_true']:<6} | {item['signal_false']:<6} | {item['close']:<10.2f} | {status}")

# Resumen de señales detectadas
total_true = sum(1 for item in signals_summary if item['signal_true'] == 1)
total_false = sum(1 for item in signals_summary if item['signal_false'] == 1)
only_true = sum(1 for item in signals_summary if item['signal_true'] == 1 and item['signal_false'] == 0)
only_false = sum(1 for item in signals_summary if item['signal_true'] == 0 and item['signal_false'] == 1)

print("\n" + "-" * 80)
print(f"\n📈 ESTADÍSTICAS:")
print(f"   Total símbolos analizados:          {len(signals_summary)}")
print(f"   Señales con live_trading=True:      {total_true}")
print(f"   Señales con live_trading=False:     {total_false}")
print(f"   Solo True (look-ahead bias):        {only_true} ⚠️")
print(f"   Solo False (operables correctas):   {only_false} ✅")

print("\n" + "=" * 80)