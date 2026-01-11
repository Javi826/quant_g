import numpy as np
import pandas as pd
from signals.add_signals_parity import parity_long
from signals.volatility_detection import detect_volatility

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
DATA_FOLDER = "../data/crypto_2022_IS"
TIMEFRAME = '1H'
TEST_SYMBOL = 'ETHUSDT'

# =============================================================================
# CARGAR DATOS
# =============================================================================
print("📊 Cargando datos...\n")

# BTC
btc_file = f"{DATA_FOLDER}/BTCUSDT_{TIMEFRAME}.parquet"
btc_df = pd.read_parquet(btc_file)

# Detectar columna timestamp
if 'ts' in btc_df.columns:
    ts_col = 'ts'
elif 'timestamp' in btc_df.columns:
    ts_col = 'timestamp'
elif btc_df.index.name in ['timestamp', 'ts']:
    btc_df = btc_df.reset_index()
    ts_col = btc_df.columns[0]
else:
    ts_col = btc_df.columns[0]

btc_arr = {
    'ts': btc_df[ts_col].values,
    'open': btc_df['open'].values,
    'high': btc_df['high'].values,
    'low': btc_df['low'].values,
    'close': btc_df['close'].values
}

# Símbolo test
sym_file = f"{DATA_FOLDER}/{TEST_SYMBOL}_{TIMEFRAME}.parquet"
sym_df = pd.read_parquet(sym_file)

if ts_col not in sym_df.columns and sym_df.index.name == ts_col:
    sym_df = sym_df.reset_index()

sym_arr = {
    'ts': sym_df[ts_col].values,
    'open': sym_df['open'].values,
    'high': sym_df['high'].values,
    'low': sym_df['low'].values,
    'close': sym_df['close'].values
}

# =============================================================================
# GENERAR SEÑALES Y FILTRO
# =============================================================================
print("🔧 Generando señales y filtro...\n")

# Señales del símbolo
signals = parity_long(sym_arr, lookback=100, tolerance=30, ma_period=50, live_trading=False)

# Filtro de volatilidad BTC
btc_vol_filter = detect_volatility(btc_arr, atr_period=14, chaos_percentile=90)

# =============================================================================
# ALINEAR FILTRO CON SÍMBOLO (COMO EN TU MAIN)
# =============================================================================
print("🔗 Alineando filtro...\n")

sym_timestamps = sym_arr['ts']
btc_timestamps = btc_arr['ts']

aligned_filter = np.ones(len(signals), dtype=np.int8)

for i, ts in enumerate(sym_timestamps):
    btc_idx = np.searchsorted(btc_timestamps, ts)
    if btc_idx < len(btc_vol_filter):
        aligned_filter[i] = btc_vol_filter[btc_idx]

# Aplicar filtro
signals_filtered = signals * aligned_filter

# =============================================================================
# MOSTRAR EJEMPLOS DE ALINEAMIENTO
# =============================================================================
print("="*100)
print("🔍 VERIFICACIÓN DE ALINEAMIENTO - EJEMPLOS ALEATORIOS")
print("="*100)

# Encontrar velas con señal
signal_indices = np.where(signals == 1)[0]

# Tomar 10 ejemplos aleatorios
np.random.seed(42)
sample_indices = np.random.choice(signal_indices, min(10, len(signal_indices)), replace=False)
sample_indices = np.sort(sample_indices)

print(f"\nMostrando {len(sample_indices)} señales aleatorias:\n")

for idx in sample_indices:
    # Timestamp del símbolo
    sym_ts = sym_timestamps[idx]
    sym_datetime = pd.to_datetime(sym_ts)
    
    # Buscar índice correspondiente en BTC
    btc_idx = np.searchsorted(btc_timestamps, sym_ts)
    
    # Timestamp de BTC
    if btc_idx < len(btc_timestamps):
        btc_ts = btc_timestamps[btc_idx]
        btc_datetime = pd.to_datetime(btc_ts)
        ts_match = "✅ MATCH" if sym_ts == btc_ts else "❌ DESFASE"
    else:
        btc_datetime = "N/A"
        ts_match = "❌ FUERA DE RANGO"
    
    # Estado del filtro
    filter_value = aligned_filter[idx]
    filter_state = "STABLE (opera)" if filter_value == 1 else "🔥 CHAOS (bloqueado)"
    
    # Resultado
    final_signal = signals_filtered[idx]
    result = "✅ EJECUTA" if final_signal == 1 else "❌ BLOQUEADA"
    
    print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"Vela {idx:5d}")
    print(f"─────────────────────────────────────────────────────────────────────────────────────────────────")
    print(f"  {TEST_SYMBOL} timestamp: {sym_datetime}")
    print(f"  BTC timestamp:     {btc_datetime}")
    print(f"  BTC índice:        {btc_idx}")
    print(f"  Timestamps:        {ts_match}")
    print(f"  ─────────────────────────────────────────────────────────────────────────────────────────────")
    print(f"  Señal original:    1 (LONG)")
    print(f"  Filtro BTC:        {filter_state}")
    print(f"  Señal final:       {result}")
    print()

# =============================================================================
# MOSTRAR EJEMPLOS DE CHAOS
# =============================================================================
print("="*100)
print("🔥 VERIFICACIÓN - MOMENTOS DE CHAOS")
print("="*100)

# Encontrar momentos donde había señal PERO se bloqueó por CHAOS
blocked_signals = np.where((signals == 1) & (aligned_filter == 0))[0]

if len(blocked_signals) == 0:
    print("\n⚠️  No hay señales bloqueadas por CHAOS en este período")
else:
    print(f"\n✅ {len(blocked_signals)} señales bloqueadas por CHAOS\n")
    print("Mostrando primeras 5:\n")
    
    for idx in blocked_signals[:5]:
        sym_ts = sym_timestamps[idx]
        sym_datetime = pd.to_datetime(sym_ts)
        
        btc_idx = np.searchsorted(btc_timestamps, sym_ts)
        btc_ts = btc_timestamps[btc_idx]
        btc_datetime = pd.to_datetime(btc_ts)
        
        print(f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        print(f"Vela {idx:5d} - {sym_datetime}")
        print(f"─────────────────────────────────────────────────────────────────────────────────────────────────")
        print(f"  {TEST_SYMBOL} quería entrar LONG")
        print(f"  BTC en CHAOS (vela {btc_idx} - {btc_datetime})")
        print(f"  Timestamps coinciden: {'✅ SÍ' if sym_ts == btc_ts else '❌ NO'}")
        print(f"  → Señal BLOQUEADA por protección ❌")
        print()

# =============================================================================
# ESTADÍSTICAS FINALES
# =============================================================================
print("="*100)
print("📊 ESTADÍSTICAS DE ALINEAMIENTO")
print("="*100)

total_signals = np.sum(signals)
total_chaos = np.sum(aligned_filter == 0)
signals_blocked = total_signals - np.sum(signals_filtered)
signals_executed = np.sum(signals_filtered)

# Verificar que timestamps siempre coinciden
mismatches = 0
for i in range(len(sym_timestamps)):
    btc_idx = np.searchsorted(btc_timestamps, sym_timestamps[i])
    if btc_idx < len(btc_timestamps):
        if sym_timestamps[i] != btc_timestamps[btc_idx]:
            mismatches += 1

print(f"\nSeñales totales generadas:     {total_signals:,}")
print(f"Señales ejecutadas:            {signals_executed:,}")
print(f"Señales bloqueadas por CHAOS:  {signals_blocked:,}")
print(f"\nMomentos CHAOS en BTC:         {total_chaos:,} ({total_chaos/len(btc_timestamps)*100:.1f}% del tiempo)")
print(f"\nTimestamps desalineados:       {mismatches:,} {'✅ PERFECTO' if mismatches == 0 else '❌ PROBLEMA'}")
print(f"Precisión de alineamiento:     {(1 - mismatches/len(sym_timestamps))*100:.2f}%")
print("="*100)