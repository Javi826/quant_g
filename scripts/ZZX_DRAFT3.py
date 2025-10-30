"""
Script de validación para paths sintéticos y señales de trading
Valida:
1. Correcta generación de paths sintéticos (OHLC coherente)
2. Derivación correcta de timeframe mayor desde menor
3. Generación correcta de señales en ambos timeframes
4. Sincronización temporal entre timeframes
"""

import numpy as np
import pandas as pd
import random
from datetime import datetime, timedelta

# ============================================================================
# FUNCIONES IMPORTADAS DEL CÓDIGO ORIGINAL
# ============================================================================

DTYPE = np.float32

def compute_candle_features(df, raw_columns=[]):
    """Calcula features de las velas para generar paths sintéticos"""
    df = df.copy()
    df["pct_open_low"]   = (df["low"] - df["open"]) / df["open"]
    df["pct_open_high"]  = (df["high"] - df["open"]) / df["open"]
    df["pct_open_close"] = (df["close"] - df["open"]) / df["open"]

    if len(df.index) >= 2:
        time_index = (df.index[1:] - df.index[:-1]).total_seconds()
        mode = pd.Series(time_index).mode()[0]
        time_index = np.insert(time_index, 0, mode)
    else:
        time_index = np.zeros(len(df.index))

    df["time_variation"] = time_index

    index_sec = df.index.view(np.int64) // 10**9
    low_sec   = pd.to_datetime(df["low_time"]).view(np.int64) // 10**9
    high_sec  = pd.to_datetime(df["high_time"]).view(np.int64) // 10**9
    df["var_low_time"]  = (low_sec - index_sec).astype(float)
    df["var_high_time"] = (high_sec - index_sec).astype(float)

    df_raw = df[raw_columns].copy() if raw_columns else pd.DataFrame(index=df.index)
    return df, df_raw


def generate_multiple_paths(df_hist, n_paths=100, n_obs=1000, raw_columns=[], base_seed=42):
    """Genera múltiples paths sintéticos a partir de datos históricos"""
    df_features, df_raw = compute_candle_features(df_hist, raw_columns)
    n_rows = len(df_features)
    if n_rows == 0 or n_obs == 0:
        return np.empty((0, 0, 0))

    cols = [
        df_features["pct_open_low"].to_numpy(np.float64),
        df_features["pct_open_high"].to_numpy(np.float64),
        df_features["pct_open_close"].to_numpy(np.float64),
        df_features["time_variation"].to_numpy(np.float64),
        df_features["var_low_time"].to_numpy(np.float64),
        df_features["var_high_time"].to_numpy(np.float64)
    ]
    for rc in raw_columns:
        cols.append(df_raw[rc].to_numpy(np.float64))
    data_array = np.column_stack(cols)

    n_features     = data_array.shape[1]
    n_raw          = n_features - 6
    n_features_out = 7 + n_raw

    start_price     = float(df_features["open"].iloc[-1])
    start_timestamp = df_features.index[-1].value // 10**9

    paths_array = np.empty((n_paths, n_obs, n_features_out), dtype=np.float64)

    for i in range(n_paths):
        rnd     = random.Random(base_seed + i)
        indices = np.array([rnd.randrange(n_rows) for _ in range(n_obs)], dtype=np.int64)
        sampled = data_array[indices]

        pct_open_low, pct_open_high, pct_open_close = sampled[:, 0], sampled[:, 1], sampled[:, 2]

        multipliers  = 1.0 + pct_open_close
        close_prices = start_price * np.cumprod(multipliers)
        open_prices  = np.empty_like(close_prices)
        open_prices[0] = start_price
        open_prices[1:] = close_prices[:-1]

        low_prices  = np.minimum(open_prices * (1.0 + pct_open_low), close_prices)
        high_prices = np.maximum(open_prices * (1.0 + pct_open_high), close_prices)

        cumul_seconds = np.cumsum(sampled[:, 3])
        times      = start_timestamp + cumul_seconds
        low_times  = times + sampled[:, 4]
        high_times = times + sampled[:, 5]

        base_cols = [
            open_prices, low_prices, high_prices, close_prices,
            low_times, high_times, times
        ]
        if n_raw > 0:
            for idx_col in range(n_raw):
                base_cols.append(sampled[:, 6 + idx_col])
        paths_array[i, :, :] = np.column_stack(base_cols)

    return paths_array.astype(DTYPE, copy=False)


def derive_major_from_minor(paths_minor: np.ndarray, factor: int = 6) -> np.ndarray:
    """Deriva timeframe mayor desde timeframe menor"""
    n_paths, n_obs, n_features = paths_minor.shape
    n_obs_major = n_obs // factor
    paths_major = np.empty((n_paths, n_obs_major, n_features), dtype=paths_minor.dtype)

    for p in range(n_paths):
        path = paths_minor[p]
        for j in range(n_obs_major):
            start_idx = j * factor
            end_idx = (j + 1) * factor
            block = path[start_idx:end_idx]

            open_ = block[0, 0]
            close = block[-1, 3]

            high_idx = np.argmax(block[:, 2])
            high = block[high_idx, 2]
            high_t = block[high_idx, 5]

            low_idx = np.argmin(block[:, 1])
            low = block[low_idx, 1]
            low_t = block[low_idx, 4]

            major_timestamp = block[0, 6]

            paths_major[p, j, 0] = open_
            paths_major[p, j, 1] = low
            paths_major[p, j, 2] = high
            paths_major[p, j, 3] = close
            paths_major[p, j, 4] = low_t
            paths_major[p, j, 5] = high_t
            paths_major[p, j, 6] = major_timestamp

    return paths_major


def get_last_closed_major_bar(ts_mayor, ts_minor_now):
    """Encuentra la última barra mayor cerrada antes del timestamp menor actual"""
    ts_mayor_close = ts_mayor + pd.Timedelta(days=1)
    mask           = ts_mayor_close <= ts_minor_now
    indices        = np.where(mask)[0]
    return indices[-1] if len(indices) > 0 else None


def explosive_signal_tf(
    high_mayor, close_mayor,
    high_menor, close_menor,
    lookback_mayor=1, lookback_menor=2,
    index_mayor=None, index_menor=None,
    live=False
):
    """Genera señales explosivas basadas en dos timeframes"""
    ts_mayor = pd.to_datetime(index_mayor)
    ts_menor = pd.to_datetime(index_menor)

    high_mayor = np.array(high_mayor)
    close_mayor = np.array(close_mayor)
    high_menor = np.array(high_menor)
    close_menor = np.array(close_menor)
        
    n_minor = len(close_menor)
    n_major = len(close_mayor)
    
    final_signal = np.zeros(n_minor, dtype=int)
    signal_minor_array = np.zeros(n_minor, dtype=int)
    signal_major_array = np.zeros(n_major, dtype=int)
    
    # Señales major
    for j in range(n_major):
        if j < lookback_mayor:
            signal_major_array[j] = 0
        else:
            close_major = close_mayor[j]
            highs_major = high_mayor[j - lookback_mayor:j]
            signal_major_array[j] = 1 if close_major > np.max(highs_major) else 0
    
    # Señales minor y combinadas
    for i in range(1, n_minor):
        ts_minor_now = ts_menor[i]
        
        if i - 1 < lookback_menor:
            signal_minor = 0
        else:
            close_prev = close_menor[i - 1]
            highs_prev = high_menor[i - 1 - lookback_menor:i - 1]
            signal_minor = 1 if close_prev > np.max(highs_prev) else 0
        signal_minor_array[i] = signal_minor
        
        idx_major = get_last_closed_major_bar(ts_mayor, ts_minor_now)
        
        if idx_major is None:
            sig_major_for_this_minor = 0
        else:
            sig_major_for_this_minor = int(signal_major_array[idx_major])
        
        final_signal[i] = 1 if (signal_minor == 1 and sig_major_for_this_minor == 1) else 0
    
    return final_signal


# ============================================================================
# FUNCIONES DE VALIDACIÓN
# ============================================================================

def create_synthetic_data():
    """Crea datos sintéticos de prueba"""
    dates = pd.date_range('2024-01-01', periods=100, freq='12H')
    np.random.seed(42)
    
    data = {
        'open': 100 + np.random.randn(100).cumsum(),
        'close': 100 + np.random.randn(100).cumsum(),
        'low_time': dates,
        'high_time': dates
    }
    
    df = pd.DataFrame(data, index=dates)
    df['high'] = df[['open', 'close']].max(axis=1) + np.abs(np.random.randn(100))
    df['low'] = df[['open', 'close']].min(axis=1) - np.abs(np.random.randn(100))
    
    return df


def validate_ohlc_consistency(paths, path_idx=0):
    """Valida que OHLC sea consistente en los paths"""
    print("\n" + "="*80)
    print("VALIDACIÓN 1: Consistencia OHLC")
    print("="*80)
    
    path = paths[path_idx]
    issues = []
    
    for i in range(len(path)):
        o, l, h, c = path[i, 0], path[i, 1], path[i, 2], path[i, 3]
        
        if not (l <= o <= h and l <= c <= h):
            issues.append(f"Barra {i}: O={o:.2f}, L={l:.2f}, H={h:.2f}, C={c:.2f}")
        
        if l > min(o, c) or h < max(o, c):
            issues.append(f"Barra {i}: Low/High no contiene Open/Close")
    
    if issues:
        print(f"❌ ENCONTRADOS {len(issues)} ERRORES:")
        for issue in issues[:10]:  # Mostrar primeros 10
            print(f"   {issue}")
    else:
        print("✅ OHLC consistente en todas las barras")
    
    return len(issues) == 0


def validate_major_derivation(paths_minor, paths_major, factor, path_idx=0):
    """Valida que la derivación del timeframe mayor sea correcta"""
    print("\n" + "="*80)
    print("VALIDACIÓN 2: Derivación Timeframe Mayor")
    print("="*80)
    
    minor = paths_minor[path_idx]
    major = paths_major[path_idx]
    issues = []
    
    for j in range(len(major)):
        start_idx = j * factor
        end_idx = (j + 1) * factor
        block = minor[start_idx:end_idx]
        
        # Validar Open
        if not np.isclose(major[j, 0], block[0, 0]):
            issues.append(f"Major {j}: Open incorrecto")
        
        # Validar Close
        if not np.isclose(major[j, 3], block[-1, 3]):
            issues.append(f"Major {j}: Close incorrecto")
        
        # Validar High
        expected_high = np.max(block[:, 2])
        if not np.isclose(major[j, 2], expected_high):
            issues.append(f"Major {j}: High incorrecto (esperado {expected_high:.2f}, obtenido {major[j, 2]:.2f})")
        
        # Validar Low
        expected_low = np.min(block[:, 1])
        if not np.isclose(major[j, 1], expected_low):
            issues.append(f"Major {j}: Low incorrecto (esperado {expected_low:.2f}, obtenido {major[j, 1]:.2f})")
    
    if issues:
        print(f"❌ ENCONTRADOS {len(issues)} ERRORES:")
        for issue in issues[:10]:
            print(f"   {issue}")
    else:
        print("✅ Derivación correcta en todas las barras major")
    
    return len(issues) == 0


def validate_timestamps(paths_minor, paths_major, factor, path_idx=0):
    """Valida la consistencia temporal"""
    print("\n" + "="*80)
    print("VALIDACIÓN 3: Consistencia Temporal")
    print("="*80)
    
    minor = paths_minor[path_idx]
    major = paths_major[path_idx]
    issues = []
    
    # Timestamps monotónicamente crecientes en minor
    ts_minor = minor[:, 6]
    if not np.all(np.diff(ts_minor) >= 0):
        issues.append("Timestamps minor NO son monotónicamente crecientes")
    
    # Timestamps monotónicamente crecientes en major
    ts_major = major[:, 6]
    if not np.all(np.diff(ts_major) >= 0):
        issues.append("Timestamps major NO son monotónicamente crecientes")
    
    # Validar alineación temporal
    for j in range(len(major)):
        start_idx = j * factor
        expected_ts = minor[start_idx, 6]
        actual_ts = major[j, 6]
        
        if not np.isclose(expected_ts, actual_ts):
            issues.append(f"Major {j}: Timestamp no alineado con minor")
    
    if issues:
        print(f"❌ ENCONTRADOS {len(issues)} ERRORES:")
        for issue in issues:
            print(f"   {issue}")
    else:
        print("✅ Timestamps consistentes")
    
    return len(issues) == 0


def validate_signals(paths_minor, paths_major, lookback_major=1, lookback_minor=2):
    """Valida la generación de señales"""
    print("\n" + "="*80)
    print("VALIDACIÓN 4: Generación de Señales")
    print("="*80)
    
    path_idx = 0
    minor = paths_minor[path_idx]
    major = paths_major[path_idx]
    
    ts_minor = pd.to_datetime(minor[:, 6], unit='s')
    ts_major = pd.to_datetime(major[:, 6], unit='s')
    
    signals = explosive_signal_tf(
        high_mayor=major[:, 2],
        close_mayor=major[:, 3],
        high_menor=minor[:, 2],
        close_menor=minor[:, 3],
        lookback_mayor=lookback_major,
        lookback_menor=lookback_minor,
        index_mayor=ts_major,
        index_menor=ts_minor,
        live=False
    )
    
    n_signals = np.sum(signals)
    signal_positions = np.where(signals == 1)[0]
    
    print(f"📊 Total señales generadas: {n_signals}")
    print(f"📊 Total barras minor: {len(signals)}")
    print(f"📊 Porcentaje de señales: {100*n_signals/len(signals):.2f}%")
    
    if n_signals > 0:
        print(f"\n🎯 Primeras 5 posiciones con señal: {signal_positions[:5]}")
        print("\nValidando lógica de señales en primeras 5 ocurrencias:")
        
        for idx in signal_positions[:5]:
            if idx >= lookback_minor:
                close_prev = minor[idx-1, 3]
                highs_prev = minor[idx-1-lookback_minor:idx-1, 2]
                signal_minor = close_prev > np.max(highs_prev)
                
                ts_minor_now = ts_minor[idx]
                idx_major = get_last_closed_major_bar(ts_major, ts_minor_now)
                
                if idx_major is not None and idx_major >= lookback_major:
                    close_major = major[idx_major, 3]
                    highs_major = major[idx_major-lookback_major:idx_major, 2]
                    signal_major = close_major > np.max(highs_major)
                    
                    print(f"  Minor {idx}: signal_minor={signal_minor}, signal_major={signal_major}, combined={signal_minor and signal_major}")
    else:
        print("⚠️  No se generaron señales. Verificar parámetros o datos.")
    
    return n_signals > 0


def run_comprehensive_validation():
    """Ejecuta todas las validaciones"""
    print("\n" + "🔍 "*20)
    print("VALIDACIÓN COMPLETA DE PATHS SINTÉTICOS Y SEÑALES")
    print("🔍 "*20)
    
    # Crear datos de prueba
    print("\n📦 Creando datos sintéticos de prueba...")
    df_test = create_synthetic_data()
    print(f"   Generados {len(df_test)} registros históricos")
    
    # Generar paths
    print("\n🎲 Generando paths sintéticos...")
    n_paths = 5
    n_obs = 48  # 2 días de datos en 12H
    paths_minor = generate_multiple_paths(df_test, n_paths=n_paths, n_obs=n_obs, base_seed=42)
    print(f"   Shape paths minor: {paths_minor.shape}")
    
    # Derivar major
    print("\n📈 Derivando timeframe mayor (factor=2)...")
    factor = 2
    paths_major = derive_major_from_minor(paths_minor, factor=factor)
    print(f"   Shape paths major: {paths_major.shape}")
    
    # Ejecutar validaciones
    results = {}
    results['ohlc'] = validate_ohlc_consistency(paths_minor, path_idx=0)
    results['derivation'] = validate_major_derivation(paths_minor, paths_major, factor, path_idx=0)
    results['timestamps'] = validate_timestamps(paths_minor, paths_major, factor, path_idx=0)
    results['signals'] = validate_signals(paths_minor, paths_major, lookback_major=1, lookback_minor=2)
    
    # Resumen final
    print("\n" + "="*80)
    print("RESUMEN DE VALIDACIÓN")
    print("="*80)
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} - {test_name.upper()}")
    
    print("\n" + ("🎉 TODAS LAS VALIDACIONES PASARON 🎉" if all_passed else "⚠️  ALGUNAS VALIDACIONES FALLARON ⚠️"))
    print("="*80 + "\n")
    
    return results


# ============================================================================
# EJECUCIÓN PRINCIPAL
# ============================================================================

if __name__ == "__main__":
    results = run_comprehensive_validation()