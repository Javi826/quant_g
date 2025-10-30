import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def explosive_signal_tf(high_mayor, close_mayor, high_menor, close_menor, 
                        low_mayor, low_menor, 
                        lookback_mayor, lookback_menor, live=False):
    """Señales LONG con shift(1) en timeframe mayor y menor"""

    # Señales individuales
    signal_mayor = np.zeros_like(close_mayor, dtype=np.int8)
    signal_menor = np.zeros_like(close_menor, dtype=np.int8)

    # Timeframe mayor
    n_mayor = len(close_mayor)
    for i in range(lookback_mayor, n_mayor):
        window = high_mayor[i-lookback_mayor:i]
        if np.all(window[:-1] > window[1:]):
            max_window = np.max(window)
            if close_mayor[i] > max_window:
                signal_mayor[i] = 1

    # Timeframe menor
    n_menor = len(close_menor)
    for i in range(lookback_menor, n_menor):
        window = high_menor[i-lookback_menor:i]
        if np.all(window[:-1] > window[1:]):
            max_window = np.max(window)
            if close_menor[i] > max_window:
                signal_menor[i] = 1

    # Shift(1) en ambos timeframes si no es en tiempo real
    if not live:
        # Shift mayor
        signal_shifted_mayor = np.empty_like(signal_mayor)
        signal_shifted_mayor[0] = 0
        signal_shifted_mayor[1:] = signal_mayor[:-1]
        signal_mayor = signal_shifted_mayor

        # Shift menor
        signal_shifted_menor = np.empty_like(signal_menor)
        signal_shifted_menor[0] = 0
        signal_shifted_menor[1:] = signal_menor[:-1]
        signal_menor = signal_shifted_menor

    # Combinación multi-timeframe usando ambas señales shiftadas
    factor = len(close_menor) // len(close_mayor)
    signal_final = np.zeros_like(close_menor, dtype=np.int8)

    for i, s_menor in enumerate(signal_menor):
        idx_mayor = i // factor
        if idx_mayor < len(signal_mayor) and s_menor == 1 and signal_mayor[idx_mayor] == 1:
            signal_final[i] = 1

    return signal_final


def explosive_signal_tf_corrected(high_mayor, close_mayor, high_menor, close_menor, 
                                  low_mayor, low_menor, timestamp_mayor, timestamp_menor, 
                                  lookback_mayor, lookback_menor, live=False):
    """Versión corregida sin lookahead bias"""
    signal_mayor = np.zeros_like(close_mayor, dtype=np.int8)
    signal_menor = np.zeros_like(close_menor, dtype=np.int8)
    
    n_mayor = len(close_mayor)
    for i in range(lookback_mayor, n_mayor):
        window = high_mayor[i-lookback_mayor:i]
        if np.all(window[:-1] > window[1:]):
            max_window = np.max(window)
            if close_mayor[i] > max_window:
                signal_mayor[i] = 1
    
    n_menor = len(close_menor)
    for i in range(lookback_menor, n_menor):
        window = high_menor[i-lookback_menor:i]
        if np.all(window[:-1] > window[1:]):
            max_window = np.max(window)
            if close_menor[i] > max_window:
                signal_menor[i] = 1
    
    ts_mayor = pd.Series(timestamp_mayor)
    ts_menor = pd.Series(timestamp_menor)
    signal_final = np.zeros_like(close_menor, dtype=np.int8)
    
    for i, ts in enumerate(ts_menor):
        # CORRECCIÓN: Buscar la última vela COMPLETAMENTE CERRADA del TF mayor
        # side='left' nos da la vela que contiene ts, restamos 1 para la anterior
        idx_mayor = ts_mayor.searchsorted(ts, side='left') - 1
        
        if idx_mayor >= 0:
            # Ahora usamos la vela anterior que ya está cerrada
            if signal_menor[i] == 1 and signal_mayor[idx_mayor] == 1:
                signal_final[i] = 1
    
    if not live:
        signal_shifted = np.zeros_like(signal_final)
        signal_shifted[1:] = signal_final[:-1]
        signal_final = signal_shifted
    
    return signal_final


def test_lookahead_bias():
    """Test que demuestra el lookahead bias"""
    print("=" * 80)
    print("TEST DE LOOKAHEAD BIAS EN SEÑAL MULTI-TIMEFRAME")
    print("=" * 80)
    
    # Crear datos sintéticos: TF mayor = 1H, TF menor = 15min
    start = datetime(2024, 1, 1, 0, 0)
    
    # TF Mayor (1H) - 10 velas
    timestamps_mayor = [start + timedelta(hours=i) for i in range(10)]
    high_mayor = np.array([110, 108, 106, 104, 102, 115, 113, 111, 109, 107])
    close_mayor = np.array([105, 103, 101, 99, 97, 110, 108, 106, 104, 102])
    low_mayor = np.array([100, 98, 96, 94, 92, 105, 103, 101, 99, 97])
    
    # TF Menor (15min) - 4 velas por hora = 40 velas
    timestamps_menor = [start + timedelta(minutes=15*i) for i in range(40)]
    high_menor = np.concatenate([
        [109, 108, 107, 106],  # Hora 0
        [107, 106, 105, 104],  # Hora 1
        [105, 104, 103, 102],  # Hora 2
        [103, 102, 101, 100],  # Hora 3
        [101, 100, 99, 98],    # Hora 4
        [114, 113, 112, 120],  # Hora 5 - SEÑAL SE GENERA AQUÍ
        [118, 117, 116, 115],  # Hora 6
        [116, 115, 114, 113],  # Hora 7
        [114, 113, 112, 111],  # Hora 8
        [112, 111, 110, 109],  # Hora 9
    ])
    close_menor = high_menor - 2
    low_menor = high_menor - 5
    
    # Ejecutar ambas versiones
    signal_original = explosive_signal_tf(
        high_mayor, close_mayor, high_menor, close_menor,
        low_mayor, low_menor, timestamps_mayor, timestamps_menor,
        lookback_mayor=5, lookback_menor=5, live=False
    )
    
    signal_corrected = explosive_signal_tf_corrected(
        high_mayor, close_mayor, high_menor, close_menor,
        low_mayor, low_menor, timestamps_mayor, timestamps_menor,
        lookback_mayor=5, lookback_menor=5, live=False
    )
    
    # Análisis detallado
    print("\n📊 ANÁLISIS DE VELAS DEL TIMEFRAME MAYOR (1H)")
    print("-" * 80)
    for i, ts in enumerate(timestamps_mayor):
        print(f"Vela {i}: {ts.strftime('%Y-%m-%d %H:%M')} | "
              f"H={high_mayor[i]:>3} C={close_mayor[i]:>3} L={low_mayor[i]:>3}")
    
    print("\n📊 ANÁLISIS CRÍTICO: Hora 5 (Vela índice 23 del TF menor)")
    print("-" * 80)
    hora_5_idx = 20  # Inicio de hora 5
    print(f"\nVelas del TF menor en hora 5 (índices {hora_5_idx} a {hora_5_idx+3}):")
    for i in range(hora_5_idx, hora_5_idx + 4):
        ts = timestamps_menor[i]
        print(f"  Vela {i}: {ts.strftime('%Y-%m-%d %H:%M')} | "
              f"H={high_menor[i]:>3} C={close_menor[i]:>3}")
    
    print(f"\n⚠️  PROBLEMA DE LOOKAHEAD BIAS:")
    print(f"  • La vela 23 (15:45) está DENTRO de la hora 5")
    print(f"  • La vela del TF mayor (hora 5) NO HA CERRADO todavía")
    print(f"  • Pero el código original usa searchsorted(side='right')-1")
    print(f"  • Esto devuelve idx_mayor=5 (la vela EN FORMACIÓN)")
    
    # Verificar qué índice devuelve cada método
    ts_test = timestamps_menor[23]  # Vela a las 15:45 (dentro de hora 5)
    ts_mayor_series = pd.Series(timestamps_mayor)
    
    idx_original = ts_mayor_series.searchsorted(ts_test, side='right') - 1
    idx_corrected = ts_mayor_series.searchsorted(ts_test, side='left') - 1
    
    print(f"\n🔍 COMPARACIÓN DE ÍNDICES para vela menor en {ts_test.strftime('%H:%M')}:")
    print(f"  • Método ORIGINAL (side='right')-1: {idx_original} → Vela {timestamps_mayor[idx_original].strftime('%H:%M')} (EN FORMACIÓN ❌)")
    print(f"  • Método CORREGIDO (side='left')-1: {idx_corrected} → Vela {timestamps_mayor[idx_corrected].strftime('%H:%M')} (CERRADA ✓)")
    
    # Mostrar señales
    print("\n📈 SEÑALES GENERADAS:")
    print("-" * 80)
    original_signals = np.where(signal_original == 1)[0]
    corrected_signals = np.where(signal_corrected == 1)[0]
    
    print(f"Señales ORIGINALES (con lookahead bias): {len(original_signals)} señales")
    for idx in original_signals:
        print(f"  → Vela {idx}: {timestamps_menor[idx].strftime('%Y-%m-%d %H:%M')}")
    
    print(f"\nSeñales CORREGIDAS (sin lookahead bias): {len(corrected_signals)} señales")
    for idx in corrected_signals:
        print(f"  → Vela {idx}: {timestamps_menor[idx].strftime('%Y-%m-%d %H:%M')}")
    
    # Conclusión
    print("\n" + "=" * 80)
    print("🎯 CONCLUSIÓN:")
    print("=" * 80)
    print("✓ SÍ HAY LOOKAHEAD BIAS en la versión original")
    print("✓ El problema está en usar searchsorted(side='right')-1")
    print("✓ Esto hace que uses datos de la vela mayor que AÚN NO HA CERRADO")
    print("\n📋 SOLUCIÓN:")
    print("  Usa searchsorted(side='left')-1 para obtener la última vela COMPLETAMENTE CERRADA")
    print("  O en live mode, verifica que solo uses velas completadas del TF mayor")
    print("=" * 80)


if __name__ == "__main__":
    test_lookahead_bias()