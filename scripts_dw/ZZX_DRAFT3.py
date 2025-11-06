import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema

# =============================================================================
# IMPORTAR LA FUNCIÓN (simulada aquí para ser autocontenido)
# =============================================================================

def detect_double_top(ohlcv_array, lookback, price_tolerance, min_candles_between_peaks, backtest=True):
    """
    Detecta patrón de doble techo (double top) y genera señal de venta (-1)
    en la vela de confirmación o, si backtest=True, una vela después.
    """
    
    high  = ohlcv_array['high']
    low   = ohlcv_array['low']
    close = ohlcv_array['close']
    n = len(high)
    
    signal = np.zeros(n, dtype=np.int8)
    
    try:
        maxima_idx = argrelextrema(high, np.greater, order=lookback)[0]
        minima_idx = argrelextrema(low, np.less, order=lookback)[0]
    except Exception:
        return signal
    
    if len(maxima_idx) < 2 or len(minima_idx) == 0:
        return signal
    
    for i in range(len(maxima_idx) - 1):
        for j in range(i + 1, len(maxima_idx)):
            peak1_idx = maxima_idx[i]
            peak2_idx = maxima_idx[j]
            
            if peak2_idx - peak1_idx < min_candles_between_peaks:
                continue
            
            peak1_price = high[peak1_idx]
            peak2_price = high[peak2_idx]
            
            rel_diff = abs(peak1_price - peak2_price) / peak1_price
            if rel_diff > price_tolerance:
                continue
            
            intermediate_valleys = minima_idx[(minima_idx > peak1_idx) & 
                                             (minima_idx < peak2_idx)]
            if len(intermediate_valleys) == 0:
                continue
            
            valley_idx = intermediate_valleys[np.argmin(low[intermediate_valleys])]
            valley_price = low[valley_idx]
            
            for k in range(peak2_idx + 1, n):
                if close[k] < valley_price:
                    signal[k] = -1
                    break
    
    # Añadir desplazamiento para backtest
    if backtest:
        signal = np.roll(signal, 1)
        signal[0] = 0  # Evitar desplazamiento fuera de rango
    
    return signal


# =============================================================================
# DATOS HARDCODEADOS - ESCENARIOS DE PRUEBA
# =============================================================================

def crear_escenario_1():
    """Doble techo claro y confirmado"""
    velas = [
        # Subida inicial
        {'open': 100, 'high': 101, 'low': 99, 'close': 100.5},
        {'open': 100.5, 'high': 102, 'low': 100, 'close': 101.5},
        {'open': 101.5, 'high': 103, 'low': 101, 'close': 102.5},
        
        # PRIMER PICO (índice 3-5, ~105)
        {'open': 102.5, 'high': 105, 'low': 102, 'close': 104},
        {'open': 104, 'high': 105.2, 'low': 103.5, 'close': 104.5},
        {'open': 104.5, 'high': 105.5, 'low': 104, 'close': 104.2},
        
        # Retroceso hacia el valle
        {'open': 104.2, 'high': 104.5, 'low': 102, 'close': 102.5},
        {'open': 102.5, 'high': 103, 'low': 100, 'close': 100.5},
        {'open': 100.5, 'high': 101, 'low': 98, 'close': 98.5},
        
        # VALLE (índice 9-10, ~97)
        {'open': 98.5, 'high': 99, 'low': 96.5, 'close': 97},
        {'open': 97, 'high': 97.5, 'low': 96.8, 'close': 97.2},
        
        # Rebote hacia segundo pico
        {'open': 97.2, 'high': 99, 'low': 97, 'close': 98.5},
        {'open': 98.5, 'high': 100, 'low': 98, 'close': 99.5},
        {'open': 99.5, 'high': 102, 'low': 99, 'close': 101.5},
        {'open': 101.5, 'high': 103.5, 'low': 101, 'close': 103},
        
        # SEGUNDO PICO (índice 15-17, ~105)
        {'open': 103, 'high': 105.3, 'low': 102.5, 'close': 104.8},
        {'open': 104.8, 'high': 105.1, 'low': 103.8, 'close': 104},
        {'open': 104, 'high': 104.5, 'low': 103, 'close': 103.2},
        
        # Caída - CONFIRMACIÓN (índice 18-20)
        {'open': 103.2, 'high': 103.5, 'low': 100, 'close': 100.5},
        {'open': 100.5, 'high': 101, 'low': 98, 'close': 98.5},
        {'open': 98.5, 'high': 99, 'low': 95, 'close': 95.5},  # ← Cierra < 97 (valle)
        {'open': 95.5, 'high': 96, 'low': 93, 'close': 94},
        {'open': 94, 'high': 95, 'low': 92, 'close': 93},
    ]
    return velas


def crear_escenario_2():
    """Doble techo NO confirmado (no rompe soporte)"""
    velas = [
        {'open': 100, 'high': 101, 'low': 99, 'close': 100.5},
        {'open': 100.5, 'high': 103, 'low': 100, 'close': 102},
        {'open': 102, 'high': 110, 'low': 101.5, 'close': 108},  # Pico 1
        {'open': 108, 'high': 110.5, 'low': 107, 'close': 108.5},
        {'open': 108.5, 'high': 109, 'low': 105, 'close': 105.5},
        {'open': 105.5, 'high': 106, 'low': 102, 'close': 103},  # Valle ~102
        {'open': 103, 'high': 104, 'low': 102.5, 'close': 103.5},
        {'open': 103.5, 'high': 106, 'low': 103, 'close': 105},
        {'open': 105, 'high': 109, 'low': 104.5, 'close': 108},  # Pico 2
        {'open': 108, 'high': 110.2, 'low': 107, 'close': 108.5},
        {'open': 108.5, 'high': 109, 'low': 106, 'close': 106.5},
        {'open': 106.5, 'high': 107, 'low': 104, 'close': 104.5},  # NO rompe valle
        {'open': 104.5, 'high': 106, 'low': 104, 'close': 105},
        {'open': 105, 'high': 107, 'low': 104.5, 'close': 106},
    ]
    return velas


def crear_escenario_3():
    """Picos muy separados en tiempo"""
    velas = [
        # Primer pico
        {'open': 100, 'high': 102, 'low': 99, 'close': 101},
        {'open': 101, 'high': 120, 'low': 100, 'close': 118},
        {'open': 118, 'high': 120.5, 'low': 117, 'close': 119},
        
        # Retroceso largo (muchas velas intermedias)
        {'open': 119, 'high': 120, 'low': 115, 'close': 116},
        {'open': 116, 'high': 117, 'low': 112, 'close': 113},
        {'open': 113, 'high': 114, 'low': 110, 'close': 111},
        {'open': 111, 'high': 112, 'low': 108, 'close': 109},
        {'open': 109, 'high': 110, 'low': 105, 'close': 106},  # Valle
        {'open': 106, 'high': 107, 'low': 105, 'close': 106.5},
        {'open': 106.5, 'high': 108, 'low': 106, 'close': 107},
        
        # Recuperación hacia segundo pico
        {'open': 107, 'high': 110, 'low': 106.5, 'close': 109},
        {'open': 109, 'high': 113, 'low': 108.5, 'close': 112},
        {'open': 112, 'high': 116, 'low': 111.5, 'close': 115},
        {'open': 115, 'high': 119, 'low': 114.5, 'close': 117},
        
        # Segundo pico (similar al primero)
        {'open': 117, 'high': 120.2, 'low': 116.5, 'close': 119},
        {'open': 119, 'high': 120.8, 'low': 118, 'close': 119.5},
        
        # Caída y confirmación
        {'open': 119.5, 'high': 120, 'low': 115, 'close': 116},
        {'open': 116, 'high': 117, 'low': 110, 'close': 111},
        {'open': 111, 'high': 112, 'low': 104, 'close': 105},  # Rompe valle
        {'open': 105, 'high': 106, 'low': 100, 'close': 101},
    ]
    return velas


# =============================================================================
# FUNCIÓN DE CONVERSIÓN Y ANÁLISIS
# =============================================================================

def velas_to_ohlcv_array(velas):
    """Convierte lista de velas a formato ohlcv_array"""
    n = len(velas)
    return {
        'open':   np.array([v['open'] for v in velas]),
        'high':   np.array([v['high'] for v in velas]),
        'low':    np.array([v['low'] for v in velas]),
        'close':  np.array([v['close'] for v in velas]),
        'volume': np.random.randint(1000, 10000, n)  # Dummy volume
    }


def analizar_escenario(velas, nombre, lookback=2, price_tolerance=0.03, min_candles=5, backtest_mode=True):
    """Analiza un escenario y muestra resultados"""
    print(f"\n{'='*70}")
    print(f"  {nombre}")
    print(f"{'='*70}")
    
    ohlcv = velas_to_ohlcv_array(velas)
    
    # Test con backtest=False
    signal_sin_bt = detect_double_top(ohlcv, lookback, price_tolerance, min_candles, backtest=False)
    
    # Test con backtest=True
    signal_con_bt = detect_double_top(ohlcv, lookback, price_tolerance, min_candles, backtest=True)
    
    print(f"\n📊 Configuración:")
    print(f"   - Total velas: {len(velas)}")
    print(f"   - lookback: {lookback}")
    print(f"   - price_tolerance: {price_tolerance*100}%")
    print(f"   - min_candles_between_peaks: {min_candles}")
    
    print(f"\n🔍 Resultados SIN desplazamiento (backtest=False):")
    sell_idx_sin = np.where(signal_sin_bt == -1)[0]
    if len(sell_idx_sin) > 0:
        print(f"   ✅ Señal de VENTA en vela(s): {sell_idx_sin.tolist()}")
        for idx in sell_idx_sin:
            print(f"      → Vela #{idx}: close={ohlcv['close'][idx]:.2f}")
    else:
        print(f"   ❌ No se detectaron señales")
    
    print(f"\n🔍 Resultados CON desplazamiento (backtest=True):")
    sell_idx_con = np.where(signal_con_bt == -1)[0]
    if len(sell_idx_con) > 0:
        print(f"   ✅ Señal de VENTA en vela(s): {sell_idx_con.tolist()}")
        for idx in sell_idx_con:
            print(f"      → Vela #{idx}: close={ohlcv['close'][idx]:.2f}")
            print(f"         (Entrada real sería en vela #{idx}, precio open siguiente)")
    else:
        print(f"   ❌ No se detectaron señales")
    
    # Visualización
    visualizar_signals(ohlcv, signal_sin_bt, signal_con_bt, nombre)
    
    return ohlcv, signal_sin_bt, signal_con_bt


def visualizar_signals(ohlcv, signal_sin_bt, signal_con_bt, titulo):
    """Visualiza candlesticks con las señales"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    n = len(ohlcv['close'])
    
    # Gráfico 1: Sin desplazamiento
    for i in range(n):
        color = 'green' if ohlcv['close'][i] >= ohlcv['open'][i] else 'red'
        ax1.plot([i, i], [ohlcv['low'][i], ohlcv['high'][i]], color='black', linewidth=1)
        body_height = abs(ohlcv['close'][i] - ohlcv['open'][i])
        body_bottom = min(ohlcv['open'][i], ohlcv['close'][i])
        rect = plt.Rectangle((i - 0.3, body_bottom), 0.6, body_height, 
                            facecolor=color, edgecolor='black', linewidth=1, alpha=0.7)
        ax1.add_patch(rect)
    
    # Marcar señales sin backtest
    sell_idx_sin = np.where(signal_sin_bt == -1)[0]
    for idx in sell_idx_sin:
        ax1.scatter(idx, ohlcv['close'][idx], color='red', s=400, marker='X', 
                   zorder=10, edgecolors='black', linewidths=2, label='Señal confirmación')
    
    ax1.set_title(f"{titulo} - backtest=False (señal en vela de confirmación)", 
                 fontsize=12, fontweight='bold')
    ax1.set_ylabel('Precio', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')
    if len(sell_idx_sin) > 0:
        ax1.legend()
    
    # Gráfico 2: Con desplazamiento
    for i in range(n):
        color = 'green' if ohlcv['close'][i] >= ohlcv['open'][i] else 'red'
        ax2.plot([i, i], [ohlcv['low'][i], ohlcv['high'][i]], color='black', linewidth=1)
        body_height = abs(ohlcv['close'][i] - ohlcv['open'][i])
        body_bottom = min(ohlcv['open'][i], ohlcv['close'][i])
        rect = plt.Rectangle((i - 0.3, body_bottom), 0.6, body_height, 
                            facecolor=color, edgecolor='black', linewidth=1, alpha=0.7)
        ax2.add_patch(rect)
    
    # Marcar señales con backtest
    sell_idx_con = np.where(signal_con_bt == -1)[0]
    for idx in sell_idx_con:
        ax2.scatter(idx, ohlcv['close'][idx], color='blue', s=400, marker='v', 
                   zorder=10, edgecolors='black', linewidths=2, 
                   label='Señal desplazada (entrada real)')
    
    ax2.set_title(f"{titulo} - backtest=True (señal desplazada +1 vela)", 
                 fontsize=12, fontweight='bold')
    ax2.set_xlabel('Vela #', fontsize=11)
    ax2.set_ylabel('Precio', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    if len(sell_idx_con) > 0:
        ax2.legend()
    
    plt.tight_layout()
    plt.show()


# =============================================================================
# EJECUCIÓN DE TESTS
# =============================================================================

if __name__ == "__main__":
    print("\n" + "🚀"*35)
    print("  TEST DE DETECT_DOUBLE_TOP CON DATOS HARDCODEADOS")
    print("🚀"*35)
    
    # Test 1: Doble techo confirmado
    velas_1 = crear_escenario_1()
    ohlcv1, sig1_sin, sig1_con = analizar_escenario(
        velas_1, 
        "ESCENARIO 1: Doble Techo CONFIRMADO",
        lookback=2,
        price_tolerance=0.03,
        min_candles=3
    )
    
    # Test 2: Doble techo NO confirmado
    velas_2 = crear_escenario_2()
    ohlcv2, sig2_sin, sig2_con = analizar_escenario(
        velas_2, 
        "ESCENARIO 2: Doble Techo NO CONFIRMADO",
        lookback=2,
        price_tolerance=0.03,
        min_candles=3
    )
    
    # Test 3: Picos muy separados
    velas_3 = crear_escenario_3()
    ohlcv3, sig3_sin, sig3_con = analizar_escenario(
        velas_3, 
        "ESCENARIO 3: Picos Separados en Tiempo",
        lookback=2,
        price_tolerance=0.02,
        min_candles=5
    )
    
    print(f"\n{'='*70}")
    print("  RESUMEN FINAL")
    print(f"{'='*70}")
    print(f"Escenario 1: {np.sum(sig1_con == -1)} señal(es) detectada(s)")
    print(f"Escenario 2: {np.sum(sig2_con == -1)} señal(es) detectada(s)")
    print(f"Escenario 3: {np.sum(sig3_con == -1)} señal(es) detectada(s)")
    print(f"{'='*70}\n")
    
    print("✅ Tests completados. Revisa los gráficos para validar visualmente.")
    print("\n💡 NOTA IMPORTANTE sobre backtest=True:")
    print("   La señal se desplaza +1 vela para simular ejecución realista.")
    print("   Esto evita 'mirar al futuro' durante el backtesting.")