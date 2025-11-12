"""
Script para comparar señales entre dos funciones de detección en BTCUSDT
Compara detect_parity_reversal_long vs trend_reversal_entry_long
"""

import os
import numpy as np
import pandas as pd
from Z_add_signals_parity import detect_parity_reversal_long
from Z_add_signals_reversal import trend_reversal_entry_long


# ============================================================================
# FUNCIONES DE CARGA Y ANÁLISIS
# ============================================================================

def load_btc_data(data_folder, timeframe):
    """Carga datos de BTCUSDT"""
    file_path = os.path.join(data_folder, f"BTCUSDT_{timeframe}.parquet")
    
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No se encontró el archivo: {file_path}")
    
    df = pd.read_parquet(file_path)
    
    # Convertir a arrays numpy
    ohlcv_data = {
        'open': df['open'].values,
        'high': df['high'].values,
        'low': df['low'].values,
        'close': df['close'].values,
        'volume': df['volume'].values if 'volume' in df.columns else None
    }
    
    return ohlcv_data, df


def compare_signals_btc(arr, df, params):
    """Compara las señales de ambas funciones para BTCUSDT"""
    
    # Generar señales con ambas funciones
    signals_parity = detect_parity_reversal_long(
        arr, 
        lookback=params['lookback'],
        tolerance=params['tolerance'],
        live_trading=False
    )
    
    signals_trend = trend_reversal_entry_long(
        arr,
        left_lookback=params['lookback'],
        tolerance=params['tolerance'],
        live_trading=False
    )
    
    # Encontrar índices donde hay señales
    idx_parity = np.where(signals_parity == 1)[0]
    idx_trend = np.where(signals_trend == 1)[0]
    
    # Señales coincidentes (mismo índice)
    idx_coincident = np.intersect1d(idx_parity, idx_trend)
    
    # Señales únicas
    idx_only_parity = np.setdiff1d(idx_parity, idx_trend)
    idx_only_trend = np.setdiff1d(idx_trend, idx_parity)
    
    return {
        'signals_parity': signals_parity,
        'signals_trend': signals_trend,
        'idx_parity': idx_parity,
        'idx_trend': idx_trend,
        'idx_coincident': idx_coincident,
        'idx_only_parity': idx_only_parity,
        'idx_only_trend': idx_only_trend,
        'n_parity': len(idx_parity),
        'n_trend': len(idx_trend),
        'n_coincident': len(idx_coincident),
        'n_only_parity': len(idx_only_parity),
        'n_only_trend': len(idx_only_trend)
    }


def print_comparison_results(results, df, params):
    """Imprime resultados de la comparación para BTCUSDT"""
    print("=" * 80)
    print("📊 COMPARACIÓN DE SEÑALES: PARITY vs TREND REVERSAL - BTCUSDT")
    print("=" * 80)
    print(f"\n⚙️  PARÁMETROS:")
    print(f"   Lookback:  {params['lookback']}")
    print(f"   Tolerance: {params['tolerance']}%")
    print(f"   Total barras: {len(df)}")
    print(f"   Período: {df.index[0]} → {df.index[-1]}")
    
    print("\n" + "=" * 80)
    print("📈 RESUMEN DE SEÑALES")
    print("=" * 80)
    
    print(f"\n🔵 PARITY REVERSAL:")
    print(f"   Total señales:     {results['n_parity']:>6}")
    
    print(f"\n🟢 TREND REVERSAL:")
    print(f"   Total señales:     {results['n_trend']:>6}")
    
    print(f"\n🟣 COINCIDENTES (mismo punto):")
    print(f"   Total:             {results['n_coincident']:>6}")
    if results['n_parity'] > 0:
        print(f"   % sobre Parity:    {results['n_coincident']/results['n_parity']*100:.1f}%")
    if results['n_trend'] > 0:
        print(f"   % sobre Trend:     {results['n_coincident']/results['n_trend']*100:.1f}%")
    
    print(f"\n🔶 SEÑALES ÚNICAS:")
    print(f"   Solo Parity:       {results['n_only_parity']:>6}")
    print(f"   Solo Trend:        {results['n_only_trend']:>6}")
    
    # Listar todas las señales coincidentes
    if results['n_coincident'] > 0:
        print("\n" + "=" * 80)
        print(f"🟣 SEÑALES COINCIDENTES - DETALLE ({results['n_coincident']} puntos)")
        print("=" * 80)
        print(f"\n{'Índice':<8} {'Fecha':<20} {'Close':<12} {'High':<12} {'Low':<12}")
        print("-" * 80)
        for idx in results['idx_coincident']:
            date = df.index[idx]
            close = df.iloc[idx]['close']
            high = df.iloc[idx]['high']
            low = df.iloc[idx]['low']
            print(f"{idx:<8} {str(date):<20} {close:<12.2f} {high:<12.2f} {low:<12.2f}")
    else:
        print("\n⚠️  No hay señales coincidentes en los mismos puntos")
    
    # Listar señales solo de Parity
    if results['n_only_parity'] > 0:
        print("\n" + "=" * 80)
        print(f"🔵 SEÑALES SOLO PARITY ({results['n_only_parity']} puntos)")
        print("=" * 80)
        print(f"\n{'Índice':<8} {'Fecha':<20} {'Close':<12} {'High':<12} {'Low':<12}")
        print("-" * 80)
        for idx in results['idx_only_parity'][:20]:  # Mostrar primeras 20
            date = df.index[idx]
            close = df.iloc[idx]['close']
            high = df.iloc[idx]['high']
            low = df.iloc[idx]['low']
            print(f"{idx:<8} {str(date):<20} {close:<12.2f} {high:<12.2f} {low:<12.2f}")
        if results['n_only_parity'] > 20:
            print(f"\n... y {results['n_only_parity'] - 20} señales más")
    
    # Listar señales solo de Trend
    if results['n_only_trend'] > 0:
        print("\n" + "=" * 80)
        print(f"🟢 SEÑALES SOLO TREND ({results['n_only_trend']} puntos)")
        print("=" * 80)
        print(f"\n{'Índice':<8} {'Fecha':<20} {'Close':<12} {'High':<12} {'Low':<12}")
        print("-" * 80)
        for idx in results['idx_only_trend'][:20]:  # Mostrar primeras 20
            date = df.index[idx]
            close = df.iloc[idx]['close']
            high = df.iloc[idx]['high']
            low = df.iloc[idx]['low']
            print(f"{idx:<8} {str(date):<20} {close:<12.2f} {high:<12.2f} {low:<12.2f}")
        if results['n_only_trend'] > 20:
            print(f"\n... y {results['n_only_trend'] - 20} señales más")
    
    print("\n" + "=" * 80)


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Configuración
    DATA_FOLDER = "data/crypto_OOS"
    TIMEFRAME = '4H'
    
    # Parámetros para comparar
    PARAMS = {
        'lookback': 50,
        'tolerance': 20
    }
    
    print("🚀 Iniciando comparación de señales para BTCUSDT...\n")
    print(f"📁 Carpeta de datos: {DATA_FOLDER}")
    print(f"⏰ Timeframe: {TIMEFRAME}\n")
    
    # Cargar datos de BTCUSDT
    print("📥 Cargando datos de BTCUSDT...")
    try:
        ohlcv_data, df = load_btc_data(DATA_FOLDER, TIMEFRAME)
        print(f"✅ Datos cargados: {len(df)} barras\n")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return
    
    # Comparar señales
    print("🔄 Generando y comparando señales...")
    results = compare_signals_btc(ohlcv_data, df, PARAMS)
    
    # Imprimir resultados
    print_comparison_results(results, df, PARAMS)
    
    print("\n✅ Comparación completada")
    print("=" * 80)


if __name__ == "__main__":
    main()