"""
market_regime/run_analysis.py

Script principal para ejecutar el análisis de régimen de una estrategia.
"""

import os
import sys
from pathlib import Path

# Añadir el directorio padre al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_regime.trade_analyzer import TradeAnalyzer
from market_regime.strategy_profiler import StrategyProfiler


def full_analysis(
    strategy_name: str,
    trades_folder: str = 'brief_trades',
    ohlc_folder: str = 'data/crypto_OOS',
    output_folder: str = 'market_regime/output',
    save_results: bool = True,
    verbose: bool = True
):
    """
    Ejecuta el análisis completo de una estrategia.
    """
    
    # Rutas
    trades_path = os.path.join(trades_folder, f'all_trades_{strategy_name}.xlsx')
    
    # Inferir timeframe del nombre
    parts = strategy_name.split('_')
    timeframe = parts[-1] if parts else '4H'
    
    if verbose:
        print("=" * 70)
        print(f"🔍 ANÁLISIS DE RÉGIMEN: {strategy_name}")
        print("=" * 70)
        print(f"\n📂 Trades: {trades_path}")
        print(f"📂 OHLC: {ohlc_folder}")
        print(f"⏱️  Timeframe: {timeframe}")
    
    # Paso 1: Analizar trades
    if verbose:
        print("\n" + "-" * 70)
        print("PASO 1: Calculando métricas de régimen por trade...")
        print("-" * 70)
    
    analyzer = TradeAnalyzer(
        trades_path=trades_path,
        ohlc_folder=ohlc_folder,
        timeframe=timeframe
    )
    
    df_enriched = analyzer.analyze(verbose=verbose)
    
    # Paso 2: Generar perfil
    if verbose:
        print("\n" + "-" * 70)
        print("PASO 2: Generando perfil de estrategia...")
        print("-" * 70)
    
    profiler = StrategyProfiler(df_enriched)
    profile = profiler.generate_profile()
    
    if verbose:
        print()
        profiler.print_report(profile)
    
    # Paso 3: Guardar resultados
    if save_results:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Guardar trades enriquecidos
        enriched_path = os.path.join(output_folder, f'trades_enriched_{strategy_name}.xlsx')
        df_enriched.to_excel(enriched_path, index=False)
        
        # Guardar perfil
        profile_path = os.path.join(output_folder, f'profile_{strategy_name}.xlsx')
        df_profile = profiler.to_dataframe(profile)
        df_profile.to_excel(profile_path, index=False)
        
        if verbose:
            print(f"\n💾 Resultados guardados en: {output_folder}/")
            print(f"   - trades_enriched_{strategy_name}.xlsx")
            print(f"   - profile_{strategy_name}.xlsx")
    
    return profile, df_enriched


# =============================================================================
# CONFIGURACIÓN - EDITA AQUÍ
# =============================================================================
STRATEGY = 'parity_long_4H'

# Directorio base (donde está el proyecto)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRADES_FOLDER = os.path.join(BASE_DIR, 'brief_trades')
OHLC_FOLDER = os.path.join(BASE_DIR, 'data', 'crypto_OOS')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'market_regime', 'output')
# =============================================================================


if __name__ == "__main__":
    print(f"📁 Base dir: {BASE_DIR}")
    print(f"📁 Trades folder: {TRADES_FOLDER}")
    print(f"📁 OHLC folder: {OHLC_FOLDER}")
    
    profile, df = full_analysis(
        strategy_name=STRATEGY,
        trades_folder=TRADES_FOLDER,
        ohlc_folder=OHLC_FOLDER,
        output_folder=OUTPUT_FOLDER
    )