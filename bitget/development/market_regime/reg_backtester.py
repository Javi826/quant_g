"""
market_regime/reg_backtester.py

Compara el rendimiento de una estrategia con y sin los filtros del Mayordomo.
Lee los trades enriquecidos generados por run_analysis.py.

Uso:
    %runfile reg_backtester.py
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List

# Añadir el directorio padre al path para imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_enriched_trades(strategy_name: str, output_folder: str = None) -> pd.DataFrame:
    """Carga el archivo de trades enriquecidos."""
    if output_folder is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_folder = os.path.join(base_dir, 'market_regime', 'output')
    
    filepath = os.path.join(output_folder, f'trades_enriched_{strategy_name}.xlsx')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró: {filepath}\nEjecuta primero run_analysis.py")
    
    return pd.read_excel(filepath)


def load_profile(strategy_name: str, output_folder: str = None) -> pd.DataFrame:
    """Carga el perfil de la estrategia."""
    if output_folder is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        output_folder = os.path.join(base_dir, 'market_regime', 'output')
    
    filepath = os.path.join(output_folder, f'profile_{strategy_name}.xlsx')
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"No se encontró: {filepath}\nEjecuta primero run_analysis.py")
    
    return pd.read_excel(filepath)


def get_activation_rules(profile_df: pd.DataFrame) -> Dict[str, Tuple[str, float]]:
    """Extrae las reglas de activación del perfil."""
    rules = {}
    
    for _, row in profile_df.iterrows():
        metric = row['metric']
        op = row['threshold_op']
        val = row['threshold_val']
        
        if pd.notna(op) and pd.notna(val):
            rules[metric] = (op, float(val))
    
    return rules


def apply_filter(df: pd.DataFrame, rules: Dict[str, Tuple[str, float]]) -> pd.DataFrame:
    """Aplica las reglas de filtro al DataFrame."""
    mask = pd.Series([True] * len(df), index=df.index)
    
    for metric, (op, val) in rules.items():
        if metric not in df.columns:
            continue
        
        if op == '>':
            mask &= df[metric] > val
        elif op == '<':
            mask &= df[metric] < val
        elif op == '>=':
            mask &= df[metric] >= val
        elif op == '<=':
            mask &= df[metric] <= val
    
    return df[mask]


def calculate_metrics(df: pd.DataFrame) -> dict:
    """Calcula métricas de rendimiento de un conjunto de trades."""
    if len(df) == 0:
        return {
            'num_trades': 0,
            'profit_total': 0,
            'profit_per_trade': 0,
            'win_rate': 0,
            'profit_factor': 0,
        }
    
    profits = df['profit']
    wins = profits[profits > 0]
    losses = profits[profits <= 0]
    
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0.0001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    
    return {
        'num_trades': len(df),
        'profit_total': profits.sum(),
        'profit_per_trade': profits.mean(),
        'win_rate': len(wins) / len(df) * 100,
        'profit_factor': profit_factor,
    }


def test_all_combinations(strategy_name: str, output_folder: str = None) -> pd.DataFrame:
    """
    Prueba todas las combinaciones posibles de reglas (1, 2, 3, 4 métricas).
    """
    from itertools import combinations
    
    df = load_enriched_trades(strategy_name, output_folder)
    profile_df = load_profile(strategy_name, output_folder)
    all_rules = get_activation_rules(profile_df)
    
    metrics_list = list(all_rules.keys())
    results = []
    
    # Baseline sin filtro
    baseline = calculate_metrics(df)
    baseline['config'] = 'SIN FILTRO'
    baseline['rules'] = '-'
    baseline['pct_trades'] = 100.0
    results.append(baseline)
    
    # Probar todas las combinaciones de 1, 2, 3, 4 métricas
    for n in range(1, len(metrics_list) + 1):
        for combo in combinations(metrics_list, n):
            rules_subset = {m: all_rules[m] for m in combo}
            df_filtered = apply_filter(df, rules_subset)
            
            metrics = calculate_metrics(df_filtered)
            
            # Formatear nombre de reglas
            rules_str = ' & '.join([f"{m}{rules_subset[m][0]}{rules_subset[m][1]:.2f}" for m in combo])
            
            metrics['config'] = f"{n} regla(s)"
            metrics['rules'] = rules_str
            metrics['pct_trades'] = (metrics['num_trades'] / baseline['num_trades'] * 100) if baseline['num_trades'] > 0 else 0
            
            results.append(metrics)
    
    return pd.DataFrame(results)


def print_results_table(df_results: pd.DataFrame):
    """Imprime tabla de resultados formateada."""
    
    print("=" * 120)
    print("📊 COMPARATIVA DE TODAS LAS COMBINACIONES DE FILTROS")
    print("=" * 120)
    
    # Ordenar por profit_per_trade descendente
    df_sorted = df_results.sort_values('profit_per_trade', ascending=False)
    
    print(f"\n{'CONFIG':<12} {'TRADES':>8} {'%TRADES':>8} {'PROFIT':>10} {'P/TRADE':>10} {'WIN%':>8} {'PF':>8}  REGLAS")
    print("-" * 120)
    
    baseline_ppt = df_results[df_results['config'] == 'SIN FILTRO']['profit_per_trade'].values[0]
    
    for _, row in df_sorted.iterrows():
        ppt = row['profit_per_trade']
        pf = row['profit_factor']
        pf_str = f"{pf:.2f}" if pf != np.inf else "∞"
        
        # Marcar si mejora el baseline
        marker = "✅" if ppt > baseline_ppt and row['config'] != 'SIN FILTRO' else "  "
        
        print(f"{row['config']:<12} {row['num_trades']:>8.0f} {row['pct_trades']:>7.1f}% {row['profit_total']:>10.2f} {ppt:>10.4f} {row['win_rate']:>7.1f}% {pf_str:>8} {marker} {row['rules']}")
    
    print("-" * 120)
    
    # Mejor configuración
    best = df_sorted[df_sorted['config'] != 'SIN FILTRO'].iloc[0]
    baseline = df_results[df_results['config'] == 'SIN FILTRO'].iloc[0]
    
    print(f"\n🏆 MEJOR CONFIGURACIÓN:")
    print(f"   Reglas: {best['rules']}")
    print(f"   Profit/Trade: {best['profit_per_trade']:.4f} vs {baseline['profit_per_trade']:.4f} (baseline)")
    
    if best['profit_per_trade'] > baseline['profit_per_trade']:
        mejora = (best['profit_per_trade'] / baseline['profit_per_trade'] - 1) * 100
        print(f"   Mejora: +{mejora:.1f}%")
    else:
        print(f"   ⚠️  Ninguna combinación mejora el baseline")
    
    print("=" * 120)


# =============================================================================
# CONFIGURACIÓN - EDITA AQUÍ
# =============================================================================
STRATEGY = 'parity_long_4H'
# =============================================================================


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_folder = os.path.join(base_dir, 'market_regime', 'output')
    
    print(f"📁 Analizando: {STRATEGY}")
    print(f"📁 Datos: {output_folder}")
    
    df_results = test_all_combinations(STRATEGY, output_folder)
    print_results_table(df_results)