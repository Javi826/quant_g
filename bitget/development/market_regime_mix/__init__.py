"""
market_regime - Módulo para análisis y gestión de régimen de mercado

El Mayordomo: sistema que activa/desactiva estrategias según el régimen actual.

Módulos:
- regime_metrics: Cálculo de Hurst, ATR%, Efficiency Ratio, Permutation Entropy
- trade_analyzer: Asocia métricas de régimen a trades históricos
- strategy_profiler: Genera perfiles óptimos por estrategia
- butler: Activación/desactivación de bots en producción (TODO)
"""

from .regime_metrics import (
    calc_hurst,
    calc_efficiency_ratio,
    calc_atr_pct,
    calc_permutation_entropy,
    calc_all_metrics,
    classify_regime
)

from .trade_analyzer import (
    TradeAnalyzer,
    analyze_strategy
)

from .strategy_profiler import (
    StrategyProfiler,
    StrategyProfile,
    MetricProfile,
    profile_strategy
)

__all__ = [
    # Métricas
    'calc_hurst',
    'calc_efficiency_ratio', 
    'calc_atr_pct',
    'calc_permutation_entropy',
    'calc_all_metrics',
    'classify_regime',
    # Analyzer
    'TradeAnalyzer',
    'analyze_strategy',
    # Profiler
    'StrategyProfiler',
    'StrategyProfile',
    'MetricProfile',
    'profile_strategy',
]

__version__ = '0.1.0'
