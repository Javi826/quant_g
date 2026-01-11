"""
market_regime/strategy_profiler.py

Analiza trades enriquecidos con métricas de régimen y genera perfiles óptimos.
Identifica en qué condiciones de mercado cada estrategia performa mejor.

Uso:
    from market_regime.strategy_profiler import StrategyProfiler
    
    profiler = StrategyProfiler(df_trades_enriched)
    profile = profiler.generate_profile()
    profiler.print_report()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class MetricProfile:
    """Perfil de una métrica para una estrategia."""
    metric_name: str
    best_quartile: str
    worst_quartile: str
    best_range: Tuple[float, float]
    worst_range: Tuple[float, float]
    profit_by_quartile: Dict[str, float]
    optimal_threshold: Optional[Tuple[str, float]]  # ('>', 0.55) o ('<', 0.45)
    correlation_with_profit: float


@dataclass 
class StrategyProfile:
    """Perfil completo de una estrategia."""
    strategy_name: str
    generator: str
    direction: str
    timeframe: str
    total_trades: int
    total_profit: float
    win_rate: float
    family: str  # mean_reversion, trend_follow, breakout, structural
    metrics: Dict[str, MetricProfile]
    activation_rules: Dict[str, Tuple[str, float]]  # {'hurst': ('>', 0.55), ...}


class StrategyProfiler:
    """
    Genera perfiles óptimos de estrategias basado en métricas de régimen.
    """
    
    METRIC_COLS = ['hurst', 'efficiency_ratio', 'atr_pct', 'permutation_entropy']
    
    FAMILY_RULES = {
        'trend_follow': {
            'hurst': ('>', 0.55),
            'efficiency_ratio': ('>', 0.5)
        },
        'mean_reversion': {
            'hurst': ('<', 0.45),
            'efficiency_ratio': ('<', 0.4)
        },
        'breakout': {
            'atr_pct': ('>', 'Q3'),  # percentil 75
            'permutation_entropy': ('>', 0.7)
        },
        'structural': {
            'permutation_entropy': ('<', 0.7),
            'atr_pct': ('<', 'Q2')  # percentil 50
        }
    }
    
    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame de trades enriquecido con métricas de régimen
        """
        self.df = df.copy()
        self._validate_data()
        self._extract_metadata()
    
    def _validate_data(self):
        """Valida que el DataFrame tenga las columnas necesarias."""
        required = ['profit', 'symbol', 'buy_time'] + self.METRIC_COLS
        missing = [col for col in required if col not in self.df.columns]
        
        if missing:
            raise ValueError(f"Columnas faltantes: {missing}")
        
        # Eliminar filas con métricas NaN
        initial_len = len(self.df)
        self.df = self.df.dropna(subset=self.METRIC_COLS)
        
        if len(self.df) < initial_len:
            dropped = initial_len - len(self.df)
            print(f"⚠️  Eliminados {dropped} trades sin métricas completas")
    
    def _extract_metadata(self):
        """Extrae metadatos de la estrategia."""
        if 'generator' in self.df.columns:
            self.generator = self.df['generator'].iloc[0]
        else:
            self.generator = 'unknown'
        
        if 'direction' in self.df.columns:
            self.direction = self.df['direction'].iloc[0]
        else:
            self.direction = 'unknown'
        
        if 'timeframe' in self.df.columns:
            self.timeframe = self.df['timeframe'].iloc[0]
        else:
            self.timeframe = 'unknown'
        
        self.strategy_name = f"{self.generator}_{self.direction}_{self.timeframe}"
    
    def _analyze_metric(self, metric_name: str) -> MetricProfile:
        """Analiza cómo una métrica correlaciona con el profit."""
        df = self.df.copy()
        
        # Crear cuartiles de la métrica
        try:
            df['metric_quartile'] = pd.qcut(
                df[metric_name], 
                q=4, 
                labels=['Q1', 'Q2', 'Q3', 'Q4'],
                duplicates='drop'
            )
        except ValueError:
            # Si hay muchos valores duplicados, usar cortes fijos
            df['metric_quartile'] = pd.cut(
                df[metric_name],
                bins=4,
                labels=['Q1', 'Q2', 'Q3', 'Q4']
            )
        
        # Profit promedio por cuartil
        profit_by_q = df.groupby('metric_quartile', observed=False)['profit'].mean().to_dict()
        
        # Encontrar mejor y peor cuartil
        if profit_by_q:
            best_q = max(profit_by_q, key=profit_by_q.get)
            worst_q = min(profit_by_q, key=profit_by_q.get)
        else:
            best_q = worst_q = 'Q1'
        
        # Rangos de valores por cuartil
        ranges_by_q = df.groupby('metric_quartile', observed=False)[metric_name].agg(['min', 'max'])
        
        best_range = (
            float(ranges_by_q.loc[best_q, 'min']) if best_q in ranges_by_q.index else np.nan,
            float(ranges_by_q.loc[best_q, 'max']) if best_q in ranges_by_q.index else np.nan
        )
        worst_range = (
            float(ranges_by_q.loc[worst_q, 'min']) if worst_q in ranges_by_q.index else np.nan,
            float(ranges_by_q.loc[worst_q, 'max']) if worst_q in ranges_by_q.index else np.nan
        )
        
        # Correlación con profit
        correlation = df[metric_name].corr(df['profit'])
        
        # Determinar umbral óptimo
        optimal_threshold = self._find_optimal_threshold(df, metric_name, profit_by_q, best_q)
        
        return MetricProfile(
            metric_name=metric_name,
            best_quartile=best_q,
            worst_quartile=worst_q,
            best_range=best_range,
            worst_range=worst_range,
            profit_by_quartile=profit_by_q,
            optimal_threshold=optimal_threshold,
            correlation_with_profit=correlation
        )
    
    def _find_optimal_threshold(
        self, 
        df: pd.DataFrame, 
        metric_name: str,
        profit_by_q: dict,
        best_q: str
    ) -> Optional[Tuple[str, float]]:
        """Encuentra el umbral óptimo para activar la estrategia."""
        
        # Si Q4 es el mejor → queremos valores altos → threshold '>'
        # Si Q1 es el mejor → queremos valores bajos → threshold '<'
        
        if best_q == 'Q4':
            # Umbral = percentil 75
            threshold_value = float(df[metric_name].quantile(0.75))
            return ('>', round(threshold_value, 4))
        
        elif best_q == 'Q3':
            # Umbral = percentil 50
            threshold_value = float(df[metric_name].quantile(0.50))
            return ('>', round(threshold_value, 4))
        
        elif best_q == 'Q1':
            # Umbral = percentil 25
            threshold_value = float(df[metric_name].quantile(0.25))
            return ('<', round(threshold_value, 4))
        
        elif best_q == 'Q2':
            # Umbral = percentil 50
            threshold_value = float(df[metric_name].quantile(0.50))
            return ('<', round(threshold_value, 4))
        
        return None
    
    def _classify_family(self, metric_profiles: Dict[str, MetricProfile]) -> str:
        """Clasifica la estrategia en una familia basado en sus métricas óptimas."""
        
        hurst_profile = metric_profiles.get('hurst')
        er_profile = metric_profiles.get('efficiency_ratio')
        atr_profile = metric_profiles.get('atr_pct')
        pe_profile = metric_profiles.get('permutation_entropy')
        
        scores = {
            'trend_follow': 0,
            'mean_reversion': 0,
            'breakout': 0,
            'structural': 0
        }
        
        # Evaluar Hurst
        if hurst_profile and hurst_profile.optimal_threshold:
            op, val = hurst_profile.optimal_threshold
            if op == '>' and val >= 0.5:
                scores['trend_follow'] += 2
            elif op == '<' and val <= 0.5:
                scores['mean_reversion'] += 2
        
        # Evaluar Efficiency Ratio
        if er_profile and er_profile.optimal_threshold:
            op, val = er_profile.optimal_threshold
            if op == '>' and val >= 0.4:
                scores['trend_follow'] += 1
            elif op == '<' and val <= 0.4:
                scores['mean_reversion'] += 1
        
        # Evaluar ATR%
        if atr_profile and atr_profile.optimal_threshold:
            op, val = atr_profile.optimal_threshold
            median_atr = self.df['atr_pct'].median()
            if op == '>' and val >= median_atr:
                scores['breakout'] += 2
            elif op == '<' and val <= median_atr:
                scores['structural'] += 1
        
        # Evaluar Permutation Entropy
        if pe_profile and pe_profile.optimal_threshold:
            op, val = pe_profile.optimal_threshold
            if op == '<' and val <= 0.75:
                scores['structural'] += 2
            elif op == '>' and val >= 0.75:
                scores['breakout'] += 1
        
        # Retornar familia con mayor score
        best_family = max(scores, key=scores.get)
        
        # Si empate o scores muy bajos, clasificar como 'mixed'
        if scores[best_family] < 2:
            return 'mixed'
        
        return best_family
    
    def generate_profile(self) -> StrategyProfile:
        """Genera el perfil completo de la estrategia."""
        
        # Analizar cada métrica
        metric_profiles = {}
        for metric in self.METRIC_COLS:
            metric_profiles[metric] = self._analyze_metric(metric)
        
        # Clasificar familia
        family = self._classify_family(metric_profiles)
        
        # Extraer reglas de activación
        activation_rules = {}
        for metric, profile in metric_profiles.items():
            if profile.optimal_threshold:
                activation_rules[metric] = profile.optimal_threshold
        
        # Estadísticas generales
        total_profit = self.df['profit'].sum()
        win_rate = (self.df['profit'] > 0).mean()
        
        return StrategyProfile(
            strategy_name=self.strategy_name,
            generator=self.generator,
            direction=self.direction,
            timeframe=self.timeframe,
            total_trades=len(self.df),
            total_profit=total_profit,
            win_rate=win_rate,
            family=family,
            metrics=metric_profiles,
            activation_rules=activation_rules
        )
    
    def print_report(self, profile: StrategyProfile = None):
        """Imprime un reporte detallado del perfil."""
        if profile is None:
            profile = self.generate_profile()
        
        print("=" * 70)
        print(f"📊 PERFIL DE ESTRATEGIA: {profile.strategy_name}")
        print("=" * 70)
        
        print(f"\n🏷️  Familia detectada: {profile.family.upper()}")
        print(f"📈 Total trades: {profile.total_trades}")
        print(f"💰 Profit total: {profile.total_profit:.2f}")
        print(f"🎯 Win rate: {profile.win_rate:.1%}")
        
        print("\n" + "-" * 70)
        print("📐 ANÁLISIS POR MÉTRICA")
        print("-" * 70)
        
        for metric_name, mp in profile.metrics.items():
            print(f"\n▸ {metric_name.upper()}")
            print(f"  Mejor cuartil: {mp.best_quartile} (rango: {mp.best_range[0]:.4f} - {mp.best_range[1]:.4f})")
            print(f"  Peor cuartil: {mp.worst_quartile} (rango: {mp.worst_range[0]:.4f} - {mp.worst_range[1]:.4f})")
            print(f"  Correlación con profit: {mp.correlation_with_profit:.4f}")
            
            if mp.optimal_threshold:
                op, val = mp.optimal_threshold
                print(f"  ✅ Umbral óptimo: {metric_name} {op} {val}")
            
            print(f"  Profit por cuartil:")
            for q, p in sorted(mp.profit_by_quartile.items()):
                bar = "█" * int(max(0, p) / 10) if p > 0 else "░" * int(abs(min(0, p)) / 10)
                print(f"    {q}: {p:>8.2f} {bar}")
        
        print("\n" + "-" * 70)
        print("🎛️  REGLAS DE ACTIVACIÓN (para el Mayordomo)")
        print("-" * 70)
        
        if profile.activation_rules:
            print("\n```python")
            print(f"PERFIL_{profile.generator.upper()}_{profile.direction.upper()} = {{")
            for metric, (op, val) in profile.activation_rules.items():
                print(f"    '{metric}': ('{op}', {val}),")
            print("}")
            print("```")
        else:
            print("\n⚠️  No se encontraron reglas claras de activación")
        
        print("\n" + "=" * 70)
    
    def to_dataframe(self, profile: StrategyProfile = None) -> pd.DataFrame:
        """Convierte el perfil a DataFrame para exportar."""
        if profile is None:
            profile = self.generate_profile()
        
        rows = []
        for metric_name, mp in profile.metrics.items():
            row = {
                'strategy': profile.strategy_name,
                'generator': profile.generator,
                'direction': profile.direction,
                'timeframe': profile.timeframe,
                'family': profile.family,
                'metric': metric_name,
                'best_quartile': mp.best_quartile,
                'worst_quartile': mp.worst_quartile,
                'best_range_min': mp.best_range[0],
                'best_range_max': mp.best_range[1],
                'correlation': mp.correlation_with_profit,
                'threshold_op': mp.optimal_threshold[0] if mp.optimal_threshold else None,
                'threshold_val': mp.optimal_threshold[1] if mp.optimal_threshold else None,
                'profit_Q1': mp.profit_by_quartile.get('Q1', np.nan),
                'profit_Q2': mp.profit_by_quartile.get('Q2', np.nan),
                'profit_Q3': mp.profit_by_quartile.get('Q3', np.nan),
                'profit_Q4': mp.profit_by_quartile.get('Q4', np.nan),
            }
            rows.append(row)
        
        return pd.DataFrame(rows)


def profile_strategy(df_enriched: pd.DataFrame, verbose: bool = True) -> StrategyProfile:
    """
    Función de conveniencia para generar perfil de una estrategia.
    
    Args:
        df_enriched: DataFrame de trades con métricas de régimen
        verbose: Si True, imprime el reporte
    
    Returns:
        StrategyProfile con toda la información
    """
    profiler = StrategyProfiler(df_enriched)
    profile = profiler.generate_profile()
    
    if verbose:
        profiler.print_report(profile)
    
    return profile


if __name__ == "__main__":
    print("=== Strategy Profiler ===")
    print("\nUso:")
    print("  from market_regime.strategy_profiler import profile_strategy")
    print("  profile = profile_strategy(df_enriched)")
    print("\nO con más control:")
    print("  profiler = StrategyProfiler(df_enriched)")
    print("  profile = profiler.generate_profile()")
    print("  profiler.print_report(profile)")
    print("  df_export = profiler.to_dataframe(profile)")
