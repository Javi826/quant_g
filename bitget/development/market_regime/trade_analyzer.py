"""
market_regime/trade_analyzer.py

Analiza trades históricos y les asocia métricas de régimen de mercado
en el momento de entrada (buy_time).

Uso:
    from market_regime.trade_analyzer import TradeAnalyzer
    
    analyzer = TradeAnalyzer(
        trades_path='brief_trades/all_trades_parity_long_4H.xlsx',
        ohlc_folder='data/crypto_OOS',
        timeframe='4H'
    )
    
    df_enriched = analyzer.analyze()
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from pathlib import Path

from .regime_metrics import calc_all_metrics


class TradeAnalyzer:
    """
    Analiza trades históricos y calcula métricas de régimen en el momento de entrada.
    """
    
    def __init__(
        self,
        trades_path: str,
        ohlc_folder: str,
        timeframe: str = '4H',
        lookback_bars: int = 100,
        hurst_window: int = 100,
        er_window: int = 14,
        atr_window: int = 14,
        pe_window: int = 50,
        pe_order: int = 3
    ):
        """
        Args:
            trades_path: Ruta al Excel de trades (brief_trades/all_trades_xxx.xlsx)
            ohlc_folder: Carpeta con los parquets OHLC (data/crypto_OOS)
            timeframe: Timeframe de los datos ('1H', '4H', '15m', etc.)
            lookback_bars: Barras hacia atrás para calcular métricas
            hurst_window: Ventana para Hurst
            er_window: Ventana para Efficiency Ratio
            atr_window: Ventana para ATR
            pe_window: Ventana para Permutation Entropy
            pe_order: Orden para Permutation Entropy
        """
        self.trades_path = Path(trades_path)
        self.ohlc_folder = Path(ohlc_folder)
        self.timeframe = timeframe
        self.lookback_bars = lookback_bars
        
        # Parámetros de métricas
        self.hurst_window = hurst_window
        self.er_window = er_window
        self.atr_window = atr_window
        self.pe_window = pe_window
        self.pe_order = pe_order
        
        # Cache de OHLC cargados
        self._ohlc_cache: Dict[str, pd.DataFrame] = {}
        
        # Extraer info del nombre del archivo
        self._parse_strategy_info()
    
    def _parse_strategy_info(self):
        """Extrae estrategia, dirección y timeframe del nombre del archivo."""
        # all_trades_parity_long_4H.xlsx → parity_long_4H
        filename = self.trades_path.stem  # sin extensión
        
        if filename.startswith('all_trades_'):
            strategy_full = filename.replace('all_trades_', '')
        else:
            strategy_full = filename
        
        self.strategy_full = strategy_full
        
        # Intentar parsear: generador_direction_timeframe
        parts = strategy_full.split('_')
        
        if len(parts) >= 3:
            # Último es timeframe, penúltimo es direction
            self.timeframe_from_name = parts[-1]
            self.direction = parts[-2]  # long/short
            self.generator = '_'.join(parts[:-2])  # todo lo demás
        elif len(parts) == 2:
            self.generator = parts[0]
            self.direction = parts[1]
            self.timeframe_from_name = self.timeframe
        else:
            self.generator = strategy_full
            self.direction = 'unknown'
            self.timeframe_from_name = self.timeframe
    
    def _load_ohlc(self, symbol: str) -> Optional[pd.DataFrame]:
        """Carga OHLC de un símbolo desde parquet (con cache)."""
        if symbol in self._ohlc_cache:
            return self._ohlc_cache[symbol]
        
        # Buscar archivo: {SYMBOL}_{TIMEFRAME}.parquet
        filename = f"{symbol}_{self.timeframe}.parquet"
        filepath = self.ohlc_folder / filename
        
        if not filepath.exists():
            # Intentar con timeframe del nombre del archivo
            filename = f"{symbol}_{self.timeframe_from_name}.parquet"
            filepath = self.ohlc_folder / filename
        
        if not filepath.exists():
            print(f"⚠️  OHLC no encontrado: {filepath}")
            return None
        
        df = pd.read_parquet(filepath)
        
        # Normalizar columnas
        df.columns = df.columns.str.lower()
        
        # Asegurar que timestamp es datetime
        if 'timestamp' in df.columns:
            df['ts'] = pd.to_datetime(df['timestamp'])
        elif 'ts' in df.columns:
            df['ts'] = pd.to_datetime(df['ts'])
        elif 'date' in df.columns:
            df['ts'] = pd.to_datetime(df['date'])
        elif 'time' in df.columns:
            df['ts'] = pd.to_datetime(df['time'])
        else:
            # Asumir que el índice es el timestamp
            df['ts'] = pd.to_datetime(df.index)
            df = df.reset_index(drop=True)
        
        df = df.sort_values('ts').reset_index(drop=True)
        
        self._ohlc_cache[symbol] = df
        return df
    
    def _get_ohlc_at_time(self, symbol: str, buy_time: pd.Timestamp) -> Optional[dict]:
        """
        Obtiene OHLC hasta el momento de entrada del trade.
        
        Returns:
            dict con arrays numpy de open, high, low, close (últimas lookback_bars)
        """
        df = self._load_ohlc(symbol)
        if df is None:
            return None
        
        # Encontrar el índice más cercano <= buy_time
        mask = df['ts'] <= buy_time
        if not mask.any():
            return None
        
        idx = mask.sum() - 1  # último índice donde ts <= buy_time
        
        # Necesitamos lookback_bars hacia atrás
        start_idx = max(0, idx - self.lookback_bars + 1)
        
        if idx - start_idx < 20:  # mínimo de datos necesarios
            return None
        
        subset = df.iloc[start_idx:idx + 1]
        
        return {
            'open': subset['open'].values.astype(np.float64),
            'high': subset['high'].values.astype(np.float64),
            'low': subset['low'].values.astype(np.float64),
            'close': subset['close'].values.astype(np.float64)
        }
    
    def _calc_metrics_for_trade(self, symbol: str, buy_time: pd.Timestamp) -> dict:
        """Calcula las 4 métricas de régimen para un trade específico."""
        ohlc = self._get_ohlc_at_time(symbol, buy_time)
        
        if ohlc is None:
            return {
                'hurst': np.nan,
                'efficiency_ratio': np.nan,
                'atr_pct': np.nan,
                'permutation_entropy': np.nan
            }
        
        return calc_all_metrics(
            ohlc,
            hurst_window=self.hurst_window,
            er_window=self.er_window,
            atr_window=self.atr_window,
            pe_window=self.pe_window,
            pe_order=self.pe_order
        )
    
    def load_trades(self) -> pd.DataFrame:
        """Carga el Excel de trades."""
        if not self.trades_path.exists():
            raise FileNotFoundError(f"No se encontró: {self.trades_path}")
        
        df = pd.read_excel(self.trades_path)
        
        # Normalizar nombres de columnas
        df.columns = df.columns.str.lower().str.strip()
        
        # Asegurar que buy_time es datetime
        if 'buy_time' in df.columns:
            df['buy_time'] = pd.to_datetime(df['buy_time'])
        elif 'buy time' in df.columns:
            df['buy_time'] = pd.to_datetime(df['buy time'])
        
        # Asegurar que sell_time es datetime
        if 'sell_time' in df.columns:
            df['sell_time'] = pd.to_datetime(df['sell_time'])
        elif 'sell time' in df.columns:
            df['sell_time'] = pd.to_datetime(df['sell time'])
        
        return df
    
    def analyze(self, verbose: bool = True) -> pd.DataFrame:
        """
        Analiza todos los trades y les asocia métricas de régimen.
        
        Returns:
            DataFrame con columnas adicionales: hurst, efficiency_ratio, atr_pct, permutation_entropy
        """
        df = self.load_trades()
        
        if verbose:
            print(f"📊 Analizando: {self.strategy_full}")
            print(f"   Generador: {self.generator}")
            print(f"   Dirección: {self.direction}")
            print(f"   Timeframe: {self.timeframe_from_name}")
            print(f"   Trades totales: {len(df)}")
            print()
        
        # Inicializar columnas de métricas
        metrics_cols = ['hurst', 'efficiency_ratio', 'atr_pct', 'permutation_entropy']
        for col in metrics_cols:
            df[col] = np.nan
        
        # Calcular métricas para cada trade
        symbols_processed = set()
        errors = 0
        
        for idx, row in df.iterrows():
            symbol = row['symbol']
            buy_time = row['buy_time']
            
            metrics = self._calc_metrics_for_trade(symbol, buy_time)
            
            for col in metrics_cols:
                df.at[idx, col] = metrics[col]
            
            symbols_processed.add(symbol)
            
            if np.isnan(metrics['hurst']):
                errors += 1
        
        if verbose:
            success = len(df) - errors
            print(f"✅ Trades procesados: {success}/{len(df)}")
            print(f"📈 Símbolos únicos: {len(symbols_processed)}")
            if errors > 0:
                print(f"⚠️  Trades sin datos OHLC: {errors}")
        
        # Agregar metadatos
        df['generator'] = self.generator
        df['direction'] = self.direction
        df['timeframe'] = self.timeframe_from_name
        
        return df
    
    def summary(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Genera resumen estadístico de las métricas por cuartiles de profit.
        
        Returns:
            DataFrame con estadísticas
        """
        if df is None:
            df = self.analyze(verbose=False)
        
        # Crear cuartiles de profit
        df['profit_quartile'] = pd.qcut(df['profit'], q=4, labels=['Q1_worst', 'Q2', 'Q3', 'Q4_best'])
        
        # Agregar por cuartil
        summary = df.groupby('profit_quartile').agg({
            'profit': ['mean', 'sum', 'count'],
            'hurst': 'mean',
            'efficiency_ratio': 'mean',
            'atr_pct': 'mean',
            'permutation_entropy': 'mean'
        }).round(4)
        
        return summary


def analyze_strategy(
    strategy_name: str,
    trades_folder: str = 'brief_trades',
    ohlc_folder: str = 'data/crypto_OOS',
    timeframe: str = None,
    output_path: str = None
) -> pd.DataFrame:
    """
    Función de conveniencia para analizar una estrategia.
    
    Args:
        strategy_name: Nombre de la estrategia (ej: 'parity_long_4H')
        trades_folder: Carpeta con los Excel de trades
        ohlc_folder: Carpeta con los parquets OHLC
        timeframe: Timeframe (si None, se infiere del nombre)
        output_path: Si se especifica, guarda el resultado
    
    Returns:
        DataFrame enriquecido con métricas de régimen
    """
    trades_path = os.path.join(trades_folder, f'all_trades_{strategy_name}.xlsx')
    
    # Inferir timeframe del nombre si no se especifica
    if timeframe is None:
        parts = strategy_name.split('_')
        timeframe = parts[-1] if parts else '4H'
    
    analyzer = TradeAnalyzer(
        trades_path=trades_path,
        ohlc_folder=ohlc_folder,
        timeframe=timeframe
    )
    
    df = analyzer.analyze()
    
    if output_path:
        df.to_excel(output_path, index=False)
        print(f"\n💾 Guardado en: {output_path}")
    
    return df


if __name__ == "__main__":
    # Ejemplo de uso
    print("=== Trade Analyzer ===")
    print("\nUso:")
    print("  from market_regime.trade_analyzer import analyze_strategy")
    print("  df = analyze_strategy('parity_long_4H')")
    print("\nO directamente:")
    print("  analyzer = TradeAnalyzer(")
    print("      trades_path='brief_trades/all_trades_parity_long_4H.xlsx',")
    print("      ohlc_folder='data/crypto_OOS',")
    print("      timeframe='4H'")
    print("  )")
    print("  df = analyzer.analyze()")
