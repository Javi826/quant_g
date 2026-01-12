"""
market_regime/trade_analyzer.py

Analyzes historical trades and associates market regime metrics
at the moment of entry (buy_time).

METRIC SOURCE:
- use_own_symbol=False: Uses BTC as market proxy (all trades use BTC metrics)
- use_own_symbol=True: Uses each trade's own symbol for metrics

Usage:
    from market_regime.trade_analyzer import TradeAnalyzer
    
    analyzer = TradeAnalyzer(
        trades_path='brief_trades/all_trades_parity_long_4H.xlsx',
        ohlc_folder='data/crypto_OOS',
        timeframe='4H',
        use_own_symbol=True  # Use symbol's own metrics instead of BTC
    )
    
    df_enriched = analyzer.analyze()
"""

import os
import numpy as np
import pandas as pd
from typing import Optional, Dict
from pathlib import Path

from .regime_metrics import calc_all_metrics


# Market proxy symbol for regime calculation
MARKET_PROXY = 'BTCUSDT'


class TradeAnalyzer:
    """
    Analyzes historical trades and calculates regime metrics at entry time.
    Can use BTC as market proxy or each symbol's own data.
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
        pe_order: int = 3,
        market_proxy: str = MARKET_PROXY,
        use_own_symbol: bool = False
    ):
        """
        Args:
            trades_path: Path to trades Excel file (brief_trades/all_trades_xxx.xlsx)
            ohlc_folder: Folder with OHLC parquet files (data/crypto_OOS)
            timeframe: Data timeframe ('1H', '4H', '6H', etc.)
            lookback_bars: Bars to look back for metric calculation
            hurst_window: Window for Hurst exponent
            er_window: Window for Efficiency Ratio
            atr_window: Window for ATR
            pe_window: Window for Permutation Entropy
            pe_order: Order for Permutation Entropy
            market_proxy: Symbol to use as market proxy (default: BTCUSDT)
            use_own_symbol: If True, use each trade's own symbol for metrics instead of market_proxy
        """
        self.trades_path = Path(trades_path)
        self.ohlc_folder = Path(ohlc_folder)
        self.timeframe = timeframe
        self.lookback_bars = lookback_bars
        self.market_proxy = market_proxy
        self.use_own_symbol = use_own_symbol
        
        # Metric parameters
        self.hurst_window = hurst_window
        self.er_window = er_window
        self.atr_window = atr_window
        self.pe_window = pe_window
        self.pe_order = pe_order
        
        # OHLC cache
        self._ohlc_cache: Dict[str, pd.DataFrame] = {}
        
        # Parse strategy info from filename
        self._parse_strategy_info()
    
    def _parse_strategy_info(self):
        """Extracts strategy, direction and timeframe from filename."""
        # all_trades_parity_long_4H_OOS.xlsx → parity_long_4H_OOS
        filename = self.trades_path.stem
        
        if filename.startswith('all_trades_'):
            strategy_full = filename.replace('all_trades_', '')
        else:
            strategy_full = filename
        
        self.strategy_full = strategy_full
        
        # Parse: generator_direction_timeframe or generator_direction_timeframe_IS/OOS
        parts = strategy_full.split('_')
        
        # Check if last part is IS/OOS
        if parts[-1].upper() in ['IS', 'OOS']:
            self.data_type = parts[-1].upper()
            parts = parts[:-1]  # Remove IS/OOS for further parsing
        else:
            self.data_type = None
        
        if len(parts) >= 3:
            self.timeframe_from_name = parts[-1]
            self.direction = parts[-2]  # long/short
            self.generator = '_'.join(parts[:-2])
        elif len(parts) == 2:
            self.generator = parts[0]
            self.direction = parts[1]
            self.timeframe_from_name = self.timeframe
        else:
            self.generator = strategy_full
            self.direction = 'unknown'
            self.timeframe_from_name = self.timeframe
    
    def _load_ohlc(self, symbol: str) -> Optional[pd.DataFrame]:
        """Loads OHLC data from parquet file (with caching)."""
        cache_key = f"{symbol}_{self.timeframe}"
        
        if cache_key in self._ohlc_cache:
            return self._ohlc_cache[cache_key]
        
        # Try different filename patterns
        patterns = [
            f"{symbol}_{self.timeframe}.parquet",
            f"{symbol}_{self.timeframe_from_name}.parquet",
        ]
        
        filepath = None
        for pattern in patterns:
            candidate = self.ohlc_folder / pattern
            if candidate.exists():
                filepath = candidate
                break
        
        if filepath is None:
            print(f"⚠️  OHLC not found: {self.ohlc_folder}/{symbol}_{self.timeframe}.parquet")
            return None
        
        df = pd.read_parquet(filepath)
        
        # Normalize column names
        df.columns = df.columns.str.lower()
        
        # Ensure timestamp is datetime
        ts_columns = ['timestamp', 'ts', 'date', 'time']
        ts_col = None
        for col in ts_columns:
            if col in df.columns:
                ts_col = col
                break
        
        if ts_col:
            df['ts'] = pd.to_datetime(df[ts_col])
        else:
            df['ts'] = pd.to_datetime(df.index)
            df = df.reset_index(drop=True)
        
        df = df.sort_values('ts').reset_index(drop=True)
        
        # DIAGNOSTIC: Show BTC data range
        print(f"\n{'='*70}")
        print(f"📊 OHLC LOADED: {filepath}")
        print(f"{'='*70}")
        print(f"   Total bars: {len(df)}")
        print(f"   Date range: {df['ts'].min()} → {df['ts'].max()}")
        print(f"   Columns: {list(df.columns)}")
        print(f"{'='*70}\n")
        
        self._ohlc_cache[cache_key] = df
        return df
    
    def _get_ohlc_at_time(self, symbol: str, buy_time: pd.Timestamp, verbose_debug: bool = False) -> Optional[dict]:
        """
        Gets OHLC data up to the trade entry time.
        
        Args:
            symbol: Symbol to load OHLC for
            buy_time: Timestamp of trade entry
            verbose_debug: If True, print debug info
        
        REQUIRES EXACT MATCH on timestamp. If no exact match found, raises error.
        
        Returns:
            dict with numpy arrays of open, high, low, close (last lookback_bars)
        """
        df = self._load_ohlc(symbol)
        
        if df is None:
            return None
        
        # Find EXACT match on timestamp
        exact_match = df[df['ts'] == buy_time]
        
        if len(exact_match) == 0:
            # No exact match - this is an error
            data_min = df['ts'].min()
            data_max = df['ts'].max()
            raise ValueError(
                f"NO EXACT MATCH for buy_time={buy_time} in {symbol}\n"
                f"   {symbol} data range: {data_min} → {data_max}\n"
                f"   Make sure your OHLC folder contains data for this timestamp."
            )
        
        idx = exact_match.index[0]
        
        # Get lookback_bars backwards
        start_idx = max(0, idx - self.lookback_bars + 1)
        
        if idx - start_idx < 20:  # Minimum data required
            raise ValueError(
                f"INSUFFICIENT DATA for buy_time={buy_time} in {symbol}\n"
                f"   Only {idx - start_idx + 1} bars available, need at least 20.\n"
                f"   {symbol} data starts at: {df['ts'].min()}"
            )
        
        subset = df.iloc[start_idx:idx + 1]
        
        if verbose_debug:
            print(f"   ✅ EXACT MATCH: buy_time={buy_time}")
            print(f"      {symbol} bar used: {df.loc[idx, 'ts']} (idx={idx})")
            print(f"      Lookback: {len(subset)} bars from {subset['ts'].iloc[0]} to {subset['ts'].iloc[-1]}")
            print(f"      {symbol} close at entry: {subset['close'].iloc[-1]:.2f}")
        
        return {
            'open': subset['open'].values.astype(np.float64),
            'high': subset['high'].values.astype(np.float64),
            'low': subset['low'].values.astype(np.float64),
            'close': subset['close'].values.astype(np.float64)
        }
    
    def _calc_metrics_for_trade(self, symbol: str, buy_time: pd.Timestamp, verbose_debug: bool = False) -> dict:
        """Calculates the 4 regime metrics for a specific trade.
        
        Args:
            symbol: Symbol of the trade (used if use_own_symbol=True)
            buy_time: Timestamp of trade entry
            verbose_debug: If True, print debug info
        """
        # Choose which symbol to use for metrics
        metric_symbol = symbol if self.use_own_symbol else self.market_proxy
        
        ohlc = self._get_ohlc_at_time(metric_symbol, buy_time, verbose_debug=verbose_debug)
        
        if ohlc is None:
            return {
                'hurst': np.nan,
                'efficiency_ratio': np.nan,
                'atr_pct': np.nan,
                'permutation_entropy': np.nan
            }
        
        metrics = calc_all_metrics(
            ohlc,
            hurst_window=self.hurst_window,
            er_window=self.er_window,
            atr_window=self.atr_window,
            pe_window=self.pe_window,
            pe_order=self.pe_order
        )
        
        if verbose_debug:
            print(f"      Metrics: hurst={metrics['hurst']:.4f}, ER={metrics['efficiency_ratio']:.4f}, "
                  f"ATR%={metrics['atr_pct']:.4f}, PE={metrics['permutation_entropy']:.4f}")
        
        return metrics
    
    def load_trades(self) -> pd.DataFrame:
        """Loads trades from Excel file."""
        if not self.trades_path.exists():
            raise FileNotFoundError(f"File not found: {self.trades_path}")
        
        df = pd.read_excel(self.trades_path)
        
        # Normalize column names
        df.columns = df.columns.str.lower().str.strip()
        
        # Ensure buy_time is datetime
        if 'buy_time' in df.columns:
            df['buy_time'] = pd.to_datetime(df['buy_time'])
        elif 'buy time' in df.columns:
            df['buy_time'] = pd.to_datetime(df['buy time'])
        
        # Ensure sell_time is datetime
        if 'sell_time' in df.columns:
            df['sell_time'] = pd.to_datetime(df['sell_time'])
        elif 'sell time' in df.columns:
            df['sell_time'] = pd.to_datetime(df['sell time'])
        
        return df
    
    def analyze(self, verbose: bool = True) -> pd.DataFrame:
        """
        Analyzes all trades and associates regime metrics.
        
        Returns:
            DataFrame with additional columns: hurst, efficiency_ratio, atr_pct, permutation_entropy
        """
        df = self.load_trades()
        
        if verbose:
            print(f"📊 Analyzing: {self.strategy_full}")
            print(f"   Generator: {self.generator}")
            print(f"   Direction: {self.direction}")
            print(f"   Timeframe: {self.timeframe_from_name}")
            if self.use_own_symbol:
                print(f"   Metric Source: OWN SYMBOL (each trade uses its own symbol)")
            else:
                print(f"   Metric Source: {self.market_proxy} (market proxy)")
            print(f"   OHLC Folder: {self.ohlc_folder}")
            print(f"   Total trades: {len(df)}")
            print()
            
            # DIAGNOSTIC: Show trades date range
            print(f"{'='*70}")
            print(f"📅 TRADES DATE RANGE")
            print(f"{'='*70}")
            print(f"   First trade: {df['buy_time'].min()}")
            print(f"   Last trade:  {df['buy_time'].max()}")
            print(f"{'='*70}\n")
        
        # Initialize metric columns
        metrics_cols = ['hurst', 'efficiency_ratio', 'atr_pct', 'permutation_entropy']
        for col in metrics_cols:
            df[col] = np.nan
        
        # Calculate metrics for each trade
        symbols_in_trades = set()
        errors = 0
        matches = 0
        
        # Determine which trades to show detailed debug
        n_trades = len(df)
        debug_indices = set()
        if n_trades > 0:
            debug_indices.add(0)  # First trade
            debug_indices.add(1)  # Second trade
            debug_indices.add(2)  # Third trade
            debug_indices.add(n_trades - 2)  # Second to last
            debug_indices.add(n_trades - 1)  # Last trade
        debug_indices = {i for i in debug_indices if 0 <= i < n_trades}
        
        if verbose:
            print(f"{'='*70}")
            print(f"🔍 DETAILED DEBUG FOR SAMPLE TRADES")
            print(f"{'='*70}")
        
        for idx, row in df.iterrows():
            buy_time = row['buy_time']
            symbol = row.get('symbol', 'UNKNOWN')
            
            # Show detailed debug for first/last trades
            show_debug = verbose and (idx in debug_indices)
            
            if show_debug:
                metric_src = symbol if self.use_own_symbol else self.market_proxy
                print(f"\n▶ Trade #{idx}: {symbol} @ {buy_time} (metrics from: {metric_src})")
            
            # Calculate metrics
            metrics = self._calc_metrics_for_trade(symbol, buy_time, verbose_debug=show_debug)
            
            for col in metrics_cols:
                df.at[idx, col] = metrics[col]
            
            symbols_in_trades.add(symbol)
            
            if np.isnan(metrics['hurst']):
                errors += 1
            else:
                matches += 1
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"📊 ENRICHMENT SUMMARY")
            print(f"{'='*70}")
            print(f"   Total trades:        {len(df)}")
            print(f"   ✅ Matched: {matches}")
            print(f"   ❌ No data: {errors}")
            if self.use_own_symbol:
                print(f"   Regime source:       OWN SYMBOL")
            else:
                print(f"   Regime source:       {self.market_proxy}")
            print(f"   Unique symbols:      {len(symbols_in_trades)}")
            print(f"   Regime source:       {self.market_proxy}")
            
            # Show metrics distribution
            if matches > 0:
                print(f"\n   📈 Metrics distribution (matched trades only):")
                for col in metrics_cols:
                    valid = df[col].dropna()
                    if len(valid) > 0:
                        print(f"      {col:25s}: min={valid.min():.4f}, max={valid.max():.4f}, mean={valid.mean():.4f}")
            
            print(f"{'='*70}\n")
        
        # Add metadata
        df['generator'] = self.generator
        df['direction'] = self.direction
        df['timeframe'] = self.timeframe_from_name
        df['market_proxy'] = 'OWN_SYMBOL' if self.use_own_symbol else self.market_proxy

        
        return df
    
    def summary(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Generates statistical summary of metrics by profit quartiles.
        
        Returns:
            DataFrame with statistics
        """
        if df is None:
            df = self.analyze(verbose=False)
        
        # Create profit quartiles
        df['profit_quartile'] = pd.qcut(df['profit'], q=4, labels=['Q1_worst', 'Q2', 'Q3', 'Q4_best'])
        
        # Aggregate by quartile
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
    output_path: str = None,
    market_proxy: str = MARKET_PROXY
) -> pd.DataFrame:
    """
    Convenience function to analyze a strategy.
    
    Args:
        strategy_name: Strategy name (e.g., 'parity_long_4H')
        trades_folder: Folder with trades Excel files
        ohlc_folder: Folder with OHLC parquet files
        timeframe: Timeframe (if None, inferred from name)
        output_path: If specified, saves result to this path
        market_proxy: Symbol to use as market proxy (default: BTCUSDT)
    
    Returns:
        DataFrame enriched with regime metrics
    """
    trades_path = os.path.join(trades_folder, f'all_trades_{strategy_name}.xlsx')
    
    # Infer timeframe from name if not specified
    if timeframe is None:
        parts = strategy_name.split('_')
        # Handle IS/OOS suffix
        if parts[-1].upper() in ['IS', 'OOS']:
            timeframe = parts[-2] if len(parts) >= 2 else '4H'
        else:
            timeframe = parts[-1] if parts else '4H'
    
    analyzer = TradeAnalyzer(
        trades_path=trades_path,
        ohlc_folder=ohlc_folder,
        timeframe=timeframe,
        market_proxy=market_proxy
    )
    
    df = analyzer.analyze()
    
    if output_path:
        df.to_excel(output_path, index=False)
        print(f"\n💾 Saved to: {output_path}")
    
    return df


if __name__ == "__main__":
    print("=== Trade Analyzer ===")
    print(f"\nMarket Proxy: {MARKET_PROXY}")
    print("\nUsage:")
    print("  from market_regime.trade_analyzer import analyze_strategy")
    print("  df = analyze_strategy('parity_long_4H')")
    print("\nOr directly:")
    print("  analyzer = TradeAnalyzer(")
    print("      trades_path='brief_trades/all_trades_parity_long_4H.xlsx',")
    print("      ohlc_folder='data/crypto_OOS',")
    print("      timeframe='4H'")
    print("  )")
    print("  df = analyzer.analyze()")