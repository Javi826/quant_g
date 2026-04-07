#!/usr/bin/env python3
"""
market_regime/regime_analyzer.py

Autonomous script that compares system performance with/without trend filtering.
Calculates BTC MAs on-the-fly - no pre-enrichment needed.

Usage:
    python regime_analyzer_STANDALONE.py
    
Parameters (edit at top of script):
    TRADES_FOLDER: Folder with all_trades_*.xlsx files
    BTC_FILE: Path to BTC 1D parquet
    MA_PERIOD: Moving average period for trend detection (5, 10, 20, 50, 200)
    INITIAL_CAPITAL: Capital per strategy (default 800)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

# =============================================================================
# CONFIGURATION - EDIT THESE PARAMETERS
# =============================================================================

TRADES_FOLDER   = '../brief_trades'
BTC_FILE        = '../data/crypto_2022_OOS/BTCUSDT_1Dutc.parquet'
MA_PERIOD       = 5  # Options: 5, 10, 20, 50, 200
LONG_TH         = 1.02  # Threshold for LONG: BTC > MA * LONG_TH
SHORT_TH        = 1.00  # Threshold for SHORT: BTC < MA * SHORT_TH
INITIAL_CAPITAL = 800

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_btc_1d(btc_file: str) -> pd.DataFrame:
    """Load BTC 1D data and calculate MA"""
    if not Path(btc_file).exists():
        raise FileNotFoundError(f"BTC file not found: {btc_file}")
    
    df = pd.read_parquet(btc_file)
    df.columns = df.columns.str.lower()
    
    if 'timestamp' in df.columns:
        df['ts'] = pd.to_datetime(df['timestamp'])
    else:
        df['ts'] = pd.to_datetime(df.index)
    
    df = df.sort_values('ts').reset_index(drop=True)
    
    # Calculate MA
    df[f'ma{MA_PERIOD}'] = df['close'].rolling(window=MA_PERIOD).mean()
    
    return df


def get_btc_value_at_trade(btc_df: pd.DataFrame, trade_time: pd.Timestamp) -> tuple:
    """Get BTC close and MA at trade time (only closed candles)"""
    closed_candles = btc_df[btc_df['ts'] < trade_time]
    
    if len(closed_candles) < MA_PERIOD:
        return None, None
    
    last_candle = closed_candles.iloc[-1]
    
    if pd.isna(last_candle[f'ma{MA_PERIOD}']):
        return None, None
    
    return last_candle['close'], last_candle[f'ma{MA_PERIOD}']


def detect_strategy_type(strategy_name: str) -> str:
    """Detect if strategy is LONG or SHORT based on name"""
    name_lower = strategy_name.lower()
    
    if '_long_' in name_lower or name_lower.endswith('_long'):
        return 'LONG'
    elif '_short_' in name_lower or name_lower.endswith('_short'):
        return 'SHORT'
    
    print(f"⚠️  Cannot detect type for '{strategy_name}', assuming LONG")
    return 'LONG'


def calculate_strategy_metrics(df: pd.DataFrame, initial_capital: float) -> dict:
    """Calculate key metrics for a strategy"""
    if len(df) == 0:
        return {
            'num_trades': 0,
            'total_profit': 0.0,
            'net_gain_pct': 0.0,
            'max_dd_pct': 0.0
        }
    
    df = df.sort_values('buy_time').copy()
    df['cumulative_profit'] = df['profit'].cumsum()
    df['balance'] = initial_capital + df['cumulative_profit']
    
    # Net gain
    final_balance = df['balance'].iloc[-1]
    net_gain_pct = (final_balance - initial_capital) / initial_capital * 100
    
    # Max DD
    cummax = df['balance'].cummax()
    drawdown_pct = ((df['balance'] - cummax) / cummax * 100)
    max_dd_pct = drawdown_pct.min()
    
    return {
        'num_trades': len(df),
        'total_profit': df['profit'].sum(),
        'net_gain_pct': net_gain_pct,
        'max_dd_pct': max_dd_pct
    }


def load_trades(filepath: str) -> pd.DataFrame:
    """Load trades from Excel file"""
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.lower().str.strip()
    
    if 'buy_time' in df.columns:
        df['buy_time'] = pd.to_datetime(df['buy_time'])
    else:
        raise ValueError(f"File {filepath} missing 'buy_time' column")
    
    return df


def classify_trades_by_trend(df: pd.DataFrame, btc_df: pd.DataFrame, strategy_type: str) -> pd.DataFrame:
    """Add trend classification to each trade based on BTC MA with asymmetric thresholds"""
    df['trend'] = 'unknown'
    
    for idx, trade in df.iterrows():
        btc_close, ma_value = get_btc_value_at_trade(btc_df, trade['buy_time'])
        
        if btc_close is not None and ma_value is not None:
            if strategy_type == 'LONG':
                # LONG: BTC > MA * LONG_TH
                df.at[idx, 'trend'] = 'uptrend' if btc_close > ma_value * LONG_TH else 'downtrend'
            else:  # SHORT
                # SHORT: BTC < MA * SHORT_TH
                df.at[idx, 'trend'] = 'downtrend' if btc_close < ma_value * SHORT_TH else 'uptrend'
    
    return df


def analyze_strategy(filepath: str, btc_df: pd.DataFrame, initial_capital: float) -> dict:
    """Analyze single strategy with and without trend filter"""
    strategy = Path(filepath).stem.replace('all_trades_', '')
    df = load_trades(filepath)
    
    # Detect strategy type
    strategy_type = detect_strategy_type(strategy)
    
    # Classify trades by trend (with asymmetric thresholds)
    df = classify_trades_by_trend(df, btc_df, strategy_type)
    
    # SCENARIO A: WITHOUT FILTER (all trades)
    metrics_without = calculate_strategy_metrics(df, initial_capital)
    
    # SCENARIO B: WITH FILTER (only matching trend)
    if strategy_type == 'LONG':
        df_filtered = df[df['trend'] == 'uptrend'].copy()
    else:  # SHORT
        df_filtered = df[df['trend'] == 'downtrend'].copy()
    
    metrics_with = calculate_strategy_metrics(df_filtered, initial_capital)
    
    return {
        'strategy': strategy,
        'type': strategy_type,
        'filepath': filepath,
        'without_filter': metrics_without,
        'with_filter': metrics_with
    }


def calculate_global_portfolio(results: list, btc_df: pd.DataFrame, initial_capital: float, use_filter: bool = False) -> dict:
    """Calculate global portfolio metrics"""
    all_trades = []
    
    for r in results:
        df = load_trades(r['filepath'])
        df = classify_trades_by_trend(df, btc_df, r['type'])
        
        if use_filter:
            if r['type'] == 'LONG':
                df = df[df['trend'] == 'uptrend'].copy()
            else:  # SHORT
                df = df[df['trend'] == 'downtrend'].copy()
        
        all_trades.append(df[['buy_time', 'profit']].copy())
    
    if not all_trades:
        return {'num_trades': 0, 'total_profit': 0.0, 'net_gain_pct': 0.0, 'max_dd_pct': 0.0}
    
    combined_trades = pd.concat(all_trades, ignore_index=True)
    combined_trades = combined_trades.sort_values('buy_time').reset_index(drop=True)
    
    if len(combined_trades) == 0:
        return {'num_trades': 0, 'total_profit': 0.0, 'net_gain_pct': 0.0, 'max_dd_pct': 0.0}
    
    total_capital = initial_capital * len(results)
    
    # Calculate equity curve
    combined_trades['cumulative_profit'] = combined_trades['profit'].cumsum()
    combined_trades['balance'] = total_capital + combined_trades['cumulative_profit']
    
    # Net gain
    final_balance = combined_trades['balance'].iloc[-1]
    net_gain_pct = (final_balance - total_capital) / total_capital * 100
    
    # Max DD
    cummax = combined_trades['balance'].cummax()
    drawdown_pct = ((combined_trades['balance'] - cummax) / cummax * 100)
    max_dd_pct = drawdown_pct.min()
    
    return {
        'num_trades': len(combined_trades),
        'total_profit': combined_trades['profit'].sum(),
        'net_gain_pct': net_gain_pct,
        'max_dd_pct': max_dd_pct
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("REGIME ANALYZER - Trend Filtering Comparison (STANDALONE)")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  Trades folder: {TRADES_FOLDER}")
    print(f"  BTC file:      {BTC_FILE}")
    print(f"  MA period:     MA{MA_PERIOD}")
    print(f"  LONG TH:       {LONG_TH}")
    print(f"  SHORT TH:      {SHORT_TH}")
    print(f"  Capital:       ${INITIAL_CAPITAL}")
    
    print("\nComparison scenarios:")
    print("  WITHOUT FILTER: All trades")
    print(f"  WITH FILTER:    LONG when BTC > MA×{LONG_TH}, SHORT when BTC < MA×{SHORT_TH}")
    
    # Load BTC 1D
    print("\n📂 Loading BTC 1D data...")
    btc_df = load_btc_1d(BTC_FILE)
    print(f"✅ Loaded {len(btc_df)} daily bars")
    
    # Find all trades files
    pattern = str(Path(TRADES_FOLDER) / 'all_trades_*.xlsx')
    files = sorted(glob(pattern))
    
    if not files:
        print(f"\n❌ No trades files found in {TRADES_FOLDER}")
        return
    
    print(f"\n📂 Found {len(files)} strategy files")
    
    # Analyze each strategy
    print("\n🔍 Analyzing strategies...")
    results = []
    for filepath in files:
        result = analyze_strategy(filepath, btc_df, INITIAL_CAPITAL)
        results.append(result)
        print(f"   ✅ {result['strategy']}")
    
    # Calculate global portfolios
    global_without = calculate_global_portfolio(results, btc_df, INITIAL_CAPITAL, use_filter=False)
    global_with = calculate_global_portfolio(results, btc_df, INITIAL_CAPITAL, use_filter=True)
    
    # ==========================================================================
    # PRINT COMPARISON TABLE
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STRATEGY-BY-STRATEGY COMPARISON")
    print("=" * 80)
    
    comparison_rows = []
    
    for r in results:
        w = r['without_filter']
        f = r['with_filter']
        
        # Calculate % change in profit
        if w['total_profit'] != 0:
            profit_change_pct = ((f['total_profit'] - w['total_profit']) / abs(w['total_profit'])) * 100
        else:
            profit_change_pct = 0.0
        
        # Calculate % change in DD
        if w['max_dd_pct'] != 0:
            dd_change_pct = ((f['max_dd_pct'] - w['max_dd_pct']) / abs(w['max_dd_pct'])) * 100
        else:
            dd_change_pct = 0.0
        
        # Indicators
        profit_indicator = "✅" if profit_change_pct > 5 else ("❌" if profit_change_pct < -5 else "=")
        dd_indicator = "✅" if dd_change_pct > 5 else ("❌" if dd_change_pct < -5 else "=")
        
        comparison_rows.append({
            'Strategy': r['strategy'],
            'Type': r['type'],
            'ΔProfit%': profit_change_pct,
            'Profit': profit_indicator,
            'ΔDD%': dd_change_pct,
            'DD': dd_indicator
        })
    
    df_comp = pd.DataFrame(comparison_rows)
    
    # Format numeric columns
    df_comp['ΔProfit%'] = df_comp['ΔProfit%'].apply(lambda x: f"{x:+.1f}%")
    df_comp['ΔDD%'] = df_comp['ΔDD%'].apply(lambda x: f"{x:+.1f}%")
    
    print(df_comp.to_string(index=False))
    
    # ==========================================================================
    # GLOBAL SUMMARY TABLE
    # ==========================================================================
    print("\n" + "=" * 100)
    print("GLOBAL PORTFOLIO SUMMARY")
    print("=" * 100)
    
    print(f"\n{'Metric':<25} {'WITHOUT FILTER':>20} {'WITH FILTER':>20} {'CHANGE':>20}")
    print("-" * 100)
    
    # Trades
    trades_change = global_with['num_trades'] - global_without['num_trades']
    trades_change_pct = (trades_change / global_without['num_trades'] * 100) if global_without['num_trades'] > 0 else 0
    trades_without_str = f"{global_without['num_trades']:,}".replace(',', '.')
    trades_with_str = f"{global_with['num_trades']:,}".replace(',', '.')
    trades_change_str = f"{trades_change_pct:+.1f}".replace('.', ',')
    print(f"{'Trades':<25} {trades_without_str:>20} {trades_with_str:>20} {trades_change_str:>19}%")
    
    # Profit
    profit_change = global_with['total_profit'] - global_without['total_profit']
    profit_change_pct = (profit_change / abs(global_without['total_profit']) * 100) if global_without['total_profit'] != 0 else 0
    profit_without_str = f"{global_without['total_profit']:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    profit_with_str = f"{global_with['total_profit']:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    change_str = f"{profit_change_pct:+.1f}".replace('.', ',')
    print(f"{'Total Profit':<25} {profit_without_str:>20} {profit_with_str:>20} {change_str:>19}%")
    
    # Net Gain
    gain_change = global_with['net_gain_pct'] - global_without['net_gain_pct']
    gain_without_str = f"{global_without['net_gain_pct']:.2f}".replace('.', ',')
    gain_with_str = f"{global_with['net_gain_pct']:.2f}".replace('.', ',')
    gain_change_str = f"{gain_change:+.2f}".replace('.', ',')
    print(f"{'Net Gain %':<25} {gain_without_str:>19}% {gain_with_str:>19}% {gain_change_str:>19}%")
    
    # Max DD
    dd_change = global_with['max_dd_pct'] - global_without['max_dd_pct']
    dd_without_str = f"{global_without['max_dd_pct']:.2f}".replace('.', ',')
    dd_with_str = f"{global_with['max_dd_pct']:.2f}".replace('.', ',')
    dd_change_str = f"{dd_change:+.2f}".replace('.', ',')
    print(f"{'Max Drawdown %':<25} {dd_without_str:>19}% {dd_with_str:>19}% {dd_change_str:>19}%")
    
    print("-" * 100)
    
    # Improvement stats
    improvements = sum(1 for r in results if r['with_filter']['net_gain_pct'] > r['without_filter']['net_gain_pct'])
    print(f"\nStrategies improved: {improvements}/{len(results)} ({improvements/len(results)*100:.1f}%)")
    
    # ==========================================================================
    # RECOMMENDATION
    # ==========================================================================
    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)
    
    delta_global_gain = global_with['net_gain_pct'] - global_without['net_gain_pct']
    delta_global_dd = global_with['max_dd_pct'] - global_without['max_dd_pct']
    
    if delta_global_gain > 2.0 and (delta_global_dd > -1.0):
        print("\n✅ RECOMMEND USING TREND FILTER")
        print(f"   • Net Gain improves by {delta_global_gain:.2f}%")
        print(f"   • Max DD similar or better")
        print(f"   • {improvements} out of {len(results)} strategies improve")
    elif delta_global_gain < -2.0:
        print("\n❌ DO NOT USE TREND FILTER")
        print(f"   • Net Gain decreases by {abs(delta_global_gain):.2f}%")
        print(f"   • System performs better without filtering")
    else:
        print("\n⚠️  MARGINAL IMPACT")
        print(f"   • Net Gain change: {delta_global_gain:+.2f}%")
        print(f"   • Consider other factors (complexity, robustness, etc.)")
    
    print("=" * 100)


if __name__ == "__main__":
    main()