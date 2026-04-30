#!/usr/bin/env python3
"""
develop/market_regime/regime01_performance.py

Unified regime analysis combining:
  - Macro BTC direction (regime0): uptrend / downtrend based on BTC 1D MA + thresholds
  - Market family (regime1): trending / volatile / ranging based on Hurst, ER, ATR, PE

Evaluates 6 cross combinations (family x direction) per strategy.
Automatically flags bins to filter: trades > MIN_TRADES and profit < 0.

Thresholds (LONG_TH, SHORT_TH) must be obtained from regime0_exhaustive.py first.

Usage:
    python regime_unified_analyzer.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared", "shared_market_regime")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared")))
from regime_common import extract_timeframe, load_btc_for_timeframe, calc_all_metrics_at_time
from regime_common import classify_trade_by_family, load_trades, calculate_max_dd_pct
from regime_common import permutation_test, format_significance, get_btc_macro_direction

# =============================================================================
# CONFIGURATION
# =============================================================================

TRADES_FOLDER   = os.path.join(os.path.dirname(__file__), "..", "brief_trades")
TRADES_LABEL  = "is_baseline" 

SPLIT_MODE      = "expanding"
SPLIT_BASE      = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "data_pipeline", "data", "04_split", SPLIT_MODE)
BTC_FOLDER      = os.path.join(SPLIT_BASE, "IS",  "crypto_full_IS")

# regime0 params — obtain from regime0_exhaustive.py
BTC_MA_PERIOD   = 2
LONG_TH         = 1.00
SHORT_TH        = 1.00

# regime1 params
FAMILIES = {
    'trending': {'hurst': ('>', 0.55), 'efficiency_ratio': ('>', 0.4)},
    'volatile': {'atr_pct': ('>', 2.0), 'permutation_entropy': ('>', 0.2)},
    'ranging':  {}
}

HURST_WINDOW    = 100
ER_WINDOW       = 14
ATR_WINDOW      = 14
PE_WINDOW       = 50
PE_ORDER        = 3
LOOKBACK_BARS   = 100

# Family source: 'strategy' = BTC at strategy timeframe | 'macro' = BTC 1D
FAMILY_SOURCE   = 'strategy'
#FAMILY_SOURCE   = 'macro'

# Analysis mode: 'family' = 3 bins | 'direction' = 2 bins | 'combined' = 6 bins
ANALYSIS_MODE   = 'combined'
#ANALYSIS_MODE   = 'family'

INITIAL_CAPITAL     = 800
MIN_TRADES          = 2   # minimum trades to trust a bin result

# =============================================================================
# CACHE
# =============================================================================

_btc_cache = {}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_btc_1d() -> pd.DataFrame:
    """Load BTC 1D OHLC for macro direction calculation"""
    filepath = Path(BTC_FOLDER) / "BTCUSDT_1Dutc.parquet"

    if not filepath.exists():
        raise FileNotFoundError(f"BTC 1D file not found: {filepath}")

    df = pd.read_parquet(filepath)
    df.columns = df.columns.str.lower()
    df['ts'] = pd.to_datetime(df['timestamp'] if 'timestamp' in df.columns else df.index)
    df = df.sort_values('ts').reset_index(drop=True)
    return df


def load_btc_for_family(timeframe: str) -> pd.DataFrame:
    """Load BTC OHLC for family metrics based on FAMILY_SOURCE setting"""
    if FAMILY_SOURCE == 'macro':
        return load_btc_1d()
    return load_btc_for_timeframe(BTC_FOLDER, timeframe, _btc_cache)


# =============================================================================
# TRADE CLASSIFICATION
# =============================================================================

def classify_trade(trade: pd.Series, btc_1d_df: pd.DataFrame, btc_family_df: pd.DataFrame) -> dict:
    """
    Classify a single trade by macro direction and family.
    Uses only closed candles — no lookahead bias.

    Returns dict with 'direction' and 'family' keys.
    """
    direction = get_btc_macro_direction(
        btc_1d_df  = btc_1d_df,
        trade_time = trade['buy_time'],
        ma_period  = BTC_MA_PERIOD,
        long_th    = LONG_TH,
        short_th   = SHORT_TH,
    )

    metrics = calc_all_metrics_at_time(
        btc_df       = btc_family_df,
        buy_time     = trade['buy_time'],
        lookback     = LOOKBACK_BARS,
        hurst_window = HURST_WINDOW,
        er_window    = ER_WINDOW,
        atr_window   = ATR_WINDOW,
        pe_window    = PE_WINDOW,
        pe_order     = PE_ORDER,
    )

    family = classify_trade_by_family(metrics, FAMILIES) if metrics else 'unknown'

    return {'direction': direction, 'family': family}


# =============================================================================
# BIN METRICS
# =============================================================================

def calc_bin_metrics(trades: pd.DataFrame) -> dict:
    """Calculate performance metrics for a subset of trades"""
    if len(trades) == 0:
        return {
            'num_trades': 0,
            'profit':     0.0,
            'win_rate':   0.0,
            'dd_pct':     0.0,
            'profits_list': [],
        }

    trades = trades.sort_values('buy_time').reset_index(drop=True)
    trades['equity'] = INITIAL_CAPITAL + trades['profit'].cumsum()

    return {
        'num_trades':   len(trades),
        'profit':       trades['profit'].sum(),
        'win_rate':     (trades['profit'] > 0).mean() * 100,
        'dd_pct':       calculate_max_dd_pct(trades['equity']),
        'profits_list': trades['profit'].tolist(),
    }


# =============================================================================
# STRATEGY ANALYSIS
# =============================================================================

def analyze_strategy(filepath: str, btc_1d_df: pd.DataFrame) -> dict:
    """
    Full regime analysis for a single strategy.
    Classifies each trade by direction x family and evaluates all 6 bins.
    """
    df         = load_trades(filepath)
    strategy   = df['strategy'].iloc[0]
    timeframe  = extract_timeframe(df)

    btc_family_df = load_btc_for_family(timeframe)

    unknown_count = 0

    directions = []
    families   = []
    
    first_trade = df.iloc[0]
    closed_1d = btc_1d_df[btc_1d_df['ts'] < first_trade['buy_time']]
    closed_tf = btc_family_df[btc_family_df['ts'] < first_trade['buy_time']]
    #print(f"DEBUG | buy_time={first_trade['buy_time']} | last 1D={closed_1d.iloc[-1]['ts']} | last {timeframe}={closed_tf.iloc[-1]['ts']}")

    for _, trade in df.iterrows():
        result = classify_trade(trade, btc_1d_df, btc_family_df)

        if result['direction'] == 'unknown':
            unknown_count += 1

        directions.append(result['direction'])
        families.append(result['family'])

    df['direction'] = directions
    df['family']    = families

    if unknown_count > 0:
        print(f"   ⚠️  {strategy}: {unknown_count} trades with unknown direction (excluded)")

    # Exclude based on mode: family needs valid family, direction needs valid direction
    if ANALYSIS_MODE in ('family', 'combined'):
        df = df[df['family'] != 'unknown'].copy()
    if ANALYSIS_MODE in ('direction', 'combined'):
        df = df[df['direction'].isin(['uptrend', 'dwtrend'])].copy()

    # Build bins according to ANALYSIS_MODE
    all_families   = list(FAMILIES.keys())
    all_directions = ['uptrend', 'dwtrend']

    bins = {}
    if ANALYSIS_MODE == 'combined':
        for family in all_families:
            for direction in all_directions:
                key    = f"{family}_{direction}"
                subset = df[(df['family'] == family) & (df['direction'] == direction)]
                bins[key] = calc_bin_metrics(subset)
    elif ANALYSIS_MODE == 'family':
        for family in all_families:
            bins[family] = calc_bin_metrics(df[df['family'] == family])
    elif ANALYSIS_MODE == 'direction':
        for direction in all_directions:
            bins[direction] = calc_bin_metrics(df[df['direction'] == direction])

    # Total metrics (all valid trades)
    df_sorted = df.sort_values('buy_time').reset_index(drop=True)
    df_sorted['equity'] = INITIAL_CAPITAL + df_sorted['profit'].cumsum()

    total_metrics = {
        'num_trades': len(df_sorted),
        'profit':     df_sorted['profit'].sum(),
        'win_rate':   (df_sorted['profit'] > 0).mean() * 100 if len(df_sorted) > 0 else 0.0,
        'dd_pct':     calculate_max_dd_pct(df_sorted['equity']),
    }

    # Auto-flag bins to filter: enough trades AND negative profit
    filter_rules = []
    for bin_key, bin_metrics in bins.items():
        if bin_metrics['num_trades'] >= MIN_TRADES and bin_metrics['profit'] < 0:
            filter_rules.append(bin_key)

    # Filtered metrics: trades that survive the filter
    if filter_rules:
        df_filtered = df.copy()
        for bin_key in filter_rules:
            if ANALYSIS_MODE == 'combined':
                family, direction = bin_key.rsplit('_', 1)
                mask = (df_filtered['family'] == family) & (df_filtered['direction'] == direction)
            elif ANALYSIS_MODE == 'family':
                mask = df_filtered['family'] == bin_key
            elif ANALYSIS_MODE == 'direction':
                mask = df_filtered['direction'] == bin_key
            df_filtered = df_filtered[~mask]
    else:
        df_filtered = df.copy()

    df_filtered = df_filtered.sort_values('buy_time').reset_index(drop=True)
    df_filtered['equity'] = INITIAL_CAPITAL + df_filtered['profit'].cumsum()

    filtered_metrics = {
        'num_trades': len(df_filtered),
        'profit':     df_filtered['profit'].sum() if len(df_filtered) > 0 else 0.0,
        'dd_pct':     calculate_max_dd_pct(df_filtered['equity']) if len(df_filtered) > 0 else 0.0,
    }

    return {
        'strategy':     strategy,
        'filepath':     filepath,
        'timeframe':    timeframe,
        'bins':         bins,
        'total':        total_metrics,
        'filtered':     filtered_metrics,
        'filter_rules': filter_rules,
    }


# =============================================================================
# PRINTING
# =============================================================================

def print_strategy_result(r: dict):
    """Print detailed bin table for a strategy"""
    t = r['total']
    print(f"\n\033[93m{'='*130}\033[0m")
    print(f"\033[93mSTRATEGY: {r['strategy']}  [{r['timeframe']}]  |  "
          f"trades={t['num_trades']}  profit=${t['profit']:.2f}  "
          f"dd={t['dd_pct']:.2f}%  wr={t['win_rate']:.1f}%\033[0m")
    print(f"\033[93m{'='*130}\033[0m")

    header = f"{'BIN':<30} {'CONF':>5} {'TRADES':>8} {'PROFIT':>12} {'WIN%':>8} {'DD%':>8} {'FILTER':>8}"
    print(f"\n{header}")
    print("-" * 90)

    sorted_bins = sorted(r['bins'].items(), key=lambda x: x[1]['profit'], reverse=True)

    for bin_key, m in sorted_bins:
        conf       = "✓" if m['num_trades'] >= MIN_TRADES else "✗"
        flag       = "🚫 FILTER" if bin_key in r['filter_rules'] else ""
        print(f"{bin_key:<30} {conf:>5} {m['num_trades']:>8} {m['profit']:>12.2f} "
              f"{m['win_rate']:>7.1f}% {m['dd_pct']:>7.2f}% {flag}")

    print("-" * 90)
    print(f"{'TOTAL':<30} {'':>5} {t['num_trades']:>8} {t['profit']:>12.2f} "
          f"{t['win_rate']:>7.1f}% {t['dd_pct']:>7.2f}%")

    if r['filter_rules']:
        print(f"\n  → Filter rules: {', '.join(r['filter_rules'])}")
    else:
        print(f"\n  → No bins to filter")


def print_summary(results: list):
    """Print summary table — one row per strategy with total vs filtered comparison"""
    print(f"\n{'='*150}")
    print("SUMMARY — TOTAL vs FILTERED PER STRATEGY")
    print(f"{'='*150}")

    header = (f"{'STRATEGY':<35} {'TR_TOT':>8} {'PF_TOT':>10} "
              f"{'TR_FILT':>8} {'PF_FILT':>10} {'Δ_PROFIT':>10} {'FILTER RULES'}")
    print(f"\n{header}")
    print("-" * 150)

    sys_trades_total    = 0
    sys_profit_total    = 0.0
    sys_trades_filtered = 0
    sys_profit_filtered = 0.0

    for r in results:
        t     = r['total']
        f     = r['filtered']
        delta = f['profit'] - t['profit']
        rules = ', '.join(r['filter_rules']) if r['filter_rules'] else 'none'

        delta_str = f"{delta:+.2f}"
        color     = "\033[92m" if delta > 0 else "\033[91m" if delta < 0 else ""
        reset     = "\033[0m" if color else ""

        print(f"{r['strategy']:<35} {t['num_trades']:>8} {t['profit']:>10.2f} "
              f"{f['num_trades']:>8} {f['profit']:>10.2f} "
              f"{color}{delta_str:>10}{reset}   {rules}")

        sys_trades_total    += t['num_trades']
        sys_profit_total    += t['profit']
        sys_trades_filtered += f['num_trades']
        sys_profit_filtered += f['profit']

    sys_delta = sys_profit_filtered - sys_profit_total
    sys_delta_str = f"{sys_delta:+.2f}"
    sys_color = "\033[92m" if sys_delta > 0 else "\033[91m" if sys_delta < 0 else ""
    reset = "\033[0m"

    print("-" * 150)
    print(f"{'SYSTEM TOTAL':<35} {sys_trades_total:>8} {sys_profit_total:>10.2f} "
          f"{sys_trades_filtered:>8} {sys_profit_filtered:>10.2f} "
          f"{sys_color}{sys_delta_str:>10}{reset}")
    print("-" * 150)

    n_filtered = sum(1 for r in results if r['filter_rules'])
    print(f"\nStrategies with filter rules: {n_filtered}/{len(results)}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("REGIME UNIFIED ANALYZER — direction x family per strategy")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Trades folder  : {TRADES_FOLDER}")
    print(f"  BTC folder     : {BTC_FOLDER}")
    print(f"  BTC MA period  : {BTC_MA_PERIOD}")
    print(f"  Long threshold : {LONG_TH}")
    print(f"  Short threshold: {SHORT_TH}")
    print(f"  Family source  : {FAMILY_SOURCE}")
    print(f"  Min trades     : {MIN_TRADES}")
    print(f"  Capital        : ${INITIAL_CAPITAL}")

    # Load BTC 1D (macro direction — always needed)
    print("\n📂 Loading BTC 1D data...")
    btc_1d_df = load_btc_1d()
    print(f"✅ {len(btc_1d_df)} daily bars loaded")

    # Find trades files
    files = sorted(glob(str(Path(TRADES_FOLDER) / f"trades_{TRADES_LABEL}_*.csv")))
    if not files:
        print(f"\n❌ No CSV files found in {TRADES_FOLDER}")
        return

    print(f"\n📂 Found {len(files)} strategy files")

    # Analyze each strategy
    print("\n🔍 Analyzing strategies...")
    results = []
    for filepath in files:
        result = analyze_strategy(filepath, btc_1d_df)
        results.append(result)
        rules_str = ', '.join(result['filter_rules']) if result['filter_rules'] else 'none'
        print(f"   ✅ {result['strategy']}  →  filter: {rules_str}")

    # Print detailed results
    for r in results:
        print_strategy_result(r)

    # Print summary
    print_summary(results)

    print(f"\n{'='*80}")
    print("INTERPRETATION:")
    print("  ✓ = reliable bin (>= 50 trades)")
    print("  ✗ = unreliable bin (< 50 trades)")
    print("  🚫 FILTER = bin flagged for filtering (trades >= 50 AND profit < 0)")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()