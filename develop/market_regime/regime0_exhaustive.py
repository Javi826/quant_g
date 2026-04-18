#!/usr/bin/env python3
"""
develop/market_regime/regime0_exhaustive.py

Find best LONG + SHORT BTC MA threshold combination by testing all pairs.
Tests all combinations (MA_TYPES x THRESHOLDS) on IS trades data.

Output: optimal LONG_TH + SHORT_TH to use in regime_unified_analyzer.py
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared", "shared_market_regime")))
from regime_common import get_btc_macro_direction

# =============================================================================
# CONFIGURATION
# =============================================================================

TRADES_FOLDER = os.path.join(os.path.dirname(__file__), "..", "brief_trades_22")

SPLIT_MODE      = "expanding"
SPLIT_BASE      = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "data_pipeline", "data", "04_split", SPLIT_MODE)
BTC_FOLDER      = os.path.join(SPLIT_BASE, "IS",  "crypto_2022-01_2026-04_IS")

MA_TYPES   = [5, 10, 20, 50]
THRESHOLDS = [0.95, 0.98, 1.00, 1.02, 1.05]


# =============================================================================
# DATA LOADING
# =============================================================================

def load_btc_1d() -> pd.DataFrame:
    """Load BTC 1D OHLC"""
    filepath = Path(BTC_FOLDER) / "BTCUSDT_1Dutc.parquet"
    if not filepath.exists():
        raise FileNotFoundError(f"BTC 1D file not found: {filepath}")

    df = pd.read_parquet(filepath)
    df.columns = df.columns.str.lower()
    df['ts'] = pd.to_datetime(df['timestamp'] if 'timestamp' in df.columns else df.index)
    return df.sort_values('ts').reset_index(drop=True)


def load_all_trades() -> pd.DataFrame:
    """Load all trades from CSV files"""
    files = sorted(glob(str(Path(TRADES_FOLDER) / "*.csv")))
    if not files:
        raise FileNotFoundError(f"No CSV files found in {TRADES_FOLDER}")

    all_trades = []
    for filepath in files:
        df = pd.read_csv(filepath)
        df.columns = df.columns.str.lower().str.strip()
        df['buy_time'] = pd.to_datetime(df['buy_time'])
        all_trades.append(df)

    combined = pd.concat(all_trades, ignore_index=True)
    return combined.sort_values('buy_time').reset_index(drop=True)


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_combination(
    df_trades: pd.DataFrame,
    btc_df: pd.DataFrame,
    ma_period: int,
    long_th: float,
    short_th: float,
) -> dict:
    """
    Evaluate a single MA + threshold combination.
    Uses get_btc_macro_direction — no lookahead bias.
    """
    long_profits_all      = []
    long_profits_filtered = []
    short_profits_all     = []
    short_profits_filtered = []

    for _, trade in df_trades.iterrows():
        profit        = trade['profit']
        position_type = trade['position_type']
        direction     = get_btc_macro_direction(btc_df, trade['buy_time'], ma_period, long_th, short_th)

        if position_type == 'LONG':
            long_profits_all.append(profit)
            if direction == 'uptrend':
                long_profits_filtered.append(profit)

        elif position_type == 'SHORT':
            short_profits_all.append(profit)
            if direction == 'downtrend':
                short_profits_filtered.append(profit)

    combined_profit = sum(long_profits_filtered) + sum(short_profits_filtered)

    return {
        'long_total_trades':    len(long_profits_all),
        'long_total_profit':    sum(long_profits_all),
        'long_filtered_trades': len(long_profits_filtered),
        'long_filtered_profit': sum(long_profits_filtered),
        'short_total_trades':    len(short_profits_all),
        'short_total_profit':    sum(short_profits_all),
        'short_filtered_trades': len(short_profits_filtered),
        'short_filtered_profit': sum(short_profits_filtered),
        'combined_profit':       combined_profit,
    }


# =============================================================================
# PRINTING
# =============================================================================

def print_combination_details(rank: int, combo: dict):
    """Print detailed results for a single combination"""
    long_rule  = f"MA{combo['ma_period']} * {combo['long_th']:.2f}"
    short_rule = f"MA{combo['ma_period']} * {combo['short_th']:.2f}"
    r          = combo['result']

    total_before = r['long_total_trades']    + r['short_total_trades']
    total_after  = r['long_filtered_trades'] + r['short_filtered_trades']
    profit_before = r['long_total_profit']    + r['short_total_profit']
    profit_after  = r['long_filtered_profit'] + r['short_filtered_profit']
    delta         = profit_after - profit_before

    print(f"\n#{rank} {'='*110}")
    print(f"  LONG:  BTC > {long_rule}")
    print(f"  SHORT: BTC < {short_rule}")
    print(f"{'='*110}")
    print(f"\n{'Direction':<10} {'TR_TOT':>8} {'TR_FILT':>8} {'PF_TOT':>12} {'PF_FILT':>12} {'Δ_PROFIT':>12}")
    print("-" * 70)
    print(f"{'LONG':<10} {r['long_total_trades']:>8} {r['long_filtered_trades']:>8} "
          f"{r['long_total_profit']:>12.2f} {r['long_filtered_profit']:>12.2f} "
          f"{r['long_filtered_profit'] - r['long_total_profit']:>+12.2f}")
    print(f"{'SHORT':<10} {r['short_total_trades']:>8} {r['short_filtered_trades']:>8} "
          f"{r['short_total_profit']:>12.2f} {r['short_filtered_profit']:>12.2f} "
          f"{r['short_filtered_profit'] - r['short_total_profit']:>+12.2f}")
    print("-" * 70)
    print(f"{'TOTAL':<10} {total_before:>8} {total_after:>8} "
          f"{profit_before:>12.2f} {profit_after:>12.2f} {delta:>+12.2f}")
    print(f"\n  Combined profit after filter: ${r['combined_profit']:,.2f}")


def print_summary_table(combinations: list):
    """Print summary table of top combinations"""
    print(f"\n{'='*110}")
    print("SUMMARY — TOP 5 COMBINATIONS")
    print(f"{'='*110}")
    print(f"\n{'#':>3} {'MA':>6} {'LONG_TH':>10} {'SHORT_TH':>10} {'TR_FILT':>10} {'PF_FILT':>14} {'Δ_PROFIT':>12}")
    print("-" * 110)

    for rank, combo in enumerate(combinations[:5], 1):
        r = combo['result']
        total_trades  = r['long_filtered_trades'] + r['short_filtered_trades']
        total_profit  = r['combined_profit']
        profit_before = r['long_total_profit'] + r['short_total_profit']
        delta         = total_profit - profit_before
        print(f"{rank:>3} {'MA'+str(combo['ma_period']):>6} {combo['long_th']:>10.2f} {combo['short_th']:>10.2f} "
              f"{total_trades:>10} {total_profit:>14.2f} {delta:>+12.2f}")

    print("-" * 110)


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("REGIME0 EXHAUSTIVE — Find best MA + threshold combination")
    print("=" * 80)

    print(f"\nConfiguration:")
    print(f"  Trades folder: {TRADES_FOLDER}")
    print(f"  BTC folder   : {BTC_FOLDER}")
    print(f"  MA types     : {MA_TYPES}")
    print(f"  Thresholds   : {THRESHOLDS}")

    print("\n📂 Loading BTC 1D data...")
    btc_df = load_btc_1d()
    print(f"✅ {len(btc_df)} daily bars loaded")

    print("\n📂 Loading trades...")
    df_trades = load_all_trades()
    print(f"✅ {len(df_trades)} trades loaded")
    print(f"   LONG : {len(df_trades[df_trades['position_type'] == 'LONG'])}")
    print(f"   SHORT: {len(df_trades[df_trades['position_type'] == 'SHORT'])}")

    # Build all combinations
    combos = [
        (ma, long_th, short_th)
        for ma in MA_TYPES
        for long_th in THRESHOLDS
        for short_th in THRESHOLDS
    ]
    total = len(combos)
    print(f"\n🔍 Testing {total} combinations ({len(MA_TYPES)} MA × {len(THRESHOLDS)} LONG_TH × {len(THRESHOLDS)} SHORT_TH)...")

    results = []
    for i, (ma_period, long_th, short_th) in enumerate(combos, 1):
        print(f"   Progress: {i}/{total} ({i/total*100:.1f}%)...", end='\r')
        result = evaluate_combination(df_trades, btc_df, ma_period, long_th, short_th)
        results.append({
            'ma_period': ma_period,
            'long_th':   long_th,
            'short_th':  short_th,
            'result':    result,
        })

    print()

    # Sort by combined profit
    results = sorted(results, key=lambda x: x['result']['combined_profit'], reverse=True)

    print(f"\n{'='*80}")
    print("TOP 3 COMBINATIONS (by combined filtered profit)")
    print(f"{'='*80}")
    for rank, combo in enumerate(results[:3], 1):
        print_combination_details(rank, combo)

    print_summary_table(results)

    best = results[0]
    print(f"\n{'='*80}")
    print("BEST COMBINATION — use these values in regime_unified_analyzer.py")
    print(f"{'='*80}")
    print(f"\n  BTC_MA_PERIOD = {best['ma_period']}")
    print(f"  LONG_TH       = {best['long_th']}")
    print(f"  SHORT_TH      = {best['short_th']}")
    print(f"\n  Combined profit after filter: ${best['result']['combined_profit']:,.2f}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()