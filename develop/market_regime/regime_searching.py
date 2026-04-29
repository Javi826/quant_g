#!/usr/bin/env python3
"""
develop/market_regime/regime01_performance_wr.py

Clone of regime01_performance.py with an additional win rate gap filter.

Extra filter logic:
    After flagging bins with profit < 0 (original logic),
    also flag bins with profit > 0 whose win rate is more than
    WR_GAP_THRESHOLD percentage points below the best valid bin win rate
    of that strategy.

    If WR_GAP_THRESHOLD = 0, output is identical to regime01_performance.py.

Usage:
    python regime01_performance_wr.py
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
TRADES_FOLDER = os.path.join(os.path.dirname(__file__), "..", "brief_trades")
TRADES_LABEL  = "is_baseline"  # "is_baseline" | "oos1_baseline" | "oos2_baseline" | "oos3_baseline"

SPLIT_MODE    = "expanding"
SPLIT_BASE    = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "data_pipeline", "data", "04_split", SPLIT_MODE)
BTC_FOLDER    = os.path.join(SPLIT_BASE, "IS", "crypto_full_IS")

BTC_MA_PERIOD = 5
LONG_TH       = 1.00
SHORT_TH      = 1.00

FAMILIES = {
    'trending': {'hurst': ('>', 0.55), 'efficiency_ratio': ('>', 0.4)},
    'volatile': {'atr_pct': ('>', 2.0), 'permutation_entropy': ('>', 0.2)},
    'ranging':  {}
}

HURST_WINDOW  = 100
ER_WINDOW     = 14
ATR_WINDOW    = 14
PE_WINDOW     = 50
PE_ORDER      = 3
LOOKBACK_BARS = 100
FAMILY_SOURCE = 'strategy'
ANALYSIS_MODE = 'combined'
INITIAL_CAPITAL = 800
MIN_TRADES      = 10

# Win rate gap filter — set to 0 to disable (identical to original)
WR_GAP_THRESHOLD = 1   # pp below best bin win rate to also filter positive-profit bins

# =============================================================================
# CACHE
# =============================================================================
_btc_cache = {}


# =============================================================================
# DATA LOADING
# =============================================================================
def load_btc_1d() -> pd.DataFrame:
    filepath = Path(BTC_FOLDER) / "BTCUSDT_1Dutc.parquet"
    if not filepath.exists():
        raise FileNotFoundError(f"BTC 1D file not found: {filepath}")
    df = pd.read_parquet(filepath)
    df.columns = df.columns.str.lower()
    df['ts'] = pd.to_datetime(df['timestamp'] if 'timestamp' in df.columns else df.index)
    return df.sort_values('ts').reset_index(drop=True)


def load_btc_for_family(timeframe: str) -> pd.DataFrame:
    if FAMILY_SOURCE == 'macro':
        return load_btc_1d()
    return load_btc_for_timeframe(BTC_FOLDER, timeframe, _btc_cache)


# =============================================================================
# TRADE CLASSIFICATION
# =============================================================================
def classify_trade(trade: pd.Series, btc_1d_df: pd.DataFrame, btc_family_df: pd.DataFrame) -> dict:
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
    if len(trades) == 0:
        return {'num_trades': 0, 'profit': 0.0, 'win_rate': 0.0, 'dd_pct': 0.0, 'profits_list': []}
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
    df        = load_trades(filepath)
    strategy  = df['strategy'].iloc[0]
    timeframe = extract_timeframe(df)
    btc_family_df = load_btc_for_family(timeframe)

    unknown_count = 0
    directions, families = [], []

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

    if ANALYSIS_MODE in ('family', 'combined'):
        df = df[df['family'] != 'unknown'].copy()
    if ANALYSIS_MODE in ('direction', 'combined'):
        df = df[df['direction'].isin(['uptrend', 'dwtrend'])].copy()

    all_families   = list(FAMILIES.keys())
    all_directions = ['uptrend', 'dwtrend']

    bins = {}
    if ANALYSIS_MODE == 'combined':
        for family in all_families:
            for direction in all_directions:
                key    = f"{family}_{direction}"
                subset = df[(df['family'] == family) & (df['direction'] == direction)]
                bins[key] = calc_bin_metrics(subset)

    # Total metrics
    df_sorted = df.sort_values('buy_time').reset_index(drop=True)
    df_sorted['equity'] = INITIAL_CAPITAL + df_sorted['profit'].cumsum()
    total_metrics = {
        'num_trades': len(df_sorted),
        'profit':     df_sorted['profit'].sum(),
        'win_rate':   (df_sorted['profit'] > 0).mean() * 100 if len(df_sorted) > 0 else 0.0,
        'dd_pct':     calculate_max_dd_pct(df_sorted['equity']),
    }

    # --- Original filter: profit < 0 ---
    filter_rules = []
    for bin_key, m in bins.items():
        if m['num_trades'] >= MIN_TRADES and m['profit'] < 0:
            filter_rules.append(bin_key)

    # --- Additional filter: positive profit but low win rate ---
    wr_filter_rules = []
    if WR_GAP_THRESHOLD > 0:
        valid_bins = {k: m for k, m in bins.items() if m['num_trades'] >= MIN_TRADES}
        if valid_bins:
            best_wr = max(m['win_rate'] for m in valid_bins.values())
            for bin_key, m in valid_bins.items():
                if bin_key not in filter_rules and m['profit'] > 0:
                    if best_wr - m['win_rate'] > WR_GAP_THRESHOLD:
                        wr_filter_rules.append(bin_key)

    all_filter_rules = filter_rules + wr_filter_rules

    # Filtered metrics
    if all_filter_rules:
        df_filtered = df.copy()
        for bin_key in all_filter_rules:
            family, direction = bin_key.rsplit('_', 1)
            mask = (df_filtered['family'] == family) & (df_filtered['direction'] == direction)
            df_filtered = df_filtered[~mask]
    else:
        df_filtered = df.copy()

    df_filtered = df_filtered.sort_values('buy_time').reset_index(drop=True)
    df_filtered['equity'] = INITIAL_CAPITAL + df_filtered['profit'].cumsum()
    filtered_metrics = {
        'num_trades': len(df_filtered),
        'profit':     df_filtered['profit'].sum() if len(df_filtered) > 0 else 0.0,
        'dd_pct':     calculate_max_dd_pct(df_filtered['equity']) if len(df_filtered) > 0 else 0.0,
        'win_rate':   (df_filtered['profit'] > 0).mean() * 100 if len(df_filtered) > 0 else 0.0,
    }

    return {
        'strategy':        strategy,
        'filepath':        filepath,
        'timeframe':       timeframe,
        'bins':            bins,
        'total':           total_metrics,
        'filtered':        filtered_metrics,
        'filter_rules':    filter_rules,
        'wr_filter_rules': wr_filter_rules,
        'all_filter_rules':all_filter_rules,
    }


# =============================================================================
# PRINTING
# =============================================================================
def print_strategy_result(r: dict):
    t = r['total']
    print(f"\n\033[93m{'='*130}\033[0m")
    print(f"\033[93mSTRATEGY: {r['strategy']}  [{r['timeframe']}]  |  "
          f"trades={t['num_trades']}  profit=${t['profit']:.2f}  "
          f"dd={t['dd_pct']:.2f}%  wr={t['win_rate']:.1f}%\033[0m")
    print(f"\033[93m{'='*130}\033[0m")

    valid_wrs = [m['win_rate'] for m in r['bins'].values() if m['num_trades'] >= MIN_TRADES]
    best_wr   = max(valid_wrs) if valid_wrs else 0.0

    header = f"{'BIN':<30} {'CONF':>5} {'TRADES':>8} {'PROFIT':>12} {'WIN%':>8} {'DD%':>8} {'GAP_WR':>8} {'FILTER'}"
    print(f"\n{header}")
    print("-" * 100)

    sorted_bins = sorted(r['bins'].items(), key=lambda x: x[1]['profit'], reverse=True)
    for bin_key, m in sorted_bins:
        conf    = "✓" if m['num_trades'] >= MIN_TRADES else "✗"
        gap_wr  = f"{best_wr - m['win_rate']:+.1f}pp" if m['num_trades'] >= MIN_TRADES else "—"
        if bin_key in r['filter_rules']:
            flag = "🚫 FILTER"
        elif bin_key in r['wr_filter_rules']:
            flag = "⚠️  WR_FILTER"
        else:
            flag = ""
        print(f"{bin_key:<30} {conf:>5} {m['num_trades']:>8} {m['profit']:>12.2f} "
              f"{m['win_rate']:>7.1f}% {m['dd_pct']:>7.2f}% {gap_wr:>8} {flag}")

    print("-" * 100)
    print(f"{'TOTAL':<30} {'':>5} {t['num_trades']:>8} {t['profit']:>12.2f} "
          f"{t['win_rate']:>7.1f}% {t['dd_pct']:>7.2f}%  best_wr={best_wr:.1f}%")

    if r['all_filter_rules']:
        print(f"\n  → Profit filter : {', '.join(r['filter_rules']) if r['filter_rules'] else 'none'}")
        print(f"  → WR filter     : {', '.join(r['wr_filter_rules']) if r['wr_filter_rules'] else 'none'}")
    else:
        print(f"\n  → No bins to filter")


def print_summary(results: list):
    print(f"\n{'='*175}")
    print(f"SUMMARY — TOTAL vs FILTERED  (WR_GAP_THRESHOLD={WR_GAP_THRESHOLD}pp)")
    print(f"{'='*175}")
    header = (f"{'STRATEGY':<35} {'TR_TOT':>8} {'TR_FILT':>8} {'%TR_ELIM':>10} "
              f"{'WR_TOT':>8} {'WR_FILT':>8} "
              f"{'PF_TOT':>10} {'PF_FILT':>10} {'Δ_PROFIT':>10} {'FILTER RULES'}")
    print(f"\n{header}")
    print("-" * 175)

    sys_trades_total    = 0
    sys_profit_total    = 0.0
    sys_trades_filtered = 0
    sys_profit_filtered = 0.0

    for r in results:
        t           = r['total']
        f           = r['filtered']
        rules       = ', '.join(r['all_filter_rules']) if r['all_filter_rules'] else 'none'
        pct_tr_elim = round((1 - f['num_trades'] / t['num_trades']) * 100, 1) if t['num_trades'] > 0 else 0.0
        delta       = f['profit'] - t['profit']
        color = "\033[92m" if f['win_rate'] > t['win_rate'] else ""
        reset = "\033[0m" if color else ""
        print(f"{r['strategy']:<35} {t['num_trades']:>8} {f['num_trades']:>8} {pct_tr_elim:>9.1f}% "
              f"{t['win_rate']:>7.1f}% {color}{f['win_rate']:>7.1f}%{reset} "
              f"{t['profit']:>10.2f} {f['profit']:>10.2f} {delta:>+10.2f}   {rules}")
        sys_trades_total    += t['num_trades']
        sys_profit_total    += t['profit']
        sys_trades_filtered += f['num_trades']
        sys_profit_filtered += f['profit']

    sys_wr_tot  = sum(r['total']['win_rate']    * r['total']['num_trades']    for r in results) / sys_trades_total if sys_trades_total else 0
    sys_wr_filt = sum(r['filtered']['win_rate'] * r['filtered']['num_trades'] for r in results) / sys_trades_filtered if sys_trades_filtered else 0
    sys_pct_tr  = round((1 - sys_trades_filtered / sys_trades_total) * 100, 1) if sys_trades_total else 0
    sys_pct_pf  = round(sys_profit_filtered / sys_profit_total * 100, 1) if sys_profit_total else 0
    sys_delta = sys_profit_filtered - sys_profit_total
    print("-" * 175)
    print(f"{'SYSTEM TOTAL':<35} {sys_trades_total:>8} {sys_trades_filtered:>8} {sys_pct_tr:>9.1f}% "
          f"{sys_wr_tot:>7.1f}% {sys_wr_filt:>7.1f}% "
          f"{sys_profit_total:>10.2f} {sys_profit_filtered:>10.2f} {sys_delta:>+10.2f}")
    print("-" * 175)
    n_filtered = sum(1 for r in results if r['all_filter_rules'])
    print(f"\nStrategies with filter rules: {n_filtered}/{len(results)}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print("REGIME ANALYZER — profit filter + win rate gap filter")
    print("=" * 80)
    print(f"\n  Trades folder     : {TRADES_FOLDER}")
    print(f"  Trades label      : {TRADES_LABEL}")
    print(f"  BTC folder        : {BTC_FOLDER}")
    print(f"  Min trades        : {MIN_TRADES}")
    print(f"  WR gap threshold  : {WR_GAP_THRESHOLD}pp  {'(disabled)' if WR_GAP_THRESHOLD == 0 else ''}")

    print("\n  Loading BTC 1D data...")
    btc_1d_df = load_btc_1d()
    print(f"  {len(btc_1d_df)} daily bars loaded")

    files = sorted(glob(str(Path(TRADES_FOLDER) / f"trades_{TRADES_LABEL}_*.csv")))
    if not files:
        print(f"\n  No CSV files found for label '{TRADES_LABEL}'")
        return
    print(f"\n  Found {len(files)} strategy files")

    print("\n  Analyzing strategies...")
    results = []
    for filepath in files:
        result    = analyze_strategy(filepath, btc_1d_df)
        results.append(result)
        rules_str = ', '.join(result['all_filter_rules']) if result['all_filter_rules'] else 'none'
        print(f"   ✅ {result['strategy']}  →  filter: {rules_str}")

    for r in results:
        print_strategy_result(r)

    print_summary(results)

    print(f"\n{'='*80}")
    print(f"  ✓ = reliable bin (>= {MIN_TRADES} trades)")
    print(f"  ✗ = unreliable bin (< {MIN_TRADES} trades)")
    print(f"  🚫 FILTER     = profit < 0")
    print(f"  ⚠️  WR_FILTER  = profit > 0 but WR gap > {WR_GAP_THRESHOLD}pp vs best bin")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()