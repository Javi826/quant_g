#!/usr/bin/env python3
"""
develop/market_regime/regime_threshold_search.py

Grid search over efficiency_ratio and atr_pct thresholds in FAMILIES.
For each (er_th, atr_th) combination, reclassifies all IS trades,
applies the regime filter, and computes % profit improvement vs baseline.

Output: summary table per strategy + system total row, sorted by system % improvement.

Easily extensible to OOS periods by adding entries to PERIODS config.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from itertools import product

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared", "shared_market_regime")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared")))
from regime_common import extract_timeframe, load_btc_for_timeframe, calc_all_metrics_at_time
from regime_common import classify_trade_by_family, load_trades, get_btc_macro_direction
from regime_common import build_direction_cache

# =============================================================================
# CONFIGURATION
# =============================================================================

SPLIT_MODE = "expanding"
SPLIT_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "data_pipeline", "data", "04_split", SPLIT_MODE)
BTC_FOLDER = os.path.join(SPLIT_BASE, "IS", "crypto_full_IS")

# Periods config — btc_folder is the same for all periods
PERIODS = {
    "IS": {
        "trades_folder": os.path.join(os.path.dirname(__file__), "..", "brief_trades"),
        "trades_label":  "is_baseline",
        "btc_folder":    BTC_FOLDER,
    },
    "OOS1": {
        "trades_folder": os.path.join(os.path.dirname(__file__), "..", "brief_trades"),
        "trades_label":  "oos1_baseline",
        "btc_folder":    BTC_FOLDER,
    },
    "OOS2": {
        "trades_folder": os.path.join(os.path.dirname(__file__), "..", "brief_trades"),
        "trades_label":  "oos2_baseline",
        "btc_folder":    BTC_FOLDER,
    },
    "OOS3": {
        "trades_folder": os.path.join(os.path.dirname(__file__), "..", "brief_trades"),
        "trades_label":  "oos3_baseline",
        "btc_folder":    BTC_FOLDER,
    },
}

# Periods to run individually + combined
PERIOD_KEYS = ["IS", "OOS1", "OOS2", "OOS3"]

# Fixed params
BTC_MA_PERIOD   = 5
LONG_TH         = 1.00
SHORT_TH        = 1.00
HURST_WINDOW    = 100
ER_WINDOW       = 14
ATR_WINDOW      = 14
PE_WINDOW       = 50
PE_ORDER        = 3
LOOKBACK_BARS   = 100
FAMILY_SOURCE   = 'strategy'
MIN_TRADES      = 2
INITIAL_CAPITAL = 800

# Grid search ranges
MA_PERIODS     = [2, 3, 4, 5]
ER_THRESHOLDS  = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
ATR_THRESHOLDS = [1.5, 2.0, 2.5]

# =============================================================================
# CACHE
# =============================================================================
_btc_cache = {}


# =============================================================================
# HELPERS
# =============================================================================

def build_families(er_th: float, atr_th: float) -> dict:
    return {
        'trending': {'hurst': ('>', 0.55), 'efficiency_ratio': ('>', er_th)},
        'volatile': {'atr_pct': ('>', atr_th), 'permutation_entropy': ('>', 0.2)},
        'ranging':  {}
    }


def load_btc_1d(btc_folder: str) -> pd.DataFrame:
    filepath = Path(btc_folder) / "BTCUSDT_1Dutc.parquet"
    if not filepath.exists():
        raise FileNotFoundError(f"BTC 1D not found: {filepath}")
    df = pd.read_parquet(filepath)
    df.columns = df.columns.str.lower()
    df['ts'] = pd.to_datetime(df['timestamp'] if 'timestamp' in df.columns else df.index)
    return df.sort_values('ts').reset_index(drop=True)


def classify_trades(df: pd.DataFrame, btc_1d_df: pd.DataFrame, btc_tf_df: pd.DataFrame,
                    direction_cache: dict, metrics_cache: dict, families: dict) -> pd.DataFrame:
    """Classify trades using precomputed caches — fast path."""
    buy_times = pd.to_datetime(df['buy_time'])

    directions = [direction_cache.get(t, 'unknown') for t in buy_times]

    families_list = []
    for t in buy_times:
        metrics = metrics_cache.get(t)
        if metrics is None:
            metrics = calc_all_metrics_at_time(
                btc_df=btc_tf_df, buy_time=t, lookback=LOOKBACK_BARS,
                hurst_window=HURST_WINDOW, er_window=ER_WINDOW,
                atr_window=ATR_WINDOW, pe_window=PE_WINDOW, pe_order=PE_ORDER,
            )
        families_list.append(classify_trade_by_family(metrics, families) if metrics else 'unknown')

    out = df.copy()
    out['direction'] = directions
    out['family']    = families_list
    return out


def apply_filter(df: pd.DataFrame, families: dict) -> tuple:
    """Apply regime filter. Returns (filtered_df, bins_to_filter)."""
    df_valid = df[
        (df['family'] != 'unknown') &
        (df['direction'].isin(['uptrend', 'dwtrend']))
    ].copy()

    bins_to_filter = set()
    for family in families:
        for direction in ['uptrend', 'dwtrend']:
            subset = df_valid[(df_valid['family'] == family) & (df_valid['direction'] == direction)]
            n      = len(subset)
            profit = subset['profit'].sum() if n > 0 else 0.0
            if n >= MIN_TRADES and profit < 0:
                bins_to_filter.add(f"{family}_{direction}")

    if bins_to_filter:
        mask = df_valid.apply(lambda r: f"{r['family']}_{r['direction']}" in bins_to_filter, axis=1)
        df_filtered = df[~df.index.isin(df_valid[mask].index)]
    else:
        df_filtered = df.copy()

    return df_filtered, bins_to_filter


def pct_improvement(profit_filtered: float, profit_baseline: float) -> float:
    if profit_baseline == 0:
        return 0.0
    return (profit_filtered - profit_baseline) / abs(profit_baseline) * 100


# =============================================================================
# PRECOMPUTE METRICS CACHE PER STRATEGY
# =============================================================================

def precompute_metrics_cache(btc_tf_df: pd.DataFrame) -> dict:
    """Precompute metrics for all BTC bars — reused across all grid combinations."""
    from shared_market_regime.regime_common import build_metrics_cache
    return build_metrics_cache(
        btc_df=btc_tf_df, lookback=LOOKBACK_BARS,
        hurst_window=HURST_WINDOW, er_window=ER_WINDOW,
        atr_window=ATR_WINDOW, pe_window=PE_WINDOW, pe_order=PE_ORDER,
    )


# =============================================================================
# MAIN GRID SEARCH
# =============================================================================

def run_grid_search(period_key: str = "IS"):
    cfg          = PERIODS[period_key]
    trades_folder = cfg["trades_folder"]
    trades_label  = cfg["trades_label"]
    btc_folder    = cfg["btc_folder"]

    print(f"\n{'='*100}")
    print(f"  REGIME THRESHOLD GRID SEARCH — {period_key}")
    print(f"  MA periods    : {MA_PERIODS}")
    print(f"  ER thresholds : {ER_THRESHOLDS}")
    print(f"  ATR thresholds: {ATR_THRESHOLDS}")
    print(f"  Total combos  : {len(MA_PERIODS) * len(ER_THRESHOLDS) * len(ATR_THRESHOLDS)}")
    print(f"{'='*100}\n")

    # Load BTC 1D once
    btc_1d_df = load_btc_1d(btc_folder)

    # Load all strategy trade files
    files = sorted(glob(str(Path(trades_folder) / f"trades_{trades_label}_*.csv")))
    if not files:
        print(f"No files found for label '{trades_label}' in {trades_folder}")
        return

    print(f"  Loading {len(files)} strategy files...")

    # Per-strategy data: trades + precomputed caches
    strategies = {}
    for filepath in files:
        df        = load_trades(filepath)
        strategy  = df['strategy'].iloc[0]
        timeframe = extract_timeframe(df)

        btc_tf_df = load_btc_for_timeframe(btc_folder, timeframe, _btc_cache) \
                    if FAMILY_SOURCE == 'strategy' else btc_1d_df

        buy_times       = pd.to_datetime(df['buy_time'])
        metrics_cache   = precompute_metrics_cache(btc_tf_df)
        strategies[strategy] = {
            'df':              df,
            'timeframe':       timeframe,
            'btc_tf_df':       btc_tf_df,
            'buy_times':       buy_times,
            'metrics_cache':   metrics_cache,
            'profit_baseline': df['profit'].sum(),
        }

    print(f"  Running grid search...\n")

    # Grid search
    results = []

    for ma_period, er_th, atr_th in product(MA_PERIODS, ER_THRESHOLDS, ATR_THRESHOLDS):
        families    = build_families(er_th, atr_th)
        row         = {'ma_period': ma_period, 'er_th': er_th, 'atr_th': atr_th}
        sys_baseline  = 0.0
        sys_filtered  = 0.0

        # Build direction cache per MA period (shared across strategies for same MA)
        direction_caches = {}
        for strategy, data in strategies.items():
            direction_caches[strategy] = build_direction_cache(
                btc_1d_df, ma_period, LONG_TH, SHORT_TH, data['buy_times']
            )

        for strategy, data in strategies.items():
            df_classified        = classify_trades(
                data['df'], btc_1d_df, data['btc_tf_df'],
                direction_caches[strategy], data['metrics_cache'], families
            )
            df_filtered, bins    = apply_filter(df_classified, families)
            profit_filtered      = df_filtered['profit'].sum()
            profit_baseline      = data['profit_baseline']
            pct_imp              = pct_improvement(profit_filtered, profit_baseline)

            row[f"{strategy}_baseline"] = round(profit_baseline, 2)
            row[f"{strategy}_filtered"] = round(profit_filtered, 2)
            row[f"{strategy}_pct"]      = round(pct_imp, 2)
            row[f"{strategy}_bins"]     = len(bins)

            sys_baseline += profit_baseline
            sys_filtered += profit_filtered

        row['sys_baseline'] = round(sys_baseline, 2)
        row['sys_filtered'] = round(sys_filtered, 2)
        row['sys_pct']      = round(pct_improvement(sys_filtered, sys_baseline), 2)
        results.append(row)

    df_results = pd.DataFrame(results).sort_values('sys_pct', ascending=False).reset_index(drop=True)

    # ==========================================================================
    # PRINT TOP 2 COMBINATIONS — one table each
    # ==========================================================================
    strategy_ids = list(strategies.keys())

    def print_combination_table(rank: int, row: pd.Series, strategies: dict, strategy_ids: list):
        ma_period = int(row['ma_period'])
        er_th     = row['er_th']
        atr_th    = row['atr_th']
        sys_pct   = row['sys_pct']

        print(f"\n{'='*100}")
        print(f"  #{rank} BEST — MA{ma_period} | ER>{er_th} | ATR>{atr_th}  →  system improvement: {sys_pct:+.2f}%")
        print(f"{'='*100}")
        print(f"  {'STRATEGY':<35} {'BASELINE':>10} {'FILTERED':>10} {'%_IMP':>8} {'BINS_FILTERED':>6}  {'BINS'}")
        print(f"  {'-'*100}")

        for s in strategy_ids:
            baseline = row[f'{s}_baseline']
            filtered = row[f'{s}_filtered']
            pct      = row[f'{s}_pct']
            n_bins   = int(row[f'{s}_bins'])
            bins_str = row.get(f'{s}_binnames', '—')
            color    = "\033[92m" if pct > 0 else "\033[91m" if pct < 0 else ""
            reset    = "\033[0m"
            print(f"  {s:<35} {baseline:>10.2f} {filtered:>10.2f} "
                  f"{color}{pct:>+7.2f}%{reset} {n_bins:>6}  {bins_str}")

        print(f"  {'-'*100}")
        sys_base = row['sys_baseline']
        sys_filt = row['sys_filtered']
        color    = "\033[92m" if sys_pct > 0 else "\033[91m"
        reset    = "\033[0m"
        print(f"  {'SYSTEM TOTAL':<35} {sys_base:>10.2f} {sys_filt:>10.2f} "
              f"{color}{sys_pct:>+7.2f}%{reset}")
        print(f"{'='*100}\n")

    # Run only best combination (#1)
    best_row = df_results.iloc[0].copy()
    ma_period = int(best_row['ma_period'])
    er_th     = best_row['er_th']
    atr_th    = best_row['atr_th']
    families  = build_families(er_th, atr_th)

    all_classified = []
    for strategy, data in strategies.items():
        direction_cache   = build_direction_cache(
            btc_1d_df, ma_period, LONG_TH, SHORT_TH, data['buy_times']
        )
        df_classified     = classify_trades(
            data['df'], btc_1d_df, data['btc_tf_df'],
            direction_cache, data['metrics_cache'], families
        )
        _, bins_to_filter = apply_filter(df_classified, families)
        best_row[f'{strategy}_binnames'] = ', '.join(sorted(bins_to_filter)) if bins_to_filter else '—'
        all_classified.append(df_classified)

    print_combination_table(1, best_row, strategies, strategy_ids)

    # --- Bin distribution table ---
    df_all   = pd.concat(all_classified, ignore_index=True)
    df_valid = df_all[
        (df_all['family'] != 'unknown') &
        (df_all['direction'].isin(['uptrend', 'dwtrend']))
    ]
    total_trades = len(df_all)

    system_bins_to_filter = set()
    for fam in families:
        for direction in ['uptrend', 'dwtrend']:
            subset = df_valid[(df_valid['family'] == fam) & (df_valid['direction'] == direction)]
            n      = len(subset)
            profit = subset['profit'].sum() if n > 0 else 0.0
            if n >= MIN_TRADES and profit < 0:
                system_bins_to_filter.add(f"{fam}_{direction}")

    print(f"\n{'='*80}")
    print(f"  BIN DISTRIBUTION — MA{ma_period} | ER>{er_th} | ATR>{atr_th}")
    print(f"  Total trades: {total_trades} | Valid (classified): {len(df_valid)}")
    print(f"{'='*80}")
    print(f"  {'BIN':<30} {'TRADES':>8} {'%_SYSTEM':>10} {'PROFIT':>10} {'FILTER':>8}")
    print(f"  {'-'*70}")

    for bin_key in [f"{fam}_{d}" for fam in ['trending', 'ranging', 'volatile'] for d in ['uptrend', 'dwtrend']]:
        fam, direction = bin_key.rsplit('_', 1)
        subset  = df_valid[(df_valid['family'] == fam) & (df_valid['direction'] == direction)]
        n       = len(subset)
        pct     = n / total_trades * 100 if total_trades > 0 else 0.0
        profit  = subset['profit'].sum() if n > 0 else 0.0
        flag    = "🚫 FILTER" if bin_key in system_bins_to_filter else ""
        print(f"  {bin_key:<30} {n:>8} {pct:>9.1f}% {profit:>10.2f} {flag}")

    n_unknown = len(df_all[df_all['family'] == 'unknown'])
    n_neutral = len(df_all[df_all['direction'] == 'neutral'])
    print(f"  {'-'*70}")
    print(f"  {'neutral_direction':<30} {n_neutral:>8} {n_neutral/total_trades*100:>9.1f}%")
    print(f"  {'unknown_family':<30} {n_unknown:>8} {n_unknown/total_trades*100:>9.1f}%")
    print(f"  {'-'*70}")
    print(f"  {'TOTAL':<30} {total_trades:>8} {'100.0%':>10}")
    print(f"{'='*80}\n")

    return df_results


# =============================================================================
# COMBINED GRID SEARCH — all periods together
# =============================================================================

def run_grid_search_combined(period_keys: list):
    """Run grid search over all periods combined — trades from all periods pooled."""

    print(f"\n{'='*100}")
    print(f"  REGIME THRESHOLD GRID SEARCH — ALL PERIODS COMBINED ({' + '.join(period_keys)})")
    print(f"  MA periods    : {MA_PERIODS}")
    print(f"  ER thresholds : {ER_THRESHOLDS}")
    print(f"  ATR thresholds: {ATR_THRESHOLDS}")
    print(f"  Total combos  : {len(MA_PERIODS) * len(ER_THRESHOLDS) * len(ATR_THRESHOLDS)}")
    print(f"{'='*100}\n")

    btc_1d_df = load_btc_1d(BTC_FOLDER)

    # Load all strategies across all periods
    all_strategies = {}  # {strategy_id: {period: data}}

    for period_key in period_keys:
        cfg           = PERIODS[period_key]
        trades_folder = cfg["trades_folder"]
        trades_label  = cfg["trades_label"]
        btc_folder    = cfg["btc_folder"]

        files = sorted(glob(str(Path(trades_folder) / f"trades_{trades_label}_*.csv")))
        if not files:
            print(f"  No files for {period_key} — skipping")
            continue

        for filepath in files:
            df        = load_trades(filepath)
            strategy  = df['strategy'].iloc[0]
            timeframe = extract_timeframe(df)

            btc_tf_df     = load_btc_for_timeframe(btc_folder, timeframe, _btc_cache) \
                            if FAMILY_SOURCE == 'strategy' else btc_1d_df
            buy_times     = pd.to_datetime(df['buy_time'])
            metrics_cache = precompute_metrics_cache(btc_tf_df)

            key = f"{strategy}__{period_key}"
            all_strategies[key] = {
                'df':              df,
                'strategy':        strategy,
                'period':          period_key,
                'timeframe':       timeframe,
                'btc_tf_df':       btc_tf_df,
                'buy_times':       buy_times,
                'metrics_cache':   metrics_cache,
                'profit_baseline': df['profit'].sum(),
            }

    if not all_strategies:
        print("  No data loaded — aborting")
        return None

    print(f"  Running combined grid search ({len(all_strategies)} strategy-period pairs)...\n")

    results = []
    for ma_period, er_th, atr_th in product(MA_PERIODS, ER_THRESHOLDS, ATR_THRESHOLDS):
        families     = build_families(er_th, atr_th)
        row          = {'ma_period': ma_period, 'er_th': er_th, 'atr_th': atr_th}
        sys_baseline = 0.0
        sys_filtered = 0.0

        for key, data in all_strategies.items():
            direction_cache = build_direction_cache(
                btc_1d_df, ma_period, LONG_TH, SHORT_TH, data['buy_times']
            )
            df_classified   = classify_trades(
                data['df'], btc_1d_df, data['btc_tf_df'],
                direction_cache, data['metrics_cache'], families
            )
            df_filtered, _  = apply_filter(df_classified, families)

            sys_baseline += data['profit_baseline']
            sys_filtered += df_filtered['profit'].sum()

        row['sys_baseline'] = round(sys_baseline, 2)
        row['sys_filtered'] = round(sys_filtered, 2)
        row['sys_pct']      = round(pct_improvement(sys_filtered, sys_baseline), 2)
        results.append(row)

    df_results = pd.DataFrame(results).sort_values('sys_pct', ascending=False).reset_index(drop=True)

    # Print best combination
    best = df_results.iloc[0]
    print(f"\n{'='*80}")
    print(f"  COMBINED BEST — MA{int(best['ma_period'])} | ER>{best['er_th']} | ATR>{best['atr_th']}")
    print(f"  System improvement: {best['sys_pct']:+.2f}%  "
          f"(baseline={best['sys_baseline']:.2f} → filtered={best['sys_filtered']:.2f})")
    print(f"{'='*80}\n")

    return df_results


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    best_per_period = {}

    # --- Individual grid search per period ---
    for period_key in PERIOD_KEYS:
        df_results = run_grid_search(period_key)
        if df_results is not None and len(df_results) > 0:
            best = df_results.iloc[0]
            best_per_period[period_key] = {
                'ma_period': int(best['ma_period']),
                'er_th':     best['er_th'],
                'atr_th':    best['atr_th'],
                'sys_pct':   best['sys_pct'],
            }

    # --- Combined grid search (all periods together) ---
    df_results_all = run_grid_search_combined(PERIOD_KEYS)
    if df_results_all is not None and len(df_results_all) > 0:
        best_all = df_results_all.iloc[0]
        best_per_period["ALL"] = {
            'ma_period': int(best_all['ma_period']),
            'er_th':     best_all['er_th'],
            'atr_th':    best_all['atr_th'],
            'sys_pct':   best_all['sys_pct'],
        }

    # --- Stability summary table ---
    print(f"\n{'='*80}")
    print(f"  STABILITY SUMMARY — Best params per period")
    print(f"{'='*80}")
    print(f"  {'PERIOD':<10} {'MA':>5} {'ER_TH':>8} {'ATR_TH':>8} {'SYS_%':>8}")
    print(f"  {'-'*50}")
    for period, params in best_per_period.items():
        print(f"  {period:<10} {params['ma_period']:>5} {params['er_th']:>8.2f} "
              f"{params['atr_th']:>8.2f} {params['sys_pct']:>+7.2f}%")
    print(f"{'='*80}\n")