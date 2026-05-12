"""
develop/market_regime/regime0_exhaustive.py
Find best LONG + SHORT BTC MA threshold combination by testing all pairs.
Tests all combinations (MA_TYPES x THRESHOLDS) on all baseline periods.
Shows results per period and aggregated.
"""
import os
import sys
import pandas as pd
from pathlib import Path
from glob import glob
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared")))
from shared.shared_batch_develop.market_regime.regime_analysis import get_macro_direction
# =============================================================================
# CONFIGURATION
# =============================================================================
REFERENCE_SYMBOL = 'QQQUSDT'
TRADES_FOLDER    = os.path.join(os.path.dirname(__file__), "..", "brief_trades")
SPLIT_MODE       = "expanding"
SPLIT_BASE       = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "data_pipeline", "data", "04_split", SPLIT_MODE)
#REF_FOLDER       = os.path.join(SPLIT_BASE, "IS", "crypto_full_IS")
REF_FOLDER      = os.path.join(SPLIT_BASE, "IS",  "rwa_2025-01_2026-03_IS")

THRESHOLDS       = [0.99,1.00,1.01]

MA_TYPES         = [2,3,4,5,6]
BTC_TIMEFRAME    = '1Dutc'

# =============================================================================
# MA_TYPES      = [30,60,120,300]
# BTC_TIMEFRAME = '4H'                            
# =============================================================================

PERIOD_LABELS = [
    ("IS",   "is_baseline"),
    ("OOS1", "oos1_baseline"),
    ("OOS2", "oos2_baseline"),
    ("OOS3", "oos3_baseline"),
]
TOP_N = 1


# =============================================================================
# DATA LOADING
# =============================================================================
def load_btc() -> pd.DataFrame:
    filepath = Path(REF_FOLDER) / f"{REFERENCE_SYMBOL}_{BTC_TIMEFRAME}.parquet"
    if not filepath.exists():
        raise FileNotFoundError(f"BTC file not found: {filepath}")
    df = pd.read_parquet(filepath)
    df.columns = df.columns.str.lower()
    df['ts'] = pd.to_datetime(df['timestamp'] if 'timestamp' in df.columns else df.index)
    return df.sort_values('ts').reset_index(drop=True)


def load_trades_for_label(label: str) -> pd.DataFrame:
    files = sorted(glob(str(Path(TRADES_FOLDER) / f"trades_{label}_*.csv")))
    if not files:
        return pd.DataFrame()
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
def evaluate_combination(df_trades: pd.DataFrame, btc_df: pd.DataFrame,
                         ma_period: int, long_th: float, short_th: float) -> dict:
    long_all, long_filt   = [], []
    short_all, short_filt = [], []

    for _, trade in df_trades.iterrows():
        profit        = trade['profit']
        position_type = trade['position_type']
        direction     = get_macro_direction(btc_df, trade['buy_time'], ma_period, long_th, short_th)

        if position_type == 'LONG':
            long_all.append(profit)
            if direction == 'uptrend':
                long_filt.append(profit)
        elif position_type == 'SHORT':
            short_all.append(profit)
            if direction == 'dwtrend':
                short_filt.append(profit)

    return {
        'long_total_trades':    len(long_all),
        'long_total_profit':    sum(long_all),
        'long_filtered_trades': len(long_filt),
        'long_filtered_profit': sum(long_filt),
        'short_total_trades':   len(short_all),
        'short_total_profit':   sum(short_all),
        'short_filtered_trades':len(short_filt),
        'short_filtered_profit':sum(short_filt),
        'combined_profit':      sum(long_filt) + sum(short_filt),
    }


def run_grid(df_trades: pd.DataFrame, btc_df: pd.DataFrame) -> list:
    combos = [(ma, lt, st) for ma in MA_TYPES for lt in THRESHOLDS for st in THRESHOLDS if st <= lt]
    results = []
    for ma_period, long_th, short_th in combos:
        result = evaluate_combination(df_trades, btc_df, ma_period, long_th, short_th)
        results.append({'ma_period': ma_period, 'long_th': long_th, 'short_th': short_th, 'result': result})
    return sorted(results, key=lambda x: x['result']['combined_profit'], reverse=True)


# =============================================================================
# PRINTING
# =============================================================================
def print_top_table(results: list, title: str, n: int = TOP_N):
    print(f"\n{'─'*110}")
    print(f"  {title}")
    print(f"{'─'*110}")
    print(f"  {'#':>3} {'MA':>6} {'LONG_TH':>10} {'SHORT_TH':>10} {'TR_TOT':>8} {'TR_FILT':>8} {'PF_TOT':>12} {'PF_FILT':>12} {'Δ_PROFIT':>12}")
    print(f"  {'-'*100}")
    for rank, combo in enumerate(results[:n], 1):
        r             = combo['result']
        tr_tot        = r['long_total_trades']    + r['short_total_trades']
        tr_filt       = r['long_filtered_trades'] + r['short_filtered_trades']
        pf_tot        = r['long_total_profit']    + r['short_total_profit']
        pf_filt       = r['combined_profit']
        delta         = pf_filt - pf_tot
        print(f"  {rank:>3} {'MA'+str(combo['ma_period']):>6} {combo['long_th']:>10.2f} {combo['short_th']:>10.2f} "
              f"{tr_tot:>8} {tr_filt:>8} {pf_tot:>12.2f} {pf_filt:>12.2f} {delta:>+12.2f}")
    print(f"  {'-'*100}")


def print_period_summary(period_results: dict, best_combo: dict):
    """Print one row per period for the best combination."""
    ma, lt, st = best_combo['ma_period'], best_combo['long_th'], best_combo['short_th']
    print(f"\n{'═'*110}")
    print(f"  BEST COMBINATION ACROSS PERIODS — MA{ma}  LONG_TH={lt}  SHORT_TH={st}")
    print(f"{'═'*110}")
    print(f"  {'Period':<8} {'TR_TOT':>8} {'TR_FILT':>8} {'%TR_ELIM':>10} {'PF_TOT':>12} {'PF_FILT':>12} {'Δ_PROFIT':>12}")
    print(f"  {'─'*80}")

    agg = {'tr_tot': 0, 'tr_filt': 0, 'pf_tot': 0.0, 'pf_filt': 0.0}
    for period_name, results in period_results.items():
        # Find this combo in results
        combo = next((r for r in results if r['ma_period'] == ma and r['long_th'] == lt and r['short_th'] == st), None)
        if not combo:
            continue
        r       = combo['result']
        tr_tot  = r['long_total_trades']    + r['short_total_trades']
        tr_filt = r['long_filtered_trades'] + r['short_filtered_trades']
        pf_tot  = r['long_total_profit']    + r['short_total_profit']
        pf_filt = r['combined_profit']
        delta   = pf_filt - pf_tot
        pct_elim = round((1 - tr_filt / tr_tot) * 100, 1) if tr_tot > 0 else 0.0
        print(f"  {period_name:<8} {tr_tot:>8} {tr_filt:>8} {pct_elim:>9.1f}% {pf_tot:>12.2f} {pf_filt:>12.2f} {delta:>+12.2f}")
        agg['tr_tot']  += tr_tot
        agg['tr_filt'] += tr_filt
        agg['pf_tot']  += pf_tot
        agg['pf_filt'] += pf_filt

    agg_pct  = round((1 - agg['tr_filt'] / agg['tr_tot']) * 100, 1) if agg['tr_tot'] > 0 else 0.0
    agg_delta = agg['pf_filt'] - agg['pf_tot']
    print(f"  {'─'*80}")
    print(f"  {'TOTAL':<8} {agg['tr_tot']:>8} {agg['tr_filt']:>8} {agg_pct:>9.1f}% "
          f"{agg['pf_tot']:>12.2f} {agg['pf_filt']:>12.2f} {agg_delta:>+12.2f}")
    print(f"  {'═'*110}")


def print_direction_distribution(df: pd.DataFrame, btc_df: pd.DataFrame,
                                  ma_period: int, long_th: float, short_th: float,
                                  period_name: str) -> None:
    """Print distribution of BTC macro direction for all trades in a period."""
    directions = []
    for _, trade in df.iterrows():
        direction = get_macro_direction(btc_df, trade['buy_time'], ma_period, long_th, short_th)
        directions.append({'direction': direction, 'position_type': trade['position_type']})

    df_dir = pd.DataFrame(directions)
    print(f"\n  Direction distribution — {period_name} (MA{ma_period}  LONG_TH={long_th}  SHORT_TH={short_th})")
    print(f"  {'─'*60}")
    for side in ['LONG', 'SHORT']:
        subset = df_dir[df_dir['position_type'] == side]
        total  = len(subset)
        for d in ['uptrend', 'dwtrend', 'neutral', 'unknown']:
            n   = (subset['direction'] == d).sum()
            pct = round(n / total * 100, 1) if total > 0 else 0.0
            print(f"  {side:<6} {d:<10} {n:>6}  ({pct:.1f}%)")
    print(f"  {'─'*60}")


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 80)
    print("REGIME0 EXHAUSTIVE — Find best MA + threshold combination per period")
    print("=" * 80)
    print(f"\n  Trades folder : {TRADES_FOLDER}")
    print(f"  REF folder    : {REF_FOLDER}")
    print(f"  MA types      : {MA_TYPES}")
    print(f"  Thresholds    : {THRESHOLDS}")
    print(f"  BTC timeframe : {BTC_TIMEFRAME}")
    n_combos = sum(1 for ma in MA_TYPES for lt in THRESHOLDS for st in THRESHOLDS if st <= lt)
    print(f"  Combinations  : {n_combos} (filtered st<=lt)")

    print("\n  Loading BTC data...")
    btc_df = load_btc()
    print(f"  {len(btc_df)} bars loaded ({BTC_TIMEFRAME})")

    period_results = {}
    all_trades_list = []

    for period_name, label in PERIOD_LABELS:
        df = load_trades_for_label(label)
        if df.empty:
            print(f"\n  No files for {label} — skipping.")
            continue
        print(f"\n  [{period_name}] {len(df)} trades ({label}) — running grid search...")
        results = run_grid(df, btc_df)
        period_results[period_name] = results
        all_trades_list.append(df)
        print_top_table(results, f"TOP {TOP_N} — {period_name}")

    # Aggregated grid on all periods combined
    if all_trades_list:
        df_all = pd.concat(all_trades_list, ignore_index=True).sort_values('buy_time').reset_index(drop=True)
        print(f"\n  [ALL PERIODS] {len(df_all)} trades combined — running grid search...")
        results_all = run_grid(df_all, btc_df)
        print_top_table(results_all, f"TOP {TOP_N} — ALL PERIODS COMBINED")

        best = results_all[0]
        print_period_summary(period_results, best)

        print(f"\n{'═'*80}")
        print("  BEST COMBINATION — use these values in regime scripts")
        print(f"{'═'*80}")
        print(f"\n  BTC_MA_PERIOD = {best['ma_period']}")
        print(f"  LONG_TH       = {best['long_th']}")
        print(f"  SHORT_TH      = {best['short_th']}")
        print(f"{'═'*80}\n")


if __name__ == "__main__":
    main()