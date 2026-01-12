"""
verify_enrichment.py

Verifies enrichment quality and diagnoses issues.
Checks for:
- Timestamp matching problems
- NaN values distribution
- Metric consistency across strategies
- Same-period strategies comparison

Usage:
    python verify_enrichment.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob


def analyze_enriched_files(output_folder='output'):
    """Analyzes all enriched files for quality and consistency."""
    
    print("=" * 80)
    print("ENRICHMENT VERIFICATION")
    print("=" * 80)
    
    # Find all enriched files
    pattern = f"{output_folder}/trades_enriched_*.xlsx"
    files = sorted(glob(pattern))
    
    if not files:
        print(f"\n❌ No enriched files found in {output_folder}")
        return
    
    print(f"\nFound {len(files)} enriched files")
    
    # Load all files
    dfs = {}
    for f in files:
        name = Path(f).stem.replace('trades_enriched_', '')
        dfs[name] = pd.read_excel(f)
        dfs[name].columns = dfs[name].columns.str.lower().str.strip()
        if 'buy_time' in dfs[name].columns:
            dfs[name]['buy_time'] = pd.to_datetime(dfs[name]['buy_time'])
    
    # =================================================================
    # CHECK 1: NaN values
    # =================================================================
    print("\n" + "=" * 80)
    print("CHECK 1: NaN VALUES IN METRICS")
    print("=" * 80)
    
    print(f"\n{'STRATEGY':<35} {'TOTAL':>7} {'NaN_HURST':>10} {'NaN_ATR':>10} {'NaN_ER':>10} {'NaN_PE':>10} {'%_NaN':>8}")
    print("-" * 95)
    
    for name, df in dfs.items():
        total = len(df)
        nan_hurst = df['hurst'].isna().sum()
        nan_atr = df['atr_pct'].isna().sum()
        nan_er = df['efficiency_ratio'].isna().sum()
        nan_pe = df['permutation_entropy'].isna().sum()
        pct_nan = (nan_hurst / total * 100) if total > 0 else 0
        
        status = "⚠️" if pct_nan > 5 else "✅"
        print(f"{name:<35} {total:>7} {nan_hurst:>10} {nan_atr:>10} {nan_er:>10} {nan_pe:>10} {pct_nan:>7.1f}% {status}")
    
    # =================================================================
    # CHECK 2: Metric ranges (detect outliers)
    # =================================================================
    print("\n" + "=" * 80)
    print("CHECK 2: METRIC RANGES (should be similar across strategies)")
    print("=" * 80)
    
    for metric in ['hurst', 'atr_pct', 'efficiency_ratio', 'permutation_entropy']:
        print(f"\n{metric.upper()}:")
        print(f"{'STRATEGY':<35} {'MIN':>10} {'MEAN':>10} {'MAX':>10} {'STD':>10}")
        print("-" * 75)
        
        for name, df in dfs.items():
            values = df[metric].dropna()
            if len(values) > 0:
                print(f"{name:<35} {values.min():>10.4f} {values.mean():>10.4f} {values.max():>10.4f} {values.std():>10.4f}")
    
    # =================================================================
    # CHECK 3: Same timeframe comparison
    # =================================================================
    print("\n" + "=" * 80)
    print("CHECK 3: SAME TIMEFRAME COMPARISON")
    print("=" * 80)
    
    # Group by timeframe
    by_tf = {}
    for name, df in dfs.items():
        # Extract timeframe more robustly
        name_upper = name.upper()
        if '1H' in name_upper and '6H' not in name_upper:
            tf = '1H'
        elif '4H' in name_upper:
            tf = '4H'
        elif '6H' in name_upper:
            tf = '6H'
        else:
            tf = 'UNKNOWN'
        
        if tf not in by_tf:
            by_tf[tf] = {}
        by_tf[tf][name] = df
    
    # Compare strategies within same timeframe
    for tf, strategies in by_tf.items():
        if len(strategies) < 2:
            continue
            
        print(f"\n{tf} STRATEGIES:")
        print("-" * 80)
        
        # Get all unique timestamps
        all_times = set()
        for name, df in strategies.items():
            all_times.update(df['buy_time'].tolist())
        
        print(f"Total unique timestamps across all {tf} strategies: {len(all_times)}")
        
        # Check overlap
        strat_names = list(strategies.keys())
        for i in range(len(strat_names)):
            for j in range(i+1, len(strat_names)):
                name1 = strat_names[i]
                name2 = strat_names[j]
                df1 = strategies[name1]
                df2 = strategies[name2]
                
                times1 = set(df1['buy_time'].tolist())
                times2 = set(df2['buy_time'].tolist())
                
                common = times1 & times2
                only1 = times1 - times2
                only2 = times2 - times1
                
                print(f"\n  {name1} vs {name2}:")
                print(f"    Common timestamps: {len(common)}")
                print(f"    Only in {name1}: {len(only1)}")
                print(f"    Only in {name2}: {len(only2)}")
                
                # For common timestamps, check if metrics match
                if len(common) > 0:
                    # Get sample of common timestamps
                    sample_times = list(common)[:5]
                    
                    print(f"\n    SAMPLE METRIC COMPARISON (first 5 common timestamps):")
                    print(f"    {'TIMESTAMP':<20} {'HURST_MATCH':>12} {'ATR_MATCH':>12} {'ER_MATCH':>12}")
                    print("    " + "-" * 60)
                    
                    mismatches = 0
                    for ts in sample_times:
                        row1 = df1[df1['buy_time'] == ts].iloc[0]
                        row2 = df2[df2['buy_time'] == ts].iloc[0]
                        
                        h_match = "✅" if abs(row1['hurst'] - row2['hurst']) < 0.0001 else "❌"
                        atr_match = "✅" if abs(row1['atr_pct'] - row2['atr_pct']) < 0.0001 else "❌"
                        er_match = "✅" if abs(row1['efficiency_ratio'] - row2['efficiency_ratio']) < 0.0001 else "❌"
                        
                        if h_match == "❌" or atr_match == "❌" or er_match == "❌":
                            mismatches += 1
                        
                        ts_str = str(ts)[:19]
                        print(f"    {ts_str:<20} {h_match:>12} {atr_match:>12} {er_match:>12}")
                    
                    if mismatches > 0:
                        print(f"\n    ⚠️  WARNING: Found {mismatches} mismatches in sample!")
                        print(f"    This should NOT happen for same timeframe strategies!")
    
    # =================================================================
    # CHECK 4: Detailed timestamp investigation for OOS 1H
    # =================================================================
    print("\n" + "=" * 80)
    print("CHECK 4: DETAILED INVESTIGATION - 1H_OOS STRATEGIES")
    print("=" * 80)
    
    oos_1h = {k: v for k, v in dfs.items() if '1H_OOS' in k.upper()}
    
    if len(oos_1h) >= 2:
        names = list(oos_1h.keys())
        print(f"\nComparing: {names[0]} vs {names[1]}")
        
        df1 = oos_1h[names[0]]
        df2 = oos_1h[names[1]]
        
        print(f"\n{names[0]}:")
        print(f"  Total trades: {len(df1)}")
        print(f"  Period: {df1['buy_time'].min()} → {df1['buy_time'].max()}")
        print(f"  NaN values: {df1['hurst'].isna().sum()}")
        
        print(f"\n{names[1]}:")
        print(f"  Total trades: {len(df2)}")
        print(f"  Period: {df2['buy_time'].min()} → {df2['buy_time'].max()}")
        print(f"  NaN values: {df2['hurst'].isna().sum()}")
        
        # Check first 10 trades of each
        print(f"\nFIRST 10 TRADES COMPARISON:")
        print(f"{'TIMESTAMP':<20} {'STRATEGY':<10} {'HURST':>10} {'ATR_PCT':>10} {'EFF_RATIO':>10}")
        print("-" * 70)
        
        for i in range(min(10, len(df1))):
            row = df1.iloc[i]
            ts_str = str(row['buy_time'])[:19]
            print(f"{ts_str:<20} {names[0][:10]:<10} {row['hurst']:>10.4f} {row['atr_pct']:>10.4f} {row['efficiency_ratio']:>10.4f}")
        
        print()
        for i in range(min(10, len(df2))):
            row = df2.iloc[i]
            ts_str = str(row['buy_time'])[:19]
            print(f"{ts_str:<20} {names[1][:10]:<10} {row['hurst']:>10.4f} {row['atr_pct']:>10.4f} {row['efficiency_ratio']:>10.4f}")
    
    # =================================================================
    # CHECK 5: BTC timestamp availability check
    # =================================================================
    print("\n" + "=" * 80)
    print("CHECK 5: RECOMMENDATIONS")
    print("=" * 80)
    
    print("\n1. Check if BTC OHLC files have ALL required timestamps")
    print("   → Run: verify_btc_coverage.py (create this to check BTC data)")
    
    print("\n2. If timestamps don't match:")
    print("   → Consider using nearest-neighbor matching instead of exact match")
    print("   → Or align all trade timestamps to BTC bar opens")
    
    print("\n3. If metrics differ for same timestamps:")
    print("   → BUG in enrichment process - investigate get_metrics_at_time()")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    analyze_enriched_files()