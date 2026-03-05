#!/usr/bin/env python3
"""
Monthly Analysis 2025 vs February 2026

Analyzes each month of 2025 to find if any month resembles February 2026.
Compares: WR, $/trade, ATR, ER, and trade distributions.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_all_2025_trades():
    """Load and combine all 2025 strategy trades"""
    
    output_folder = Path('/home/javi/projects/quant/quant_g/bitget/development/market_regime/output_2025')
    pattern = str(output_folder / "trades_enriched_*.xlsx")
    files = glob(pattern)
    
    all_trades = []
    
    for filepath in files:
        df = pd.read_excel(filepath)
        df.columns = df.columns.str.lower()
        
        # Find timestamp column
        ts_col = None
        for col in ['open_at', 'buy_time']:
            if col in df.columns:
                ts_col = col
                break
        
        if ts_col:
            df[ts_col] = pd.to_datetime(df[ts_col])
            df['month'] = df[ts_col].dt.to_period('M')
            all_trades.append(df)
    
    if not all_trades:
        return None
    
    combined = pd.concat(all_trades, ignore_index=True)
    return combined


def load_feb_2026_trades():
    """Load February 2026 trades"""
    
    feb_file = Path('/home/javi/projects/quant/quant_g/bitget/development/defense_mode/enriched_bot_trades_00_feb.xlsx')
    
    if not feb_file.exists():
        return None
    
    df = pd.read_excel(feb_file)
    df.columns = df.columns.str.lower()
    
    ts_col = None
    for col in ['open_at', 'buy_time']:
        if col in df.columns:
            ts_col = col
            break
    
    if ts_col:
        df[ts_col] = pd.to_datetime(df[ts_col])
    
    return df


def analyze_month(df, month_name):
    """Analyze single month metrics"""
    
    if len(df) == 0:
        return None
    
    # Find ER and ATR columns
    er_col = None
    atr_col = None
    
    for col in df.columns:
        if 'efficiency' in col.lower() and 'ratio' in col.lower():
            er_col = col
        if 'atr' in col.lower() and ('pct' in col.lower() or 'percent' in col.lower()):
            atr_col = col
    
    # Basic metrics
    total_trades = len(df)
    wr = (df['profit'] > 0).mean() * 100
    avg_profit = df['profit'].mean()
    total_profit = df['profit'].sum()
    
    # BTC metrics
    avg_er = df[er_col].mean() if er_col and er_col in df.columns else np.nan
    avg_atr = df[atr_col].mean() if atr_col and atr_col in df.columns else np.nan
    
    # ER distribution
    er_020_040_count = 0
    er_020_040_wr = 0
    er_020_040_avg = 0
    
    if er_col and er_col in df.columns:
        er_020_040 = df[(df[er_col] >= 0.20) & (df[er_col] < 0.40)]
        er_020_040_count = len(er_020_040)
        if er_020_040_count > 0:
            er_020_040_wr = (er_020_040['profit'] > 0).mean() * 100
            er_020_040_avg = er_020_040['profit'].mean()
    
    return {
        'month': month_name,
        'trades': total_trades,
        'wr': wr,
        'avg_profit': avg_profit,
        'total_profit': total_profit,
        'avg_er': avg_er,
        'avg_atr': avg_atr,
        'er_020_040_count': er_020_040_count,
        'er_020_040_pct': (er_020_040_count / total_trades * 100) if total_trades > 0 else 0,
        'er_020_040_wr': er_020_040_wr,
        'er_020_040_avg': er_020_040_avg
    }


def main():
    print("="*120)
    print("MONTHLY ANALYSIS: 2025 vs February 2026")
    print("="*120)
    
    # Load 2025 trades
    print("\n📂 Loading 2025 trades...")
    df_2025 = load_all_2025_trades()
    
    if df_2025 is None:
        print("❌ Could not load 2025 trades")
        return
    
    print(f"✅ Loaded {len(df_2025)} trades from 2025")
    
    # Load February 2026
    print("📂 Loading February 2026...")
    df_feb = load_feb_2026_trades()
    
    if df_feb is None:
        print("❌ Could not load February 2026")
        return
    
    print(f"✅ Loaded {len(df_feb)} trades from February 2026")
    
    # Analyze each month of 2025
    print(f"\n{'='*120}")
    print("ANALYZING 2025 MONTHS")
    print(f"{'='*120}")
    
    months_2025 = []
    
    for month in df_2025['month'].unique():
        month_df = df_2025[df_2025['month'] == month]
        result = analyze_month(month_df, str(month))
        if result:
            months_2025.append(result)
    
    # Analyze February 2026
    feb_result = analyze_month(df_feb, "Feb 2026")
    
    # Sort by month
    months_2025 = sorted(months_2025, key=lambda x: x['month'])
    
    # Print results
    print(f"\n{'Month':<12} {'Trades':>8} {'WR%':>8} {'$/Trade':>10} {'AvgER':>8} {'AvgATR':>8} {'ER0.2-0.4':>10} {'WR_ER':>8} {'$/T_ER':>10}")
    print("-"*120)
    
    for m in months_2025:
        print(f"{m['month']:<12} "
              f"{m['trades']:>8} "
              f"{m['wr']:>7.1f}% "
              f"{m['avg_profit']:>10.2f} "
              f"{m['avg_er']:>8.3f} "
              f"{m['avg_atr']:>8.2f} "
              f"{m['er_020_040_pct']:>9.1f}% "
              f"{m['er_020_040_wr']:>7.1f}% "
              f"{m['er_020_040_avg']:>10.2f}")
    
    print("-"*120)
    
    if feb_result:
        print(f"{'FEB 2026':<12} "
              f"{feb_result['trades']:>8} "
              f"{feb_result['wr']:>7.1f}% "
              f"{feb_result['avg_profit']:>10.2f} "
              f"{feb_result['avg_er']:>8.3f} "
              f"{feb_result['avg_atr']:>8.2f} "
              f"{feb_result['er_020_040_pct']:>9.1f}% "
              f"{feb_result['er_020_040_wr']:>7.1f}% "
              f"{feb_result['er_020_040_avg']:>10.2f}")
    
    # Statistical comparison
    print(f"\n{'='*120}")
    print("COMPARISON WITH FEBRUARY 2026")
    print(f"{'='*120}")
    
    if feb_result:
        # Find most similar month
        print(f"\nFebruary 2026 characteristics:")
        print(f"  WR: {feb_result['wr']:.1f}%")
        print(f"  $/trade: {feb_result['avg_profit']:.2f}")
        print(f"  Avg ER: {feb_result['avg_er']:.3f}")
        print(f"  Avg ATR: {feb_result['avg_atr']:.2f}%")
        print(f"  ER 0.2-0.4: {feb_result['er_020_040_pct']:.1f}% of trades")
        print(f"  ER 0.2-0.4 WR: {feb_result['er_020_040_wr']:.1f}%")
        print(f"  ER 0.2-0.4 $/trade: {feb_result['er_020_040_avg']:.2f}")
        
        # Compare with 2025 months
        print(f"\n2025 Months comparison:")
        
        # Find worst month in 2025
        worst_2025 = min(months_2025, key=lambda x: x['avg_profit'])
        best_2025 = max(months_2025, key=lambda x: x['avg_profit'])
        
        print(f"\nWorst 2025 month: {worst_2025['month']}")
        print(f"  WR: {worst_2025['wr']:.1f}% (Feb: {feb_result['wr']:.1f}%)")
        print(f"  $/trade: {worst_2025['avg_profit']:.2f} (Feb: {feb_result['avg_profit']:.2f})")
        print(f"  Avg ER: {worst_2025['avg_er']:.3f} (Feb: {feb_result['avg_er']:.3f})")
        print(f"  Avg ATR: {worst_2025['avg_atr']:.2f}% (Feb: {feb_result['avg_atr']:.2f}%)")
        
        print(f"\nBest 2025 month: {best_2025['month']}")
        print(f"  WR: {best_2025['wr']:.1f}%")
        print(f"  $/trade: {best_2025['avg_profit']:.2f}")
        
        # Statistical summary
        avg_2025_wr = np.mean([m['wr'] for m in months_2025])
        avg_2025_profit = np.mean([m['avg_profit'] for m in months_2025])
        avg_2025_er = np.mean([m['avg_er'] for m in months_2025 if not np.isnan(m['avg_er'])])
        avg_2025_atr = np.mean([m['avg_atr'] for m in months_2025 if not np.isnan(m['avg_atr'])])
        
        print(f"\n2025 Average:")
        print(f"  WR: {avg_2025_wr:.1f}% (Feb: {feb_result['wr']:.1f}%, Δ: {feb_result['wr'] - avg_2025_wr:+.1f}pp)")
        print(f"  $/trade: {avg_2025_profit:.2f} (Feb: {feb_result['avg_profit']:.2f}, Δ: {feb_result['avg_profit'] - avg_2025_profit:+.2f})")
        print(f"  Avg ER: {avg_2025_er:.3f} (Feb: {feb_result['avg_er']:.3f}, Δ: {feb_result['avg_er'] - avg_2025_er:+.3f})")
        print(f"  Avg ATR: {avg_2025_atr:.2f}% (Feb: {feb_result['avg_atr']:.2f}%, Δ: {feb_result['avg_atr'] - avg_2025_atr:+.2f})")
        
        # Is February an outlier?
        print(f"\n{'='*120}")
        print("OUTLIER ANALYSIS")
        print(f"{'='*120}")
        
        # Calculate z-scores
        wr_values = [m['wr'] for m in months_2025]
        profit_values = [m['avg_profit'] for m in months_2025]
        
        wr_std = np.std(wr_values)
        profit_std = np.std(profit_values)
        
        wr_zscore = (feb_result['wr'] - avg_2025_wr) / wr_std if wr_std > 0 else 0
        profit_zscore = (feb_result['avg_profit'] - avg_2025_profit) / profit_std if profit_std > 0 else 0
        
        print(f"\nFebruary 2026 deviation from 2025 average:")
        print(f"  WR z-score: {wr_zscore:.2f} ({'OUTLIER' if abs(wr_zscore) > 2 else 'normal'})")
        print(f"  $/trade z-score: {profit_zscore:.2f} ({'OUTLIER' if abs(profit_zscore) > 2 else 'normal'})")
        
        if abs(wr_zscore) > 2 or abs(profit_zscore) > 2:
            print(f"\n⚠️  FEBRUARY 2026 IS A STATISTICAL OUTLIER")
            print(f"   It is significantly different from ALL 2025 months")
            print(f"   This was a rare, unpredictable event")
        else:
            print(f"\n✅ February 2026 is within normal variance")
            print(f"   Similar conditions occurred in 2025")
    
    print(f"\n{'='*120}")


if __name__ == "__main__":
    main()