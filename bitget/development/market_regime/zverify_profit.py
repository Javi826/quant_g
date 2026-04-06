"""
market_regime/verify_profit_consistency.py

Verifies that total profits match between:
1. Sum of all trades (from brief_trades_2025)
2. Final equity minus initial capital (from brief_equities)

Usage:
    python verify_profit_consistency.py
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

# Configuration
TRADES_FOLDER   = '../brief_trades'
EQUITIES_FOLDER = "../brief_equities"
INITIAL_CAPITAL = 800

print("=" * 80)
print("PROFIT CONSISTENCY VERIFICATION")
print("=" * 80)

# ================================================ex=============================
# STEP 1: Calculate total profit from TRADES
# =============================================================================
print("\n[1] ANALYZING TRADES...")
print("-" * 80)

trades_pattern = os.path.join(TRADES_FOLDER, "all_trades_*.xlsx")
trades_files = sorted(glob(trades_pattern))

if not trades_files:
    print(f"❌ No trades files found in {TRADES_FOLDER}")
    exit(1)

print(f"Found {len(trades_files)} trade files\n")

trades_summary = []
trades_total_profit = 0.0

for filepath in trades_files:
    filename = Path(filepath).stem
    strategy_name = filename.replace('all_trades_', '')
    
    try:
        df = pd.read_excel(filepath)
        df.columns = df.columns.str.lower().str.strip()
        
        if 'profit' not in df.columns:
            print(f"⚠️  {filename}: No 'profit' column, skipping")
            continue
        
        profit = df['profit'].sum()
        num_trades = len(df)
        
        trades_summary.append({
            'strategy': strategy_name,
            'num_trades': num_trades,
            'profit': profit
        })
        
        trades_total_profit += profit
        
        print(f"{strategy_name:<40} Trades: {num_trades:>6}  Profit: ${profit:>12.2f}")
        
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")

print("-" * 80)
print(f"{'TOTAL FROM TRADES':<40} Trades: {sum(s['num_trades'] for s in trades_summary):>6}  Profit: ${trades_total_profit:>12.2f}")

# =============================================================================
# STEP 2: Calculate total profit from EQUITIES
# =============================================================================
print("\n[2] ANALYZING EQUITIES...")
print("-" * 80)

equities_pattern = os.path.join(EQUITIES_FOLDER, "equity_*.xlsx")
equities_files = sorted(glob(equities_pattern))

if not equities_files:
    print(f"❌ No equity files found in {EQUITIES_FOLDER}")
    exit(1)

print(f"Found {len(equities_files)} equity files\n")

equities_summary = []
equities_total_profit = 0.0
num_strategies_equity = 0

for filepath in equities_files:
    filename = Path(filepath).stem
    strategy_name = filename.replace('equity_', '')
    
    try:
        df = pd.read_excel(filepath)
        df.columns = df.columns.str.lower().str.strip()
        
        if 'balance' not in df.columns:
            print(f"⚠️  {filename}: No 'balance' column, skipping")
            continue
        
        final_balance = df['balance'].iloc[-1]
        profit = final_balance - INITIAL_CAPITAL
        
        equities_summary.append({
            'strategy': strategy_name,
            'final_balance': final_balance,
            'profit': profit
        })
        
        equities_total_profit += profit
        num_strategies_equity += 1
        
        print(f"{strategy_name:<40} Final: ${final_balance:>10.2f}  Profit: ${profit:>12.2f}")
        
    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")

print("-" * 80)
total_capital_equity = INITIAL_CAPITAL * num_strategies_equity
combined_final_balance = total_capital_equity + equities_total_profit
print(f"{'TOTAL FROM EQUITIES':<40} Final: ${combined_final_balance:>10.2f}  Profit: ${equities_total_profit:>12.2f}")

# =============================================================================
# STEP 3: COMPARISON
# =============================================================================
print("\n[3] COMPARISON")
print("=" * 80)

print(f"\nTotal Profit from TRADES:   ${trades_total_profit:>15.2f}")
print(f"Total Profit from EQUITIES: ${equities_total_profit:>15.2f}")
print(f"Difference:                 ${abs(trades_total_profit - equities_total_profit):>15.2f}")

difference_pct = abs(trades_total_profit - equities_total_profit) / trades_total_profit * 100 if trades_total_profit != 0 else 0

if difference_pct < 0.01:
    print(f"\n✅ MATCH! (Difference: {difference_pct:.4f}%)")
elif difference_pct < 1.0:
    print(f"\n⚠️  SMALL DIFFERENCE (Difference: {difference_pct:.2f}%)")
else:
    print(f"\n❌ LARGE DIFFERENCE! (Difference: {difference_pct:.2f}%)")

# =============================================================================
# STEP 4: NET GAIN PERCENTAGES
# =============================================================================
print("\n[4] NET GAIN PERCENTAGES")
print("=" * 80)

# From trades
trades_capital = INITIAL_CAPITAL * len(trades_summary)
trades_net_gain_pct = (trades_total_profit / trades_capital * 100) if trades_capital != 0 else 0

# From equities
equities_capital = INITIAL_CAPITAL * num_strategies_equity
equities_net_gain_pct = (equities_total_profit / equities_capital * 100) if equities_capital != 0 else 0

print(f"\nTrades approach:")
print(f"  Capital: ${trades_capital:>15.2f} ({len(trades_summary)} strategies × ${INITIAL_CAPITAL})")
print(f"  Profit:  ${trades_total_profit:>15.2f}")
print(f"  Net Gain: {trades_net_gain_pct:>14.2f}%")

print(f"\nEquities approach:")
print(f"  Capital: ${equities_capital:>15.2f} ({num_strategies_equity} strategies × ${INITIAL_CAPITAL})")
print(f"  Profit:  ${equities_total_profit:>15.2f}")
print(f"  Net Gain: {equities_net_gain_pct:>14.2f}%")

# =============================================================================
# STEP 5: STRATEGY-BY-STRATEGY COMPARISON
# =============================================================================
print("\n[5] STRATEGY-BY-STRATEGY COMPARISON")
print("=" * 80)

# Match strategies
trades_dict = {s['strategy']: s for s in trades_summary}
equities_dict = {s['strategy']: s for s in equities_summary}

all_strategies = sorted(set(trades_dict.keys()) | set(equities_dict.keys()))

print(f"\n{'STRATEGY':<40} {'TRADES $':>15} {'EQUITIES $':>15} {'DIFF $':>15} {'MATCH':>10}")
print("-" * 100)

mismatches = []

for strategy in all_strategies:
    trades_profit = trades_dict.get(strategy, {}).get('profit', np.nan)
    equities_profit = equities_dict.get(strategy, {}).get('profit', np.nan)
    
    if pd.isna(trades_profit):
        status = "❌ MISSING"
        diff = np.nan
    elif pd.isna(equities_profit):
        status = "❌ MISSING"
        diff = np.nan
    else:
        diff = abs(trades_profit - equities_profit)
        diff_pct = (diff / trades_profit * 100) if trades_profit != 0 else 0
        
        if diff_pct < 0.01:
            status = "✅"
        elif diff_pct < 1.0:
            status = "⚠️"
            mismatches.append((strategy, diff_pct))
        else:
            status = "❌"
            mismatches.append((strategy, diff_pct))
    
    print(f"{strategy:<40} {trades_profit:>15.2f} {equities_profit:>15.2f} {diff:>15.2f} {status:>10}")

print("-" * 100)

if mismatches:
    print(f"\n⚠️  {len(mismatches)} strategies with differences:")
    for strategy, diff_pct in mismatches:
        print(f"   • {strategy}: {diff_pct:.2f}%")
else:
    print("\n✅ All strategies match!")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)