#!/usr/bin/env python3
"""
Standalone script to test regime calculation and compare with dashboard.

Usage:
    python test_regime_comparison.py

Requirements:
    - Bot must be running
    - Run from BOT_trading directory with venv activated
"""

import sys
import pandas as pd
import numpy as np
import requests
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, '/home/javi/projects/quant/quant_g/bitget/BOT_trading')

from market_data.data_utils import fetch_ohlcv_data
from market_regime.regime_metrics import calc_all_metrics
from config.settings import (
    REGIME_REFERENCE_SYMBOL,
    REGIME_FAMILIES,
    REGIME_FAMILY_SIZING,
    REGIME_HURST_WINDOW,
    REGIME_ER_WINDOW,
    REGIME_ATR_WINDOW,
    REGIME_PE_WINDOW,
    REGIME_PE_ORDER
)


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def print_candles(df, timeframe, num=10):
    """Print last N candles"""
    print(f"\nLast {num} candles for {timeframe}:")
    print("-" * 80)
    for i in range(max(0, len(df)-num), len(df)):
        row = df.iloc[i]
        # Convert to float in case they're strings
        o = float(row['open'])
        h = float(row['high'])
        l = float(row['low'])
        c = float(row['close'])
        print(f"  [{i:3d}] {row['timestamp']} | "
              f"O:{o:8.2f} H:{h:8.2f} "
              f"L:{l:8.2f} C:{c:8.2f}")


def calculate_local_regime(timeframe):
    """Calculate regime locally (same as regime_classifier.py)"""
    print(f"\n📊 Calculating regime for {timeframe}...")
    
    # Fetch data
    ohlcv_data = fetch_ohlcv_data([REGIME_REFERENCE_SYMBOL], timeframe)
    df = ohlcv_data.get(REGIME_REFERENCE_SYMBOL)
    
    if df is None or df.empty:
        print(f"❌ No data for {timeframe}")
        return None
    
    print(f"✅ Fetched {len(df)} bars")
    print(f"   First: {df.iloc[0]['timestamp']}")
    print(f"   Last:  {df.iloc[-1]['timestamp']}")
    
    # Print last 10 candles
    print_candles(df, timeframe, 10)
    
    # Normalize and convert to arrays
    from market_data.data_utils import normalize_live_ohlcv, df_to_arrays_live
    df_norm = normalize_live_ohlcv(df)
    arrays = df_to_arrays_live(df_norm)
    
    # Prepare OHLC dict
    ohlc = {
        'open': arrays['open'],
        'high': arrays['high'],
        'low': arrays['low'],
        'close': arrays['close']
    }
    
    # Calculate metrics
    metrics = calc_all_metrics(
        ohlc=ohlc,
        hurst_window=REGIME_HURST_WINDOW,
        er_window=REGIME_ER_WINDOW,
        atr_window=REGIME_ATR_WINDOW,
        pe_window=REGIME_PE_WINDOW,
        pe_order=REGIME_PE_ORDER
    )
    
    print("\n📈 Metrics calculated:")
    print(f"   Hurst:               {metrics.get('hurst', 'N/A'):.6f}")
    print(f"   Efficiency Ratio:    {metrics.get('efficiency_ratio', 'N/A'):.6f}")
    print(f"   ATR %:               {metrics.get('atr_pct', 'N/A'):.4f}%")
    print(f"   Permutation Entropy: {metrics.get('permutation_entropy', 'N/A'):.6f}")
    
    # Classify regime
    family = classify_regime_local(metrics)
    multiplier = REGIME_FAMILY_SIZING.get(family, 1.0)
    
    print(f"\n🎯 Classified as: {family.upper()} ({multiplier}x)")
    
    return {
        'timeframe': timeframe,
        'family': family,
        'multiplier': multiplier,
        'metrics': metrics,
        'num_bars': len(df),
        'first_timestamp': df.iloc[0]['timestamp'],
        'last_timestamp': df.iloc[-1]['timestamp']
    }


def classify_regime_local(metrics):
    """Classify regime (same logic as regime_classifier.py)"""
    # Check for NaN
    if any(pd.isna(v) for v in metrics.values()):
        return 'default'
    
    # First-match-wins
    for family_name, rules in REGIME_FAMILIES.items():
        if not rules:
            continue
        
        match = True
        for metric, (op, threshold) in rules.items():
            if metric not in metrics:
                match = False
                break
            
            value = metrics[metric]
            
            if pd.isna(value):
                match = False
                break
            
            if op == '>' and not (value > threshold):
                match = False
                break
            elif op == '<' and not (value < threshold):
                match = False
                break
        
        if match:
            return family_name
    
    # Default
    for family_name, rules in REGIME_FAMILIES.items():
        if not rules:
            return family_name
    
    return 'default'


def fetch_dashboard_regime(timeframe):
    """Fetch regime from dashboard API"""
    print(f"\n🌐 Fetching regime from dashboard API for {timeframe}...")
    
    try:
        # Create fresh session to avoid Spyder/IPython cookie jar issues
        session = requests.Session()
        
        response = session.get(
            f"http://localhost:5099/api/regime/current?timeframe={timeframe}",
            timeout=5
        )
        
        session.close()
        
        if response.status_code != 200:
            print(f"❌ API returned status {response.status_code}")
            return None
        
        data = response.json()
        
        if not data.get('success'):
            print(f"❌ API returned error: {data.get('error', 'Unknown')}")
            return None
        
        print(f"✅ API response received")
        print(f"   Family:     {data.get('family', 'N/A').upper()}")
        print(f"   Multiplier: {data.get('multiplier', 'N/A')}x")
        
        metrics = data.get('metrics', {})
        print(f"\n📈 API Metrics:")
        print(f"   Hurst:               {metrics.get('hurst', 'N/A')}")
        print(f"   Efficiency Ratio:    {metrics.get('efficiency_ratio', 'N/A')}")
        print(f"   ATR %:               {metrics.get('atr_pct', 'N/A')}")
        print(f"   Permutation Entropy: {metrics.get('permutation_entropy', 'N/A')}")
        
        return data
        
    except Exception as e:
        print(f"❌ Error calling API: {e}")
        import traceback
        traceback.print_exc()
        return None


def compare_results(local, dashboard):
    """Compare local calculation with dashboard"""
    print_header("COMPARISON")
    
    if local is None or dashboard is None:
        print("❌ Cannot compare - missing data")
        return
    
    # Compare family
    local_family = local['family']
    dash_family = dashboard.get('family', 'unknown')
    
    family_match = "✅" if local_family == dash_family else "❌"
    print(f"\n{family_match} Family:")
    print(f"   Local:     {local_family.upper()}")
    print(f"   Dashboard: {dash_family.upper()}")
    
    # Compare multiplier
    local_mult = local['multiplier']
    dash_mult = dashboard.get('multiplier', 0)
    
    mult_match = "✅" if abs(local_mult - dash_mult) < 0.01 else "❌"
    print(f"\n{mult_match} Multiplier:")
    print(f"   Local:     {local_mult:.1f}x")
    print(f"   Dashboard: {dash_mult:.1f}x")
    
    # Compare metrics
    local_metrics = local['metrics']
    dash_metrics = dashboard.get('metrics', {})
    
    print("\n📊 Metrics Comparison:")
    
    for metric in ['hurst', 'efficiency_ratio', 'atr_pct', 'permutation_entropy']:
        local_val = local_metrics.get(metric)
        dash_val = dash_metrics.get(metric)
        
        if local_val is None or dash_val is None:
            match = "⚠️"
            diff = "N/A"
        elif pd.isna(local_val) or pd.isna(dash_val):
            match = "⚠️"
            diff = "NaN"
        else:
            diff_val = abs(local_val - dash_val)
            match = "✅" if diff_val < 0.001 else "❌"
            diff = f"{diff_val:.6f}"
        
        print(f"   {match} {metric:20s}: Local={local_val}, Dashboard={dash_val}, Diff={diff}")


def main():
    """Main test function"""
    print_header("REGIME CALCULATION COMPARISON TEST")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    timeframes = ['4H', '6Hutc']
    
    for tf in timeframes:
        print_header(f"Testing {tf}")
        
        # Calculate locally
        local_result = calculate_local_regime(tf)
        
        # Fetch from dashboard
        dashboard_result = fetch_dashboard_regime(tf)
        
        # Compare
        compare_results(local_result, dashboard_result)
        
        print("\n" + "-"*80)
    
    print_header("TEST COMPLETE")


if __name__ == "__main__":
    main()