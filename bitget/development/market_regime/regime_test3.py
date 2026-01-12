"""
validate_metrics.py

Validates that regime metrics are calculated correctly by:
1. Testing with synthetic data (known expected values)
2. Checking mathematical properties
3. Comparing with manual calculations on real data samples

Usage:
    python validate_metrics.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
import os

# Import the metrics module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_regime.regime_metrics import (
    calc_hurst, calc_efficiency_ratio, calc_atr_pct, calc_permutation_entropy
)


def test_hurst_synthetic():
    """Test Hurst with known synthetic data."""
    print("\n" + "="*80)
    print("TEST 1: HURST EXPONENT - SYNTHETIC DATA")
    print("="*80)
    
    # Test 1: Perfect trend (should be > 0.5)
    print("\n1.1 Perfect uptrend:")
    trend_data = np.linspace(100, 200, 100)
    h_trend = calc_hurst(trend_data, window=100)
    print(f"    Data: Linear uptrend from 100 to 200")
    print(f"    Hurst: {h_trend:.4f}")
    print(f"    Expected: > 0.5 (trending)")
    print(f"    Status: {'✅ PASS' if h_trend > 0.5 else '❌ FAIL'}")
    
    # Test 2: Random walk (should be ≈ 0.5)
    print("\n1.2 Random walk:")
    np.random.seed(42)
    random_walk = np.cumsum(np.random.randn(100)) + 100
    h_random = calc_hurst(random_walk, window=100)
    print(f"    Data: Cumulative sum of random normal")
    print(f"    Hurst: {h_random:.4f}")
    print(f"    Expected: ≈ 0.5 (random walk)")
    print(f"    Status: {'✅ PASS' if 0.4 < h_random < 0.6 else '⚠️  WARNING'}")
    
    # Test 3: Mean-reverting (oscillating)
    print("\n1.3 Mean-reverting oscillation:")
    oscillating = 100 + 10 * np.sin(np.linspace(0, 4*np.pi, 100))
    h_osc = calc_hurst(oscillating, window=100)
    print(f"    Data: Sine wave oscillation")
    print(f"    Hurst: {h_osc:.4f}")
    print(f"    Expected: < 0.5 (mean-reverting)")
    print(f"    Status: {'✅ PASS' if h_osc < 0.5 else '❌ FAIL'}")


def test_efficiency_ratio_synthetic():
    """Test Efficiency Ratio with known synthetic data."""
    print("\n" + "="*80)
    print("TEST 2: EFFICIENCY RATIO - SYNTHETIC DATA")
    print("="*80)
    
    # Test 1: Perfect directional move
    print("\n2.1 Perfect directional move:")
    perfect_trend = np.linspace(100, 150, 15)
    er_perfect = calc_efficiency_ratio(perfect_trend, window=14)
    print(f"    Data: Linear move from 100 to 150")
    print(f"    Net change: {perfect_trend[-1] - perfect_trend[0]:.2f}")
    print(f"    Sum of changes: {np.sum(np.abs(np.diff(perfect_trend))):.2f}")
    print(f"    ER: {er_perfect:.4f}")
    print(f"    Expected: 1.0 (perfect efficiency)")
    print(f"    Status: {'✅ PASS' if er_perfect > 0.99 else '❌ FAIL'}")
    
    # Test 2: Choppy sideways
    print("\n2.2 Choppy sideways:")
    choppy = np.array([100, 105, 100, 105, 100, 105, 100, 105, 100, 105, 100, 105, 100, 105, 100])
    er_choppy = calc_efficiency_ratio(choppy, window=14)
    net = abs(choppy[-1] - choppy[0])
    total = np.sum(np.abs(np.diff(choppy)))
    print(f"    Data: Oscillating 100-105-100...")
    print(f"    Net change: {net:.2f}")
    print(f"    Sum of changes: {total:.2f}")
    print(f"    ER: {er_choppy:.4f}")
    print(f"    Expected: 0.0 (no net progress)")
    print(f"    Status: {'✅ PASS' if er_choppy < 0.1 else '❌ FAIL'}")


def test_atr_pct_synthetic():
    """Test ATR% with known synthetic data."""
    print("\n" + "="*80)
    print("TEST 3: ATR% - SYNTHETIC DATA")
    print("="*80)
    
    # Test 1: Stable price with consistent range
    print("\n3.1 Stable 1% daily range:")
    n = 14
    close = np.full(n+1, 100.0)
    high = close[1:] * 1.005  # 0.5% above close
    low = close[1:] * 0.995   # 0.5% below close
    
    atr = calc_atr_pct(high, low, close[1:], window=14)
    print(f"    Setup: Close=100, High=100.5, Low=99.5")
    print(f"    Expected range: ~1.0%")
    print(f"    ATR%: {atr:.4f}%")
    print(f"    Status: {'✅ PASS' if 0.9 < atr < 1.1 else '❌ FAIL'}")
    
    # Test 2: Volatile price with large range
    print("\n3.2 Volatile 5% daily range:")
    close = np.full(n+1, 100.0)
    high = close[1:] * 1.025  # 2.5% above
    low = close[1:] * 0.975   # 2.5% below
    
    atr = calc_atr_pct(high, low, close[1:], window=14)
    print(f"    Setup: Close=100, High=102.5, Low=97.5")
    print(f"    Expected range: ~5.0%")
    print(f"    ATR%: {atr:.4f}%")
    print(f"    Status: {'✅ PASS' if 4.5 < atr < 5.5 else '❌ FAIL'}")


def test_permutation_entropy_synthetic():
    """Test Permutation Entropy with known patterns."""
    print("\n" + "="*80)
    print("TEST 4: PERMUTATION ENTROPY - SYNTHETIC DATA")
    print("="*80)
    
    # Test 1: Deterministic pattern (same pattern repeated)
    print("\n4.1 Deterministic pattern:")
    deterministic = np.tile([1, 2, 3], 17)  # Repeat [1,2,3] pattern
    pe_det = calc_permutation_entropy(deterministic, window=50, order=3)
    print(f"    Data: Repeating [1,2,3] pattern")
    print(f"    PE: {pe_det:.4f}")
    print(f"    Expected: Low (< 0.5, deterministic)")
    print(f"    Status: {'✅ PASS' if pe_det < 0.5 else '⚠️  WARNING'}")
    
    # Test 2: Random noise
    print("\n4.2 Random noise:")
    np.random.seed(42)
    random_data = np.random.randn(50)
    pe_random = calc_permutation_entropy(random_data, window=50, order=3)
    print(f"    Data: Random normal distribution")
    print(f"    PE: {pe_random:.4f}")
    print(f"    Expected: High (> 0.9, random)")
    print(f"    Status: {'✅ PASS' if pe_random > 0.85 else '⚠️  WARNING'}")


def test_real_data_sample():
    """Test with real enriched data sample."""
    print("\n" + "="*80)
    print("TEST 5: REAL DATA VALIDATION")
    print("="*80)
    
    # Try to load an enriched file
    enriched_files = list(Path('output').glob('trades_enriched_*.xlsx'))
    
    if not enriched_files:
        print("\n⚠️  No enriched files found. Skipping real data test.")
        return
    
    print(f"\nLoading: {enriched_files[0].name}")
    df = pd.read_excel(enriched_files[0])
    df.columns = df.columns.str.lower().str.strip()
    
    print(f"Total trades: {len(df)}")
    
    # Check for obvious errors
    print("\n5.1 Range checks:")
    
    # Hurst should be [0, 1]
    hurst_valid = df['hurst'].between(0, 1).all()
    hurst_min, hurst_max = df['hurst'].min(), df['hurst'].max()
    print(f"    Hurst range: [{hurst_min:.4f}, {hurst_max:.4f}]")
    print(f"    Expected: [0, 1]")
    print(f"    Status: {'✅ PASS' if hurst_valid else '❌ FAIL - OUT OF RANGE!'}")
    
    # Efficiency Ratio should be [0, 1]
    er_valid = df['efficiency_ratio'].between(0, 1).all()
    er_min, er_max = df['efficiency_ratio'].min(), df['efficiency_ratio'].max()
    print(f"    ER range: [{er_min:.4f}, {er_max:.4f}]")
    print(f"    Expected: [0, 1]")
    print(f"    Status: {'✅ PASS' if er_valid else '❌ FAIL - OUT OF RANGE!'}")
    
    # ATR% should be positive
    atr_valid = (df['atr_pct'] > 0).all()
    atr_min, atr_max = df['atr_pct'].min(), df['atr_pct'].max()
    print(f"    ATR% range: [{atr_min:.4f}, {atr_max:.4f}]")
    print(f"    Expected: > 0")
    print(f"    Status: {'✅ PASS' if atr_valid else '❌ FAIL - NEGATIVE VALUES!'}")
    
    # Permutation Entropy should be [0, 1]
    pe_valid = df['permutation_entropy'].between(0, 1).all()
    pe_min, pe_max = df['permutation_entropy'].min(), df['permutation_entropy'].max()
    print(f"    PE range: [{pe_min:.4f}, {pe_max:.4f}]")
    print(f"    Expected: [0, 1]")
    print(f"    Status: {'✅ PASS' if pe_valid else '❌ FAIL - OUT OF RANGE!'}")
    
    # Check for suspicious distributions
    print("\n5.2 Distribution checks:")
    
    # Hurst should have reasonable spread
    hurst_std = df['hurst'].std()
    print(f"    Hurst std: {hurst_std:.4f}")
    print(f"    Status: {'✅ Good variance' if hurst_std > 0.05 else '⚠️  Low variance - check if stuck'}")
    
    # Check for duplicate values (possible calculation error)
    hurst_unique = df['hurst'].nunique()
    hurst_dupes = len(df) - hurst_unique
    print(f"    Hurst unique values: {hurst_unique}/{len(df)}")
    print(f"    Duplicates: {hurst_dupes}")
    print(f"    Status: {'✅ Good diversity' if hurst_unique > len(df)*0.5 else '⚠️  Many duplicates'}")
    
    # Show sample values
    print("\n5.3 Sample values (first 5 trades):")
    print(f"    {'HURST':>10} {'ER':>10} {'ATR%':>10} {'PE':>10}")
    print("    " + "-"*44)
    for i in range(min(5, len(df))):
        print(f"    {df.iloc[i]['hurst']:>10.4f} {df.iloc[i]['efficiency_ratio']:>10.4f} "
              f"{df.iloc[i]['atr_pct']:>10.4f} {df.iloc[i]['permutation_entropy']:>10.4f}")


def test_mathematical_properties():
    """Test mathematical properties and edge cases."""
    print("\n" + "="*80)
    print("TEST 6: MATHEMATICAL PROPERTIES")
    print("="*80)
    
    print("\n6.1 Correlation between metrics:")
    print("    Testing if ER and Hurst correlate (both measure trend strength)")
    
    # Generate various patterns
    np.random.seed(42)
    n_samples = 50
    
    hurst_vals = []
    er_vals = []
    
    for i in range(n_samples):
        # Random trend strength
        trend_strength = np.random.rand()
        noise_level = 1 - trend_strength
        
        # Generate data
        trend = np.linspace(0, trend_strength*50, 100)
        noise = np.random.randn(100) * noise_level * 5
        data = 100 + trend + noise
        
        h = calc_hurst(data, window=100)
        er = calc_efficiency_ratio(data, window=14)
        
        if not np.isnan(h) and not np.isnan(er):
            hurst_vals.append(h)
            er_vals.append(er)
    
    correlation = np.corrcoef(hurst_vals, er_vals)[0, 1]
    print(f"    Correlation(Hurst, ER): {correlation:.4f}")
    print(f"    Expected: Positive (both measure trend)")
    print(f"    Status: {'✅ PASS' if correlation > 0.3 else '⚠️  WARNING - weak correlation'}")
    
    print("\n6.2 Stability test (same input → same output):")
    test_data = np.random.randn(100) + 100
    
    h1 = calc_hurst(test_data, window=100)
    h2 = calc_hurst(test_data, window=100)
    
    print(f"    First call:  Hurst = {h1:.6f}")
    print(f"    Second call: Hurst = {h2:.6f}")
    print(f"    Status: {'✅ PASS - deterministic' if h1 == h2 else '❌ FAIL - non-deterministic!'}")


def main():
    """Run all validation tests."""
    print("="*80)
    print("REGIME METRICS VALIDATION")
    print("="*80)
    print("\nThis script validates that regime metrics are calculated correctly.")
    print("It tests synthetic data, mathematical properties, and real data samples.")
    
    test_hurst_synthetic()
    test_efficiency_ratio_synthetic()
    test_atr_pct_synthetic()
    test_permutation_entropy_synthetic()
    test_real_data_sample()
    test_mathematical_properties()
    
    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)
    print("\nSummary:")
    print("  ✅ PASS = Metric behaves as expected")
    print("  ⚠️  WARNING = Result is questionable but not necessarily wrong")
    print("  ❌ FAIL = Clear error in calculation")
    print("\nIf you see any ❌ FAIL, investigate the metric calculation!")
    print("="*80)


if __name__ == "__main__":
    main()