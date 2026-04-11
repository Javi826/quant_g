"""
Exhaustive Test Suite for detect_intrabar_exit function - CORRECTED VERSION

Tests all possible scenarios:
- LONG: TP/SL combinations
- SHORT: TP/SL combinations  
- Edge cases: same candle, no exit, invalid ranges
- Time priority when both hit same candle

IMPORTANT: Buy candle (buy_idx) is checked for TP/SL hits.
Test data must ensure buy candle doesn't immediately hit TP/SL unless that's the test case.

Run with:
    python3 test_detect_intrabar_exit.py
"""

import sys
import os
import numpy as np

# Import function to test
# Adjust path to your backtest module
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Change this import to match your actual module path
from backtesters.ZX_compute_BT import detect_intrabar_exit


# ==========================================================================
# HELPER: Create sample data
# ==========================================================================
def create_sample_data(high, low, high_time, low_time):
    """Create sample data dictionary for testing."""
    return {
        'high': np.array(high, dtype=np.float64),
        'low': np.array(low, dtype=np.float64),
        'high_time': np.array(high_time, dtype=np.int64),
        'low_time': np.array(low_time, dtype=np.int64)
    }


# ==========================================================================
# TEST SUITE - CORRECTED
# ==========================================================================

def test_long_tp_hit_first():
    """LONG: TP is hit before SL."""
    print("TEST 1: LONG - TP hit first")
    
    # Entry at 100, vela 0 doesn't hit anything, vela 1 hits TP
    d = create_sample_data(
        high=[102, 105, 110],  # Candle 0: doesn't reach TP(104), Candle 1: hits TP
        low=[98, 100, 105],    # Candle 0: doesn't reach SL(96)
        high_time=[1000, 2000, 3000],
        low_time=[500, 1500, 2500]
    )
    
    buy_idx = 0
    sell_idx = 2
    tp_price = 104.0  # TP at +4%
    sl_price = 96.0   # SL at -4%
    
    detected, idx, reason, price = detect_intrabar_exit(
        d, buy_idx, sell_idx, tp_price, sl_price, False
    )
    
    assert detected == True, "Should detect exit"
    assert idx == 1, f"Should exit at candle 1, got {idx}"
    assert reason == 'TP', f"Should be TP, got {reason}"
    assert price == tp_price
    
    print("   ✅ PASSED: TP detected at correct candle")


def test_long_sl_hit_first():
    """LONG: SL is hit before TP."""
    print("TEST 2: LONG - SL hit first")
    
    # Entry at 100, vela 1 hits SL
    d = create_sample_data(
        high=[102, 98, 105],   # Candle 0: safe, Candle 1: drops
        low=[98, 93, 100],     # Candle 0: safe, Candle 1: hits SL(96)
        high_time=[1000, 2000, 3000],
        low_time=[500, 1500, 2500]
    )
    
    buy_idx = 0
    sell_idx = 2
    tp_price = 104.0
    sl_price = 96.0
    
    detected, idx, reason, price = detect_intrabar_exit(
        d, buy_idx, sell_idx, tp_price, sl_price, False
    )
    
    assert detected == True
    assert idx == 1, f"Should exit at candle 1, got {idx}"
    assert reason == 'SL', f"Should be SL, got {reason}"
    assert price == sl_price
    
    print("   ✅ PASSED: SL detected at correct candle")


def test_long_both_same_candle_tp_first():
    """LONG: Both TP and SL hit same candle, TP happens first (high_time < low_time)."""
    print("TEST 3: LONG - Both same candle, TP first")
    
    d = create_sample_data(
        high=[102, 110],  # Candle 1: Hits TP (104)
        low=[98, 90],     # Candle 1: Hits SL (96)
        high_time=[1000, 1500],  # TP time (earlier)
        low_time=[500, 2000]     # SL time (later)
    )
    
    buy_idx = 0
    sell_idx = 1
    tp_price = 104.0
    sl_price = 96.0
    
    detected, idx, reason, price = detect_intrabar_exit(
        d, buy_idx, sell_idx, tp_price, sl_price, False
    )
    
    assert detected == True
    assert idx == 1
    assert reason == 'TP', f"TP should happen first (high_time < low_time), got {reason}"
    assert price == tp_price
    
    print("   ✅ PASSED: TP prioritized (earlier timestamp)")


def test_long_both_same_candle_sl_first():
    """LONG: Both TP and SL hit same candle, SL happens first (low_time < high_time)."""
    print("TEST 4: LONG - Both same candle, SL first")
    
    d = create_sample_data(
        high=[102, 110],  # Candle 1: hits TP
        low=[98, 90],     # Candle 1: hits SL
        high_time=[1000, 2000],  # TP time (later)
        low_time=[500, 1000]     # SL time (earlier)
    )
    
    buy_idx = 0
    sell_idx = 1
    tp_price = 104.0
    sl_price = 96.0
    
    detected, idx, reason, price = detect_intrabar_exit(
        d, buy_idx, sell_idx, tp_price, sl_price, False
    )
    
    assert detected == True
    assert idx == 1
    assert reason == 'SL', f"SL should happen first (low_time < high_time), got {reason}"
    
    print("   ✅ PASSED: SL prioritized (earlier timestamp)")


def test_long_only_tp():
    """LONG: Only TP defined (SL is None)."""
    print("TEST 5: LONG - Only TP")
    
    d = create_sample_data(
        high=[102, 105, 110],  # Candle 1: hits TP
        low=[98, 100, 105],
        high_time=[1000, 2000, 3000],
        low_time=[500, 1500, 2500]
    )
    
    buy_idx = 0
    sell_idx = 2
    tp_price = 104.0
    sl_price = None
    
    detected, idx, reason, price = detect_intrabar_exit(
        d, buy_idx, sell_idx, tp_price, sl_price, False
    )
    
    assert detected == True
    assert reason == 'TP'
    assert idx == 1
    
    print("   ✅ PASSED: TP detected with SL=None")


def test_long_only_sl():
    """LONG: Only SL defined (TP is None)."""
    print("TEST 6: LONG - Only SL")
    
    d = create_sample_data(
        high=[102, 98, 96],
        low=[98, 93, 90],  # Candle 1: hits SL
        high_time=[1000, 2000, 3000],
        low_time=[500, 1500, 2500]
    )
    
    buy_idx = 0
    sell_idx = 2
    tp_price = None
    sl_price = 96.0
    
    detected, idx, reason, price = detect_intrabar_exit(
        d, buy_idx, sell_idx, tp_price, sl_price, False
    )
    
    assert detected == True
    assert reason == 'SL'
    assert idx == 1
    
    print("   ✅ PASSED: SL detected with TP=None")


def test_long_no_exit():
    """LONG: Neither TP nor SL is hit."""
    print("TEST 7: LONG - No exit")
    
    d = create_sample_data(
        high=[102, 102, 103],  # Never reaches TP(105)
        low=[98, 99, 100],     # Never reaches SL(95)
        high_time=[1000, 2000, 3000],
        low_time=[500, 1500, 2500]
    )
    
    buy_idx = 0
    sell_idx = 2
    tp_price = 105.0  # Not reached
    sl_price = 95.0   # Not reached
    
    detected, idx, reason, price = detect_intrabar_exit(
        d, buy_idx, sell_idx, tp_price, sl_price, False
    )
    
    assert detected == False, "Should not detect exit"
    assert idx is None
    assert reason is None
    assert price is None
    
    print("   ✅ PASSED: No exit detected")


# ==========================================================================
# SHORT TESTS - CORRECTED
# ==========================================================================

def test_short_tp_hit_first():
    """SHORT: TP is hit before SL (price goes DOWN to TP)."""
    print("TEST 8: SHORT - TP hit first")
    
    # Entry: 100, TP: 96 (price down), SL: 104 (price up)
    d = create_sample_data(
        high=[102, 98, 105],
        low=[98, 93, 100],   # Candle 1: low reaches TP (96)
        high_time=[1000, 2000, 3000],
        low_time=[500, 1500, 2500]
    )
    
    buy_idx = 0
    sell_idx = 2
    tp_price = 96.0   # TP at -4% (SHORT profit when price drops)
    sl_price = 104.0  # SL at +4% (SHORT loss when price rises)
    
    detected, idx, reason, price = detect_intrabar_exit(
        d, buy_idx, sell_idx, tp_price, sl_price, is_short=True
    )
    
    assert detected == True
    assert idx == 1, f"Should exit at candle 1, got {idx}"
    assert reason == 'TP', f"Should be TP, got {reason}"
    assert price == tp_price
    
    print("   ✅ PASSED: SHORT TP detected")


def test_short_sl_hit_first():
    """SHORT: SL is hit before TP (price goes UP to SL)."""
    print("TEST 9: SHORT - SL hit first")
    
    # Entry: 100, SL: 104 (price up), TP: 96 (price down)
    d = create_sample_data(
        high=[102, 105, 110],  # Candle 1: high reaches SL (104)
        low=[98, 100, 105],
        high_time=[1000, 2000, 3000],
        low_time=[500, 1500, 2500]
    )
    
    buy_idx = 0
    sell_idx = 2
    tp_price = 96.0
    sl_price = 104.0
    
    detected, idx, reason, price = detect_intrabar_exit(
        d, buy_idx, sell_idx, tp_price, sl_price, is_short=True
    )
    
    assert detected == True
    assert idx == 1
    assert reason == 'SL', f"Should be SL, got {reason}"
    assert price == sl_price
    
    print("   ✅ PASSED: SHORT SL detected")


def test_short_both_same_candle_tp_first():
    """SHORT: Both hit same candle, TP first (low_time < high_time for SHORT)."""
    print("TEST 10: SHORT - Both same candle, TP first")
    
    d = create_sample_data(
        high=[102, 110],  # Candle 1: Hits SL (104)
        low=[98, 90],     # Candle 1: Hits TP (96)
        high_time=[1000, 2000],  # SL time (later)
        low_time=[500, 1500]     # TP time (earlier)
    )
    
    buy_idx = 0
    sell_idx = 1
    tp_price = 96.0
    sl_price = 104.0
    
    detected, idx, reason, price = detect_intrabar_exit(
        d, buy_idx, sell_idx, tp_price, sl_price, is_short=True
    )
    
    assert detected == True
    assert idx == 1
    assert reason == 'TP', f"TP should happen first for SHORT (low_time earlier), got {reason}"
    
    print("   ✅ PASSED: SHORT TP prioritized")


def test_short_both_same_candle_sl_first():
    """SHORT: Both hit same candle, SL first (high_time < low_time for SHORT)."""
    print("TEST 11: SHORT - Both same candle, SL first")
    
    d = create_sample_data(
        high=[102, 110],  # Candle 1: hits SL
        low=[98, 90],     # Candle 1: hits TP
        high_time=[1000, 1500],  # SL time (earlier)
        low_time=[500, 2000]     # TP time (later)
    )
    
    buy_idx = 0
    sell_idx = 1
    tp_price = 96.0
    sl_price = 104.0
    
    detected, idx, reason, price = detect_intrabar_exit(
        d, buy_idx, sell_idx, tp_price, sl_price, is_short=True
    )
    
    assert detected == True
    assert idx == 1
    assert reason == 'SL', f"SL should happen first for SHORT (high_time earlier), got {reason}"
    
    print("   ✅ PASSED: SHORT SL prioritized")


def test_short_only_tp():
    """SHORT: Only TP defined."""
    print("TEST 12: SHORT - Only TP")
    
    d = create_sample_data(
        high=[102, 98, 96],
        low=[98, 93, 90],  # Candle 1: hits TP
        high_time=[1000, 2000, 3000],
        low_time=[500, 1500, 2500]
    )
    
    buy_idx = 0
    sell_idx = 2
    tp_price = 96.0
    sl_price = None
    
    detected, idx, reason, price = detect_intrabar_exit(
        d, buy_idx, sell_idx, tp_price, sl_price, is_short=True
    )
    
    assert detected == True
    assert reason == 'TP'
    assert idx == 1
    
    print("   ✅ PASSED: SHORT TP with SL=None")


def test_short_only_sl():
    """SHORT: Only SL defined."""
    print("TEST 13: SHORT - Only SL")
    
    d = create_sample_data(
        high=[102, 105, 110],  # Candle 1: hits SL
        low=[98, 100, 105],
        high_time=[1000, 2000, 3000],
        low_time=[500, 1500, 2500]
    )
    
    buy_idx = 0
    sell_idx = 2
    tp_price = None
    sl_price = 104.0
    
    detected, idx, reason, price = detect_intrabar_exit(
        d, buy_idx, sell_idx, tp_price, sl_price, is_short=True
    )
    
    assert detected == True
    assert reason == 'SL'
    assert idx == 1
    
    print("   ✅ PASSED: SHORT SL with TP=None")


def test_short_no_exit():
    """SHORT: Neither TP nor SL hit."""
    print("TEST 14: SHORT - No exit")
    
    d = create_sample_data(
        high=[102, 102, 103],  # Never reaches SL(105)
        low=[98, 99, 100],     # Never reaches TP(95)
        high_time=[1000, 2000, 3000],
        low_time=[500, 1500, 2500]
    )
    
    buy_idx = 0
    sell_idx = 2
    tp_price = 95.0   # Not reached (would need price to drop more)
    sl_price = 105.0  # Not reached (would need price to rise more)
    
    detected, idx, reason, price = detect_intrabar_exit(
        d, buy_idx, sell_idx, tp_price, sl_price, is_short=True
    )
    
    assert detected == False
    
    print("   ✅ PASSED: SHORT no exit")


# ==========================================================================
# EDGE CASES
# ==========================================================================

def test_edge_case_invalid_range():
    """Edge case: buy_idx > sell_idx."""
    print("TEST 15: Edge case - Invalid range")
    
    d = create_sample_data(
        high=[100, 105],
        low=[95, 100],
        high_time=[1000, 2000],
        low_time=[500, 1500]
    )
    
    buy_idx = 1
    sell_idx = 0  # INVALID: before buy
    
    detected, idx, reason, price = detect_intrabar_exit(
        d, buy_idx, sell_idx, 105.0, 95.0, False
    )
    
    assert detected == False, "Should not detect with invalid range"
    
    print("   ✅ PASSED: Invalid range handled")


def test_edge_case_both_none():
    """Edge case: Both TP and SL are None."""
    print("TEST 16: Edge case - Both None")
    
    d = create_sample_data(
        high=[100, 105],
        low=[95, 100],
        high_time=[1000, 2000],
        low_time=[500, 1500]
    )
    
    detected, idx, reason, price = detect_intrabar_exit(
        d, 0, 1, None, None, False
    )
    
    assert detected == False
    
    print("   ✅ PASSED: Both None handled")


def test_edge_case_single_candle():
    """Edge case: buy_idx == sell_idx."""
    print("TEST 17: Edge case - Single candle")
    
    d = create_sample_data(
        high=[110],  # Hits TP(105)
        low=[90],    # Hits SL(95)
        high_time=[1500],  # TP time (later)
        low_time=[1000]    # SL time (earlier)
    )
    
    buy_idx = 0
    sell_idx = 0
    
    detected, idx, reason, price = detect_intrabar_exit(
        d, buy_idx, sell_idx, 105.0, 95.0, False
    )
    
    # Should detect SL first (earlier time)
    assert detected == True, "Should detect within single candle"
    assert reason == 'SL', f"Should be SL (earlier time), got {reason}"
    
    print("   ✅ PASSED: Single candle handled")


def test_long_exact_tp_price():
    """LONG: Price exactly matches TP."""
    print("TEST 18: LONG - Exact TP match")
    
    d = create_sample_data(
        high=[102, 104, 105],  # Candle 1: Exactly hits TP
        low=[98, 100, 102],
        high_time=[1000, 2000, 3000],
        low_time=[500, 1500, 2500]
    )
    
    buy_idx = 0
    sell_idx = 2
    tp_price = 104.0  # Exact match
    sl_price = 96.0
    
    detected, idx, reason, price = detect_intrabar_exit(
        d, buy_idx, sell_idx, tp_price, sl_price, False
    )
    
    assert detected == True
    assert reason == 'TP'
    assert idx == 1
    
    print("   ✅ PASSED: Exact TP match")


def test_short_exact_tp_price():
    """SHORT: Price exactly matches TP."""
    print("TEST 19: SHORT - Exact TP match")
    
    d = create_sample_data(
        high=[102, 98, 95],
        low=[98, 96, 93],  # Candle 1: Exactly hits TP
        high_time=[1000, 2000, 3000],
        low_time=[500, 1500, 2500]
    )
    
    buy_idx = 0
    sell_idx = 2
    tp_price = 96.0  # Exact match
    sl_price = 104.0
    
    detected, idx, reason, price = detect_intrabar_exit(
        d, buy_idx, sell_idx, tp_price, sl_price, is_short=True
    )
    
    assert detected == True
    assert reason == 'TP'
    assert idx == 1
    
    print("   ✅ PASSED: SHORT exact TP match")


def test_long_multiple_tp_hits():
    """LONG: TP is hit multiple times, should return first."""
    print("TEST 20: LONG - Multiple TP hits")
    
    d = create_sample_data(
        high=[102, 105, 110, 115],  # TP hit at candles 1, 2, 3
        low=[98, 100, 105, 110],
        high_time=[1000, 2000, 3000, 4000],
        low_time=[500, 1500, 2500, 3500]
    )
    
    buy_idx = 0
    sell_idx = 3
    tp_price = 104.0
    sl_price = 96.0
    
    detected, idx, reason, price = detect_intrabar_exit(
        d, buy_idx, sell_idx, tp_price, sl_price, False
    )
    
    assert detected == True
    assert idx == 1, f"Should exit at first TP (candle 1), got {idx}"
    assert reason == 'TP'
    
    print("   ✅ PASSED: First TP returned")


def test_long_tp_on_buy_candle():
    """LONG: TP is hit on the buy candle itself."""
    print("TEST 21: LONG - TP on buy candle")
    
    d = create_sample_data(
        high=[110, 102, 103],  # Candle 0: hits TP immediately
        low=[98, 99, 100],
        high_time=[1500, 2000, 3000],
        low_time=[1000, 1500, 2500]
    )
    
    buy_idx = 0
    sell_idx = 2
    tp_price = 105.0
    sl_price = 95.0
    
    detected, idx, reason, price = detect_intrabar_exit(
        d, buy_idx, sell_idx, tp_price, sl_price, False
    )
    
    assert detected == True
    assert idx == 0, f"Should exit at buy candle, got {idx}"
    assert reason == 'TP'
    
    print("   ✅ PASSED: TP on buy candle detected")


# ==========================================================================
# MAIN - Run all tests
# ==========================================================================

def main():
    print("\n" + "="*70)
    print("EXHAUSTIVE TEST SUITE: detect_intrabar_exit() - CORRECTED")
    print("="*70 + "\n")
    
    tests = [
        # LONG tests
        test_long_tp_hit_first,
        test_long_sl_hit_first,
        test_long_both_same_candle_tp_first,
        test_long_both_same_candle_sl_first,
        test_long_only_tp,
        test_long_only_sl,
        test_long_no_exit,
        # SHORT tests
        test_short_tp_hit_first,
        test_short_sl_hit_first,
        test_short_both_same_candle_tp_first,
        test_short_both_same_candle_sl_first,
        test_short_only_tp,
        test_short_only_sl,
        test_short_no_exit,
        # Edge cases
        test_edge_case_invalid_range,
        test_edge_case_both_none,
        test_edge_case_single_candle,
        test_long_exact_tp_price,
        test_short_exact_tp_price,
        test_long_multiple_tp_hits,
        test_long_tp_on_buy_candle
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"   ❌ FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"   ❌ ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
        print()
    
    print("="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())