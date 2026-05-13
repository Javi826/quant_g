"""
test_regime_verification.py
Verifies that get_ref_1d_direction() and get_current_regime() produce
correct results by comparing against manual calculation from raw candles.

Usage:
    python test_regime_verification.py

Run from BOT_trading root with env_quant activated.
"""

import sys
import os
import pandas as pd

# =============================================================================
# PATH SETUP
# =============================================================================
BOT_DIR    = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR   = os.path.abspath(os.path.join(BOT_DIR, ".."))
ACCOUNT    = "E1"
TIMEFRAME  = "1H"

sys.path.insert(0, BOT_DIR)
sys.path.insert(0, ROOT_DIR)

# =============================================================================
# CONFIGURE ACCOUNT
# =============================================================================
from market_regime.regime_classifier import configure_regime, fetch_ref_ohlcv
from market_regime.regime_classifier import get_ref_1d_direction, get_current_regime
from config.settings import ACCOUNTS

configure_regime(ACCOUNT)

from market_regime.regime_classifier import (
    REGIME_REFERENCE_SYMBOL,
    REGIME0_MA_PERIOD,
    GLOBAL_SYSTEM_REGIME_TH1,
    GLOBAL_SYSTEM_REGIME_TH2,
)

print(f"\n{'='*60}")
print(f"  REGIME VERIFICATION — Account: {ACCOUNT}")
print(f"{'='*60}")
print(f"  Reference symbol : {REGIME_REFERENCE_SYMBOL}")
print(f"  MA period        : {REGIME0_MA_PERIOD}")
print(f"  Long threshold   : {GLOBAL_SYSTEM_REGIME_TH2}")
print(f"  Short threshold  : {GLOBAL_SYSTEM_REGIME_TH1}")
print(f"{'='*60}\n")

# =============================================================================
# TEST 1 — get_ref_1d_direction() vs manual calculation
# =============================================================================
print("TEST 1 — Direction (1D)")
print("-" * 60)

df_1d = fetch_ref_ohlcv("1Dutc")

if df_1d is None or df_1d.empty:
    print("  ERROR: Could not fetch 1D data")
else:
    close      = pd.to_numeric(df_1d["close"], errors="coerce")
    last_close = float(close.iloc[-1])
    ma         = float(close.tail(REGIME0_MA_PERIOD).mean())

    # Manual calculation
    if last_close > ma * GLOBAL_SYSTEM_REGIME_TH2:
        manual_direction = "uptrend"
    elif last_close < ma * GLOBAL_SYSTEM_REGIME_TH1:
        manual_direction = "dwtrend"
    else:
        manual_direction = "uptrend"

    # Function result
    func_direction = get_ref_1d_direction()

    # Print candles used
    print(f"  Last {REGIME0_MA_PERIOD} closes used for MA:")
    for i, (idx, val) in enumerate(close.tail(REGIME0_MA_PERIOD).items()):
        print(f"    [{i+1}] {idx} → {val:.4f}")

    print(f"\n  Last close : ${last_close:,.2f}")
    print(f"  MA{REGIME0_MA_PERIOD}        : ${ma:,.2f}")
    print(f"  Long TH    : ${ma * GLOBAL_SYSTEM_REGIME_TH2:,.2f}")
    print(f"  Short TH   : ${ma * GLOBAL_SYSTEM_REGIME_TH1:,.2f}")
    print(f"\n  Manual     : {manual_direction.upper()}")
    print(f"  Function   : {func_direction.upper()}")

    if manual_direction == func_direction:
        print(f"\n  ✅ MATCH")
    else:
        print(f"\n  ❌ MISMATCH")

# =============================================================================
# TEST 2 — get_current_regime() vs manual metrics
# =============================================================================
print(f"\nTEST 2 — Regime ({TIMEFRAME})")
print("-" * 60)

df_tf = fetch_ref_ohlcv(TIMEFRAME)

if df_tf is None or df_tf.empty:
    print(f"  ERROR: Could not fetch {TIMEFRAME} data")
else:
    print(f"  Candles loaded : {len(df_tf)}")
    print(f"  Last candle    : {df_tf.index[-1]}")
    print(f"  Last close     : ${float(pd.to_numeric(df_tf['close'].iloc[-1], errors='coerce')):,.4f}")

    family, metrics = get_current_regime(TIMEFRAME)

    print(f"\n  Regime family  : {family.upper()}")
    if metrics:
        print(f"  Metrics:")
        for k, v in metrics.items():
            print(f"    {k:<25} : {v:.4f}" if v is not None else f"    {k:<25} : None")

    print(f"\n  ✅ Regime calculated successfully" if family != "default" else f"\n  ⚠️  Default regime returned")

print(f"\n{'='*60}\n")