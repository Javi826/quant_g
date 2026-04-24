"""
test_metrics_cache.py
---------------------
Verifies that build_metrics_cache returns identical results to calc_all_metrics_at_time.
Run from BOT_batch/ directory.
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_market_regime")))

import numpy as np
import pandas as pd
from regime_common import calc_all_metrics_at_time, build_metrics_cache, load_btc_for_timeframe
from shared_config import (
    REGIME_HURST_WINDOW as HURST_WINDOW,
    REGIME_ER_WINDOW    as ER_WINDOW,
    REGIME_ATR_WINDOW   as ATR_WINDOW,
    REGIME_PE_WINDOW    as PE_WINDOW,
    REGIME_PE_ORDER     as PE_ORDER,
)

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
SPLIT_MODE   = "expanding"
SPLIT_BASE   = os.path.join(os.path.dirname(__file__), "..", "data_pipeline", "data", "04_split", SPLIT_MODE)
DATA_FOLDER  = os.path.join(SPLIT_BASE, "OOS", "crypto_2025-04_2026-04_OOS")
TIMEFRAME    = "1Dutc"
LOOKBACK     = 100
N_SAMPLES    = 20  # number of random timestamps to compare

# ---------------------------------------------------------------------------
# LOAD BTC
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  METRICS CACHE TEST")
print(f"{'='*60}")

btc_cache = {}
btc_df = load_btc_for_timeframe(DATA_FOLDER, TIMEFRAME, btc_cache)
print(f"  Loaded {len(btc_df)} BTC bars")

# ---------------------------------------------------------------------------
# BUILD CACHE
# ---------------------------------------------------------------------------
print(f"\n  Building metrics cache...")
import time
t0 = time.time()
cache = build_metrics_cache(
    btc_df       = btc_df,
    lookback     = LOOKBACK,
    hurst_window = HURST_WINDOW,
    er_window    = ER_WINDOW,
    atr_window   = ATR_WINDOW,
    pe_window    = PE_WINDOW,
    pe_order     = PE_ORDER,
)
t_cache = time.time() - t0
print(f"  Cache built: {len(cache)} entries in {t_cache:.2f}s")

# ---------------------------------------------------------------------------
# COMPARE CACHE vs calc_all_metrics_at_time on N random timestamps
# ---------------------------------------------------------------------------
sample_ts = btc_df['ts'].iloc[LOOKBACK:LOOKBACK + N_SAMPLES * 5:5].tolist()

print(f"\n{'─'*80}")
print(f"  COMPARISON — {len(sample_ts)} samples")
print(f"{'─'*80}")
print(f"  {'Timestamp':<25} {'Metric':<22} {'Cache':>10} {'Direct':>10} {'Match':>7}")
print(f"  {'-'*75}")

metrics_keys = ['hurst', 'efficiency_ratio', 'atr_pct', 'permutation_entropy']
mismatches   = 0
t0 = time.time()

for ts in sample_ts:
    # Cache lookup — find closest bar at or before ts
    cache_ts = max((k for k in cache if k <= pd.Timestamp(ts)), default=None)
    if cache_ts is None:
        continue
    cached = cache[cache_ts]

    # Direct calculation
    direct = calc_all_metrics_at_time(
        btc_df       = btc_df,
        buy_time     = pd.Timestamp(ts),
        lookback     = LOOKBACK,
        hurst_window = HURST_WINDOW,
        er_window    = ER_WINDOW,
        atr_window   = ATR_WINDOW,
        pe_window    = PE_WINDOW,
        pe_order     = PE_ORDER,
    )

    if direct is None:
        continue

    for key in metrics_keys:
        c_val = cached.get(key, np.nan)
        d_val = direct.get(key, np.nan)
        if pd.isna(c_val) and pd.isna(d_val):
            match = "✅"
        elif pd.isna(c_val) or pd.isna(d_val):
            match = "❌ NaN diff"
            mismatches += 1
        elif abs(c_val - d_val) < 1e-6:
            match = "✅"
        else:
            match = f"❌ diff={abs(c_val - d_val):.6f}"
            mismatches += 1
        print(f"  {str(ts):<25} {key:<22} {c_val:>10.4f} {d_val:>10.4f} {match:>7}")

t_direct = time.time() - t0

print(f"\n{'='*60}")
print(f"  SUMMARY")
print(f"{'='*60}")
print(f"  Cache build time   : {t_cache:.2f}s  ({len(cache)} bars)")
print(f"  Direct calc time   : {t_direct:.2f}s  ({len(sample_ts)} samples)")
print(f"  Mismatches         : {mismatches}")
print(f"  Result             : {'✅ PASS' if mismatches == 0 else '❌ FAIL'}")
print(f"{'='*60}\n")