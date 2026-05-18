"""
validate_regime.py
==================
Self-contained regime filter validation script.
Run from BOT_batch/ directory.

Usage:
    python validate_regime.py
"""

import os
import sys

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "market_regime")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_batch")))

import importlib
for mod in list(sys.modules.keys()):
    if any(x in mod for x in ("shared_batchs", "shared_batch", "market_regime", "strategies_files")):
        del sys.modules[mod]

import glob
import numpy as np
import pandas as pd
from importlib import import_module

from shared_batchs.regime.regime_filter import build_metrics_cache
from shared_batchs.regime.regime_config import (
    REGIME_LOOKBACK_BARS, REGIME_REFERENCE, REGIME_FAMILY_SOURCE,
    REGIME0_MA_PERIOD as R0_MA, REGIME0_LONG_TH as R0_LTH, REGIME0_SHORT_TH as R0_STH,
)
from shared.shared_trading_batch.config_trading_batch import (
    REGIME_ATR_WINDOW   as ATR_WINDOW,
    REGIME_PE_WINDOW    as PE_WINDOW,
    REGIME_PE_ORDER     as PE_ORDER,
    REGIME_HURST_WINDOW as HURST_WINDOW,
    REGIME_ER_WINDOW    as ER_WINDOW,
    REGIME_FAMILIES     as FAMILIES,
)
from shared.shared_batch_develop.market_regime.regime_analysis import (
    load_reference_symbol_for_timeframe,
    filter_signals_by_regime,
    build_direction_cache,
    classify_trade_by_family,
)
from shared_batchs.registry.signal_registry import SIGNAL_REGISTRY
from shared_batchs.utils.torque import prepare_ohlcv_arrays
from shared_batchs.utils.utils import filter_symbols

# =============================================================================
# CONFIGURATION
# =============================================================================

SPLIT_MODE       = "expanding"
SPLIT_BASE       = os.path.join(os.path.dirname(__file__), "..", "data_pipeline", "data", "04_split", SPLIT_MODE)
DATA_FOLDER_OOS1 = os.path.join(SPLIT_BASE, "OOS", "crypto_2025-05_2026-05_OOS")

# Full BTC history for regime cache warmup (no missing cache keys)
DATA_FOLDER_FULL = os.path.join(os.path.dirname(__file__), "..", "data_pipeline", "data",
                                "04_split", SPLIT_MODE, "IS", "crypto_full_IS")

STRATEGY_ID          = "20_flag_short_1H"
STRATEGIES_SET       = "E1"
STRATEGIES_LOOP_NAME = "strategies_loop_E1_01"
N_SYMBOLS            = 3      # number of OOS1 symbols to load
N_ROWS_TO_SHOW       = 15     # rows in sample table

# =============================================================================
# HARDCODED BINS — real bins for strategy 20_flag_short_1H
# Set to empty set to test REGIME=1 (no filtering)
# =============================================================================

BINS_TO_FILTER = {
    "trending_uptrend",
    "ranging_uptrend",
    "volatile_uptrend",
}

# =============================================================================
# HELPERS
# =============================================================================

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results: list[dict] = []

def _record(block: str, name: str, passed: bool, detail: str = "") -> None:
    results.append({"block": block, "check": name, "status": PASS if passed else FAIL, "detail": detail})

def _load_strategy(strategy_id: str) -> dict:
    batch    = import_module(f"strategies_files.files_{STRATEGIES_SET}.strategies_BT_{STRATEGIES_SET}_batch").STRATEGIES
    loop     = import_module(f"strategies_files.files_{STRATEGIES_SET}.{STRATEGIES_LOOP_NAME}").STRATEGIES_LOOP
    loop_map = {s["id"]: s for s in loop}
    for s in batch:
        if s["id"] == strategy_id:
            return {**s, **loop_map.get(strategy_id, {})}
    raise ValueError(f"Strategy {strategy_id} not found")

# =============================================================================
# R3 — SIGNAL COMPOSITION
# When regime filters a timestamp (family_direction in bins_to_filter),
# ALL symbols must have composed signal = 0 for that timestamp.
# When regime does not filter, composed signal == baseline signal.
# =============================================================================

def validate_r3() -> None:
    block = "R3 — Signal composition"

    strategy           = _load_strategy(STRATEGY_ID)
    timeframe          = strategy["timeframe"]
    signal_key         = "_".join(strategy["name"].split("_")[:-1])
    registry           = SIGNAL_REGISTRY[signal_key]
    signal_fn          = registry["fn"]
    signal_params_keys = registry["params"]
    signal_params      = {k: strategy.get(k) or strategy.get(k.upper()) for k in signal_params_keys}

    # Load reference symbol using full history for cache warmup
    ref_cache = {}
    ref_tf_full = load_reference_symbol_for_timeframe(DATA_FOLDER_FULL, REGIME_REFERENCE, timeframe, ref_cache) \
                  if REGIME_FAMILY_SOURCE == "strategy" \
                  else load_reference_symbol_for_timeframe(DATA_FOLDER_FULL, REGIME_REFERENCE, "1Dutc", ref_cache)
    ref_1d_full = load_reference_symbol_for_timeframe(DATA_FOLDER_FULL, REGIME_REFERENCE, "1Dutc", ref_cache)

    # Trim ref data to OOS1 period only (to simulate what happens in production)
    oos1_files   = glob.glob(os.path.join(DATA_FOLDER_OOS1, f"*_{timeframe}.parquet"))
    oos1_syms    = [os.path.basename(f).replace(f"_{timeframe}.parquet", "") for f in oos1_files][:N_SYMBOLS]
    ohlcv_dfs, _ = filter_symbols(
        symbols      = oos1_syms,
        min_vol_usdt = 0,
        timeframe    = timeframe,
        data_folder  = DATA_FOLDER_OOS1,
        min_price    = 0.0001,
        vol_window   = 50,
        my_symbols   = False,
    )
    arr_arrays = prepare_ohlcv_arrays(ohlcv_dfs)

    # OOS1 date range
    sample_arr  = next(iter(arr_arrays.values()))
    oos1_start  = pd.Timestamp(sample_arr["ts"][0])
    oos1_end    = pd.Timestamp(sample_arr["ts"][-1])

    # Trim ref_tf to OOS1 period for the metrics cache
    ref_tf_oos1 = ref_tf_full[
        (pd.to_datetime(ref_tf_full["ts"]) >= oos1_start) &
        (pd.to_datetime(ref_tf_full["ts"]) <= oos1_end)
    ].reset_index(drop=True)

    # Build metrics cache using full BTC history (so first OOS1 candles are covered)
    metrics_cache = build_metrics_cache(
        ref_df       = ref_tf_full,
        lookback     = REGIME_LOOKBACK_BARS,
        hurst_window = HURST_WINDOW,
        er_window    = ER_WINDOW,
        atr_window   = ATR_WINDOW,
        pe_window    = PE_WINDOW,
        pe_order     = PE_ORDER,
    )
    # Keep only keys within OOS1 period
    metrics_cache = {k: v for k, v in metrics_cache.items() if oos1_start <= k <= oos1_end}

    _record(block, "Metrics cache covers OOS1 period",
            len(metrics_cache) > 0,
            f"entries={len(metrics_cache)}")

    # Compute baseline and composed signals
    baseline = {}
    composed = {}
    for sym, arr in arr_arrays.items():
        sigs_base = signal_fn(arr, **signal_params, live_trading=False)
        sigs_comp = filter_signals_by_regime(
            signals        = sigs_base.copy(),
            ts             = arr["ts"],
            ref_1d_df      = ref_1d_full,
            ref_tf_df      = ref_tf_full,
            bins_to_filter = BINS_TO_FILTER,
            ma_period      = R0_MA,
            long_th        = R0_LTH,
            short_th       = R0_STH,
            families       = FAMILIES,
            lookback_bars  = REGIME_LOOKBACK_BARS,
            hurst_window   = HURST_WINDOW,
            er_window      = ER_WINDOW,
            atr_window     = ATR_WINDOW,
            pe_window      = PE_WINDOW,
            pe_order       = PE_ORDER,
            metrics_cache  = metrics_cache,
        )
        baseline[sym] = sigs_base
        composed[sym]  = sigs_comp

    # Build direction cache for all signal timestamps
    all_signal_ts = pd.Series(dtype="datetime64[ns]")
    for sym, arr in arr_arrays.items():
        idxs = np.nonzero(baseline[sym])[0]
        if len(idxs) > 0:
            all_signal_ts = pd.concat([all_signal_ts, pd.Series(pd.to_datetime(arr["ts"][idxs]))], ignore_index=True)
    dir_cache = build_direction_cache(ref_1d_full, R0_MA, R0_LTH, R0_STH, all_signal_ts)

    # Collect all timestamps with at least one baseline signal
    all_ts    = next(iter(arr_arrays.values()))["ts"]
    sig_times = set()
    for sym, sigs in baseline.items():
        sig_times.update(pd.to_datetime(arr_arrays[sym]["ts"][np.nonzero(sigs)[0]]).tolist())
    sig_times = sorted(sig_times)

    # For each timestamp, determine regime and expected composed value
    # Print sample: show timestamps where regime FILTERS (bin active)
    filtered_times = []
    for t in sig_times:
        t_ts      = pd.Timestamp(t)
        metrics   = metrics_cache.get(t_ts)
        direction = dir_cache.get(t_ts, "unknown")
        if metrics is None or direction == "unknown":
            continue
        family = classify_trade_by_family(metrics, FAMILIES)
        if f"{family}_{direction}" in BINS_TO_FILTER:
            filtered_times.append((t_ts, family, direction))

    passed_times = []
    for t in sig_times:
        t_ts      = pd.Timestamp(t)
        metrics   = metrics_cache.get(t_ts)
        direction = dir_cache.get(t_ts, "unknown")
        if metrics is None or direction == "unknown":
            continue
        family = classify_trade_by_family(metrics, FAMILIES)
        if f"{family}_{direction}" not in BINS_TO_FILTER:
            passed_times.append((t_ts, family, direction))

    # Print filtered sample
    print(f"\n  Strategy: {STRATEGY_ID}  |  Symbols: {list(arr_arrays.keys())}")
    print(f"  Bins to filter: {BINS_TO_FILTER}")
    print(f"  Filtered timestamps: {len(filtered_times)}  |  Passed timestamps: {len(passed_times)}")

    print(f"\n  --- FILTERED TIMESTAMPS (regime active → all composed must be 0) ---")
    print(f"  {'timestamp':<28} {'family':<12} {'direction':<12}", end="")
    for sym in arr_arrays:
        print(f"  {sym[:10]+'_base':<16} {sym[:10]+'_comp':<16}", end="")
    print()
    print(f"  {'-'*120}")

    shown = 0
    for t_ts, family, direction in filtered_times[:N_ROWS_TO_SHOW]:
        row = f"  {str(t_ts):<28} {family:<12} {direction:<12}"
        for sym, arr in arr_arrays.items():
            idx  = np.searchsorted(arr["ts"], t_ts.to_datetime64())
            base = int(baseline[sym][idx]) if idx < len(baseline[sym]) else 0
            comp = int(composed[sym][idx])  if idx < len(composed[sym])  else 0
            row += f"  {str(base):<16} {str(comp):<16}"
        print(row)
        shown += 1

    print(f"\n  --- PASSED TIMESTAMPS (regime inactive → composed must equal baseline) ---")
    print(f"  {'timestamp':<28} {'family':<12} {'direction':<12}", end="")
    for sym in arr_arrays:
        print(f"  {sym[:10]+'_base':<16} {sym[:10]+'_comp':<16}", end="")
    print()
    print(f"  {'-'*120}")

    shown = 0
    for t_ts, family, direction in passed_times[:N_ROWS_TO_SHOW]:
        row = f"  {str(t_ts):<28} {family:<12} {direction:<12}"
        for sym, arr in arr_arrays.items():
            idx  = np.searchsorted(arr["ts"], t_ts.to_datetime64())
            base = int(baseline[sym][idx]) if idx < len(baseline[sym]) else 0
            comp = int(composed[sym][idx])  if idx < len(composed[sym])  else 0
            row += f"  {str(base):<16} {str(comp):<16}"
        print(row)
        shown += 1
        if shown >= N_ROWS_TO_SHOW:
            break
    print()

    # Check 1: filtered timestamps → all composed == 0
    filter_violations = []
    for t_ts, family, direction in filtered_times:
        for sym, arr in arr_arrays.items():
            idx  = np.searchsorted(arr["ts"], t_ts.to_datetime64())
            comp = int(composed[sym][idx]) if idx < len(composed[sym]) else 0
            if comp != 0:
                filter_violations.append(f"{sym}@{t_ts}")

    _record(block, "Filtered timestamps: all symbols composed == 0",
            len(filter_violations) == 0,
            f"{len(filter_violations)} violations" if filter_violations else f"checked {len(filtered_times)} timestamps")

    # Check 2: passed timestamps → composed == baseline
    pass_violations = []
    for t_ts, family, direction in passed_times:
        for sym, arr in arr_arrays.items():
            idx  = np.searchsorted(arr["ts"], t_ts.to_datetime64())
            base = int(baseline[sym][idx]) if idx < len(baseline[sym]) else 0
            comp = int(composed[sym][idx])  if idx < len(composed[sym])  else 0
            if base != comp:
                pass_violations.append(f"{sym}@{t_ts} base={base} comp={comp}")

    _record(block, "Passed timestamps: composed == baseline for all symbols",
            len(pass_violations) == 0,
            f"{len(pass_violations)} violations" if pass_violations else f"checked {len(passed_times)} timestamps")

# =============================================================================
# REPORT
# =============================================================================

def _print_report() -> None:
    sep = "─" * 110
    print(f"\n{'═'*110}")
    print(f"  REGIME VALIDATION REPORT")
    print(f"{'═'*110}")
    print(f"  {'BLOCK':<30} {'CHECK':<50} {'STATUS':<12} DETAIL")
    print(f"  {sep}")

    current_block = None
    for r in results:
        if r["block"] != current_block:
            if current_block is not None:
                print(f"  {sep}")
            current_block = r["block"]
        detail = r["detail"][:45] if r["detail"] else ""
        print(f"  {r['block']:<30} {r['check']:<50} {r['status']:<12} {detail}")

    print(f"  {sep}")
    n_pass = sum(1 for r in results if r["status"] == PASS)
    n_fail = sum(1 for r in results if r["status"] == FAIL)
    verdict = "✅ ALL CHECKS PASSED" if n_fail == 0 else f"❌ {n_fail} CHECK(S) FAILED"
    print(f"\n  {verdict}  ({n_pass} passed / {n_fail} failed / {len(results)} total)")
    print(f"{'═'*110}\n")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import time
    t0 = time.time()

    print("\n  Running R3 — Signal composition ...")
    validate_r3()

    _print_report()
    print(f"  Elapsed: {time.time() - t0:.1f}s\n")