"""
validate_backtest.py
====================
Self-contained backtest validation script.
Run from BOT_batch/ directory.

Usage:
    python validate_backtest.py
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
    if "shared_batchs" in mod:
        del sys.modules[mod]

import numpy as np
import pandas as pd

from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest, COMISION

# =============================================================================
# GLOBAL CONFIG
# =============================================================================

COMI_FACTOR  = COMISION / 100.0   # 0.001
ORDER_AMOUNT = 100.0

# =============================================================================
# A1 — LONG TP
# Columns: open, high, low, close, signal
# signal: 1=long entry (pre-shifted, equivalent to live_trading=False)
# Candle 4 : signal=1  (detected at close of candle 4, shift already applied)
# Candle 5 : entry     (open=100 — this is where the engine opens the trade)
# Candle 6-7: high below TP (104.99) — does not trigger
# Candle 8 : high hits TP (105.00)   — TP triggered here
# sell_after=10, tp_pct=5.0%
# =============================================================================

A1_SELL_AFTER  = 10
A1_TP_PCT      = 5.0
A1_SL_PCT      = 0.0

# Manual calculation inputs — edit these to match your array
A1_ENTRY_PRICE = 100.0   # open of entry candle (candle 5)
A1_TP_PRICE    = A1_ENTRY_PRICE * (1 + A1_TP_PCT / 100)  # derived from entry price and tp_pct

#            open    high     low     close  signal
A1_BARS = [
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 0
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 1
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 2
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 3
    (100.0,  100.0,  100.0,  100.0,   1),   # candle 4  ← signal (pre-shifted)
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 5  ← entry (open=100)
    (100.0,  105.0,  100.0,  100.0,   0),   # candle 6  ← TP hit (high=105.00)
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 7
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 8
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 9
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 10
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 11
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 12
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 13
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 14
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 15
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 16
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 17
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 18
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 19
]

# =============================================================================
# A2 — LONG SL
# Candle 4: signal (pre-shifted). Candle 5: entry (open=100).
# Candle 6-7: low above SL, high below TP — neither triggers.
# Candle 8: low hits SL. TP not reached.
# Tip: to test TP wins, raise high on candle 6 or 7 above tp_price,
#      or lower sl so it doesn't trigger before TP.
# =============================================================================

A2_SELL_AFTER  = 10
A2_TP_PCT      = 5.0
A2_SL_PCT      = 3.0

# Manual calculation inputs
A2_ENTRY_PRICE = 100.0
A2_SL_PRICE    = A2_ENTRY_PRICE * (1 - A2_SL_PCT / 100)  # 97.0
A2_TP_PRICE    = A2_ENTRY_PRICE * (1 + A2_TP_PCT / 100)  # 105.0

#            open    high     low     close  signal
A2_BARS = [
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 0
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 1
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 2
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 3
    (100.0,  100.0,  100.0,  100.0,   1),   # candle 4  ← signal (pre-shifted)
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 5  ← entry (open=100)
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 6  — high below TP, low above SL
    (100.0,  105.0,  100.0,  100.0,   0),   # candle 7  — TP reachable here if needed
    (100.0,  100.0,   97.0,  100.0,   0),   # candle 8  ← SL hit (low=97.00)
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 9
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 10
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 11
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 12
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 13
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 14
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 15
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 16
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 17
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 18
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 19
]

# =============================================================================
# A3 — SHORT TP / tie-break
# signal: -1=short entry (pre-shifted)
# Candle 4: signal. Candle 5: entry (open=100).
# TP is BELOW entry: open * (1 - tp_pct/100) = 95.0
# SL is ABOVE entry: open * (1 + sl_pct/100) = 103.0
# Candle 6: both TP and SL hit — tie-break decided by high_time vs low_time
#
# Tuple format extended: (open, high, low, close, signal, high_time_offset_min, low_time_offset_min)
# high_time_offset_min / low_time_offset_min: minutes after candle open when high/low was reached
# Example: (100, 103, 90, 100, 0, 10, 30) → high reached at +10min, low at +30min → SL hits first
#          (100, 103, 90, 100, 0, 30, 10) → low reached at +10min, high at +30min → TP hits first
#          (100, 103, 90, 100, 0,  0,  0) → same timestamp → TP wins (engine tie-break rule)
# =============================================================================

A3_SELL_AFTER  = 10
A3_TP_PCT      = 5.0
A3_SL_PCT      = 3.0

# Manual calculation inputs
A3_ENTRY_PRICE = 100.0
A3_TP_PRICE    = A3_ENTRY_PRICE * (1 - A3_TP_PCT / 100)  # 95.0
A3_SL_PRICE    = A3_ENTRY_PRICE * (1 + A3_SL_PCT / 100)  # 103.0

#            open    high     low     close  signal
A3_BARS = [
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 0
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 1
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 2
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 3
    (100.0,  100.0,  100.0,  100.0,  -1),   # candle 4  ← signal (pre-shifted, short)
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 5  ← entry (open=100)
    #(100.0,  103.0,   90.0,  100.0,   0,   0,   0),   # candle 6  — tie: high_time==low_time → TP wins
    #(100.0,  103.0,   90.0,  100.0,   0,  10,  30),  # candle 6  — high+10min, low+30min → SL wins first
    (100.0,  103.0,   90.0,  100.0,   0,  5,  10),  # candle 6  — high+30min, low+10min → TP wins first
    (100.0,  103.0,  100.0,  100.0,   0),   # candle 7
    (100.0,  103.0,  100.0,  100.0,   0),   # candle 8
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 9
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 10
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 11
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 12
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 13
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 14
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 15
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 16
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 17
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 18
    (100.0,  100.0,  100.0,  100.0,   0),   # candle 19
]

# =============================================================================
# HELPERS
# =============================================================================

PASS = "✅ PASS"
FAIL = "❌ FAIL"
results: list[dict] = []

def _record(block: str, name: str, passed: bool, detail: str = "") -> None:
    results.append({
        "block":  block,
        "check":  name,
        "status": PASS if passed else FAIL,
        "detail": detail,
    })

def _approx(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol

def _make_ts(n: int, freq_h: int = 1) -> np.ndarray:
    base = np.datetime64("2024-01-01T00:00:00", "ns")
    step = np.timedelta64(freq_h * 3_600_000_000_000, "ns")
    return np.array([base + i * step for i in range(n)], dtype="datetime64[ns]")

def _bars_to_arr(bars: list[tuple]) -> dict:
    """
    Convert explicit bar list to ohlcv dict expected by run_grid_backtest.
    Tuple format: (open, high, low, close, signal) — basic
                  (open, high, low, close, signal, high_time_offset_min, low_time_offset_min) — extended
    high_time_offset_min / low_time_offset_min: minutes after bar open when high/low was reached.
    Default 0 (same as bar open timestamp) when not provided.
    """
    n       = len(bars)
    ts      = _make_ts(n)
    step_ns = np.timedelta64(60_000_000_000, "ns")  # 1 minute in nanoseconds

    high_times = np.array([
        ts[i] + int(b[5]) * step_ns if len(b) > 5 else ts[i]
        for i, b in enumerate(bars)
    ], dtype="datetime64[ns]")

    low_times = np.array([
        ts[i] + int(b[6]) * step_ns if len(b) > 6 else ts[i]
        for i, b in enumerate(bars)
    ], dtype="datetime64[ns]")

    return {
        "ts":        ts,
        "open":      np.array([b[0] for b in bars], dtype=np.float64),
        "high":      np.array([b[1] for b in bars], dtype=np.float64),
        "low":       np.array([b[2] for b in bars], dtype=np.float64),
        "close":     np.array([b[3] for b in bars], dtype=np.float64),
        "volume":    np.zeros(n, dtype=np.float64),
        "high_time": high_times,
        "low_time":  low_times,
        "signal":    np.array([b[4] for b in bars], dtype=np.int8),
    }

def _run(ohlcv_arrays: dict, sell_after: int, tp_pct: float, sl_pct: float) -> pd.DataFrame:
    result    = run_grid_backtest(ohlcv_arrays, sell_after=sell_after, tp_pct=tp_pct, sl_pct=sl_pct, order_amount=ORDER_AMOUNT)
    trade_log = result["__PORTFOLIO__"]["trade_log"].copy()
    trade_log.columns = trade_log.columns.str.lower().str.strip()
    return trade_log

# =============================================================================
# BLOCK A1 — LONG TP
# =============================================================================

def validate_a1() -> None:
    block  = "A1 — Long TP"
    arr    = _bars_to_arr(A1_BARS)
    trades = _run({"SYM_A1": arr}, sell_after=A1_SELL_AFTER, tp_pct=A1_TP_PCT, sl_pct=A1_SL_PCT)

    signal_candle     = next(i for i, b in enumerate(A1_BARS) if b[4] != 0)
    expected_entry_ts = arr["ts"][signal_candle]

    tp_hit_candle     = next(i for i, b in enumerate(A1_BARS) if b[1] >= A1_BARS[4][0] * (1 + A1_TP_PCT / 100))
    expected_exit_ts  = arr["ts"][tp_hit_candle]

    # Manual profit calculation
    qty             = ORDER_AMOUNT / A1_ENTRY_PRICE
    commission_buy  = ORDER_AMOUNT * COMI_FACTOR
    commission_sell = qty * A1_TP_PRICE * COMI_FACTOR
    expected_profit = (A1_TP_PRICE - A1_ENTRY_PRICE) * qty - commission_buy - commission_sell

    if len(trades) == 1:
        row            = trades.iloc[0]
        actual_buy_ts  = pd.Timestamp(row["buy_time"]).to_datetime64().astype("datetime64[ns]")
        actual_sell_ts = pd.Timestamp(row["sell_time"]).to_datetime64().astype("datetime64[ns]")
        _record(block, "entry candle == signal_candle (pre-shifted)",
                actual_buy_ts == expected_entry_ts,
                f"actual={pd.Timestamp(actual_buy_ts)} expected={pd.Timestamp(expected_entry_ts)}")
        _record(block, "exit candle == first candle where high >= TP",
                actual_sell_ts == expected_exit_ts,
                f"actual={pd.Timestamp(actual_sell_ts)} expected={pd.Timestamp(expected_exit_ts)}")
        _record(block, "profit matches manual calculation",
                abs(row["profit"] - expected_profit) <= 1e-6,
                f"actual={row['profit']:.6f} expected={expected_profit:.6f}")
    else:
        _record(block, "entry candle == signal_candle (pre-shifted)", False, "no trades produced")
        _record(block, "exit candle == first candle where high >= TP",  False, "no trades produced")
        _record(block, "profit matches manual calculation",             False, "no trades produced")

# =============================================================================
# BLOCK A2 — LONG SL
# =============================================================================

def validate_a2() -> None:
    block  = "A2 — Long SL"
    arr    = _bars_to_arr(A2_BARS)
    trades = _run({"SYM_A2": arr}, sell_after=A2_SELL_AFTER, tp_pct=A2_TP_PCT, sl_pct=A2_SL_PCT)

    signal_candle     = next(i for i, b in enumerate(A2_BARS) if b[4] != 0)
    expected_entry_ts = arr["ts"][signal_candle]
    sl_hit_candle     = next((i for i, b in enumerate(A2_BARS) if b[2] <= A2_SL_PRICE), None)
    tp_hit_candle     = next((i for i, b in enumerate(A2_BARS) if b[1] >= A2_TP_PRICE), None)

    # First trigger wins
    if sl_hit_candle and tp_hit_candle:
        expected_exit_candle = min(sl_hit_candle, tp_hit_candle)
        expected_exit_reason = "SL" if sl_hit_candle <= tp_hit_candle else "TP"
        expected_exec_price  = A2_SL_PRICE if sl_hit_candle <= tp_hit_candle else A2_TP_PRICE
    elif sl_hit_candle:
        expected_exit_candle = sl_hit_candle
        expected_exit_reason = "SL"
        expected_exec_price  = A2_SL_PRICE
    elif tp_hit_candle:
        expected_exit_candle = tp_hit_candle
        expected_exit_reason = "TP"
        expected_exec_price  = A2_TP_PRICE
    else:
        expected_exit_candle = None
        expected_exit_reason = "SELL_AFTER"
        expected_exec_price  = None

    expected_exit_ts = arr["ts"][expected_exit_candle] if expected_exit_candle else None

    # Manual profit calculation
    qty             = ORDER_AMOUNT / A2_ENTRY_PRICE
    commission_buy  = ORDER_AMOUNT * COMI_FACTOR
    commission_sell = qty * expected_exec_price * COMI_FACTOR if expected_exec_price else 0.0
    expected_profit = (expected_exec_price - A2_ENTRY_PRICE) * qty - commission_buy - commission_sell if expected_exec_price else None

    if len(trades) == 1:
        row            = trades.iloc[0]
        actual_buy_ts  = pd.Timestamp(row["buy_time"]).to_datetime64().astype("datetime64[ns]")
        actual_sell_ts = pd.Timestamp(row["sell_time"]).to_datetime64().astype("datetime64[ns]")
        _record(block, "entry candle == signal_candle (pre-shifted)",
                actual_buy_ts == expected_entry_ts,
                f"actual={pd.Timestamp(actual_buy_ts)} expected={pd.Timestamp(expected_entry_ts)}")
        _record(block, f"exit_reason == {expected_exit_reason}",
                row["exit_reason"] == expected_exit_reason,
                f"actual={row['exit_reason']} expected={expected_exit_reason}")
        if expected_exit_ts is not None:
            _record(block, "exit candle matches first SL/TP hit",
                    actual_sell_ts == expected_exit_ts,
                    f"actual={pd.Timestamp(actual_sell_ts)} expected={pd.Timestamp(expected_exit_ts)}")
        if expected_profit is not None:
            _record(block, "profit matches manual calculation",
                    abs(row["profit"] - expected_profit) <= 1e-6,
                    f"actual={row['profit']:.6f} expected={expected_profit:.6f}")
    else:
        _record(block, "entry candle == signal_candle (pre-shifted)", False, "no trades produced")

# =============================================================================
# BLOCK A3 — SHORT TP
# =============================================================================

def validate_a3() -> None:
    block  = "A3 — Short TP"
    arr    = _bars_to_arr(A3_BARS)
    trades = _run({"SYM_A3": arr}, sell_after=A3_SELL_AFTER, tp_pct=A3_TP_PCT, sl_pct=A3_SL_PCT)

    signal_candle     = next(i for i, b in enumerate(A3_BARS) if b[4] != 0)
    expected_entry_ts = arr["ts"][signal_candle]
    tp_hit_candle     = next((i for i, b in enumerate(A3_BARS) if b[2] <= A3_TP_PRICE), None)
    sl_hit_candle     = next((i for i, b in enumerate(A3_BARS) if b[1] >= A3_SL_PRICE), None)

    # First trigger wins — same candle: use high_time vs low_time to decide
    if tp_hit_candle is not None and sl_hit_candle is not None:
        if tp_hit_candle < sl_hit_candle:
            expected_exit_candle = tp_hit_candle
            expected_exit_reason = "TP"
            expected_exec_price  = A3_TP_PRICE
        elif sl_hit_candle < tp_hit_candle:
            expected_exit_candle = sl_hit_candle
            expected_exit_reason = "SL"
            expected_exec_price  = A3_SL_PRICE
        else:
            # Same candle — winner decided by high_time vs low_time
            c        = tp_hit_candle
            high_t   = arr["high_time"][c]
            low_t    = arr["low_time"][c]
            tp_wins  = low_t <= high_t   # for SHORT: low=TP, high=SL
            expected_exit_candle = c
            expected_exit_reason = "TP" if tp_wins else "SL"
            expected_exec_price  = A3_TP_PRICE if tp_wins else A3_SL_PRICE
    elif tp_hit_candle is not None:
        expected_exit_candle = tp_hit_candle
        expected_exit_reason = "TP"
        expected_exec_price  = A3_TP_PRICE
    elif sl_hit_candle is not None:
        expected_exit_candle = sl_hit_candle
        expected_exit_reason = "SL"
        expected_exec_price  = A3_SL_PRICE
    else:
        expected_exit_candle = None
        expected_exit_reason = "SELL_AFTER"
        expected_exec_price  = None

    expected_exit_ts = arr["ts"][expected_exit_candle] if expected_exit_candle else None

    # Manual profit for SHORT: (entry - exec) * qty - commissions
    qty             = ORDER_AMOUNT / A3_ENTRY_PRICE
    commission_buy  = ORDER_AMOUNT * COMI_FACTOR
    commission_sell = qty * expected_exec_price * COMI_FACTOR if expected_exec_price else 0.0
    expected_profit = (A3_ENTRY_PRICE - expected_exec_price) * qty - commission_buy - commission_sell if expected_exec_price else None

    if len(trades) == 1:
        row            = trades.iloc[0]
        actual_buy_ts  = pd.Timestamp(row["buy_time"]).to_datetime64().astype("datetime64[ns]")
        actual_sell_ts = pd.Timestamp(row["sell_time"]).to_datetime64().astype("datetime64[ns]")
        _record(block, "entry candle == signal_candle (pre-shifted)",
                actual_buy_ts == expected_entry_ts,
                f"actual={pd.Timestamp(actual_buy_ts)} expected={pd.Timestamp(expected_entry_ts)}")
        _record(block, f"exit_reason == {expected_exit_reason}",
                row["exit_reason"] == expected_exit_reason,
                f"actual={row['exit_reason']} expected={expected_exit_reason}")
        if expected_exit_ts is not None:
            _record(block, "exit candle matches first TP/SL hit",
                    actual_sell_ts == expected_exit_ts,
                    f"actual={pd.Timestamp(actual_sell_ts)} expected={pd.Timestamp(expected_exit_ts)}")
        if expected_profit is not None:
            _record(block, "profit matches manual calculation",
                    abs(row["profit"] - expected_profit) <= 1e-6,
                    f"actual={row['profit']:.6f} expected={expected_profit:.6f}")

        # Tie-break check: when TP and SL hit same candle, winner depends on high_time vs low_time
        same_candle = tp_hit_candle is not None and sl_hit_candle is not None and tp_hit_candle == sl_hit_candle
        if same_candle:
            c          = tp_hit_candle
            high_t     = arr["high_time"][c]
            low_t      = arr["low_time"][c]
            if high_t == low_t:
                expected_winner = "TP"   # engine tie-break: tp_time <= sl_time → TP
            elif low_t < high_t:
                expected_winner = "TP"   # low (TP for short) reached first
            else:
                expected_winner = "SL"   # high (SL for short) reached first
            _record(block, f"tie-break: {expected_winner} wins (high_time={high_t} low_time={low_t})",
                    row["exit_reason"] == expected_winner,
                    f"actual={row['exit_reason']} expected={expected_winner}")
    else:
        _record(block, "entry candle == signal_candle (pre-shifted)", False, f"got {len(trades)} trades")

# =============================================================================
# REPORT
# =============================================================================

def _print_report() -> None:
    sep = "─" * 110
    print(f"\n{'═'*110}")
    print(f"  BACKTEST VALIDATION REPORT")
    print(f"{'═'*110}")
    print(f"  {'BLOCK':<20} {'CHECK':<50} {'STATUS':<12} DETAIL")
    print(f"  {sep}")

    current_block = None
    for r in results:
        if r["block"] != current_block:
            if current_block is not None:
                print(f"  {sep}")
            current_block = r["block"]
        detail = r["detail"][:50] if r["detail"] else ""
        print(f"  {r['block']:<20} {r['check']:<50} {r['status']:<12} {detail}")

    print(f"  {sep}")
    n_pass  = sum(1 for r in results if r["status"] == PASS)
    n_fail  = sum(1 for r in results if r["status"] == FAIL)
    verdict = "✅ ALL CHECKS PASSED" if n_fail == 0 else f"❌ {n_fail} CHECK(S) FAILED"
    print(f"\n  {verdict}  ({n_pass} passed / {n_fail} failed / {len(results)} total)")
    print(f"{'═'*110}\n")

# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import time
    t0 = time.time()

    print("\n  Running A1 — Long TP ...")
    validate_a1()

    print("  Running A2 — Long SL ...")
    validate_a2()

    print("  Running A3 — Short TP ...")
    validate_a3()

    _print_report()
    print(f"  Elapsed: {time.time() - t0:.1f}s\n")