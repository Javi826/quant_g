import os
import sys

import numpy as np
import pandas as pd

BACKTESTERS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "bitget", "shared", "shared_batchs", "backtesters"
)
sys.path.insert(0, BACKTESTERS_PATH)

from ZX_compute_oo import run_grid_backtest as run_grid_backtest_oo
from ZX_compute_bb import run_grid_backtest as run_grid_backtest_bb


def build_ohlcv_arrays():
    base = np.datetime64("2024-01-01T18:00:00", "ns")
    ts = base + np.arange(6) * np.timedelta64(1, "h")

    open_  = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    high   = np.array([100.0, 110.0, 100.0, 100.0, 100.0, 100.0])
    low    = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    close  = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    signal = np.array([1, 1, 0, 0, 0, 0])

    high_time = ts.copy()
    low_time  = ts.copy()

    return {
        "BTCUSDT": {
            "ts": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "signal": signal,
            "high_time": high_time,
            "low_time": low_time,
        }
    }


def print_trade_log(label, results):
    trade_log = results["__PORTFOLIO__"]["trade_log"]
    print(f"\n=== {label} ===")
    print(f"num_signals_executed: {results['__PORTFOLIO__']['num_signals']}")
    if trade_log.empty:
        print("trade_log: EMPTY")
    else:
        print(trade_log[["symbol", "buy_time", "buy_price", "sell_time", "sell_price", "exit_reason"]])


def main():
    ohlcv_arrays = build_ohlcv_arrays()
    sell_after = 3
    tp_pct = 5.0
    sl_pct = 50.0
    order_amount = 100.0

    results_oo = run_grid_backtest_oo(ohlcv_arrays, sell_after, tp_pct, sl_pct, order_amount)
    results_bb = run_grid_backtest_bb(ohlcv_arrays, sell_after, tp_pct, sl_pct, order_amount)

    print_trade_log("ZX_compute_oo", results_oo)
    print_trade_log("ZX_compute_bb", results_bb)


if __name__ == "__main__":
    main()