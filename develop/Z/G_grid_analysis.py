# grid_search_is_reversal.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared", "shared_batch")))

import time
import numpy as np
import pandas as pd
from itertools import product

from shared_batchs.utils.utils import filter_symbols
from shared_batchs.backtesters.ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE
from shared_batchs.utils.torque import prepare_ohlcv_arrays, compile_grid_results
from signals.add_signals_reversal import reversal_long
from signals.add_signals_reversal import reversal_short

start_time = time.time()

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
SPLIT_MODE = "expanding"
SPLIT_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "data_pipeline", "data", "04_split", SPLIT_MODE)

DATA_FOLDERS = {
    "IS"   : os.path.join(SPLIT_BASE, "IS",  "crypto_2024-01_2025-05_IS"),
    "OOS1" : os.path.join(SPLIT_BASE, "OOS", "crypto_2025-05_2026-05_OOS"),
    "OOS2" : os.path.join(SPLIT_BASE, "OOS", "crypto_2022-01_2023-01_OOS"),
    "OOS3" : os.path.join(SPLIT_BASE, "OOS", "crypto_2023-01_2024-01_OOS"),
}

TIMEFRAME    = "4H"
ORDER_AMOUNT = 80
MIN_VOL_USDT = 10_000_000
N_SYMBOLS    = 10
MY_SYMBOLS   = False

# -----------------------------------------------------------------------------
# SIGNAL FUNCTION — comment/uncomment as needed
# -----------------------------------------------------------------------------
signal_fn = reversal_long
# signal_fn = reversal_short

# -----------------------------------------------------------------------------
# PARAMETER GRIDS
# -----------------------------------------------------------------------------
GRIDS = {
    "LARGE": {
        "SELL_AFTER" : [0],
        "LOOKBACK"   : [2, 3, 5, 6],
        "TOLERANCE"  : [30, 40, 50],
        "MA_PERIOD"  : [10, 25],
        "TP_PCT"     : [2, 3, 4, 5],
        "SL_PCT"     : [7, 8, 9, 10],
    },
    "SMALL": {
        "SELL_AFTER" : [0],
        "LOOKBACK"   : [2, 3, 5, 6],
        "TOLERANCE"  : [30, 40, 50],
        "MA_PERIOD"  : [10, 25],
        "TP_PCT"     : [2, 3, 4],
        "SL_PCT"     : [2, 3],
    },
}

SIGNAL_PARAMS = ["LOOKBACK", "TOLERANCE", "MA_PERIOD"]

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------
def load_symbols(data_folder, reference_symbols=None):
    symbols = [
        f.split('_')[0]
        for f in os.listdir(data_folder)
        if f.endswith(f"_{TIMEFRAME}.parquet")
    ]
    if reference_symbols is not None:
        symbols = [s for s in symbols if s in reference_symbols]

    ohlcv_data, _ = filter_symbols(
        symbols,
        min_vol_usdt=0 if reference_symbols else MIN_VOL_USDT,
        timeframe=TIMEFRAME,
        data_folder=data_folder,
        min_price=MIN_PRICE,
        vol_window=50,
        my_symbols=MY_SYMBOLS if reference_symbols is None else False
    )
    selected = sorted(ohlcv_data.keys())[:N_SYMBOLS] if reference_symbols is None else sorted(ohlcv_data.keys())
    return {s: ohlcv_data[s] for s in selected}


def run_grid(ohlcv_arrays, param_grid):
    param_names    = list(param_grid.keys())
    lists_for_grid = [param_grid[k] for k in param_names]
    results        = []

    for comb in product(*lists_for_grid):
        params = dict(zip(param_names, comb))

        ohlcv_with_signals = {}
        for sym, arr in ohlcv_arrays.items():
            signal_kwargs = {k.lower(): params[k] for k in SIGNAL_PARAMS}
            signals = signal_fn(arr, **signal_kwargs, live_trading=False)
            ohlcv_with_signals[sym] = {**arr, 'signal': signals}

        result = run_grid_backtest(
            ohlcv_with_signals,
            sell_after=params['SELL_AFTER'],
            tp_pct=params['TP_PCT'],
            sl_pct=params['SL_PCT'],
            order_amount=ORDER_AMOUNT
        )
        results.append((tuple(params[p] for p in param_names), result))

    df = pd.DataFrame(compile_grid_results(results, param_names, INITIAL_BALANCE))
    return df.sort_values("Net_Gain_pct", ascending=False).reset_index(drop=True)


def print_top10(label, df, param_names):
    display_cols = param_names + ["Net_Gain_pct", "Win_Ratio", "DD_pct", "Num_Trades", "Sharpe"]
    print(f"\n{'=' * 90}")
    print(f"  TOP 10 — {label}")
    print(f"{'=' * 90}")
    print(df[display_cols].head(10).to_string(index=False))
    best = df.iloc[0]
    print(f"\n  BEST: " + " | ".join(f"{p}={best[p]}" for p in param_names))
    print(f"  Net_Gain_pct={best['Net_Gain_pct']:.2f}%  Win_Ratio={best['Win_Ratio']:.2f}  DD_pct={best['DD_pct']:.2f}%  Trades={int(best['Num_Trades'])}")


def print_oos_rank(grid_label, best_is_comb, param_names, oos_dfs):
    print(f"\n  [{grid_label}] IS best: " + " | ".join(f"{p}={best_is_comb[i]}" for i, p in enumerate(param_names)))
    for label, df_oos in oos_dfs.items():
        mask  = np.all(df_oos[param_names].values == list(best_is_comb), axis=1)
        match = df_oos[mask]
        if not match.empty:
            rank = match.index[0] + 1
            row  = match.iloc[0]
            print(f"    {label}: rank {rank}/{len(df_oos)} — Net_Gain_pct={row['Net_Gain_pct']:.2f}%  DD_pct={row['DD_pct']:.2f}%  Win_Ratio={row['Win_Ratio']:.2f}")
        else:
            print(f"    {label}: combination not found in OOS grid")


# -----------------------------------------------------------------------------
# LOAD IS SYMBOLS (reference universe)
# -----------------------------------------------------------------------------
ohlcv_is   = load_symbols(DATA_FOLDERS["IS"])
is_symbols = set(ohlcv_is.keys())
print(f"IS Symbols ({len(ohlcv_is)}): {sorted(is_symbols)}")

# Preload OOS arrays (shared across grids)
ohlcv_oos_all = {
    label: load_symbols(folder, reference_symbols=is_symbols)
    for label, folder in [("OOS1", DATA_FOLDERS["OOS1"]), ("OOS2", DATA_FOLDERS["OOS2"]), ("OOS3", DATA_FOLDERS["OOS3"])]
}

# -----------------------------------------------------------------------------
# RUN BOTH GRIDS
# -----------------------------------------------------------------------------
summary = {}

for grid_label, param_grid in GRIDS.items():
    param_names    = list(param_grid.keys())
    n_combinations = int(np.prod([len(v) for v in param_grid.values()]))
    print(f"\n{'#' * 90}")
    print(f"  GRID: {grid_label} — {n_combinations} combinations")
    print(f"{'#' * 90}")

    # IS
    df_is         = run_grid(prepare_ohlcv_arrays(ohlcv_is), param_grid)
    print_top10(f"IS [{grid_label}]", df_is, param_names)
    best_is_comb  = tuple(df_is.iloc[0][p] for p in param_names)

    # OOS — run grid on each period
    oos_dfs = {}
    for label, ohlcv_oos in ohlcv_oos_all.items():
        df_oos         = run_grid(prepare_ohlcv_arrays(ohlcv_oos), param_grid)
        oos_dfs[label] = df_oos
        print_top10(f"{label} [{grid_label}]", df_oos, param_names)

    summary[grid_label] = (best_is_comb, param_names, oos_dfs)

# -----------------------------------------------------------------------------
# SUMMARY — IS best params rank in each OOS for both grids
# -----------------------------------------------------------------------------
print(f"\n{'=' * 90}")
print(f"  SUMMARY — IS BEST PARAMS RANK IN EACH OOS")
print(f"{'=' * 90}")
for grid_label, (best_is_comb, param_names, oos_dfs) in summary.items():
    print_oos_rank(grid_label, best_is_comb, param_names, oos_dfs)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600}h {(elapsed%3600)//60}m {elapsed%60}s")