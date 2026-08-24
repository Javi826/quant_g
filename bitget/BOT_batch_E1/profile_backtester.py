#profile_backtester.py
import os
import sys
import time
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_batch")))

from shared_batchs.symbols.universe import filter_symbols, select_universe, select_top_n_by_volume
from shared_batchs.setup.config_paths import DATA_FOLDER_IS
from shared_batchs.setup.config_backtest import MIN_PRICE, ORDER_AMOUNT
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_batchs.rule_mining.rule_generator import generate_all_rules
from shared_batchs.backtesters.ZX_compute_BT import (
    prepare_static_arrays,
    prepare_signal_arrays,
    backtest_core,
)
from shared_config import VOLUME_COL
from signals.condition_bank import ConditionBank

# =============================================================================
# CLI ARGS
# =============================================================================
parser = argparse.ArgumentParser()
parser.add_argument("--timeframe", type=str, default="1H")
parser.add_argument("--n_symbols", type=int, default=10)
parser.add_argument("--n_rules", type=int, default=300)
parser.add_argument("--max_depth", type=int, default=None)
parser.add_argument("--shared_banks", action="store_true", default=True,
                     help="Reuse one ConditionBank per symbol across all rules (new production path). "
                          "Omit this flag to reproduce the old per-rule bank=None behavior.")
args = parser.parse_args()

TIMEFRAME  = args.timeframe
N_SYMBOLS  = args.n_symbols
N_RULES    = args.n_rules

PARAM_GRID = {
    "SELL_AFTER": [50],
    "TP_PCT":     [6, 8, 10],
    "SL_PCT":     [6, 8],
}

COMI_FACTOR     = 0.001
INITIAL_BALANCE = 10000.0


# =============================================================================
# TIMER HELPER
# =============================================================================
class Timer:
    def __init__(self):
        self.totals = {}

    def add(self, key, seconds):
        self.totals[key] = self.totals.get(key, 0.0) + seconds

    def report(self, n_rules):
        grand_total = sum(self.totals.values())
        rows = sorted(self.totals.items(), key=lambda x: -x[1])
        key_width = max([len(k) for k in self.totals] + [len("TOTAL")]) + 2

        print(f"\n{'=' * 90}")
        print(f"PROFILING REPORT — timeframe={TIMEFRAME} n_symbols={N_SYMBOLS} n_rules={n_rules}")
        print(f"{'=' * 90}")
        header = f"  {'STAGE':<{key_width}}{'TOTAL(s)':>12}{'PCT':>9}{'PER_RULE(ms)':>16}"
        print(header)
        print(f"  {'-' * (len(header) - 2)}")
        for key, seconds in rows:
            pct = (seconds / grand_total * 100) if grand_total else 0.0
            per_rule_ms = (seconds / n_rules * 1000) if n_rules else 0.0
            print(f"  {key:<{key_width}}{seconds:12.3f}{pct:8.1f}%{per_rule_ms:15.3f} ms")
        print(f"  {'-' * (len(header) - 2)}")
        print(f"  {'TOTAL':<{key_width}}{grand_total:12.3f}")
        print(f"{'=' * 90}\n")


timer = Timer()

# =============================================================================
# LOAD DATA
# =============================================================================
t0 = time.perf_counter()
ohlcv_is = select_universe(
    data_folder_is    = DATA_FOLDER_IS,
    timeframe         = TIMEFRAME,
    min_price         = MIN_PRICE,
    filter_symbols_fn = filter_symbols,
)
ohlcv_is  = select_top_n_by_volume(ohlcv_is, N_SYMBOLS)
ohlcv_arr = prepare_ohlcv_arrays(ohlcv_is)
timer.add("data_loading", time.perf_counter() - t0)

# =============================================================================
# STATIC BUNDLE (built once, same as the shared-memory cache path)
# =============================================================================
t0 = time.perf_counter()
static_bundle = prepare_static_arrays(ohlcv_arr)
timer.add("prepare_static_once", time.perf_counter() - t0)

# =============================================================================
# CONDITION BANKS — one per symbol, built once, reused across all rules
# (mirrors _get_condition_banks in the production backtest_runner.py path)
# =============================================================================
t0 = time.perf_counter()
condition_banks = {sym: ConditionBank(arr) for sym, arr in ohlcv_arr.items()} if args.shared_banks else None
timer.add("prepare_condition_banks_once", time.perf_counter() - t0)

print(f"shared_banks={'ON (new production path)' if args.shared_banks else 'OFF (old per-rule bank=None)'}")

# =============================================================================
# GENERATE RULES
# =============================================================================
arr_sample = next(iter(ohlcv_arr.values()))
t0 = time.perf_counter()
all_rules = generate_all_rules({
    "open":  arr_sample["open"],
    "high":  arr_sample["high"],
    "low":   arr_sample["low"],
    "close": arr_sample["close"],
    VOLUME_COL: arr_sample[VOLUME_COL],
}, max_depth=args.max_depth) if args.max_depth else generate_all_rules({
    "open":  arr_sample["open"],
    "high":  arr_sample["high"],
    "low":   arr_sample["low"],
    "close": arr_sample["close"],
    VOLUME_COL: arr_sample[VOLUME_COL],
})
timer.add("rule_generation_total", time.perf_counter() - t0)

sample_rules = all_rules[:N_RULES]
print(f"Sampling {len(sample_rules)} rules out of {len(all_rules)} generated.")

keys   = list(PARAM_GRID.keys())
combos = []
for sell_after in PARAM_GRID["SELL_AFTER"]:
    for tp in PARAM_GRID["TP_PCT"]:
        for sl in PARAM_GRID["SL_PCT"]:
            combos.append({"SELL_AFTER": sell_after, "TP_PCT": tp, "SL_PCT": sl})

# =============================================================================
# PER-RULE BREAKDOWN — signal_fn / prepare / backtest_core
# =============================================================================
total_trades_all = 0

for rule in sample_rules:
    signal_fn = rule["signal_fn"]

    # ---- signal_fn ----
    t0 = time.perf_counter()
    ohlcv_arrays_for_rule = {}
    for sym, arr in ohlcv_arr.items():
        bank    = condition_banks.get(sym) if condition_banks else None
        signals = signal_fn(arr, live_trading=False, bank=bank)
        ohlcv_arrays_for_rule[sym] = {**arr, "signal": np.asarray(signals, dtype=np.float32)}
    timer.add("signal_fn", time.perf_counter() - t0)

    # ---- prepare_signal_arrays (using cached static bundle, real prod path) ----
    t0 = time.perf_counter()
    prepared_data = prepare_signal_arrays(static_bundle, ohlcv_arrays_for_rule)
    timer.add("prepare_signal_arrays", time.perf_counter() - t0)

    prepared_arrays = prepared_data[7]
    (open_2d, close_2d, high_2d, low_2d,
     high_time_2d, low_time_2d, ts_int_2d, signal_2d, sym_len,
     signal_events, all_timestamps_int, ev_col0) = prepared_arrays

    # ---- backtest_core, summed over all combos (same as _run_full_period_for_rule) ----
    t0 = time.perf_counter()
    for combo in combos:
        core_output = backtest_core(
            open_2d, close_2d, high_2d, low_2d,
            high_time_2d, low_time_2d, ts_int_2d, signal_2d, sym_len,
            signal_events, all_timestamps_int, ev_col0,
            INITIAL_BALANCE, COMI_FACTOR, float(ORDER_AMOUNT),
            int(combo["SELL_AFTER"]), float(combo["TP_PCT"]), float(combo["SL_PCT"]),
        )
        total_trades_all += core_output[0]
    timer.add("backtest_core", time.perf_counter() - t0)

print(f"Total trades executed across all rules/combos: {total_trades_all}")

# =============================================================================
# JOBLIB / IPC OVERHEAD — serial vs single-rule parallel dispatch
# =============================================================================
from joblib import Parallel, delayed

def _noop_task(x):
    return x

t0 = time.perf_counter()
for i in range(min(50, N_RULES)):
    _noop_task(i)
serial_baseline = time.perf_counter() - t0

t0 = time.perf_counter()
Parallel(n_jobs=-1, batch_size=1)(delayed(_noop_task)(i) for i in range(min(50, N_RULES)))
parallel_overhead = time.perf_counter() - t0

timer.add("ipc_overhead_50_noop_tasks", max(parallel_overhead - serial_baseline, 0.0))

timer.report(len(sample_rules))