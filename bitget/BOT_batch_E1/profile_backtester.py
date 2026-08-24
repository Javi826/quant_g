#verify_lexsort_argsort.py
import os
import sys
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "shared_batch")))

from shared_batchs.symbols.universe import filter_symbols, select_universe, select_top_n_by_volume
from shared_batchs.setup.config_paths import DATA_FOLDER_IS
from shared_batchs.setup.config_backtest import MIN_PRICE
from shared_batchs.utils.ohlcv_utils import prepare_ohlcv_arrays
from shared_batchs.rule_mining.rule_generator import generate_all_rules
from shared_batchs.backtesters.ZX_compute_BT import prepare_static_arrays, prepare_signal_arrays as prepare_signal_arrays_original
from shared_config import VOLUME_COL

parser = argparse.ArgumentParser()
parser.add_argument("--timeframe", type=str, default="1H")
parser.add_argument("--n_symbols", type=int, default=10)
parser.add_argument("--n_rules", type=int, default=300)
args = parser.parse_args()

TIMEFRAME = args.timeframe
N_SYMBOLS = args.n_symbols
N_RULES   = args.n_rules

# =============================================================================
# LOAD DATA
# =============================================================================
ohlcv_is  = select_universe(
    data_folder_is    = DATA_FOLDER_IS,
    timeframe         = TIMEFRAME,
    min_price         = MIN_PRICE,
    filter_symbols_fn = filter_symbols,
)
ohlcv_is  = select_top_n_by_volume(ohlcv_is, N_SYMBOLS)
ohlcv_arr = prepare_ohlcv_arrays(ohlcv_is)

static_bundle = prepare_static_arrays(ohlcv_arr)

symbols_dict_order = list(ohlcv_arr.keys())
sym_ids            = static_bundle["sym_ids"]
symbols_sid_order  = sorted(symbols_dict_order, key=lambda s: sym_ids[s])

print(f"symbols (dict/insertion order): {symbols_dict_order}")
print(f"symbols (sid-ascending order) : {symbols_sid_order}")
if symbols_dict_order == symbols_sid_order:
    print("WARNING: dict order already matches sid order for this universe — "
          "this run does NOT exercise the ordering risk. Consider a different "
          "--n_symbols or universe to get a non-trivial test case.")
else:
    print("Dict order DIFFERS from sid order — this run DOES exercise the risk case.")

# =============================================================================
# MODIFIED VARIANT — argsort(ts, kind='stable'), iterating symbols in
# sid-ascending order so ties resolve identically to lexsort((sid, ts)).
# Local copy only, for verification — production .pyx untouched.
# =============================================================================
def prepare_signal_arrays_argsort(static_bundle, ohlcv_arrays):
    symbols = static_bundle["symbols"]
    sym_ids = static_bundle["sym_ids"]
    sym_len = static_bundle["sym_len"]
    symbols_sorted_by_sid = sorted(symbols, key=lambda s: sym_ids[s])

    sym_data = {}
    for sym in symbols:
        data   = ohlcv_arrays[sym]
        sid    = sym_ids[sym]
        n      = int(sym_len[sid])
        ts_int = static_bundle["ts_int_arrays"][sym]
        sym_data[sym] = {
            'ts_int': ts_int,
            'signal': data['signal'][:n],
            'len':    n,
        }

    event_chunks = []
    for sym in symbols_sorted_by_sid:
        sid        = sym_ids[sym]
        d          = sym_data[sym]
        signal_arr = d['signal']
        n          = d['len']

        sig_idxs = np.flatnonzero(signal_arr)
        sig_idxs = sig_idxs[sig_idxs < n]
        if sig_idxs.size:
            ts_ints = d['ts_int'][sig_idxs]
            chunk   = np.empty((sig_idxs.size, 3), dtype=np.int64)
            chunk[:, 0] = ts_ints
            chunk[:, 1] = sid
            chunk[:, 2] = sig_idxs
            event_chunks.append(chunk)

    if event_chunks:
        signal_events = np.concatenate(event_chunks, axis=0)
        order         = np.argsort(signal_events[:, 0], kind="stable")
        signal_events = signal_events[order]
    else:
        signal_events = np.empty((0, 3), dtype=np.int64)

    return signal_events

# =============================================================================
# GENERATE RULES + COMPARE
# =============================================================================
arr_sample = next(iter(ohlcv_arr.values()))
all_rules = generate_all_rules({
    "open":  arr_sample["open"],
    "high":  arr_sample["high"],
    "low":   arr_sample["low"],
    "close": arr_sample["close"],
    VOLUME_COL: arr_sample[VOLUME_COL],
})
sample_rules = all_rules[:N_RULES]
print(f"\nSampling {len(sample_rules)} rules out of {len(all_rules)} generated. Comparing signal_events...")

n_mismatches   = 0
n_empty        = 0
first_mismatch = None

for i, rule in enumerate(sample_rules):
    signal_fn = rule["signal_fn"]
    ohlcv_arrays_for_rule = {}
    for sym, arr in ohlcv_arr.items():
        signals = signal_fn(arr, live_trading=False)
        ohlcv_arrays_for_rule[sym] = {**arr, "signal": np.asarray(signals, dtype=np.float32)}

    prepared_original = prepare_signal_arrays_original(static_bundle, ohlcv_arrays_for_rule)
    signal_events_original = prepared_original[7][9]  # arrays tuple index 9 = signal_events

    signal_events_new = prepare_signal_arrays_argsort(static_bundle, ohlcv_arrays_for_rule)

    if signal_events_original.shape[0] == 0 and signal_events_new.shape[0] == 0:
        n_empty += 1
        continue

    identical = np.array_equal(signal_events_original, signal_events_new)
    if not identical:
        n_mismatches += 1
        if first_mismatch is None:
            first_mismatch = (i, rule["label"], signal_events_original, signal_events_new)

print(f"\n{'=' * 70}")
print(f"Rules compared      : {len(sample_rules)}")
print(f"Empty on both sides  : {n_empty}")
print(f"MISMATCHES          : {n_mismatches}")
print(f"{'=' * 70}")

if n_mismatches == 0:
    print("RESULT (natural order): IDENTICAL for every sampled rule.")
else:
    idx, label, orig, new = first_mismatch
    print("RESULT (natural order): MISMATCH FOUND.")
    print(f"First mismatching rule [{idx}]: {label}")
    print(f"original.shape={orig.shape}  new.shape={new.shape}")
    diff_rows = min(orig.shape[0], new.shape[0])
    for r in range(min(diff_rows, 10)):
        if not np.array_equal(orig[r], new[r]):
            print(f"  row {r}: original={orig[r].tolist()}  new={new[r].tolist()}")

# =============================================================================
# FORCED-DIVERGENCE TEST — artificially reorder the input dict (same data,
# different insertion order) so dict order no longer matches sid-ascending
# order. This is the actual risk case the natural universe order did not
# exercise above.
# =============================================================================
print(f"\n{'=' * 70}")
print("FORCED-DIVERGENCE TEST — reordering symbol dict insertion order")
print(f"{'=' * 70}")

ohlcv_arr_reordered = {sym: ohlcv_arr[sym] for sym in reversed(list(ohlcv_arr.keys()))}
static_bundle_reordered = prepare_static_arrays(ohlcv_arr_reordered)

symbols_dict_order_r = static_bundle_reordered["symbols"]
sym_ids_r            = static_bundle_reordered["sym_ids"]
symbols_sid_order_r  = sorted(symbols_dict_order_r, key=lambda s: sym_ids_r[s])

print(f"reordered dict order : {symbols_dict_order_r}")
print(f"sid-ascending order  : {symbols_sid_order_r}")
if symbols_dict_order_r == symbols_sid_order_r:
    print("WARNING: reordering did not break the coincidence either — "
          "try a universe with non-alphabetically-sortable symbol names, "
          "or manually shuffle instead of reversing.")
else:
    print("Reordered dict order DIFFERS from sid order — divergence risk is now exercised.")

n_mismatches_r   = 0
n_empty_r        = 0
first_mismatch_r = None

for i, rule in enumerate(sample_rules):
    signal_fn = rule["signal_fn"]
    ohlcv_arrays_for_rule_r = {}
    for sym, arr in ohlcv_arr_reordered.items():
        signals = signal_fn(arr, live_trading=False)
        ohlcv_arrays_for_rule_r[sym] = {**arr, "signal": np.asarray(signals, dtype=np.float32)}

    prepared_original_r = prepare_signal_arrays_original(static_bundle_reordered, ohlcv_arrays_for_rule_r)
    signal_events_original_r = prepared_original_r[7][9]

    signal_events_new_r = prepare_signal_arrays_argsort(static_bundle_reordered, ohlcv_arrays_for_rule_r)

    if signal_events_original_r.shape[0] == 0 and signal_events_new_r.shape[0] == 0:
        n_empty_r += 1
        continue

    identical_r = np.array_equal(signal_events_original_r, signal_events_new_r)
    if not identical_r:
        n_mismatches_r += 1
        if first_mismatch_r is None:
            first_mismatch_r = (i, rule["label"], signal_events_original_r, signal_events_new_r)

print(f"\nRules compared      : {len(sample_rules)}")
print(f"Empty on both sides  : {n_empty_r}")
print(f"MISMATCHES          : {n_mismatches_r}")

if n_mismatches_r == 0:
    print("RESULT (reordered dict): IDENTICAL for every sampled rule. "
          "Safe to apply the .pyx change.")
else:
    idx, label, orig, new = first_mismatch_r
    print("RESULT (reordered dict): MISMATCH FOUND — do NOT apply the .pyx change as-is.")
    print(f"First mismatching rule [{idx}]: {label}")
    print(f"original.shape={orig.shape}  new.shape={new.shape}")
    diff_rows = min(orig.shape[0], new.shape[0])
    for r in range(min(diff_rows, 10)):
        if not np.array_equal(orig[r], new[r]):
            print(f"  row {r}: original={orig[r].tolist()}  new={new[r].tolist()}")