import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import time
import pandas as pd
from itertools import product
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

from shared.backtesters.ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE
from shared.utils.st_tools import prepare_ohlcv_arrays, compile_grid_results, save_all_trades_to_excel, save_results,save_equity_to_excel
from shared.utils.utils import filter_symbols, save_filtered_symbols

from signals.add_signals_flag       import flag_long, flag_short
from signals.add_signals_orderblocks import orderblocks_long, orderblocks_short
from signals.add_signals_parity     import parity_long, parity_short
from signals.add_signals_ranging    import ranging_long, ranging_short
from signals.add_signals_reversal   import reversal_long, reversal_short

# -----------------------------------------------------------------------------
# GLOBAL CONFIG
# -----------------------------------------------------------------------------
DATA_FOLDER  = "../../BOT_batch/data/crypto_2026_OOS"
SYMBOLS_DIR  = "../../BOT_trading/symbols_live"
ORDER_AMOUNT = 80
N_JOBS       = -1
VOL_WINDOW   = 50

# -----------------------------------------------------------------------------
# SIGNAL REGISTRY
# signal_fn    : function to call
# param_names  : ordered list of parameter keys for the grid
# signal_params: keys from strategy config passed to the signal function
# -----------------------------------------------------------------------------
SIGNAL_REGISTRY = {
    "flag_long":         {"fn": flag_long,         "params": ["lookback", "impulse", "flag", "ma_period"]},
    "flag_short":        {"fn": flag_short,         "params": ["lookback", "impulse", "flag", "ma_period"]},
    "orderblocks_long":  {"fn": orderblocks_long,   "params": ["lookback", "tolerance", "impulse"]},
    "orderblocks_short": {"fn": orderblocks_short,  "params": ["lookback", "tolerance", "impulse"]},
    "parity_long":       {"fn": parity_long,        "params": ["lookback", "tolerance", "ma_period"]},
    "parity_short":      {"fn": parity_short,       "params": ["lookback", "tolerance", "ma_period"]},
    "ranging_long":      {"fn": ranging_long,       "params": ["lookback", "tolerance", "ma_period", "ranges"]},
    "ranging_short":     {"fn": ranging_short,      "params": ["lookback", "tolerance", "ma_period", "ranges"]},
    "reversal_long":     {"fn": reversal_long,      "params": ["lookback", "tolerance", "ma_period"]},
    "reversal_short":    {"fn": reversal_short,     "params": ["lookback", "tolerance", "ma_period"]},
}

# -----------------------------------------------------------------------------
# STRATEGIES TABLE
# Each entry is one strategy with fixed (single-value) parameters.
# All values are lists so they plug directly into the grid (single-combo grid).
# -----------------------------------------------------------------------------
STRATEGIES = [
    {
        "id":        "02",
        "name":      "reversal_long_4H",
        "signal":    "reversal_long",
        "timeframe": "4H",
        "sell_after": [0],
        "tp_pct":    [3],
        "sl_pct":    [10],
        "lookback":  [4],
        "ma_period": [50],
        "tolerance": [20],
    },
    {
        "id":        "03",
        "name":      "parity_long_4H",
        "signal":    "parity_long",
        "timeframe": "4H",
        "sell_after": [0],
        "tp_pct":    [3],
        "sl_pct":    [10],
        "lookback":  [150],
        "ma_period": [50],
        "tolerance": [40],
    },
    {
        "id":        "04",
        "name":      "reversal_short_4H",
        "signal":    "reversal_short",
        "timeframe": "4H",
        "sell_after": [0],
        "tp_pct":    [3],
        "sl_pct":    [9],
        "lookback":  [4],
        "ma_period": [50],
        "tolerance": [25],
    },
    {
        "id":        "06",
        "name":      "reversal_long_1H",
        "signal":    "reversal_long",
        "timeframe": "1H",
        "sell_after": [0],
        "tp_pct":    [2],
        "sl_pct":    [10],
        "lookback":  [7],
        "ma_period": [25],
        "tolerance": [40],
    },
    {
        "id":        "07",
        "name":      "reversal_short_1H",
        "signal":    "reversal_short",
        "timeframe": "1H",
        "sell_after": [0],
        "tp_pct":    [2],
        "sl_pct":    [5],
        "lookback":  [5],
        "ma_period": [50],
        "tolerance": [30],
    },
    {
        "id":        "08",
        "name":      "reversal_long_6Hutc",
        "signal":    "reversal_long",
        "timeframe": "6Hutc",
        "sell_after": [0],
        "tp_pct":    [4],
        "sl_pct":    [10],
        "lookback":  [3],
        "ma_period": [50],
        "tolerance": [20],
    },
    {
        "id":        "09",
        "name":      "reversal_short_6Hutc",
        "signal":    "reversal_short",
        "timeframe": "6Hutc",
        "sell_after": [0],
        "tp_pct":    [4],
        "sl_pct":    [7.5],
        "lookback":  [6],
        "ma_period": [25],
        "tolerance": [30],
    },
    {
        "id":        "10",
        "name":      "parity_long_1H",
        "signal":    "parity_long",
        "timeframe": "1H",
        "sell_after": [0],
        "tp_pct":    [2],
        "sl_pct":    [10],
        "lookback":  [150],
        "ma_period": [25],
        "tolerance": [15],
    },
    {
        "id":        "11",
        "name":      "parity_short_1H",
        "signal":    "parity_short",
        "timeframe": "1H",
        "sell_after": [0],
        "tp_pct":    [2],
        "sl_pct":    [7.5],
        "lookback":  [150],
        "ma_period": [50],
        "tolerance": [20],
    },
    {
        "id":        "12",
        "name":      "parity_long_6Hutc",
        "signal":    "parity_long",
        "timeframe": "6Hutc",
        "sell_after": [0],
        "tp_pct":    [3.5],
        "sl_pct":    [10],
        "lookback":  [50],
        "ma_period": [25],
        "tolerance": [40],
    },
    {
        "id":        "13",
        "name":      "orderblocks_short_4H",
        "signal":    "orderblocks_short",
        "timeframe": "4H",
        "sell_after": [0],
        "tp_pct":    [4],
        "sl_pct":    [11],
        "lookback":  [50],
        "impulse":   [0.01],
        "tolerance": [35],
    },
    {
        "id":        "16",
        "name":      "ranging_short_6Hutc",
        "signal":    "ranging_short",
        "timeframe": "6Hutc",
        "sell_after": [0],
        "tp_pct":    [4],
        "sl_pct":    [6],
        "lookback":  [10],
        "ma_period": [25],
        "tolerance": [5],
        "ranges": [25],
    },
    {
        "id":        "17",
        "name":      "flag_long_4H",
        "signal":    "flag_long",
        "timeframe": "4H",
        "sell_after": [0],
        "tp_pct":    [4],
        "sl_pct":    [10],
        "lookback":  [13],
        "flag":      [40],
        "impulse":   [5],
        "ma_period": [50],
    },
    {
        "id":        "19",
        "name":      "flag_short_4H",
        "signal":    "flag_short",
        "timeframe": "4H",
        "sell_after": [0],
        "tp_pct":    [3],
        "sl_pct":    [9],
        "lookback":  [10],
        "flag":      [50],
        "impulse":   [3],
        "ma_period": [50],
    },
    {
        "id":        "20",
        "name":      "flag_short_1H",
        "signal":    "flag_short",
        "timeframe": "1H",
        "sell_after": [0],
        "tp_pct":    [2],
        "sl_pct":    [8],
        "lookback":  [20],
        "flag":      [60],
        "impulse":   [3],
        "ma_period": [25],
    },
]

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def load_custom_symbols(strategy_id: str, strategy_name: str, timeframe: str) -> list[str]:
    """Read symbol list from the corresponding CSV file."""
    filename = f"symbols_live_{strategy_id}_{strategy_name}_{timeframe}.csv"
    filepath = os.path.join(SYMBOLS_DIR, filename)
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Symbols file not found: {filepath}")
    df = pd.read_csv(filepath, header=None)
    return df.iloc[:, 0].dropna().str.strip().tolist()


def build_param_grid(strategy: dict, signal_params: list[str]) -> tuple[list[str], list]:
    """Extract parameter names and their value lists for the grid."""
    grid_param_names = ["sell_after", "tp_pct", "sl_pct"] + signal_params
    lists_for_grid   = [
        strategy.get("sell_after", [0]),
        strategy["tp_pct"],
        strategy["sl_pct"],
        *[strategy[p] for p in signal_params],
    ]
    return grid_param_names, lists_for_grid


def make_process_combo(ohlcv_arr, signal_fn, signal_param_keys, grid_param_names):
    """Return a process_combo closure bound to the current strategy's data."""
    def process_combo(comb):
        params       = dict(zip(grid_param_names, comb))
        signal_kwargs = {k: params[k] for k in signal_param_keys}
        ohlcv_arrays = {}

        for sym, arr in ohlcv_arr.items():
            signals = signal_fn(arr, live_trading=False, **signal_kwargs)
            ohlcv_arrays[sym] = {**arr, "signal": signals}

        results = run_grid_backtest(
            ohlcv_arrays,
            sell_after=params["sell_after"],
            tp_pct=params["tp_pct"],
            sl_pct=params["sl_pct"],
            order_amount=ORDER_AMOUNT,
        )
        return comb, results

    return process_combo


# -----------------------------------------------------------------------------
# MAIN RUNNER
# -----------------------------------------------------------------------------

def run_strategy(strategy: dict) -> None:
    sid       = strategy["id"]
    sname     = strategy["name"]
    timeframe = strategy["timeframe"]
    signal_key = strategy["signal"]

    registry      = SIGNAL_REGISTRY[signal_key]
    signal_fn     = registry["fn"]
    signal_params = registry["params"]

    strategy_label = f"{sid}_{sname}"
    print(f"\n{'='*60}")
    print(f"  Running: {strategy_label}")
    print(f"{'='*60}")

    # Load symbols
    custom_symbols = load_custom_symbols(sid, sname, timeframe)

    # Load & filter data
    all_files     = [f.split("_")[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{timeframe}.parquet")]
    ohlcv_data, _ = filter_symbols(
        all_files,
        min_vol_usdt=0,
        timeframe=timeframe,
        data_folder=DATA_FOLDER,
        min_price=MIN_PRICE,
        vol_window=VOL_WINDOW,
        my_symbols=True,
        custom_symbols=custom_symbols,
    )
    ohlcv_arr = prepare_ohlcv_arrays(ohlcv_data)
    missing = [s for s in custom_symbols if s not in ohlcv_data.keys()]
    print(f"Missing parquets: {missing}")

    # Build grid
    grid_param_names, lists_for_grid = build_param_grid(strategy, signal_params)
    all_combinations                  = list(product(*lists_for_grid))
    process_combo                     = make_process_combo(ohlcv_arr, signal_fn, signal_params, grid_param_names)

    # Run backtest
    with tqdm_joblib(tqdm(desc=f"🔄 {strategy_label}", total=len(all_combinations))):
        grid_results_list = Parallel(n_jobs=N_JOBS)(
            delayed(process_combo)(comb) for comb in all_combinations
        )

    # Compile & save
    grid_records    = compile_grid_results(grid_results_list, grid_param_names, INITIAL_BALANCE)
    grid_results_df = pd.DataFrame(grid_records)

    save_all_trades_to_excel(
        grid_results_list, grid_param_names,
        f"all_trades_{strategy_label}.xlsx",
        strategy_name=strategy_label,
        save=True,
        output_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "brief_trades"),
    )
    save_equity_to_excel(
        grid_results_list,
        folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "brief_equities"),
        initial_capital=INITIAL_BALANCE,
        strategy_name=strategy_label,
        save_file=True,
        output_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "brief_equities"),
    )

    print(f"✅ Done: {strategy_label}")


def main():
    total_start = time.time()

    for strategy in STRATEGIES:
        try:
            run_strategy(strategy)
        except Exception as e:
            print(f"❌ Error in strategy {strategy['id']}_{strategy['name']}: {e}")

    elapsed = int(time.time() - total_start)
    print(f"\n🏁 Total execution time: {elapsed//3600}h {(elapsed%3600)//60}m {elapsed%60}s")


if __name__ == "__main__":
    main()