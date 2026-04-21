#develop/Z_grid/G_grid_central.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget")))
import time
import pandas as pd
from itertools import product
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed

from shared.backtesters.ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE
from shared.utils.st_tools import prepare_ohlcv_arrays, compile_grid_results, save_all_trades_to_csv, save_results, save_equity_to_excel
from shared.utils.utils import filter_symbols, save_filtered_symbols

from signals.add_signals_flag        import flag_long, flag_short
from signals.add_signals_orderblocks import orderblocks_long, orderblocks_short
from signals.add_signals_parity      import parity_long, parity_short
from signals.add_signals_reversal    import reversal_long, reversal_short

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "shared", "shared_market_regime")))
from regime_common import load_btc_for_timeframe, filter_signals_by_regime

# -----------------------------------------------------------------------------
# GLOBAL CONFIG
# -----------------------------------------------------------------------------
SPLIT_MODE   = "expanding"
SPLIT_BASE   = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "data_pipeline", "data", "04_split", SPLIT_MODE)
DATA_FOLDER  = os.path.join(SPLIT_BASE, "OOS", "crypto_2025-04_2026-04_OOS")
# DATA_FOLDER = os.path.join(SPLIT_BASE, "IS",  "crypto_2022-01_2025-01_IS")
BTC_FOLDER   = DATA_FOLDER
SYMBOLS_DIR  = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "BOT_batch", "symbols_live_old")
ORDER_AMOUNT = 80
N_JOBS       = -1
VOL_WINDOW   = 50

# -----------------------------------------------------------------------------
# REGIME FILTER CONFIG
# Set enabled=False to disable completely
# -----------------------------------------------------------------------------
REGIME_FILTER = {
    'enabled':   True,
    'ma_period': 5,     # BTC MA period for macro direction
    'long_th':   1.0,   # BTC uptrend threshold
    'short_th':  1.0,   # BTC downtrend threshold
}

# Family source for regime filter: 'macro' = BTC 1D | 'strategy' = BTC at strategy timeframe
REGIME_FAMILY_SOURCE = 'macro'

FAMILIES = {
    'trending': {'hurst': ('>', 0.55), 'efficiency_ratio': ('>', 0.4)},
    'volatile': {'atr_pct': ('>', 2.0), 'permutation_entropy': ('>', 0.2)},
    'ranging':  {}
}

REGIME_LOOKBACK_BARS = 100
REGIME_MA_PERIOD     = 50
REGIME_HURST_WINDOW  = 100
REGIME_ER_WINDOW     = 14
REGIME_ATR_WINDOW    = 14
REGIME_PE_WINDOW     = 50
REGIME_PE_ORDER      = 3

# -----------------------------------------------------------------------------
# SIGNAL REGISTRY
# -----------------------------------------------------------------------------
SIGNAL_REGISTRY = {
    "flag_long":         {"fn": flag_long,         "params": ["lookback", "impulse", "flag", "ma_period"]},
    "flag_short":        {"fn": flag_short,         "params": ["lookback", "impulse", "flag", "ma_period"]},
    "orderblocks_long":  {"fn": orderblocks_long,   "params": ["lookback", "tolerance", "impulse"]},
    "orderblocks_short": {"fn": orderblocks_short,  "params": ["lookback", "tolerance", "impulse"]},
    "parity_long":       {"fn": parity_long,        "params": ["lookback", "tolerance", "ma_period"]},
    "parity_short":      {"fn": parity_short,       "params": ["lookback", "tolerance", "ma_period"]},
    "reversal_long":     {"fn": reversal_long,      "params": ["lookback", "tolerance", "ma_period"]},
    "reversal_short":    {"fn": reversal_short,     "params": ["lookback", "tolerance", "ma_period"]},
}

# -----------------------------------------------------------------------------
# STRATEGIES TABLE
# -----------------------------------------------------------------------------
STRATEGIES = [
    {
        "id":               "02",
        "name":             "reversal_long_4H",
        "signal":           "reversal_long",
        "timeframe":        "4H",
        "direction_mode":   "long_only",
        "regime_trending":  1,
        "regime_ranging":   1,
        "regime_volatile":  0,
        "sell_after":       [0],
        "tp_pct":           [3],
        "sl_pct":           [10],
        "lookback":         [4],
        "ma_period":        [50],
        "tolerance":        [20],
    },
    {
        "id":               "03",
        "name":             "parity_long_4H",
        "signal":           "parity_long",
        "timeframe":        "4H",
        "direction_mode":   "long_only",
        "regime_trending":  1,
        "regime_ranging":   1,
        "regime_volatile":  0,
        "sell_after":       [0],
        "tp_pct":           [3],
        "sl_pct":           [10],
        "lookback":         [150],
        "ma_period":        [50],
        "tolerance":        [40],
    },
    {
        "id":               "04",
        "name":             "reversal_short_4H",
        "signal":           "reversal_short",
        "timeframe":        "4H",
        "direction_mode":   "short_only",
        "regime_trending":  1,
        "regime_ranging":   1,
        "regime_volatile":  0,
        "sell_after":       [0],
        "tp_pct":           [3],
        "sl_pct":           [9],
        "lookback":         [4],
        "ma_period":        [50],
        "tolerance":        [25],
    },
    {
        "id":               "06",
        "name":             "reversal_long_1H",
        "signal":           "reversal_long",
        "timeframe":        "1H",
        "direction_mode":   "long_only",
        "regime_trending":  1,
        "regime_ranging":   1,
        "regime_volatile":  0,
        "sell_after":       [0],
        "tp_pct":           [2],
        "sl_pct":           [10],
        "lookback":         [7],
        "ma_period":        [25],
        "tolerance":        [40],
    },
    {
        "id":               "07",
        "name":             "reversal_short_1H",
        "signal":           "reversal_short",
        "timeframe":        "1H",
        "direction_mode":   "short_only",
        "regime_trending":  1,
        "regime_ranging":   1,
        "regime_volatile":  0,
        "sell_after":       [0],
        "tp_pct":           [2],
        "sl_pct":           [5],
        "lookback":         [5],
        "ma_period":        [50],
        "tolerance":        [30],
    },
    {
        "id":               "08",
        "name":             "reversal_long_6Hutc",
        "signal":           "reversal_long",
        "timeframe":        "6Hutc",
        "direction_mode":   "long_only",
        "regime_trending":  1,
        "regime_ranging":   1,
        "regime_volatile":  0,
        "sell_after":       [0],
        "tp_pct":           [4],
        "sl_pct":           [10],
        "lookback":         [3],
        "ma_period":        [50],
        "tolerance":        [20],
    },
    {
        "id":               "09",
        "name":             "reversal_short_6Hutc",
        "signal":           "reversal_short",
        "timeframe":        "6Hutc",
        "direction_mode":   "short_only",
        "regime_trending":  1,
        "regime_ranging":   1,
        "regime_volatile":  0,
        "sell_after":       [0],
        "tp_pct":           [4],
        "sl_pct":           [7.5],
        "lookback":         [6],
        "ma_period":        [25],
        "tolerance":        [30],
    },
    {
        "id":               "10",
        "name":             "parity_long_1H",
        "signal":           "parity_long",
        "timeframe":        "1H",
        "direction_mode":   "long_only",
        "regime_trending":  1,
        "regime_ranging":   1,
        "regime_volatile":  0,
        "sell_after":       [0],
        "tp_pct":           [2],
        "sl_pct":           [10],
        "lookback":         [150],
        "ma_period":        [25],
        "tolerance":        [15],
    },
    {
        "id":               "11",
        "name":             "parity_short_1H",
        "signal":           "parity_short",
        "timeframe":        "1H",
        "direction_mode":   "short_only",
        "regime_trending":  1,
        "regime_ranging":   1,
        "regime_volatile":  0,
        "sell_after":       [0],
        "tp_pct":           [2],
        "sl_pct":           [7.5],
        "lookback":         [150],
        "ma_period":        [50],
        "tolerance":        [20],
    },
    {
        "id":               "12",
        "name":             "parity_long_6Hutc",
        "signal":           "parity_long",
        "timeframe":        "6Hutc",
        "direction_mode":   "long_only",
        "regime_trending":  1,
        "regime_ranging":   1,
        "regime_volatile":  0,
        "sell_after":       [0],
        "tp_pct":           [3.5],
        "sl_pct":           [10],
        "lookback":         [50],
        "ma_period":        [25],
        "tolerance":        [40],
    },
    {
        "id":               "13",
        "name":             "orderblocks_short_4H",
        "signal":           "orderblocks_short",
        "timeframe":        "4H",
        "direction_mode":   "short_only",
        "regime_trending":  1,
        "regime_ranging":   1,
        "regime_volatile":  0,
        "sell_after":       [0],
        "tp_pct":           [4],
        "sl_pct":           [11],
        "lookback":         [50],
        "impulse":          [0.01],
        "tolerance":        [35],
    },
    {
        "id":               "17",
        "name":             "flag_long_4H",
        "signal":           "flag_long",
        "timeframe":        "4H",
        "direction_mode":   "long_only",
        "regime_trending":  1,
        "regime_ranging":   1,
        "regime_volatile":  0,
        "sell_after":       [0],
        "tp_pct":           [4],
        "sl_pct":           [10],
        "lookback":         [13],
        "flag":             [40],
        "impulse":          [5],
        "ma_period":        [50],
    },
    {
        "id":               "19",
        "name":             "flag_short_4H",
        "signal":           "flag_short",
        "timeframe":        "4H",
        "direction_mode":   "short_only",
        "regime_trending":  1,
        "regime_ranging":   1,
        "regime_volatile":  0,
        "sell_after":       [0],
        "tp_pct":           [3],
        "sl_pct":           [9],
        "lookback":         [10],
        "flag":             [50],
        "impulse":          [3],
        "ma_period":        [50],
    },
    {
        "id":               "20",
        "name":             "flag_short_1H",
        "signal":           "flag_short",
        "timeframe":        "1H",
        "direction_mode":   "short_only",
        "regime_trending":  1,
        "regime_ranging":   1,
        "regime_volatile":  0,
        "sell_after":       [0],
        "tp_pct":           [2],
        "sl_pct":           [8],
        "lookback":         [20],
        "flag":             [60],
        "impulse":          [3],
        "ma_period":        [25],
    },
]

# -----------------------------------------------------------------------------
# HELPERS
# -----------------------------------------------------------------------------

def load_btc_1d() -> pd.DataFrame:
    """Load BTC 1D OHLC for macro direction"""
    from pathlib import Path
    filepath = Path(BTC_FOLDER) / "BTCUSDT_1Dutc.parquet"
    if not filepath.exists():
        raise FileNotFoundError(f"BTC 1D not found: {filepath}")
    df = pd.read_parquet(filepath)
    df.columns = df.columns.str.lower()
    df['ts'] = pd.to_datetime(df['timestamp'] if 'timestamp' in df.columns else df.index)
    return df.sort_values('ts').reset_index(drop=True)


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


def make_process_combo(ohlcv_arr, signal_fn, signal_param_keys, grid_param_names,
                       btc_1d_df, btc_tf_df, strategy_config):
    """Return a process_combo closure bound to the current strategy's data."""
    def process_combo(comb):
        params        = dict(zip(grid_param_names, comb))
        signal_kwargs = {k: params[k] for k in signal_param_keys}
        ohlcv_arrays  = {}

        for sym, arr in ohlcv_arr.items():
            signals = signal_fn(arr, live_trading=False, **signal_kwargs)

            # Apply regime filter if enabled
            if REGIME_FILTER.get('enabled', False):
                signals = filter_signals_by_regime(
                    signals         = signals,
                    ts              = arr['ts'],
                    btc_1d_df       = btc_1d_df,
                    btc_tf_df       = btc_tf_df,
                    regime_filter   = REGIME_FILTER,
                    strategy_config = strategy_config,
                    families        = FAMILIES,
                    lookback_bars   = REGIME_LOOKBACK_BARS,
                    ma_period       = REGIME_MA_PERIOD,
                    hurst_window    = REGIME_HURST_WINDOW,
                    er_window       = REGIME_ER_WINDOW,
                    atr_window      = REGIME_ATR_WINDOW,
                    pe_window       = REGIME_PE_WINDOW,
                    pe_order        = REGIME_PE_ORDER,
                )

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

_btc_tf_cache = {}

def run_strategy(strategy: dict, btc_1d_df: pd.DataFrame) -> None:
    sid        = strategy["id"]
    sname      = strategy["name"]
    timeframe  = strategy["timeframe"]
    signal_key = strategy["signal"]

    registry      = SIGNAL_REGISTRY[signal_key]
    signal_fn     = registry["fn"]
    signal_params = registry["params"]

    strategy_label = f"{sid}_{sname}"
    print(f"\n{'='*60}")
    print(f"  Running: {strategy_label}")
    print(f"{'='*60}")

    # Load BTC for family metrics based on REGIME_FAMILY_SOURCE
    if REGIME_FAMILY_SOURCE == 'macro':
        btc_tf_df = btc_1d_df  # reuse already-loaded BTC 1D
    else:
        btc_tf_df = load_btc_for_timeframe(BTC_FOLDER, timeframe, _btc_tf_cache)

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
    missing   = [s for s in custom_symbols if s not in ohlcv_data.keys()]
    print(f"Missing parquets: {missing}")

    # Build grid
    grid_param_names, lists_for_grid = build_param_grid(strategy, signal_params)
    all_combinations                  = list(product(*lists_for_grid))
    process_combo                     = make_process_combo(
        ohlcv_arr, signal_fn, signal_params, grid_param_names,
        btc_1d_df, btc_tf_df, strategy
    )

    # Run backtest
    with tqdm_joblib(tqdm(desc=f"🔄 {strategy_label}", total=len(all_combinations))):
        grid_results_list = Parallel(n_jobs=N_JOBS)(
            delayed(process_combo)(comb) for comb in all_combinations
        )

    # Compile & save
    grid_records    = compile_grid_results(grid_results_list, grid_param_names, INITIAL_BALANCE)
    grid_results_df = pd.DataFrame(grid_records)

    save_all_trades_to_csv(
        grid_results_list, grid_param_names,
        f"all_trades_{strategy_label}.csv",
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

    # Load BTC 1D once for all strategies
    print("📂 Loading BTC 1D data...")
    btc_1d_df = load_btc_1d()
    print(f"✅ {len(btc_1d_df)} daily bars loaded")

    for strategy in STRATEGIES:
        try:
            run_strategy(strategy, btc_1d_df)
        except Exception as e:
            print(f"❌ Error in strategy {strategy['id']}_{strategy['name']}: {e}")

    elapsed = int(time.time() - total_start)
    print(f"\n🏁 Total execution time: {elapsed//3600}h {(elapsed%3600)//60}m {elapsed%60}s")


if __name__ == "__main__":
    main()