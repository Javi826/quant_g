import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import time
from utils.ZX_utils import filter_symbols, final_prints
from tools.ZX_WFO_MC import walk_forward_optimization_mc
from tools.ZX_st_tools import get_n_obs
import pandas as pd
from tools.ZX_st_tools import prepare_ohlcv_arrays, compile_grid_results
from utils.ZX_analysis import report_backtesting
from backtesters.ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE
from signals.add_signals_orderblocks import orderblocks_long
from signals.add_signals_orderblocks import orderblocks_short

start_time  = time.time()
N_JOBS      = -1
STRATEGY    = "orderblocks"
MY_SYMBOLS  = True
# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATA_FOLDER         = "../data/crypto_2022_IS"
DATA_FOLDER_OOS     = "../data/crypto_2025_OOS"
TIMEFRAME_MINOR     = '4H'
ORDER_AMOUNT        = 80
MIN_VOL_USDT        = 10_000_000
# -----------------------------------------------------------------------------
# WFO SETTINGS
# -----------------------------------------------------------------------------
ANCHORED            = False
MONTHS_TRAIN        = 6
MONTHS_TEST         = 2
# -----------------------------------------------------------------------------
# MONTE CARLO SETTINGS
# -----------------------------------------------------------------------------
FINAL_N_PATHS       = 100
# -----------------------------------------------------------------------------
# PARAMETER GRID
# -----------------------------------------------------------------------------
SELL_AFTER_LIST      = [0]  
LOOKBACK_LIST        = [50,100,150]
TOLERANCE_LIST       = [35,40,45] 
IMPULSE_LIST         = [0.005,0.01,0.015]

TP_PCT_LIST          = [2,3,4,5]
SL_PCT_LIST          = [8,9,10,11]

param_names     = ['SELL_AFTER', 'LOOKBACK', 'TOLERANCE', 'IMPULSE', 'TP_PCT', 'SL_PCT']
param_ranges    = {name: globals()[f"{name}_LIST"] for name in param_names}
# -----------------------------------------------------------------------------
# CANDLES PER MONTH BY TIMEFRAME
# -----------------------------------------------------------------------------
_CANDLES_PER_MONTH = {
    '1H':    24 * 30,
    '4H':     6 * 30,
    '6Hutc':  4 * 30,
    '12Hutc': 2 * 30,
    '1Dutc':      30,
}

if TIMEFRAME_MINOR not in _CANDLES_PER_MONTH:
    raise ValueError(f"Timeframe not supported: {TIMEFRAME_MINOR}")

_cpm            = _CANDLES_PER_MONTH[TIMEFRAME_MINOR]
LENGTH_TRAIN    = int(MONTHS_TRAIN * _cpm)
PCT_TRAIN       = MONTHS_TRAIN / (MONTHS_TRAIN + MONTHS_TEST)
FINAL_N_OBS     = get_n_obs(TIMEFRAME_MINOR)

# -----------------------------------------------------------------------------
# SIGNAL FUNCTION — comment/uncomment as needed
# -----------------------------------------------------------------------------
signal_fn = orderblocks_short
# signal_fn = orderblocks_long

# -----------------------------------------------------------------------------
# LOAD AND FILTER DATA
# -----------------------------------------------------------------------------
symbols_minor = [
    f.split('_')[0]
    for f in os.listdir(DATA_FOLDER)
    if f.endswith(f"_{TIMEFRAME_MINOR}.parquet")
]

ohlcv_data_minor, filtered_minor = filter_symbols(
    symbols_minor,
    min_vol_usdt=MIN_VOL_USDT,
    timeframe=TIMEFRAME_MINOR,
    data_folder=DATA_FOLDER,
    min_price=MIN_PRICE,
    vol_window=50,
    my_symbols=MY_SYMBOLS
)
print(f"Symbols: {sorted(list(ohlcv_data_minor.keys()))}")
# -----------------------------------------------------------------------------
# RUN WFO + MC
# -----------------------------------------------------------------------------
final_prints(
    f"🔁 WFO_MC_{STRATEGY}",
    DATA_FOLDER,
    TIMEFRAME_MINOR,
    min_vol_usdt=MIN_VOL_USDT,
    order_amount=ORDER_AMOUNT,
    param_names=param_names,
    lists_for_grid=[param_ranges[n] for n in param_names]
)

print(f"\n⚙️  Anchored: {ANCHORED} | Train: {MONTHS_TRAIN}m | Test: {MONTHS_TEST}m | Paths: {FINAL_N_PATHS} | N_obs: {FINAL_N_OBS}")

df_wfo_results = walk_forward_optimization_mc(
    ohlcv_data=ohlcv_data_minor,
    param_ranges=param_ranges,
    signal_fn=signal_fn,
    run_grid_backtest=run_grid_backtest,
    length_train_set=LENGTH_TRAIN,
    pct_train_set=PCT_TRAIN,
    anchored=ANCHORED,
    n_paths=FINAL_N_PATHS,
    n_obs=FINAL_N_OBS,
    order_amount=ORDER_AMOUNT,
    initial_balance=INITIAL_BALANCE,
    n_jobs=N_JOBS
)
# -----------------------------------------------------------------------------
# OOS ANALYSIS — mean, mode, ewm
# -----------------------------------------------------------------------------
symbols_oos       = [f.split('_')[0] for f in os.listdir(DATA_FOLDER_OOS) if f.endswith(f"_{TIMEFRAME_MINOR}.parquet")]
ohlcv_data_oos, _ = filter_symbols(symbols_oos, min_vol_usdt=MIN_VOL_USDT, timeframe=TIMEFRAME_MINOR, data_folder=DATA_FOLDER_OOS, min_price=MIN_PRICE, vol_window=50, my_symbols=MY_SYMBOLS)
ohlcv_arr_oos     = prepare_ohlcv_arrays(ohlcv_data_oos)

_int_params = {k for k in param_names if all(isinstance(x, int) for x in param_ranges[k])}

def _extract_params(row):
    return {
        k: int(round(row[k])) if k in _int_params else round(float(row[k]), 4)
        for k in param_names
    }

for method in ['mean', 'mode', 'ewm']:
    print(f"\n{'='*60}")
    print(f"🔭 OOS analysis — {method.upper()} params")
    print(f"{'='*60}")

    params = _extract_params(df_wfo_results[df_wfo_results['window'] == method.upper()].iloc[0])
    print("   " + " | ".join(f"{k}: {v}" for k, v in params.items()))

    ohlcv_arrays_oos = {}
    for sym, arr in ohlcv_arr_oos.items():
        signals = signal_fn(arr, 
                            lookback=params['LOOKBACK'], 
                            tolerance=params['TOLERANCE'], 
                            impulse=params['IMPULSE'], 
                            live_trading=False)
        ohlcv_arrays_oos[sym] = {**arr, 'signal': signals}

    oos_result  = run_grid_backtest(ohlcv_arrays_oos,
                                    sell_after=params['SELL_AFTER'], 
                                    tp_pct=params['TP_PCT'],
                                    sl_pct=params['SL_PCT'], 
                                    order_amount=ORDER_AMOUNT)
    best_comb   = tuple(params[p] for p in param_names)
    oos_df      = pd.DataFrame(compile_grid_results([(best_comb, oos_result)], param_names, INITIAL_BALANCE))

    final_prints(f"🔭 OOS_{method.upper()}_{STRATEGY}", DATA_FOLDER_OOS, TIMEFRAME_MINOR, MIN_VOL_USDT, ORDER_AMOUNT, param_names, [param_ranges[n] for n in param_names])
    report_backtesting(df=oos_df, parameters=param_names, data_folder=DATA_FOLDER_OOS, initial_capital=INITIAL_BALANCE)
# -----------------------------------------------------------------------------
# ELAPSED TIME
# -----------------------------------------------------------------------------
elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")