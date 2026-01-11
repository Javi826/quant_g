import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from itertools import product
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from utils.ZX_analysis import report_montecarlo
from utils.ZX_utils import filter_symbols, final_prints, align_filter_to_symbol
from backtesters.ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE
from tools.ZX_st_tools import extract_ohlcv_from_path_v, compile_MC_results, get_n_obs
from tools.ZX_optimize_MCf_tf import generate_paths_for_all_symbols_functional
from signals.add_signals_parity import parity_long, parity_short
from signals.volatility_detection import detect_volatility

DTYPE               = np.float32
start_time          = time.time()
N_JOBS              = -1
STRATEGY            = "parity_short"

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATA_FOLDER         = "../data/crypto_2022_IS"
TIMEFRAME_MINOR     = '1H'

ORDER_AMOUNT        = 80
MIN_VOL_USDT        = 10_000_000

# -----------------------------------------------------------------------------
# PARAMETER GRID
# -----------------------------------------------------------------------------
SELL_AFTER_LIST      = [0]  
LOOKBACK_LIST        = [50,100,150]
TOLERANCE_LIST       = [10,20,30,40] 
MA_PERIOD_LIST       = [25,50]
TP_PCT_LIST          = [2,3]
SL_PCT_LIST          = [9,10]

# VOLATILITY FILTER 
USE_VOL_FILTER_LIST   = [True]  
CHAOS_PERCENTILE_LIST = [95,99]
ATR_PERIOD_LIST       = [14] 

param_names     = ['SELL_AFTER', 'LOOKBACK', 'TOLERANCE', 'MA_PERIOD', 'TP_PCT', 'SL_PCT',
                   'USE_VOL_FILTER', 'CHAOS_PERCENTILE', 'ATR_PERIOD']
lists_for_grid  = [globals()[name + "_LIST"] for name in param_names]
param_dict_list = [dict(zip(param_names, comb)) for comb in product(*lists_for_grid)]

# -----------------------------------------------------------------------------
# MONTE CARLO SETTINGS
# -----------------------------------------------------------------------------
FINAL_N_PATHS        = 100
FINAL_N_OBS_PER_PATH = get_n_obs(TIMEFRAME_MINOR)

# -----------------------------------------------------------------------------
# LOAD AND FILTER DATA
# -----------------------------------------------------------------------------
symbols_minor = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) 
                 if f.endswith(f"_{TIMEFRAME_MINOR}.parquet")]
ohlcv_data_minor, filtered_minor = filter_symbols(
    symbols_minor,
    min_vol_usdt=MIN_VOL_USDT,
    timeframe=TIMEFRAME_MINOR,
    data_folder=DATA_FOLDER,
    min_price=MIN_PRICE,
    vol_window=50
)

# -----------------------------------------------------------------------------
# AÑADIR CAMPOS BTC A TODOS LOS SÍMBOLOS
# -----------------------------------------------------------------------------
if 'BTCUSDT' not in ohlcv_data_minor:
    raise ValueError("BTCUSDT no encontrado. Necesario para volatility filter.")

btc_df = ohlcv_data_minor['BTCUSDT'].copy()

for sym in list(ohlcv_data_minor.keys()):
    sym_df = ohlcv_data_minor[sym]
    
    btc_aligned = btc_df.reindex(sym_df.index, method='ffill')
    
    if btc_aligned['close'].isna().any():
        first_valid           = btc_aligned['close'].first_valid_index()
        sym_df                = sym_df.loc[first_valid:]
        btc_aligned           = btc_aligned.loc[first_valid:]
        ohlcv_data_minor[sym] = sym_df
    
    sym_df['btc_open']  = btc_aligned['open'].values
    sym_df['btc_high']  = btc_aligned['high'].values
    sym_df['btc_low']   = btc_aligned['low'].values
    sym_df['btc_close'] = btc_aligned['close'].values

# -----------------------------------------------------------------------------
# HELPER FUNCTIONS
# -----------------------------------------------------------------------------
BTC_RAW_COLUMNS = ['btc_open', 'btc_high', 'btc_low', 'btc_close']

def process_path_IDX(path_idx, paths_minor, param_dict_list):
    all_results = []
    
    for param_dict in param_dict_list:
        ohlcv_arrays_minor = extract_ohlcv_from_path_v(
            paths_minor, 
            path_idx, 
            dtype=DTYPE,
            raw_columns=BTC_RAW_COLUMNS
        )
        
        if len(ohlcv_arrays_minor) == 0:
            continue
        
        btc_arr = {
            'ts': ohlcv_arrays_minor['BTCUSDT']['ts'],
            'open': ohlcv_arrays_minor['BTCUSDT']['btc_open'],
            'high': ohlcv_arrays_minor['BTCUSDT']['btc_high'],
            'low': ohlcv_arrays_minor['BTCUSDT']['btc_low'],
            'close': ohlcv_arrays_minor['BTCUSDT']['btc_close']
        }
            
        if param_dict.get('USE_VOL_FILTER'):
            btc_vol_filter = detect_volatility(
                btc_arr,
                atr_period=param_dict.get('ATR_PERIOD'),
                chaos_percentile=param_dict.get('CHAOS_PERCENTILE')
            )
        else:
            btc_vol_filter = None
        
        for sym in ohlcv_arrays_minor.keys():
            arr_minor = ohlcv_arrays_minor[sym]
 
            signals = parity_long(
                arr_minor,
                lookback=param_dict.get('LOOKBACK'),
                tolerance=param_dict.get('TOLERANCE'),
                ma_period=param_dict.get('MA_PERIOD'),
                live_trading=False
            )

            if param_dict.get('USE_VOL_FILTER') and btc_vol_filter is not None:
                aligned_filter = align_filter_to_symbol(
                    symbol_timestamps=arr_minor.get('ts'),
                    btc_timestamps=btc_arr['ts'],
                    btc_filter=btc_vol_filter
                )
                signals = signals * aligned_filter

            arr_minor['signal'] = np.asarray(signals, dtype=DTYPE)

        result = run_grid_backtest(
            ohlcv_arrays_minor,
            sell_after=param_dict.get('SELL_AFTER'),
            tp_pct=param_dict.get('TP_PCT'),
            sl_pct=param_dict.get('SL_PCT'),
            order_amount=ORDER_AMOUNT
        )

        portfolio_record = compile_MC_results(
            result, param_dict, path_idx, INITIAL_BALANCE, dtype=DTYPE
        )
        all_results.append(portfolio_record)

    return all_results

def parallel_with_progress(tasks, desc: str, n_jobs: int = N_JOBS):
    with tqdm_joblib(tqdm(total=len(tasks), desc=desc)):
        return Parallel(n_jobs=n_jobs)(tasks)

# -----------------------------------------------------------------------------
# GENERATE & EVALUATE PATHS
# -----------------------------------------------------------------------------
paths_minor = generate_paths_for_all_symbols_functional(
    ohlcv_data_minor,
    n_paths=FINAL_N_PATHS,
    n_obs=FINAL_N_OBS_PER_PATH,
    raw_columns=BTC_RAW_COLUMNS
)

results_list = parallel_with_progress(
    [delayed(process_path_IDX)(i, paths_minor, param_dict_list) 
     for i in range(FINAL_N_PATHS)],
    desc="🔄 Evaluating Paths"
)

all_results = [r for sublist in results_list for r in sublist]
df_portfolio = pd.DataFrame(all_results)

# -----------------------------------------------------------------------------
# SUMMARY / REPORT
# -----------------------------------------------------------------------------
final_prints(
    f"🎲 MC_{STRATEGY} 🎲", 
    DATA_FOLDER, 
    f"{TIMEFRAME_MINOR}", 
    min_vol_usdt=MIN_VOL_USDT, 
    order_amount=ORDER_AMOUNT, 
    param_names=param_names, 
    lists_for_grid=lists_for_grid
)

df_summary = report_montecarlo(
    df_portfolio=df_portfolio, 
    param_names=param_names, 
    initial_balance=INITIAL_BALANCE
)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600}h {(elapsed%3600)//60}m {elapsed%60}s")