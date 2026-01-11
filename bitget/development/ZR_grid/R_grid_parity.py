import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import time
import numpy as np
import pandas as pd
from itertools import product
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from backtesters.ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE
from tools.ZX_st_tools import prepare_ohlcv_arrays, compile_grid_results, save_all_trades_to_excel, save_results
from utils.ZX_analysis import report_backtesting
from utils.ZX_utils import filter_symbols, save_filtered_symbols, final_prints,save_equity_to_excel
from signals.add_signals_parity import parity_long
from signals.add_signals_parity import parity_short  

from signals.regime_detection import detect_regime
start_time   = time.time()
SAVE_SYMBOLS = False
MY_SYMBOLS   = False
STRATEGY     = "parity_long_4H"
N_JOBS       = -1

# -----------------------------------------------------------------------------
# CONFIGURATION
# -----------------------------------------------------------------------------
DATA_FOLDER         = "../data/crypto_OOS"
DATA_FOLDER         = "../data/crypto_2022_IS"
TIMEFRAME_MINOR     = '4H'

ORDER_AMOUNT        = 80
MIN_VOL_USDT        = 10_000_000

# -----------------------------------------------------------------------------
# PARAMETER GRID
# -----------------------------------------------------------------------------
SELL_AFTER_LIST      = [0]  
LOOKBACK_LIST        = [50,100,150]
TOLERANCE_LIST       = [10,20,30,40] 
MA_PERIOD_LIST       = [50]
TP_PCT_LIST          = [3,4,5]
SL_PCT_LIST          = [8,9,10]

# ===== NUEVOS PARÁMETROS PARA RÉGIMEN =====
ADX_THRESHOLD_LIST = [20, 25, 30]  # Diferentes niveles de exigencia
ADX_PERIOD_LIST    = [14,50,75]  # Diferentes períodos de cálculo

REGIME_FILTER_LIST = [
    None,         # Sin filtro
    [0],          # Solo RANGING
    [1],          # Solo UPTREND
    [2],          # Solo DOWNTREND
    [1, 2],       # Cualquier trending (evita ranging)
    [0, 1],       # RANGING + UPTREND (evita downtrend)
    [0, 2]        # RANGING + DOWNTREND (evita uptrend)
]
# ==========================================

param_names    = ['SELL_AFTER','LOOKBACK','TOLERANCE','MA_PERIOD','TP_PCT','SL_PCT',
                  'ADX_THRESHOLD','ADX_PERIOD','REGIME_FILTER']  # ← Añadidos
param_ranges   = {name: globals()[f"{name}_LIST"] for name in param_names}
lists_for_grid = [param_ranges[name] for name in param_names]

# -----------------------------------------------------------------------------
# LOAD AND FILTER DATA
# -----------------------------------------------------------------------------
symbols_minor = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME_MINOR}.parquet")]
ohlcv_data_minor, filtered_minor = filter_symbols(symbols_minor, min_vol_usdt=MIN_VOL_USDT, timeframe=TIMEFRAME_MINOR, data_folder=DATA_FOLDER, min_price=MIN_PRICE, vol_window=50,my_symbols=MY_SYMBOLS)

save_filtered_symbols(filtered_minor, strategy=STRATEGY, timeframe=TIMEFRAME_MINOR, save_symbols=SAVE_SYMBOLS)

ohlcv_arr_minor = prepare_ohlcv_arrays(ohlcv_data_minor)

# -----------------------------------------------------------------------------
# FUNCTION TO PROCESS ONE PARAMETER COMBINATION
# -----------------------------------------------------------------------------
def process_combo(comb):
    params = dict(zip(param_names, comb))
    ohlcv_arrays = {}

    for sym in ohlcv_arr_minor.keys():
        arr_minor = ohlcv_arr_minor[sym]

        # ===== Detectar régimen con parámetros variables =====
        regimes = detect_regime(
            arr_minor, 
            adx_threshold=params['ADX_THRESHOLD'],  # ← Variable
            adx_period=params['ADX_PERIOD'],        # ← Variable
            live_trading=True
        )
        arr_minor['regime'] = regimes
        # =====================================================

        # Generar señales
        signals = parity_long(
            arr=arr_minor,
            lookback=params['LOOKBACK'],
            tolerance=params['TOLERANCE'],
            ma_period=params['MA_PERIOD'],
            live_trading=False
        )

        # Filtrar por régimen
        regime_filter = params['REGIME_FILTER']
        
        if regime_filter is not None:
            mask = np.isin(regimes, regime_filter)
            signals = signals * mask.astype(np.int8)

        ohlcv_arrays[sym] = {**arr_minor, 'signal': signals}

    # Backtest
    results = run_grid_backtest(
        ohlcv_arrays,
        sell_after=params['SELL_AFTER'],
        tp_pct=params['TP_PCT'],
        sl_pct=params['SL_PCT'],
        order_amount=ORDER_AMOUNT
    )
    return comb, results

# -----------------------------------------------------------------------------
# PARALLELIZED BACKTESTING
# -----------------------------------------------------------------------------
all_combinations = list(product(*lists_for_grid))

print(f"\n📊 GRID SEARCH INFO:")
print(f"Total combinations: {len(all_combinations):,}")
print(f"  Strategy params: {3*4*1*3*3} = {3*4*1*3*3}")
print(f"  Regime params: {3*3*7} = {3*3*7}")
print(f"  Total: {3*4*1*3*3} × {3*3*7} = {len(all_combinations):,}\n")

with tqdm_joblib(tqdm(desc="🔄 Backtesting Grid... \n", total=len(all_combinations))) as progress:
    grid_results_list = Parallel(n_jobs=N_JOBS)(
        delayed(process_combo)(comb) for comb in all_combinations
    )

# -----------------------------------------------------------------------------
# COMPILE RESULTS INTO DATAFRAME
# -----------------------------------------------------------------------------
grid_records    = compile_grid_results(grid_results_list, param_names, INITIAL_BALANCE)
grid_results_df = pd.DataFrame(grid_records)

# -----------------------------------------------------------------------------
# SAVE RESULTS + EXECUTION TIME
# -----------------------------------------------------------------------------
save_results(grid_results_df.to_dict('records'), grid_results_df, f"grid_backtest_{DATA_FOLDER}_{TIMEFRAME_MINOR}.xlsx", save=False)
save_all_trades_to_excel(grid_results_list, param_names, f"all_trades_{TIMEFRAME_MINOR}.xlsx", save=False)
save_equity_to_excel(grid_results_list,"brief_equities", INITIAL_BALANCE,STRATEGY,save_file=False)

final_prints(f" 🥇 Grid_{STRATEGY} 🥇", DATA_FOLDER, f"{TIMEFRAME_MINOR}", MIN_VOL_USDT, ORDER_AMOUNT, param_names, lists_for_grid)

df_portfolio, mi_series = report_backtesting(df=grid_results_df, parameters=param_names, data_folder=DATA_FOLDER, initial_capital=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")