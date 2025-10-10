# === FILE: main_BACKTESTING.py ===
# -----------------------------
import os
import time
import numpy as np
import warnings
import pandas as pd
from tqdm import tqdm
from itertools import product
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from ZZX_DRAFT3 import analyze_grid_results
from ZX_utils import filter_symbols, save_results,save_filtered_symbols
from ZZX_DRAFT2 import run_grid_backtest, explosive_signal, add_indicators

warnings.filterwarnings("ignore") 
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)

start_time = time.time()

SAVE_SYMBOLS = False
# -----------------------------
# CONFIGURATION
# -----------------------------
TIMEFRAME    = '4H'
DATA_FOLDER  = "data/crypto_2023_highlow_UPTO"
DATE_MIN     = "2025-06-03"
MIN_VOL_USDT = 500_000
MIN_PRICE           = 0.0001

INITIAL_BALANCE     = 10000
ORDER_AMOUNT        = 100

# -----------------------------
# GRID
# -----------------------------
SELL_AFTER_LIST     = [5,10,15,20,25]
ENTROPY_MAX_LIST    = [0.6,0.8,1.0,1.2,1.4]
ACCEL_SPAN_LIST     = [6,8,10]

TP_PCT_LIST         = [0,10,15,20,25]   
SL_PCT_LIST         = [0,10,15,20,25] 
# =============================================================================
SELL_AFTER_LIST    = [25]
ENTROPY_MAX_LIST   = [0.4]
ACCEL_SPAN_LIST    = [5]

TP_PCT_LIST        = [0]
SL_PCT_LIST        = [5]
# =============================================================================     

param_names = ['SELL_AFTER','ENTROPY_MAX','ACCEL_SPAN','TP_PCT','SL_PCT']

# -----------------------------
# SYMBOLS
# -----------------------------
symbols = [f.split('_')[0] for f in os.listdir(DATA_FOLDER) if f.endswith(f"_{TIMEFRAME}.parquet")]

ohlcv_data, filtered_symbols, removed_symbols = filter_symbols(
    symbols,
    min_vol_usdt=MIN_VOL_USDT,
    timeframe=TIMEFRAME,
    data_folder=DATA_FOLDER,
    min_price=MIN_PRICE,
    vol_window=50,
    date_min=DATE_MIN
)

# -----------------------------
# SAVE SYMBOLS
# -----------------------------
save_filtered_symbols(filtered_symbols,strategy="entropy",timeframe=TIMEFRAME, save_symbols=SAVE_SYMBOLS)

# -----------------------------
# FUNCIÓN PARA CADA COMBINACIÓN
# -----------------------------
def process_combination(comb):
    sell_after, entropia_max, accel_span, tp_pct, sl_pct = comb
    
    ohlcv_prepared = {}
    for sym, df in ohlcv_data.items():
        df_proc = df.copy()
        
        if not isinstance(df_proc.index, pd.DatetimeIndex):
            df_proc.index = pd.to_datetime(df_proc.index)
        df_proc = df_proc.sort_index()

        df_proc = add_indicators(df_proc, m_accel=accel_span)
        df_proc = explosive_signal(df_proc, entropia_max=entropia_max, live=False)
        ohlcv_prepared[sym] = df_proc
    #
    results = run_grid_backtest(
        ohlcv_prepared,
        sell_after=sell_after,
        n_jobs=1,
        initial_balance=INITIAL_BALANCE,
        order_amount=ORDER_AMOUNT,
        keep_df=False,
        tp_pct=tp_pct,
        sl_pct=sl_pct,
        comi_pct=0.05   
    )
    
    return comb, results

# -----------------------------
# GRID COMBINATIONS
# -----------------------------
param_lists      = [
    SELL_AFTER_LIST,
    ENTROPY_MAX_LIST,
    ACCEL_SPAN_LIST,
    TP_PCT_LIST,
    SL_PCT_LIST
]
all_combinations = list(product(*param_lists))

# -----------------------------
# BACKTEST
# -----------------------------
with tqdm_joblib(tqdm(desc="🔁 Backtesting Grid... \n", total=len(all_combinations))) as progress:
    grid_results_list = Parallel(
        n_jobs=-1,
        batch_size='auto'
    )(delayed(process_combination)(comb) for comb in all_combinations)

# -----------------------------
# TO DF (POR NÚMERO DE COMBINACIÓN: MÉTRICAS A NIVEL PORTAFOLIO)
# -----------------------------
# Para cada combinación guardamos UNA SOLA FILA con las métricas del portafolio
grid_records = []
for comb, results in grid_results_list:
    # extraer datos del portafolio
    port = results.get("__PORTFOLIO__", None)
    if port is None:
        # si no existe (improbable), saltamos
        continue
   
    net_gain      = np.sum(port['trades']) if len(port.get('trades', [])) > 0 else 0.0   
    net_gain_pct  = (net_gain / INITIAL_BALANCE) * 100.0 if INITIAL_BALANCE != 0 else np.nan   
    num_signals   = int(port.get('num_signals', 0))    
    num_trades    = len(port.get('trades', [])) 
    win_ratio     = port.get('proportion_winners', np.nan)   
    dd_pct        = port.get('max_dd', 0.0) * 100.0  
    final_balance = float(port.get('final_balance', INITIAL_BALANCE))   
    avg_trade     = np.nan if num_trades == 0 else (np.mean(port['trades']))
    median_trade  = np.nan if num_trades == 0 else (np.median(port['trades']))
    
    row = {
        param: value for param, value in zip(param_names, comb)
    }
    
    row.update({
        "symbol": "__PORTFOLIO__",
        "Net_Gain": float(net_gain),
        "Net_Gain_pct": float(net_gain_pct),
        "Final_Balance": final_balance,
        "Num_Signals": num_signals,      # aperturas ejecutadas en todo el portafolio
        "Num_Trades": num_trades,        # cierres reales
        "Win_Ratio": float(win_ratio) if not pd.isna(win_ratio) else np.nan,
        "Avg_Trade": float(avg_trade) if not pd.isna(avg_trade) else np.nan,
        "Median_Trade": float(median_trade) if not pd.isna(median_trade) else np.nan,
        "DD_pct": float(dd_pct)
    })

    grid_records.append(row)

# convertir a DataFrame (una fila por combinación de parámetros — portafolio)
grid_results_df = pd.DataFrame(grid_records, columns=[
    *param_names,
    "symbol","Net_Gain","Net_Gain_pct","Final_Balance",
    "Num_Signals","Num_Trades","Win_Ratio","Avg_Trade","Median_Trade","DD_pct"
])

# -----------------------------
# SAVE RESULTS + TIMING
# -----------------------------
#save_results(grid_results_df.to_dict('records'), grid_results_df, filename=f"grid_backtest_entropy_{DATA_FOLDER}_{TIMEFRAME}.xlsx")
# -----------------------------
# INFO FINAL
# -----------------------------
print(f"🕒 TIMEFRAME     : {TIMEFRAME}")
print(f"💰 MIN_VOL_USDT  : {MIN_VOL_USDT}")
print(f"📅 DATE_MIN      : {DATE_MIN}")
print(f"SELL_AFTER_LIST  = {SELL_AFTER_LIST}")
print(f"ENTROPY_MAX_LIST = {ENTROPY_MAX_LIST}")
print(f"ACCEL_SPAN_LIST  = {ACCEL_SPAN_LIST}")
print(f"TP_PCT_LIST      = {TP_PCT_LIST}")
print(f"SL_PCT_LIST      = {SL_PCT_LIST}")


df_portfolio, mi_series = analyze_grid_results(df=grid_results_df, parameters=param_names, initial_capital=INITIAL_BALANCE)

end_time = time.time()
hours, remainder = divmod(end_time - start_time, 3600)
minutes, seconds = divmod(remainder, 60)
print(f"\n🏁 Total execution time: {int(hours)} h {int(minutes)} min {int(seconds)} s")