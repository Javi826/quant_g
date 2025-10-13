# === FILE: main_BACKTESTING.py ===
# ---------------------------------
import os
import time
import numpy as np
import pandas as pd
from itertools import product
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
from joblib import Parallel, delayed
from ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE, ORDER_AMOUNT
from utils.ZX_analysis import report_backtesting
from utils.ZX_utils import filter_symbols, save_results, save_filtered_symbols
from Z_add_signals_04 import add_indicators, explosive_signal

start_time = time.time()
SAVE_SYMBOLS = False

# -----------------------------------------------------------------------------
# CONFIGURACIÓN
# -----------------------------------------------------------------------------
DATA_FOLDER  = "data/crypto_2023_highlow_UPTO"
DATE_MIN     = "2025-06-03"
TIMEFRAME    = '4H'
MIN_VOL_USDT = 500_000

# -----------------------------------------------------------------------------
# GRID DE PARÁMETROS
# -----------------------------------------------------------------------------
SELL_AFTER_LIST        = [5,10,15,20]
DOJI_LIST              = [True, False]
HAMMER_LIST            = [True, False]
SHOOTING_STAR_LIST     = [True, False]
BULLISH_ENGULFING_LIST = [True, False]
BEARISH_ENGULFING_LIST = [True, False]
PIERCING_LINE_LIST     = [True, False]
DARK_CLOUD_COVER_LIST  = [True, False]

TP_PCT_LIST            = [0,5,10,15]
SL_PCT_LIST            = [0,5,10,15]

# -----------------------------
# GRID DE PARÁMETROS
# -----------------------------
# =============================================================================
SELL_AFTER_LIST        = [30]
DOJI_LIST              = [False]
HAMMER_LIST            = [True]
SHOOTING_STAR_LIST     = [True]
BULLISH_ENGULFING_LIST = [True]
BEARISH_ENGULFING_LIST = [True]
PIERCING_LINE_LIST     = [False]
DARK_CLOUD_COVER_LIST  = [True]
TP_PCT_LIST            = [0]
SL_PCT_LIST            = [0]
# =============================================================================

param_names = [
    'SELL_AFTER', 'DOJI', 'HAMMER', 'SHOOTING_STAR',
    'BULLISH_ENGULFING', 'BEARISH_ENGULFING',
    'PIERCING_LINE', 'DARK_CLOUD_COVER',
    'TP_PCT', 'SL_PCT'
]
lists_for_grid = [globals()[name + "_LIST"] for name in param_names]

# -----------------------------------------------------------------------------
# CARGA Y FILTRADO DE DATOS
# -----------------------------------------------------------------------------
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

save_filtered_symbols(filtered_symbols, strategy="patterns", timeframe=TIMEFRAME, save_symbols=SAVE_SYMBOLS)

# -----------------------------------------------------------------------------
# FUNCIÓN DE PROCESO PARA UNA COMBINACIÓN
# -----------------------------------------------------------------------------
def process_combo(comb):
    params = dict(zip(param_names, comb))
    ohlcv_arrays = {}

    for sym, df in ohlcv_data.items():
        df_ind = add_indicators(df.copy())

        # Crear pattern_flags según la combinación de parámetros
        pattern_flags = [
            params['DOJI'],
            params['HAMMER'],
            params['SHOOTING_STAR'],
            params['BULLISH_ENGULFING'],
            params['BEARISH_ENGULFING'],
            params['PIERCING_LINE'],
            params['DARK_CLOUD_COVER']
        ]

        df_signal = explosive_signal(df_ind, pattern_flags, live=False)

        ohlcv_arrays[sym] = {
            'ts': df_signal.index.values.astype('datetime64[ns]'),
            'open': df_signal['open'].to_numpy(dtype=np.float64),
            'high': df_signal['high'].to_numpy(dtype=np.float64),
            'low': df_signal['low'].to_numpy(dtype=np.float64),
            'close': df_signal['close'].to_numpy(dtype=np.float64),
            'volume': df_signal['volume'].to_numpy(dtype=np.float64) if 'volume' in df_signal.columns else np.zeros(len(df_signal)),
            'signal': df_signal['signal'].to_numpy(dtype=bool)
        }

    results = run_grid_backtest(
        ohlcv_arrays,
        sell_after=params['SELL_AFTER'],
        initial_balance=INITIAL_BALANCE,
        order_amount=ORDER_AMOUNT,
        tp_pct=params['TP_PCT'],
        sl_pct=params['SL_PCT'],
        comi_pct=0.05
    )

    return comb, results

# -----------------------------------------------------------------------------
# GENERAR TODAS LAS COMBINACIONES DEL GRID
# -----------------------------------------------------------------------------
all_combinations = list(product(*lists_for_grid))

# -----------------------------------------------------------------------------
# BACKTESTING PARALIZADO
# -----------------------------------------------------------------------------
with tqdm_joblib(tqdm(desc="🔁 Backtesting Grid... \n", total=len(all_combinations))) as progress:
    grid_results_list = Parallel(n_jobs=-1)(
        delayed(process_combo)(comb) for comb in all_combinations
    )

# -----------------------------------------------------------------------------
# COMPILAR RESULTADOS A DATAFRAME
# -----------------------------------------------------------------------------
grid_records = []
for comb, results in grid_results_list:
    port = results.get("__PORTFOLIO__", None)
    if port is None:
        continue

    net_gain      = np.sum(port.get('trades', [])) if len(port.get('trades', [])) > 0 else 0.0
    net_gain_pct  = (net_gain / INITIAL_BALANCE) * 100.0
    num_signals   = int(port.get('num_signals', 0))
    num_trades    = len(port.get('trades', []))
    win_ratio     = port.get('proportion_winners', np.nan)
    dd_pct        = port.get('max_dd', 0.0) * 100.0
    final_balance = float(port.get('final_balance', INITIAL_BALANCE))
    avg_trade     = np.nan if num_trades == 0 else np.mean(port['trades'])
    median_trade  = np.nan if num_trades == 0 else np.median(port['trades'])
    sharpe_ratio  = float(port.get('sharpe', np.nan))

    row = {param: value for param, value in zip(param_names, comb)}
    row.update({
        "symbol": "__PORTFOLIO__",
        "Net_Gain": float(net_gain),
        "Net_Gain_pct": float(net_gain_pct),
        "Final_Balance": final_balance,
        "Num_Signals": num_signals,
        "Num_Trades": num_trades,
        "Win_Ratio": float(win_ratio) if not pd.isna(win_ratio) else np.nan,
        "Avg_Trade": float(avg_trade) if not pd.isna(avg_trade) else np.nan,
        "Median_Trade": float(median_trade) if not pd.isna(median_trade) else np.nan,
        "DD_pct": float(dd_pct),
        "Sharpe": sharpe_ratio
    })
    grid_records.append(row)

grid_results_df = pd.DataFrame(grid_records, columns=[*param_names,"symbol", "Net_Gain", "Net_Gain_pct", "Final_Balance","Num_Signals", "Num_Trades", "Win_Ratio", "Avg_Trade", "Median_Trade", "DD_pct", "Sharpe"])

# -----------------------------------------------------------------------------
# SAVE RESULTS + TIMING
# -----------------------------------------------------------------------------
save_results(grid_results_df.to_dict('records'), grid_results_df, filename=f"grid_backtest_{DATA_FOLDER}_{TIMEFRAME}.xlsx", save=False)

print(f"\nTIMEFRAME        : {TIMEFRAME}")
print(f"MIN_VOL_USDT     : {MIN_VOL_USDT}")
print(f"DATE_MIN         : {DATE_MIN}")
print(f"SELL_AFTER_LIST  = {SELL_AFTER_LIST}")

df_portfolio, mi_series = report_backtesting(df=grid_results_df, parameters=param_names, initial_capital=INITIAL_BALANCE)

elapsed = int(time.time() - start_time)
print(f"\n🏁 Total execution time: {elapsed//3600} h {(elapsed%3600)//60} min {elapsed%60} s")
