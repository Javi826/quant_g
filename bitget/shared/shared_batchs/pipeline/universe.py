#shared_batch/pipeline/universe.py
import os
import logging
import pandas as pd
from shared_config import VOLUME_COL
logger = logging.getLogger("BOT_batch.pipeline.universe")

# =============================================================================
# FILTER SYMBOLS
# =============================================================================
MY_SYMBOLS = False
#symbols_to_exclude        = {"BTCUSDT","ETHUSDT"}
symbols_to_exclude        = {}

# Symbols must have data available from this date onward (covers WFO window 0 train_start).
MIN_START_DATE = "2022-01-03"

symbols_to_include = [
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "SUIUSDT",
    "HYPEUSDT",
    "ADAUSDT",
    "TAOUSDT",
    "PIPPINUSDT",
]

def filter_symbols(symbols, min_vol_usdt, timeframe=None, data_folder=None, exchange=None,
                   min_price=None, vol_window=50, custom_symbols=None):
    ohlcv_data         = {}
    filtered_symbols   = []
    removed_symbols    = []
    removed_by_reasons = {"No data": 0, "Last close too low": 0, "Avg volume too low": 0, "File missing": 0, "Starts too late": 0}
    
    #Inclusion
    if MY_SYMBOLS:
        inclusion_list = custom_symbols if custom_symbols is not None else symbols_to_include
        symbols = [s for s in symbols if s in inclusion_list]
        
    #Exclusion
    for sym in symbols:
        if sym in symbols_to_exclude:
            removed_symbols.append(sym)
            continue
        
        #Si my_symbols=True, solo cargar sin filtros
        if MY_SYMBOLS:
            file_path = os.path.join(data_folder, f"{sym}_{timeframe}.parquet")
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                if not df.empty:
                    ohlcv_data[sym] = df
                    filtered_symbols.append(sym)
                else:
                    removed_symbols.append(sym)
            else:
                removed_symbols.append(sym)
            continue
        
        df        = None
        reasons   = []
        file_path = os.path.join(data_folder, f"{sym}_{timeframe}.parquet")
        
        if not os.path.exists(file_path):
            reasons.append("File missing")
        else:
            df = pd.read_parquet(file_path)
            if df.empty:
                reasons.append("No data")
            if df is not None and min_price is not None:
                last_close = df['close'].iloc[-1]
                if last_close <= min_price:
                    reasons.append("Last close too low")
                    
            if df is not None:
                avg_vol = df[VOLUME_COL].tail(vol_window).mean()
                if avg_vol < min_vol_usdt:
                    reasons.append("Avg volume too low")
                    
            if df is not None and not df.empty:
                if df.index[0] > pd.Timestamp(MIN_START_DATE):
                    reasons.append("Starts too late")

        if reasons:
            removed_symbols.append(sym)
            for r in reasons:
                removed_by_reasons[r] += 1
        else:
            ohlcv_data[sym] = df
            filtered_symbols.append(sym)
            
    logger.debug(f"🔹Total symbols     : {len(symbols)}")
    logger.debug(f"🔹Symbols removed   : {len(removed_symbols)}")
    logger.debug(f"🔹Symbols remaining : {len(filtered_symbols)}")
    
    return ohlcv_data, filtered_symbols

# =============================================================================
# UNIVERSE SELECTION
# =============================================================================
def select_universe(
    data_folder_is:    str,
    timeframe:         str,
    min_price:         float,
    filter_symbols_fn: callable,
) -> dict:

    raw_is = sorted([f.split("_")[0] for f in os.listdir(data_folder_is) if f.endswith(f"_{timeframe}.parquet")])

    ohlcv_is, filtered_is = filter_symbols_fn(
        raw_is, min_vol_usdt=0, timeframe=timeframe, data_folder=data_folder_is, min_price=min_price, vol_window=50
    )

    logger.debug(f"IS pool ({len(filtered_is):>3}): {filtered_is}")

    return ohlcv_is