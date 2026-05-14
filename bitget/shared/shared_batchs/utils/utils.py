#shared_batch/utils.py
import os
import pandas as pd
from shared_config import VOLUME_COL
import logging
logger = logging.getLogger("BOT_batch.utils")
symbols_to_exclude = {}

symbols_to_include = [
    "NVDAUSDT",   # NVIDIA        - corr 0.77, 202 rows
    "PLTRUSDT",   # Palantir      - corr 0.73, 181 rows
    "HOODUSDT",   # Robinhood     - corr 0.71, 194 rows
    "ASMLUSDT",   # ASML          - corr 0.70, 182 rows
    "GOOGLUSDT",  # Alphabet      - corr 0.65, 195 rows
    "AMZNUSDT",   # Amazon        - corr 0.65, 195 rows
    "TSLAUSDT",   # Tesla         - corr 0.64, 202 rows
    "COINUSDT",   # Coinbase      - corr 0.63, 194 rows
    "MRVLUSDT",   # Marvell       - corr 0.63, 180 rows
    "METAUSDT",   # Meta          - corr 0.59, 195 rows
    "MSFTUSDT",   # Microsoft     - corr 0.48, 168 rows
]

def filter_symbols(symbols, min_vol_usdt, timeframe=None, data_folder=None, exchange=None,
                   min_price=None, vol_window=50, my_symbols=False, custom_symbols=None):
    ohlcv_data         = {}
    filtered_symbols   = []
    removed_symbols    = []
    removed_by_reasons = {"No data": 0, "Not enough bars": 0, "Last close too low": 0, "Avg volume too low": 0, "File missing": 0}
    
    #Inclusion
    if my_symbols:
        inclusion_list = custom_symbols if custom_symbols is not None else symbols_to_include
        symbols = [s for s in symbols if s in inclusion_list]
        
    #Exclusion
    for sym in symbols:
        if sym in symbols_to_exclude:
            removed_symbols.append(sym)
            continue
        
        #Si my_symbols=True, solo cargar sin filtros
        if my_symbols:
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
                    
            if df is not None:
                n_rows = len(df)            
                if timeframe == "1Dutc":
                    min_bars = 300
                elif timeframe == "12Hutc":
                    min_bars = 600
                elif timeframe == "6Hutc":
                    min_bars = 1200
                elif timeframe == "4H":
                    min_bars = 1800
                elif timeframe == "1H":
                    min_bars = 7200
                elif timeframe == "30m":
                    min_bars = 14400                    
                elif timeframe == "15m":
                    min_bars = 28800
                else:
                    min_bars = 999999999                 
                if n_rows < min_bars:
                    reasons.append("Not enough bars")
                    
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