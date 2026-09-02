#shared_batch/pipeline/universe.py (forex)
import os
import logging
import pandas as pd
from shared_config import VOLUME_COL
logger = logging.getLogger("BOT_batch.pipeline.universe")

# =============================================================================
# FILTER SYMBOLS
# =============================================================================
ENABLE_INCLUDE_FILTER = True
INCLUDED_SYMBOLS = ["ADAUSDT","AVAXUSDT","BCHUSDT","BNBUSDT","DOGEUSDT","LINKUSDT","NEARUSDT","SOLUSDT","XLMUSDT","XRPUSDT"]

ENABLE_EXCLUDE_FILTER = False
EXCLUDED_SYMBOLS = {"BTCUSDT", "ETHUSDT"}

# Symbols must have data available from this date onward (covers WFO window 0 train_start).
MIN_START_DATE = "2022-01-02"

def filter_symbols(symbols, timeframe=None, data_folder=None, exchange=None,
                   min_price=None, vol_window=50, custom_symbols=None):
    ohlcv_data         = {}
    filtered_symbols   = []
    removed_symbols    = []
    removed_by_reasons = {"No data": 0, "Last close too low": 0, "File missing": 0, "Starts too late": 0}
    
    #Inclusion
    if ENABLE_INCLUDE_FILTER:
        inclusion_list = custom_symbols if custom_symbols is not None else INCLUDED_SYMBOLS
        symbols = [s for s in symbols if s in inclusion_list]
        
    #Exclusion
    for sym in symbols:
        if ENABLE_EXCLUDE_FILTER and sym in EXCLUDED_SYMBOLS:
            removed_symbols.append(sym)
            continue
        
        #Si ENABLE_INCLUDE_FILTER=True, solo cargar sin filtros
        if ENABLE_INCLUDE_FILTER:
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

    active_reasons = {reason: count for reason, count in removed_by_reasons.items() if count > 0}
    if active_reasons:
        reasons_str = "  |  ".join(f"{reason}: {count}" for reason, count in active_reasons.items())
        logger.debug(f"🔹Removed reasons   : {reasons_str}")

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
        raw_is, timeframe=timeframe, data_folder=data_folder_is, min_price=min_price, vol_window=50
    )
    cutoff_ts = pd.Timestamp(MIN_START_DATE)
    ohlcv_is_cut = {}
    for sym, df in ohlcv_is.items():
        start_before = df.index[0]
        df_cut = df[df.index >= cutoff_ts]
        ohlcv_is_cut[sym] = df_cut
        start_after = df_cut.index[0] if len(df_cut) else None
        logger.debug(f"  {sym:<12} before: {str(start_before):<19}  after: {str(start_after):<19}")
    ohlcv_is = ohlcv_is_cut
    logger.debug(f"IS pool ({len(filtered_is):>3}): {filtered_is}")
    return ohlcv_is

# =============================================================================
# TOP-N SYMBOL SELECTION BY VOLUME — applied once, upstream of every pipeline
# =============================================================================
def select_top_n_by_volume(ohlcv_data: dict, n_symbols: int | None) -> dict:

    if n_symbols is None or n_symbols >= len(ohlcv_data):
        return ohlcv_data

    avg_volume_by_symbol = {
        sym: float(df[VOLUME_COL].mean())
        for sym, df in ohlcv_data.items()
    }

    top_symbols = sorted(avg_volume_by_symbol, key=avg_volume_by_symbol.get, reverse=True)[:n_symbols]

    logger.debug(f"Top {n_symbols} symbols by avg volume: {top_symbols}")

    return {sym: ohlcv_data[sym] for sym in top_symbols}