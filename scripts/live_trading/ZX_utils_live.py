import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import numpy as np
import pandas as pd
from parquet_process.Z_parquet_A0_extraction import  _call_history_candles, to_dataframe_from_api
from pandas.api.types import is_datetime64_any_dtype
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

STATE_FILE   = os.path.join(os.path.dirname(__file__), 'tracked_orders_state.json')
MADRID_TZ    = ZoneInfo("Europe/Madrid")
BASE_URL     = "https://api.bitget.com"
PRODUCT_TYPE = 'usdt-futures'


#==============================================================================================
#HELPERS
#==============================================================================================
def normalize_live_ohlcv(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df.index = pd.to_datetime(df['timestamp'])
        else:
            df.index = pd.to_datetime(df.index)

    for col in ['open', 'high', 'low', 'close', 'volume_base', 'volume_quote']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

def df_to_arrays_live(df):
    if not is_datetime64_any_dtype(df.index):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    arrays = {
        'ts': df.index.to_numpy(dtype='datetime64[ns]'),
        'open': df['open'].to_numpy(dtype=np.float64),
        'high': df['high'].to_numpy(dtype=np.float64),
        'low': df['low'].to_numpy(dtype=np.float64),
        'close': df['close'].to_numpy(dtype=np.float64),
        'volume_quote': (
            df['volume_quote'].to_numpy(dtype=np.float64)
            if 'volume_quote' in df
            else np.zeros(len(df))
        )
    }

    return arrays

def load_final_symbols(all_symbols, strategy="_", timeframe="4H"):
    folder = os.path.join(os.path.dirname(__file__), "symbols_live")
    folder = os.path.abspath(folder)
    try:
        path_live = os.path.join(folder, f"symbols_live_{strategy}_{timeframe}.xlsx")
        df_live = pd.read_excel(path_live)
        live_symbols = set(df_live.iloc[:, 0].dropna().astype(str))
        final_symbols = set(all_symbols) & live_symbols

        print(f"🔹 Symbols for Live: {len(final_symbols)}")
        return sorted(final_symbols)

    except Exception as e:
        print(f"⚠️ Error loading symbols: {e}")
        return []

def wait_for_next_candle(timeframe='4H'):
    now = datetime.utcnow()
    
    if timeframe.endswith('H'):
        minutes = int(timeframe[:-1]) * 60
    elif timeframe.endswith('m'):
        minutes = int(timeframe[:-1])
    elif timeframe.endswith('Dutc'):
        minutes = int(timeframe[:-4]) * 24 * 60
    else:
        raise ValueError("Invalid timeframe, use 'm', 'H', or 'Dutc'.")

    total_minutes      = now.hour * 60 + now.minute
    next_total_minutes = ((total_minutes // minutes) + 1) * minutes
    delta_minutes      = next_total_minutes - total_minutes
    next_run           = now + timedelta(minutes=delta_minutes, seconds=-now.second, microseconds=-now.microsecond)
    
    sleep_seconds = (next_run - now).total_seconds() + 45
    print('\n')
    print(f"🔷 === Waiting for next candle ===: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print('\n')
    time.sleep(sleep_seconds)
    


def fetch_ohlcv_data(symbols, timeframe):
    ohlcv_data = {}
    for sym in symbols:
        recent_minor    = _call_history_candles(symbol=sym, granularity=timeframe, limit=180)
        df_minor        = to_dataframe_from_api(recent_minor)
        ohlcv_data[sym] = df_minor
    return ohlcv_data


