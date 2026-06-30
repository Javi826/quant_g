#shared/shared_batch/ohlcv_utils.py
import pandas as pd
import numpy as np
from shared_config import VOLUME_COL

import logging
logger = logging.getLogger("shared.utils.ohlcv_utils")

NIGHT_CONSOLIDATION_FILTER_ENABLED = False  # toggle to compare with/without

def apply_night_consolidation_filter(ts, signals, hour_start: int = 0, hour_end: int = 3):

    hours    = pd.DatetimeIndex(ts).hour
    in_gap   = (hours >= hour_start) & (hours < hour_end)
    n_zeroed = int((signals[in_gap] != 0).sum())
    if n_zeroed:
        logger.debug(f"  [night-filter] zeroed {n_zeroed} signals in [{hour_start:02d}:00-{hour_end:02d}:00) UTC")
    filtered = signals.copy()
    filtered[in_gap] = 0
    return filtered

def get_n_obs(timeframe: str) -> int:
    mapping = {
        '5m'     : 34560,
        '15m'    : 17280,
        '30m'    : 8640,
        '1H'     : 4320,
        '4H'     : 1080,
        '6Hutc'  : 720,
        '12Hutc' : 360,
        '1Dutc'  : 180
    }
    if timeframe not in mapping:
        raise ValueError(f"Timeframe no in Mapping: {timeframe}")
    return mapping[timeframe]

def get_bars_per_year(timeframe: str) -> int:
    mapping = {
        '15m'    : 365 * 96,
        '30m'    : 365 * 48,
        '1H'     : 365 * 24,
        '4H'     : 365 * 6,
        '6Hutc'  : 365 * 4,
        '12Hutc' : 365 * 2,
        '1Dutc'  : 365,
    }
    if timeframe not in mapping:
        raise ValueError(f"Timeframe not in mapping: {timeframe}")
    return mapping[timeframe]

def prepare_ohlcv_arrays(ohlcv_data):
    ohlcv_arr = {}
    for sym, df in ohlcv_data.items():
        ohlcv_arr[sym] = {
            'ts': df.index.values.astype('datetime64[ns]'),
            'open': df['open'].to_numpy(dtype=np.float64),
            'high': df['high'].to_numpy(dtype=np.float64),
            'low': df['low'].to_numpy(dtype=np.float64),
            'close': df['close'].to_numpy(dtype=np.float64),
            VOLUME_COL: df[VOLUME_COL].to_numpy(dtype=np.float64),
            'low_time': (pd.to_datetime(df['low_time']).to_numpy(dtype='datetime64[ns]')),
            'high_time': (pd.to_datetime(df['high_time']).to_numpy(dtype='datetime64[ns]'))          
        }
        
    return ohlcv_arr


def extract_ohlcv_from_path(paths_per_symbol, path_idx, ts_index=None, dtype=np.float32):
    ohlcv_arrays = {}

    for sym, arr_paths in paths_per_symbol.items():
        if path_idx >= arr_paths.shape[0]:
            continue

        arr = arr_paths[path_idx]  # (n_obs, n_features)
        ohlcv_arrays[sym] = {
            'ts': ts_index if ts_index is not None else np.arange(arr.shape[0]),
            'open': arr[:, 0].astype(dtype),
            'low':  arr[:, 1].astype(dtype),
            'high': arr[:, 2].astype(dtype),
            'close': arr[:, 3].astype(dtype),
            'low_time': np.array(arr[:, 4], dtype='datetime64[ns]'),
            'high_time': np.array(arr[:, 5], dtype='datetime64[ns]'),
        }

    return ohlcv_arrays




