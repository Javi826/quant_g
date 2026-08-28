#shared/shared_batchs/utils/ohlcv_utils.py (crypto)
import pandas as pd
import numpy as np
from shared_config import VOLUME_COL

import logging
logger = logging.getLogger("shared.utils.ohlcv_utils")

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