# === FILE: add_signals03.py ===
# ---------------------------------
import logging
import warnings
import numpy as np
import pandas as pd
from numba import njit
from utils.ZX_indicators import rolling_entropy_pandas,ewm_pandas
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")



# =========================
# FUNCIONES BASE
# =========================
def add_indicators_03_pandas(df, m_accel=5):
    df = df.copy()
    delta = df['close'].diff().fillna(0)
    df['entropia'] = rolling_entropy_pandas(delta, window=5, bins=10)
    accel_raw = df['close'].diff().diff().fillna(0)  # segunda diferencia
    df['accel'] = accel_raw.ewm(span=m_accel, adjust=False).mean()
    return df

def explosive_signal_03_pandas(df, entropia_max=2.0, live=False):
    df = df.copy()
    df['signal'] = (df['entropia'] < entropia_max) & (df['accel'] > 0)
    if not live:
        df['signal'] = df['signal'].shift(1, fill_value=False)
    return df
    
