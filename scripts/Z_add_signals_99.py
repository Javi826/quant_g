# === FILE: add_signals04.py ===
# ---------------------------------
import warnings
import logging
import pandas as pd
import numpy as np
from utils.ZX_indicators import rolling_entropy_numba, delta_numba

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

import numpy as np

def explosive_signal_99(high, close, live=False):
    signal = np.zeros_like(close)
    if len(signal) > 0:
        signal[0] = 1  # Comprar en el primer timestamp
    return signal





