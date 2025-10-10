import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from numba import njit
import logging
logging.basicConfig(level=logging.INFO)

# -----------------------------
# CÁLCULO DE GANANCIAS NETAS
# -----------------------------
def calculate_data(results, initial_balance=10000):
    portfolio = results.get("__PORTFOLIO__", None)
    if portfolio is None:
        return 0.0, 0.0
    final_balance  = portfolio['final_balance']
    net_gain_total = final_balance - initial_balance
    net_gain_pct   = (net_gain_total / initial_balance) * 100
    return net_gain_total, net_gain_pct

def add_indicators(df):
    """
    Detecta patrones clásicos de velas japonesas:
    - Doji
    - Martillo (Hammer)
    - Estrella fugaz (Shooting Star)
    - Engulfing alcista y bajista
    """
    # Cuerpo y mechas
    df['body'] = (df['close'] - df['open']).abs()
    df['upper_wick'] = df['high'] - df[['open','close']].max(axis=1)
    df['lower_wick'] = df[['open','close']].min(axis=1) - df['low']

    # Doji: cuerpo muy pequeño (apertura ≈ cierre)
    df['doji'] = df['body'] <= (df['high'] - df['low']) * 0.1

    # Martillo (Hammer): cuerpo pequeño con mecha inferior larga
    df['hammer'] = (df['body'] < (df['upper_wick'] + df['lower_wick']) * 0.3) & \
                   (df['lower_wick'] > df['body'] * 2)

    # Estrella fugaz (Shooting Star): cuerpo pequeño con mecha superior larga
    df['shooting_star'] = (df['body'] < (df['upper_wick'] + df['lower_wick']) * 0.3) & \
                          (df['upper_wick'] > df['body'] * 2)

    # Engulfing alcista: vela verde que envuelve la roja anterior
    df['bullish_engulfing'] = (
        (df['close'] > df['open']) &
        (df['close'].shift(1) < df['open'].shift(1)) &
        (df['close'] >= df['open'].shift(1)) &
        (df['open'] <= df['close'].shift(1))
    )

    # Engulfing bajista: vela roja que envuelve la verde anterior
    df['bearish_engulfing'] = (
        (df['close'] < df['open']) &
        (df['close'].shift(1) > df['open'].shift(1)) &
        (df['open'] >= df['close'].shift(1)) &
        (df['close'] <= df['open'].shift(1))
    )

    return df

def explosive_signal(df, pattern_flags, live=False):
    patterns = ['doji', 'hammer', 'shooting_star', 'bullish_engulfing', 'bearish_engulfing']
    active_patterns = [pat for pat, flag in zip(patterns, pattern_flags) if flag]

    signal = pd.Series(False, index=df.index)
    for pat in active_patterns:
        signal |= df[pat]

    if not live:
        signal = signal.shift(1)  # evitar lookahead bias

    df['signal'] = signal.fillna(False)
    return df

