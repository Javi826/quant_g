# === FILE: add_signals04.py ===
# ---------------------------------
import warnings
import logging
import pandas as pd
from utils.ZX_indicators import rolling_entropy_numba, delta_numba

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)


# -----------------------------
# CÁLCULO DE PATRONES DE VELAS
# -----------------------------

def add_indicators_04(df):

    # Cuerpo y mechas
    df['body'] = (df['close'] - df['open']).abs()
    df['upper_wick'] = df['high'] - df[['open','close']].max(axis=1)
    df['lower_wick'] = df[['open','close']].min(axis=1) - df['low']

    # Doji: cuerpo muy pequeño (apertura ≈ cierre)
    df['doji'] = df['body'] <= (df['high'] - df['low']) * 0.1

    # Martillo (Hammer): cuerpo pequeño con mecha inferior larga
    df['hammer'] = (df['body'] < (df['upper_wick'] + df['lower_wick']) * 0.3) & \
                   (df['lower_wick'] > df['body'] * 2)

    # Estrella fugaz (Shooting Star)
    df['shooting_star'] = (df['body'] < (df['upper_wick'] + df['lower_wick']) * 0.3) & \
                          (df['upper_wick'] > df['body'] * 2)

    # Engulfing alcista
    df['bullish_engulfing'] = (
        (df['close'] > df['open']) &
        (df['close'].shift(1) < df['open'].shift(1)) &
        (df['close'] >= df['open'].shift(1)) &
        (df['open'] <= df['close'].shift(1))
    )

    # Engulfing bajista
    df['bearish_engulfing'] = (
        (df['close'] < df['open']) &
        (df['close'].shift(1) > df['open'].shift(1)) &
        (df['open'] >= df['close'].shift(1)) &
        (df['close'] <= df['open'].shift(1))
    )

    # Piercing Line (alcista)
    df['piercing_line'] = (
        (df['close'].shift(1) < df['open'].shift(1)) &
        (df['close'] > df['open']) &
        (df['open'] < df['close'].shift(1)) &
        (df['close'] >= df['open'].shift(1) + 0.5 * (df['close'].shift(1) - df['open'].shift(1)))
    )

    # Dark Cloud Cover (bajista)
    df['dark_cloud_cover'] = (
        (df['close'].shift(1) > df['open'].shift(1)) &
        (df['close'] < df['open']) &
        (df['open'] > df['close'].shift(1)) &
        (df['close'] <= df['open'].shift(1) - 0.5 * (df['close'].shift(1) - df['open'].shift(1)))
    )

    # -----------------------------
    # CÁLCULO DE ENTROPÍA (igual que en add_signals03)
    # -----------------------------
    close = df['close'].values
    delta = delta_numba(close)
    entropia = rolling_entropy_numba(delta, 5, 10)
    df['entropia'] = entropia

    return df


def explosive_signal_04(df, pattern_flags, entropia_max=2.0, live=False):
    patterns = ['doji', 'hammer', 'shooting_star', 'bullish_engulfing', 'bearish_engulfing', 'piercing_line', 'dark_cloud_cover']
    active_patterns = [pat for pat, flag in zip(patterns, pattern_flags) if flag]

    signal = pd.Series(False, index=df.index)
    for pat in active_patterns:
        signal |= df[pat]

    signal &= (df['entropia'] < entropia_max)

    if not live:
        signal = signal.shift(1)

    df['signal'] = signal.fillna(False)
    return df
