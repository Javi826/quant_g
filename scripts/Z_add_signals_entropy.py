import numpy as np
import pandas as pd
import ta

def signal_99_long(arr,
                   sma_cross=False, rsi_14=False, macd_cross=False,
                   momentum=False, stoch=False, cci=False, adx=False,
                   roc=False, ema_cross=False,
                   umbral=None, live_trading=True):

    closes  = pd.Series(arr['close'])
    highs   = pd.Series(arr.get('high', arr['close']))
    lows    = pd.Series(arr.get('low', arr['close']))

    # Lista de señales parciales
    signals_list = []

    # --- Indicadores ---
    if sma_cross:
        sma_fast = closes.rolling(10).mean()
        sma_slow = closes.rolling(50).mean()
        signals_list.append((sma_fast > sma_slow).astype(int))

    if rsi_14:
        rsi = ta.momentum.RSIIndicator(closes, window=14).rsi()
        signals_list.append((rsi < 70).astype(int))

    if macd_cross:
        macd = ta.trend.MACD(closes)
        signals_list.append((macd.macd_diff() > 0).astype(int))

    if momentum:
        mom = closes - closes.shift(10)
        signals_list.append((mom > 0).astype(int))

    if stoch:
        st = ta.momentum.StochasticOscillator(high=highs, low=lows, close=closes)
        signals_list.append((st.stoch_signal() < 80).astype(int))

    if cci:
        cci_val = ta.trend.CCIIndicator(high=highs, low=lows, close=closes, window=20).cci()
        signals_list.append((cci_val > -100).astype(int))

    if adx:
        adx_val = ta.trend.ADXIndicator(high=highs, low=lows, close=closes, window=14).adx()
        signals_list.append((adx_val > 20).astype(int))


    if roc:
        roc_val = ta.momentum.ROCIndicator(closes, window=12).roc()
        signals_list.append((roc_val > 0).astype(int))

    if ema_cross:
        ema_fast = closes.ewm(span=20).mean()
        ema_slow = closes.ewm(span=50).mean()
        signals_list.append((ema_fast > ema_slow).astype(int))

    # --- Combinar señales ---
    if not signals_list:
        signals = np.zeros(len(closes), dtype=np.int8)
    else:
        df_signals = pd.concat(signals_list, axis=1)
        if umbral is None:
            umbral = len(signals_list) // 2 + 1
        signals = (df_signals.sum(axis=1) >= umbral).astype(np.int8).to_numpy()

    # --- Shift para backtesting ---
    if not live_trading:
        signals = np.roll(signals, 1)
        signals[0] = 0

    return signals

def signal_99_short(arr,
                    sma_cross=False, rsi_14=False, macd_cross=False,
                    momentum=False, stoch=False, cci=False, adx=False,
                    roc=False, ema_cross=False,
                    umbral=None, live_trading=True):

    closes  = pd.Series(arr['close'])
    highs   = pd.Series(arr.get('high', arr['close']))
    lows    = pd.Series(arr.get('low', arr['close']))

    # Lista de señales parciales
    signals_list = []

    # --- Indicadores invertidos para short ---
    if sma_cross:
        sma_fast = closes.rolling(10).mean()
        sma_slow = closes.rolling(50).mean()
        signals_list.append((sma_fast < sma_slow).astype(int))  # 🔹 invertido

    if rsi_14:
        rsi = ta.momentum.RSIIndicator(closes, window=14).rsi()
        signals_list.append((rsi > 30).astype(int))  # 🔹 invertido

    if macd_cross:
        macd = ta.trend.MACD(closes)
        signals_list.append((macd.macd_diff() < 0).astype(int))  # 🔹 invertido

    if momentum:
        mom = closes - closes.shift(10)
        signals_list.append((mom < 0).astype(int))  # 🔹 invertido

    if stoch:
        st = ta.momentum.StochasticOscillator(high=highs, low=lows, close=closes)
        signals_list.append((st.stoch_signal() > 20).astype(int))  # 🔹 invertido

    if cci:
        cci_val = ta.trend.CCIIndicator(high=highs, low=lows, close=closes, window=20).cci()
        signals_list.append((cci_val < 100).astype(int))  # 🔹 invertido

    if adx:
        adx_val = ta.trend.ADXIndicator(high=highs, low=lows, close=closes, window=14).adx()
        signals_list.append((adx_val > 20).astype(int))  # 🔹 ADX se mantiene igual, solo indica fuerza de tendencia

    if roc:
        roc_val = ta.momentum.ROCIndicator(closes, window=12).roc()
        signals_list.append((roc_val < 0).astype(int))  # 🔹 invertido

    if ema_cross:
        ema_fast = closes.ewm(span=20).mean()
        ema_slow = closes.ewm(span=50).mean()
        signals_list.append((ema_fast < ema_slow).astype(int))  # 🔹 invertido

    # --- Combinar señales ---
    if not signals_list:
        signals = np.zeros(len(closes), dtype=np.int8)
    else:
        df_signals = pd.concat(signals_list, axis=1)
        if umbral is None:
            umbral = len(signals_list) // 2 + 1
        signals = (df_signals.sum(axis=1) >= umbral).astype(np.int8).to_numpy()

    # --- Shift para backtesting ---
    if not live_trading:
        signals = np.roll(signals, 1)
        signals[0] = 0

    return signals
