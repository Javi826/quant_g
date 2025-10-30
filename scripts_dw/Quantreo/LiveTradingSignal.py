import numpy as np
from Quantreo.DataPreprocessing import *
from Quantreo.MetaTrader5 import *

from joblib import load


def random(symbol):
    values = [True, False]
    buy = np.random.choice(values)
    sell = not buy
    return buy, sell


def li_2023_02_RsiSma(symbol, timeframe, fast_sma_period, slow_sma_period, rsi_period):
    df = get_rates(symbol=symbol, number_of_data=500, timeframe=timeframe)
    df = sma(df, "close", fast_sma_period)
    df = sma(df, "close", slow_sma_period)
    df = rsi(df, "close", rsi_period)

    # def signal
    df["RSI_retarded"] = df[f"RSI"].shift(1)
    condition_1_buy = df[f"SMA_{fast_sma_period}"].iloc[-1] < df[f"SMA_{slow_sma_period}"].iloc[-1]
    condition_1_sell = df[f"SMA_{fast_sma_period}"].iloc[-1] > df[f"SMA_{slow_sma_period}"].iloc[-1]

    condition_2_buy = df[f"RSI"].iloc[-1] > df["RSI_retarded"].iloc[-1]
    condition_2_sell = df[f"RSI"].iloc[-1] < df["RSI_retarded"].iloc[-1]

    buy = condition_1_buy & condition_2_buy
    sell = condition_1_sell & condition_2_sell
    return buy, sell

