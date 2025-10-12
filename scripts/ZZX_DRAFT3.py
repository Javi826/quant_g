import random
import pandas as pd
import numpy as np
from numba import njit

DTYPE = np.float32

# ============================================================
# 1️⃣ Preprocesamiento: cálculo de features
# ============================================================
def compute_candle_features(df, raw_columns=[]):
    df = df.copy()

    # Porcentajes relativos vectorizados
    df["pct_open_low"] = (df["low"] - df["open"]) / df["open"]
    df["pct_open_high"] = (df["high"] - df["open"]) / df["open"]
    df["pct_open_close"] = (df["close"] - df["open"]) / df["open"]

    if len(df.index) >= 2:
        time_index = (df.index[1:] - df.index[:-1]).total_seconds()
        mode = pd.Series(time_index).mode()[0]
        time_index = np.insert(time_index, 0, mode)
    else:
        time_index = np.zeros(len(df.index))

    df["time_variation"] = time_index

    # Diferencias de tiempo low/high
    index_seconds = df.index.view(np.int64) // 10**9
    low_seconds = pd.to_datetime(df["low_time"]).view(np.int64) // 10**9
    high_seconds = pd.to_datetime(df["high_time"]).view(np.int64) // 10**9
    df["var_low_time"] = (low_seconds - index_seconds).astype(float)
    df["var_high_time"] = (high_seconds - index_seconds).astype(float)

    df_raw = df[raw_columns].copy() if raw_columns else pd.DataFrame(index=df.index)
    return df, df_raw


# ============================================================
# 2️⃣ Núcleo numérico compilado con Numba
# ============================================================
@njit
def _simulate_single_path_numba(sampled, start_price, start_timestamp):
    """
    Cálculo vectorizado de un solo path — compilado con Numba.
    sampled: array (n_obs, n_features)
    """
    n_obs = sampled.shape[0]
    open_prices = np.empty(n_obs, dtype=np.float64)
    close_prices = np.empty(n_obs, dtype=np.float64)
    low_prices = np.empty(n_obs, dtype=np.float64)
    high_prices = np.empty(n_obs, dtype=np.float64)
    times = np.empty(n_obs, dtype=np.float64)
    low_times = np.empty(n_obs, dtype=np.float64)
    high_times = np.empty(n_obs, dtype=np.float64)

    open_price = start_price
    current_time = start_timestamp

    for t in range(n_obs):
        pct_open_low  = sampled[t, 0]
        pct_open_high = sampled[t, 1]
        pct_open_close= sampled[t, 2]
        time_var      = sampled[t, 3]
        var_low_time  = sampled[t, 4]
        var_high_time = sampled[t, 5]

        current_time += time_var
        times[t] = current_time
        low_times[t]  = current_time + var_low_time
        high_times[t] = current_time + var_high_time

        close_price = open_price * (1.0 + pct_open_close)
        low_price   = open_price * (1.0 + pct_open_low)
        high_price  = open_price * (1.0 + pct_open_high)

        if low_price > close_price:
            low_price = close_price
        if high_price < close_price:
            high_price = close_price

        open_prices[t]  = open_price
        close_prices[t] = close_price
        low_prices[t]   = low_price
        high_prices[t]  = high_price

        open_price = close_price

    return open_prices, low_prices, high_prices, close_prices, times, low_times, high_times


# ============================================================
# 3️⃣ Función principal: genera múltiples paths con Numba
# ============================================================
def generate_multiple_paths(df_hist, n_paths=100, n_obs=1000, raw_columns=[], base_seed=42):
    df_features, df_raw = compute_candle_features(df_hist, raw_columns)
    n_rows = len(df_features)
    if n_rows == 0 or n_obs == 0:
        return []

    # Construcción del array base
    cols = [
        df_features["pct_open_low"].to_numpy(np.float64),
        df_features["pct_open_high"].to_numpy(np.float64),
        df_features["pct_open_close"].to_numpy(np.float64),
        df_features["time_variation"].to_numpy(np.float64),
        df_features["var_low_time"].to_numpy(np.float64),
        df_features["var_high_time"].to_numpy(np.float64)
    ]
    for rc in raw_columns:
        cols.append(df_raw[rc].to_numpy(np.float64))

    data_array = np.column_stack(cols)
    n_features = data_array.shape[1]

    start_price = float(df_features["open"].iloc[-1])
    start_timestamp = df_features.index[-1].timestamp()  # segundos UNIX

    paths = []

    # ========================================================
    # Bucle externo por path (Numba optimiza el interno)
    # ========================================================
    for i in range(n_paths):
        rnd = random.Random(base_seed + i)
        indices = np.array([rnd.randrange(n_rows) for _ in range(n_obs)], dtype=np.int64)
        sampled = data_array[indices]

        # Cálculo rápido con Numba
        open_p, low_p, high_p, close_p, times_s, low_s, high_s = _simulate_single_path_numba(
            sampled, start_price, start_timestamp
        )

        # Reconstrucción de fechas Pandas
        times = pd.to_datetime(times_s, unit="s")
        low_times = pd.to_datetime(low_s, unit="s")
        high_times = pd.to_datetime(high_s, unit="s")

        # Construcción del DataFrame
        data_dict = {
            "open": open_p,
            "low": low_p,
            "high": high_p,
            "close": close_p,
            "low_time": low_times,
            "high_time": high_times
        }

        # Añadir raw columns (si las hay)
        n_raw = n_features - 6
        if n_raw > 0:
            for idx_col, col_name in enumerate(raw_columns):
                data_dict[col_name] = sampled[:, 6 + idx_col]

        df_path = pd.DataFrame(data_dict, index=times)
        df_path.index.name = "time"
        df_path = df_path.astype({c: DTYPE for c in ["open", "low", "high", "close"]}, copy=False)
        paths.append(df_path)

    return paths
