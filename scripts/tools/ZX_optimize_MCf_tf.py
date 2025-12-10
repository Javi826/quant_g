import random
import numpy as np
import pandas as pd

DTYPE = np.float32


def compute_candle_features(df, raw_columns=[]):
    df = df.copy()
    df["pct_open_low"]   = (df["low"] - df["open"]) / df["open"]
    df["pct_open_high"]  = (df["high"] - df["open"]) / df["open"]
    df["pct_open_close"] = (df["close"] - df["open"]) / df["open"]

    if len(df.index) >= 2:
        time_index = (df.index[1:] - df.index[:-1]).total_seconds()
        mode = pd.Series(time_index).mode()[0]
        time_index = np.insert(time_index, 0, mode)
    else:
        time_index = np.zeros(len(df.index))

    df["time_variation"] = time_index

    index_sec = df.index.view(np.int64) // 10**9
    low_sec   = pd.to_datetime(df["low_time"]).view(np.int64) // 10**9
    high_sec  = pd.to_datetime(df["high_time"]).view(np.int64) // 10**9
    df["var_low_time"]  = (low_sec - index_sec).astype(float)
    df["var_high_time"] = (high_sec - index_sec).astype(float)

    df_raw = df[raw_columns].copy() if raw_columns else pd.DataFrame(index=df.index)
    return df, df_raw


def generate_multiple_paths(df_hist, n_paths, n_obs, raw_columns=[], base_seed=42):
    df_features, df_raw = compute_candle_features(df_hist, raw_columns)
    n_rows = len(df_features)
    if n_rows == 0 or n_obs == 0:
        return np.empty((0, 0, 0))

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

    n_features     = data_array.shape[1]
    n_raw          = n_features - 6
    n_features_out = 7 + n_raw

    start_price     = float(df_features["open"].iloc[-1])
    start_timestamp = df_features.index[-1].value // 10**9

    paths_array = np.empty((n_paths, n_obs, n_features_out), dtype=np.float64)

    for i in range(n_paths):
        rnd     = random.Random(base_seed + i)
        indices = np.array([rnd.randrange(n_rows) for _ in range(n_obs)], dtype=np.int64)
        sampled = data_array[indices]

        pct_open_low, pct_open_high, pct_open_close = sampled[:, 0], sampled[:, 1], sampled[:, 2]

        multipliers  = 1.0 + pct_open_close
        close_prices = start_price * np.cumprod(multipliers)
        open_prices  = np.empty_like(close_prices)
        open_prices[0] = start_price
        open_prices[1:] = close_prices[:-1]

        low_prices  = np.minimum(open_prices * (1.0 + pct_open_low), close_prices)
        high_prices = np.maximum(open_prices * (1.0 + pct_open_high), close_prices)

        cumul_seconds = np.cumsum(sampled[:, 3])
        times      = start_timestamp + cumul_seconds
        low_times  = times + sampled[:, 4]
        high_times = times + sampled[:, 5]

        # stack completo
        base_cols = [
                open_prices, 
                low_prices, 
                high_prices, 
                close_prices, 
                low_times,      # Timestamp cuando ocurrió el low
                high_times,     # Timestamp cuando ocurrió el high
                times           # NUEVO: Timestamp de inicio de la vela
            ]
        if n_raw > 0:
            for idx_col in range(n_raw):
                base_cols.append(sampled[:, 6 + idx_col])
        paths_array[i, :, :] = np.column_stack(base_cols)

    return paths_array.astype(DTYPE, copy=False)


def derive_major_from_minor(paths_minor: np.ndarray, factor: int = 6) -> np.ndarray:
    n_paths, n_obs, n_features = paths_minor.shape
    n_obs_major = n_obs // factor
    paths_major = np.empty((n_paths, n_obs_major, n_features), dtype=paths_minor.dtype)

    for p in range(n_paths):
        path = paths_minor[p]
        for j in range(n_obs_major):
            start_idx = j * factor
            end_idx = (j + 1) * factor
            block = path[start_idx:end_idx]

            open_ = block[0, 0]
            close = block[-1, 3]

            # High y low con timestamps CORRECTOS
            high_idx = np.argmax(block[:, 2])
            high = block[high_idx, 2]
            high_t = block[high_idx, 5]  # Ya es timestamp absoluto

            low_idx = np.argmin(block[:, 1])
            low = block[low_idx, 1]
            low_t = block[low_idx, 4]  # Ya es timestamp absoluto

            # Timestamp del major = timestamp de inicio del bloque
            major_timestamp = block[0, 6]  # Nueva columna 6

            paths_major[p, j, 0] = open_
            paths_major[p, j, 1] = low
            paths_major[p, j, 2] = high
            paths_major[p, j, 3] = close
            paths_major[p, j, 4] = low_t
            paths_major[p, j, 5] = high_t
            paths_major[p, j, 6] = major_timestamp  # Timestamp del major

    return paths_major

def generate_paths_for_all_symbols_functional(ohlcv_data, n_paths, n_obs, raw_columns=[]):
    paths_per_symbol = {}
    for symbol, df_hist in ohlcv_data.items():
        arr_paths = generate_multiple_paths(df_hist, n_paths=n_paths, n_obs=n_obs, raw_columns=raw_columns)
        if arr_paths is not None and arr_paths.shape[0] > 0:
            paths_per_symbol[symbol] = arr_paths
    return paths_per_symbol