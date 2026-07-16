#shared/shared_batchs/tools/optimize_MC.py
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

def _sample_indices(rnd, n_rows, n_obs, block_size):
    if block_size <= 1:
        return np.array([rnd.randrange(n_rows) for _ in range(n_obs)], dtype=np.int64)
    n_blocks = n_rows - block_size + 1
    n_blocks_needed = int(np.ceil(n_obs / block_size))
    starts = np.array([rnd.randrange(n_blocks) for _ in range(n_blocks_needed)], dtype=np.int64)
    indices = np.concatenate([np.arange(s, s + block_size) for s in starts])[:n_obs]
    return indices

def generate_multiple_paths(df_hist, n_paths, n_obs, raw_columns=[], base_seed=42, block_size=1):
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
# =============================================================================
#     start_price     = float(df_features["open"].iloc[0])
#     start_timestamp = df_features.index[0].value // 10**9
# =============================================================================
    
    paths_array   = np.empty((n_paths, n_obs, n_features_out), dtype=np.float64)
    effective_block_size = min(block_size, n_rows) if block_size > 1 else 1
    for i in range(n_paths):
        rnd     = random.Random(base_seed + i)
        indices = _sample_indices(rnd, n_rows, n_obs, effective_block_size)
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

def generate_paths_for_all_symbols_functional(ohlcv_data, n_paths, n_obs, raw_columns=[], block_size=1):
    paths_per_symbol = {}
    for symbol, df_hist in ohlcv_data.items():
        arr_paths = generate_multiple_paths(df_hist, n_paths=n_paths, n_obs=n_obs, raw_columns=raw_columns, block_size=block_size)
        if arr_paths is not None and arr_paths.shape[0] > 0:
            paths_per_symbol[symbol] = arr_paths
    return paths_per_symbol