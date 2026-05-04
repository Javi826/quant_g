import os
import sys
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
BASE   = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SHARED = os.path.join(BASE, "shared")
sys.path.insert(0, BASE)
sys.path.insert(0, SHARED)
sys.path.insert(0, os.path.dirname(__file__))

from ta.trend import ADXIndicator, SMAIndicator
from ta.momentum import RSIIndicator



# ---------------------------------------------------------------------------
# Inline numpy implementation of _compute_indicators
# ---------------------------------------------------------------------------
def _sma(close: np.ndarray, window: int) -> np.ndarray:
    out = np.full(len(close), np.nan)
    for i in range(window - 1, len(close)):
        out[i] = np.mean(close[i - window + 1:i + 1])
    return out


def _rsi(close: np.ndarray, window: int) -> np.ndarray:
    n   = len(close)
    out = np.full(n, np.nan)

    # ta uses close.diff(1) which has NaN at idx=0, so up/down arrays start with 0
    # np.diff produces length n-1 starting from idx=1 equivalent
    # prepend 0 to align with pandas behavior
    diff           = np.diff(close)
    up_direction   = np.concatenate([[0.0], np.where(diff > 0, diff,  0.0)])
    down_direction = np.concatenate([[0.0], np.where(diff < 0, -diff, 0.0)])

    alpha = 1.0 / window

    # EWM with adjust=False — matches pandas ewm(alpha=1/window, adjust=False)
    emaup = np.zeros(n)
    emadn = np.zeros(n)
    emaup[0] = up_direction[0]
    emadn[0] = down_direction[0]
    for i in range(1, n):
        emaup[i] = alpha * up_direction[i] + (1 - alpha) * emaup[i - 1]
        emadn[i] = alpha * down_direction[i] + (1 - alpha) * emadn[i - 1]

    # min_periods=window: first valid at index window-1 (0-indexed), ta confirmed idx=13
    for i in range(window - 1, n):
        if emadn[i] == 0:
            out[i] = 100.0
        else:
            rs     = emaup[i] / emadn[i]
            out[i] = 100.0 - (100.0 / (1.0 + rs))

    return out


def _adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, window: int) -> tuple:
    n           = len(close)
    close_shift = np.empty(n)
    close_shift[0]  = np.nan
    close_shift[1:] = close[:-1]

    # Use np.amax/amin to match ta._get_min_max (NaN propagation)
    pdm     = np.amax([high, close_shift], axis=0)
    pdn     = np.amin([low,  close_shift], axis=0)
    diff_dm = pdm - pdn   # NaN at idx 0

    diff_up   = high - np.roll(high, 1);  diff_up[0]   = np.nan
    diff_down = np.roll(low, 1) - low;    diff_down[0] = np.nan

    pos = np.abs(np.where((diff_up > diff_down)   & (diff_up > 0),   diff_up,   0.0))
    neg = np.abs(np.where((diff_down > diff_up)   & (diff_down > 0), diff_down, 0.0))

    # ta: diff_dm.dropna() for seed, diff_dm.reset_index() for loop (keeps original idx)
    diff_dm_dropna = diff_dm[~np.isnan(diff_dm)]  # = diff_dm[1:]
    # pos/neg: reset_index only (no dropna) — original indices preserved

    k = n - (window - 1)

    trs = np.zeros(k)
    dip = np.zeros(k)
    din = np.zeros(k)

    trs[0] = diff_dm_dropna[:window].sum()
    # ta pos/neg have NaN at idx 0 (pandas shift), so dropna().iloc[:window] = [1:window+1]
    dip[0] = pos[1:window + 1].sum()
    din[0] = neg[1:window + 1].sum()

    # ta loop: range(1, len(trs)-1), uses diff_dm.reset_index()[window+i]
    # diff_dm.reset_index() = diff_dm (same positions), so diff_dm[window+i]
    for i in range(1, k - 1):
        trs[i] = trs[i - 1] - trs[i - 1] / window + diff_dm[window + i]
        dip[i] = dip[i - 1] - dip[i - 1] / window + pos[window + i]
        din[i] = din[i - 1] - din[i - 1] / window + neg[window + i]

    with np.errstate(divide='ignore', invalid='ignore'):
        di_pos = np.where(trs != 0, 100.0 * dip / trs, 0.0)
        di_neg = np.where(trs != 0, 100.0 * din / trs, 0.0)

    denom = di_pos + di_neg
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = np.where(denom != 0, 100.0 * np.abs(di_pos - di_neg) / denom, 0.0)

    adx_smooth         = np.zeros(k)
    adx_smooth[window] = dx[:window].mean()
    for i in range(window + 1, k):
        adx_smooth[i] = (adx_smooth[i - 1] * (window - 1) + dx[i - 1]) / window

    prefix       = window - 1
    adx_full     = np.zeros(n)
    adx_pos_full = np.zeros(n)
    adx_neg_full = np.zeros(n)

    adx_full[prefix:prefix + k] = adx_smooth

    # ta adx_pos: for i in range(1, len(trs)-1): output at i + window
    for i in range(1, k - 1):
        out_idx = i + window
        if out_idx < n:
            adx_pos_full[out_idx] = 100.0 * dip[i] / trs[i] if trs[i] != 0 else 0.0
            adx_neg_full[out_idx] = 100.0 * din[i] / trs[i] if trs[i] != 0 else 0.0

    return adx_full, adx_pos_full, adx_neg_full


def _compute_indicators(arr: dict) -> dict:
    open_  = np.asarray(arr['open'],  dtype=np.float64)
    high   = np.asarray(arr['high'],  dtype=np.float64)
    low    = np.asarray(arr['low'],   dtype=np.float64)
    close  = np.asarray(arr['close'], dtype=np.float64)

    ma50                   = _sma(close, 50)
    rsi                    = _rsi(close, 14)
    adx, adx_pos, adx_neg = _adx(high, low, close, 14)

    return {
        'open': open_, 'high': high, 'low': low, 'close': close,
        'ma50': ma50, 'rsi': rsi, 'adx': adx,
        'plus_di': adx_pos, 'minus_di': adx_neg,
    }

# ---------------------------------------------------------------------------
# Load one symbol from IS data
# ---------------------------------------------------------------------------
DATA_FOLDER_IS = os.path.join(BASE, "data_pipeline", "data", "04_split",
                               "expanding", "IS", "crypto_2024-01_2025-04_IS")

parquet_file = next(f for f in os.listdir(DATA_FOLDER_IS) if f.endswith("_1H.parquet"))
symbol       = parquet_file.replace("_1H.parquet", "")
df_raw       = pd.read_parquet(os.path.join(DATA_FOLDER_IS, parquet_file))

arr = {
    'open':  df_raw['open'].to_numpy(dtype=np.float64),
    'high':  df_raw['high'].to_numpy(dtype=np.float64),
    'low':   df_raw['low'].to_numpy(dtype=np.float64),
    'close': df_raw['close'].to_numpy(dtype=np.float64),
}
print(f"Testing with symbol: {symbol}  n={len(arr['close'])}")

# ---------------------------------------------------------------------------
# ta reference
# ---------------------------------------------------------------------------
df          = pd.DataFrame({'open': arr['open'], 'high': arr['high'],
                            'low':  arr['low'],  'close': arr['close']})
ta_ma50     = SMAIndicator(df['close'], 50).sma_indicator().values
ta_rsi      = RSIIndicator(df['close'], 14).rsi().values
adx_ind     = ADXIndicator(df['high'], df['low'], df['close'], 14)
ta_adx      = adx_ind.adx().values
ta_plus_di  = adx_ind.adx_pos().values
ta_minus_di = adx_ind.adx_neg().values

# ---------------------------------------------------------------------------
# numpy version
# ---------------------------------------------------------------------------
ind = _compute_indicators(arr)

# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------
def check(name, ref, got):
    match = np.allclose(ref, got, equal_nan=True, rtol=1e-5, atol=1e-8)
    if match:
        print(f"  [PASS] {name}")
    else:
        diff_idx = np.where(~np.isclose(ref, got, equal_nan=True, rtol=1e-5, atol=1e-8))[0]
        print(f"  [FAIL] {name} — {len(diff_idx)} mismatches")
        print(f"         first 5 idx: {diff_idx[:5]}")
        for i in diff_idx[:5]:
            print(f"         [{i}] ref={ref[i]:.8f}  got={got[i]:.8f}")

print("\n--- DEBUG RSI values idx 12-18 ---")
for i in range(12, 19):
    print(f"  [{i}] ta={ta_rsi[i]:.8f}  np={ind['rsi'][i]:.8f}")
print("--- DEBUG +DI values idx 14-20 ---")
for i in range(14, 21):
    print(f"  [{i}] ta={ta_plus_di[i]:.8f}  np={ind['plus_di'][i]:.8f}")
print("---\n")
first_rsi     = next(i for i, v in enumerate(ta_rsi)     if not np.isnan(v) and v != 0)
first_adx     = next(i for i, v in enumerate(ta_adx)     if v != 0)
first_plus_di = next(i for i, v in enumerate(ta_plus_di) if v != 0)
first_minus_di= next(i for i, v in enumerate(ta_minus_di)if v != 0)
print(f"  ta RSI first non-nan:    idx={first_rsi}  val={ta_rsi[first_rsi]:.6f}")
print(f"  ta ADX first non-zero:   idx={first_adx}  val={ta_adx[first_adx]:.6f}")
print(f"  ta +DI first non-zero:   idx={first_plus_di}  val={ta_plus_di[first_plus_di]:.6f}")
print(f"  ta -DI first non-zero:   idx={first_minus_di}  val={ta_minus_di[first_minus_di]:.6f}")
print("------------------------\n")
print("Indicators clone test — ta vs numpy")
print("=" * 50)
check("SMA50",    ta_ma50,     ind['ma50'])
check("RSI14",    ta_rsi,      ind['rsi'])
check("ADX14",    ta_adx,      ind['adx'])
check("+DI14",    ta_plus_di,  ind['plus_di'])
check("-DI14",    ta_minus_di, ind['minus_di'])
print("=" * 50)