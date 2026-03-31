#!/usr/bin/env python3
"""
check_ranging_short_signals.py

Standalone script to check ranging_short signals for 16_ranging_short_6Hutc.
Fetches live candles from Bitget API and runs signal detection.

Usage:
    python check_ranging_short_signals.py
    python check_ranging_short_signals.py --symbols BTCUSDT ETHUSDT
    python check_ranging_short_signals.py --show-all
"""

import os
import sys
import argparse
import time
import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# PATH SETUP - adjust to your project structure
# ---------------------------------------------------------------------------
BOT_PATH = "/home/javi/projects/quant/quant_g/bitget/BOT_trading"
sys.path.insert(0, BOT_PATH)

# ---------------------------------------------------------------------------
# IMPORT SIGNAL FUNCTION
# ---------------------------------------------------------------------------
from signals.add_signals_ranging import ranging_short

# ---------------------------------------------------------------------------
# STRATEGY PARAMETERS (16_ranging_short_6Hutc)
# ---------------------------------------------------------------------------
STRATEGY_ID  = "16_ranging_short_6Hutc"
TIMEFRAME    = "6Hutc"
LOOKBACK     = 10
TOLERANCE    = 5
RANGE_STR    = 25
MA_PERIOD    = 10
TP_PCT       = 4
SL_PCT       = 6

# ---------------------------------------------------------------------------
# API CONFIG
# ---------------------------------------------------------------------------
BASE_URL     = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"
API_LIMIT    = 180
API_TIMEOUT  = 10
API_RETRIES  = 3

# ---------------------------------------------------------------------------
# SYMBOLS FILE
# ---------------------------------------------------------------------------
SYMBOLS_FILE = os.path.join(BOT_PATH, "symbols_live", f"symbols_live_{STRATEGY_ID}_{TIMEFRAME}.csv")


# ===========================================================================
# API FUNCTIONS
# ===========================================================================

def _http_get(url, params=None):
    for attempt in range(1, API_RETRIES + 1):
        try:
            r = requests.get(url, params=params, timeout=API_TIMEOUT)
            if r.status_code in (429, 502, 503, 504) or r.status_code >= 500:
                time.sleep(0.5 * attempt)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            time.sleep(0.5 * attempt)
    raise Exception(f"API request failed after {API_RETRIES} retries: {url}")


def fetch_candles(symbol: str, timeframe: str, limit: int = API_LIMIT) -> pd.DataFrame:
    url = f"{BASE_URL}/api/v2/mix/market/history-candles"
    params = {
        "symbol": symbol,
        "granularity": timeframe,
        "limit": limit,
        "productType": PRODUCT_TYPE
    }
    try:
        r = _http_get(url, params=params)
        data = r.json().get("data", [])
        if not data:
            return pd.DataFrame()

        clean = []
        for row in data:
            if not row or len(row) < 7:
                continue
            try:
                clean.append([int(row[0]), row[1], row[2], row[3], row[4], row[5], row[6]])
            except Exception:
                continue

        df = pd.DataFrame(clean, columns=["timestamp", "open", "high", "low", "close", "volume_base", "volume_quote"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        df = df.sort_values("timestamp").reset_index(drop=True)

        for col in ["open", "high", "low", "close", "volume_quote"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        return df

    except Exception as e:
        print(f"  [ERROR] Failed to fetch {symbol}: {e}")
        return pd.DataFrame()


def df_to_arrays(df: pd.DataFrame) -> dict:
    return {
        "ts":           df["timestamp"].to_numpy(),
        "open":         df["open"].to_numpy(dtype=np.float64),
        "high":         df["high"].to_numpy(dtype=np.float64),
        "low":          df["low"].to_numpy(dtype=np.float64),
        "close":        df["close"].to_numpy(dtype=np.float64),
        "volume_quote": df["volume_quote"].to_numpy(dtype=np.float64),
    }


# ===========================================================================
# SYMBOL LOADING
# ===========================================================================

def load_symbols(override: list = None) -> list:
    if override:
        return sorted(override)

    if not os.path.exists(SYMBOLS_FILE):
        raise FileNotFoundError(f"Symbol file not found: {SYMBOLS_FILE}")

    df = pd.read_csv(SYMBOLS_FILE, header=None)
    symbols = df.iloc[:, 0].dropna().astype(str).tolist()
    print(f"Loaded {len(symbols)} symbols from {SYMBOLS_FILE}")
    return sorted(symbols)


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description=f"Check signals for {STRATEGY_ID}")
    parser.add_argument("--symbols", nargs="+", help="Override symbols list")
    parser.add_argument("--show-all", action="store_true", help="Show all symbols, not just signals")
    args = parser.parse_args()

    symbols = load_symbols(args.symbols)

    print()
    print("=" * 60)
    print(f"  Strategy  : {STRATEGY_ID}")
    print(f"  Timeframe : {TIMEFRAME}")
    print(f"  Params    : lookback={LOOKBACK}, tolerance={TOLERANCE}, range_str={RANGE_STR}, ma_period={MA_PERIOD}")
    print(f"  Symbols   : {len(symbols)}")
    print("=" * 60)

    signals_found = []
    errors        = []


    for sym in symbols:
        df = fetch_candles(sym, TIMEFRAME)
    
        if sym == "BTCUSDT":
            print(f"\n--- Sample candles for {sym} (last 5) ---")
            print(df.tail())
            print(f"Total candles: {len(df)}\n")
    
        if df.empty or len(df) < LOOKBACK + MA_PERIOD:
            errors.append(sym)
            if args.show_all:
                print(f"  {sym:<20} | NO DATA")
            continue

        arr     = df_to_arrays(df)
        signals = ranging_short(
            arr,
            lookback=LOOKBACK,
            tolerance=TOLERANCE,
            range_str=RANGE_STR,
            ma_period=MA_PERIOD,
            live_trading=True
        )

        last_signal    = signals[-1]
        last_candle_ts = df["timestamp"].iloc[-1]
        last_close     = df["close"].iloc[-1]

        if last_signal == -1:
            signals_found.append({
                "symbol": sym,
                "ts":     last_candle_ts,
                "close":  last_close
            })
            print(f"  {sym:<20} | SIGNAL -1 | {last_candle_ts} | close={last_close:.4f}")
        else:
            if args.show_all:
                print(f"  {sym:<20} | no signal | {last_candle_ts} | close={last_close:.4f}")

    print()
    print("=" * 60)
    print(f"  Signals found : {len(signals_found)} / {len(symbols)}")
    if errors:
        print(f"  Errors        : {len(errors)} symbols with no data")
    print("=" * 60)

    if signals_found:
        print()
        print("SIGNALS SUMMARY:")
        for s in signals_found:
            print(f"  {s['symbol']:<20} | {s['ts']} | close={s['close']:.4f}")


if __name__ == "__main__":
    main()