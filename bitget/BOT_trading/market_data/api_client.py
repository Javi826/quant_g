"""
Bitget API Client - OHLCV data fetching utilities.

This module provides low-level API functions for fetching candle data
from Bitget futures API. Used by BOT_trading for live market data.

Extracted from parquet_process/Z_parquet_A0_extraction.py
"""

import time
import requests
import pandas as pd
from config.settings import BASE_URL,PRODUCT_TYPE,API_TIMEOUT,API_MAX_RETRIES
import logging
logger = logging.getLogger('BOT_trading.market_data.api_client')

def _http_get(url, params=None, timeout=API_TIMEOUT, max_retries=API_MAX_RETRIES):
    attempt = 0
    while attempt < max_retries:
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code in (429, 502, 503, 504) or r.status_code >= 500:
                attempt += 1
                time.sleep(0.5 * attempt)
                continue
            r.raise_for_status()
            return r
        except requests.RequestException:
            attempt += 1
            time.sleep(0.5 * attempt)
    raise Exception("No more tryies.")


def _call_history_candles(symbol, granularity, limit=200, startTime=None, endTime=None):
    url = f"{BASE_URL}/api/v2/mix/market/history-candles"
    params = {
        "symbol": symbol,
        "granularity": granularity,
        "limit": limit,
        "productType": PRODUCT_TYPE
    }
    if startTime is not None:
        params["startTime"] = str(int(startTime))
    if endTime is not None:
        params["endTime"] = str(int(endTime))
    try:
        r = _http_get(url, params=params)
        j = r.json()
        if isinstance(j, dict) and j.get("code") and j.get("code") != "00000":
            return []
        data = j.get("data") if isinstance(j, dict) else j
        if not data:
            return []
        if isinstance(data, list):
            return data
        return []
    except Exception as e:
        logger.error(f"Error-API (symbol={symbol} start={startTime} end={endTime}): {e}")
        return []

def to_dataframe_from_api(data):
    if not data:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume_base", "volume_quote"])
    clean = []
    for row in data:
        if not row or len(row) < 7:
            continue
        try:
            ts_int = int(row[0])
        except Exception:
            continue
        clean.append([ts_int, row[1], row[2], row[3], row[4], row[5], row[6]])
    df = pd.DataFrame(clean, columns=["timestamp", "open", "high", "low", "close", "volume_base", "volume_quote"])
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df

def get_futures_symbols_from_api(product_type=PRODUCT_TYPE):
    """Devuelve lista de símbolos (robusto)."""
    url = f"{BASE_URL}/api/v2/mix/market/contracts"
    params = {'productType': product_type}
    try:
        r = _http_get(url, params=params)
        j = r.json()
        data = j.get('data') if isinstance(j, dict) else None
        if not data:
            return []
        symbols = []
        for item in data:
            s = item.get('symbol') or item.get('contract') or item.get('symbolName')
            if s:
                symbols.append(str(s))
        return sorted(set(symbols))
    except Exception as e:
        logger.error(f"Error-from API: {e}")
        return []