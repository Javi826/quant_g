#MAIN EXTRACTION
# -----------------------------
import os
import time
import re
import requests
import pandas as pd
from datetime import datetime, timezone

# ---------------- CONFIG ----------------
BASE_URL               = "https://api.bitget.com"
PRODUCT_TYPE           = "usdt-futures"

TIMEFRAME              = "4H"          
LIMIT                  = 200           
DATA_FOLDER            = "crypto_2021"
START_DATE             = "2021-01-02"  
REQUEST_TIMEOUT        = 20
SLEEP_BETWEEN_REQUESTS = 0.06  
MAX_ITER_PER_SYMBOL    = 2000     
MAX_RETRIES            = 3                
# ----------------------------------------

MS_90_DAYS = 90 * 24 * 60 * 60 * 1000

# ---------------- UTILIDADES ----------------

def sanitize_filename(name):
    """Sanitiza nombre de fichero quitando caracteres peligrosos y espacios."""
    safe = re.sub(r'[^\w\-_\. ]', '_', name)
    return safe.strip()

def parse_timeframe_to_ms(tf):
    """
    Convierte TIMEFRAME (ej "1m","1H","1D","1W","1M", con o sin 'utc') a milisegundos aproximados.
    """
    s = str(tf).strip()
    s = s.lower().replace('utc', '')
    m = re.match(r'^(\d+)([mhdwM])$', s)
    if not m:
        raise ValueError(f"Timeframe no reconocido: '{tf}'")
    n = int(m.group(1))
    u = m.group(2)
    if u == 'm':
        return n * 60 * 1000
    if u == 'h':
        return n * 60 * 60 * 1000
    if u == 'd':
        return n * 24 * 60 * 60 * 1000
    if u == 'w':
        return n * 7 * 24 * 60 * 60 * 1000
    if u == 'M':
        return n * 30 * 24 * 60 * 60 * 1000
    raise ValueError(f"Unidad de timeframe no soportada: '{tf}'")

# ---------------- API WRAPPERS ----------------

def _http_get(url, params=None, timeout=REQUEST_TIMEOUT, max_retries=MAX_RETRIES):
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
    raise Exception("Reintentos agotados")

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
        print(f"⚠️ Error from API: {e}")
        return []

def _call_history_candles(symbol, granularity, limit=LIMIT, startTime=None, endTime=None):
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
        print(f"⚠ Error API (symbol={symbol} start={startTime} end={endTime}): {e}")
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

# ---------------- FUNCIONES DE DESCARGA ----------------

def find_earliest_available_timestamp(symbol, gran_ms, timeframe, max_iters=500):
    end = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    earliest_found = None
    iters = 0
    prev_end = None
    while iters < max_iters:
        iters += 1
        data = _call_history_candles(symbol, timeframe, limit=LIMIT, startTime=None, endTime=end)
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        if not data:
            return earliest_found
        timestamps = []
        for item in data:
            try:
                timestamps.append(int(item[0]))
            except Exception:
                continue
        if not timestamps:
            return earliest_found
        min_ts = min(timestamps)
        new_end = min_ts - gran_ms
        if prev_end is not None and new_end == prev_end:
            return earliest_found or min_ts
        prev_end = end
        earliest_found = min(earliest_found, min_ts) if earliest_found else min_ts
        if new_end < 0 or new_end >= end:
            return earliest_found
        end = new_end
    return earliest_found

def download_candles_from_start(symbol, start_ms, gran_ms, timeframe, max_iters=2000):
    all_rows = []
    now_ms = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    current_start = int(start_ms)
    if current_start >= now_ms:
        return pd.DataFrame()
    window_ms = min(gran_ms * LIMIT, MS_90_DAYS)
    iters = 0
    prev_start = None
    no_progress_count = 0
    while current_start < now_ms and iters < max_iters:
        iters += 1
        window_end = min(current_start + window_ms, now_ms)
        data = _call_history_candles(symbol, timeframe, limit=LIMIT, startTime=current_start, endTime=window_end)
        time.sleep(SLEEP_BETWEEN_REQUESTS)
        if not data:
            next_start = window_end + gran_ms
            if next_start <= current_start:
                no_progress_count += 1
            else:
                no_progress_count = 0
            current_start = next_start
            if no_progress_count >= 3:
                break
            continue
        valid_rows = []
        timestamps = []
        for row in data:
            try:
                ts = int(row[0])
                timestamps.append(ts)
                valid_rows.append(row)
            except Exception:
                continue
        if not valid_rows:
            current_start = window_end + gran_ms
            continue
        all_rows.extend(valid_rows)
        max_ts = max(timestamps)
        if max_ts <= current_start:
            current_start = window_end + gran_ms
        else:
            current_start = max_ts + gran_ms
        if prev_start is not None and current_start <= prev_start:
            no_progress_count += 1
        else:
            no_progress_count = 0
        prev_start = current_start
        if no_progress_count >= 5:
            break
    df = to_dataframe_from_api(all_rows)
    if not df.empty:
        df = df.drop_duplicates(subset=["timestamp"], keep="first").reset_index(drop=True)
    return df


def process_symbol(sym, start_ms_requested, gran_ms, timeframe, data_folder=DATA_FOLDER):
    """
    Descarga las velas de un símbolo y guarda en Excel y Parquet (tz-naive para timestamp).
    """
    try:
        df = download_candles_from_start(sym, start_ms_requested, gran_ms, timeframe)
        start_dt = pd.to_datetime(START_DATE).tz_localize("UTC")
        first_ts = df["timestamp"].min()
        if first_ts > start_dt:
           
            print(f"⚠ No data from initial date for {sym}. Searching firest candle date...")
            earliest_ts = find_earliest_available_timestamp(sym, gran_ms, timeframe)
            if earliest_ts is None:
                print(f"❌ No candles for {sym}. Discarded.")
                return
            earliest_dt = datetime.fromtimestamp(earliest_ts / 1000, tz=timezone.utc)
            print(f"✅ First candle found: {earliest_dt.strftime('%Y-%m-%d %H:%M:%S')} UTC. Downloading data from this date.")
            df = download_candles_from_start(sym, earliest_ts, gran_ms, timeframe)
            if df.empty:
                print(f"❌ Error downloading frome first candle for {sym}. Discarded.")
                return

        # tz-naive
        if "timestamp" in df.columns and pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)

        # Crear carpeta si no existe
        os.makedirs(data_folder, exist_ok=True)
        
        # Convertir columnas numéricas a float antes de guardar
        for col in ["open", "high", "low", "close", "volume_base", "volume_quote"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Guardar Parquet
        parquet_filename = sanitize_filename(f"{sym}_{timeframe}.parquet")
        parquet_path = os.path.join(data_folder, parquet_filename)
        df.to_parquet(parquet_path, index=False)

        # Guardar Excel
        excel_filename = sanitize_filename(f"{sym}_{timeframe}.xlsx")
        excel_path = os.path.join(data_folder, excel_filename)
        df.to_excel(excel_path, index=False)

        print(f"💾 Saved {len(df)} candles in '{os.path.basename(parquet_path)}' and excel.")

    except Exception as e:
        print(f"⚠ Error processing {sym}: {e}")


# ---------------- SCRIPT PRINCIPAL ----------------

def process_all_symbols(start_date_str=START_DATE, timeframe=TIMEFRAME):
    try:
        start_dt = pd.to_datetime(start_date_str)
        if start_dt.tzinfo is None:
            start_dt = start_dt.tz_localize("UTC")
        else:
            start_dt = start_dt.tz_convert("UTC")
    except Exception:
        start_dt = pd.to_datetime(start_date_str, utc=True)
    start_ms_requested = int(start_dt.timestamp() * 1000)
    gran_ms = parse_timeframe_to_ms(timeframe)
    os.makedirs(DATA_FOLDER, exist_ok=True)
    symbols = get_futures_symbols_from_api(PRODUCT_TYPE)
    if not symbols:
        print("⚠️ No symbols. Aborting.")
        return
    print(f"🔁 Will be donwloaded candles from {start_dt.isoformat()} (UTC) until for {len(symbols)} símbolos.")
    for i, sym in enumerate(symbols, start=1):
        print(f"\n[{i}/{len(symbols)}] Processing {sym} ...")
        process_symbol(sym, start_ms_requested, gran_ms, timeframe, DATA_FOLDER)
    print("\n🏁 Process END.")


# ------------- Ejecutar -------------
if __name__ == "__main__":
    import time
    start_time = time.time()
    process_all_symbols(START_DATE, TIMEFRAME)
    end_time = time.time()    # <-- termina de contar
    elapsed = end_time - start_time
    minutes, seconds = divmod(elapsed, 60)
    print(f"\n⏱ Tiempo total de ejecución: {int(minutes)} min {int(seconds)} seg")
