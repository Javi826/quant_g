import os
import time
import json
import random
import hashlib
import base64
import smtplib
import hmac
import numpy as np
import pandas as pd
import requests
from typing import Union
from urllib.parse import urlencode
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_DOWN
from email.mime.multipart import MIMEMultipart

np.random.seed(42)
random.seed(42)

# EMAIL CONFIG
# -----------------------------
EMAIL_FROM     = "jlahoz.ferrandez@gmail.com"
EMAIL_PASSWORD = "tvli cxgk duwh yzdd"
EMAIL_TO       = "jlahoz.ferrandez@gmail.com"

API_KEY        = "bg_afdcb9221ad98efb3b0b7bdd4c236338"
API_SECRET     = "0c4214cbfccfb648f841b43ca5d68531c8fb44b75ab271fdd222da9a74ee413f"
API_PASSPHRASE = "Cryptobitget86"

BASE_URL       = "https://api.bitget.com"
PRODUCT_TYPE   = 'usdt-futures'  

def seed_for_symbol(symbol: Union[str, object], base_seed: int = 42, path_idx: int = 0, mod: int = 100000) -> int:

    s = str(getattr(symbol, "name", symbol))
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    
    return int(base_seed) + (int(h, 16) % mod) + int(path_idx)
# SYMBOLS
# -----------------------------
def normalize_live_ohlcv(df):
    # Asegurarse de que los índices sean datetime
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df.index = pd.to_datetime(df['timestamp'])
        else:
            df.index = pd.to_datetime(df.index)

    # Convertir columnas clave a float
    for col in ['open', 'high', 'low', 'close', 'volume_base', 'volume_quote']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

def filter_symbols(symbols, 
                   min_vol_usdt, 
                   timeframe=None, 
                   data_folder=None, 
                   exchange=None,
                   min_price=None, 
                   vol_window=50,
                   date_min=None):  # <-- Nuevo parámetro


    ohlcv_data = {}
    filtered_symbols = []
    removed_symbols = []

    # Contadores por motivo de eliminación
    removed_by_reasons = {
        "No data": 0,
        "Not enough bars": 0,
        "Last close too low": 0,
        "Avg volume too low": 0,
        "Contains zeros": 0,
        "File missing": 0,
        "First candle before DATE_MIN": 0  # Nuevo motivo
    }

    for sym in symbols:
        df = None
        removed_reason = None

        # -------------------
        # Load data
        # -------------------
        if data_folder is None or timeframe is None:
            raise ValueError("Backtesting mode requires data_folder and timeframe")

        file_path = os.path.join(data_folder, f"{sym}_{timeframe}.parquet")

        if not os.path.exists(file_path):
            removed_reason = "File missing"
        else:
            df = pd.read_parquet(file_path)


        if removed_reason:
            removed_symbols.append(sym)
            removed_by_reasons[removed_reason] += 1
            continue

        # -------------------
        # 0 Cleaning
        # -------------------
        df = df.dropna(subset=['open','high','low','close','volume_quote'])
        df = df[(df[['open','high','low','close','volume_quote']] != 0).all(axis=1)]
        if df.empty:
            removed_reason = "Contains zeros"

        # ------------------- 
        # Index 
        # -------------------

        if not pd.api.types.is_datetime64_any_dtype(df.index):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")   # ✅ fijamos timestamp como índice y eliminamos la columna
            else:
                df.index = pd.to_datetime(df.index, errors="coerce")


        # ------------------- 
        # DATE_MIN
        # -------------------
        
        if removed_reason is None and date_min is not None:
            first_date = df.index.min()
            if first_date > pd.to_datetime(date_min):
                removed_reason = "First candle before DATE_MIN"


        # -------------------
        # Min price
        # -------------------
        if removed_reason is None and min_price is not None:
            last_close = df['close'].iloc[-1]
            if last_close <= min_price:
                removed_reason = "Last close too low"

        # -------------------
        # Avg volume (últimas vol_window velas)
        # -------------------
        if removed_reason is None:
            avg_vol = df['volume_quote'].tail(vol_window).mean()
            if avg_vol < min_vol_usdt:
                removed_reason = "Avg volume too low"

        # -------------------
        # Registrar resultado
        # -------------------
        if removed_reason:
            removed_symbols.append(sym)
            removed_by_reasons[removed_reason] += 1
        else:
            ohlcv_data[sym] = df
            filtered_symbols.append(sym)

    # -------------------
    # Summary
    # -------------------
    print(f"🔹 Total symbols BROKER        : {len(symbols)}")
    print(f"❌ Symbols removed total       : {len(removed_symbols)}")
    print(f"✅ Symbols remaining           : {len(filtered_symbols)}\n")

# =============================================================================
#     print("📊 Details by reason:")
#    for reason, count in removed_by_reasons.items():
#        print(f"   - {reason:<25}: {count}")
# =============================================================================

    return ohlcv_data, filtered_symbols, removed_symbols

def save_filtered_symbols(filtered_symbols, strategy="_",timeframe="10H",save_symbols=False, folder="symbols_live"):

    if save_symbols:
        os.makedirs(folder, exist_ok=True)  
        df_symbols   = pd.DataFrame({"Filtered_symbols": filtered_symbols})
        path_symbols = os.path.join(folder, f"symbols_live_{strategy}_{timeframe}.xlsx")
        df_symbols.to_excel(path_symbols, index=False)   
        print(f"📂 {len(filtered_symbols)} símbolos filtrados guardados en '{path_symbols}'")


def load_final_symbols(all_symbols,strategy="_",timeframe="10H"):

    folder = "symbols_live"
    try:

        path_live    = os.path.join(folder, f"symbols_live_{strategy}_{timeframe}.xlsx")
        df_live      = pd.read_excel(path_live)
        live_symbols = set(df_live.iloc[:, 0].dropna().astype(str))

        final_symbols = set(all_symbols) & live_symbols 

        print(f"🔹 symbols for Live: {len(final_symbols)}")
        return sorted(final_symbols)

    except Exception as e:
        print(f"⚠️ Error loading symbols: {e}")
        return []
    
def _now_ms():
    return str(int(time.time() * 1000))

def _body_to_str(body):
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False) if body else ""

def sign_request(timestamp, method, path, query_string, body_str):
    to_sign = timestamp + method.upper() + path
    if query_string:
        to_sign += "?" + query_string
    to_sign += body_str
    digest = hmac.new(API_SECRET.encode('utf-8'), to_sign.encode('utf-8'), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()

def send_request(method, path, params=None, body=None):
    ts           = _now_ms()
    query_string = urlencode(params) if params else ""
    body_str     = _body_to_str(body)
    sign         = sign_request(ts, method, path, query_string, body_str)
    headers = {
        "ACCESS-KEY": API_KEY,
        "ACCESS-SIGN": sign,
        "ACCESS-TIMESTAMP": ts,
        "ACCESS-PASSPHRASE": API_PASSPHRASE,
        "Content-Type": "application/json"
    }
    url = BASE_URL + path + (f"?{query_string}" if query_string else "")
    try:
        if method.upper() != "GET":
            r = requests.post(url, headers=headers, data=body_str.encode('utf-8'), timeout=15)
        else:
            r = requests.get(url, headers=headers, timeout=15)
        ct = r.headers.get("Content-Type", "")
        return r.status_code, r.json() if ct.startswith("application/json") else r.text
    except Exception as e:
        return 0, {"error": str(e)}
# =============================================================================
# PLACE ORDER
# =============================================================================
def place_order(symbol: str, usdt_amount: float = 100, tp_percent: float = 5, sl_percent: float = 5,
                product_type: str = "USDT-FUTURES", margin_coin: str = "USDT", margin_mode: str = "isolated"):

    # 1) último precio
    code, resp = send_request("GET", "/api/v2/mix/market/ticker", params={"productType": product_type, "symbol": symbol})
    if code != 200 or resp.get("code") != "00000":
        print("⚠️ Error for ticker:", resp)
        return None, None
    last_price = Decimal(str(resp['data'][0]['lastPr']))
    time.sleep(0.5)

    # 2) tamaño estimado (base)
    size_base = (Decimal(str(usdt_amount)) / last_price)

    # 3) obtener metadata de símbolos para sizeScale y price tick (robusto)
    code_info, resp_info = send_request("GET", "/api/v2/mix/market/symbols")
    price_tick = None
    size_scale = None
    if code_info == 200 and resp_info.get("code") == "00000":
        for s in resp_info.get("data", []):
            if s.get("symbol") == symbol:
                # varios posibles campos según la API/versión
                if "priceScale" in s and isinstance(s.get("priceScale"), int):
                    price_tick = Decimal(f"1e-{int(s.get('priceScale'))}")
                elif "tickSize" in s:
                    try:
                        price_tick = Decimal(str(s.get("tickSize")))
                    except:
                        pass
                elif "pricePrecision" in s:
                    price_tick = Decimal(f"1e-{int(s.get('pricePrecision'))}")
                # size scale
                if "sizeScale" in s:
                    try:
                        size_scale = int(s.get("sizeScale"))
                    except:
                        pass
                elif "qtyScale" in s:
                    try:
                        size_scale = int(s.get("qtyScale"))
                    except:
                        pass
                break

    # fallbacks seguros
    if price_tick is None:
        # fallback razonable: determina tick por magnitud del precio
        if last_price >= 1:
            price_tick = Decimal("0.01")
        elif last_price >= 0.1:
            price_tick = Decimal("0.001")
        else:
            price_tick = Decimal("0.00001")
    if size_scale is None or size_scale < 0:
        size_scale = 6

    precision_size = Decimal(f"1e-{size_scale}")

    # 4) quantizar size al sizeScale y asegurarnos > 0
    size_q = size_base.quantize(precision_size, rounding=ROUND_DOWN)
    if size_q == 0:
        # intentar fallback con 1e-6
        size_q = size_base.quantize(Decimal("1e-6"), rounding=ROUND_DOWN)
    if size_q == 0:
        print("⚠️ Size obtained = 0. Increase usdt_amount o adjust precision.")
        return None, None

    # 5) calcular TP/SL y quantizar al tick del símbolo
    tp_price = (last_price * (Decimal("1") + Decimal(str(tp_percent)) / 100)).quantize(price_tick, rounding=ROUND_DOWN)
    sl_price = (last_price * (Decimal("1") - Decimal(str(sl_percent)) / 100)).quantize(price_tick, rounding=ROUND_DOWN)

    # 6) colocar orden market incluyendo preset TP/SL (pre-quantized)
    body_order = {
        "symbol": symbol,
        "productType": product_type,
        "marginMode": margin_mode,
        "marginCoin": margin_coin,
        "size": format(size_q, "f"),
        "side": "buy",
        "tradeSide": "open",
        "orderType": "market",
        "clientOid": f"script-{int(time.time())}",
        "presetStopSurplusPrice": format(tp_price, "f"),
        "presetStopLossPrice": format(sl_price, "f")
    }

    code_order, resp_order = send_request("POST", "/api/v2/mix/order/place-order", body=body_order)
    if code_order != 200 or resp_order.get("code") != "00000":
        # Si la API responde error por tick, imprimimos detalle adicional para depuración
        print("⚠️ Error in market order:", resp_order)
        return None, None
    
    # 7) obtener cantidad ejecutada (robusto)
    filled_amount = Decimal("0")
    data          = resp_order.get("data") or {}
    for key in ("size", "filledSize", "filledQty", "filled_amount"):
        if key in data and data[key] is not None:
            filled_amount = Decimal(str(data[key]))
            break
    if filled_amount == 0:
        filled_amount = size_q

    # 8) tamaño para TP/SL — no exceder lo fillado y quantizar
    size_tpsl = filled_amount.quantize(precision_size, rounding=ROUND_DOWN)
    if size_tpsl == 0:
        size_tpsl = filled_amount.quantize(Decimal("1e-6"), rounding=ROUND_DOWN)
    if size_tpsl == 0:
        print("⚠️ After execution size_tpsl = 0. Aborting TP/SL.")
        return resp_order, None
    
    # precio real de compra (long)
    buy_price = Decimal(str(resp_order['data'].get('price', last_price)))
    
    print(f"⬆️ & 🎯 Position for {symbol} | Size: {filled_amount} | Price: {buy_price} | TP: {tp_price} | SL: {sl_price}")


    return resp_order, {"size_tpsl": format(size_tpsl, "f"), "tp_price": format(tp_price, "f"), "sl_price": format(sl_price, "f")}

def get_usdt_balance(exchange):
    balance = exchange.fetch_balance()
    return balance['free']['USDT']

def wait_for_next_candle(timeframe='4h'):
    now = datetime.utcnow()
    if timeframe.endswith('H'):
        minutes = int(timeframe[:-1]) * 60
    elif timeframe.endswith('m'):
        minutes = int(timeframe[:-1])
    else:
        raise ValueError("Timeframe incorrect, use 'm' or 'h'.")
    total_minutes      = now.hour * 60 + now.minute
    next_total_minutes = ((total_minutes // minutes) + 1) * minutes
    delta_minutes      = next_total_minutes - total_minutes
    next_run           = now + timedelta(minutes=delta_minutes, seconds=-now.second, microseconds=-now.microsecond)
    sleep_seconds      = (next_run - now).total_seconds()
    now                = datetime.utcnow()
    print(f"🕒 Waiting for next candle: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    time.sleep(sleep_seconds)
    
def save_results(grid_results, grid_results_df, filename="grid_backtest.xlsx",save=False):
    
    if save:
        # Crear directorio si no existe
        folder = os.path.dirname(filename)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

        grid_results_df.to_excel(filename, index=False)
        print(f"📂 File saved successfully as: {filename}")
      

def send_email(detected_cryptos):
    if not detected_cryptos: return
    msg = MIMEMultipart()
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_TO
    msg['Subject'] = f"Crypto_signals: {', '.join([d['symbol'] for d in detected_cryptos])}"
    body = "\n".join([f"{d['symbol']} | Signal: {d['signal_type']} | Close: {d['close']:.2f}" for d in detected_cryptos])
    msg.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"📧 Email sent: {', '.join([d['symbol'] for d in detected_cryptos])}")
    except Exception as e:
        print(f"⚠️ Error sending email: {e}")



