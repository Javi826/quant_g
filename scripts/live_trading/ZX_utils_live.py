import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import numpy as np
import pandas as pd
import json
import uuid
from decimal import Decimal
from parquet_process.Z_parquet_A0_extraction import  _call_history_candles, to_dataframe_from_api
from Z_add_signals_reversal import trend_reversal_entry_short
from Z_add_signals_parity import detect_parity_short
from ZX_place_orders_sub import place_order
from pandas.api.types import is_datetime64_any_dtype
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
STATE_FILE    = os.path.join(os.path.dirname(__file__), 'tracked_orders_state.json')

MADRID_TZ = ZoneInfo("Europe/Madrid")
BASE_URL     = "https://api.bitget.com"
PRODUCT_TYPE = 'usdt-futures'

# =============================================================================
# SYMBOLS & CANDLES
# =============================================================================
def load_state(strategies, path=STATE_FILE):
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Error cargando estado: {e}")
    return {'strategies': {s['id']: [] for s in strategies}}

def save_state(state, path=STATE_FILE):
    try:
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, default=str)
    except Exception as e:
        print(f"⚠️ Error guardando estado: {e}")

def normalize_live_ohlcv(df):
    if not isinstance(df.index, pd.DatetimeIndex):
        if 'timestamp' in df.columns:
            df.index = pd.to_datetime(df['timestamp'])
        else:
            df.index = pd.to_datetime(df.index)

    for col in ['open', 'high', 'low', 'close', 'volume_base', 'volume_quote']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    return df

def make_client_oid(strategy_id):
    return f"{strategy_id}-{int(time.time())}-{uuid.uuid4().hex[:6]}"

def extract_filled_size_from_resp(resp_order):
    if not resp_order or 'data' not in resp_order:
        return 0.0
    data = resp_order.get('data') or {}
    for k in ('size', 'filledSize', 'filledQty', 'filled_amount', 'filled_size'):
        v = data.get(k)
        if v is not None:
            try:
                return float(v)
            except Exception:
                try:
                    return float(str(v))
                except Exception:
                    pass
    try:
        return float(data.get('size', 0) or 0)
    except Exception:
        return 0.0
    
def determine_side_for_open(direction):
    return 'buy' if direction.lower() == 'long' else 'sell'

def determine_side_for_close(direction):
    return 'sell' if direction.lower() == 'long' else 'buy'

def detect_signal_for_strategy(strategy, final_symbols):
    """
    Normaliza la salida de las funciones de señal y evita evaluar arrays directamente.
    Devuelve lista de dicts {'symbol', 'timestamp', 'close'}.
    """
    detected = []
    if not final_symbols:
        return detected

    ohlcv = fetch_ohlcv_data(final_symbols, strategy['timeframe'])
    for sym, df in ohlcv.items():
        if df is None or df.empty:
            continue
        df_norm = normalize_live_ohlcv(df)
        arr = df_to_arrays_live(df_norm)

        # obtener señales según estrategia
        try:
            if strategy['name'] == 'reversal_short':
                signals = trend_reversal_entry_short(
                    arr,
                    left_lookback=strategy['left_lookback'],
                    tolerance=strategy['tolerance'],
                    live_trading=True
                )
            elif strategy['name'] == 'parity_short':
                signals = detect_parity_short(
                    arr,
                    lookback=strategy['lookback'],
                    tolerance=strategy['tolerance'],
                    live_trading=True
                )
            else:
                signals = None
        except Exception as e:
            print(f"⚠️ Error ejecutando la función de señales para {sym} ({strategy['name']}): {e}")
            signals = None

        # Normalizar signals para evitar truthiness ambiguo
        if signals is None:
            continue

        # convertir a array numpy para inspección segura
        try:
            signals_arr = np.asarray(signals)
        except Exception:
            # fallback: intentar convertir a lista
            try:
                signals_arr = np.array(list(signals))
            except Exception:
                continue

        if signals_arr.size == 0:
            continue

        # tomar el último elemento
        last = signals_arr.flat[-1]

        # convertir last a array/numpy para comprobar si hay valores no nulos
        last_arr = np.asarray(last)

        # si cualquier elemento del último valor es distinto de 0, consideramos señal
        try:
            has_signal = np.any(last_arr != 0)
        except Exception:
            # si comparación falla, intentar comparación escalar
            try:
                has_signal = (float(last_arr) != 0.0)
            except Exception:
                has_signal = False

        if has_signal:
            last_row = df_norm.iloc[-1]
            detected.append({
                'symbol': sym,
                'timestamp': last_row.name if 'timestamp' not in df_norm.columns else last_row['timestamp'],
                'close': float(last_row['close'])
            })
    #FAKE
    now = datetime.now(MADRID_TZ).isoformat()
    return [
        {
            'symbol': 'BTCUSDT',
            'timestamp': now,
            'close': 50000.0
        },
        {
            'symbol': 'BNBUSDT',
            'timestamp': now,
            'close': 600.0
        }
    ]
       
    #return detected

def get_contract_info(symbol, product_type, send_request_fn):

    defaults = {
        'min_trade': Decimal('0'),
        'size_mult': Decimal('0.00000001'),
        'volume_place': 8,
        'max_market_order_qty': None
    }

    try:
        path = f"/api/v2/mix/market/contracts?productType={product_type}&symbol={symbol}"
        code_cfg, resp_cfg = send_request_fn('GET', path)

        if code_cfg == 200 and resp_cfg.get('code') == '00000' and resp_cfg.get('data'):
            data0 = resp_cfg['data'][0]

            min_trade = Decimal(str(data0.get('minTradeNum', '0') or '0'))
            size_mult = Decimal(str(data0.get('sizeMultiplier', '0.00000001') or '0.00000001'))
            volume_place = int(data0.get('volumePlace', 8))
            mmq = data0.get('maxMarketOrderQty')
            max_market_order_qty = Decimal(str(mmq)) if mmq not in (None, '') else None

            return {
                'min_trade': min_trade,
                'size_mult': size_mult,
                'volume_place': volume_place,
                'max_market_order_qty': max_market_order_qty
            }

    except Exception as e:
        print(f"⚠️ Error obteniendo info del contrato para {symbol}: {e}")

    return defaults

def get_order_detail(symbol, order_id=None, client_oid=None, product_type='USDT-FUTURES', send_request_fn=None):
    """
    Consulta el detalle de una orden para obtener información completa de ejecución.
    """
    if not send_request_fn:
        return None
    
    params = {
        "symbol": symbol,
        "productType": product_type
    }
    
    if order_id:
        params["orderId"] = order_id
    elif client_oid:
        params["clientOid"] = client_oid
    else:
        return None
    
    try:
        code, resp = send_request_fn("GET", "/api/v2/mix/order/detail", params=params)
        
        if code == 200 and resp.get("code") == "00000":
            return resp.get("data", {})
    except Exception as e:
        print(f"⚠️ Error consultando detalle de orden: {e}")
    
    return None
#==============================================================================================
#SUBACCOUNTS
#==============================================================================================
def df_to_arrays_live(df):
    if not is_datetime64_any_dtype(df.index):
        df = df.copy()
        df.index = pd.to_datetime(df.index)

    arrays = {
        'ts': df.index.to_numpy(dtype='datetime64[ns]'),
        'open': df['open'].to_numpy(dtype=np.float64),
        'high': df['high'].to_numpy(dtype=np.float64),
        'low': df['low'].to_numpy(dtype=np.float64),
        'close': df['close'].to_numpy(dtype=np.float64),
        'volume_quote': (
            df['volume_quote'].to_numpy(dtype=np.float64)
            if 'volume_quote' in df
            else np.zeros(len(df))
        )
    }

    return arrays

def load_final_symbols(all_symbols, strategy="_", timeframe="4H"):
    folder = os.path.join(os.path.dirname(__file__), "symbols_live")
    folder = os.path.abspath(folder)
    try:
        path_live = os.path.join(folder, f"symbols_live_{strategy}_{timeframe}.xlsx")
        df_live = pd.read_excel(path_live)
        live_symbols = set(df_live.iloc[:, 0].dropna().astype(str))
        final_symbols = set(all_symbols) & live_symbols

        print(f"🔹 Symbols for Live: {len(final_symbols)}")
        return sorted(final_symbols)

    except Exception as e:
        print(f"⚠️ Error loading symbols: {e}")
        return []

def wait_for_next_candle(timeframe='4H'):
    now = datetime.utcnow()
    
    if timeframe.endswith('H'):
        minutes = int(timeframe[:-1]) * 60
    elif timeframe.endswith('m'):
        minutes = int(timeframe[:-1])
    elif timeframe.endswith('Dutc'):
        minutes = int(timeframe[:-4]) * 24 * 60
    else:
        raise ValueError("Invalid timeframe, use 'm', 'H', or 'Dutc'.")

    total_minutes      = now.hour * 60 + now.minute
    next_total_minutes = ((total_minutes // minutes) + 1) * minutes
    delta_minutes      = next_total_minutes - total_minutes
    next_run           = now + timedelta(minutes=delta_minutes, seconds=-now.second, microseconds=-now.microsecond)
    
    sleep_seconds = (next_run - now).total_seconds() + 45
    print('\n')
    print(f"🔷 ===Waiting for next candle===: {now.strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print('\n')
    time.sleep(sleep_seconds)
    
# archivo utils_positions.py
def has_open_positions_on_exchange(get_open_fn, product_type: str):

    try:
        pos_list = get_open_fn(product_type=product_type.upper())
        return bool(pos_list)
    except Exception as e:
        print(f"⚠️ Error checking positions: {e}")
        return True

def fetch_ohlcv_data(symbols, timeframe):
    ohlcv_data = {}
    for sym in symbols:
        recent_minor = _call_history_candles(symbol=sym, granularity=timeframe, limit=180)
        df_minor = to_dataframe_from_api(recent_minor)
        ohlcv_data[sym] = df_minor
    return ohlcv_data


def process_signals_and_buy(
    final_symbols,
    exchange,
    open_positions,
    order_amount,
    timeframe_minor,
    sell_after_n_candles,
    tp_pct,
    sl_pct,
    direction,
    send_request_fn,
    get_balance_fn,
    check_signal_fn 
):

    ohlcv_data = fetch_ohlcv_data(final_symbols, timeframe_minor)
    ohlcv_data = {sym: {"minor": df} for sym, df in ohlcv_data.items()}

    # ----------------------------------------
    # 2️⃣ Detectar señales
    # ----------------------------------------
    detected_signals = []
    for sym, dfs in ohlcv_data.items():
        signal = check_signal_fn(dfs["minor"], sym)
        if signal:
            detected_signals.append(signal)

    print(f"\n✨ {datetime.now(MADRID_TZ).strftime('%H:%M')} - Signals detected: {len(detected_signals)}")

    # ----------------------------------------
    # 3️⃣ Ejecutar órdenes
    # ----------------------------------------
    for signal in detected_signals:
        sym = signal["symbol"]
        usdt_balance = get_balance_fn(exchange)
        now = datetime.now(MADRID_TZ).replace(second=0, microsecond=0) 

        if usdt_balance < order_amount:
            print(f"⚠️ {now} - USDT balance too low to place order for {sym}")
            continue

        order, tpsl_info = place_order(
            sym,
            direction=direction,
            usdt_amount=order_amount,
            tp_percent=tp_pct,
            sl_percent=sl_pct,
            send_request_func=send_request_fn
        )

        if order is not None:
            buy_price     = float(order['data'].get('price', signal['close']))
            filled_amount = float(order['data'].get('size', order_amount / buy_price))

            open_positions.append({
                'symbol': sym,
                'buy_price': buy_price,
                'amount': filled_amount,
                'candles_to_sell': sell_after_n_candles,
                'just_bought': True
            })

            usdt_after = get_balance_fn(exchange)
            print(f"💵 {now} - ORDER executed: {sym} | Remaining USDT: {usdt_after:.2f}\n")
            time.sleep(2)

        else:
            print(f"⚠️ {now} - Order for {sym} was not executed or returned None.")

    return open_positions

def manage_open_positions(open_positions, send_request_fn, product_type=PRODUCT_TYPE):

    for pos in open_positions[:]:
        if pos.get('just_bought', False):
            pos['just_bought'] = False
            continue

        pos['candles_to_sell'] -= 1

        if pos['candles_to_sell'] <= 0:
            try:
                body = {"symbol": pos['symbol'], "productType": product_type}
                code, resp = send_request_fn("POST", "/api/v2/mix/order/close-positions", body=body)
                now = datetime.now(MADRID_TZ).replace(second=0, microsecond=0)

                if code == 200 and resp.get("code") == "00000":
                    for success in resp['data']['successList']:
                        code_ticker, resp_ticker = send_request_fn("GET", "/api/v2/mix/market/ticker", params={"productType": product_type, "symbol": success['symbol']})
                        sell_price = None
                        if code_ticker == 200 and resp_ticker.get("code") == "00000":
                            sell_price = resp_ticker['data'][0]['lastPr']

                        print(f"💰 {now.strftime('%Y-%m-%d %H:%M:%S')} - FLASH CLOSE: {success['symbol']} | Sold at: {sell_price}")
                else:
                    print(f"⚠️ {now} - Failed Flash Close for {pos['symbol']}: {resp}")

            except Exception as e:
                now = datetime.now(MADRID_TZ).replace(second=0, microsecond=0)
                print(f"⚠️ {now} - Error closing position {pos['symbol']}: {e}")
            finally:
                try:
                    open_positions.remove(pos)
                except ValueError:
                    pass

            time.sleep(1.1)

