import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import json
import copy
import traceback
import logging
import builtins
import pandas as pd
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.console import Group
from BOT_metrics import bot_metrics
from collections import defaultdict
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from datetime import timedelta

STATE_FILE   = os.path.join(os.path.dirname(__file__), 'tracked_orders_state.json')
BASE_URL     = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"
MARGIN_MODE  = "crossed"
BLUE_BOLD    = "\033[1;94m"
RESET        = "\033[0m"

# ==========================================================================
# STATE MANAGEMENT
# ========================================================================== 

def load_state(state_file):
    OPEN_POSITIONS = {}
    STRATEGY_CANDLES = {}
    if not os.path.exists(state_file):
        print("📂 No previous state file found")
        return OPEN_POSITIONS, STRATEGY_CANDLES
    try:
        with open(state_file, 'r') as f:
            data = json.load(f)
        # Cargar contador de velas por estrategia
        STRATEGY_CANDLES = data.get('strategy_candles', {})
        # Convertir strings a Decimal donde sea necesario
        positions_data = data.get('positions', {})
        for strat_id, positions in positions_data.items():
            OPEN_POSITIONS[strat_id] = []
            for pos in positions:
                OPEN_POSITIONS[strat_id].append({
                    'symbol': pos.get('symbol'),
                    'size': Decimal(pos.get('size')),
                    'entry_price': Decimal(pos.get('entry_price')),
                    'direction': pos.get('direction'),
                    'tp': Decimal(pos.get('tp')),
                    'sl': Decimal(pos.get('sl')),
                    'order_id': pos.get('order_id'),
                    'opened_at': datetime.fromisoformat(pos.get('opened_at')),
                    'usdt_amount': float(pos.get('usdt_amount', 0)) 
                })
        total_positions = sum(len(p) for p in OPEN_POSITIONS.values())
        print(f"✅  State loaded: {total_positions} positions recovered")
        for strat_id, positions in OPEN_POSITIONS.items():
            if positions:
                candles = STRATEGY_CANDLES.get(strat_id, 0)
                print(f"   ➡️  {strat_id}: {len(positions)} positions | Candles: {candles}")
                for pos in positions:
                    size_str = f"{float(pos['size']):.6f}".rstrip('0').rstrip('.')
                    entry_str = f"{float(pos['entry_price']):.6f}".rstrip('0').rstrip('.')
                    print(f"      - {pos['symbol']:<12} | Size: {size_str:<10} | Entry: {entry_str:<10}")
        return OPEN_POSITIONS, STRATEGY_CANDLES
    except Exception as e:
        print(f"❌ Error loading state: {e}")
        traceback.print_exc()
        return OPEN_POSITIONS, STRATEGY_CANDLES


def save_state_local(open_positions, strategy_candles, state_file):
    try:
        positions_copy        = copy.deepcopy(open_positions)
        strategy_candles_copy = copy.deepcopy(strategy_candles)

        serializable_positions = {}
        for strat_id, positions in positions_copy.items():
            serializable_positions[strat_id] = []
            for pos in positions:
                serializable_positions[strat_id].append({
                    'symbol': pos['symbol'],
                    'size': str(pos['size']),
                    'entry_price': str(pos['entry_price']),
                    'direction': pos['direction'],
                    'tp': str(pos['tp']),
                    'sl': str(pos['sl']),
                    'order_id': pos['order_id'],
                    'opened_at': pos['opened_at'].isoformat(),
                    'usdt_amount': float(pos.get('usdt_amount', 0))  
                })

        state_data = {
            'positions': serializable_positions,
            'strategy_candles': strategy_candles_copy
        }

        with open(state_file, 'w') as f:
            json.dump(state_data, f, indent=2)
        #print(f"💾 Saving state...")
    except Exception as e:
        print(f"❌ Error saving state: {e}")
        import traceback
        traceback.print_exc()

def sync_broker(open_positions, strategy_candles, state_file, send_request_func):
    print("🌐 Syncronizing positions in broker...")
    total_removed = 0
    
    for strat_id, positions in list(open_positions.items()):
        positions_to_remove = []
        
        for i, pos in enumerate(positions):
            try:
                # Consultar si la posición existe en el broker
                code, resp = send_request_func("GET", "/api/v2/mix/position/single-position",
                    params={
                        "productType": "USDT-FUTURES",
                        "symbol": pos['symbol'],
                        "marginCoin": "USDT"
                    }
                )
                
                if code != 200 or resp.get("code") != "00000":
                    continue
                
                data = resp.get("data", [])
                
                # Si no existe la posición, marcarla para eliminar
                if not data or float(data[0].get('total', 0)) == 0:
                    print(f"→ Position {pos['symbol']} doesn't exist in broker - treating as SL")
                    
                    # ⭐ Usar el precio del SL en lugar del precio actual
                    sl_price = pos['sl']
                    
                    position_data = {
                        'opened_at': pos['opened_at'],
                        'strategy_id': strat_id,
                        'usdt_amount': pos.get('usdt_amount', 0),
                        'entry_price': pos['entry_price']
                    }
                    log_closed_position(
                        opened_at=position_data['opened_at'],
                        strategy_id=position_data['strategy_id'],
                        symbol=pos['symbol'],
                        direction=pos['direction'],
                        usdt_amount=position_data['usdt_amount'],
                        entry_price=position_data['entry_price'],
                        close_price=sl_price,  
                        reason="NOT_FOUND",  
                        size=pos['size'],
                        profit_from_api=None,
                        fee_from_api=None
                    )
                    
                    positions_to_remove.append(i)
                    total_removed += 1
                
                time.sleep(0.05)  # Rate limiting
                
            except Exception as e:
                print(f"❌ Error checking {pos['symbol']}: {e}")
        
        # Eliminar posiciones que no existen
        for i in reversed(positions_to_remove):
            if i < len(open_positions[strat_id]):
                open_positions[strat_id].pop(i)
        
        # Si no quedan posiciones, resetear contador de velas
        if not open_positions[strat_id]:
            if strategy_candles.get(strat_id, 0) > 0:
                strategy_candles[strat_id] = 0
    
    # Guardar estado solo si hubo cambios
    if total_removed > 0:
        save_state_local(open_positions, strategy_candles, state_file)
        print(f"✅ Sync completed: {total_removed} position(s) removed")
    else:
        print(f"✅ Sync completed: All positions exist in broker")
        
# ==========================================================================
# PLACE ORDER
# ==========================================================================   
def fetch_ticker(send_request_func, product_type, symbol):
    code, resp = send_request_func("GET", "/api/v2/mix/market/ticker",params={"productType": product_type, "symbol": symbol})
    if code != 200 or resp.get("code") != "00000":
        print("🔔 No fecth of ticker:", resp)
        return None, None
    last_price = Decimal(str(resp['data'][0]['lastPr']))
    time.sleep(0.05)
    return last_price, resp


def compute_size_base(usdt_amount, last_price):
    return Decimal(str(usdt_amount)) / last_price


def fetch_contracts(send_request_func, product_type, symbol):
    code_info, resp_info = send_request_func("GET", "/api/v2/mix/market/contracts",params={"productType": product_type, "symbol": symbol})
    if code_info == 200 and resp_info.get("code") == "00000":
        data_list = resp_info.get("data", [])
        if data_list:
            return data_list[0]
    return None


def extract_contract_params(c, last_price):
    """Extrae parámetros de configuración del contrato"""
    if c is None:
        return None, None, None, None, None
    
    try:
        # pricePlace siempre existe en la respuesta
        price_tick = Decimal(f"1e-{int(c['pricePlace'])}")
        
        # volumePlace siempre existe
        size_scale = int(c['volumePlace'])
        
        # Estos tres siempre existen como strings
        min_trade_num   = Decimal(c['minTradeNum'])
        size_multiplier = Decimal(c['sizeMultiplier'])
        min_trade_usdt  = Decimal(c['minTradeUSDT'])
        
        return price_tick, size_scale, min_trade_num, size_multiplier, min_trade_usdt
        
    except (KeyError, ValueError, TypeError) as e:
        # Solo si hay un error inesperado en los datos
        print(f"Error extrayendo parámetros del contrato: {e}")
        return None, None, None, None, None


def fallback_params(price_tick, size_scale, last_price, min_trade_num=None, min_trade_usdt=None):
    """Aplica valores por defecto si faltan parámetros"""
    if price_tick is None:
        if last_price >= 1000:
            price_tick = Decimal("0.1")
        elif last_price >= 1:
            price_tick = Decimal("0.01")
        elif last_price >= 0.1:
            price_tick = Decimal("0.001")
        else:
            price_tick = Decimal("0.00001")
    
    if size_scale is None or size_scale < 0:
        size_scale = 6
    
    if min_trade_num is None:
        if last_price >= 100:
            min_trade_num = Decimal("0.01")
        elif last_price >= 10:
            min_trade_num = Decimal("0.1")
        else:
            min_trade_num = Decimal("1")
    
    return price_tick, size_scale, min_trade_num, min_trade_usdt


def quantize_size(size_base, size_scale):
    precision_size = Decimal(f"1e-{size_scale}")
    size_q = size_base.quantize(precision_size, rounding=ROUND_DOWN)
    if size_q == 0:
        size_q = size_base.quantize(Decimal("1e-6"), rounding=ROUND_DOWN)
    if size_q == 0:
        print("🔔 Size = 0")
        return None, precision_size
    return size_q, precision_size


def build_order_body(symbol, product_type, margin_mode, margin_coin, size_q, side, client_oid):
    body = {
        "symbol": symbol,
        "productType": product_type,
        "marginMode": margin_mode,
        "marginCoin": margin_coin,
        "size": format(size_q, "f"),
        "side": side,
        "tradeSide": "open",
        "orderType": "market",
        "clientOid": client_oid if client_oid else f"script-{int(time.time())}"
    }
    return body


def place_market_order(send_request_func, body_order):
    code_order, resp_order = send_request_func("POST", "/api/v2/mix/order/place-order", body=body_order)
    if code_order != 200 or resp_order.get("code") != "00000":
        print("❌ Error order:", resp_order)
        return None, None
    return code_order, resp_order


def extract_filled_amount(resp_order, size_q):
    filled_amount = Decimal("0")
    data = resp_order.get("data") or {}
    for k in ("baseVolume", "filledQty", "size", "filledSize", "sz", "filled_amount"):
        if k in data and data[k] is not None:
            try:
                filled_amount = Decimal(str(data[k]))
                if filled_amount > 0:
                    break
            except Exception:
                continue
    if filled_amount == 0:
        filled_amount = size_q
    return filled_amount


def get_exec_price(resp_order, last_price):
    return Decimal(str(resp_order['data'].get('price', last_price)))


def place_order(symbol: str,
                direction: str,
                usdt_amount: float = 100,
                product_type: str = PRODUCT_TYPE,
                margin_coin: str = "USDT",
                margin_mode: str = MARGIN_MODE,
                send_request_func=None,
                client_oid: str = None):

    if send_request_func is None:
        raise ValueError("Send request error.")

    last_price, _ = fetch_ticker(send_request_func, product_type, symbol)
    if last_price is None:
        return None

    size_base = compute_size_base(usdt_amount, last_price)
    c         = fetch_contracts(send_request_func, product_type, symbol)
    price_tick, size_scale, min_trade_num, size_multiplier, min_trade_usdt = extract_contract_params(c, last_price)
    price_tick, size_scale, min_trade_num, min_trade_usdt = fallback_params(price_tick, size_scale, last_price, min_trade_num, min_trade_usdt)

    size_q, _ = quantize_size(size_base, size_scale)
    if size_q is None:
        return None

    side       = "buy" if direction.lower() == "long" else "sell"
    body_order = build_order_body(symbol, product_type, margin_mode, margin_coin, size_q, side, client_oid)

    code_order, resp_order = place_market_order(send_request_func, body_order)
    if code_order is None:
        print(f"🔔 Debug: last_price={last_price}, price_tick={price_tick},min_num: {min_trade_num}, min_usdt: {min_trade_usdt}")
        return None

    filled_amount = extract_filled_amount(resp_order, size_q)
    exec_price    = get_exec_price(resp_order, last_price)

    print(f"✅ {('⬆️ ' if direction=='long' else '⬇️ '):2} {direction.upper():<6} {symbol:<10} | Size: {filled_amount:<8} | Price: {exec_price:<10}")

    return resp_order


# ==========================================================================
# PRICING
# ==========================================================================   
def get_fills_for_order(order_id, symbol, product_type=PRODUCT_TYPE, send_request_func=None, retries=5, delay=0.05):
    
    time.sleep(delay)
    
    for attempt in range(retries):
        try:
            code, resp = send_request_func("GET","/api/v2/mix/order/fills",params={"productType": product_type, "orderId": order_id, "symbol": symbol})
            if code == 200 and resp.get("code") == "00000":
                data = resp.get("data") or {}
                fill_list = data.get("fillList") or []
                if fill_list:
                    total_base = Decimal('0')
                    weighted = Decimal('0')
                    total_profit = Decimal('0')
                    total_fee = Decimal('0')  # ⭐ NUEVO
                    
                    for f in fill_list:
                        bv = f.get("baseVolume")
                        price = f.get("price")
                        profit = f.get("profit")
                        fee_detail = f.get("feeDetail", [])  # ⭐ NUEVO
                        
                        if bv is None or price is None:
                            continue
                        
                        try:
                            bv_d = Decimal(str(bv))
                            p_d = Decimal(str(price))
                            total_base += bv_d
                            weighted += p_d * bv_d
                            
                            if profit is not None:
                                total_profit += Decimal(str(profit))
                            
                            # ⭐ SUMAR totalFee de cada elemento en feeDetail
                            for fee_item in fee_detail:
                                total_fee_val = fee_item.get("totalFee")
                                if total_fee_val is not None:
                                    total_fee += abs(Decimal(str(total_fee_val)))  # abs porque viene negativo
                                    
                        except Exception:
                            pass
                    
                    entry_price = (weighted / total_base) if total_base > 0 and weighted > 0 else None
                    return total_base, entry_price, total_profit, total_fee  # ⭐ RETORNAR FEE
                
        except Exception as e:
            print(f"🔔 Error consulting fills (attempt {attempt+1}): {e}")
        time.sleep(delay)
    return None, None, None, None  # ⭐ CUATRO VALORES

def get_current_price(symbol, send_request_func):
    """Obtiene el precio actual del mercado usando send_request_func"""
    try:
        code, resp = send_request_func("GET","/api/v2/mix/market/ticker",params={"productType": "USDT-FUTURES", "symbol": symbol})
        if code == 200 and resp.get("code") == "00000":
            return Decimal(str(resp['data'][0]['lastPr']))
    except Exception as e:
        print(f"🔔 No price of {symbol}: {e}")
    return None

def calculate_tp_sl_prices(entry_price, direction, tp_pct, sl_pct):
    """Calcula los precios de TP y SL basados en el precio de entrada"""
    entry = Decimal(str(entry_price))
    tp_decimal = Decimal(str(tp_pct)) / Decimal('100')
    sl_decimal = Decimal(str(sl_pct)) / Decimal('100')
    
    if direction.lower() == 'long':
        tp_price = entry * (Decimal('1') + tp_decimal)
        sl_price = entry * (Decimal('1') - sl_decimal)
    else:  # short
        tp_price = entry * (Decimal('1') - tp_decimal)
        sl_price = entry * (Decimal('1') + sl_decimal)
    
    return tp_price, sl_price
        
 # ==========================================================================
 # TIMEFRMES & CANDLES
 # ==========================================================================        

def calculate_next_candle_time(timeframe='4H', hour_zone=None):

    from datetime import datetime, timedelta
    
    now = datetime.now(hour_zone)
    
    # Detectar tipo de timeframe
    if timeframe.endswith('Hutc'):
        # Formato especial UTC: 6Hutc, 12Hutc
        hours = int(timeframe[:-4])
        minutes = hours * 60
    elif timeframe.endswith('H'):
        # Formato estándar de horas: 1H, 4H, etc.
        hours = int(timeframe[:-1])
        minutes = hours * 60
    elif timeframe.endswith('m'):
        # Formato de minutos: 15m, 30m, etc.
        minutes = int(timeframe[:-1])
    elif timeframe.endswith('Dutc'):
        # Formato de días UTC: 1Dutc
        days = int(timeframe[:-4])
        minutes = days * 24 * 60
    else:
        raise ValueError("Invalid timeframe. Use 'm', 'H', 'Hutc', or 'Dutc'. Examples: '15m', '4H', '6Hutc', '1Dutc'")
    
    # Calcular siguiente vela
    total_minutes = now.hour * 60 + now.minute
    next_total_minutes = ((total_minutes // minutes) + 1) * minutes
    delta_minutes = next_total_minutes - total_minutes
    next_candle = now + timedelta(minutes=delta_minutes, seconds=-now.second, microseconds=-now.microsecond)
    
    # Añadir 45 segundos de margen
    next_candle = next_candle + timedelta(seconds=45)
    
    return next_candle

def group_strategies_by_timeframe(strategies):
    """Agrupa estrategias por timeframe."""
    grouped = defaultdict(list)
    for strat in strategies:
        grouped[strat['timeframe']].append(strat)
    return grouped


def get_unique_timeframes(strategies):
    """Obtiene lista de timeframes únicos."""
    return list(set(s['timeframe'] for s in strategies))

def increment_strategy_candles(strat_id, strategy_candles, open_positions, state_file):
    if strat_id not in strategy_candles:
        strategy_candles[strat_id] = 0

    strategy_candles[strat_id] += 1
    save_state_local(open_positions, strategy_candles, state_file)
    
def reset_strategy_candles(strat_id, strategy_candles, open_positions, state_file):
    strategy_candles[strat_id] = 0
    save_state_local(open_positions, strategy_candles, state_file)

def check_candles_timeout_for_strategy(strat_id, sell_after_ncandles,
                                       open_positions, strategy_candles,
                                       state_file, send_request_func):
    """Cierra todas las posiciones de una estrategia si superan el límite de velas"""
    candles_elapsed = strategy_candles.get(strat_id, 0)
    if candles_elapsed < sell_after_ncandles:
        return

    if strat_id not in open_positions or not open_positions[strat_id]:
        return

    positions = open_positions[strat_id][:]

    if not positions:
        return

    print(f"\n⏱ TIMEOUT REACHED for strategy {strat_id}")
    print(f"➡ Candles ongoing         : {candles_elapsed}/{sell_after_ncandles}")
    print(f"→ Closing {len(positions)} positions...")

    all_closed = True
    for pos in positions:
        position_data = {
            'opened_at': pos['opened_at'],
            'strategy_id': strat_id,
            'usdt_amount': pos.get('usdt_amount', 0),
            'entry_price': pos['entry_price']
        }
        if not close_position(pos['symbol'], pos['size'], pos['direction'],
                              send_request_func, reason="TIMEOUT", position_data=position_data):
            all_closed = False

    if all_closed:
        open_positions[strat_id] = []
        strategy_candles[strat_id] = 0
        save_state_local(open_positions, strategy_candles, state_file)
        bot_metrics()
# ==========================================================================
# DISPLAY
# ==========================================================================
console       = Console()
_live_display = None
def format_price(price):
    """Formatea precios con decimales apropiados según su magnitud"""
    price_float = float(price)
    if price_float < 0.01:
        return f"{price_float:.6f}"
    elif price_float < 1:
        return f"{price_float:.4f}"
    elif price_float < 100:
        return f"{price_float:.2f}"
    else:
        return f"{price_float:.1f}"


def get_pnl_arrow(direction, entry_price, current_price):
    """Determina la flecha según si la posición está en profit o loss"""
    entry_float = float(entry_price)
    current_float = float(current_price)
    
    if direction.lower() == 'long':
        # Para LONG: profit si current > entry
        if current_float > entry_float:
            return "[bold green]↑[/bold green]"
        else:
            return "[bold red]↓[/bold red]"
    else:  # short
        # Para SHORT: profit si current < entry
        if current_float < entry_float:
            return "[bold green]↑[/bold green]"
        else:
            return "[bold red]↓[/bold red]"


def calculate_pnl(direction, entry_price, current_price, size):
    """Calcula el PnL en USDT de una posición"""
    entry_float = float(entry_price)
    current_float = float(current_price)
    size_float = float(size)
    
    if direction.lower() == 'long':
        pnl = (current_float - entry_float) * size_float
    else:  # short
        pnl = (entry_float - current_float) * size_float
    
    return pnl


def add_position_to_table(table, strat_id, pos, current_price, pnl_accumulator, strategy_candles, sell_after_ncandles):
    """Añade una fila de posición a la tabla de Rich"""
    direction = pos['direction']
    tp_price = pos['tp']
    sl_price = pos['sl']
    entry_price = pos['entry_price']
    symbol = pos['symbol']
    size = pos['size']
    
    # Calcular distancias al TP y SL
    if direction.lower() == 'short':
        dist_to_tp = float(current_price - tp_price)
        dist_to_sl = float(sl_price - current_price)
        tp_pct_away = (dist_to_tp / float(entry_price)) * 100
        sl_pct_away = (dist_to_sl / float(entry_price)) * 100
    else:  # long
        dist_to_tp = float(tp_price - current_price)
        dist_to_sl = float(current_price - sl_price)
        tp_pct_away = (dist_to_tp / float(entry_price)) * 100
        sl_pct_away = (dist_to_sl / float(entry_price)) * 100
    
    direction_style = "white"
    pnl_arrow = get_pnl_arrow(direction, entry_price, current_price)
    
    # Calcular PnL numérico
    pnl = calculate_pnl(direction, entry_price, current_price, size)
    pnl_accumulator['total'] += pnl
    
    # Formatear PnL con color
    pnl_color = "green" if pnl >= 0 else "red"
    pnl_text = f"[{pnl_color}]{pnl:+.2f}[/{pnl_color}]"
    
    # Extraer opened_at y formatear solo la fecha
    opened_at = pos.get('opened_at', '')
    if opened_at:
        # Si es datetime, convertir a string con solo fecha
        if hasattr(opened_at, 'strftime'):
            opened_at_str = opened_at.strftime('%Y-%m-%d')
        # Si es string, extraer solo YYYY-MM-DD
        elif isinstance(opened_at, str):
            opened_at_str = opened_at.split('T')[0] if 'T' in opened_at else opened_at[:10]
        else:
            opened_at_str = str(opened_at)[:10]
    else:
        opened_at_str = '-'
    
    # Obtener candles elapsed y sell_after_ncandles
    candles_elapsed = strategy_candles.get(strat_id, 0)
    candles_str = f"{candles_elapsed}/{sell_after_ncandles}" if sell_after_ncandles else f"{candles_elapsed}"
    
    # Formatear TP con color condicional
    tp_color = "bold green" if tp_pct_away < 1 else "cyan"
    tp_text = f"[white]{format_price(tp_price)}[/white] [{tp_color}](Δ {tp_pct_away:+.2f}%)[/{tp_color}]"
    
    # Formatear SL con color condicional
    sl_color = "bold red" if sl_pct_away < 1 else "magenta"
    sl_text = f"[white]{format_price(sl_price)}[/white] [{sl_color}](Δ {sl_pct_away:+.2f}%)[/{sl_color}]"
    
    # Formatear size
    size_str = f"{float(size):.6f}".rstrip('0').rstrip('.')
    
    table.add_row(
        strat_id,
        f"[{direction_style}]{symbol}[/{direction_style}]",
        f"[{direction_style}]{direction.upper()}[/{direction_style}]",
        f"[white]{opened_at_str}[/white]",
        f"[white]{candles_str}[/white]",
        f"{format_price(entry_price)}",
        f"[white]{size_str}[/white]",  # Nueva columna SIZE
        f"[yellow]{format_price(current_price)}[/yellow]",
        pnl_arrow,
        pnl_text,
        tp_text,
        sl_text
    )


def create_tp_sl_display(now, total_pnl=None):
    """Crea el header y la tabla para el display de TP/SL"""
    # Crear el header con PnL total si se proporciona
    header = Text()
    header.append(f"{BLUE_BOLD}{'─'*115}\n")
    header.append(f"{BLUE_BOLD}🔷 Checking TP/SL - {now}\n")
    if total_pnl is not None:
        pnl_color = "bold green" if total_pnl >= 0 else "bold red"
        header.append(f"💰 Total PnL: ", style="white")
        header.append(f"{total_pnl:+.2f} USDT\n", style=pnl_color)
    header.append(f"{BLUE_BOLD}{'─'*115}\n")
    
    # Crear tabla con columnas adicionales: opened_at y candles
    table = Table(show_header=True, header_style="bold white", border_style="white")
    table.add_column("Strategy", style="white", width=15)
    table.add_column("Symbol", style="bold", width=11)
    table.add_column("Side", justify="center", width=5)
    table.add_column("Opened", style="white", width=10)
    table.add_column("Candles", justify="center", width=8)
    table.add_column("Entry", justify="right", width=8)
    table.add_column("Size", justify="right", width=7)  # Nueva columna
    table.add_column("Current", justify="right", width=8)
    table.add_column("↕", justify="center", width=1)
    table.add_column("PnL (USDT)", justify="right", width=6)
    table.add_column("TP", justify="right", width=20)
    table.add_column("SL", justify="right", width=20)
    
    return header, table


# ==========================================================================
# TP/SL CHECKINGS
# ========================================================================== 
def check_tp_sl_for_strategy(strat_id, strat_config, open_positions, strategy_candles, state_file, send_request_func, table=None, pnl_accumulator=None):
    """Comprueba TP/SL para todas las posiciones de una estrategia"""
    if strat_id not in open_positions or not open_positions[strat_id]:
        return
    
    positions = open_positions[strat_id][:]
    positions_to_remove = []
    
    # Obtener sell_after_ncandles de la configuración de estrategia
    sell_after_ncandles = strat_config.get('sell_after_ncandles') if strat_config else None
    
    for i, pos in enumerate(positions):
        symbol = pos['symbol']
        current_price = get_current_price(symbol, send_request_func=send_request_func)
        
        if current_price is None:
            print(f"🔔 No price for {symbol}")
            continue
        
        direction = pos['direction']
        tp_price = pos['tp']
        sl_price = pos['sl']
        entry_price = pos['entry_price']
        
        current_price = Decimal(str(current_price))
        
        # Añadir fila a la tabla si se proporciona
        if table and pnl_accumulator is not None:
            add_position_to_table(table, strat_id, pos, current_price, pnl_accumulator, strategy_candles, sell_after_ncandles)
        
        # Verificar si se alcanzó TP o SL
        hit_tp = current_price >= tp_price if direction.lower() == 'long' else current_price <= tp_price
        hit_sl = current_price <= sl_price if direction.lower() == 'long' else current_price >= sl_price
        
        if hit_tp:
            position_data = {
                'opened_at': pos['opened_at'],
                'strategy_id': strat_id,
                'usdt_amount': pos.get('usdt_amount', 0),
                'entry_price': pos['entry_price']
            }
            if close_position(symbol, pos['size'], direction, send_request_func, reason="TP", position_data=position_data):
                positions_to_remove.append(i)
                
        elif hit_sl:
            position_data = {
                'opened_at': pos['opened_at'],
                'strategy_id': strat_id,
                'usdt_amount': pos.get('usdt_amount', 0),
                'entry_price': pos['entry_price']
            }
            if close_position(symbol, pos['size'], direction, send_request_func, reason="SL", position_data=position_data):
                positions_to_remove.append(i)
    
    # Eliminar posiciones cerradas
    if positions_to_remove:
        for i in reversed(positions_to_remove):
            if i < len(open_positions[strat_id]):
                open_positions[strat_id].pop(i)
                
        save_state_local(open_positions, strategy_candles, state_file)


def check_all_tp_sl(strategies, open_positions, strategy_candles, state_file, send_request_func, hour_zone):
    """Chequea TP/SL para todas las estrategias"""
    global _live_display
    
    now = datetime.now(hour_zone).strftime('%Y-%m-%d %H:%M:%S')
    
    # Acumulador para el PnL total
    pnl_accumulator = {'total': 0.0}
    
    # Crear header y tabla (sin total aún)
    header, table = create_tp_sl_display(now)
    
    # Crear un diccionario de estrategias por ID para acceso rápido
    strat_dict = {strat['id']: strat for strat in strategies}
    
    # Llenar la tabla con todas las estrategias
    for idx, strat in enumerate(strategies):
        strat_id = strat['id']
        num_positions = len(open_positions.get(strat_id, []))
        
        if num_positions > 0:
            check_tp_sl_for_strategy(strat_id, strat, open_positions, strategy_candles, state_file, send_request_func, table, pnl_accumulator)
            
            # Añadir fila vacía entre estrategias (excepto la última)
            if idx < len(strategies) - 1:
                next_has_positions = any(
                    len(open_positions.get(strategies[next_idx]['id'], [])) > 0 
                    for next_idx in range(idx + 1, len(strategies))
                )
                if next_has_positions:
                    table.add_row("", "", "", "", "", "", "", "", "", "", "")
    
    # Recrear header con el total PnL calculado
    header, _ = create_tp_sl_display(now, pnl_accumulator['total'])
    
    # Combinar header + tabla
    display = Group(header, table)
    
    # Inicializar o actualizar Live display
    if _live_display is None:
        _live_display = Live(display, console=console, refresh_per_second=4)
        _live_display.start()
    else:
        _live_display.update(display)
        
# ==========================================================================
# STRATEGY MANAGMENT
# ========================================================================== 
def process_strategy(
    strat,
    final_symbols,
    exchange,
    open_positions,
    strategy_candles,
    state_file,
    send_request_func,
    get_balance_func,
    hour_zone,
    use_hardcoded=False,
    detect_signal_func=None
):
    """Procesa una estrategia: busca señales y abre posiciones.
    Usa las utilidades locales del módulo (place_order, get_fills_for_order, add_position, reset_strategy_candles, etc.).
    Recibe las estructuras y funciones externas como parámetros para poder importarla.
    """
    strat_id = strat['id']

    print(f"\n🔄 Processing strategy: {strat_id}")

    # Detectar señales
    if use_hardcoded:
        signals = get_hardcoded_signals(strat_id, send_request_func, hour_zone)
    else:
        if detect_signal_func is None:
            raise ValueError("No se proporcionó detect_signal_func")
        signals = detect_signal_func(strat, final_symbols)

    print(f"✨ Signals detected for {strat_id}: {len(signals)}")

    if not signals:
        return

    # Resetear contador de velas al abrir nuevas posiciones
    reset_strategy_candles(strat_id, strategy_candles, open_positions, state_file)

    # Procesar cada señal
    for sig in signals:
        usdt_balance = get_balance_func(exchange)
        if usdt_balance < strat['order_amount']:
            print(f"🔔 Insufficient balance ({usdt_balance:.2f} USDT) for {sig['symbol']}")
            continue

        resp_order = place_order(
            symbol=sig['symbol'],
            direction=strat['direction'],
            usdt_amount=strat['order_amount'],
            send_request_func=send_request_func
        )

        if resp_order is None:
            print(f"❌ Error placing order for {sig['symbol']}")
            continue

        data = resp_order.get('data', {}) if isinstance(resp_order, dict) else {}
        order_id = data.get('orderId')

        if order_id:
            filled_size, entry_price_from_fills, _, _ = get_fills_for_order(
                    order_id=order_id,
                    symbol=sig['symbol'],
                    send_request_func=send_request_func
                )
            time.sleep(0.05)

            if filled_size is None or filled_size == 0:
                size = Decimal(str(data.get('size', data.get('filledQty', data.get('baseVolume', 0)))))
                entry_price = Decimal(str(data.get('price', sig.get('close', 0))))
            else:
                size = filled_size
                entry_price = entry_price_from_fills if entry_price_from_fills is not None else Decimal(str(sig.get('close', 0)))

            #print(f"➡️ Orden ejecutada - ID: {order_id}")

            # Registrar posición usando la función local add_position (que espera open_positions, strategy_candles, state_file, hour_zone)
            add_position(strat_id=strat_id, symbol=sig['symbol'], size=size, entry_price=entry_price, direction=strat['direction'], tp_pct=strat['tp_pct'], sl_pct=strat['sl_pct'], order_id=order_id, open_positions=open_positions, strategy_candles=strategy_candles, state_file=state_file, hour_zone=hour_zone, usdt_amount=strat['order_amount'])

        else:
            print(f"🔔 Order executed but no orderId in response")

        time.sleep(0.05)
        
def get_hardcoded_signals(strat_id, send_request_func, hour_zone):
    """Genera señales de prueba para testing"""
    symbols = ['BTCUSDT', 'BNBUSDT']
    signals = []
    for symbol in symbols:
        code, resp = send_request_func("GET","/api/v2/mix/market/ticker",params={"productType": PRODUCT_TYPE, "symbol": symbol})
        current_price = 50000.0
        if code == 200 and isinstance(resp, dict) and resp.get("code") == "00000":
            try:
                current_price = float(resp['data'][0]['lastPr'])
            except Exception:
                pass
        signals.append({
            'symbol': symbol,
            'close': current_price,
            'timestamp': datetime.now(hour_zone).isoformat()
        })
    return signals

# ==========================================================================
# POSITIONS MANAGEMENT
# ========================================================================== 
def close_position(symbol, size, direction, send_request_func, reason="NO_INFO", position_data=None):
    """Cierra una posición con orden market en HEDGE MODE"""
    try:
        close_side = "sell" if direction.lower() == "short" else "buy"
        
        body = {
            "symbol": symbol,
            "productType": PRODUCT_TYPE,
            "marginMode": "isolated",
            "marginCoin": "USDT",
            "size": format(size, "f"),
            "side": close_side,
            "tradeSide": "close",
            "orderType": "market"
        }
        
        # Mensajes ANTES de cerrar
        if reason == "TP":
            print(f"\n💲 TP REACHED for {symbol} ({position_data.get('strategy_id', 'N/A') if position_data else 'N/A'}) at {datetime.now().strftime('%H:%M')}")
        elif reason == "SL":
            print(f"\n🔻 SL REACHED for {symbol} ({position_data.get('strategy_id', 'N/A') if position_data else 'N/A'}) at {datetime.now().strftime('%H:%M')}")

        print(f"→  Closing {direction} position on {symbol}:")   
        code, resp = send_request_func("POST", "/api/v2/mix/order/place-order", body=body)
        time.sleep(0.05)
        
        if code == 200 and resp.get("code") == "00000":
            print(f"✅ Position closed due to {reason}: {symbol} | Size: {size}")
            
            # ⭐ OBTENER PRECIO REAL DE EJECUCIÓN
            if position_data:
                data = resp.get('data', {})
                order_id = data.get('orderId')
                
                # Obtener precio real desde fills (igual que al abrir)
                if order_id:
                    _, close_price_from_fills, profit_from_api, fee_from_api = get_fills_for_order(
                        order_id=order_id,
                        symbol=symbol,
                        send_request_func=send_request_func
                    )
                    
                    # Si no hay fills, usar el precio del response o ticker como fallback
                    if close_price_from_fills is None:
                        close_price_from_fills = Decimal(str(data.get('price', 0)))
                        if close_price_from_fills == 0:
                            close_price_from_fills = get_current_price(symbol, send_request_func)
                    
                    if close_price_from_fills:
                        log_closed_position(
                            opened_at=position_data.get('opened_at'), 
                            strategy_id=position_data.get('strategy_id'), 
                            symbol=symbol, 
                            direction=direction, 
                            usdt_amount=position_data.get('usdt_amount', 0), 
                            entry_price=position_data.get('entry_price'), 
                            close_price=close_price_from_fills,     
                            reason=reason,
                            size=size,
                            profit_from_api=profit_from_api,
                            fee_from_api=fee_from_api
                        )
                        
            bot_metrics()
            return True
        else:
            print(f"🔔 No closing position available {symbol}: {resp}")
            if resp.get("code") == "22002":
                print(f"   → Removing from local record (nonexistent position)")
                if position_data:
                    current_price = get_current_price(symbol, send_request_func)
                    if current_price:
                        log_closed_position(
                            opened_at=position_data.get('opened_at'), 
                            strategy_id=position_data.get('strategy_id'), 
                            symbol=symbol, 
                            direction=direction, 
                            usdt_amount=position_data.get('usdt_amount', 0), 
                            entry_price=position_data.get('entry_price'), 
                            close_price=current_price, 
                            reason="OUT_OF_MARGIN",
                            size=size,
                            profit_from_api=None,
                            fee_from_api=None
                        )
                return True
            return False
            
    except Exception as e:
        print(f"❌ Error closing position {symbol}: {e}")
        import traceback
        traceback.print_exc()
        return False
    
def add_position(strat_id, symbol, size, entry_price, direction, tp_pct, sl_pct, order_id,
                 open_positions, strategy_candles, state_file,hour_zone,usdt_amount=0):
    """Registra una nueva posición abierta en open_positions y guarda estado"""
    
    if strat_id not in open_positions:
        open_positions[strat_id] = []

    tp_price, sl_price = calculate_tp_sl_prices(entry_price, direction, tp_pct, sl_pct)
    
    position = {
        'symbol': symbol,
        'size': size,
        'entry_price': entry_price,
        'direction': direction,
        'tp': tp_price,
        'sl': sl_price,
        'order_id': order_id,
        'opened_at': datetime.now(hour_zone),
        'usdt_amount': usdt_amount
    }
    
    open_positions[strat_id].append(position)
    
# =============================================================================
#     print(f"➡️ Registered position:")
#     print(f"  Symbol: {symbol} | Size: {size} | Entry: {entry_price}")
#     print(f"  TP: {tp_price} | SL: {sl_price}")
# =============================================================================
    
    # Guardar estado actualizado
    save_state_local(open_positions, strategy_candles, state_file)
    

def log_closed_position(
    opened_at,
    strategy_id,
    symbol,
    direction,
    usdt_amount,
    entry_price,
    close_price,
    reason,
    size,
    profit_from_api=None,
    fee_from_api=None,
    excel_file='bot_trading_trades.xlsx'
):
    try:
        # Carpeta 'files' en el mismo directorio que excel_file
        base_dir = os.path.dirname(os.path.abspath(excel_file))
        files_dir = os.path.join(base_dir, 'bot_files')
        os.makedirs(files_dir, exist_ok=True)  # crea la carpeta si no existe

        # Excel final en la carpeta 'files'
        excel_file_path = os.path.join(files_dir, os.path.basename(excel_file))

        # Convertir precios y montos a float
        entry_price = float(entry_price)
        close_price = float(close_price)
        usdt_amount = float(usdt_amount)

        # Intentar convertir size a float si viene
        size_val = None
        if size is not None:
            try:
                size_val = float(size)
            except Exception:
                size_val = None

        # Si no hay usdt_amount pero sí size, calcular usdt_amount = size * entry_price (valor de la posición)
        if usdt_amount == 0 and size_val is not None:
            usdt_amount = size_val * entry_price

        # ⭐ PRIORIZAR PROFIT DEL API
        if profit_from_api is not None:
            profit_gross = float(profit_from_api)
            fee = float(fee_from_api) if fee_from_api is not None else 0
            fee = 2*fee
            profit = profit_gross - fee  # ⭐ PROFIT NETO
            
            # Calcular profit_pct basado en el profit neto
            if usdt_amount > 0:
                profit_pct = (profit / usdt_amount) * 100
            else:
                profit_pct = 0
        else:
            # Fallback al cálculo manual
            if size_val is not None:
                if direction.lower() == 'long':
                    profit = (close_price - entry_price) * size_val
                    profit_pct = ((close_price - entry_price) / entry_price) * 100
                else:
                    profit = (entry_price - close_price) * size_val
                    profit_pct = ((entry_price - close_price) / entry_price) * 100
            else:
                if direction.lower() == 'long':
                    profit = (close_price - entry_price) * (usdt_amount / entry_price)
                    profit_pct = ((close_price - entry_price) / entry_price) * 100
                else:
                    profit = (entry_price - close_price) * (usdt_amount / entry_price)
                    profit_pct = ((entry_price - close_price) / entry_price) * 100


        closed_at = datetime.now()

        if isinstance(opened_at, str):
            opened_at_dt = datetime.strptime(opened_at, '%Y-%m-%d %H:%M:%S')
        else:
            opened_at_dt = opened_at

        if opened_at_dt.tzinfo is not None:
            opened_at_dt = opened_at_dt.replace(tzinfo=None)
        if closed_at.tzinfo is not None:
            closed_at = closed_at.replace(tzinfo=None)

        delta_days = (closed_at - opened_at_dt).total_seconds() / (3600*24)

        new_record = {
            'OPEN_AT': opened_at_dt.strftime('%Y-%m-%d %H:%M:%S'),
            'CLOSE_AT': closed_at.strftime('%Y-%m-%d %H:%M:%S'),
            'DURATION_DAYS': round(delta_days, 4),
            'STRATEGY': strategy_id,
            'SYMBOL': symbol,
            'DIRECTION': direction.upper(),
            'USDT_AMOUNT': round(usdt_amount, 2),
            'SIZE': round(size_val, 6),
            'PRICE_ENTRY': round(entry_price, 6),
            'PRICE_CLOSE': round(close_price, 6),
            'PROFIT': round(profit, 2),
            'FEE': round(fee, 4) if profit_from_api is not None else 0,
            'PROFIT_PCT': round(profit_pct, 1),
            'REASON_OUT': reason
        }

        # Cargar o crear DataFrame
        if os.path.exists(excel_file_path):
            df = pd.read_excel(excel_file_path)
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        else:
            df = pd.DataFrame([new_record])

        df.to_excel(excel_file_path, index=False, engine='openpyxl')

        print(f"📋 Trade logged: {symbol} | Profit: {profit:.2f} USDT ({profit_pct:+.2f}%) | Duration: {delta_days:.4f} days")

    except Exception as e:
        print(f"❌ Error logging trade to Excel: {e}")
        import traceback
        traceback.print_exc()

    
def setup_print_logger(logdir, logfile_name="BOT_all_stratagies.log"):
    """
    Configura un logger que duplica print() al archivo y a consola.
    Solo muestra el print normal en consola, sin prefijos.
    """
    
    # Crear carpeta
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, logfile_name)
    
    logger = logging.getLogger('bot_logger')
    logger.setLevel(logging.INFO)
    logger.propagate = False  # ⭐ IMPORTANTE: Evita que propague a otros handlers
    
    fh = logging.FileHandler(logfile, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    
    # Evitar múltiples handlers si se llama varias veces
    if not logger.handlers:
        logger.addHandler(fh)
    
    # Reemplazar print
    old_print = builtins.print
    
    def _print_and_log(*args, **kwargs):
        # ⭐ PRIMERO: Imprimir SOLO en consola (sin logger)
        old_print(*args, **kwargs)
        
        # ⭐ DESPUÉS: Escribir solo en el archivo (sin mostrar en consola)
        text = kwargs.get("sep", " ").join(str(a) for a in args)
        logger.info(text)
    
    builtins.print = _print_and_log