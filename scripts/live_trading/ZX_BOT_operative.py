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
from rich.console import Group
from ZX_BOT_metrics import bot_metrics
from collections import defaultdict
from datetime import datetime
from decimal import Decimal, ROUND_DOWN
from ZX_BOT_display import add_position_to_table

# Import WebSocket manager
import ZX_BOT_ws_manager

BASE_URL     = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"
MARGIN_MODE  = "crossed"
BLUE_BOLD    = "\033[1;94m"
RESET        = "\033[0m"

# ⭐ Configuración de archivo Excel para trades
TRADES_LOG_DIR  = os.path.expanduser('~/projects/quant/quant_g/scripts/live_trading/bot_files')
TRADES_LOG_FILE = 'bot_trading_trades.xlsx'
TRADES_LOG_PATH = os.path.join(TRADES_LOG_DIR, TRADES_LOG_FILE)

# Crear directorio si no existe
os.makedirs(TRADES_LOG_DIR, exist_ok=True)


# ==========================================================================
# WEBSOCKET-BASED FUNCTIONS 
# ==========================================================================

def fetch_ticker_ws(symbol):
    """Obtiene ticker SOLO via WebSocket"""
    if not ZX_BOT_ws_manager._ws_manager:
        raise RuntimeError("WebSocket not initialized")
    
    # Suscribir si no está suscrito
    if symbol not in ZX_BOT_ws_manager._ws_manager.subscribed_public:
        ZX_BOT_ws_manager._ws_manager.subscribe_ticker(symbol)
        time.sleep(0.05)
    
    # Obtener del caché
    price_data = ZX_BOT_ws_manager._ws_manager.prices.get(symbol)
    
    if price_data:
        age = time.time() - price_data['timestamp']
        if age < 5.0:
            return price_data['price'], None
    
    # Esperar dato fresco
    initial_ts = price_data['timestamp'] if price_data else 0
    timeout_start = time.time()
    
    while (time.time() - timeout_start) < 2.0:
        price_data = ZX_BOT_ws_manager._ws_manager.prices.get(symbol)
        if price_data and price_data['timestamp'] > initial_ts:
            return price_data['price'], None
        time.sleep(0.02)
    
    raise TimeoutError(f"No WebSocket data for {symbol}")

def fetch_contracts_ws(symbol):
    """Obtiene contratos SOLO desde caché WebSocket"""
    if not ZX_BOT_ws_manager._ws_manager:
        raise RuntimeError("WebSocket not initialized")
    
    contract = ZX_BOT_ws_manager._ws_manager.get_contract(symbol)
    if contract:
        return contract
    
    raise ValueError(f"Contract for {symbol} not in cache")

def get_usdt_balance_ws(exchange=None):
    """
    Obtiene el balance USDT desde WebSocket (canal equity).
    Parámetro exchange ignorado (compatibilidad).
    """
    if not ZX_BOT_ws_manager._ws_manager:
        print("❌ WebSocket manager not initialized for balance check")
        return 0.0
    
    balance = ZX_BOT_ws_manager._ws_manager.get_usdt_balance()
    
    # Si no hay datos de equity todavía, esperar un poco
    if balance == 0.0 and not ZX_BOT_ws_manager._ws_manager.equity:
        print("⏳ Waiting for equity data from WebSocket...")
        time.sleep(0.2)
        balance = ZX_BOT_ws_manager._ws_manager.get_usdt_balance()
    
    return balance

# ==========================================================================
# STATE MANAGEMENT
# ==========================================================================
def load_state(state_file):
    OPEN_POSITIONS   = {}
    STRATEGY_CANDLES = {}
    if not os.path.exists(state_file):
        print("📂 No previous state file found")
        return OPEN_POSITIONS, STRATEGY_CANDLES
    try:
        with open(state_file, 'r') as f:
            data = json.load(f)
        STRATEGY_CANDLES = data.get('strategy_candles', {})
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
        positions_copy = copy.deepcopy(open_positions)
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
    except Exception as e:
        print(f"❌ Error saving state: {e}")
        traceback.print_exc()

def sync_broker(open_positions, strategy_candles, state_file):
    """
    Sincroniza posiciones locales con el broker via WebSocket (canal positions).
    Refresca datos para asegurar que estén actualizados.
    """
    print("🌐 Syncronizing positions in broker...")
    total_removed = 0
    
    if not ZX_BOT_ws_manager._ws_manager:
        raise RuntimeError("WebSocket manager not initialized")
    
    # ⭐ REFRESCAR datos de posiciones del WebSocket
    ZX_BOT_ws_manager._ws_manager.refresh_positions()
    
    for strat_id, positions in list(open_positions.items()):
        positions_to_remove = []
        
        for i, pos in enumerate(positions):
            try:
                symbol = pos['symbol']
                
                # Obtener posición desde WebSocket (ya refrescado)
                ws_position = ZX_BOT_ws_manager._ws_manager.get_position(symbol)
                
                # Verificar si la posición existe
                position_exists = False
                if ws_position:
                    total_size = float(ws_position.get('total', 0))
                    position_exists = (total_size > 0)
                    
                    # Debug: mostrar info de la posición
                    if not position_exists:
                        print(f"   📊 {symbol}: total={total_size} (position closed)")
                
                # Si no existe en WS data, tampoco existe
                if not position_exists:
                    print(f"→ Position {symbol} doesn't exist in broker - treating as SL")
                    
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
                        symbol=symbol,
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
                
            except Exception as e:
                print(f"❌ Error checking {pos['symbol']}: {e}")
                import traceback
                traceback.print_exc()
        
        # Eliminar posiciones que no existen
        for i in reversed(positions_to_remove):
            if i < len(open_positions[strat_id]):
                open_positions[strat_id].pop(i)
        
        # Resetear contador de velas si no quedan posiciones
        if not open_positions[strat_id]:
            if strategy_candles.get(strat_id, 0) > 0:
                strategy_candles[strat_id] = 0
    
    # Guardar estado si hubo cambios
    if total_removed > 0:
        save_state_local(open_positions, strategy_candles, state_file)
        print(f"✅ Sync completed: {total_removed} position(s) removed")
    else:
        print(f"✅ Sync completed: All positions exist in broker")
    
# ==========================================================================
# PLACE ORDER
# ==========================================================================
def fetch_ticker(symbol):
    """Obtiene el ticker del símbolo via WebSocket"""
    return fetch_ticker_ws(symbol)

def compute_size_base(usdt_amount, last_price):
    return Decimal(str(usdt_amount)) / last_price

def fetch_contracts(symbol):
    """Obtiene información del contrato desde caché WebSocket"""
    return fetch_contracts_ws(symbol)

def extract_contract_params(c, last_price):
    """Extrae parámetros de configuración del contrato"""
    if c is None:
        return None, None, None, None, None
    
    try:
        price_tick = Decimal(f"1e-{int(c['pricePlace'])}")
        size_scale = int(c['volumePlace'])
        min_trade_num = Decimal(c['minTradeNum'])
        size_multiplier = Decimal(c['sizeMultiplier'])
        min_trade_usdt = Decimal(c['minTradeUSDT'])
        
        return price_tick, size_scale, min_trade_num, size_multiplier, min_trade_usdt
        
    except (KeyError, ValueError, TypeError) as e:
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
        print("⚠️ Size = 0")
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
        "clientOid": client_oid if client_oid else f"script-{int(time.time() * 1000000)}"
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
    """
    Coloca una orden via REST API
    """
    if send_request_func is None:
        raise ValueError("Send request error.")

    # Obtener precio actual via WebSocket
    last_price, _ = fetch_ticker(symbol)
    if last_price is None:
        return None

    # Calcular tamaño via WebSocket
    size_base = compute_size_base(usdt_amount, last_price)
    c = fetch_contracts(symbol)
    price_tick, size_scale, min_trade_num, size_multiplier, min_trade_usdt = extract_contract_params(c, last_price)
    price_tick, size_scale, min_trade_num, min_trade_usdt = fallback_params(price_tick, size_scale, last_price, min_trade_num, min_trade_usdt)

    size_q, _ = quantize_size(size_base, size_scale)
    if size_q is None:
        return None

    side = "buy" if direction.lower() == "long" else "sell"
    
    # Usar REST API
    body_order = build_order_body(symbol, product_type, margin_mode, margin_coin, size_q, side, client_oid)
    code_order, resp_order = place_market_order(send_request_func, body_order)
    if code_order is None:
        print(f"⚠️ Debug: last_price={last_price}, price_tick={price_tick}, min_num: {min_trade_num}, min_usdt: {min_trade_usdt}")
        return None

    filled_amount = extract_filled_amount(resp_order, size_q)
    exec_price    = get_exec_price(resp_order, last_price)

    print(f"✅ {('⬆️ ' if direction=='long' else '⬇️ '):2} {direction.upper():<6} {symbol:<10} | Size: {filled_amount:<8} | Price: {exec_price:<10}")

    return resp_order

# ==========================================================================
# PRICING
# ==========================================================================
def get_fills_for_order(order_id, symbol, product_type=PRODUCT_TYPE, send_request_func=None, retries=5, delay=0.05):
    """
    Obtiene fills via WebSocket.
    """
    time.sleep(delay)
    
    if not ZX_BOT_ws_manager._ws_manager:
        raise RuntimeError("WebSocket manager not initialized")
    
    # Esperar a recibir fills via WebSocket
    start_time = time.time()
    timeout = 1.0
    
    while time.time() - start_time < timeout:
        fills = ZX_BOT_ws_manager._ws_manager.get_fills(order_id)
        if fills:
            # Procesar fills
            total_base   = Decimal('0')
            weighted     = Decimal('0')
            total_profit = Decimal('0')
            total_fee    = Decimal('0')
            
            for f in fills:
                bv         = f.get("baseVolume")
                price      = f.get("price")
                profit     = f.get("profit")
                fee_detail = f.get("feeDetail", [])
                
                if bv is None or price is None:
                    continue
                
                try:
                    bv_d = Decimal(str(bv))
                    p_d = Decimal(str(price))
                    total_base += bv_d
                    weighted += p_d * bv_d
                    
                    if profit is not None:
                        total_profit += Decimal(str(profit))
                    
                    for fee_item in fee_detail:
                        total_fee_val = fee_item.get("totalFee")
                        if total_fee_val is not None:
                            total_fee += abs(Decimal(str(total_fee_val)))
                except Exception:
                    pass
            
            entry_price = (weighted / total_base) if total_base > 0 and weighted > 0 else None
                        
            return total_base, entry_price, total_profit, total_fee
        
        time.sleep(0.05)
    
    # Timeout - retornar None
    print(f"⚠️  No fills received for order {order_id} via WebSocket (timeout)")
    return None, None, None, None

def get_current_price(symbol, max_cache_age=0.5):
    """Obtiene el precio actual del mercado via WebSocket"""
    if not ZX_BOT_ws_manager._ws_manager:
        raise RuntimeError("WebSocket manager not initialized")
    
    # Suscribir si no está suscrito
    if symbol not in ZX_BOT_ws_manager._ws_manager.subscribed_public:
        ZX_BOT_ws_manager._ws_manager.subscribe_ticker(symbol)
    
    price_data = ZX_BOT_ws_manager._ws_manager.prices.get(symbol)
    
    # Usar caché si es suficientemente fresco
    if price_data:
        age = time.time() - price_data['timestamp']
        if age < max_cache_age:
            return price_data['price']
    
    # Esperar dato fresco
    initial_timestamp = price_data.get('timestamp', 0) if price_data else 0
    timeout = 1.0
    start_time = time.time()
    
    while (time.time() - start_time) < timeout:
        price_data = ZX_BOT_ws_manager._ws_manager.prices.get(symbol)
        if price_data and price_data['timestamp'] > initial_timestamp:
            return price_data['price']
        time.sleep(0.01)
    
    # Timeout
    raise TimeoutError(f"No fresh price data for {symbol}")

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
# TIMEFRAMES & CANDLES
# ==========================================================================
def calculate_next_candle_time(timeframe='4H', hour_zone=None):
    from datetime import datetime, timedelta
    
    now = datetime.now(hour_zone)
    
    if timeframe.endswith('Hutc'):
        hours = int(timeframe[:-4])
        minutes = hours * 60
    elif timeframe.endswith('H'):
        hours = int(timeframe[:-1])
        minutes = hours * 60
    elif timeframe.endswith('m'):
        minutes = int(timeframe[:-1])
    elif timeframe.endswith('Dutc'):
        days = int(timeframe[:-4])
        minutes = days * 24 * 60
    else:
        raise ValueError("Invalid timeframe. Use 'm', 'H', 'Hutc', or 'Dutc'. Examples: '15m', '4H', '6Hutc', '1Dutc'")
    
    total_minutes = now.hour * 60 + now.minute
    next_total_minutes = ((total_minutes // minutes) + 1) * minutes
    delta_minutes = next_total_minutes - total_minutes
    next_candle = now + timedelta(minutes=delta_minutes, seconds=-now.second, microseconds=-now.microsecond)
    
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

    print(f"\n🕓 TIMEOUT REACHED for strategy {strat_id}")
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
# TP/SL CHECKINGS
# ==========================================================================
# ==========================================================================
# TP/SL CHECKINGS
# ==========================================================================
def check_tp_sl_for_strategy(strat_id, strat_config, open_positions, strategy_candles,
                              state_file, send_request_func, table=None, pnl_accumulator=None):
    """Comprueba TP/SL para todas las posiciones de una estrategia via WebSocket"""
    if strat_id not in open_positions or not open_positions[strat_id]:
        return
    
    positions = open_positions[strat_id][:]
    positions_to_remove = []
    
    sell_after_ncandles = strat_config.get('sell_after_ncandles') if strat_config else None
    
    for i, pos in enumerate(positions):
        symbol = pos['symbol']
        
        # Obtener precio via WebSocket
        try:
            current_price = get_current_price(symbol, max_cache_age=0.5)
        except (TimeoutError, RuntimeError) as e:
            print(f"⚠️ No price for {symbol}: {e}")
            continue
        
        if current_price is None:
            continue
        
        direction = pos['direction']
        tp_price = pos['tp']
        sl_price = pos['sl']
        entry_price = pos['entry_price']
        
        current_price = Decimal(str(current_price))
        
        # ⭐ CALCULAR PnL SIEMPRE (aunque no haya tabla)
        if pnl_accumulator is not None:
            from ZX_BOT_display import calculate_pnl
            pnl = calculate_pnl(direction, entry_price, current_price, pos['size'])
            pnl_accumulator['total'] += pnl
        
        # LUEGO añadir a tabla si existe
        if table is not None:
            add_position_to_table(table, strat_id, pos, current_price, pnl_accumulator,
                                 strategy_candles, sell_after_ncandles)
        
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
    
    if positions_to_remove:
        for i in reversed(positions_to_remove):
            if i < len(open_positions[strat_id]):
                open_positions[strat_id].pop(i)
        
        save_state_local(open_positions, strategy_candles, state_file)



# ==========================================================================
# STRATEGY MANAGEMENT
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
    """Procesa una estrategia: busca señales y abre posiciones."""
    strat_id = strat['id']

    print(f"\n🔄 Processing strategy: {strat_id}")

    if use_hardcoded:
        signals = get_hardcoded_signals(strat_id, send_request_func, hour_zone)
    else:
        if detect_signal_func is None:
            raise ValueError("No se proporcionó detect_signal_func")
        signals = detect_signal_func(strat, final_symbols)

    print(f"💫 Signals detected for {strat_id}: {len(signals)}")

    if not signals:
        return

    reset_strategy_candles(strat_id, strategy_candles, open_positions, state_file)

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

            add_position(strat_id=strat_id, symbol=sig['symbol'], size=size, entry_price=entry_price,
                        direction=strat['direction'], tp_pct=strat['tp_pct'], sl_pct=strat['sl_pct'],
                        order_id=order_id, open_positions=open_positions, strategy_candles=strategy_candles,
                        state_file=state_file, hour_zone=hour_zone, usdt_amount=strat['order_amount'])
        else:
            print(f"⚠️ Order executed but no orderId in response")

        time.sleep(0.05)

def get_hardcoded_signals(strat_id, send_request_func, hour_zone):
    """Genera señales de prueba para testing"""
    symbols = ['BTCUSDT', 'BNBUSDT']
    signals = []
    for symbol in symbols:
        code, resp = send_request_func("GET", "/api/v2/mix/market/ticker",params={"productType": PRODUCT_TYPE, "symbol": symbol})
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
    """Cierra una posición con orden market"""
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
        
        if reason == "TP":
            print(f"\n💲 TP REACHED for {symbol} ({position_data.get('strategy_id', 'N/A') if position_data else 'N/A'}) at {datetime.now().strftime('%H:%M')}")
        elif reason == "SL":
            print(f"\n🔻 SL REACHED for {symbol} ({position_data.get('strategy_id', 'N/A') if position_data else 'N/A'}) at {datetime.now().strftime('%H:%M')}")

        print(f"→  Closing {direction} position on {symbol}:")
        code, resp = send_request_func("POST", "/api/v2/mix/order/place-order", body=body)
        time.sleep(0.05)
        
        if code == 200 and resp.get("code") == "00000":
            print(f"✅ Position closed due to {reason}: {symbol} | Size: {size}")
            
            if position_data:
                data = resp.get('data', {})
                order_id = data.get('orderId')
                
                if order_id:
                    _, close_price_from_fills, profit_from_api, fee_from_api = get_fills_for_order(
                        order_id=order_id,
                        symbol=symbol,
                        send_request_func=send_request_func
                    )
                    
                    if close_price_from_fills is None:
                        close_price_from_fills = Decimal(str(data.get('price', 0)))
                        if close_price_from_fills == 0:
                            close_price_from_fills = get_current_price(symbol)
                    
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
            print(f"⚠️ No closing position available {symbol}: {resp}")
            if resp.get("code") == "22002":
                print(f"   → Removing from local record (nonexistent position)")
                if position_data:
                    current_price = get_current_price(symbol)
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
        traceback.print_exc()
        return False

def add_position(strat_id, symbol, size, entry_price, direction, tp_pct, sl_pct, order_id,
                 open_positions, strategy_candles, state_file, hour_zone, usdt_amount=0):
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
    fee_from_api=None
):
    """
    Registra una posición cerrada en el archivo Excel.
    Usa la ruta definida en TRADES_LOG_PATH (constante global).
    """
    try:
        entry_price = float(entry_price)
        close_price = float(close_price)
        usdt_amount = float(usdt_amount)

        size_val = None
        if size is not None:
            try:
                size_val = float(size)
            except Exception:
                size_val = None

        if usdt_amount == 0 and size_val is not None:
            usdt_amount = size_val * entry_price

        if profit_from_api is not None:
            profit_gross = float(profit_from_api)
            fee = float(fee_from_api) if fee_from_api is not None else 0
            fee = 2 * fee
            profit = profit_gross - fee
            
            if usdt_amount > 0:
                profit_pct = (profit / usdt_amount) * 100
            else:
                profit_pct = 0
        else:
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

        # ⭐ Usar la ruta global
        if os.path.exists(TRADES_LOG_PATH):
            df = pd.read_excel(TRADES_LOG_PATH)
            df = pd.concat([df, pd.DataFrame([new_record])], ignore_index=True)
        else:
            df = pd.DataFrame([new_record])

        df.to_excel(TRADES_LOG_PATH, index=False, engine='openpyxl')

        print(f"📋 Trade logged: {symbol} | Profit: {profit:.2f} USDT ({profit_pct:+.2f}%) | Duration: {delta_days:.4f} days")

    except Exception as e:
        print(f"❌ Error logging trade to Excel: {e}")
        traceback.print_exc()

def setup_print_logger(logdir, logfile_name="BOT_all_stratagies.log"):
    """Configura un logger que duplica print() al archivo y a consola."""
    os.makedirs(logdir, exist_ok=True)
    logfile = os.path.join(logdir, logfile_name)
    
    logger = logging.getLogger('bot_logger')
    logger.setLevel(logging.INFO)
    logger.propagate = False
    
    fh = logging.FileHandler(logfile, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    
    if not logger.handlers:
        logger.addHandler(fh)
    
    old_print = builtins.print
    
    def _print_and_log(*args, **kwargs):
        old_print(*args, **kwargs)
        text = kwargs.get("sep", " ").join(str(a) for a in args)
        logger.info(text)
    
    builtins.print = _print_and_log

# Import display functions (assuming they exist)
try:
    from ZX_BOT_display import create_tp_sl_display
    console = Console()
    _live_display = None
except ImportError:
    def create_tp_sl_display(now, total=None):
        return None, None
    console = None
    _live_display = None