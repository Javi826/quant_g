"""
Módulo de visualización para el bot de trading.
Maneja toda la lógica de display con ANSI puro (sin Rich)
"""

from datetime import datetime
import time

# ==========================================================================
# CONSTANTES Y VARIABLES GLOBALES
# ==========================================================================
BLUE_BOLD    = "\033[1;94m"
BLUE         = "\033[0;94m"
RESET        = "\033[0m"

# Constantes de color
WHITE        = "\033[37m"
WHITE_BOLD   = "\033[1;37m"
GREEN        = "\033[92m"
GREEN_BOLD   = "\033[1;92m"
RED          = "\033[91m"
RED_BOLD     = "\033[1;91m"
YELLOW       = "\033[93m"
CYAN         = "\033[96m"
MAGENTA      = "\033[95m"

# ==========================================================================
# FUNCIONES DE FORMATEO
# ==========================================================================
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
        if current_float > entry_float:
            return f"{GREEN_BOLD}↑{RESET}"
        else:
            return f"{RED_BOLD}↓{RESET}"
    else:  # short
        if current_float < entry_float:
            return f"{GREEN_BOLD}↑{RESET}"
        else:
            return f"{RED_BOLD}↓{RESET}"


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


# ==========================================================================
# FUNCIONES DE DISPLAY
# ==========================================================================
def stop_live_display():
    """Detiene el display ANSI si está activo y limpia la pantalla"""
    
    # Si hay un display activo, limpiarlo
    if hasattr(check_all_tp_sl, '_last_lines') and check_all_tp_sl._last_lines > 0:
        # Mover cursor abajo del display actual para no sobreescribir nada
        print(f"\033[{check_all_tp_sl._last_lines}B", end='', flush=True)
        # Resetear contador
        delattr(check_all_tp_sl, '_last_lines')
    
    # Resetear flag de inicialización para forzar nueva tabla completa
    if hasattr(check_all_tp_sl, '_initialized'):
        delattr(check_all_tp_sl, '_initialized')
    
    # Pequeña pausa para que el terminal procese
    time.sleep(0.05)


# ==========================================================================
# FUNCIÓN PRINCIPAL DE DISPLAY
# ==========================================================================
def check_all_tp_sl(strategies, open_positions, strategy_candles, state_file, 
                    send_request_func, hour_zone, check_tp_sl_for_strategy_func, 
                    get_current_price_func, display_mode="summary", account_number=None,
                    display_color=None, bot_state=None):
    
    now = datetime.now(hour_zone).strftime('%Y-%m-%d %H:%M:%S')
    
    # Acumulador para el PnL total
    pnl_accumulator = {'total': 0.0}
    
    # =======================================================================
    # MODE "none": PRINT SIMPLE
    # =======================================================================
    if display_mode == "none":
        print(f"\n{BLUE}{'─'*115}")
        print(f"🛰️  Checking TP/SL - {now}")
        
        for strat in strategies:
            strat_id = strat['id']
            num_positions = len(open_positions.get(strat_id, []))
            
            if num_positions > 0:
                check_tp_sl_for_strategy_func(
                    strat_id, strat, open_positions, strategy_candles, 
                    state_file, send_request_func, None, pnl_accumulator, bot_state
                )
        
        total_pnl = pnl_accumulator['total']
        pnl_color = GREEN_BOLD if total_pnl >= 0 else RED_BOLD
        print(f"💰 Total PnL: {pnl_color}{total_pnl:+.2f} USDT{RESET}")
        print(f"{BLUE_BOLD}{'─'*115}{RESET}\n")
        return
    
    # =======================================================================
    # MODE "summary": TABLA RESUMIDA
    # =======================================================================
    if display_mode == "summary":
        # Procesar estrategias y acumular datos
        rows = []
        for strat in sorted(strategies, key=lambda s: s['id']):
            strat_id = strat['id']
            positions = open_positions.get(strat_id, [])
            
            if positions:
                strat_pnl_acc = {'total': 0.0}
                check_tp_sl_for_strategy_func(
                    strat_id, strat, open_positions, strategy_candles, 
                    state_file, send_request_func, None, strat_pnl_acc, bot_state
                )
                pnl_accumulator['total'] += strat_pnl_acc['total']
                
                # Re-chequear después de check_tp_sl
                positions = open_positions.get(strat_id, [])
                if not positions:
                    continue
                
                first_pos = positions[0]
                direction = first_pos['direction'].upper()
                opened_at = first_pos.get('opened_at', '')
                if hasattr(opened_at, 'strftime'):
                    opened_at_str = opened_at.strftime('%Y-%m-%d')
                elif isinstance(opened_at, str):
                    opened_at_str = opened_at.split('T')[0] if 'T' in opened_at else opened_at[:10]
                else:
                    opened_at_str = str(opened_at)[:10]
                
                candles_elapsed = strategy_candles.get(strat_id, 0)
                sell_after = strat.get('sell_after_ncandles', 0)
                candles_str = f"{candles_elapsed}/{sell_after}"
                
                strat_pnl = strat_pnl_acc['total']
                pnl_color = GREEN if strat_pnl >= 0 else RED
                
                rows.append((strat_id, direction, opened_at_str, candles_str, len(positions), strat_pnl, pnl_color))
        
        # Calcular número total de líneas de la tabla
        total_lines = 8 + len(rows)
        
        # Detectar si el terminal cambió de tamaño
        import os
        current_term_size = os.get_terminal_size().lines if hasattr(os, 'get_terminal_size') else None
        
        # Si no es la primera vez Y el tamaño no cambió, mover cursor arriba
        should_overwrite = (
            hasattr(check_all_tp_sl, '_last_lines') and 
            hasattr(check_all_tp_sl, '_last_term_size') and
            current_term_size == check_all_tp_sl._last_term_size
        )
        
        if should_overwrite:
            print(f"\033[{check_all_tp_sl._last_lines}A", end='', flush=True)
        
        # Guardar estado para próxima vez
        check_all_tp_sl._last_lines = total_lines
        check_all_tp_sl._last_term_size = current_term_size
        
        # Header (limpiar línea completa con \033[K)
        print(f"\033[K{display_color}{'─'*72}{RESET}")
        account_str = f" (ACC: {account_number})" if account_number else ""
        print(f"\033[K{display_color}🛰️  Checking TP/SL{account_str}{RESET} - {WHITE}{now}{RESET}")
        
        # Mostrar Closed P/L y Open P/L
        line_pnl = ""
        if bot_state is not None:
            total_profit = bot_state.closed_total_profit
            profit_color = GREEN_BOLD if total_profit >= 0 else RED_BOLD
            line_pnl += f"{WHITE_BOLD}💰 Closed PnL: {profit_color}{total_profit:+.2f} USDT{RESET}"
            line_pnl += f"{WHITE} | {RESET}"
        
        total_pnl = pnl_accumulator['total']
        pnl_color = GREEN if total_pnl >= 0 else RED
        line_pnl += f"{WHITE}Open PnL: {pnl_color}{total_pnl:+.2f} USDT{RESET}"
        
        # BTC price
        try:
            btc_price = get_current_price_func('BTCUSDT')
            line_pnl += f"{WHITE} | BTC: {RESET}{YELLOW}{btc_price:,.2f}{RESET}{WHITE} USDT{RESET}"
        except:
            pass
        
        print(f"\033[K{line_pnl}")
        print(f"\033[K{display_color}{'─'*72}{RESET}")
        
        # Tabla con box drawing characters
        border_color = CYAN if display_color == "\033[1;96m" else "\033[94m"
        
        # Header de tabla
        print(f"\033[K{border_color}┏{'━'*22}┳{'━'*8}┳{'━'*12}┳{'━'*9}┳{'━'*5}┳{'━'*8}┓{RESET}")
        print(f"\033[K{border_color}┃{WHITE_BOLD}{'Strategy':<22}{RESET}{border_color}┃{WHITE_BOLD}{'Side':<8}{RESET}{border_color}┃{WHITE_BOLD}{'Opened':<12}{RESET}{border_color}┃{WHITE_BOLD}{'Candles':>9}{RESET}{border_color}┃{WHITE_BOLD}{'#pos':>5}{RESET}{border_color}┃{WHITE_BOLD}{'PnL':>8}{RESET}{border_color}┃{RESET}")
        print(f"\033[K{border_color}┡{'━'*22}╇{'━'*8}╇{'━'*12}╇{'━'*9}╇{'━'*5}╇{'━'*8}┩{RESET}")
        
        # Filas de datos
        for strat_id, direction, opened_at_str, candles_str, num_pos, pnl, pnl_color in rows:
            pnl_str = f"{pnl:+8.2f}"
            print(f"\033[K{border_color}│{WHITE}{strat_id:<22}{RESET}{border_color}│{WHITE}{direction:<8}{RESET}{border_color}│{WHITE}{opened_at_str:<12}{RESET}{border_color}│{WHITE}{candles_str:>9}{RESET}{border_color}│{CYAN}{num_pos:>5}{RESET}{border_color}│{pnl_color}{pnl_str}{RESET}{border_color}│{RESET}")
        
        # Footer de tabla
        print(f"\033[K{border_color}└{'─'*22}┴{'─'*8}┴{'─'*12}┴{'─'*9}┴{'─'*5}┴{'─'*8}┘{RESET}")
        
        return
    
    # =======================================================================
    # MODE "detailed": TABLA DETALLADA
    # =======================================================================
    if display_mode == "detailed":
        # Limpiar pantalla
        print("\033[2J\033[H", end='', flush=True)
        
        # Header
        print(f"{display_color}{'─'*115}")
        account_str = f" (ACC: {account_number})" if account_number else ""
        print(f"{display_color}🛰️  Checking TP/SL{account_str} - {RESET}{now}")
        
        # Procesar todas las estrategias
        for strat in strategies:
            strat_id = strat['id']
            num_positions = len(open_positions.get(strat_id, []))
            
            if num_positions > 0:
                check_tp_sl_for_strategy_func(
                    strat_id, strat, open_positions, strategy_candles, 
                    state_file, send_request_func, None, pnl_accumulator, bot_state
                )
        
        # Mostrar PnL total
        total_pnl = pnl_accumulator['total']
        pnl_color = GREEN_BOLD if total_pnl >= 0 else RED_BOLD
        print(f"{WHITE_BOLD}💰 Total PnL: {pnl_color}{total_pnl:+.2f} USDT{RESET}")
        print(f"{display_color}{'─'*115}{RESET}")
        
        # Tabla detallada (header)
        print(f"{WHITE}┏{'━'*20}┳{'━'*11}┳{'━'*5}┳{'━'*10}┳{'━'*8}┳{'━'*8}┳{'━'*7}┳{'━'*8}┳{'━'*1}┳{'━'*6}┳{'━'*20}┳{'━'*20}┓{RESET}")
        print(f"{WHITE}┃{WHITE_BOLD}{'Strategy':<20}{RESET}{WHITE}┃{WHITE_BOLD}{'Symbol':<11}{RESET}{WHITE}┃{WHITE_BOLD}{'Side':<5}{RESET}{WHITE}┃{WHITE_BOLD}{'Opened':<10}{RESET}{WHITE}┃{WHITE_BOLD}{'Candles':^8}{RESET}{WHITE}┃{WHITE_BOLD}{'Entry':>8}{RESET}{WHITE}┃{WHITE_BOLD}{'Size':>7}{RESET}{WHITE}┃{WHITE_BOLD}{'Current':>8}{RESET}{WHITE}┃{WHITE_BOLD}{'↕':^1}{RESET}{WHITE}┃{WHITE_BOLD}{'PnL':>6}{RESET}{WHITE}┃{WHITE_BOLD}{'TP':>20}{RESET}{WHITE}┃{WHITE_BOLD}{'SL':>20}{RESET}{WHITE}┃{RESET}")
        print(f"{WHITE}┡{'━'*20}╇{'━'*11}╇{'━'*5}╇{'━'*10}╇{'━'*8}╇{'━'*8}╇{'━'*7}╇{'━'*8}╇{'━'*1}╇{'━'*6}╇{'━'*20}╇{'━'*20}┩{RESET}")
        
        # Filas de posiciones
        for strat in strategies:
            strat_id = strat['id']
            positions = open_positions.get(strat_id, [])
            
            for pos in positions:
                direction = pos['direction']
                symbol = pos['symbol']
                entry_price = pos['entry_price']
                size = pos['size']
                tp_price = pos['tp']
                sl_price = pos['sl']
                
                # Obtener precio actual
                try:
                    current_price = get_current_price_func(symbol)
                except:
                    current_price = entry_price
                
                # Calcular distancias
                if direction.lower() == 'short':
                    dist_to_tp = float(current_price - tp_price)
                    dist_to_sl = float(sl_price - current_price)
                    tp_pct_away = (dist_to_tp / float(entry_price)) * 100
                    sl_pct_away = (dist_to_sl / float(entry_price)) * 100
                else:
                    dist_to_tp = float(tp_price - current_price)
                    dist_to_sl = float(current_price - sl_price)
                    tp_pct_away = (dist_to_tp / float(entry_price)) * 100
                    sl_pct_away = (dist_to_sl / float(entry_price)) * 100
                
                # PnL
                pnl = calculate_pnl(direction, entry_price, current_price, size)
                pnl_color = GREEN if pnl >= 0 else RED
                pnl_arrow = get_pnl_arrow(direction, entry_price, current_price)
                
                # Opened date
                opened_at = pos.get('opened_at', '')
                if hasattr(opened_at, 'strftime'):
                    opened_at_str = opened_at.strftime('%Y-%m-%d')
                elif isinstance(opened_at, str):
                    opened_at_str = opened_at.split('T')[0] if 'T' in opened_at else opened_at[:10]
                else:
                    opened_at_str = str(opened_at)[:10]
                
                # Candles
                candles_elapsed = strategy_candles.get(strat_id, 0)
                sell_after = strat.get('sell_after_ncandles', 0)
                candles_str = f"{candles_elapsed}/{sell_after}"
                
                # TP/SL con colores
                tp_color = GREEN_BOLD if tp_pct_away < 1 else CYAN
                sl_color = RED_BOLD if sl_pct_away < 1 else MAGENTA
                
                tp_text = f"{format_price(tp_price)} {tp_color}(Δ {tp_pct_away:+.2f}%){RESET}"
                sl_text = f"{format_price(sl_price)} {sl_color}(Δ {sl_pct_away:+.2f}%){RESET}"
                
                size_str = f"{float(size):.6f}".rstrip('0').rstrip('.')
                
                # Imprimir fila
                print(f"{WHITE}│{strat_id:<20}│{symbol:<11}│{direction.upper():<5}│{opened_at_str:<10}│{candles_str:^8}│{format_price(entry_price):>8}│{size_str:>7}│{YELLOW}{format_price(current_price):>8}{WHITE}│{pnl_arrow}│{pnl_color}{pnl:+.2f}{WHITE}│{tp_text:<20}│{sl_text:<20}│{RESET}")
        
        # Footer
        print(f"{WHITE}└{'─'*20}┴{'─'*11}┴{'─'*5}┴{'─'*10}┴{'─'*8}┴{'─'*8}┴{'─'*7}┴{'─'*8}┴{'─'*1}┴{'─'*6}┴{'─'*20}┴{'─'*20}┘{RESET}")
        
        return
    
    # =======================================================================
    # MODO DESCONOCIDO: Usar "summary" por defecto
    # =======================================================================
    print(f"⚠️  Unknown display_mode '{display_mode}', using 'summary' by default")
    check_all_tp_sl(strategies, open_positions, strategy_candles, state_file,
                    send_request_func, hour_zone, check_tp_sl_for_strategy_func,
                    get_current_price_func, display_mode="summary", account_number=account_number,
                    display_color=display_color, bot_state=bot_state)