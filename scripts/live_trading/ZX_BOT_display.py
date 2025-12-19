"""
Módulo de visualización para el bot de trading.
Maneja toda la lógica de display con Rich (tablas, formateo, etc.)
"""

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.console import Group
from datetime import datetime

# ==========================================================================
# CONSTANTES Y VARIABLES GLOBALES
# ==========================================================================
BLUE_BOLD = "\033[1;94m"
RESET     = "\033[0m"

console = Console()
_live_display = None

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


# ==========================================================================
# FUNCIONES DE CONSTRUCCIÓN DE TABLA
# ==========================================================================
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
    
    # ⭐ Calcular PnL solo para mostrar, NO acumular (ya se acumuló antes)
    pnl = calculate_pnl(direction, entry_price, current_price, size)
    
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
        f"[white]{size_str}[/white]",
        f"[yellow]{format_price(current_price)}[/yellow]",
        pnl_arrow,
        pnl_text,
        tp_text,
        sl_text
    )


def create_tp_sl_display(now, total_pnl=None, account_number=None, display_color=None):  # ⭐ AÑADIR
    """Crea el header y la tabla para el display de TP/SL"""
    # Crear el header con PnL total si se proporciona
    header = Text()
    header.append(f"{display_color}{'─'*115}\n")  # ⭐ USAR display_color
    account_str = f" (ACC: {account_number})" if account_number else ""
    header.append(f"{display_color}🔷 Checking TP/SL{account_str} - {now}\n")  # ⭐ USAR display_color
    if total_pnl is not None:
        pnl_color = "bold green" if total_pnl >= 0 else "bold red"
        header.append(f"💰 Total PnL: ", style="white")
        header.append(f"{total_pnl:+.2f} USDT\n", style=pnl_color)
    header.append(f"{display_color}{'─'*115}\n")  
    
    
    # Crear tabla con columnas adicionales: opened_at y candles
    table = Table(show_header=True, header_style="bold white", border_style="white")
    table.add_column("Strategy", style="white", width=20)
    table.add_column("Symbol", style="bold", width=11)
    table.add_column("Side", justify="left", width=5)
    table.add_column("Opened", style="white", width=10)
    table.add_column("Candles", justify="center", width=8)
    table.add_column("Entry", justify="right", width=8)
    table.add_column("Size", justify="right", width=7)
    table.add_column("Current", justify="right", width=8)
    table.add_column("↕", justify="center", width=1)
    table.add_column("PnL (USDT)", justify="right", width=6)
    table.add_column("TP", justify="right", width=20)
    table.add_column("SL", justify="right", width=20)
    
    return header, table


# ==========================================================================
# FUNCIÓN PRINCIPAL DE DISPLAY
# ==========================================================================
def check_all_tp_sl(strategies, open_positions, strategy_candles, state_file, 
                    send_request_func, hour_zone, check_tp_sl_for_strategy_func, 
                    get_current_price_func, display_mode="summary", account_number=None,
                    display_color=None):

    global _live_display
    
    now = datetime.now(hour_zone).strftime('%Y-%m-%d %H:%M:%S')
    
    # Acumulador para el PnL total
    pnl_accumulator = {'total': 0.0}
    
    # =======================================================================
    # MODE "none": PRINT SIMPLE (sin Rich)
    # =======================================================================
    if display_mode == "none":
        print(f"\n{BLUE_BOLD}{'─'*115}")
        print(f"🔷 Checking TP/SL - {now}")
        
        for strat in strategies:
            strat_id = strat['id']
            num_positions = len(open_positions.get(strat_id, []))
            
            if num_positions > 0:
                check_tp_sl_for_strategy_func(
                    strat_id, strat, open_positions, strategy_candles, 
                    state_file, send_request_func, None, pnl_accumulator
                )
        
        total_pnl = pnl_accumulator['total']
        pnl_color = "\033[1;92m" if total_pnl >= 0 else "\033[1;91m"
        print(f"💰 Total PnL: {pnl_color}{total_pnl:+.2f} USDT{RESET}")
        print(f"{BLUE_BOLD}{'─'*115}{RESET}\n")
        return
    
    # =======================================================================
    # MODE "summary": BRIEF TABLE 
    # =======================================================================
    if display_mode == "summary":
        header = Text()
        header.append(f"{display_color}{'─'*72}\n")  # ⭐ USAR display_color
        account_str = f" (ACC: {account_number})" if account_number else ""
        header.append(f"{display_color}🔷 Checking TP/SL{account_str} - {now}\n")
        
        # Crear tabla resumida
        border_color = "bright_cyan" if display_color == "\033[1;96m" else "bright_blue"  # ⭐ AÑADIR
        summary_table = Table(show_header=True, header_style="bold white", border_style=border_color)  # ⭐ CAMBIAR
        summary_table.add_column("Strategy", style="white", width=20)
        summary_table.add_column("Side", justify="left", width=6)
        summary_table.add_column("Opened", style="white", width=10)
        summary_table.add_column("Candles", justify="right", width=7)
        summary_table.add_column("#pos", justify="right", width=4)
        summary_table.add_column("PnL", justify="right", width=6)
        
        # Procesar cada estrategia

        for strat in sorted(strategies, key=lambda s: s['id']):
            strat_id = strat['id']
            positions = open_positions.get(strat_id, [])
            num_positions = len(positions)
            
            if num_positions > 0:
                # Acumulador local para esta estrategia
                strat_pnl_acc = {'total': 0.0}
                
                # Chequear TP/SL y acumular PnL
                check_tp_sl_for_strategy_func(
                    strat_id, strat, open_positions, strategy_candles, 
                    state_file, send_request_func, None, strat_pnl_acc
                )
                
                # Acumular al total
                pnl_accumulator['total'] += strat_pnl_acc['total']
                
                # ⭐ RE-CHEQUEAR positions después de check_tp_sl (puede haberse cerrado)
                positions = open_positions.get(strat_id, [])
                if not positions:
                    continue
                
                # Datos de la primera posición
                first_pos = positions[0]
                direction = first_pos['direction'].upper()
                opened_at = first_pos.get('opened_at', '')
                if hasattr(opened_at, 'strftime'):
                    opened_at_str = opened_at.strftime('%Y-%m-%d')
                elif isinstance(opened_at, str):
                    opened_at_str = opened_at.split('T')[0] if 'T' in opened_at else opened_at[:10]
                else:
                    opened_at_str = str(opened_at)[:10]
                
                # Candles
                candles_elapsed = strategy_candles.get(strat_id, 0)
                sell_after      = strat.get('sell_after_ncandles', 0)
                candles_str     = f"{candles_elapsed}/{sell_after}"
                
                # PnL con color
                strat_pnl = strat_pnl_acc['total']
                pnl_color = "green" if strat_pnl >= 0 else "red"
                pnl_text = f"[{pnl_color}]{strat_pnl:+.2f}[/{pnl_color}]"
                
                # Side con color
                side_color = "white"
                
                summary_table.add_row(
                    strat_id,
                    f"[{side_color}]{direction}[/{side_color}]",
                    f"[white]{opened_at_str}[/white]",
                    f"[white]{candles_str}[/white]",
                    f"[cyan]{num_positions}[/cyan]",
                    pnl_text
                )
        
        # Añadir total PnL al header
        total_pnl = pnl_accumulator['total']
        pnl_color = "bold green" if total_pnl >= 0 else "bold red"
        
        # Obtener precio de BTCUSDT
        try:
            btc_price = get_current_price_func('BTCUSDT')
        except Exception:
            btc_price = None
        
        header.append(f"💰 Total PnL: ", style="white")
        header.append(f"{total_pnl:+.2f} USDT", style=pnl_color)
        if btc_price:
            header.append(f" | BTC: ", style="white")
            header.append(f"{btc_price:,.2f}", style="yellow")
            header.append(f" USDT\n", style="white")
        else:
            header.append("\n")
        header.append(f"{display_color}{'─'*72}\n")
        
        display = Group(header, summary_table)
        
        if _live_display is None:
            _live_display = Live(display, console=console, refresh_per_second=4)
            _live_display.start()
        else:
            _live_display.update(display)
        return
    
    # =======================================================================
    # MODO "detailed": TABLA DETALLADA (una fila por posición)
    # =======================================================================
    if display_mode == "detailed":
        header, table = create_tp_sl_display(now, account_number=account_number, display_color=display_color)
        
        for idx, strat in enumerate(strategies):
            strat_id = strat['id']
            num_positions = len(open_positions.get(strat_id, []))
            
            if num_positions > 0:
                check_tp_sl_for_strategy_func(
                    strat_id, strat, open_positions, strategy_candles, 
                    state_file, send_request_func, table, pnl_accumulator
                )
                
                if idx < len(strategies) - 1:
                    next_has_positions = any(
                        len(open_positions.get(strategies[next_idx]['id'], [])) > 0 
                        for next_idx in range(idx + 1, len(strategies))
                    )
                    if next_has_positions:
                        table.add_row("", "", "", "", "", "", "", "", "", "", "", "")
        
        header, _ = create_tp_sl_display(now, pnl_accumulator['total'], account_number, display_color)
        display = Group(header, table)
        
        if _live_display is None:
            _live_display = Live(display, console=console, refresh_per_second=4)
            _live_display.start()
        else:
            _live_display.update(display)
        return
    
    # =======================================================================
    # MODO DESCONOCIDO: Usar "summary" por defecto
    # =======================================================================
    print(f"⚠️  Unknown display_mode '{display_mode}', using 'summary' by default")
    check_all_tp_sl(strategies, open_positions, strategy_candles, state_file,
                    send_request_func, hour_zone, check_tp_sl_for_strategy_func,
                    get_current_price_func, display_mode="summary")


def stop_live_display():
    """Detiene el display de Rich si está activo"""
    global _live_display
    if _live_display is not None:
        _live_display.stop()
        _live_display = None