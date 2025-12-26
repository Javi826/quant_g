#ZX_BOT_display.py
"""
Módulo de visualización para el bot de trading.
Versión simplificada para uso con dashboard web.
"""

from datetime import datetime


# ==========================================================================
# FUNCIÓN PRINCIPAL - VERSIÓN SIMPLIFICADA
# ==========================================================================
def check_all_tp_sl(strategies, open_positions, strategy_candles, state_file, 
                    send_request_func, hour_zone, check_tp_sl_for_strategy_func, 
                    get_current_price_func=None, display_mode="simple", account_number=None,
                    display_color=None, bot_state=None):
    """
    Versión simplificada para uso con dashboard.
    Mantiene toda la lógica de negocio pero solo imprime mensaje básico.
    
    Parámetros opcionales (ignorados en modo simple):
    - get_current_price_func: No se usa (mantenido por compatibilidad)
    - display_mode: No se usa (mantenido por compatibilidad)
    - display_color: No se usa (mantenido por compatibilidad)
    """
    now = datetime.now(hour_zone).strftime('%Y-%m-%d %H:%M:%S')
    
    
    # Acumulador para el PnL total
    pnl_accumulator = {'total': 0.0}
    
    # Lógica de negocio: procesar todas las estrategias
    for strat in strategies:
        strat_id = strat['id']
        positions = open_positions.get(strat_id, [])
        
        if positions:
            strat_pnl_acc = {'total': 0.0}
            check_tp_sl_for_strategy_func(
                strat_id, strat, open_positions, strategy_candles, 
                state_file, send_request_func, None, strat_pnl_acc, bot_state
            )
            pnl_accumulator['total'] += strat_pnl_acc['total']
    
    return pnl_accumulator

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