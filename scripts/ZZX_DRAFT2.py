import heapq
import numpy as np

# ================================
# Función a testear
# ================================
def update_sim_balance(t_int, open_heap, cash, sym_data_local, ts_int_arrays, close_arrays, sim_balance_cols):
    active_positions = [pos for _, _, pos in open_heap if not pos.get('closed', False)]
    
    if not active_positions:
        total_value = 0.0
    else:
        # Agrupar por símbolo
        symbol_groups = {}
        for pos in active_positions:
            sym = pos['symbol']
            symbol_groups.setdefault(sym, []).append(pos['qty'])
        
        total_value = 0.0
        for sym, qtys in symbol_groups.items():
            # Obtener precio actual solo una vez por símbolo
            ts_int = ts_int_arrays[sym]
            close_arr = close_arrays[sym]

            # Buscar índice más cercano sin pasar el timestamp actual
            idx = np.searchsorted(ts_int, t_int, side='right') - 1
            if idx < 0:
                price = close_arr[0]
            else:
                price = close_arr[idx]

            # Multiplicar suma de cantidades * precio
            total_value += np.sum(qtys) * price

    sim_balance_cols['timestamp'].append(np.datetime64(int(t_int), 'ns'))
    sim_balance_cols['balance'].append(cash + total_value)
    return sim_balance_cols

# ================================
# Test unitario autocontenido
# ================================
def test_update_sim_balance():
    # Datos simulados
    cash = 10000.0
    t_int = np.int64(1_700_000_000_000_000_000)  # timestamp simulado

    sym_data_local = {
        'BTCUSD': {
            'ts_int': np.array([t_int - 10_000_000_000, t_int, t_int + 10_000_000_000]),
            'close': np.array([20000.0, 21000.0, 22000.0])
        }
    }

    ts_int_arrays = {'BTCUSD': sym_data_local['BTCUSD']['ts_int']}
    close_arrays = {'BTCUSD': sym_data_local['BTCUSD']['close']}

    # Posición abierta simulada: 0.1 BTC
    open_heap = [(t_int, 0, {'symbol': 'BTCUSD', 'qty': 0.1, 'closed': False})]

    sim_balance_cols = {'timestamp': [], 'balance': []}

    sim_balance_cols = update_sim_balance(t_int, open_heap, cash, sym_data_local, ts_int_arrays, close_arrays, sim_balance_cols)

    # Print para ver resultados
    print("=== Test update_sim_balance ===")
    print("Cash inicial:", cash)
    print("Valor posiciones abiertas:", 0.1 * 21000.0)
    print("Balance final esperado:", cash + 0.1 * 21000.0)
    print("Balance calculado por función:", sim_balance_cols['balance'][0])
    print("Timestamp registrado:", sim_balance_cols['timestamp'][0])

# Ejecutar test
test_update_sim_balance()
