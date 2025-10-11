import heapq
import logging
import warnings
import numpy as np
import pandas as pd
from numba import njit
logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

MIN_PRICE            = 0.0001
INITIAL_BALANCE      = 10000
ORDER_AMOUNT         = 100

# ============================
# Helper: get_price_at_int (nivel módulo)
# ============================

# Función numba para acelerar searchsorted
@njit
def _search_price_numba(ts_int_arr, close_arr, t_int):
    idx = np.searchsorted(ts_int_arr, t_int, side='right') - 1
    if idx >= 0:
        return close_arr[idx]
    else:
        return np.nan

def get_price_at_int(sym, t, sym_data, ts_index_map_by_sym_int):
    """
    Devuelve el precio de cierre para el símbolo `sym` en el timestamp `t`.
    Usa diccionario de índices para acceso rápido y fallback acelerado con Numba.
    """
    t_int = int(t)

    # acceso rápido por índice
    idx_map = ts_index_map_by_sym_int[sym]
    if t_int in idx_map:
        return float(sym_data[sym]['close'][idx_map[t_int]])

    # fallback con numba
    ts_arr = sym_data[sym]['ts_int']
    close_arr = sym_data[sym]['close']
    return float(_search_price_numba(ts_arr, close_arr, t_int))


def run_grid_backtest(
    ohlcv_arrays,
    sell_after,
    initial_balance=10000,
    order_amount=100,
    tp_pct=0.0,
    sl_pct=0.0,
    comi_pct=0.05
):

    # -------------------------
    # Configuración inicial
    # -------------------------
    comi_factor = float(comi_pct) / 100.0
    cash        = float(initial_balance)
    num_signals_executed = 0

    # Preparar sym_data con arrays y ts_int
    symbols = list(ohlcv_arrays.keys())
    sym_data = {}
    for sym in symbols:
        data = ohlcv_arrays[sym]
        ts = data['ts']
        ts_int = ts.astype('int64')
        sym_data[sym] = {
            'ts': ts,
            'ts_int': ts_int,
            'close': data['close'],
            'high': data.get('high', None),
            'low': data.get('low', None),
            'signal': data['signal'],
            'len': len(ts)
        }

    # Señales por timestamp
    signals_by_time = {}
    for sym in symbols:
        d = sym_data[sym]
        sig_idxs = np.nonzero(d['signal'])[0]
        ts_int_arr = d['ts_int']
        for idx in sig_idxs:
            t_int = int(ts_int_arr[idx])
            lst = signals_by_time.get(t_int)
            if lst is None:
                signals_by_time[t_int] = [(sym, int(idx))]
            else:
                lst.append((sym, int(idx)))

    # Array ordenado de todos los timestamps
    all_ts_set = set()
    for d in sym_data.values():
        all_ts_set.update(d['ts_int'].tolist())
    all_timestamps_int = np.array(sorted(all_ts_set), dtype=np.int64)
    all_timestamps_dt = all_timestamps_int.astype('datetime64[ns]')

    # -------------------------
    # Estructuras auxiliares
    # -------------------------
    trades = {sym: [] for sym in symbols}
    trade_times = {sym: [] for sym in symbols}

    trade_log_cols = {
        'symbol': [],
        'buy_time': [],
        'buy_price': [],
        'sell_time': [],
        'sell_price': [],
        'qty': [],
        'profit': [],
        'exit_reason': [],
        'commission_buy': [],
        'commission_sell': []
    }

    sim_balance_cols = {
        'timestamp': [],
        'balance': []
    }

    open_positions_heap = []
    counter = 0

    symbol_order = {s: i for i, s in enumerate(symbols)}
    ts_index_map_by_sym_int = {sym: {int(t): idx for idx, t in enumerate(d['ts_int'])} for sym, d in sym_data.items()}

    def close_position(pos, exec_time, exec_price, exit_reason):
        nonlocal cash
        qty = pos['qty']
        buy_price = pos['buy_price']
        commission_buy = pos.get('commission_buy', 0.0)
        commission_sell = (qty * exec_price) * comi_factor if comi_factor != 0.0 else 0.0
        cash += qty * exec_price - commission_sell
        profit = (exec_price - buy_price) * qty - commission_buy - commission_sell
        trades[pos['symbol']].append(profit)
        trade_times[pos['symbol']].append(exec_time)

        for key, val in zip(trade_log_cols.keys(),
                            [pos['symbol'], pos['buy_time'], buy_price, exec_time, exec_price,
                             qty, profit, exit_reason, commission_buy, commission_sell]):
            trade_log_cols[key].append(val)

    # -------------------------
    # Bucle principal
    # -------------------------
    open_heap = open_positions_heap
    sym_data_local = sym_data
    signals_local = signals_by_time
    symbol_order_local = symbol_order

    for t_int in all_timestamps_int:
        # Cerrar posiciones vencidas
        while open_heap and open_heap[0][0] <= t_int:
            _, _, pos = heapq.heappop(open_heap)
            if pos.get('closed', False):
                continue
            sym = pos['symbol']
            sell_ts_int = pos.get('sell_time_int', int(sym_data_local[sym]['ts_int'][-1]))
            exec_price = get_price_at_int(sym, sell_ts_int, sym_data_local, ts_index_map_by_sym_int)
            exec_time_dt = np.datetime64(int(sell_ts_int), 'ns')
            close_position(pos, exec_time_dt, exec_price, 'SELL_AFTER')
            pos['closed'] = True

        # Abrir nuevas posiciones
        events = signals_local.get(int(t_int), [])
        if events:
            events = sorted(events, key=lambda x: symbol_order_local[x[0]])
            for sym, buy_idx in events:
                if cash < order_amount:
                    break
                d = sym_data_local[sym]
                price_t = float(d['close'][buy_idx])
                qty = order_amount / price_t
                commission_buy = order_amount * comi_factor if comi_factor != 0.0 else 0.0
                cash -= (order_amount + commission_buy)
                num_signals_executed += 1

                sell_idx = min(buy_idx + sell_after, d['len'] - 1)
                sell_time_dt = d['ts'][sell_idx]
                sell_time_int = int(d['ts_int'][sell_idx])
                tp_price = price_t * (1.0 + tp_pct / 100.0) if tp_pct != 0.0 else np.inf
                sl_price = price_t * (1.0 - sl_pct / 100.0) if sl_pct != 0.0 else -np.inf

                position = {
                    'symbol': sym,
                    'qty': qty,
                    'buy_price': price_t,
                    'buy_time': np.datetime64(int(t_int), 'ns'),
                    'sell_time': sell_time_dt,
                    'sell_time_int': sell_time_int,
                    'commission_buy': commission_buy
                }

                heapq.heappush(open_heap, (sell_time_int, counter, position))
                counter += 1

        # Registrar balance actual
        positions_value = sum(pos['qty'] * get_price_at_int(pos['symbol'], t_int, sym_data_local, ts_index_map_by_sym_int) 
                              for _, _, pos in open_heap if not pos.get('closed', False))
        sim_balance_cols['timestamp'].append(np.datetime64(int(t_int), 'ns'))
        sim_balance_cols['balance'].append(cash + positions_value)

    # -------------------------
    # Cierre final de posiciones
    # -------------------------
    while open_heap:
        _, _, pos = heapq.heappop(open_heap)
        if pos.get('closed', False):
            continue
        sym = pos['symbol']
        d = sym_data_local[sym]
        sell_ts_int = pos.get('sell_time_int', int(d['ts_int'][-1]))
        exec_price = get_price_at_int(sym, sell_ts_int, sym_data_local, ts_index_map_by_sym_int)
        exec_time_dt = np.datetime64(int(sell_ts_int), 'ns')
        close_position(pos, exec_time_dt, exec_price, 'SELL_AFTER')

    # -------------------------
    # Resultados finales
    # -------------------------
    final_balance = cash
    all_trades = []
    for sym in symbols:
        all_trades.extend(trades[sym])
    num_trades = len(all_trades)
    proportion_winners = np.sum(np.array(all_trades) > 0.0) / num_trades if num_trades > 0 else np.nan

    results = {}
    for sym in symbols:
        results[sym] = {
            'df': None,
            'trades': trades[sym],
            'final_balance': final_balance,
            'num_signals': len(trades[sym]),
            'proportion_winners': (np.nan if len(trades[sym]) == 0 else np.sum(np.array(trades[sym]) > 0.0) / len(trades[sym])),
            'max_dd': 0.0
        }

    results["__PORTFOLIO__"] = {
        'df': None,
        'trades': all_trades,
        'final_balance': final_balance,
        'num_signals': num_signals_executed,
        'proportion_winners': proportion_winners,
        'max_dd': 0.0,
        'sim_balance_history': sim_balance_cols,
        'trade_log': pd.DataFrame(trade_log_cols)
    }

    return results
