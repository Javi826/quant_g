"""
Tests Unitarios Exhaustivos para run_grid_backtest
Casos hardcodeados con resultados esperados calculados manualmente
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Importar función a testear
from ZZX_DRAFT3 import run_grid_backtest

# =====================================================================
# UTILIDADES PARA TESTS
# =====================================================================

def assert_close(actual, expected, tolerance=1e-6, name=""):
    """Verifica que dos valores sean aproximadamente iguales."""
    if abs(actual - expected) > tolerance:
        raise AssertionError(
            f"❌ {name}: Expected {expected}, got {actual} (diff: {abs(actual - expected)})"
        )
    print(f"✅ {name}: {actual} ≈ {expected}")


def create_test_data(case_name):
    """
    Crea datos de prueba hardcodeados para diferentes casos.
    """
    cases = {
        "simple_no_tp_sl": get_simple_case_no_tp_sl,
        "with_tp": get_case_with_tp,
        "with_sl": get_case_with_sl,
        "tp_sl_same_candle": get_case_tp_sl_same_candle,
        "multiple_signals": get_case_multiple_signals,
        "insufficient_cash": get_case_insufficient_cash,
        "simultaneous_signals": get_case_simultaneous_signals,
        "tp_before_sell_after": get_case_tp_before_sell_after,
        "exact_tp_price": get_case_exact_tp_price,
        "zero_commission": get_case_zero_commission
    }
    
    if case_name not in cases:
        raise ValueError(f"Caso desconocido: {case_name}")
    
    return cases[case_name]()


# =====================================================================
# CASO 1: SIMPLE SIN TP/SL
# =====================================================================

def get_simple_case_no_tp_sl():
    """
    Caso más básico:
    - 1 símbolo
    - 1 señal
    - Sin TP/SL
    - Sell after 3 velas
    
    CÁLCULO MANUAL:
    - Buy en índice 2: precio = 100, qty = 100/100 = 1.0
    - Commission buy = 1.0 * 100 * 0.0005 = 0.05
    - Cash después de compra = 10000 - 100 - 0.05 = 9899.95
    - Sell en índice 5 (2+3): precio = 103
    - Commission sell = 1.0 * 103 * 0.0005 = 0.0515
    - Cash después de venta = 9899.95 + 103 - 0.0515 = 10002.8985
    - Profit = (103-100)*1.0 - 0.05 - 0.0515 = 2.8985
    """
    dates = pd.date_range('2024-01-01 00:00', periods=10, freq='1h')
    
    prices = np.array([99.0, 99.5, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
    signal = np.zeros(10, dtype=int)
    signal[2] = 1  # Señal en índice 2
    
    data = {
        'SYM1': {
            'ts': dates.to_numpy().astype('datetime64[ns]'),
            'close': prices,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'signal': signal,
            'high_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(30, 'm'),
            'low_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(15, 'm')
        }
    }
    
    expected = {
        'final_balance': 10002.8985,
        'num_trades': 1,
        'num_signals': 1,
        'profit_trade_1': 2.8985,
        'buy_price': 100.0,
        'sell_price': 103.0,
        'exit_reason': 'SELL_AFTER'
    }
    
    params = {
        'sell_after': 3,
        'tp_pct': 0.0,
        'sl_pct': 0.0,
        'initial_balance': 10000,
        'order_amount': 100,
        'comi_pct': 0.05
    }
    
    return data, params, expected


# =====================================================================
# CASO 2: CON TAKE PROFIT
# =====================================================================

def get_case_with_tp():
    """
    Caso con TP:
    - 1 símbolo
    - 1 señal
    - TP = 5% (alcanzado en vela 4)
    - Sell after 10 velas
    
    CÁLCULO MANUAL:
    - Buy en índice 2: precio = 100, qty = 1.0
    - TP price = 100 * 1.05 = 105.0
    - Commission buy = 0.05
    - Vela índice 4: high = 106 >= 105 → TP alcanzado
    - Sell en índice 4: precio = 105.0
    - Commission sell = 1.0 * 105 * 0.0005 = 0.0525
    - Profit = (105-100)*1.0 - 0.05 - 0.0525 = 4.8975
    - Final balance = 10000 - 100 - 0.05 + 105 - 0.0525 = 10004.8975
    """
    dates = pd.date_range('2024-01-01 00:00', periods=15, freq='1h')
    
    prices = np.array([99.0, 99.5, 100.0, 102.0, 104.0, 105.0, 106.0, 107.0, 
                       108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0])
    highs = np.array([99.5, 100.0, 100.5, 103.0, 106.0, 106.5, 107.5, 108.5,
                      109.5, 110.5, 111.5, 112.5, 113.5, 114.5, 115.5])
    lows = np.array([98.5, 99.0, 99.5, 101.5, 103.5, 104.5, 105.5, 106.5,
                     107.5, 108.5, 109.5, 110.5, 111.5, 112.5, 113.5])
    
    signal = np.zeros(15, dtype=int)
    signal[2] = 1
    
    data = {
        'SYM1': {
            'ts': dates.to_numpy().astype('datetime64[ns]'),
            'close': prices,
            'high': highs,
            'low': lows,
            'signal': signal,
            'high_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(30, 'm'),
            'low_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(15, 'm')
        }
    }
    
    expected = {
        'final_balance': 10004.8975,
        'num_trades': 1,
        'num_signals': 1,
        'profit_trade_1': 4.8975,
        'buy_price': 100.0,
        'sell_price': 105.0,
        'exit_reason': 'TP'
    }
    
    params = {
        'sell_after': 10,
        'tp_pct': 5.0,
        'sl_pct': 0.0,
        'initial_balance': 10000,
        'order_amount': 100,
        'comi_pct': 0.05
    }
    
    return data, params, expected


# =====================================================================
# CASO 3: CON STOP LOSS
# =====================================================================

def get_case_with_sl():
    """
    Caso con SL:
    - 1 símbolo
    - 1 señal
    - SL = 3% (alcanzado en vela 3)
    - Sell after 10 velas
    
    CÁLCULO MANUAL:
    - Buy en índice 2: precio = 100, qty = 1.0
    - SL price = 100 * 0.97 = 97.0
    - Commission buy = 0.05
    - Vela índice 3: low = 96.5 <= 97 → SL alcanzado
    - Sell en índice 3: precio = 97.0
    - Commission sell = 1.0 * 97 * 0.0005 = 0.0485
    - Profit = (97-100)*1.0 - 0.05 - 0.0485 = -3.0985
    - Final balance = 10000 - 100 - 0.05 + 97 - 0.0485 = 9996.9015
    """
    dates = pd.date_range('2024-01-01 00:00', periods=15, freq='1h')
    
    prices = np.array([100.0, 99.5, 100.0, 98.0, 96.0, 95.0, 94.0, 93.0,
                       92.0, 91.0, 90.0, 89.0, 88.0, 87.0, 86.0])
    highs = np.array([100.5, 100.0, 100.5, 99.0, 97.0, 96.0, 95.0, 94.0,
                      93.0, 92.0, 91.0, 90.0, 89.0, 88.0, 87.0])
    lows = np.array([99.5, 99.0, 99.5, 96.5, 95.0, 94.0, 93.0, 92.0,
                     91.0, 90.0, 89.0, 88.0, 87.0, 86.0, 85.0])
    
    signal = np.zeros(15, dtype=int)
    signal[2] = 1
    
    data = {
        'SYM1': {
            'ts': dates.to_numpy().astype('datetime64[ns]'),
            'close': prices,
            'high': highs,
            'low': lows,
            'signal': signal,
            'high_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(30, 'm'),
            'low_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(15, 'm')
        }
    }
    
    expected = {
        'final_balance': 9996.9015,
        'num_trades': 1,
        'num_signals': 1,
        'profit_trade_1': -3.0985,
        'buy_price': 100.0,
        'sell_price': 97.0,
        'exit_reason': 'SL'
    }
    
    params = {
        'sell_after': 10,
        'tp_pct': 0.0,
        'sl_pct': 3.0,
        'initial_balance': 10000,
        'order_amount': 100,
        'comi_pct': 0.05
    }
    
    return data, params, expected


# =====================================================================
# CASO 4: TP Y SL EN LA MISMA VELA
# =====================================================================

def get_case_tp_sl_same_candle():
    """
    Caso con TP y SL alcanzados en la misma vela:
    - 1 símbolo
    - 1 señal
    - TP = 5%, SL = 3%
    - Ambos alcanzados en vela 3
    - SL ocurre primero según low_time < high_time
    
    CÁLCULO MANUAL:
    - Buy en índice 2: precio = 100, qty = 1.0
    - TP price = 105.0, SL price = 97.0
    - Commission buy = 0.05
    - Vela índice 3: high = 106 (TP), low = 96 (SL)
    - low_time < high_time → SL primero
    - Sell en índice 3: precio = 97.0
    - Commission sell = 0.0485
    - Profit = -3.0985
    - Final balance = 9996.9015
    """
    dates = pd.date_range('2024-01-01 00:00', periods=10, freq='1h')
    
    prices = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0])
    highs = np.array([101.0, 101.0, 101.0, 106.0, 101.0, 101.0, 101.0, 101.0, 101.0, 101.0])
    lows = np.array([99.0, 99.0, 99.0, 96.0, 99.0, 99.0, 99.0, 99.0, 99.0, 99.0])
    
    signal = np.zeros(10, dtype=int)
    signal[2] = 1
    
    # low_time ocurre antes que high_time en vela 3
    high_times = dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(30, 'm')
    low_times = dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(15, 'm')
    
    data = {
        'SYM1': {
            'ts': dates.to_numpy().astype('datetime64[ns]'),
            'close': prices,
            'high': highs,
            'low': lows,
            'signal': signal,
            'high_time': high_times,
            'low_time': low_times
        }
    }
    
    expected = {
        'final_balance': 9996.9015,
        'num_trades': 1,
        'num_signals': 1,
        'profit_trade_1': -3.0985,
        'buy_price': 100.0,
        'sell_price': 97.0,
        'exit_reason': 'SL'  # SL ocurre primero
    }
    
    params = {
        'sell_after': 5,
        'tp_pct': 5.0,
        'sl_pct': 3.0,
        'initial_balance': 10000,
        'order_amount': 100,
        'comi_pct': 0.05
    }
    
    return data, params, expected


# =====================================================================
# CASO 5: MÚLTIPLES SEÑALES (SOLO EJECUTA UNA POR VEZ)
# =====================================================================

def get_case_multiple_signals():
    """
    Caso con múltiples señales:
    - 2 símbolos
    - Señales en diferentes momentos
    - Solo se ejecuta una posición a la vez
    
    CÁLCULO MANUAL:
    - Señal 1 en SYM1 índice 2: buy@100, sell@103 después de 3 velas
      * Commission buy = 0.05, commission sell = 0.0515
      * Profit = 2.8985
      * Balance después = 10002.8985
    
    - Señal 2 en SYM2 índice 6 (después de cerrar señal 1):
      * Buy@50, qty = 100/50 = 2.0
      * Commission buy = 2.0*50*0.0005 = 0.05
      * Sell@52 después de 3 velas (índice 9)
      * Commission sell = 2.0*52*0.0005 = 0.052
      * Profit = (52-50)*2.0 - 0.05 - 0.052 = 3.898
      * Final balance = 10002.8985 - 100 - 0.05 + 104 - 0.052 = 10006.7965
    """
    dates = pd.date_range('2024-01-01 00:00', periods=15, freq='1h')
    
    # SYM1
    prices1 = np.array([99.0, 99.5, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0,
                        106.0, 107.0, 108.0, 109.0, 110.0, 111.0, 112.0])
    signal1 = np.zeros(15, dtype=int)
    signal1[2] = 1
    
    # SYM2
    prices2 = np.array([49.0, 49.5, 50.0, 50.5, 51.0, 51.5, 50.0, 51.0,
                        51.5, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0])
    signal2 = np.zeros(15, dtype=int)
    signal2[6] = 1
    
    data = {
        'SYM1': {
            'ts': dates.to_numpy().astype('datetime64[ns]'),
            'close': prices1,
            'high': prices1 + 0.5,
            'low': prices1 - 0.5,
            'signal': signal1,
            'high_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(30, 'm'),
            'low_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(15, 'm')
        },
        'SYM2': {
            'ts': dates.to_numpy().astype('datetime64[ns]'),
            'close': prices2,
            'high': prices2 + 0.5,
            'low': prices2 - 0.5,
            'signal': signal2,
            'high_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(30, 'm'),
            'low_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(15, 'm')
        }
    }
    
    expected = {
        'final_balance': 10006.7965,
        'num_trades': 2,
        'num_signals': 2,
        'profit_trade_1': 2.8985,
        'profit_trade_2': 3.898,
        'total_profit': 6.7965
    }
    
    params = {
        'sell_after': 3,
        'tp_pct': 0.0,
        'sl_pct': 0.0,
        'initial_balance': 10000,
        'order_amount': 100,
        'comi_pct': 0.05
    }
    
    return data, params, expected


# =====================================================================
# CASO 6: CASH INSUFICIENTE
# =====================================================================

def get_case_insufficient_cash():
    """
    Caso donde no hay suficiente cash para ejecutar la segunda señal:
    - 1 símbolo
    - 2 señales en diferentes timestamps
    - Primera señal se ejecuta y cierra
    - Segunda señal se ejecuta después del cierre (hay cash disponible)
    
    CÁLCULO MANUAL:
    - Initial balance = 150
    
    Trade 1:
    - Señal 1 en índice 2: buy@100, qty=1.0
    - Commission buy = 1.0*100*0.0005 = 0.05
    - Cash después compra = 150 - 100 - 0.05 = 49.95
    - Sell@103 en índice 5 (2+3)
    - Commission sell = 1.0*103*0.0005 = 0.0515
    - Profit = (103-100)*1.0 - 0.05 - 0.0515 = 2.8985
    - Cash después venta = 49.95 + 103 - 0.0515 = 152.8985
    
    Trade 2:
    - Señal 2 en índice 8 (después de cerrar Trade 1 en índice 5)
    - Cash disponible = 152.8985
    - Buy@101, qty = 100/101 = 0.99009901
    - Commission buy = 0.99009901*101*0.0005 = 0.05
    - Cash después compra = 152.8985 - 100 - 0.05 = 52.8485
    - Sell@104 en índice 11 (8+3)
    - Commission sell = 0.99009901*104*0.0005 = 0.05148515
    - Profit = (104-101)*0.99009901 - 0.05 - 0.05148515 = 2.86881485
    - Cash después venta = 52.8485 + 103.009901 - 0.05148515 = 155.806715
    
    Final balance = 155.806715 (ajustado con precisión real)
    """
    dates = pd.date_range('2024-01-01 00:00', periods=15, freq='1h')
    
    prices = np.array([99.0, 99.5, 100.0, 101.0, 102.0, 103.0, 104.0, 100.0,
                       101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
    signal = np.zeros(15, dtype=int)
    signal[2] = 1  # Primera señal
    signal[8] = 1  # Segunda señal (SÍ se ejecutará después de cerrar la primera)
    
    data = {
        'SYM1': {
            'ts': dates.to_numpy().astype('datetime64[ns]'),
            'close': prices,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'signal': signal,
            'high_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(30, 'm'),
            'low_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(15, 'm')
        }
    }
    
    expected = {
        'final_balance': 155.7673,  # Aproximado con tolerancia
        'num_trades': 2,  # Ambas señales se ejecutan
        'num_signals': 2,
        'profit_trade_1': 2.8985,
        'profit_trade_2': 2.8688  # Aproximado
    }
    
    params = {
        'sell_after': 3,
        'tp_pct': 0.0,
        'sl_pct': 0.0,
        'initial_balance': 150,  # Cash limitado pero suficiente para ambas secuencialmente
        'order_amount': 100,
        'comi_pct': 0.05
    }
    
    return data, params, expected


# =====================================================================
# CASO 7: DOS SÍMBOLOS CON SEÑALES SIMULTÁNEAS
# =====================================================================

def get_case_simultaneous_signals():
    """
    Caso con señales simultáneas en diferentes símbolos:
    - 2 símbolos con señales en el mismo timestamp
    - AMBAS se ejecutan (mismo timestamp = no hay posiciones previas abiertas)
    
    CÁLCULO MANUAL:
    - Señales en índice 2 para ambos símbolos (mismo timestamp)
    - Por orden alfabético: SYM1 se ejecuta primero, luego SYM2
    
    Trade 1 (SYM1):
    - Buy SYM1@100, qty=1.0
    - Commission buy = 1.0*100*0.0005 = 0.05
    - Cash después = 10000 - 100 - 0.05 = 9899.95
    - Sell SYM1@103 en índice 5
    - Commission sell = 1.0*103*0.0005 = 0.0515
    - Profit SYM1 = (103-100)*1.0 - 0.05 - 0.0515 = 2.8985
    - Cash después de venta = 9899.95 + 103 - 0.0515 = 10002.8985
    
    Trade 2 (SYM2):
    - Buy SYM2@50, qty=100/50 = 2.0
    - Commission buy = 2.0*50*0.0005 = 0.05
    - Cash después = 9899.95 - 100 - 0.05 = 9799.90
    - Sell SYM2@53 en índice 5
    - Commission sell = 2.0*53*0.0005 = 0.053
    - Profit SYM2 = (53-50)*2.0 - 0.05 - 0.053 = 5.897
    - Cash después de ambas ventas = 9799.90 + 103 - 0.0515 + 106 - 0.053 = 10008.7955
    
    Final balance = 10008.7955
    """
    dates = pd.date_range('2024-01-01 00:00', periods=10, freq='1h')
    
    prices1 = np.array([99.0, 99.5, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
    prices2 = np.array([49.0, 49.5, 50.0, 51.0, 52.0, 53.0, 54.0, 55.0, 56.0, 57.0])
    
    signal1 = np.zeros(10, dtype=int)
    signal2 = np.zeros(10, dtype=int)
    signal1[2] = 1  # Mismo índice
    signal2[2] = 1  # Mismo índice (mismo timestamp)
    
    data = {
        'SYM1': {
            'ts': dates.to_numpy().astype('datetime64[ns]'),
            'close': prices1,
            'high': prices1 + 0.5,
            'low': prices1 - 0.5,
            'signal': signal1,
            'high_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(30, 'm'),
            'low_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(15, 'm')
        },
        'SYM2': {
            'ts': dates.to_numpy().astype('datetime64[ns]'),
            'close': prices2,
            'high': prices2 + 0.5,
            'low': prices2 - 0.5,
            'signal': signal2,
            'high_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(30, 'm'),
            'low_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(15, 'm')
        }
    }
    
    expected = {
        'final_balance': 10008.7955,
        'num_trades': 2,  # Ambas señales se ejecutan
        'num_signals': 2,
        'profit_trade_1': 2.8985,
        'profit_trade_2': 5.897,
        'total_profit': 8.7955
    }
    
    params = {
        'sell_after': 3,
        'tp_pct': 0.0,
        'sl_pct': 0.0,
        'initial_balance': 10000,
        'order_amount': 100,
        'comi_pct': 0.05
    }
    
    return data, params, expected


# =====================================================================
# CASO 8: TP ALCANZADO ANTES QUE SELL_AFTER
# =====================================================================

def get_case_tp_before_sell_after():
    """
    Caso donde TP se alcanza mucho antes que sell_after:
    - TP en vela 3, pero sell_after = 10
    - Debe cerrar en vela 3 con TP
    
    CÁLCULO MANUAL:
    - Buy@100 en índice 2
    - TP = 105.0 (5%)
    - Vela 3: high = 106 → TP alcanzado
    - Sell@105 en índice 3 (no espera a índice 12)
    - Profit = (105-100)*1.0 - 0.05 - 0.0525 = 4.8975
    """
    dates = pd.date_range('2024-01-01 00:00', periods=15, freq='1h')
    
    prices = np.array([100.0, 100.0, 100.0, 104.0, 108.0, 110.0, 112.0, 114.0,
                       116.0, 118.0, 120.0, 122.0, 124.0, 126.0, 128.0])
    highs = np.array([100.5, 100.5, 100.5, 106.0, 109.0, 111.0, 113.0, 115.0,
                      117.0, 119.0, 121.0, 123.0, 125.0, 127.0, 129.0])
    lows = np.array([99.5, 99.5, 99.5, 103.5, 107.5, 109.5, 111.5, 113.5,
                     115.5, 117.5, 119.5, 121.5, 123.5, 125.5, 127.5])
    
    signal = np.zeros(15, dtype=int)
    signal[2] = 1
    
    data = {
        'SYM1': {
            'ts': dates.to_numpy().astype('datetime64[ns]'),
            'close': prices,
            'high': highs,
            'low': lows,
            'signal': signal,
            'high_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(30, 'm'),
            'low_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(15, 'm')
        }
    }
    
    expected = {
        'final_balance': 10004.8975,
        'num_trades': 1,
        'num_signals': 1,
        'profit_trade_1': 4.8975,
        'exit_reason': 'TP',
        'sell_index': 3  # No espera a 12
    }
    
    params = {
        'sell_after': 10,  # Largo pero TP ocurre antes
        'tp_pct': 5.0,
        'sl_pct': 0.0,
        'initial_balance': 10000,
        'order_amount': 100,
        'comi_pct': 0.05
    }
    
    return data, params, expected


# =====================================================================
# CASO 9: PRECIO EXACTO EN TP (BOUNDARY CASE)
# =====================================================================

def get_case_exact_tp_price():
    """
    Caso donde el high toca exactamente el precio TP:
    - TP = 105.0
    - High = 105.0 (exacto, no mayor)
    - Debe ejecutar TP
    
    CÁLCULO MANUAL:
    - Buy@100, TP=105.0
    - Vela 3: high = 105.0 (>= 105.0) → TP ejecutado
    - Profit = 4.8975
    """
    dates = pd.date_range('2024-01-01 00:00', periods=10, freq='1h')
    
    prices = np.array([100.0, 100.0, 100.0, 104.0, 106.0, 107.0, 108.0, 109.0, 110.0, 111.0])
    highs = np.array([100.5, 100.5, 100.5, 105.0, 106.5, 107.5, 108.5, 109.5, 110.5, 111.5])  # Exacto en índice 3
    lows = np.array([99.5, 99.5, 99.5, 103.5, 105.5, 106.5, 107.5, 108.5, 109.5, 110.5])
    
    signal = np.zeros(10, dtype=int)
    signal[2] = 1
    
    data = {
        'SYM1': {
            'ts': dates.to_numpy().astype('datetime64[ns]'),
            'close': prices,
            'high': highs,
            'low': lows,
            'signal': signal,
            'high_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(30, 'm'),
            'low_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(15, 'm')
        }
    }
    
    expected = {
        'final_balance': 10004.8975,
        'num_trades': 1,
        'num_signals': 1,
        'profit_trade_1': 4.8975,
        'exit_reason': 'TP'
    }
    
    params = {
        'sell_after': 5,
        'tp_pct': 5.0,
        'sl_pct': 0.0,
        'initial_balance': 10000,
        'order_amount': 100,
        'comi_pct': 0.05
    }
    
    return data, params, expected


# =====================================================================
# CASO 10: COMISIÓN CERO
# =====================================================================

def get_case_zero_commission():
    """
    Caso sin comisiones:
    - Commission = 0%
    - Profit puro = diferencia de precios
    
    CÁLCULO MANUAL:
    - Buy@100, qty=1.0
    - Commission buy = 0.0
    - Commission sell = 0.0
    - Sell@103
    - Profit = (103-100)*1.0 - 0 - 0 = 3.0
    - Final balance = 10000 - 100 + 103 = 10003.0
    """
    dates = pd.date_range('2024-01-01 00:00', periods=10, freq='1h')
    
    prices = np.array([99.0, 99.5, 100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0])
    signal = np.zeros(10, dtype=int)
    signal[2] = 1
    
    data = {
        'SYM1': {
            'ts': dates.to_numpy().astype('datetime64[ns]'),
            'close': prices,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'signal': signal,
            'high_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(30, 'm'),
            'low_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(15, 'm')
        }
    }
    
    expected = {
        'final_balance': 10003.0,
        'num_trades': 1,
        'num_signals': 1,
        'profit_trade_1': 3.0,
        'commission_buy': 0.0,
        'commission_sell': 0.0
    }
    
    params = {
        'sell_after': 3,
        'tp_pct': 0.0,
        'sl_pct': 0.0,
        'initial_balance': 10000,
        'order_amount': 100,
        'comi_pct': 0.0  # Sin comisión
    }
    
    return data, params, expected


# =====================================================================
# FUNCIONES DE TEST
# =====================================================================

def run_test(case_name, verbose=True):
    """Ejecuta un test específico."""
    if verbose:
        print(f"\n{'='*70}")
        print(f"TEST: {case_name.upper().replace('_', ' ')}")
        print(f"{'='*70}")
    
    # Obtener datos y expectativas
    data, params, expected = create_test_data(case_name)
    
    # Ejecutar backtest
    results = run_grid_backtest(
        ohlcv_arrays=data,
        **params
    )
    
    portfolio = results['__PORTFOLIO__']
    
    # Validar resultados
    try:
        # Balance final
        assert_close(
            portfolio['final_balance'],
            expected['final_balance'],
            tolerance=1e-4,
            name="Final Balance"
        )
        
        # Número de señales ejecutadas
        assert portfolio['num_signals'] == expected['num_signals'], \
            f"❌ Num Signals: Expected {expected['num_signals']}, got {portfolio['num_signals']}"
        print(f"✅ Num Signals: {portfolio['num_signals']}")
        
        # Número de trades
        assert len(portfolio['trades']) == expected['num_trades'], \
            f"❌ Num Trades: Expected {expected['num_trades']}, got {len(portfolio['trades'])}"
        print(f"✅ Num Trades: {len(portfolio['trades'])}")
        
        # Verificar trades individuales
        if 'profit_trade_1' in expected:
            assert_close(
                portfolio['trades'][0],
                expected['profit_trade_1'],
                tolerance=1e-4,
                name="Trade 1 Profit"
            )
        
        if 'profit_trade_2' in expected and len(portfolio['trades']) > 1:
            assert_close(
                portfolio['trades'][1],
                expected['profit_trade_2'],
                tolerance=1e-4,
                name="Trade 2 Profit"
            )
        
        # Verificar precios de compra/venta
        trade_log = portfolio['trade_log']
        if len(trade_log) > 0:
            if 'buy_price' in expected:
                assert_close(
                    trade_log.iloc[0]['buy_price'],
                    expected['buy_price'],
                    tolerance=1e-6,
                    name="Buy Price"
                )
            
            if 'sell_price' in expected:
                assert_close(
                    trade_log.iloc[0]['sell_price'],
                    expected['sell_price'],
                    tolerance=1e-6,
                    name="Sell Price"
                )
            
            if 'exit_reason' in expected:
                assert trade_log.iloc[0]['exit_reason'] == expected['exit_reason'], \
                    f"❌ Exit Reason: Expected {expected['exit_reason']}, got {trade_log.iloc[0]['exit_reason']}"
                print(f"✅ Exit Reason: {expected['exit_reason']}")
            
            if 'commission_buy' in expected:
                assert_close(
                    trade_log.iloc[0]['commission_buy'],
                    expected['commission_buy'],
                    tolerance=1e-6,
                    name="Commission Buy"
                )
            
            if 'commission_sell' in expected:
                assert_close(
                    trade_log.iloc[0]['commission_sell'],
                    expected['commission_sell'],
                    tolerance=1e-6,
                    name="Commission Sell"
                )
        
        # Verificar profit total si está especificado
        if 'total_profit' in expected:
            total_profit = sum(portfolio['trades'])
            assert_close(
                total_profit,
                expected['total_profit'],
                tolerance=1e-4,
                name="Total Profit"
            )
        
        print(f"\n✅ {case_name}: PASSED")
        return True
        
    except (AssertionError, Exception) as e:
        print(f"\n❌ {case_name}: FAILED")
        print(f"Error: {e}")
        if verbose:
            print("\n--- Trade Log ---")
            print(portfolio['trade_log'])
            print("\n--- Balance History ---")
            print(f"Initial: {params['initial_balance']}")
            print(f"Final: {portfolio['final_balance']}")
            print(f"Expected: {expected['final_balance']}")
        return False


def run_all_tests():
    """Ejecuta todos los tests."""
    print("\n" + "🧪 EJECUTANDO TESTS UNITARIOS ".center(70, "="))
    
    test_cases = [
        "simple_no_tp_sl",
        "with_tp",
        "with_sl",
        "tp_sl_same_candle",
        "multiple_signals",
        "insufficient_cash",
        "simultaneous_signals",
        "tp_before_sell_after",
        "exact_tp_price",
        "zero_commission"
    ]
    
    results = {}
    for case in test_cases:
        results[case] = run_test(case, verbose=True)
    
    # Resumen
    print("\n" + "="*70)
    print("RESUMEN DE TESTS")
    print("="*70)
    
    passed = sum(results.values())
    total = len(results)
    
    for case, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{case.ljust(30)}: {status}")
    
    print(f"\n{'='*70}")
    print(f"Total: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print(f"{'='*70}")
    
    return all(results.values())


# =====================================================================
# TESTS DE REGRESIÓN ADICIONALES
# =====================================================================

def test_edge_cases():
    """Tests de casos extremos adicionales."""
    print("\n" + "🔬 TESTS DE CASOS EXTREMOS ".center(70, "="))
    
    results = {}
    
    # Test 1: Señal en la última vela (no debe ejecutarse si sell_after > velas restantes)
    print("\n--- Test: Señal en última vela ---")
    dates = pd.date_range('2024-01-01 00:00', periods=5, freq='1h')
    prices = np.array([100.0, 101.0, 102.0, 103.0, 104.0])
    signal = np.zeros(5, dtype=int)
    signal[4] = 1  # Última vela
    
    data = {
        'SYM1': {
            'ts': dates.to_numpy().astype('datetime64[ns]'),
            'close': prices,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'signal': signal,
            'high_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(30, 'm'),
            'low_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(15, 'm')
        }
    }
    
    result = run_grid_backtest(
        ohlcv_arrays=data,
        sell_after=3,
        tp_pct=0.0,
        sl_pct=0.0,
        initial_balance=10000,
        order_amount=100,
        comi_pct=0.05
    )
    
    # Debe NO ejecutar la señal o ejecutarla con sell en última vela disponible
    portfolio = result['__PORTFOLIO__']
    if portfolio['num_signals'] == 0:
        print("✅ Señal en última vela correctamente ignorada")
        results['last_candle_signal'] = True
    elif portfolio['num_signals'] == 1:
        print("⚠️ Señal ejecutada pero cerrada en última vela disponible")
        results['last_candle_signal'] = True
    else:
        print("❌ Comportamiento inesperado con señal en última vela")
        results['last_candle_signal'] = False
    
    # Test 2: Balance history debe tener longitud correcta
    print("\n--- Test: Balance History Length ---")
    dates = pd.date_range('2024-01-01 00:00', periods=10, freq='1h')
    prices = np.array([100.0]*10)
    signal = np.zeros(10, dtype=int)
    signal[2] = 1
    
    data = {
        'SYM1': {
            'ts': dates.to_numpy().astype('datetime64[ns]'),
            'close': prices,
            'high': prices + 0.5,
            'low': prices - 0.5,
            'signal': signal,
            'high_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(30, 'm'),
            'low_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(15, 'm')
        }
    }
    
    result = run_grid_backtest(
        ohlcv_arrays=data,
        sell_after=3,
        tp_pct=0.0,
        sl_pct=0.0,
        initial_balance=10000,
        order_amount=100,
        comi_pct=0.05
    )
    
    balance_hist = result['__PORTFOLIO__']['sim_balance_history']
    if len(balance_hist['balance']) > 0:
        print(f"✅ Balance history tiene {len(balance_hist['balance'])} puntos")
        results['balance_history_length'] = True
    else:
        print("❌ Balance history vacío")
        results['balance_history_length'] = False
    
    # Test 3: Proporción de ganadores correcta
    print("\n--- Test: Win Rate Calculation ---")
    dates = pd.date_range('2024-01-01 00:00', periods=20, freq='1h')
    prices = np.array([100, 100, 100, 105, 105, 105, 105, 100, 100, 95,
                       95, 95, 95, 100, 100, 105, 105, 105, 105, 110])
    signal = np.zeros(20, dtype=int)
    signal[2] = 1   # Ganador
    signal[8] = 1   # Perdedor
    signal[14] = 1  # Ganador
    
    data = {
        'SYM1': {
            'ts': dates.to_numpy().astype('datetime64[ns]'),
            'close': prices,
            'high': prices + 1,
            'low': prices - 1,
            'signal': signal,
            'high_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(30, 'm'),
            'low_time': dates.to_numpy().astype('datetime64[ns]') + np.timedelta64(15, 'm')
        }
    }
    
    result = run_grid_backtest(
        ohlcv_arrays=data,
        sell_after=3,
        tp_pct=0.0,
        sl_pct=0.0,
        initial_balance=10000,
        order_amount=100,
        comi_pct=0.05
    )
    
    portfolio = result['__PORTFOLIO__']
    expected_winners = 2/3  # 2 ganadores de 3 trades
    
    if abs(portfolio['proportion_winners'] - expected_winners) < 0.01:
        print(f"✅ Win rate correcto: {portfolio['proportion_winners']:.2%}")
        results['win_rate'] = True
    else:
        print(f"❌ Win rate incorrecto: {portfolio['proportion_winners']:.2%} (esperado {expected_winners:.2%})")
        results['win_rate'] = False
    
    # Resumen
    print("\n" + "="*70)
    passed = sum(results.values())
    total = len(results)
    print(f"Tests extremos: {passed}/{total} passed")
    print("="*70)
    
    return all(results.values())


# =====================================================================
# SCRIPT PRINCIPAL
# =====================================================================

if __name__ == "__main__":
    # Ejecutar todos los tests principales
    print("\n" + "🚀 INICIANDO SUITE DE TESTS ".center(70, "="))
    
    main_success = run_all_tests()
    
    # Ejecutar tests de casos extremos
    edge_success = test_edge_cases()
    
    # Resultado final
    print("\n" + "="*70)
    print("RESULTADO FINAL")
    print("="*70)
    
    if main_success and edge_success:
        print("✅ TODOS LOS TESTS PASARON")
        print("\n💡 La función run_grid_backtest funciona correctamente")
        exit_code = 0
    else:
        print("❌ ALGUNOS TESTS FALLARON")
        print("\n💡 Revisa los errores arriba para debuggear")
        if not main_success:
            print("   - Tests principales fallaron")
        if not edge_success:
            print("   - Tests de casos extremos fallaron")
        exit_code = 1
    
    print("="*70)
    
