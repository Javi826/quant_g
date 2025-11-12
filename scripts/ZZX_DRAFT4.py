"""
Test Exhaustivo para validar la gestión de posiciones SHORT en el backtester.

Valida:
- Cash management (cash_bank y blocked_cash)
- Apertura y cierre de posiciones SHORT
- TP/SL en SHORT (invertidos respecto a LONG)
- Múltiples posiciones simultáneas
- Edge cases y límites de capital
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Importar la función del backtester
from ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE

# =============================================================================
# COLORES PARA OUTPUT
# =============================================================================
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_test(name):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}TEST: {name}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*70}{Colors.END}")

def print_pass(msg):
    print(f"{Colors.GREEN}✓ PASS:{Colors.END} {msg}")

def print_fail(msg):
    print(f"{Colors.RED}✗ FAIL:{Colors.END} {msg}")

def print_info(msg):
    print(f"{Colors.YELLOW}ℹ INFO:{Colors.END} {msg}")

# =============================================================================
# FUNCIONES AUXILIARES
# =============================================================================

def create_synthetic_data(n_candles=100, base_price=100.0, volatility=0.02):
    """Crea datos sintéticos OHLCV con precios controlados."""
    timestamps = pd.date_range(start='2024-01-01', periods=n_candles, freq='1H')
    
    # Precio base con ruido
    closes = base_price + np.cumsum(np.random.randn(n_candles) * volatility * base_price)
    opens = closes + np.random.randn(n_candles) * volatility * base_price * 0.5
    highs = np.maximum(opens, closes) + np.abs(np.random.randn(n_candles)) * volatility * base_price
    lows = np.minimum(opens, closes) - np.abs(np.random.randn(n_candles)) * volatility * base_price
    
    # Timestamps para high/low (simplificado)
    high_times = timestamps.values.astype('datetime64[ns]')
    low_times = timestamps.values.astype('datetime64[ns]')
    
    return {
        'ts': timestamps.values.astype('datetime64[ns]'),
        'open': opens,
        'high': highs,
        'low': lows,
        'close': closes,
        'signal': np.zeros(n_candles),
        'high_time': high_times,
        'low_time': low_times
    }

def validate_trade_log(trade_log, expected_type, test_name):
    """Valida que el trade log tenga la información correcta."""
    errors = []
    
    if trade_log.empty:
        errors.append("Trade log está vacío")
        return errors
    
    # Verificar columnas requeridas
    required_cols = ['symbol', 'buy_time', 'buy_price', 'sell_time', 'sell_price',
                     'qty', 'profit', 'exit_reason', 'commission_buy', 
                     'commission_sell', 'position_type']
    
    missing_cols = [col for col in required_cols if col not in trade_log.columns]
    if missing_cols:
        errors.append(f"Faltan columnas: {missing_cols}")
        return errors
    
    # Verificar tipo de posición
    if not all(trade_log['position_type'] == expected_type):
        errors.append(f"Tipo de posición incorrecto. Esperado: {expected_type}")
    
    # Verificar precios positivos
    if any(trade_log['buy_price'] <= 0) or any(trade_log['sell_price'] <= 0):
        errors.append("Precios negativos o cero detectados")
    
    # Verificar cantidades positivas
    if any(trade_log['qty'] <= 0):
        errors.append("Cantidades negativas o cero detectadas")
    
    # Verificar comisiones
    if any(trade_log['commission_buy'] < 0) or any(trade_log['commission_sell'] < 0):
        errors.append("Comisiones negativas detectadas")
    
    return errors

# =============================================================================
# TEST 1: SHORT BÁSICO CON TP
# =============================================================================
def test_short_basic_tp():
    """Test básico de SHORT que alcanza TP (precio baja)."""
    print_test("SHORT Básico con TP")
    
    # Crear datos donde el precio baja 10%
    data = create_synthetic_data(n_candles=50, base_price=100.0, volatility=0.0)
    # Precio constante excepto después de la señal
    data['close'][:] = 100.0
    data['open'][:] = 100.0
    data['high'][:] = 101.0
    data['low'][:] = 99.0
    
    # Señal SHORT en vela 10
    data['signal'][10] = -1
    
    # Después de la señal, precio baja a 90 (TP debería activarse)
    data['close'][11:] = 90.0
    data['open'][11:] = 90.0
    data['high'][11:] = 91.0
    data['low'][11:] = 89.0
    
    ohlcv = {'TEST': data}
    
    # Ejecutar backtest: SHORT con TP=10%
    results = run_grid_backtest(
        ohlcv_arrays=ohlcv,
        sell_after=20,
        tp_pct=10.0,  # TP al 10% (precio debe bajar a 90)
        sl_pct=5.0,   # SL al 5% (precio sube a 105)
        order_amount=1000.0
    )
    
    trade_log = results['__PORTFOLIO__']['trade_log']
    
    print_info(f"Trades ejecutados: {len(trade_log)}")
    if not trade_log.empty:
        print_info(f"Exit reason: {trade_log['exit_reason'].values[0]}")
        print_info(f"Buy price: {trade_log['buy_price'].values[0]:.2f}")
        print_info(f"Sell price: {trade_log['sell_price'].values[0]:.2f}")
        print_info(f"Profit: {trade_log['profit'].values[0]:.2f}")
    
    # Validaciones
    errors = validate_trade_log(trade_log, 'SHORT', 'SHORT Básico TP')
    
    if not errors:
        # Verificar que se ejecutó 1 trade
        if len(trade_log) != 1:
            print_fail(f"Se esperaba 1 trade, se ejecutaron {len(trade_log)}")
            return False
        
        # Verificar que el exit reason es TP
        if trade_log['exit_reason'].values[0] != 'TP':
            print_fail(f"Exit reason debería ser 'TP', es '{trade_log['exit_reason'].values[0]}'")
            return False
        
        # Verificar profit positivo (vendimos a 100, compramos a ~90)
        profit = trade_log['profit'].values[0]
        if profit <= 0:
            print_fail(f"Profit debería ser positivo en SHORT con TP, es {profit:.2f}")
            return False
        
        # Verificar balance final > inicial
        final_balance = results['__PORTFOLIO__']['final_balance']
        if final_balance <= INITIAL_BALANCE:
            print_fail(f"Balance final ({final_balance:.2f}) debería ser mayor que inicial ({INITIAL_BALANCE})")
            return False
        
        print_pass(f"SHORT con TP ejecutado correctamente. Profit: {profit:.2f}")
        print_pass(f"Balance final: {final_balance:.2f}")
        return True
    else:
        for error in errors:
            print_fail(error)
        return False

# =============================================================================
# TEST 2: SHORT BÁSICO CON SL
# =============================================================================
def test_short_basic_sl():
    """Test básico de SHORT que alcanza SL (precio sube)."""
    print_test("SHORT Básico con SL")
    
    # Crear datos donde el precio sube 5%
    data = create_synthetic_data(n_candles=50, base_price=100.0, volatility=0.0)
    data['close'][:] = 100.0
    data['open'][:] = 100.0
    data['high'][:] = 101.0
    data['low'][:] = 99.0
    
    # Señal SHORT en vela 10
    data['signal'][10] = -1
    
    # Después de la señal, precio sube a 106 (SL debería activarse)
    data['close'][11:] = 106.0
    data['open'][11:] = 106.0
    data['high'][11:] = 107.0
    data['low'][11:] = 105.0
    
    ohlcv = {'TEST': data}
    
    # Ejecutar backtest
    results = run_grid_backtest(
        ohlcv_arrays=ohlcv,
        sell_after=20,
        tp_pct=10.0,
        sl_pct=5.0,  # SL al 5% (precio sube a 105)
        order_amount=1000.0
    )
    
    trade_log = results['__PORTFOLIO__']['trade_log']
    
    print_info(f"Trades ejecutados: {len(trade_log)}")
    if not trade_log.empty:
        print_info(f"Exit reason: {trade_log['exit_reason'].values[0]}")
        print_info(f"Buy price: {trade_log['buy_price'].values[0]:.2f}")
        print_info(f"Sell price: {trade_log['sell_price'].values[0]:.2f}")
        print_info(f"Profit: {trade_log['profit'].values[0]:.2f}")
    
    # Validaciones
    errors = validate_trade_log(trade_log, 'SHORT', 'SHORT Básico SL')
    
    if not errors:
        if len(trade_log) != 1:
            print_fail(f"Se esperaba 1 trade, se ejecutaron {len(trade_log)}")
            return False
        
        if trade_log['exit_reason'].values[0] != 'SL':
            print_fail(f"Exit reason debería ser 'SL', es '{trade_log['exit_reason'].values[0]}'")
            return False
        
        # Verificar profit negativo (vendimos a 100, compramos a ~105)
        profit = trade_log['profit'].values[0]
        if profit >= 0:
            print_fail(f"Profit debería ser negativo en SHORT con SL, es {profit:.2f}")
            return False
        
        final_balance = results['__PORTFOLIO__']['final_balance']
        if final_balance >= INITIAL_BALANCE:
            print_fail(f"Balance final ({final_balance:.2f}) debería ser menor que inicial ({INITIAL_BALANCE})")
            return False
        
        print_pass(f"SHORT con SL ejecutado correctamente. Loss: {profit:.2f}")
        print_pass(f"Balance final: {final_balance:.2f}")
        return True
    else:
        for error in errors:
            print_fail(error)
        return False

# =============================================================================
# TEST 3: CASH MANAGEMENT - BLOCKED CASH
# =============================================================================
def test_cash_management():
    """Verifica que el blocked_cash se gestiona correctamente."""
    print_test("Cash Management - Blocked Cash en SHORT")
    
    # Crear datos con múltiples señales SHORT
    data = create_synthetic_data(n_candles=100, base_price=100.0, volatility=0.0)
    data['close'][:] = 100.0
    data['open'][:] = 100.0
    data['high'][:] = 101.0
    data['low'][:] = 99.0
    
    # 3 señales SHORT espaciadas
    data['signal'][10] = -1
    data['signal'][20] = -1
    data['signal'][30] = -1
    
    # Precio baja después de cada señal (TP)
    data['close'][15:] = 90.0
    data['low'][15:] = 89.0
    
    ohlcv = {'TEST': data}
    
    # Ejecutar con order_amount pequeño para permitir múltiples trades
    results = run_grid_backtest(
        ohlcv_arrays=ohlcv,
        sell_after=10,
        tp_pct=10.0,
        sl_pct=5.0,
        order_amount=2000.0  # Capital suficiente para 3 trades
    )
    
    trade_log = results['__PORTFOLIO__']['trade_log']
    sim_balance = pd.DataFrame(results['__PORTFOLIO__']['sim_balance_history'])
    
    print_info(f"Trades ejecutados: {len(trade_log)}")
    print_info(f"Balance final: {results['__PORTFOLIO__']['final_balance']:.2f}")
    
    # Verificar que se ejecutó solo 1 trade (por el sistema de blocked_cash)
    # El primer SHORT bloquea capital, impidiendo abrir más posiciones simultáneas
    
    if len(trade_log) == 0:
        print_fail("No se ejecutó ningún trade")
        return False
    
    # Verificar que todas las posiciones son SHORT
    if not all(trade_log['position_type'] == 'SHORT'):
        print_fail("No todas las posiciones son SHORT")
        return False
    
    # Verificar que el balance nunca es negativo
    if any(sim_balance['balance'] < 0):
        print_fail("Balance negativo detectado durante la simulación")
        return False
    
    # Calcular profit total esperado
    total_profit = trade_log['profit'].sum()
    balance_change = results['__PORTFOLIO__']['final_balance'] - INITIAL_BALANCE
    
    # Verificar coherencia (considerando pequeños errores de redondeo)
    if abs(total_profit - balance_change) > 0.01:
        print_fail(f"Incoherencia: total_profit={total_profit:.2f}, balance_change={balance_change:.2f}")
        return False
    
    print_pass(f"Cash management correcto. Trades: {len(trade_log)}, Total profit: {total_profit:.2f}")
    print_pass(f"Balance nunca fue negativo. Min: {sim_balance['balance'].min():.2f}")
    return True

# =============================================================================
# TEST 4: MÚLTIPLES SHORTS SIN SIMULTANEIDAD
# =============================================================================
def test_multiple_shorts_sequential():
    """Verifica múltiples SHORTs secuenciales (no simultáneos)."""
    print_test("Múltiples SHORTs Secuenciales")
    
    data = create_synthetic_data(n_candles=100, base_price=100.0, volatility=0.0)
    data['close'][:] = 100.0
    data['open'][:] = 100.0
    data['high'][:] = 101.0
    data['low'][:] = 99.0
    
    # Señales espaciadas para que se cierren antes de la siguiente
    data['signal'][10] = -1  # SHORT 1
    data['signal'][30] = -1  # SHORT 2
    data['signal'][50] = -1  # SHORT 3
    
    # Precios bajan después de cada señal (TP)
    data['close'][15:25] = 90.0
    data['low'][15:25] = 89.0
    
    data['close'][35:45] = 90.0
    data['low'][35:45] = 89.0
    
    data['close'][55:65] = 90.0
    data['low'][55:65] = 89.0
    
    ohlcv = {'TEST': data}
    
    results = run_grid_backtest(
        ohlcv_arrays=ohlcv,
        sell_after=5,
        tp_pct=10.0,
        sl_pct=5.0,
        order_amount=1000.0
    )
    
    trade_log = results['__PORTFOLIO__']['trade_log']
    
    print_info(f"Trades ejecutados: {len(trade_log)}")
    print_info(f"Balance final: {results['__PORTFOLIO__']['final_balance']:.2f}")
    
    if len(trade_log) != 3:
        print_fail(f"Se esperaban 3 trades, se ejecutaron {len(trade_log)}")
        return False
    
    # Verificar que todos son SHORT y con TP
    if not all(trade_log['position_type'] == 'SHORT'):
        print_fail("No todas las posiciones son SHORT")
        return False
    
    if not all(trade_log['exit_reason'] == 'TP'):
        print_fail("No todos los exits son TP")
        return False
    
    # Verificar profit positivo en todos
    if not all(trade_log['profit'] > 0):
        print_fail("Algunos trades tienen profit negativo")
        return False
    
    total_profit = trade_log['profit'].sum()
    print_pass(f"3 SHORTs secuenciales ejecutados correctamente. Total profit: {total_profit:.2f}")
    return True

# =============================================================================
# TEST 5: SHORT SIN SL (DEBE RECHAZARSE)
# =============================================================================
def test_short_without_sl():
    """Verifica que SHORTs sin SL sean rechazados (riesgo ilimitado)."""
    print_test("SHORT sin SL (debe rechazarse)")
    
    data = create_synthetic_data(n_candles=50, base_price=100.0, volatility=0.0)
    data['signal'][10] = -1
    
    ohlcv = {'TEST': data}
    
    results = run_grid_backtest(
        ohlcv_arrays=ohlcv,
        sell_after=20,
        tp_pct=10.0,
        sl_pct=0.0,  # SIN SL
        order_amount=1000.0
    )
    
    trade_log = results['__PORTFOLIO__']['trade_log']
    
    print_info(f"Trades ejecutados: {len(trade_log)}")
    
    # El backtest debería rechazar el SHORT sin SL
    if len(trade_log) > 0:
        print_fail("Se ejecutó SHORT sin SL (debería rechazarse)")
        return False
    
    print_pass("SHORT sin SL correctamente rechazado")
    return True

# =============================================================================
# TEST 6: CAPITAL INSUFICIENTE PARA SHORT
# =============================================================================
def test_insufficient_capital():
    """Verifica que SHORTs con capital insuficiente sean rechazados."""
    print_test("Capital Insuficiente para SHORT")
    
    data = create_synthetic_data(n_candles=50, base_price=100.0, volatility=0.0)
    data['signal'][10] = -1
    
    ohlcv = {'TEST': data}
    
    # Order amount muy grande con SL alto (requiere mucho margen)
    results = run_grid_backtest(
        ohlcv_arrays=ohlcv,
        sell_after=20,
        tp_pct=10.0,
        sl_pct=50.0,  # SL 50% requiere margen = 50% de order_amount
        order_amount=15000.0  # Muy grande para el capital inicial
    )
    
    trade_log = results['__PORTFOLIO__']['trade_log']
    
    print_info(f"Trades ejecutados: {len(trade_log)}")
    
    # No debería ejecutarse por falta de capital
    if len(trade_log) > 0:
        print_fail("Se ejecutó SHORT sin capital suficiente")
        return False
    
    print_pass("SHORT con capital insuficiente correctamente rechazado")
    return True

# =============================================================================
# TEST 7: COMPARACIÓN LONG vs SHORT (MISMA MAGNITUD, SIGNO OPUESTO)
# =============================================================================
def test_long_vs_short_symmetry():
    """Compara LONG y SHORT con el mismo movimiento de precio (simetría)."""
    print_test("Simetría LONG vs SHORT")
    
    # Datos para LONG: precio sube 10%
    data_long = create_synthetic_data(n_candles=50, base_price=100.0, volatility=0.0)
    data_long['close'][:] = 100.0
    data_long['open'][:] = 100.0
    data_long['signal'][10] = 1  # LONG
    data_long['close'][11:] = 110.0
    data_long['high'][11:] = 111.0
    
    # Datos para SHORT: precio baja 10%
    data_short = create_synthetic_data(n_candles=50, base_price=100.0, volatility=0.0)
    data_short['close'][:] = 100.0
    data_short['open'][:] = 100.0
    data_short['signal'][10] = -1  # SHORT
    data_short['close'][11:] = 90.0
    data_short['low'][11:] = 89.0
    
    # Ejecutar LONG
    results_long = run_grid_backtest(
        ohlcv_arrays={'TEST': data_long},
        sell_after=20,
        tp_pct=10.0,
        sl_pct=5.0,
        order_amount=1000.0
    )
    
    # Ejecutar SHORT
    results_short = run_grid_backtest(
        ohlcv_arrays={'TEST': data_short},
        sell_after=20,
        tp_pct=10.0,
        sl_pct=5.0,
        order_amount=1000.0
    )
    
    profit_long = results_long['__PORTFOLIO__']['trade_log']['profit'].values[0]
    profit_short = results_short['__PORTFOLIO__']['trade_log']['profit'].values[0]
    
    print_info(f"Profit LONG: {profit_long:.2f}")
    print_info(f"Profit SHORT: {profit_short:.2f}")
    
    # Los profits deberían ser similares (considerando comisiones)
    # Ambos deberían activar TP y tener profit positivo similar
    if abs(profit_long - profit_short) > 5.0:  # Tolerancia de 5
        print_fail(f"Profits muy diferentes: LONG={profit_long:.2f}, SHORT={profit_short:.2f}")
        return False
    
    if profit_long <= 0 or profit_short <= 0:
        print_fail("Alguno de los profits es negativo o cero")
        return False
    
    print_pass(f"Simetría correcta. Diferencia: {abs(profit_long - profit_short):.2f}")
    return True

# =============================================================================
# TEST 8: BALANCE CONSISTENCY
# =============================================================================
def test_balance_consistency():
    """Verifica que el balance sea consistente durante toda la simulación."""
    print_test("Consistencia del Balance")
    
    data = create_synthetic_data(n_candles=100, base_price=100.0, volatility=0.01)
    
    # Mezcla de señales LONG y SHORT
    data['signal'][10] = 1   # LONG
    data['signal'][30] = -1  # SHORT
    data['signal'][50] = 1   # LONG
    data['signal'][70] = -1  # SHORT
    
    ohlcv = {'TEST': data}
    
    results = run_grid_backtest(
        ohlcv_arrays=ohlcv,
        sell_after=10,
        tp_pct=5.0,
        sl_pct=3.0,
        order_amount=500.0
    )
    
    sim_balance = pd.DataFrame(results['__PORTFOLIO__']['sim_balance_history'])
    trade_log = results['__PORTFOLIO__']['trade_log']
    
    print_info(f"Trades ejecutados: {len(trade_log)}")
    print_info(f"Balance inicial: {INITIAL_BALANCE}")
    print_info(f"Balance final: {results['__PORTFOLIO__']['final_balance']:.2f}")
    print_info(f"Min balance: {sim_balance['balance'].min():.2f}")
    print_info(f"Max balance: {sim_balance['balance'].max():.2f}")
    
    # Verificar que balance nunca es negativo
    if any(sim_balance['balance'] < 0):
        print_fail("Balance negativo detectado")
        return False
    
    # Verificar que balance final = inicial + sum(profits)
    total_profit = trade_log['profit'].sum() if not trade_log.empty else 0
    expected_final = INITIAL_BALANCE + total_profit
    actual_final = results['__PORTFOLIO__']['final_balance']
    
    if abs(expected_final - actual_final) > 0.01:
        print_fail(f"Balance inconsistente. Esperado: {expected_final:.2f}, Actual: {actual_final:.2f}")
        return False
    
    # Verificar que sim_balance es monótono o coherente (no saltos extraños)
    balance_diff = sim_balance['balance'].diff().dropna()
    max_jump = balance_diff.abs().max()
    
    print_info(f"Mayor salto de balance: {max_jump:.2f}")
    
    # Un salto de balance no debería ser mayor que el order_amount (eso sería sospechoso)
    if max_jump > 2000:  # Límite razonable
        print_fail(f"Salto de balance sospechoso: {max_jump:.2f}")
        return False
    
    print_pass("Balance consistente durante toda la simulación")
    return True

# =============================================================================
# EJECUTAR TODOS LOS TESTS
# =============================================================================
def run_all_tests():
    """Ejecuta todos los tests y muestra un resumen."""
    print(f"\n{Colors.BOLD}{'='*70}")
    print(f"INICIO DE TESTS DE VALIDACIÓN - GESTIÓN SHORT")
    print(f"{'='*70}{Colors.END}\n")
    
    tests = [
        ("SHORT Básico con TP", test_short_basic_tp),
        ("SHORT Básico con SL", test_short_basic_sl),
        ("Cash Management", test_cash_management),
        ("Múltiples SHORTs Secuenciales", test_multiple_shorts_sequential),
        ("SHORT sin SL (rechazo)", test_short_without_sl),
        ("Capital Insuficiente", test_insufficient_capital),
        ("Simetría LONG vs SHORT", test_long_vs_short_symmetry),
        ("Consistencia de Balance", test_balance_consistency),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print_fail(f"Exception en {name}: {str(e)}")
            results.append((name, False))
    
    # Resumen
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*70}")
    print(f"RESUMEN DE TESTS")
    print(f"{'='*70}{Colors.END}\n")
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        if result:
            print(f"{Colors.GREEN}✓{Colors.END} {name}")
        else:
            print(f"{Colors.RED}✗{Colors.END} {name}")
    
    print(f"\n{Colors.BOLD}Total: {passed}/{total} tests pasados{Colors.END}")
    
    if passed == total:
        print(f"{Colors.GREEN}{Colors.BOLD}¡TODOS LOS TESTS PASARON!{Colors.END}")
    else:
        print(f"{Colors.RED}{Colors.BOLD}ALGUNOS TESTS FALLARON{Colors.END}")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
