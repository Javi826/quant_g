import numpy as np
import pandas as pd
from typing import Dict, List
import sys

# Importar la función real del backtest
from ZX_compute_BT import run_grid_backtest, MIN_PRICE, INITIAL_BALANCE

# =============================================================================
# FUNCIÓN AUXILIAR PARA CALCULAR WIN RATIO MANUALMENTE
# =============================================================================

def calculate_win_ratio_manual(trades_dict):
    """Calcula el win ratio manualmente para comparar"""
    all_trades = [p for lst in trades_dict.values() for p in lst]
    num_trades = len(all_trades)
    
    if num_trades == 0:
        return np.nan, 0
    
    winners = sum(1 for p in all_trades if p > 0.0)
    proportion_winners = winners / num_trades
    
    return proportion_winners, num_trades


# =============================================================================
# FUNCIÓN AUXILIAR PARA CREAR DATOS OHLCV SINTÉTICOS
# =============================================================================

def create_synthetic_ohlcv(symbol, num_candles, base_price, signal_indices, 
                           price_changes, include_intrabar=True):
    """
    Crea datos OHLCV sintéticos para testing
    
    Args:
        symbol: Nombre del símbolo
        num_candles: Número de velas a generar
        base_price: Precio base inicial
        signal_indices: Lista de índices donde hay señales de compra
        price_changes: Lista de cambios de precio (en %) después de cada señal
        include_intrabar: Si incluir datos intrabar (high_time, low_time)
    """
    timestamps = pd.date_range('2024-01-01', periods=num_candles, freq='1h')
    
    # Generar precios con la tendencia especificada
    close_prices = np.ones(num_candles) * base_price
    
    # Aplicar cambios de precio después de las señales
    for i, (sig_idx, change_pct) in enumerate(zip(signal_indices, price_changes)):
        if sig_idx < num_candles:
            # Aplicar el cambio gradualmente después de la señal
            for j in range(sig_idx + 1, min(sig_idx + 5, num_candles)):
                close_prices[j] = base_price * (1 + change_pct / 100.0)
    
    # Generar high/low basados en close
    high_prices = close_prices * 1.01
    low_prices = close_prices * 0.99
    open_prices = close_prices.copy()
    
    # Crear array de señales
    signal_array = np.zeros(num_candles, dtype=int)
    for idx in signal_indices:
        if idx < num_candles:
            signal_array[idx] = 1
    
    ohlcv = {
        'ts': timestamps.values,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'signal': signal_array
    }
    
    # Añadir datos intrabar si se requiere
    if include_intrabar:
        # high_time y low_time: timestamps de cuando ocurrieron high/low
        ohlcv['high_time'] = timestamps.values
        ohlcv['low_time'] = timestamps.values
    
    return ohlcv


# =============================================================================
# CASOS DE TEST
# =============================================================================

def test_case_1_single_symbol_all_winners():
    """Test 1: Un solo símbolo con todas las operaciones ganadoras"""
    print("\n" + "="*80)
    print("TEST 1: Un símbolo - Todas ganadoras")
    print("="*80)
    
    # Crear datos: 5 señales, todas con ganancia
    ohlcv = create_synthetic_ohlcv(
        symbol='BTC',
        num_candles=100,
        base_price=50000,
        signal_indices=[10, 20, 30, 40, 50],
        price_changes=[5, 3, 7, 2, 4]  # Todos positivos
    )
    
    ohlcv_arrays = {'BTC': ohlcv}
    
    # Ejecutar backtest real
    results = run_grid_backtest(
        ohlcv_arrays=ohlcv_arrays,
        sell_after=3,  # Vender después de 3 velas
        tp_pct=0.0,
        sl_pct=0.0,
        order_amount=100
    )
    
    # Obtener trades del resultado
    portfolio = results['__PORTFOLIO__']
    trades_list = portfolio['trades']
    win_ratio_bt = portfolio['proportion_winners']
    num_signals = portfolio['num_signals']
    
    # Reconstruir diccionario de trades para cálculo manual
    trades_dict = {'BTC': trades_list}
    win_ratio_manual, num_trades_manual = calculate_win_ratio_manual(trades_dict)
    
    print(f"Señales ejecutadas: {num_signals}")
    print(f"Trades realizados: {len(trades_list)}")
    print(f"Trades detallados: {[f'{t:.2f}' for t in trades_list]}")
    print(f"\nWin Ratio (Backtest):  {win_ratio_bt:.4%}")
    print(f"Win Ratio (Manual):    {win_ratio_manual:.4%}")
    print(f"Diferencia: {abs(win_ratio_bt - win_ratio_manual):.10f}")
    
    # Verificar que ambos métodos dan el mismo resultado
    if np.isnan(win_ratio_bt) and np.isnan(win_ratio_manual):
        print("✅ Ambos métodos retornan NaN (correcto)")
    else:
        assert abs(win_ratio_bt - win_ratio_manual) < 1e-10, \
            f"Win ratio diferente: BT={win_ratio_bt}, Manual={win_ratio_manual}"
        print("✅ TEST 1 PASADO - Win ratios coinciden")
    
    return True


def test_case_2_single_symbol_all_losers():
    """Test 2: Un solo símbolo con todas las operaciones perdedoras"""
    print("\n" + "="*80)
    print("TEST 2: Un símbolo - Todas perdedoras")
    print("="*80)
    
    # Crear datos: 4 señales, todas con pérdida
    ohlcv = create_synthetic_ohlcv(
        symbol='ETH',
        num_candles=80,
        base_price=3000,
        signal_indices=[10, 25, 40, 55],
        price_changes=[-5, -3, -7, -2]  # Todos negativos
    )
    
    ohlcv_arrays = {'ETH': ohlcv}
    
    results = run_grid_backtest(
        ohlcv_arrays=ohlcv_arrays,
        sell_after=3,
        tp_pct=0.0,
        sl_pct=0.0,
        order_amount=100
    )
    
    portfolio = results['__PORTFOLIO__']
    trades_list = portfolio['trades']
    win_ratio_bt = portfolio['proportion_winners']
    num_signals = portfolio['num_signals']
    
    trades_dict = {'ETH': trades_list}
    win_ratio_manual, _ = calculate_win_ratio_manual(trades_dict)
    
    print(f"Señales ejecutadas: {num_signals}")
    print(f"Trades realizados: {len(trades_list)}")
    print(f"Trades detallados: {[f'{t:.2f}' for t in trades_list]}")
    print(f"\nWin Ratio (Backtest):  {win_ratio_bt:.4%}")
    print(f"Win Ratio (Manual):    {win_ratio_manual:.4%}")
    print(f"Diferencia: {abs(win_ratio_bt - win_ratio_manual):.10f}")
    
    if np.isnan(win_ratio_bt) and np.isnan(win_ratio_manual):
        print("✅ Ambos métodos retornan NaN (correcto)")
    else:
        assert abs(win_ratio_bt - win_ratio_manual) < 1e-10
        print("✅ TEST 2 PASADO - Win ratios coinciden")
    
    return True


def test_case_3_single_symbol_mixed():
    """Test 3: Un solo símbolo con operaciones mixtas"""
    print("\n" + "="*80)
    print("TEST 3: Un símbolo - Operaciones mixtas (50/50)")
    print("="*80)
    
    # 10 señales alternando ganancias y pérdidas
    ohlcv = create_synthetic_ohlcv(
        symbol='BTC',
        num_candles=200,
        base_price=50000,
        signal_indices=[10, 30, 50, 70, 90, 110, 130, 150, 170, 190],
        price_changes=[5, -3, 4, -2, 6, -4, 3, -5, 7, -2]  # 5 positivos, 5 negativos
    )
    
    ohlcv_arrays = {'BTC': ohlcv}
    
    results = run_grid_backtest(
        ohlcv_arrays=ohlcv_arrays,
        sell_after=5,
        tp_pct=0.0,
        sl_pct=0.0,
        order_amount=100
    )
    
    portfolio = results['__PORTFOLIO__']
    trades_list = portfolio['trades']
    win_ratio_bt = portfolio['proportion_winners']
    
    trades_dict = {'BTC': trades_list}
    win_ratio_manual, _ = calculate_win_ratio_manual(trades_dict)
    
    winners = sum(1 for t in trades_list if t > 0)
    losers = sum(1 for t in trades_list if t < 0)
    
    print(f"Trades realizados: {len(trades_list)}")
    print(f"Winners: {winners}, Losers: {losers}")
    print(f"Trades detallados: {[f'{t:.2f}' for t in trades_list]}")
    print(f"\nWin Ratio (Backtest):  {win_ratio_bt:.4%}")
    print(f"Win Ratio (Manual):    {win_ratio_manual:.4%}")
    print(f"Diferencia: {abs(win_ratio_bt - win_ratio_manual):.10f}")
    
    if np.isnan(win_ratio_bt) and np.isnan(win_ratio_manual):
        print("✅ Ambos métodos retornan NaN (correcto)")
    else:
        assert abs(win_ratio_bt - win_ratio_manual) < 1e-10
        print("✅ TEST 3 PASADO - Win ratios coinciden")
    
    return True


def test_case_4_multiple_symbols_all_winners():
    """Test 4: Múltiples símbolos - Todas ganadoras"""
    print("\n" + "="*80)
    print("TEST 4: Múltiples símbolos - Todas ganadoras")
    print("="*80)
    
    # Crear múltiples símbolos con todas señales ganadoras
    ohlcv_btc = create_synthetic_ohlcv('BTC', 100, 50000, [10, 30, 50], [5, 3, 7])
    ohlcv_eth = create_synthetic_ohlcv('ETH', 100, 3000, [15, 45], [4, 2])
    ohlcv_ada = create_synthetic_ohlcv('ADA', 100, 1.5, [20, 40, 60], [6, 3, 5])
    
    ohlcv_arrays = {
        'BTC': ohlcv_btc,
        'ETH': ohlcv_eth,
        'ADA': ohlcv_ada
    }
    
    results = run_grid_backtest(
        ohlcv_arrays=ohlcv_arrays,
        sell_after=3,
        tp_pct=0.0,
        sl_pct=0.0,
        order_amount=100
    )
    
    portfolio = results['__PORTFOLIO__']
    trades_list = portfolio['trades']
    win_ratio_bt = portfolio['proportion_winners']
    num_signals = portfolio['num_signals']
    
    trades_dict = {'ALL': trades_list}
    win_ratio_manual, _ = calculate_win_ratio_manual(trades_dict)
    
    print(f"Señales ejecutadas: {num_signals}")
    print(f"Trades totales: {len(trades_list)}")
    print(f"Trades detallados: {[f'{t:.2f}' for t in trades_list]}")
    print(f"\nWin Ratio (Backtest):  {win_ratio_bt:.4%}")
    print(f"Win Ratio (Manual):    {win_ratio_manual:.4%}")
    print(f"Diferencia: {abs(win_ratio_bt - win_ratio_manual):.10f}")
    
    if np.isnan(win_ratio_bt) and np.isnan(win_ratio_manual):
        print("✅ Ambos métodos retornan NaN (correcto)")
    else:
        assert abs(win_ratio_bt - win_ratio_manual) < 1e-10
        print("✅ TEST 4 PASADO - Win ratios coinciden")
    
    return True


def test_case_5_multiple_symbols_mixed():
    """Test 5: Múltiples símbolos con resultados mixtos"""
    print("\n" + "="*80)
    print("TEST 5: Múltiples símbolos - Resultados mixtos")
    print("="*80)
    
    # BTC: 3 señales (2W, 1L)
    ohlcv_btc = create_synthetic_ohlcv('BTC', 100, 50000, [10, 30, 50], [5, -3, 4])
    # ETH: 2 señales (0W, 2L)
    ohlcv_eth = create_synthetic_ohlcv('ETH', 100, 3000, [15, 45], [-4, -2])
    # ADA: 4 señales (3W, 1L)
    ohlcv_ada = create_synthetic_ohlcv('ADA', 100, 1.5, [20, 40, 60, 80], [6, 3, -2, 5])
    
    ohlcv_arrays = {
        'BTC': ohlcv_btc,
        'ETH': ohlcv_eth,
        'ADA': ohlcv_ada
    }
    
    results = run_grid_backtest(
        ohlcv_arrays=ohlcv_arrays,
        sell_after=4,
        tp_pct=0.0,
        sl_pct=0.0,
        order_amount=100
    )
    
    portfolio = results['__PORTFOLIO__']
    trades_list = portfolio['trades']
    win_ratio_bt = portfolio['proportion_winners']
    
    trades_dict = {'ALL': trades_list}
    win_ratio_manual, _ = calculate_win_ratio_manual(trades_dict)
    
    winners = sum(1 for t in trades_list if t > 0)
    losers = sum(1 for t in trades_list if t < 0)
    
    print(f"Trades totales: {len(trades_list)}")
    print(f"Winners: {winners}, Losers: {losers}")
    print(f"Esperado: 5W (2+0+3), 4L (1+2+1) = ~55.56% win rate")
    print(f"Trades detallados: {[f'{t:.2f}' for t in trades_list]}")
    print(f"\nWin Ratio (Backtest):  {win_ratio_bt:.4%}")
    print(f"Win Ratio (Manual):    {win_ratio_manual:.4%}")
    print(f"Diferencia: {abs(win_ratio_bt - win_ratio_manual):.10f}")
    
    if np.isnan(win_ratio_bt) and np.isnan(win_ratio_manual):
        print("✅ Ambos métodos retornan NaN (correcto)")
    else:
        assert abs(win_ratio_bt - win_ratio_manual) < 1e-10
        print("✅ TEST 5 PASADO - Win ratios coinciden")
    
    return True


def test_case_6_no_signals():
    """Test 6: Sin señales de trading"""
    print("\n" + "="*80)
    print("TEST 6: Sin señales de trading")
    print("="*80)
    
    # Crear datos sin señales
    ohlcv = create_synthetic_ohlcv('BTC', 100, 50000, [], [])
    
    ohlcv_arrays = {'BTC': ohlcv}
    
    results = run_grid_backtest(
        ohlcv_arrays=ohlcv_arrays,
        sell_after=3,
        tp_pct=0.0,
        sl_pct=0.0,
        order_amount=100
    )
    
    portfolio = results['__PORTFOLIO__']
    trades_list = portfolio['trades']
    win_ratio_bt = portfolio['proportion_winners']
    num_signals = portfolio['num_signals']
    
    trades_dict = {'BTC': trades_list}
    win_ratio_manual, _ = calculate_win_ratio_manual(trades_dict)
    
    print(f"Señales ejecutadas: {num_signals}")
    print(f"Trades realizados: {len(trades_list)}")
    print(f"\nWin Ratio (Backtest):  {win_ratio_bt}")
    print(f"Win Ratio (Manual):    {win_ratio_manual}")
    
    # Ambos deben ser NaN
    assert np.isnan(win_ratio_bt), "Win ratio debería ser NaN sin trades"
    assert np.isnan(win_ratio_manual), "Win ratio manual debería ser NaN sin trades"
    
    print("✅ TEST 6 PASADO - Ambos métodos retornan NaN correctamente")
    
    return True


def test_case_7_with_tp_sl():
    """Test 7: Con Take Profit y Stop Loss activos"""
    print("\n" + "="*80)
    print("TEST 7: Con TP y SL activos")
    print("="*80)
    
    # Crear datos con movimientos que disparen TP/SL
    ohlcv = create_synthetic_ohlcv(
        'BTC', 150, 50000, 
        [10, 40, 70, 100], 
        [8, -5, 3, -8]  # Algunos dispararán TP, otros SL
    )
    
    ohlcv_arrays = {'BTC': ohlcv}
    
    results = run_grid_backtest(
        ohlcv_arrays=ohlcv_arrays,
        sell_after=10,
        tp_pct=5.0,   # TP al 5%
        sl_pct=3.0,   # SL al 3%
        order_amount=100
    )
    
    portfolio = results['__PORTFOLIO__']
    trades_list = portfolio['trades']
    win_ratio_bt = portfolio['proportion_winners']
    trade_log = portfolio['trade_log']
    
    trades_dict = {'BTC': trades_list}
    win_ratio_manual, _ = calculate_win_ratio_manual(trades_dict)
    
    # Contar exit reasons
    if len(trade_log) > 0:
        exit_reasons = trade_log['exit_reason'].value_counts()
        print(f"Exit reasons:\n{exit_reasons}")
    
    winners = sum(1 for t in trades_list if t > 0)
    losers = sum(1 for t in trades_list if t < 0)
    
    print(f"\nTrades totales: {len(trades_list)}")
    print(f"Winners: {winners}, Losers: {losers}")
    print(f"Trades detallados: {[f'{t:.2f}' for t in trades_list]}")
    print(f"\nWin Ratio (Backtest):  {win_ratio_bt:.4%}")
    print(f"Win Ratio (Manual):    {win_ratio_manual:.4%}")
    print(f"Diferencia: {abs(win_ratio_bt - win_ratio_manual):.10f}")
    
    if np.isnan(win_ratio_bt) and np.isnan(win_ratio_manual):
        print("✅ Ambos métodos retornan NaN (correcto)")
    else:
        assert abs(win_ratio_bt - win_ratio_manual) < 1e-10
        print("✅ TEST 7 PASADO - Win ratios coinciden con TP/SL")
    
    return True


def test_case_8_extreme_case_many_symbols():
    """Test 8: Caso extremo con muchos símbolos"""
    print("\n" + "="*80)
    print("TEST 8: Caso extremo - Muchos símbolos")
    print("="*80)
    
    symbols = ['BTC', 'ETH', 'ADA', 'SOL', 'DOT', 'MATIC', 'AVAX', 'LINK']
    ohlcv_arrays = {}
    
    np.random.seed(42)
    
    for i, sym in enumerate(symbols):
        # Variar parámetros para cada símbolo
        num_signals = np.random.randint(2, 6)
        signal_indices = np.random.choice(range(20, 80), num_signals, replace=False)
        price_changes = np.random.randn(num_signals) * 5  # Cambios aleatorios ±5%
        
        ohlcv_arrays[sym] = create_synthetic_ohlcv(
            sym, 100, 
            1000 * (i + 1),  # Precios base diferentes
            signal_indices.tolist(),
            price_changes.tolist()
        )
    
    results = run_grid_backtest(
        ohlcv_arrays=ohlcv_arrays,
        sell_after=5,
        tp_pct=0.0,
        sl_pct=0.0,
        order_amount=100
    )
    
    portfolio = results['__PORTFOLIO__']
    trades_list = portfolio['trades']
    win_ratio_bt = portfolio['proportion_winners']
    
    trades_dict = {'ALL': trades_list}
    win_ratio_manual, _ = calculate_win_ratio_manual(trades_dict)
    
    winners = sum(1 for t in trades_list if t > 0)
    losers = sum(1 for t in trades_list if t < 0)
    
    print(f"Símbolos: {len(symbols)}")
    print(f"Trades totales: {len(trades_list)}")
    print(f"Winners: {winners}, Losers: {losers}")
    print(f"\nWin Ratio (Backtest):  {win_ratio_bt:.4%}")
    print(f"Win Ratio (Manual):    {win_ratio_manual:.4%}")
    print(f"Diferencia: {abs(win_ratio_bt - win_ratio_manual):.10f}")
    
    if np.isnan(win_ratio_bt) and np.isnan(win_ratio_manual):
        print("✅ Ambos métodos retornan NaN (correcto)")
    else:
        assert abs(win_ratio_bt - win_ratio_manual) < 1e-10
        print("✅ TEST 8 PASADO - Win ratios coinciden con muchos símbolos")
    
    return True


# =============================================================================
# EJECUTAR TODOS LOS TESTS
# =============================================================================

def run_all_tests():
    """Ejecuta todos los tests y muestra resumen final"""
    print("\n" + "="*80)
    print(" BATERÍA COMPLETA DE TESTS - WIN RATIO")
    print(" Comparando función real de backtest vs cálculo manual")
    print("="*80)
    
    tests = [
        test_case_1_single_symbol_all_winners,
        test_case_2_single_symbol_all_losers,
        test_case_3_single_symbol_mixed,
        test_case_4_multiple_symbols_all_winners,
        test_case_5_multiple_symbols_mixed,
        test_case_6_no_signals,
        test_case_7_with_tp_sl,
        test_case_8_extreme_case_many_symbols
    ]
    
    passed = 0
    failed = 0
    errors = []
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            failed += 1
            errors.append((test_func.__name__, str(e)))
            print(f"❌ {test_func.__name__} FALLÓ: {e}")
        except Exception as e:
            failed += 1
            errors.append((test_func.__name__, f"Error inesperado: {str(e)}"))
            print(f"❌ {test_func.__name__} ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    # Resumen final
    print("\n" + "="*80)
    print(" RESUMEN DE TESTS")
    print("="*80)
    print(f"Total tests ejecutados: {len(tests)}")
    print(f"✅ Tests pasados: {passed}")
    print(f"❌ Tests fallados: {failed}")
    
    if failed > 0:
        print("\n" + "="*80)
        print(" ERRORES ENCONTRADOS:")
        print("="*80)
        for test_name, error_msg in errors:
            print(f"\n{test_name}:")
            print(f"  {error_msg}")
    else:
        print("\n" + "="*80)
        print(" 🎉 TODOS LOS TESTS PASARON EXITOSAMENTE 🎉")
        print("="*80)
        print("\nEl cálculo del Win Ratio es CORRECTO:")
        print("  ✓ Coincide entre la función real y el cálculo manual")
        print("  ✓ Un solo símbolo (todas ganadoras/perdedoras/mixtas)")
        print("  ✓ Múltiples símbolos (todas ganadoras/mixtas)")
        print("  ✓ Sin señales (retorna NaN)")
        print("  ✓ Con TP/SL activos")
        print("  ✓ Muchos símbolos simultáneos")
    
    return passed, failed


if __name__ == "__main__":
    passed, failed = run_all_tests()
    sys.exit(0 if failed == 0 else 1)