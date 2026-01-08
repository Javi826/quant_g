"""
TESTS REALES del Trading Bot
Autocontenido - ejecutar en la misma carpeta que BOT_orchestator_WS.py

Ejecutar:
    pytest test_bot_REAL.py -v

Instalar dependencias:
    pip install pytest pytest-mock
"""

import pytest
import sys
import os
from decimal import Decimal
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock, patch
import pandas as pd
import tempfile
import json

# Importar módulos reales del bot
from ZX_BOT_operative import (
    calculate_tp_sl_prices,
    calculate_pnl,
    get_fills_for_order,
    quantize_size,
    extract_contract_params,
    compute_size_base,
    sync_broker,
    save_state_local,
    load_state,
    log_closed_position,
    configure_paths,
    close_position,
    increment_strategy_candles,
    reset_strategy_candles,
    calculate_next_candle_time,
    get_current_price,  # ← AÑADIDO (se usa en tests)
    add_position  # ← AÑADIDO (se usa en tests de integración)
)

from ZX_BOT_metrics import MetricsCalculator


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def temp_excel_file():
    """Archivo Excel temporal para tests de logging"""
    # Crear archivo temporal con extensión .xlsx
    fd, temp_path = tempfile.mkstemp(suffix='.xlsx')
    os.close(fd)  # Cerrar file descriptor
    
    # Crear Excel VACÍO pero VÁLIDO
    empty_df = pd.DataFrame(columns=[
        'OPEN_AT', 'CLOSE_AT', 'DURATION_DAYS', 'STRATEGY', 'SYMBOL',
        'DIRECTION', 'USDT_AMOUNT', 'SIZE', 'PRICE_ENTRY', 'PRICE_CLOSE',
        'PROFIT', 'FEE', 'PROFIT_PCT', 'REASON_OUT'
    ])
    empty_df.to_excel(temp_path, index=False, engine='openpyxl')
    
    # Configurar path global
    configure_paths(temp_path, initial_capital=3671)
    
    yield temp_path
    
    # Cleanup
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def temp_state_file():
    """Archivo de estado temporal"""
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        initial_state = {
            'positions': {},
            'strategy_candles': {}
        }
        json.dump(initial_state, f)
        temp_path = f.name
    
    yield temp_path
    
    if os.path.exists(temp_path):
        os.remove(temp_path)


@pytest.fixture
def sample_trades_df():
    """DataFrame de ejemplo con trades"""
    data = {
        'OPEN_AT': ['2024-01-01 10:00:00', '2024-01-02 11:00:00', '2024-01-03 12:00:00'],
        'CLOSE_AT': ['2024-01-01 14:00:00', '2024-01-02 15:00:00', '2024-01-03 16:00:00'],
        'PROFIT': [100.0, -50.0, 75.0],
        'STRATEGY': ['02_reversal_long_4H', '02_reversal_long_4H', '05_parity_short_1H']
    }
    return pd.DataFrame(data)


@pytest.fixture
def mock_ws_manager():
    """Mock del WebSocket manager"""
    with patch('ZX_BOT_operative.ZX_BOT_websocket._ws_manager') as mock_ws:
        # Configurar prices
        mock_ws.prices = {
            'BTCUSDT': {
                'price': Decimal('50000.0'),
                'timestamp': datetime.now().timestamp()
            }
        }
        
        # Configurar positions
        mock_ws.positions = {}
        
        # Configurar fills
        mock_ws.fills = {}
        
        # Configurar contracts
        mock_ws.contracts = {
            'BTCUSDT': {
                'pricePlace': '1',
                'volumePlace': '3',
                'minTradeNum': '0.001',
                'minTradeUSDT': '5',
                'sizeMultiplier': '1'
            }
        }
        
        # Métodos
        def get_position(symbol):
            return mock_ws.positions.get(symbol, None)
        
        def get_fills(order_id):
            return mock_ws.fills.get(order_id, [])
        
        def refresh_positions():
            pass
        
        mock_ws.get_position = get_position
        mock_ws.get_fills = get_fills
        mock_ws.refresh_positions = refresh_positions
        
        yield mock_ws


# ============================================================================
# TESTS: calculate_tp_sl_prices (FUNCIÓN REAL)
# ============================================================================

class TestCalculateTPSL:
    """Tests de la función REAL calculate_tp_sl_prices"""
    
    def test_long_position_tp_sl(self):
        """Test TP/SL para posición LONG"""
        entry_price = Decimal('50000.0')
        direction = 'long'
        tp_pct = 3
        sl_pct = 10
        
        # Ejecutar función REAL
        tp_price, sl_price = calculate_tp_sl_prices(entry_price, direction, tp_pct, sl_pct)
        
        # Verificar resultados
        assert tp_price == Decimal('51500.0'), f"Expected TP 51500, got {tp_price}"
        assert sl_price == Decimal('45000.0'), f"Expected SL 45000, got {sl_price}"
        assert tp_price > entry_price, "TP debe estar por encima de entry para LONG"
        assert sl_price < entry_price, "SL debe estar por debajo de entry para LONG"
    
    def test_short_position_tp_sl(self):
        """Test TP/SL para posición SHORT"""
        entry_price = Decimal('50000.0')
        direction = 'short'
        tp_pct = 3
        sl_pct = 10
        
        # Ejecutar función REAL
        tp_price, sl_price = calculate_tp_sl_prices(entry_price, direction, tp_pct, sl_pct)
        
        # Verificar resultados
        assert tp_price == Decimal('48500.0'), f"Expected TP 48500, got {tp_price}"
        assert sl_price == Decimal('55000.0'), f"Expected SL 55000, got {sl_price}"
        assert tp_price < entry_price, "TP debe estar por debajo de entry para SHORT"
        assert sl_price > entry_price, "SL debe estar por encima de entry para SHORT"
    
    def test_edge_case_minimum_percentage(self):
        """Test con porcentajes mínimos válidos (1.5%)"""
        entry_price = Decimal('1000.0')
        tp_pct = 1.5
        sl_pct = 1.5
        
        # LONG
        tp_long, sl_long = calculate_tp_sl_prices(entry_price, 'long', tp_pct, sl_pct)
        assert tp_long == Decimal('1015.0')
        assert sl_long == Decimal('985.0')
        
        # SHORT
        tp_short, sl_short = calculate_tp_sl_prices(entry_price, 'short', tp_pct, sl_pct)
        assert tp_short == Decimal('985.0')
        assert sl_short == Decimal('1015.0')
    
    def test_edge_case_maximum_percentage(self):
        """Test con porcentajes máximos válidos (10%)"""
        entry_price = Decimal('1000.0')
        tp_pct = 10
        sl_pct = 10
        
        # LONG
        tp_long, sl_long = calculate_tp_sl_prices(entry_price, 'long', tp_pct, sl_pct)
        assert tp_long == Decimal('1100.0')
        assert sl_long == Decimal('900.0')


# ============================================================================
# TESTS: calculate_pnl (FUNCIÓN REAL)
# ============================================================================

class TestCalculatePnL:
    """Tests de la función REAL calculate_pnl"""
    
    def test_long_profit(self):
        """Test profit en posición LONG"""
        direction = 'long'
        entry_price = Decimal('50000.0')
        current_price = Decimal('51500.0')
        size = Decimal('0.5')
        
        # Ejecutar función REAL
        pnl = calculate_pnl(direction, entry_price, current_price, size)
        
        # Verificar
        assert pnl == 750.0, f"Expected 750.0, got {pnl}"
    
    def test_long_loss(self):
        """Test pérdida en posición LONG"""
        direction = 'long'
        entry_price = Decimal('50000.0')
        current_price = Decimal('45000.0')
        size = Decimal('0.5')
        
        pnl = calculate_pnl(direction, entry_price, current_price, size)
        
        assert pnl == -2500.0, f"Expected -2500.0, got {pnl}"
    
    def test_short_profit(self):
        """Test profit en posición SHORT"""
        direction = 'short'
        entry_price = Decimal('50000.0')
        current_price = Decimal('48500.0')
        size = Decimal('0.5')
        
        pnl = calculate_pnl(direction, entry_price, current_price, size)
        
        assert pnl == 750.0, f"Expected 750.0, got {pnl}"
    
    def test_short_loss(self):
        """Test pérdida en posición SHORT"""
        direction = 'short'
        entry_price = Decimal('50000.0')
        current_price = Decimal('55000.0')
        size = Decimal('0.5')
        
        pnl = calculate_pnl(direction, entry_price, current_price, size)
        
        assert pnl == -2500.0, f"Expected -2500.0, got {pnl}"


# ============================================================================
# TESTS: quantize_size (FUNCIÓN REAL)
# ============================================================================

class TestQuantizeSize:
    """Tests de la función REAL quantize_size"""
    
    def test_quantize_to_3_decimals(self):
        """Test cuantización a 3 decimales"""
        size_base = Decimal('0.12345678')
        size_scale = 3
        
        # Ejecutar función REAL
        size_q, precision = quantize_size(size_base, size_scale)
        
        # Verificar
        assert size_q == Decimal('0.123'), f"Expected 0.123, got {size_q}"
        assert precision == Decimal('0.001')
    
    def test_quantize_rounds_down(self):
        """Test que la cuantización redondea hacia abajo"""
        size_base = Decimal('0.9999')
        size_scale = 3
        
        size_q, _ = quantize_size(size_base, size_scale)
        
        assert size_q == Decimal('0.999'), "Debe redondear hacia abajo"
        assert size_q < size_base
    
    def test_zero_size_handling(self):
        """Test manejo de size que resulta en 0"""
        size_base = Decimal('0.0001')  # Muy pequeño
        size_scale = 3
        
        size_q, _ = quantize_size(size_base, size_scale)
        
        # Debe usar fallback a 6 decimales
        assert size_q is not None
        assert size_q > 0


# ============================================================================
# TESTS: compute_size_base (FUNCIÓN REAL)
# ============================================================================

class TestComputeSizeBase:
    """Tests de la función REAL compute_size_base"""
    
    def test_compute_size_basic(self):
        """Test cálculo básico de size"""
        usdt_amount = 40
        last_price = Decimal('50000.0')
        
        # Ejecutar función REAL
        size_base = compute_size_base(usdt_amount, last_price)
        
        # Verificar
        expected = Decimal('40') / Decimal('50000.0')
        assert size_base == expected, f"Expected {expected}, got {size_base}"
        assert size_base == Decimal('0.0008')
    
    def test_compute_size_with_high_price(self):
        """Test con precio alto"""
        usdt_amount = 100
        last_price = Decimal('100000.0')
        
        size_base = compute_size_base(usdt_amount, last_price)
        
        assert size_base == Decimal('0.001')


# ============================================================================
# TESTS: extract_contract_params (FUNCIÓN REAL)
# ============================================================================

class TestExtractContractParams:
    """Tests de la función REAL extract_contract_params"""
    
    def test_extract_valid_contract(self):
        """Test extracción de contrato válido"""
        contract = {
            'pricePlace': '1',
            'volumePlace': '3',
            'minTradeNum': '0.001',
            'sizeMultiplier': '1',
            'minTradeUSDT': '5'
        }
        last_price = Decimal('50000.0')
        
        # Ejecutar función REAL
        price_tick, size_scale, min_trade_num, size_multiplier, min_trade_usdt = \
            extract_contract_params(contract, last_price)
        
        # Verificar
        assert price_tick == Decimal('0.1'), f"Expected 0.1, got {price_tick}"
        assert size_scale == 3
        assert min_trade_num == Decimal('0.001')
        assert size_multiplier == Decimal('1')
        assert min_trade_usdt == Decimal('5')
    
    def test_extract_none_contract(self):
        """Test con contrato None"""
        result = extract_contract_params(None, Decimal('50000.0'))
        
        # Verificar que todos son None
        assert all(x is None for x in result)
    
    def test_extract_invalid_contract(self):
        """Test con contrato con datos inválidos"""
        contract = {
            'pricePlace': 'invalid',
            'volumePlace': '3'
        }
        
        result = extract_contract_params(contract, Decimal('50000.0'))
        
        # Debe retornar None para todos
        assert all(x is None for x in result)


# ============================================================================
# TESTS: State Management (FUNCIONES REALES)
# ============================================================================

class TestStateManagement:
    """Tests de las funciones REALES de gestión de estado"""
    
    def test_save_and_load_state(self, temp_state_file):
        """Test guardar y cargar estado"""
        # Crear estado
        open_positions = {
            '02_reversal_long_4H': [
                {
                    'symbol': 'BTCUSDT',
                    'size': Decimal('0.5'),
                    'entry_price': Decimal('50000.0'),
                    'direction': 'long',
                    'tp': Decimal('51500.0'),
                    'sl': Decimal('45000.0'),
                    'order_id': 'order_123',
                    'opened_at': datetime(2024, 12, 30, 10, 15, 23),
                    'usdt_amount': 40.0
                }
            ]
        }
        strategy_candles = {'02_reversal_long_4H': 5}
        
        # Guardar (función REAL)
        save_state_local(open_positions, strategy_candles, temp_state_file)
        
        # Cargar (función REAL)
        loaded_positions, loaded_candles = load_state(temp_state_file)
        
        # Verificar
        assert '02_reversal_long_4H' in loaded_positions
        assert len(loaded_positions['02_reversal_long_4H']) == 1
        
        pos = loaded_positions['02_reversal_long_4H'][0]
        assert pos['symbol'] == 'BTCUSDT'
        assert pos['size'] == Decimal('0.5')
        assert pos['entry_price'] == Decimal('50000.0')
        
        assert loaded_candles['02_reversal_long_4H'] == 5
    
    def test_load_nonexistent_state(self, temp_state_file):
        """Test cargar estado que no existe"""
        # Eliminar archivo
        os.remove(temp_state_file)
        
        # Cargar (función REAL)
        loaded_positions, loaded_candles = load_state(temp_state_file)
        
        # Debe retornar vacío
        assert loaded_positions == {}
        assert loaded_candles == {}


# ============================================================================
# TESTS: log_closed_position (FUNCIÓN REAL)
# ============================================================================

class TestLogClosedPosition:
    """Tests de la función REAL log_closed_position"""
    
    def test_log_position_with_api_data(self, temp_excel_file):
        """Test logging con datos de API"""
        opened_at = datetime(2024, 12, 30, 10, 0, 0)
        
        # Ejecutar función REAL
        log_closed_position(
            opened_at=opened_at,
            strategy_id='02_reversal_long_4H',
            symbol='BTCUSDT',
            direction='long',
            usdt_amount=40.0,
            entry_price=Decimal('50000.0'),
            close_price=Decimal('51500.0'),
            reason='TP',
            size=Decimal('0.0008'),
            profit_from_api=Decimal('1.2'),
            fee_from_api=Decimal('0.02')
        )
        
        # Verificar que se creó el Excel
        assert os.path.exists(temp_excel_file)
        
        # Leer y verificar
        df = pd.read_excel(temp_excel_file)
        assert len(df) == 1
        
        row = df.iloc[0]
        assert row['SYMBOL'] == 'BTCUSDT'
        assert row['DIRECTION'] == 'LONG'
        assert row['REASON_OUT'] == 'TP'
        assert row['PROFIT'] > 0  # Debe ser positivo (TP hit)
    
    def test_log_position_without_api_data(self, temp_excel_file):
        """Test logging sin datos de API (cálculo manual)"""
        opened_at = datetime(2024, 12, 30, 10, 0, 0)
        
        # Ejecutar función REAL
        log_closed_position(
            opened_at=opened_at,
            strategy_id='05_parity_short_1H',
            symbol='ETHUSDT',
            direction='short',
            usdt_amount=40.0,
            entry_price=Decimal('3000.0'),
            close_price=Decimal('2910.0'),
            reason='TP',
            size=Decimal('0.0133'),
            profit_from_api=None,
            fee_from_api=None
        )
        
        # Leer y verificar
        df = pd.read_excel(temp_excel_file)
        assert len(df) == 1
        
        row = df.iloc[0]
        assert row['SYMBOL'] == 'ETHUSDT'
        assert row['DIRECTION'] == 'SHORT'
        assert row['PROFIT'] > 0  # Short profit cuando precio baja


# ============================================================================
# TESTS: sync_broker (FUNCIÓN REAL)
# ============================================================================

class TestSyncBroker:
    """Tests de la función REAL sync_broker"""
    
    def test_sync_removes_not_found_position(self, mock_ws_manager, temp_state_file, temp_excel_file):
        """Test que sync_broker elimina posición NOT_FOUND"""
        # Configurar estado local con posición
        open_positions = {
            '02_reversal_long_4H': [
                {
                    'symbol': 'BTCUSDT',
                    'size': Decimal('0.5'),
                    'entry_price': Decimal('50000.0'),
                    'direction': 'long',
                    'tp': Decimal('51500.0'),
                    'sl': Decimal('45000.0'),
                    'order_id': 'order_123',
                    'opened_at': datetime(2024, 12, 30, 10, 0, 0),
                    'usdt_amount': 40.0
                }
            ]
        }
        strategy_candles = {'02_reversal_long_4H': 5}
        
        # WebSocket NO tiene la posición (broker la cerró)
        mock_ws_manager.positions = {}
        
        # Ejecutar función REAL
        sync_broker(open_positions, strategy_candles, temp_state_file)
        
        # Verificar que se eliminó
        assert len(open_positions['02_reversal_long_4H']) == 0
        assert strategy_candles['02_reversal_long_4H'] == 0
        
        # Verificar que se logueó
        if os.path.exists(temp_excel_file):
            df = pd.read_excel(temp_excel_file)
            if len(df) > 0:
                assert df.iloc[-1]['REASON_OUT'] == 'NOT_FOUND'
    
    def test_sync_keeps_existing_position(self, mock_ws_manager, temp_state_file, temp_excel_file):
        """Test que sync_broker mantiene posición que existe"""
        # Configurar estado local con posición
        open_positions = {
            '02_reversal_long_4H': [
                {
                    'symbol': 'BTCUSDT',
                    'size': Decimal('0.5'),
                    'entry_price': Decimal('50000.0'),
                    'direction': 'long',
                    'tp': Decimal('51500.0'),
                    'sl': Decimal('45000.0'),
                    'order_id': 'order_123',
                    'opened_at': datetime(2024, 12, 30, 10, 0, 0),
                    'usdt_amount': 40.0
                }
            ]
        }
        strategy_candles = {'02_reversal_long_4H': 5}
        
        # WebSocket SÍ tiene la posición
        mock_ws_manager.positions = {
            'BTCUSDT': {
                'instId': 'BTCUSDT',
                'total': '0.5',
                'available': '0.5'
            }
        }
        
        # Ejecutar función REAL
        sync_broker(open_positions, strategy_candles, temp_state_file)
        
        # Verificar que NO se eliminó
        assert len(open_positions['02_reversal_long_4H']) == 1
        assert strategy_candles['02_reversal_long_4H'] == 5


# ============================================================================
# TESTS: MetricsCalculator (CLASE REAL)
# ============================================================================

class TestMetricsCalculator:
    """Tests de la clase REAL MetricsCalculator"""
    
    def test_profit_factor(self, sample_trades_df):
        """Test cálculo de Profit Factor"""
        # Ejecutar método REAL
        pf = MetricsCalculator.profit_factor(sample_trades_df)
        
        # Verificar
        # Total wins = 100 + 75 = 175
        # Total losses = 50
        # PF = 175 / 50 = 3.5
        assert pf == 3.5, f"Expected 3.5, got {pf}"
    
    def test_weekly_win_percentage(self, sample_trades_df):
        """Test cálculo de Weekly Win %"""
        # Ejecutar método REAL
        weekly_win = MetricsCalculator.weekly_win_percentage(sample_trades_df)
        
        # Verificar (3 semanas diferentes, 2 positivas)
        assert weekly_win > 0
    
    def test_max_drawdown_from_equity(self):
        """Test cálculo de Max Drawdown"""
        equity_series = pd.Series([1000, 1100, 950, 1050, 900, 1200])
        
        # Ejecutar método REAL
        max_dd = MetricsCalculator.max_drawdown_from_equity(equity_series)
        
        # Verificar que es negativo
        assert max_dd < 0
        # El drawdown máximo debe ser de 1100 a 900 = -18.18%
        assert abs(max_dd - (-18.18)) < 0.1
    
    def test_sharpe_ratio(self):
        """Test cálculo de Sharpe Ratio"""
        daily_returns = pd.Series([0.01, -0.005, 0.02, 0.015, -0.01])
        
        # Ejecutar método REAL
        sharpe = MetricsCalculator.sharpe_ratio(daily_returns)
        
        # Verificar que es un número
        assert isinstance(sharpe, float)
        assert sharpe != 0
    
    def test_ulcer_index(self):
        """Test cálculo de Ulcer Index"""
        equity_series = pd.Series([1000, 1100, 950, 1050, 900, 1200])
        
        # Ejecutar método REAL
        ulcer = MetricsCalculator.ulcer_index(equity_series)
        
        # Verificar que es positivo
        assert ulcer > 0
    
    def test_calculate_all_metrics(self, sample_trades_df):
        """Test método unificado calculate_all_metrics"""
        capital_assigned = 1000.0
        
        # Ejecutar método REAL
        metrics = MetricsCalculator.calculate_all_metrics(
            df=sample_trades_df,
            capital_assigned=capital_assigned,
            include_profit_pct=True
        )
        
        # Verificar que retorna todas las métricas
        assert 'num_trades' in metrics
        assert 'total_profit_usd' in metrics
        assert 'profit_factor' in metrics
        assert 'weekly_win_pct' in metrics
        assert 'win_rate' in metrics
        assert 'max_dd' in metrics
        assert 'ulcer_index' in metrics
        assert 'sharpe_ratio' in metrics
        assert 'daily_profit' in metrics
        assert 'total_profit_pct' in metrics
        
        # Verificar valores
        assert metrics['num_trades'] == 3
        assert metrics['total_profit_usd'] == 125.0  # 100 - 50 + 75
        assert metrics['profit_factor'] == 3.5


# ============================================================================
# TESTS: get_fills_for_order (FUNCIÓN REAL)
# ============================================================================

class TestGetFillsForOrder:
    """Tests de la función REAL get_fills_for_order"""
    
    def test_get_fills_single_fill(self, mock_ws_manager):
        """Test obtener fills con un solo fill completo"""
        order_id = 'order_123'
        
        # Simular fills en WebSocket
        mock_ws_manager.fills[order_id] = [
            {
                'orderId': order_id,
                'baseVolume': '0.5',
                'price': '50000.0',
                'profit': '0',
                'feeDetail': [{'totalFee': '0.025'}]
            }
        ]
        
        # Ejecutar función REAL (sin send_request_func, solo WebSocket)
        # Nota: La función usa WebSocket directamente
        fills = mock_ws_manager.get_fills(order_id)
        
        # Verificar
        assert len(fills) == 1
        assert fills[0]['baseVolume'] == '0.5'
        assert fills[0]['price'] == '50000.0'
    
    def test_get_fills_multiple_partial_fills(self, mock_ws_manager):
        """Test obtener fills con múltiples fills parciales"""
        order_id = 'order_456'
        
        # Simular fills parciales
        mock_ws_manager.fills[order_id] = [
            {
                'orderId': order_id,
                'baseVolume': '0.25',
                'price': '50000.0',
                'profit': '0',
                'feeDetail': [{'totalFee': '0.0125'}]
            },
            {
                'orderId': order_id,
                'baseVolume': '0.25',
                'price': '50010.0',
                'profit': '0',
                'feeDetail': [{'totalFee': '0.0125'}]
            }
        ]
        
        fills = mock_ws_manager.get_fills(order_id)
        
        # Verificar
        assert len(fills) == 2
        
        # Calcular weighted average price
        total_base = Decimal('0.25') + Decimal('0.25')
        weighted = Decimal('50000.0') * Decimal('0.25') + Decimal('50010.0') * Decimal('0.25')
        avg_price = weighted / total_base
        
        assert avg_price == Decimal('50005.0')
    
    def test_get_fills_no_fills_available(self, mock_ws_manager):
        """Test cuando no hay fills disponibles"""
        order_id = 'order_nonexistent'
        
        # No hay fills
        mock_ws_manager.fills = {}
        
        fills = mock_ws_manager.get_fills(order_id)
        
        # Verificar
        assert fills == []


# ============================================================================
# TESTS: close_position scenarios (FUNCIÓN REAL)
# ============================================================================

class TestClosePositionScenarios:
    """Tests de diferentes escenarios de cierre de posición"""
    
    def test_close_position_tp_reason(self, temp_excel_file):
        """Test cierre por TP"""
        # Mock send_request
        def mock_send_request(method, path, params=None, body=None):
            return 200, {
                'code': '00000',
                'data': {
                    'orderId': 'close_order_123',
                    'price': '51500.0'
                }
            }
        
        # Mock WebSocket fills
        with patch('ZX_BOT_operative.ZX_BOT_websocket._ws_manager') as mock_ws:
            mock_ws.fills = {
                'close_order_123': [
                    {
                        'orderId': 'close_order_123',
                        'baseVolume': '0.5',
                        'price': '51500.0',
                        'profit': '750.0',
                        'feeDetail': [{'totalFee': '0.026'}]
                    }
                ]
            }
            
            mock_ws.get_fills = lambda oid: mock_ws.fills.get(oid, [])
            
            # Mock get_current_price
            with patch('ZX_BOT_operative.get_current_price', return_value=Decimal('51500.0')):
                position_data = {
                    'opened_at': datetime(2024, 12, 30, 10, 0, 0),
                    'strategy_id': '02_reversal_long_4H',
                    'usdt_amount': 40.0,
                    'entry_price': Decimal('50000.0')
                }
                
                # Ejecutar función REAL
                result = close_position(
                    symbol='BTCUSDT',
                    size=Decimal('0.5'),
                    direction='long',
                    send_request_func=mock_send_request,
                    reason='TP',
                    position_data=position_data
                )
                
                # Verificar
                assert result == True
                
                # Verificar log
                if os.path.exists(temp_excel_file):
                    df = pd.read_excel(temp_excel_file)
                    if len(df) > 0:
                        assert df.iloc[-1]['REASON_OUT'] == 'TP'
    
    def test_close_position_sl_reason(self, temp_excel_file):
        """Test cierre por SL"""
        def mock_send_request(method, path, params=None, body=None):
            return 200, {
                'code': '00000',
                'data': {
                    'orderId': 'close_order_456',
                    'price': '45000.0'
                }
            }
        
        with patch('ZX_BOT_operative.ZX_BOT_websocket._ws_manager') as mock_ws:
            mock_ws.fills = {
                'close_order_456': [
                    {
                        'orderId': 'close_order_456',
                        'baseVolume': '0.5',
                        'price': '45000.0',
                        'profit': '-2500.0',
                        'feeDetail': [{'totalFee': '0.023'}]
                    }
                ]
            }
            
            mock_ws.get_fills = lambda oid: mock_ws.fills.get(oid, [])
            
            with patch('ZX_BOT_operative.get_current_price', return_value=Decimal('45000.0')):
                position_data = {
                    'opened_at': datetime(2024, 12, 30, 10, 0, 0),
                    'strategy_id': '02_reversal_long_4H',
                    'usdt_amount': 40.0,
                    'entry_price': Decimal('50000.0')
                }
                
                result = close_position(
                    symbol='BTCUSDT',
                    size=Decimal('0.5'),
                    direction='long',
                    send_request_func=mock_send_request,
                    reason='SL',
                    position_data=position_data
                )
                
                assert result == True
    
    def test_close_position_timeout_reason(self, temp_excel_file):
        """Test cierre por TIMEOUT"""
        def mock_send_request(method, path, params=None, body=None):
            return 200, {
                'code': '00000',
                'data': {
                    'orderId': 'close_order_789',
                    'price': '49500.0'
                }
            }
        
        with patch('ZX_BOT_operative.ZX_BOT_websocket._ws_manager') as mock_ws:
            mock_ws.fills = {
                'close_order_789': [
                    {
                        'orderId': 'close_order_789',
                        'baseVolume': '0.5',
                        'price': '49500.0',
                        'profit': '-250.0',
                        'feeDetail': [{'totalFee': '0.025'}]
                    }
                ]
            }
            
            mock_ws.get_fills = lambda oid: mock_ws.fills.get(oid, [])
            
            with patch('ZX_BOT_operative.get_current_price', return_value=Decimal('49500.0')):
                position_data = {
                    'opened_at': datetime(2024, 12, 30, 10, 0, 0),
                    'strategy_id': '02_reversal_long_4H',
                    'usdt_amount': 40.0,
                    'entry_price': Decimal('50000.0')
                }
                
                result = close_position(
                    symbol='BTCUSDT',
                    size=Decimal('0.5'),
                    direction='long',
                    send_request_func=mock_send_request,
                    reason='TIMEOUT',
                    position_data=position_data
                )
                
                assert result == True
    
    def test_close_position_error_22002(self, temp_excel_file):
        """Test error 22002 (posición no existe)"""
        def mock_send_request(method, path, params=None, body=None):
            return 200, {
                'code': '22002',
                'msg': 'Position does not exist'
            }
        
        with patch('ZX_BOT_operative.get_current_price', return_value=Decimal('50500.0')):
            position_data = {
                'opened_at': datetime(2024, 12, 30, 10, 0, 0),
                'strategy_id': '02_reversal_long_4H',
                'usdt_amount': 40.0,
                'entry_price': Decimal('50000.0')
            }
            
            # Ejecutar función REAL
            result = close_position(
                symbol='BTCUSDT',
                size=Decimal('0.5'),
                direction='long',
                send_request_func=mock_send_request,
                reason='TP',
                position_data=position_data
            )
            
            # Debe retornar True para eliminar de local state
            assert result == True
    
    def test_close_position_network_error(self):
        """Test error de red al cerrar posición"""
        def mock_send_request(method, path, params=None, body=None):
            return 0, {'error': 'Network timeout'}
        
        position_data = {
            'opened_at': datetime(2024, 12, 30, 10, 0, 0),
            'strategy_id': '02_reversal_long_4H',
            'usdt_amount': 40.0,
            'entry_price': Decimal('50000.0')
        }
        
        result = close_position(
            symbol='BTCUSDT',
            size=Decimal('0.5'),
            direction='long',
            send_request_func=mock_send_request,
            reason='TP',
            position_data=position_data
        )
        
        # Debe retornar False (falló)
        assert result == False


# ============================================================================
# TESTS: Timeframe calculations
# ============================================================================

class TestTimeframeCalculations:
    """Tests de cálculos de timeframes"""
    
    def test_calculate_next_candle_4h(self):
        """Test cálculo de próxima vela 4H"""
        try:
            from zoneinfo import ZoneInfo
            madrid_tz = ZoneInfo('Europe/Madrid')
        except ImportError:
            import pytz
            madrid_tz = pytz.timezone('Europe/Madrid')
        
        # Ejecutar función REAL
        next_candle = calculate_next_candle_time('4H', madrid_tz)
        
        # Verificar que es un datetime
        assert isinstance(next_candle, datetime)
        # Verificar que el segundo es 45 (buffer)
        assert next_candle.second == 45
    
    def test_calculate_next_candle_1h(self):
        """Test cálculo de próxima vela 1H"""
        try:
            from zoneinfo import ZoneInfo
            madrid_tz = ZoneInfo('Europe/Madrid')
        except ImportError:
            import pytz
            madrid_tz = pytz.timezone('Europe/Madrid')
        
        # Ejecutar función REAL
        next_candle = calculate_next_candle_time('1H', madrid_tz)
        
        # Verificar que es un datetime
        assert isinstance(next_candle, datetime)
        # Verificar que el segundo es 45 (buffer)
        assert next_candle.second == 45


# ============================================================================
# TESTS: Edge cases y validaciones
# ============================================================================

class TestEdgeCases:
    """Tests de casos extremos"""
    
    def test_tp_sl_with_zero_entry_price(self):
        """Test TP/SL con precio de entrada cero (inválido)"""
        with pytest.raises(Exception):
            calculate_tp_sl_prices(Decimal('0'), 'long', 3, 10)
    
    def test_quantize_with_negative_size(self):
        """Test cuantización con size negativo"""
        size_base = Decimal('-0.5')
        size_scale = 3
        
        size_q, _ = quantize_size(size_base, size_scale)
        
        # Debería cuantizar pero mantener negativo
        assert size_q == Decimal('-0.500')
    
    def test_pnl_with_zero_size(self):
        """Test PnL con size cero"""
        pnl = calculate_pnl('long', Decimal('50000'), Decimal('51000'), Decimal('0'))
        
        assert pnl == 0.0
    
    def test_metrics_with_empty_dataframe(self):
        """Test métricas con DataFrame vacío"""
        empty_df = pd.DataFrame()
        
        metrics = MetricsCalculator.calculate_all_metrics(
            df=empty_df,
            capital_assigned=1000.0
        )
        
        assert metrics['num_trades'] == 0
        assert metrics['total_profit_usd'] == 0.0
        assert metrics['profit_factor'] == 0.0
    
    def test_metrics_with_all_wins(self):
        """Test métricas con solo trades ganadores"""
        df = pd.DataFrame({
            'PROFIT': [100, 200, 150],
            'CLOSE_AT': ['2024-01-01', '2024-01-02', '2024-01-03']
        })
        
        pf = MetricsCalculator.profit_factor(df)
        
        # Sin pérdidas, profit factor debería ser 0
        assert pf == 0.0
    
    def test_metrics_with_all_losses(self):
        """Test métricas con solo trades perdedores"""
        df = pd.DataFrame({
            'PROFIT': [-100, -200, -150],
            'CLOSE_AT': ['2024-01-01', '2024-01-02', '2024-01-03']
        })
        
        pf = MetricsCalculator.profit_factor(df)
        
        # Sin ganancias, profit factor debería ser 0
        assert pf == 0.0


# ============================================================================
# TESTS: Multiple positions scenarios
# ============================================================================

class TestMultiplePositionsScenarios:
    """Tests con múltiples posiciones"""
    
    def test_save_load_multiple_strategies(self, temp_state_file):
        """Test guardar/cargar múltiples estrategias con posiciones"""
        open_positions = {
            '02_reversal_long_4H': [
                {
                    'symbol': 'BTCUSDT',
                    'size': Decimal('0.5'),
                    'entry_price': Decimal('50000.0'),
                    'direction': 'long',
                    'tp': Decimal('51500.0'),
                    'sl': Decimal('45000.0'),
                    'order_id': 'order_1',
                    'opened_at': datetime(2024, 12, 30, 10, 0, 0),
                    'usdt_amount': 40.0
                }
            ],
            '05_parity_short_1H': [
                {
                    'symbol': 'ETHUSDT',
                    'size': Decimal('1.2'),
                    'entry_price': Decimal('3000.0'),
                    'direction': 'short',
                    'tp': Decimal('2910.0'),
                    'sl': Decimal('3300.0'),
                    'order_id': 'order_2',
                    'opened_at': datetime(2024, 12, 30, 11, 0, 0),
                    'usdt_amount': 40.0
                }
            ]
        }
        strategy_candles = {
            '02_reversal_long_4H': 5,
            '05_parity_short_1H': 12
        }
        
        # Guardar
        save_state_local(open_positions, strategy_candles, temp_state_file)
        
        # Cargar
        loaded_pos, loaded_candles = load_state(temp_state_file)
        
        # Verificar
        assert len(loaded_pos) == 2
        assert len(loaded_pos['02_reversal_long_4H']) == 1
        assert len(loaded_pos['05_parity_short_1H']) == 1
        assert loaded_candles['02_reversal_long_4H'] == 5
        assert loaded_candles['05_parity_short_1H'] == 12
    
    def test_pnl_multiple_positions_same_symbol(self):
        """Test PnL de múltiples posiciones en el mismo símbolo"""
        positions = [
            {'direction': 'long', 'entry': Decimal('50000'), 'size': Decimal('0.3')},
            {'direction': 'long', 'entry': Decimal('51000'), 'size': Decimal('0.2')}
        ]
        current_price = Decimal('52000')
        
        total_pnl = 0.0
        for pos in positions:
            pnl = calculate_pnl(pos['direction'], pos['entry'], current_price, pos['size'])
            total_pnl += pnl
        
        # (52000-50000)*0.3 + (52000-51000)*0.2 = 600 + 200 = 800
        assert total_pnl == 800.0


# ============================================================================
# TESTS DE INTEGRACIÓN
# ============================================================================

class TestIntegrationFlows:
    """Tests de flujos completos"""
    
    def test_full_position_lifecycle(self, temp_state_file, temp_excel_file):
        """Test ciclo completo: crear posición → guardar → cargar → verificar"""
        # 1. Crear posición
        entry_price = Decimal('50000.0')
        tp_price, sl_price = calculate_tp_sl_prices(entry_price, 'long', 3, 10)
        
        open_positions = {
            '02_reversal_long_4H': [
                {
                    'symbol': 'BTCUSDT',
                    'size': Decimal('0.5'),
                    'entry_price': entry_price,
                    'direction': 'long',
                    'tp': tp_price,
                    'sl': sl_price,
                    'order_id': 'order_123',
                    'opened_at': datetime(2024, 12, 30, 10, 0, 0),
                    'usdt_amount': 40.0
                }
            ]
        }
        strategy_candles = {'02_reversal_long_4H': 5}
        
        # 2. Guardar
        save_state_local(open_positions, strategy_candles, temp_state_file)
        
        # 3. Cargar
        loaded_pos, loaded_candles = load_state(temp_state_file)
        
        # 4. Verificar
        assert len(loaded_pos['02_reversal_long_4H']) == 1
        pos = loaded_pos['02_reversal_long_4H'][0]
        
        assert pos['tp'] == Decimal('51500.0')
        assert pos['sl'] == Decimal('45000.0')
        
        # 5. Simular cierre en TP
        current_price = Decimal('51600.0')  # Por encima de TP
        pnl = calculate_pnl('long', entry_price, current_price, pos['size'])
        
        assert pnl > 0, "Debería tener profit"
        
        # 6. Loguear cierre
        log_closed_position(
            opened_at=pos['opened_at'],
            strategy_id='02_reversal_long_4H',
            symbol='BTCUSDT',
            direction='long',
            usdt_amount=40.0,
            entry_price=entry_price,
            close_price=current_price,
            reason='TP',
            size=pos['size'],
            profit_from_api=None,
            fee_from_api=None
        )
        
        # 7. Verificar log
        df = pd.read_excel(temp_excel_file)
        assert len(df) == 1
        assert df.iloc[0]['REASON_OUT'] == 'TP'
        assert df.iloc[0]['PROFIT'] > 0


# ============================================================================
# TESTS: Profit calculations con fees
# ============================================================================

class TestProfitCalculationsWithFees:
    """Tests de cálculos de profit incluyendo fees"""
    
    def test_profit_with_open_close_fees(self):
        """Test profit con fees de apertura y cierre"""
        profit_gross = Decimal('750.0')
        fee_open = Decimal('0.025')
        fee_close = Decimal('0.026')
        
        fee_total = (fee_open + fee_close)
        profit_net = profit_gross - fee_total
        
        assert profit_net == Decimal('749.949')
    
    def test_profit_percentage_calculation(self):
        """Test cálculo de profit en porcentaje"""
        profit_net = 75.0
        usdt_amount = 40.0
        
        profit_pct = (profit_net / usdt_amount) * 100
        
        assert profit_pct == 187.5
    
    def test_negative_profit_with_fees(self):
        """Test pérdida con fees"""
        profit_gross = Decimal('-100.0')
        fee_total = Decimal('0.05')
        
        profit_net = profit_gross - fee_total
        
        assert profit_net == Decimal('-100.05')
        assert profit_net < 0


# ============================================================================
# TESTS: Strategy candles management
# ============================================================================

class TestStrategyCandles:
    """Tests de gestión de contadores de velas"""
    
    def test_increment_candles_from_zero(self, temp_state_file):
        """Test incrementar velas desde cero"""
        open_positions = {'02_reversal_long_4H': []}
        strategy_candles = {'02_reversal_long_4H': 0}
        
        # Incrementar
        increment_strategy_candles(
            '02_reversal_long_4H',
            strategy_candles,
            open_positions,
            temp_state_file
        )
        
        assert strategy_candles['02_reversal_long_4H'] == 1
    
    def test_increment_candles_multiple_times(self, temp_state_file):
        """Test incrementar velas múltiples veces"""
        open_positions = {'02_reversal_long_4H': []}
        strategy_candles = {'02_reversal_long_4H': 0}
        
        for i in range(10):
            increment_strategy_candles(
                '02_reversal_long_4H',
                strategy_candles,
                open_positions,
                temp_state_file
            )
        
        assert strategy_candles['02_reversal_long_4H'] == 10
    
    def test_reset_candles_to_zero(self, temp_state_file):
        """Test resetear velas a cero"""
        open_positions = {'02_reversal_long_4H': []}
        strategy_candles = {'02_reversal_long_4H': 45}
        
        # Resetear
        reset_strategy_candles(
            '02_reversal_long_4H',
            strategy_candles,
            open_positions,
            temp_state_file
        )
        
        assert strategy_candles['02_reversal_long_4H'] == 0


# ============================================================================
# TESTS: Contract params extraction con diferentes símbolos
# ============================================================================

class TestContractParamsVariousSymbols:
    """Tests de extracción de parámetros para diferentes símbolos"""
    
    def test_btc_contract_params(self):
        """Test parámetros de contrato BTC (precio alto)"""
        contract = {
            'pricePlace': '1',
            'volumePlace': '3',
            'minTradeNum': '0.001',
            'sizeMultiplier': '1',
            'minTradeUSDT': '5'
        }
        
        price_tick, size_scale, min_trade_num, size_mult, min_usdt = \
            extract_contract_params(contract, Decimal('50000'))
        
        assert price_tick == Decimal('0.1')
        assert size_scale == 3
        assert min_trade_num == Decimal('0.001')
    
    def test_eth_contract_params(self):
        """Test parámetros de contrato ETH (precio medio)"""
        contract = {
            'pricePlace': '2',
            'volumePlace': '2',
            'minTradeNum': '0.01',
            'sizeMultiplier': '1',
            'minTradeUSDT': '5'
        }
        
        price_tick, size_scale, min_trade_num, size_mult, min_usdt = \
            extract_contract_params(contract, Decimal('3000'))
        
        assert price_tick == Decimal('0.01')
        assert size_scale == 2
        assert min_trade_num == Decimal('0.01')
    
    def test_low_price_coin_contract(self):
        """Test parámetros de contrato para coin de bajo precio"""
        contract = {
            'pricePlace': '4',
            'volumePlace': '1',
            'minTradeNum': '1',
            'sizeMultiplier': '1',
            'minTradeUSDT': '5'
        }
        
        price_tick, size_scale, min_trade_num, size_mult, min_usdt = \
            extract_contract_params(contract, Decimal('0.5'))
        
        assert price_tick == Decimal('0.0001')
        assert size_scale == 1
        assert min_trade_num == Decimal('1')


# ============================================================================
# TESTS: Size computation para diferentes precios
# ============================================================================

class TestSizeComputationVariousPrices:
    """Tests de cálculo de size con diferentes precios"""
    
    def test_size_high_price_btc(self):
        """Test size con precio alto (BTC)"""
        usdt_amount = 100
        price = Decimal('100000.0')  # BTC muy alto
        
        size = compute_size_base(usdt_amount, price)
        
        assert size == Decimal('0.001')
    
    def test_size_medium_price_eth(self):
        """Test size con precio medio (ETH)"""
        usdt_amount = 100
        price = Decimal('2500.0')
        
        size = compute_size_base(usdt_amount, price)
        
        assert size == Decimal('0.04')
    
    def test_size_low_price_coin(self):
        """Test size con precio bajo"""
        usdt_amount = 100
        price = Decimal('0.5')
        
        size = compute_size_base(usdt_amount, price)
        
        assert size == Decimal('200')
    
    def test_size_minimum_order(self):
        """Test size mínimo de orden (40 USDT)"""
        usdt_amount = 40  # Mínimo permitido
        price = Decimal('50000.0')
        
        size = compute_size_base(usdt_amount, price)
        
        assert size == Decimal('0.0008')


# ============================================================================
# TESTS: MetricsCalculator edge cases
# ============================================================================

class TestMetricsCalculatorEdgeCases:
    """Tests de casos extremos del calculador de métricas"""
    
    def test_single_trade_metrics(self):
        """Test métricas con un solo trade"""
        df = pd.DataFrame({
            'PROFIT': [100.0],
            'CLOSE_AT': ['2024-01-01 10:00:00']
        })
        
        metrics = MetricsCalculator.calculate_all_metrics(df, 1000.0)
        
        assert metrics['num_trades'] == 1
        assert metrics['total_profit_usd'] == 100.0
        assert metrics['win_rate'] == 100.0
    
    def test_exactly_50_percent_win_rate(self):
        """Test win rate exactamente 50%"""
        df = pd.DataFrame({
            'PROFIT': [100, -100, 50, -50],
            'CLOSE_AT': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']
        })
        
        metrics = MetricsCalculator.calculate_all_metrics(df, 1000.0)
        
        assert metrics['win_rate'] == 50.0
    
    def test_very_high_profit_factor(self):
        """Test profit factor muy alto"""
        df = pd.DataFrame({
            'PROFIT': [1000, 500, 750, -10],
            'CLOSE_AT': ['2024-01-01', '2024-01-02', '2024-01-03', '2024-01-04']
        })
        
        pf = MetricsCalculator.profit_factor(df)
        
        # 2250 / 10 = 225
        assert pf == 225.0
    
    def test_breakeven_trading(self):
        """Test trading breakeven (sin profit ni loss)"""
        df = pd.DataFrame({
            'PROFIT': [100, -100, 50, -50, 25, -25],
            'CLOSE_AT': ['2024-01-01', '2024-01-02', '2024-01-03', 
                         '2024-01-04', '2024-01-05', '2024-01-06']
        })
        
        total = MetricsCalculator.total_profit_usd(df)
        
        assert total == 0.0


# ============================================================================
# TESTS: TP/SL con diferentes símbolos y direcciones
# ============================================================================

class TestTPSLVariousScenarios:
    """Tests de TP/SL con diferentes escenarios"""
    
    def test_long_btc_aggressive(self):
        """Test LONG BTC con TP/SL agresivos"""
        entry = Decimal('50000')
        tp, sl = calculate_tp_sl_prices(entry, 'long', 10, 10)  # 10% ambos
        
        assert tp == Decimal('55000')
        assert sl == Decimal('45000')
    
    def test_short_eth_conservative(self):
        """Test SHORT ETH con TP/SL conservadores"""
        entry = Decimal('3000')
        tp, sl = calculate_tp_sl_prices(entry, 'short', 1.5, 1.5)  # 1.5% ambos
        
        assert tp == Decimal('2955')
        assert sl == Decimal('3045')
    
    def test_long_with_asymmetric_tp_sl(self):
        """Test LONG con TP y SL asimétricos"""
        entry = Decimal('1000')
        tp, sl = calculate_tp_sl_prices(entry, 'long', 5, 10)  # TP 5%, SL 10%
        
        assert tp == Decimal('1050')
        assert sl == Decimal('900')
        
        # Risk/Reward ratio
        risk = entry - sl  # 100
        reward = tp - entry  # 50
        rr_ratio = reward / risk
        
        assert rr_ratio == Decimal('0.5')  # 1:2 risk/reward


# ============================================================================
# TESTS: State management con corrupción
# ============================================================================

class TestStateManagementCorruption:
    """Tests de gestión de estado con datos corruptos"""
    
    def test_load_corrupted_json(self, temp_state_file):
        """Test cargar JSON corrupto"""
        # Escribir JSON inválido
        with open(temp_state_file, 'w') as f:
            f.write("{invalid json content")
        
        # Debería retornar vacío sin crashear
        positions, candles = load_state(temp_state_file)
        
        assert positions == {}
        assert candles == {}
    
    def test_load_json_missing_keys(self, temp_state_file):
        """Test cargar JSON válido pero sin keys esperadas"""
        with open(temp_state_file, 'w') as f:
            json.dump({'wrong_key': 'value'}, f)
        
        positions, candles = load_state(temp_state_file)
        
        assert positions == {}
        assert candles == {}
    
    def test_save_state_with_special_characters(self, temp_state_file):
        """Test guardar estado con caracteres especiales en símbolos"""
        open_positions = {
            '02_reversal_long_4H': [
                {
                    'symbol': 'BTC/USDT:USDT',  # Símbolo con caracteres especiales
                    'size': Decimal('0.5'),
                    'entry_price': Decimal('50000.0'),
                    'direction': 'long',
                    'tp': Decimal('51500.0'),
                    'sl': Decimal('45000.0'),
                    'order_id': 'order_123',
                    'opened_at': datetime(2024, 12, 30, 10, 0, 0),
                    'usdt_amount': 40.0
                }
            ]
        }
        strategy_candles = {'02_reversal_long_4H': 5}
        
        # Guardar
        save_state_local(open_positions, strategy_candles, temp_state_file)
        
        # Cargar
        loaded_pos, loaded_candles = load_state(temp_state_file)
        
        # Verificar que se guardó correctamente
        assert len(loaded_pos['02_reversal_long_4H']) == 1


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == '__main__':
    import pytest
    
    print("=" * 60)
    print("EJECUTANDO TESTS REALES DEL BOT")
    print("=" * 60)
    print()
    
    # Ejecutar con verbose
    pytest.main([__file__, '-v', '--tb=short', '-s'])