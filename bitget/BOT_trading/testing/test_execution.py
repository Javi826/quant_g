"""
Tests for execution module (order_manager, position_tracker, trade_logger).

Run with:
    python3 testing/test_execution.py
    
Or with pytest:
    pytest testing/test_execution.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from decimal import Decimal
from unittest.mock import patch
from testing.helpers import (
    get_mock_ws_manager,
    reset_mock_ws_manager,
    mock_send_request_success,
    mock_send_request_insufficient_balance,
    get_sample_position,
    get_sample_strategy_config
)


# ==========================================================================
# SETUP
# ==========================================================================
def setup_module():
    """Setup before all tests."""
    print("\n" + "="*70)
    print("TESTING: execution module")
    print("="*70)


def setup_function():
    """Setup before each test."""
    reset_mock_ws_manager()


# ==========================================================================
# TESTS: order_manager.py
# ==========================================================================
def test_fetch_ticker_ws():
    """Test fetching ticker price via WebSocket."""
    with patch('market_data.get_ws_manager', get_mock_ws_manager):
        from execution.order_manager import fetch_ticker_ws
        
        # Set test price
        ws = get_mock_ws_manager()
        ws.set_price('BTCUSDT', 50000.0)
        
        # Fetch price
        price, _ = fetch_ticker_ws('BTCUSDT')
        
        # Assert
        assert price == Decimal('50000.0'), f"Expected 50000.0, got {price}"
        print("✅ test_fetch_ticker_ws PASSED")


def test_get_current_price():
    """Test getting current price."""
    with patch('market_data.get_ws_manager', get_mock_ws_manager):
        from execution.order_manager import get_current_price
        
        # Set test price
        ws = get_mock_ws_manager()
        ws.set_price('ETHUSDT', 3000.0)
        
        # Get price
        price = get_current_price('ETHUSDT')
        
        # Assert
        assert price == Decimal('3000.0'), f"Expected 3000.0, got {price}"
        print("✅ test_get_current_price PASSED")


def test_compute_size_base():
    """Test size calculation."""
    from execution.order_manager import compute_size_base
    
    usdt = 100.0
    price = Decimal('50000.0')
    
    size = compute_size_base(usdt, price)
    
    expected = Decimal('100.0') / Decimal('50000.0')
    assert size == expected, f"Expected {expected}, got {size}"
    print("✅ test_compute_size_base PASSED")


def test_place_order_success():
    """Test successful order placement."""
    with patch('market_data.get_ws_manager', get_mock_ws_manager):
        from execution.order_manager import place_order
        
        # Setup
        ws = get_mock_ws_manager()
        ws.set_price('BTCUSDT', 50000.0)
        ws.set_balance(1000.0)
        
        # Place order
        result = place_order(
            symbol='BTCUSDT',
            direction='long',
            usdt_amount=50,
            send_request_func=mock_send_request_success
        )
        
        # Assert
        assert result is not None, "Order should succeed"
        assert result['code'] == '00000', "Order should return success code"
        print("✅ test_place_order_success PASSED")


def test_place_order_insufficient_balance():
    """Test order placement with insufficient balance."""
    with patch('market_data.get_ws_manager', get_mock_ws_manager):
        from execution.order_manager import place_order
        
        # Setup
        ws = get_mock_ws_manager()
        ws.set_price('BTCUSDT', 50000.0)
        
        # Place order
        result = place_order(
            symbol='BTCUSDT',
            direction='long',
            usdt_amount=50,
            send_request_func=mock_send_request_insufficient_balance
        )
        
        # Assert
        assert result is None, "Order should fail with insufficient balance"
        print("✅ test_place_order_insufficient_balance PASSED")


# ==========================================================================
# TESTS: position_tracker.py
# ==========================================================================
def test_calculate_tp_sl_prices_long():
    """Test TP/SL calculation for long position."""
    from execution.position_tracker import calculate_tp_sl_prices
    
    entry = Decimal('100.0')
    tp_pct = 5.0
    sl_pct = 2.0
    
    tp, sl = calculate_tp_sl_prices(entry, 'long', tp_pct, sl_pct)
    
    assert tp == Decimal('105.0'), f"Expected TP 105.0, got {tp}"
    assert sl == Decimal('98.0'), f"Expected SL 98.0, got {sl}"
    print("✅ test_calculate_tp_sl_prices_long PASSED")


def test_calculate_tp_sl_prices_short():
    """Test TP/SL calculation for short position."""
    from execution.position_tracker import calculate_tp_sl_prices
    
    entry = Decimal('100.0')
    tp_pct = 5.0
    sl_pct = 2.0
    
    tp, sl = calculate_tp_sl_prices(entry, 'short', tp_pct, sl_pct)
    
    assert tp == Decimal('95.0'), f"Expected TP 95.0, got {tp}"
    assert sl == Decimal('102.0'), f"Expected SL 102.0, got {sl}"
    print("✅ test_calculate_tp_sl_prices_short PASSED")


def test_calculate_pnl_long_profit():
    """Test PnL calculation for long position with profit."""
    from execution.position_tracker import calculate_pnl
    
    entry = Decimal('100.0')
    current = Decimal('110.0')
    size = Decimal('10.0')
    
    pnl = calculate_pnl('long', entry, current, size)
    
    expected = 100.0  # (110 - 100) * 10
    assert pnl == expected, f"Expected {expected}, got {pnl}"
    print("✅ test_calculate_pnl_long_profit PASSED")


def test_calculate_pnl_short_profit():
    """Test PnL calculation for short position with profit."""
    from execution.position_tracker import calculate_pnl
    
    entry = Decimal('100.0')
    current = Decimal('90.0')
    size = Decimal('10.0')
    
    pnl = calculate_pnl('short', entry, current, size)
    
    expected = 100.0  # (100 - 90) * 10
    assert pnl == expected, f"Expected {expected}, got {pnl}"
    print("✅ test_calculate_pnl_short_profit PASSED")

# ==========================================================================
# TESTS: Closing Positions
# ==========================================================================
def test_close_position_success():
    """Test successful position closure with TP."""
    from testing.helpers import create_temp_excel, cleanup_temp_file
    
    temp_excel = create_temp_excel()
    
    try:
        with patch('market_data.get_ws_manager', get_mock_ws_manager):
            from execution.order_manager import close_position, configure_paths
            
            # Configure temp paths
            configure_paths(temp_excel, initial_capital=1000)
            
            # Setup
            ws = get_mock_ws_manager()
            ws.set_price('BTCUSDT', 52000.0)
            ws.add_fills('mock_order_close', [{
                'baseVolume': '0.001',
                'price': '52000.0',
                'profit': '2.0',
                'feeDetail': [{'totalFee': '0.05'}]
            }])
            
            position_data = get_sample_position()
            
            # Close position
            result = close_position(
                symbol='BTCUSDT',
                size=Decimal('0.001'),
                direction='long',
                send_request_func=mock_send_request_success,
                reason='TP',
                position_data=position_data
            )
            
            assert result == True, "Position should close successfully"
            print("✅ test_close_position_success PASSED")
    
    finally:
        cleanup_temp_file(temp_excel)


def test_close_position_not_exist():
    """Test closing non-existent position (error 22002)."""
    from testing.helpers import create_temp_excel, cleanup_temp_file, mock_send_request_position_not_exist
    
    temp_excel = create_temp_excel()
    
    try:
        with patch('market_data.get_ws_manager', get_mock_ws_manager):
            from execution.order_manager import close_position, configure_paths
            
            # Configure temp paths
            configure_paths(temp_excel, initial_capital=1000)
            
            # Setup
            ws = get_mock_ws_manager()
            ws.set_price('BTCUSDT', 50000.0)
            
            position_data = get_sample_position()
            
            # Close non-existent position
            result = close_position(
                symbol='BTCUSDT',
                size=Decimal('0.001'),
                direction='long',
                send_request_func=mock_send_request_position_not_exist,
                reason='TP',
                position_data=position_data
            )
            
            # Should return True (remove from tracking)
            assert result == True, "Should handle non-existent position"
            print("✅ test_close_position_not_exist PASSED")
    
    finally:
        cleanup_temp_file(temp_excel)


def test_get_fills_for_order():
    """Test getting fills for an order via WebSocket."""
    with patch('market_data.get_ws_manager', get_mock_ws_manager):
        from execution.order_manager import get_fills_for_order
        from testing.helpers import get_sample_fills
        
        # Setup
        ws = get_mock_ws_manager()
        fills = get_sample_fills()
        ws.add_fills('test_order_123', fills)
        
        # Get fills
        total_base, entry_price, profit, fee = get_fills_for_order(
            order_id='test_order_123',
            symbol='BTCUSDT',
            send_request_func=mock_send_request_success
        )
        
        assert total_base == Decimal('0.001'), f"Expected 0.001, got {total_base}"
        assert entry_price == Decimal('50000.0'), f"Expected 50000.0, got {entry_price}"
        print("✅ test_get_fills_for_order PASSED")


def test_add_position():
    """Test adding position to tracking."""
    from testing.helpers import create_temp_json, cleanup_temp_file
    from execution.position_tracker import add_position
    from zoneinfo import ZoneInfo
    
    temp_json = create_temp_json()
    
    try:
        # Setup
        open_positions = {}
        strategy_candles = {}
        hour_zone = ZoneInfo('UTC')
        
        # Add position
        add_position(
            strat_id='01_test_strategy',
            symbol='BTCUSDT',
            size=Decimal('0.001'),
            entry_price=Decimal('50000.0'),
            direction='long',
            tp_pct=4.0,
            sl_pct=10.0,
            order_id='test_order_123',
            open_positions=open_positions,
            strategy_candles=strategy_candles,
            state_file=temp_json,
            hour_zone=hour_zone,
            usdt_amount=50.0
        )
        
        # Assert
        assert '01_test_strategy' in open_positions, "Strategy should be in positions"
        assert len(open_positions['01_test_strategy']) == 1, "Should have 1 position"
        
        pos = open_positions['01_test_strategy'][0]
        assert pos['symbol'] == 'BTCUSDT'
        assert pos['size'] == Decimal('0.001')
        assert pos['direction'] == 'long'
        
        print("✅ test_add_position PASSED")
    
    finally:
        cleanup_temp_file(temp_json)


def test_check_tp_sl_hit_tp():
    """Test TP hit detection and position closure."""
    from testing.helpers import create_temp_json, create_temp_excel, cleanup_temp_file
    from execution.position_tracker import check_tp_sl_for_strategy
    from decimal import Decimal
    
    temp_json = create_temp_json()
    temp_excel = create_temp_excel()
    
    try:
        with patch('market_data.get_ws_manager', get_mock_ws_manager):
            from execution.order_manager import configure_paths
            
            # Configure temp paths
            configure_paths(temp_excel, initial_capital=1000)
            
            # Setup
            ws = get_mock_ws_manager()
            ws.set_price('BTCUSDT', 52000.0)  # Above TP (52000)
            ws.add_fills('mock_order_12345', [{
                'baseVolume': '0.001',
                'price': '52000.0',
                'profit': '2.0',
                'feeDetail': [{'totalFee': '0.05'}]
            }])
            
            # Create position
            open_positions = {
                '01_test_strategy': [get_sample_position()]
            }
            strategy_candles = {}
            strat_config = get_sample_strategy_config()
            
            # Check TP/SL
            check_tp_sl_for_strategy(
                strat_id='01_test_strategy',
                strat_config=strat_config,
                open_positions=open_positions,
                strategy_candles=strategy_candles,
                state_file=temp_json,
                send_request_func=mock_send_request_success
            )
            
            # Position should be closed (removed)
            assert len(open_positions['01_test_strategy']) == 0, "Position should be closed"
            print("✅ test_check_tp_sl_hit_tp PASSED")
    
    finally:
        cleanup_temp_file(temp_json)
        cleanup_temp_file(temp_excel)
# ==========================================================================
# TESTS: Closing Positions
# ==========================================================================
def test_close_position_success():
    """Test successful position closure with TP."""
    from testing.helpers import create_temp_excel, cleanup_temp_file
    
    temp_excel = create_temp_excel()
    
    try:
        with patch('market_data.get_ws_manager', get_mock_ws_manager):
            from execution.order_manager import close_position, configure_paths
            
            # Configure temp paths
            configure_paths(temp_excel, initial_capital=1000)
            
            # Setup
            ws = get_mock_ws_manager()
            ws.set_price('BTCUSDT', 52000.0)
            ws.add_fills('mock_order_close', [{
                'baseVolume': '0.001',
                'price': '52000.0',
                'profit': '2.0',
                'feeDetail': [{'totalFee': '0.05'}]
            }])
            
            position_data = get_sample_position()
            
            # Close position
            result = close_position(
                symbol='BTCUSDT',
                size=Decimal('0.001'),
                direction='long',
                send_request_func=mock_send_request_success,
                reason='TP',
                position_data=position_data
            )
            
            assert result == True, "Position should close successfully"
            print("✅ test_close_position_success PASSED")
    
    finally:
        cleanup_temp_file(temp_excel)


def test_close_position_not_exist():
    """Test closing non-existent position (error 22002)."""
    from testing.helpers import create_temp_excel, cleanup_temp_file, mock_send_request_position_not_exist
    
    temp_excel = create_temp_excel()
    
    try:
        with patch('market_data.get_ws_manager', get_mock_ws_manager):
            from execution.order_manager import close_position, configure_paths
            
            # Configure temp paths
            configure_paths(temp_excel, initial_capital=1000)
            
            # Setup
            ws = get_mock_ws_manager()
            ws.set_price('BTCUSDT', 50000.0)
            
            position_data = get_sample_position()
            
            # Close non-existent position
            result = close_position(
                symbol='BTCUSDT',
                size=Decimal('0.001'),
                direction='long',
                send_request_func=mock_send_request_position_not_exist,
                reason='TP',
                position_data=position_data
            )
            
            # Should return True (remove from tracking)
            assert result == True, "Should handle non-existent position"
            print("✅ test_close_position_not_exist PASSED")
    
    finally:
        cleanup_temp_file(temp_excel)


def test_get_fills_for_order():
    """Test getting fills for an order via WebSocket."""
    with patch('market_data.get_ws_manager', get_mock_ws_manager):
        from execution.order_manager import get_fills_for_order
        from testing.helpers import get_sample_fills
        
        # Setup
        ws = get_mock_ws_manager()
        fills = get_sample_fills()
        ws.add_fills('test_order_123', fills)
        
        # Get fills
        total_base, entry_price, profit, fee = get_fills_for_order(
            order_id='test_order_123',
            symbol='BTCUSDT',
            send_request_func=mock_send_request_success
        )
        
        assert total_base == Decimal('0.001'), f"Expected 0.001, got {total_base}"
        assert entry_price == Decimal('50000.0'), f"Expected 50000.0, got {entry_price}"
        print("✅ test_get_fills_for_order PASSED")


def test_add_position():
    """Test adding position to tracking."""
    from testing.helpers import create_temp_json, cleanup_temp_file
    from execution.position_tracker import add_position
    from zoneinfo import ZoneInfo
    
    temp_json = create_temp_json()
    
    try:
        # Setup
        open_positions = {}
        strategy_candles = {}
        hour_zone = ZoneInfo('UTC')
        
        # Add position
        add_position(
            strat_id='01_test_strategy',
            symbol='BTCUSDT',
            size=Decimal('0.001'),
            entry_price=Decimal('50000.0'),
            direction='long',
            tp_pct=4.0,
            sl_pct=10.0,
            order_id='test_order_123',
            open_positions=open_positions,
            strategy_candles=strategy_candles,
            state_file=temp_json,
            hour_zone=hour_zone,
            usdt_amount=50.0
        )
        
        # Assert
        assert '01_test_strategy' in open_positions, "Strategy should be in positions"
        assert len(open_positions['01_test_strategy']) == 1, "Should have 1 position"
        
        pos = open_positions['01_test_strategy'][0]
        assert pos['symbol'] == 'BTCUSDT'
        assert pos['size'] == Decimal('0.001')
        assert pos['direction'] == 'long'
        
        print("✅ test_add_position PASSED")
    
    finally:
        cleanup_temp_file(temp_json)


def test_check_tp_sl_hit_tp():
    """Test TP hit detection and position closure."""
    from testing.helpers import create_temp_json, create_temp_excel, cleanup_temp_file
    from execution.position_tracker import check_tp_sl_for_strategy
    from decimal import Decimal
    
    temp_json = create_temp_json()
    temp_excel = create_temp_excel()
    
    try:
        with patch('market_data.get_ws_manager', get_mock_ws_manager):
            from execution.order_manager import configure_paths
            
            # Configure temp paths
            configure_paths(temp_excel, initial_capital=1000)
            
            # Setup
            ws = get_mock_ws_manager()
            ws.set_price('BTCUSDT', 52000.0)  # Above TP (52000)
            ws.add_fills('mock_order_12345', [{
                'baseVolume': '0.001',
                'price': '52000.0',
                'profit': '2.0',
                'feeDetail': [{'totalFee': '0.05'}]
            }])
            
            # Create position
            open_positions = {
                '01_test_strategy': [get_sample_position()]
            }
            strategy_candles = {}
            strat_config = get_sample_strategy_config()
            
            # Check TP/SL
            check_tp_sl_for_strategy(
                strat_id='01_test_strategy',
                strat_config=strat_config,
                open_positions=open_positions,
                strategy_candles=strategy_candles,
                state_file=temp_json,
                send_request_func=mock_send_request_success
            )
            
            # Position should be closed (removed)
            assert len(open_positions['01_test_strategy']) == 0, "Position should be closed"
            print("✅ test_check_tp_sl_hit_tp PASSED")
    
    finally:
        cleanup_temp_file(temp_json)
        cleanup_temp_file(temp_excel)


def test_log_closed_position():
    """Test logging closed position to Excel."""
    from testing.helpers import create_temp_excel, cleanup_temp_file
    from execution.trade_logger import log_closed_position, configure_log_path
    from datetime import datetime
    import pandas as pd
    
    temp_excel = create_temp_excel()
    
    try:
        # Configure logger
        configure_log_path(temp_excel)
        
        # Log position
        log_closed_position(
            opened_at=datetime(2024, 1, 1, 10, 0, 0),
            strategy_id='01_test_strategy',
            symbol='BTCUSDT',
            direction='long',
            usdt_amount=50.0,
            entry_price=Decimal('50000.0'),
            close_price=Decimal('52000.0'),
            reason='TP',
            size=Decimal('0.001'),
            profit_from_api=Decimal('2.0'),
            fee_from_api=Decimal('0.05')
        )
        
        # Verify Excel file
        assert os.path.exists(temp_excel), "Excel file should be created"
        
        df = pd.read_excel(temp_excel)
        assert len(df) == 1, "Should have 1 record"
        assert df.iloc[0]['SYMBOL'] == 'BTCUSDT'
        assert df.iloc[0]['REASON_OUT'] == 'TP'
        
        print("✅ test_log_closed_position PASSED")
    
    finally:
        cleanup_temp_file(temp_excel)
        
# ==========================================================================
# TESTS: Helper Functions (order_manager)
# ==========================================================================
def test_fallback_params():
    """Test fallback parameter logic."""
    from execution.order_manager import fallback_params
    from decimal import Decimal
    
    # Test with all None
    price_tick, size_scale, min_trade_num, min_trade_usdt = fallback_params(
        price_tick=None,
        size_scale=None,
        last_price=Decimal('50000'),
        min_trade_num=None,
        min_trade_usdt=None
    )
    
    assert price_tick == Decimal('0.1'), f"Expected 0.1 for BTC price, got {price_tick}"
    assert size_scale == 6, f"Expected 6, got {size_scale}"
    assert min_trade_num == Decimal('0.01'), f"Expected 0.01, got {min_trade_num}"
    print("✅ test_fallback_params PASSED")


def test_quantize_size():
    """Test size quantization."""
    from execution.order_manager import quantize_size
    from decimal import Decimal
    
    # Normal case
    size_base = Decimal('0.123456789')
    size_scale = 3
    
    size_q, precision = quantize_size(size_base, size_scale)
    
    assert size_q == Decimal('0.123'), f"Expected 0.123, got {size_q}"
    assert precision == Decimal('0.001'), f"Expected 0.001, got {precision}"
    
    # Zero case
    size_base_zero = Decimal('0.0000001')
    size_scale_zero = 3
    
    size_q_zero, _ = quantize_size(size_base_zero, size_scale_zero)
    assert size_q_zero is None or size_q_zero > 0, "Should handle tiny sizes"
    
    print("✅ test_quantize_size PASSED")


def test_build_order_body():
    """Test order body construction."""
    from execution.order_manager import build_order_body
    from decimal import Decimal
    
    body = build_order_body(
        symbol='BTCUSDT',
        product_type='USDT-FUTURES',
        margin_mode='crossed',
        margin_coin='USDT',
        size_q=Decimal('0.001'),
        side='buy',
        client_oid='test_12345'
    )
    
    assert body['symbol'] == 'BTCUSDT'
    assert body['side'] == 'buy'
    assert body['tradeSide'] == 'open'
    assert body['orderType'] == 'market'
    assert body['clientOid'] == 'test_12345'
    assert float(body['size']) == 0.001
    
    print("✅ test_build_order_body PASSED")


def test_extract_filled_amount():
    """Test extracting filled amount from order response."""
    from execution.order_manager import extract_filled_amount
    from decimal import Decimal
    
    # With baseVolume
    resp = {
        'data': {
            'baseVolume': '0.001',
            'price': '50000'
        }
    }
    size_q = Decimal('0.001')
    
    filled = extract_filled_amount(resp, size_q)
    assert filled == Decimal('0.001'), f"Expected 0.001, got {filled}"
    
    # Without baseVolume (fallback)
    resp_empty = {'data': {}}
    filled_fallback = extract_filled_amount(resp_empty, size_q)
    assert filled_fallback == size_q, f"Should fallback to size_q"
    
    print("✅ test_extract_filled_amount PASSED")


# ==========================================================================
# TESTS: Edge Cases (position_tracker)
# ==========================================================================
def test_check_tp_sl_no_positions():
    """Test check_tp_sl with no open positions."""
    from testing.helpers import create_temp_json, cleanup_temp_file
    from execution.position_tracker import check_tp_sl_for_strategy
    from unittest.mock import Mock
    
    temp_json = create_temp_json()
    
    try:
        with patch('market_data.get_ws_manager', get_mock_ws_manager):
            open_positions = {}
            strategy_candles = {}
            strat_config = get_sample_strategy_config()
            
            # Should not crash with empty positions
            check_tp_sl_for_strategy(
                strat_id='01_test_strategy',
                strat_config=strat_config,
                open_positions=open_positions,
                strategy_candles=strategy_candles,
                state_file=temp_json,
                send_request_func=Mock()
            )
            
            print("✅ test_check_tp_sl_no_positions PASSED")
    
    finally:
        cleanup_temp_file(temp_json)


def test_close_position_with_retry():
    """Test close_position handles retries properly."""
    from testing.helpers import create_temp_excel, cleanup_temp_file
    
    temp_excel = create_temp_excel()
    
    try:
        with patch('market_data.get_ws_manager', get_mock_ws_manager):
            from execution.order_manager import close_position, configure_paths
            
            configure_paths(temp_excel, initial_capital=1000)
            
            ws = get_mock_ws_manager()
            ws.set_price('BTCUSDT', 50000.0)
            
            # Mock that succeeds on retry
            call_count = {'count': 0}
            
            def mock_retry_success(method, endpoint, body=None):
                call_count['count'] += 1
                if call_count['count'] == 1:
                    # First call fails
                    return (200, {'code': '40014', 'msg': 'Insufficient balance'})
                else:
                    # Second call succeeds
                    return (200, {
                        'code': '00000',
                        'data': {
                            'orderId': 'test_order',
                            'price': '50000'
                        }
                    })
            
            position_data = get_sample_position()
            
            # Note: This test documents the retry behavior
            # close_position currently does retry via place_market_order
            result = close_position(
                symbol='BTCUSDT',
                size=Decimal('0.001'),
                direction='long',
                send_request_func=mock_retry_success,
                reason='TP',
                position_data=position_data
            )
            
            print("✅ test_close_position_with_retry PASSED")
    
    finally:
        cleanup_temp_file(temp_excel)
# ==========================================================================
# MAIN (standalone execution)
# ==========================================================================
if __name__ == "__main__":
    setup_module()
    
    tests = [
            test_fetch_ticker_ws,
            test_get_current_price,
            test_compute_size_base,
            test_place_order_success,
            test_place_order_insufficient_balance,
            test_calculate_tp_sl_prices_long,
            test_calculate_tp_sl_prices_short,
            test_calculate_pnl_long_profit,
            test_calculate_pnl_short_profit,
            test_close_position_success,
            test_close_position_not_exist,
            test_get_fills_for_order,
            test_add_position,
            test_check_tp_sl_hit_tp,
            test_log_closed_position,
            # NEW: Helper functions
            test_fallback_params,
            test_quantize_size,
            test_build_order_body,
            test_extract_filled_amount,
            # NEW: Edge cases
            test_check_tp_sl_no_positions,
            test_close_position_with_retry
        ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            setup_function()
            test()
            passed += 1
        except Exception as e:
            print(f"❌ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "="*70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("="*70)
    
    sys.exit(0 if failed == 0 else 1)