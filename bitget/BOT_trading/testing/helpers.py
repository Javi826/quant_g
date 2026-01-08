"""
Testing Helpers - Shared mocks and fixtures for BOT_trading tests.

This module provides:
- MockWebSocketManager: Simulates WebSocket data
- mock_send_request: Simulates REST API calls
- Sample data for testing
"""

import time
from decimal import Decimal
from datetime import datetime
from typing import Dict, List, Optional, Any


# ==========================================================================
# MOCK WEBSOCKET MANAGER
# ==========================================================================
class MockWebSocketManager:
    """Mock WebSocket Manager for testing."""
    
    def __init__(self):
        self.prices = {}
        self.contracts = {}
        self.equity = {'USDT': {'available': '1000.0'}}
        self.fills = {}
        self.subscribed_public = set()
        
        # Preload common symbols
        self._setup_default_data()
    
    def _setup_default_data(self):
        """Setup default test data."""
        # Bitcoin
        self.prices['BTCUSDT'] = {
            'price': Decimal('50000.0'),
            'timestamp': time.time()
        }
        self.contracts['BTCUSDT'] = {
            'pricePlace': '1',
            'volumePlace': '3',
            'minTradeNum': '0.001',
            'sizeMultiplier': '1',
            'minTradeUSDT': '5'
        }
        
        # Ethereum
        self.prices['ETHUSDT'] = {
            'price': Decimal('3000.0'),
            'timestamp': time.time()
        }
        self.contracts['ETHUSDT'] = {
            'pricePlace': '2',
            'volumePlace': '2',
            'minTradeNum': '0.01',
            'sizeMultiplier': '1',
            'minTradeUSDT': '5'
        }
    
    def subscribe_ticker(self, symbol: str):
        """Simulate ticker subscription."""
        self.subscribed_public.add(symbol)
        
        # Add default price if not exists
        if symbol not in self.prices:
            self.prices[symbol] = {
                'price': Decimal('100.0'),
                'timestamp': time.time()
            }
    
    def get_contract(self, symbol: str) -> Optional[Dict]:
        """Get contract info."""
        return self.contracts.get(symbol)
    
    def get_usdt_balance(self) -> float:
        """Get USDT balance."""
        return float(self.equity.get('USDT', {}).get('available', '0'))
    
    def get_fills(self, order_id: str) -> Optional[List[Dict]]:
        """Get fills for order."""
        return self.fills.get(order_id)
    
    def set_price(self, symbol: str, price: float):
        """Set price for symbol (test helper)."""
        self.prices[symbol] = {
            'price': Decimal(str(price)),
            'timestamp': time.time()
        }
    
    def set_balance(self, balance: float):
        """Set balance (test helper)."""
        self.equity['USDT'] = {'available': str(balance)}
    
    def add_fills(self, order_id: str, fills: List[Dict]):
        """Add fills for order (test helper)."""
        self.fills[order_id] = fills


# Global mock instance
_mock_ws_manager = None


def get_mock_ws_manager() -> MockWebSocketManager:
    """Get or create global mock WebSocket manager."""
    global _mock_ws_manager
    if _mock_ws_manager is None:
        _mock_ws_manager = MockWebSocketManager()
    return _mock_ws_manager


def reset_mock_ws_manager():
    """Reset mock WebSocket manager."""
    global _mock_ws_manager
    _mock_ws_manager = MockWebSocketManager()


# ==========================================================================
# MOCK REST API (send_request_func)
# ==========================================================================
def mock_send_request_success(method: str, endpoint: str, body: Dict = None) -> tuple:
    """
    Mock successful REST API request.
    
    Returns:
        Tuple of (status_code, response_dict)
    """
    if endpoint == "/api/v2/mix/order/place-order":
        return (200, {
            "code": "00000",
            "msg": "success",
            "data": {
                "orderId": "mock_order_12345",
                "clientOid": body.get('clientOid', 'mock_client_oid'),
                "price": "50000.0",
                "size": body.get('size', '0.001'),
                "baseVolume": body.get('size', '0.001'),
                "filledQty": body.get('size', '0.001')
            }
        })
    
    return (200, {"code": "00000", "msg": "success", "data": {}})


def mock_send_request_insufficient_balance(method: str, endpoint: str, body: Dict = None) -> tuple:
    """Mock REST API request with insufficient balance error."""
    return (200, {
        "code": "40014",
        "msg": "Insufficient balance",
        "data": None
    })


def mock_send_request_position_not_exist(method: str, endpoint: str, body: Dict = None) -> tuple:
    """Mock REST API request with position not exist error."""
    return (200, {
        "code": "22002",
        "msg": "Position does not exist",
        "data": None
    })


# ==========================================================================
# SAMPLE DATA
# ==========================================================================
def get_sample_position() -> Dict:
    """Get sample position for testing."""
    return {
        'symbol': 'BTCUSDT',
        'size': Decimal('0.001'),
        'entry_price': Decimal('50000.0'),
        'direction': 'long',
        'tp': Decimal('52000.0'),  # +4%
        'sl': Decimal('45000.0'),  # -10%
        'order_id': 'mock_order_12345',
        'opened_at': datetime.now(),
        'usdt_amount': 50.0
    }


def get_sample_strategy_config() -> Dict:
    """Get sample strategy config for testing."""
    return {
        'id': '01_test_strategy',
        'name': 'test_strategy',
        'timeframe': '4H',
        'active': True,
        'direction': 'long',
        'sell_after_ncandles': 50,
        'order_amount': 50,
        'tp_pct': 4.0,
        'sl_pct': 10.0,
        'lookback': 100,
        'tolerance': 20
    }


def get_sample_fills() -> List[Dict]:
    """Get sample fills data."""
    return [
        {
            'baseVolume': '0.001',
            'price': '50000.0',
            'profit': '2.0',
            'feeDetail': [
                {'totalFee': '0.05'}
            ]
        }
    ]


# ==========================================================================
# TEST UTILITIES
# ==========================================================================
def patch_ws_manager(monkeypatch):
    """
    Patch WebSocket manager for testing.
    
    Usage in test:
        patch_ws_manager(monkeypatch)
    """
    reset_mock_ws_manager()
    mock = get_mock_ws_manager()
    
    # Patch get_ws_manager to return mock
    import market_data
    monkeypatch.setattr(market_data, 'get_ws_manager', lambda: mock)
    
    return mock


def create_temp_state_file(tmp_path):
    """Create temporary state file for testing."""
    state_file = tmp_path / "bot_state_test.json"
    return str(state_file)


def create_temp_trades_file(tmp_path):
    """Create temporary trades Excel file for testing."""
    trades_file = tmp_path / "bot_trades_test.xlsx"
    return str(trades_file)

# ==========================================================================
# TEMPORARY FILES
# ==========================================================================
import tempfile
import json


def create_temp_excel(filename='bot_trades_test.xlsx'):
    """Create temporary Excel file for testing."""
    temp_dir = tempfile.mkdtemp()
    return os.path.join(temp_dir, filename)


def create_temp_json(filename='bot_state_test.json'):
    """Create temporary JSON file for testing."""
    temp_dir = tempfile.mkdtemp()
    filepath = os.path.join(temp_dir, filename)
    
    # Initialize with empty state
    with open(filepath, 'w') as f:
        json.dump({
            'positions': {},
            'strategy_candles': {}
        }, f)
    
    return filepath


def cleanup_temp_file(filepath):
    """Remove temporary file."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
        # Remove parent dir if empty
        parent = os.path.dirname(filepath)
        if os.path.exists(parent) and not os.listdir(parent):
            os.rmdir(parent)
    except Exception:
        pass


# Add to imports at top
import os