#BOT_trading/execution/brokers/base_client.py
"""
Base Broker Client - Abstract base class for exchange clients.

This module provides an abstract base class that can be implemented
for different exchanges (Bitget, Binance, Bybit, etc.).

This is optional but follows best practices for extensibility.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Tuple, Optional


class BaseBrokerClient(ABC):
    """
    Abstract base class for exchange API clients.
    
    This allows the bot to support multiple exchanges in the future
    by implementing this interface for each exchange.
    
    Example:
        >>> class BinanceClient(BaseBrokerClient):
        ...     def send_request(self, method, path, params, body):
        ...         # Binance-specific implementation
        ...         pass
    """
    
    @abstractmethod
    def send_request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        body: Optional[Dict[str, Any]] = None
    ) -> Tuple[int, Any]:
        """
        Send authenticated request to exchange API.
        
        Args:
            method: HTTP method (GET/POST)
            path: API endpoint path
            params: Query parameters
            body: Request body
        
        Returns:
            Tuple of (status_code, response_data)
        """
        pass
    
    @abstractmethod
    def get_open_positions(
        self,
        product_type: str = "USDT-FUTURES"
    ) -> List[Dict[str, Any]]:
        """
        Get all open positions.
        
        Args:
            product_type: Product type
        
        Returns:
            List of position dictionaries
        """
        pass
    
    @abstractmethod
    def get_usdt_balance(self, exchange=None) -> float:
        """
        Get USDT balance.
        
        Args:
            exchange: Exchange object (optional, for compatibility)
        
        Returns:
            USDT balance
        """
        pass
