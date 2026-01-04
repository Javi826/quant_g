"""
Professional logging system for trading bot.

This module provides a production-ready logging configuration with:
- Multiple log levels (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Console and file handlers
- Automatic log rotation
- Rich formatting with timestamps and module information
"""

import logging
import os
from logging.handlers import RotatingFileHandler


def setup_logger(
    log_dir: str,
    logfile_name: str = 'bot.log',
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> logging.Logger:
    """
    Setup professional logging system with console and file handlers.
    
    Args:
        log_dir: Directory where log files will be stored
        logfile_name: Name of the log file
        console_level: Minimum level for console output (default: INFO)
        file_level: Minimum level for file output (default: DEBUG)
        max_bytes: Maximum size of log file before rotation (default: 10MB)
        backup_count: Number of backup files to keep (default: 5)
    
    Returns:
        Configured root logger
    
    Example:
        >>> logger = setup_professional_logger('/path/to/logs', 'trading.log')
        >>> logger.info("Bot started")
    """
    # Create log directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, logfile_name)
    
    # Get root logger
    root_logger = logging.getLogger('BOT_trading')
    root_logger.setLevel(logging.DEBUG)  # Capture everything
    
    # Remove existing handlers to avoid duplicates
    root_logger.handlers.clear()
    root_logger.propagate = False
    
    # ========================================================================
    # CONSOLE HANDLER (Simple format for readability)
    # ========================================================================
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    
    # Simple format for console (no timestamps, just message)
    console_format = logging.Formatter('%(message)s')
    console_handler.setFormatter(console_format)
    
    # ========================================================================
    # FILE HANDLER (Detailed format with rotation)
    # ========================================================================
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding='utf-8'
    )
    file_handler.setLevel(file_level)
    
    # Detailed format for file
    #file_format = logging.Formatter('%(message)s')
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # ========================================================================
    # ADD HANDLERS TO LOGGER
    # ========================================================================
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    
    # Log initial message
    root_logger.info("=" * 60)
    root_logger.info("Logging system initialized")
    root_logger.info(f"Log file: {log_path}")
    root_logger.info(f"Console level: {logging.getLevelName(console_level)}")
    root_logger.info(f"File level: {logging.getLevelName(file_level)}")
    root_logger.info("=" * 60)
    
    return root_logger


def get_module_logger(module_name: str) -> logging.Logger:
    """
    Get a logger for a specific module.
    
    Args:
        module_name: Name of the module (e.g., 'execution.order_manager')
    
    Returns:
        Logger instance for the module
    
    Example:
        >>> logger = get_module_logger('execution.order_manager')
        >>> logger.info("Order placed")
    """
    return logging.getLogger(f'BOT_trading.{module_name}')


# ============================================================================
# BACKWARD COMPATIBILITY (for gradual migration)
# ============================================================================
def setup_print_logger(logdir: str, logfile_name: str = None):
    """
    Backward compatibility wrapper.
    Calls new professional logger setup.
    
    Args:
        logdir: Log directory
        logfile_name: Log file name
    """
    if logfile_name is None:
        logfile_name = 'bot.log'
    
    return setup_logger(
        log_dir=logdir,
        logfile_name=logfile_name
    )
