"""
Configuration utility functions
"""
import os
from typing import Dict
from config.settings import ACCOUNTS, PERSISTENCE_DIR


def get_account_paths(account_number: str) -> dict:
    """Get all file paths for a specific account."""
    base_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        PERSISTENCE_DIR,
        f'bot_files_{account_number}'
    )
    
    return {
        'base_dir': base_dir,
        'state_file': os.path.join(base_dir, f'bot_state_{account_number}.json'),
        'trades_file': os.path.join(base_dir, f'bot_trades_{account_number}.xlsx'),
        'log_file': os.path.join(base_dir, f'BOT_orchestator_{account_number}.log')
    }


def get_account_config(account_number: str) -> Dict:
    """Get complete configuration for an account."""
    if account_number not in ACCOUNTS:
        available = ', '.join(ACCOUNTS.keys())
        raise ValueError(
            f"Invalid account number: {account_number}. "
            f"Available: {available}"
        )
    
    config = ACCOUNTS[account_number].copy()
    config['account_number'] = account_number
    config['paths'] = get_account_paths(account_number)
    
    return config


def get_strategies_yaml_path(account_number: str) -> str:
    """
    Get path to strategies YAML file for specific account.
    
    Each account has its own YAML file with pre-configured strategies
    and order amounts already adjusted for that account.
    
    Args:
        account_number: Account identifier ('00', 'E1', '01')
    
    Returns:
        Full path to strategies_XX.yaml file
    
    Raises:
        ValueError: If account number is invalid
        FileNotFoundError: If YAML file doesn't exist
    
    Example:
        >>> get_strategies_yaml_path('00')
        '/path/to/config/strategies_00.yaml'
        
        >>> get_strategies_yaml_path('E1')
        '/path/to/config/strategies_E1.yaml'
    """
    if account_number not in ACCOUNTS:
        available = ', '.join(ACCOUNTS.keys())
        raise ValueError(
            f"Invalid account number: {account_number}. "
            f"Available: {available}"
        )
    
    # Get config directory (same directory as this file)
    config_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Build path to account-specific YAML
    yaml_filename = f"strategies_{account_number}.yaml"
    yaml_path = os.path.join(config_dir, yaml_filename)
    
    # Verify file exists
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(
            f"Strategies YAML not found for account {account_number}: {yaml_path}\n"
            f"Expected file: {yaml_filename}"
        )
    
    return yaml_path