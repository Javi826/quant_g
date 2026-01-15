# config/utils.py
"""
Configuration utility functions
"""
import os
from typing import Dict, List
from .settings import ACCOUNTS, ACCOUNT_STRATEGIES, PERSISTENCE_DIR


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


def get_account_strategies(account_number: str) -> List[str]:
    """Get list of strategy IDs assigned to an account."""
    if account_number not in ACCOUNT_STRATEGIES:
        available = ', '.join(ACCOUNT_STRATEGIES.keys())
        raise ValueError(
            f"Invalid account number: {account_number}. "
            f"Available: {available}"
        )
    
    return ACCOUNT_STRATEGIES[account_number]