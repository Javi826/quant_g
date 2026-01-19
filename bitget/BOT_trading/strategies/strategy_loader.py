"""
Strategy Loader

Loads strategy configurations from per-account YAML files.

Each account has its own YAML file:
  - config/strategies_00.yaml (Main Account)
  - config/strategies_E1.yaml (Elite Account)
  - config/strategies_01.yaml (Testing Account)

The YAML files contain only the strategies for that account,
with order_amount already adjusted appropriately.
"""

import os
import yaml
import logging
from typing import List, Dict, Optional

logger = logging.getLogger('BOT_trading.strategies.strategy_loader')


def load_strategies_from_yaml(yaml_path: str) -> List[Dict]:
    """
    Load all strategies from YAML configuration file.
    
    Args:
        yaml_path: Full path to YAML file
    
    Returns:
        List of strategy dictionaries
    
    Raises:
        FileNotFoundError: If YAML file doesn't exist
        ValueError: If YAML is invalid or missing required keys
    
    Example:
        >>> strategies = load_strategies_from_yaml('/path/to/strategies_00.yaml')
        >>> len(strategies)
        16
    """
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Strategies YAML file not found: {yaml_path}")
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if not data or 'strategies' not in data:
            raise ValueError(
                f"Invalid YAML format in {yaml_path}: missing 'strategies' key"
            )
        
        strategies = data['strategies']
        
        if not strategies:
            raise ValueError(
                f"No strategies found in {yaml_path}. "
                f"YAML must contain at least one strategy."
            )
        
        # Validate each strategy has required keys
        for i, strat in enumerate(strategies):
            try:
                validate_strategy_config(strat)
            except ValueError as e:
                raise ValueError(
                    f"Strategy #{i+1} in {yaml_path} is invalid: {e}"
                )
        
        logger.info(
            f"Loaded {len(strategies)} strategies from {os.path.basename(yaml_path)}"
        )
        return strategies
    
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax in {yaml_path}: {e}")
    except Exception as e:
        raise ValueError(f"Error loading YAML {yaml_path}: {e}")


def load_strategies(
    account_number: str,
    yaml_path: Optional[str] = None
) -> List[Dict]:
    """
    Load strategies for a specific account.
    
    This is the main entry point for loading strategies. It automatically
    determines the correct YAML file based on the account number.
    
    Args:
        account_number: Account identifier ('00', 'E1', '01')
        yaml_path: Optional custom path to YAML file. If None, uses
                  config/strategies_{account_number}.yaml
    
    Returns:
        List of strategy configurations for this account
    
    Raises:
        ValueError: If account number is invalid
        FileNotFoundError: If YAML file doesn't exist
    
    Example:
        >>> # Load strategies for account 00
        >>> strategies = load_strategies('00')
        >>> len(strategies)
        16
        
        >>> # Load strategies for account E1
        >>> strategies = load_strategies('E1')
        >>> len(strategies)
        15
        
        >>> # Load strategies for testing account
        >>> strategies = load_strategies('01')
        >>> len(strategies)
        2
    """
    if yaml_path is None:
        # Auto-detect YAML path based on account
        from config.utils import get_strategies_yaml_path
        yaml_path = get_strategies_yaml_path(account_number)
        logger.debug(f"Auto-detected YAML path for account {account_number}: {yaml_path}")
    
    # Load all strategies from the account-specific YAML
    # No filtering needed - the YAML only contains strategies for this account
    return load_strategies_from_yaml(yaml_path)


def apply_set_active_argument(
    strategies: List[Dict],
    active_ids: List[str]
) -> None:
    """
    Apply --set-active command line argument.
    
    Sets 'active' flag to True only for specified strategy IDs.
    Modifies strategies in-place.
    
    This is useful for temporarily activating only specific strategies
    without modifying the YAML file.
    
    Args:
        strategies: List of strategy configurations
        active_ids: List of strategy IDs to set as active
    
    Raises:
        ValueError: If any active_id is not found in strategies
    
    Example:
        >>> strategies = load_strategies('00')
        >>> # Only activate strategies 06 and 07
        >>> apply_set_active_argument(strategies, ['06_reversal_long_1H', '07_reversal_short_1H'])
        >>> active = [s['id'] for s in strategies if s.get('active', True)]
        >>> len(active)
        2
    """
    # Verify all requested IDs exist
    available_ids = {s['id'] for s in strategies}
    missing_ids = set(active_ids) - available_ids
    
    if missing_ids:
        raise ValueError(
            f"Strategy IDs not found: {', '.join(sorted(missing_ids))}. "
            f"Available IDs: {', '.join(sorted(available_ids))}"
        )
    
    # Set active flags
    for strat in strategies:
        if strat['id'] in active_ids:
            strat['active'] = True
        else:
            strat['active'] = False
    
    active_count = sum(1 for s in strategies if s.get('active', True))
    logger.info(
        f"Applied --set-active: {active_count}/{len(strategies)} strategies active"
    )


def validate_strategy_config(strategy: Dict) -> None:
    """
    Validate that strategy configuration has all required keys.
    
    Args:
        strategy: Strategy configuration dictionary
    
    Raises:
        ValueError: If any required key is missing or invalid
    """
    required_keys = [
        'id', 'name', 'timeframe', 'active', 'direction',
        'sell_after_ncandles', 'order_amount', 'tp_pct', 'sl_pct'
    ]
    
    missing_keys = [key for key in required_keys if key not in strategy]
    
    if missing_keys:
        raise ValueError(
            f"Strategy '{strategy.get('id', 'UNKNOWN')}' missing required keys: "
            f"{', '.join(missing_keys)}"
        )
    
    # Additional validation
    strat_id = strategy.get('id', 'UNKNOWN')
    
    if strategy['order_amount'] <= 0:
        raise ValueError(
            f"Strategy '{strat_id}' has invalid order_amount: "
            f"{strategy['order_amount']} (must be > 0)"
        )
    
    if strategy['tp_pct'] <= 0:
        raise ValueError(
            f"Strategy '{strat_id}' has invalid tp_pct: "
            f"{strategy['tp_pct']} (must be > 0)"
        )
    
    if strategy['sl_pct'] <= 0:
        raise ValueError(
            f"Strategy '{strat_id}' has invalid sl_pct: "
            f"{strategy['sl_pct']} (must be > 0)"
        )


def get_all_strategy_ids(account_number: str) -> List[str]:
    """
    Get list of all strategy IDs for a specific account.
    
    Args:
        account_number: Account identifier ('00', 'E1', '01')
    
    Returns:
        List of strategy IDs for this account
    
    Example:
        >>> ids = get_all_strategy_ids('00')
        >>> '01_double_top_long_4H' in ids
        True
        >>> len(ids)
        16
    """
    strategies = load_strategies(account_number)
    return [s['id'] for s in strategies]


def get_strategy_config(account_number: str, strategy_id: str) -> Dict:
    """
    Get configuration for a single strategy from an account.
    
    Args:
        account_number: Account identifier ('00', 'E1', '01')
        strategy_id: Strategy ID to retrieve
    
    Returns:
        Strategy configuration dictionary
    
    Raises:
        ValueError: If strategy ID not found in account's strategies
    
    Example:
        >>> config = get_strategy_config('00', '06_reversal_long_1H')
        >>> config['order_amount']
        40
    """
    strategies = load_strategies(account_number)
    
    for strat in strategies:
        if strat['id'] == strategy_id:
            return strat
    
    available_ids = [s['id'] for s in strategies]
    raise ValueError(
        f"Strategy ID '{strategy_id}' not found in account {account_number}. "
        f"Available IDs: {', '.join(available_ids)}"
    )


if __name__ == '__main__':
    """Quick test of strategy loading"""
    import sys
    
    print("Testing strategy loader...")
    print("-" * 60)
    
    for account in ['00', 'E1', '01']:
        try:
            strategies = load_strategies(account)
            active_count = sum(1 for s in strategies if s.get('active', True))
            
            print(f"\n✓ Account {account}:")
            print(f"  Total strategies: {len(strategies)}")
            print(f"  Active strategies: {active_count}")
            print(f"  First strategy: {strategies[0]['id']}")
            
        except Exception as e:
            print(f"\n✗ Account {account}: ERROR")
            print(f"  {e}")
            sys.exit(1)
    
    print("\n" + "-" * 60)
    print("✓ All tests passed!")