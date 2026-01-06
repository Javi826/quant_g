"""
Strategy Loader

Loads strategy configurations from YAML file.
"""

import os
import yaml
import logging
from typing import List, Dict, Optional

logger = logging.getLogger('BOT_trading.strategies.strategy_loader')


def load_strategies_from_yaml(yaml_path: Optional[str] = None) -> List[Dict]:
    """
    Load all strategies from YAML configuration file.
    
    Args:
        yaml_path: Path to YAML file. If None, uses default path.
    
    Returns:
        List of strategy dictionaries
    
    Raises:
        FileNotFoundError: If YAML file doesn't exist
        ValueError: If YAML is invalid
    """
    if yaml_path is None:
        # Default: strategies/strategies.yaml (in same directory as this file)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        yaml_path = os.path.join(current_dir, 'strategies.yaml')
    
    if not os.path.exists(yaml_path):
        raise FileNotFoundError(f"Strategies YAML file not found: {yaml_path}")
    
    try:
        with open(yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        if not data or 'strategies' not in data:
            raise ValueError("Invalid YAML format: missing 'strategies' key")
        
        strategies = data['strategies']
        
        # Validate each strategy has required keys
        for strat in strategies:
            validate_strategy_config(strat)
        
        logger.info(f"Loaded {len(strategies)} strategies from {yaml_path}")
        return strategies
    
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML syntax: {e}")
    except Exception as e:
        raise ValueError(f"Error loading YAML: {e}")


def filter_strategies_by_ids(
    all_strategies: List[Dict],
    strategy_ids: List[str]
) -> List[Dict]:
    """
    Filter strategies to only include those with matching IDs.
    
    Args:
        all_strategies: List of all strategy configurations
        strategy_ids: List of strategy IDs to include
    
    Returns:
        Filtered list of strategies
    
    Raises:
        ValueError: If any requested strategy ID is not found
    """
    filtered = []
    available_ids = {s['id'] for s in all_strategies}
    
    for strat_id in strategy_ids:
        matching = [s for s in all_strategies if s['id'] == strat_id]
        
        if not matching:
            raise ValueError(
                f"Strategy ID '{strat_id}' not found in YAML. "
                f"Available IDs: {sorted(available_ids)}"
            )
        
        filtered.append(matching[0])
    
    logger.info(
        f"Filtered to {len(filtered)}/{len(all_strategies)} strategies: "
        f"{', '.join(strategy_ids)}"
    )
    
    return filtered


def load_strategies(
    strategy_ids: List[str],
    yaml_path: Optional[str] = None
) -> List[Dict]:
    """
    Main entry point: Load and filter strategies.
    
    Args:
        strategy_ids: List of strategy IDs to load
        yaml_path: Optional custom path to YAML file
    
    Returns:
        List of strategy configurations
    
    Example:
        >>> strategies = load_strategies(['01_double_top_long_2m', '02_reversal_long_5m'])
        >>> len(strategies)
        2
    """
    all_strategies = load_strategies_from_yaml(yaml_path)
    return filter_strategies_by_ids(all_strategies, strategy_ids)


def apply_set_active_argument(
    strategies: List[Dict],
    active_ids: List[str]
) -> None:
    """
    Apply --set-active command line argument.
    
    Sets 'active' flag to True only for specified strategy IDs.
    Modifies strategies in-place.
    
    Args:
        strategies: List of strategy configurations
        active_ids: List of strategy IDs to set as active
    
    Example:
        >>> strategies = [{'id': 'A', 'active': True}, {'id': 'B', 'active': True}]
        >>> apply_set_active_argument(strategies, ['A'])
        >>> strategies[0]['active']
        True
        >>> strategies[1]['active']
        False
    """
    for strat in strategies:
        if strat['id'] in active_ids:
            strat['active'] = True
        else:
            strat['active'] = False
    
    active_count = sum(1 for s in strategies if s.get('active', True))
    logger.info(f"Set {active_count}/{len(strategies)} strategies as active")


def validate_strategy_config(strategy: Dict) -> None:
    """
    Validate that strategy configuration has all required keys.
    
    Args:
        strategy: Strategy configuration dictionary
    
    Raises:
        ValueError: If any required key is missing
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


def get_all_strategy_ids(yaml_path: Optional[str] = None) -> List[str]:
    """
    Get list of all available strategy IDs from YAML.
    
    Args:
        yaml_path: Optional custom path to YAML file
    
    Returns:
        List of strategy IDs
    
    Example:
        >>> ids = get_all_strategy_ids()
        >>> '01_double_top_long_2m' in ids
        True
    """
    all_strategies = load_strategies_from_yaml(yaml_path)
    return [s['id'] for s in all_strategies]


# For backward compatibility
def get_strategy_config(strategy_id: str, yaml_path: Optional[str] = None) -> Dict:
    """
    Get configuration for a single strategy.
    
    Args:
        strategy_id: Strategy ID to retrieve
        yaml_path: Optional custom path to YAML file
    
    Returns:
        Strategy configuration dictionary
    
    Raises:
        ValueError: If strategy ID not found
    """
    all_strategies = load_strategies_from_yaml(yaml_path)
    
    for strat in all_strategies:
        if strat['id'] == strategy_id:
            return strat
    
    raise ValueError(f"Strategy ID '{strategy_id}' not found")


if __name__ == '__main__':
    # Quick test
    import sys
    
    try:
        strategies = load_strategies_from_yaml()
                
        for s in strategies:
            active = "ACTIVE" if s.get('active', True) else "INACTIVE"
            #print(f"  {s['id']:<30} {s['timeframe']:<5} {active}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)
