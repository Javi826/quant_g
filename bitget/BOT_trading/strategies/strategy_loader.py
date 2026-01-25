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
import logging
from typing import List, Dict, Optional

logger = logging.getLogger('BOT_trading.strategies.strategy_loader')


def load_strategies(account_number: str, yaml_path: Optional[str] = None) -> List[Dict]:
    # Importar módulo según cuenta
    if account_number == '00':
        from config.strategies_00 import STRATEGIES
    elif account_number == 'E1':
        from config.strategies_E1 import STRATEGIES
    elif account_number == '01':
        from config.strategies_01 import STRATEGIES
    else:
        raise ValueError(f"Unknown account: {account_number}")
    
    logger.info(f"Loaded {len(STRATEGIES)} strategies for account {account_number}")
    return STRATEGIES


def apply_set_active_argument(
    strategies: List[Dict],
    active_ids: List[str]
) -> None:

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

    strategies = load_strategies(account_number)
    return [s['id'] for s in strategies]


def get_strategy_config(account_number: str, strategy_id: str) -> Dict:

    strategies = load_strategies(account_number)
    
    for strat in strategies:
        if strat['id'] == strategy_id:
            return strat
    
    available_ids = [s['id'] for s in strategies]
    raise ValueError(
        f"Strategy ID '{strategy_id}' not found in account {account_number}. "
        f"Available IDs: {', '.join(available_ids)}"
    )

# =============================================================================
# def load_strategies_from_yaml(yaml_path: str) -> List[Dict]:
#     """
#     Load all strategies from YAML configuration file.
#     
#     Args:
#         yaml_path: Full path to YAML file
#     
#     Returns:
#         List of strategy dictionaries
#     
#     Raises:
#         FileNotFoundError: If YAML file doesn't exist
#         ValueError: If YAML is invalid or missing required keys
#     
#     Example:
#         >>> strategies = load_strategies_from_yaml('/path/to/strategies_00.yaml')
#         >>> len(strategies)
#         16
#     """
#     if not os.path.exists(yaml_path):
#         raise FileNotFoundError(f"Strategies YAML file not found: {yaml_path}")
#     
#     try:
#         with open(yaml_path, 'r', encoding='utf-8') as f:
#             data = yaml.safe_load(f)
#         
#         if not data or 'strategies' not in data:
#             raise ValueError(
#                 f"Invalid YAML format in {yaml_path}: missing 'strategies' key"
#             )
#         
#         strategies = data['strategies']
#         
#         if not strategies:
#             raise ValueError(
#                 f"No strategies found in {yaml_path}. "
#                 f"YAML must contain at least one strategy."
#             )
#         
#         # Validate each strategy has required keys
#         for i, strat in enumerate(strategies):
#             try:
#                 validate_strategy_config(strat)
#             except ValueError as e:
#                 raise ValueError(
#                     f"Strategy #{i+1} in {yaml_path} is invalid: {e}"
#                 )
#         
#         logger.info(
#             f"Loaded {len(strategies)} strategies from {os.path.basename(yaml_path)}"
#         )
#         return strategies
#     
#     except yaml.YAMLError as e:
#         raise ValueError(f"Invalid YAML syntax in {yaml_path}: {e}")
#     except Exception as e:
#         raise ValueError(f"Error loading YAML {yaml_path}: {e}")
# 
# =============================================================================