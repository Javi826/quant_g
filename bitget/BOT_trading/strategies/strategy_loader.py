#BOT_trading/strategies/strategy_loader.py

import logging
from typing import List, Dict

logger = logging.getLogger('BOT_trading.strategies.strategy_loader')


def load_strategies(account_number: str) -> List[Dict]:
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

