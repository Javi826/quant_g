#BOT_trading/market_regime/position_sizer.py
"""
Position sizing based on market regime and direction alignment.

Calculates adjusted order amounts by applying multipliers from:
- regime_trending/ranging/volatile: Strategy's own regime multipliers
- DIRECTION_MATRIX: Strategy direction mode vs market direction alignment
"""

import logging

class PositionSizer:


    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def calculate_adjusted_amount(
        self,
        base_amount:   float,
        strat:         dict,
        market_regime: str = 'neutral',
    ) -> tuple:

        bin_key = f"regime_{market_regime}"
        if bin_key not in strat:
            raise KeyError(f"[SIZING] '{bin_key}' not found in strategy '{strat.get('id')}'")
        flag = strat[bin_key]
        blocked = flag == 0

        adjusted_amount = 0.0 if blocked else base_amount

        metadata = {
            'base_amount':     base_amount,
            'market_regime':   market_regime,
            'bin_key':         bin_key,
            'flag':            flag,
            'adjusted_amount': adjusted_amount,
            'blocked':         blocked,
        }

        return adjusted_amount, metadata

    def format_log_message(self, strategy_id: str, metadata: dict) -> str:

        if metadata['blocked']:
            return (
                f"[SIZING] Skip {strategy_id}: "
                f"bin={metadata['bin_key']} → BLOCKED"
            )
        else:
            return (
                f"[SIZING] {strategy_id}: "
                f"bin={metadata['bin_key']} | "
                f"Base=${metadata['base_amount']:.0f} → ${metadata['adjusted_amount']:.0f}"
            )