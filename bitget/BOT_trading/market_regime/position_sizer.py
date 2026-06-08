#BOT_trading/market_regime/position_sizer.py
"""
Position sizing based on market regime and direction alignment.

Calculates adjusted order amounts by applying multipliers from:
- regime_trending/ranging/volatile: Strategy's own regime multipliers
- DIRECTION_MATRIX: Strategy direction mode vs market direction alignment
"""

import logging

class PositionSizer:
    """
    Handles position sizing adjustments based on market regime.

    Uses 3 bin flags from strategy config (regime_ranging / regime_trending / regime_neutral)
    to determine if a strategy is blocked or allowed in the current market condition.

    Usage:
        sizer = PositionSizer(logger)
        adjusted_amount, metadata = sizer.calculate_adjusted_amount(
            base_amount=40.0,
            strat=strat,
            market_regime='trending',
        )
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def calculate_adjusted_amount(
        self,
        base_amount:   float,
        strat:         dict,
        market_regime: str = 'neutral',
    ) -> tuple:
        """
        Calculate adjusted order amount based on regime bin flag.

        Constructs bin key as f"regime_{market_regime}",
        looks up the flag in strat config (0 = blocked, 1 = allowed).

        Args:
            base_amount   : Base order amount from strategy config
            strat         : Strategy config dict with 3 bin flags
            market_regime : Current market regime ('trending' | 'ranging' | 'neutral')

        Returns:
            Tuple of (adjusted_amount, metadata_dict)
        """
        bin_key = f"regime_{market_regime}"
        flag    = strat.get(bin_key, 1)
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
        """
        Format standardized log message for sizing decision.

        Args:
            strategy_id: Strategy identifier
            metadata   : Metadata dict from calculate_adjusted_amount()

        Returns:
            Formatted log string
        """
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