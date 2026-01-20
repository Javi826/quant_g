"""
Position sizing based on market regime and direction alignment.

Calculates adjusted order amounts by applying multipliers from:
- regime_trending/ranging/volatile: Strategy's own regime multipliers (from YAML)
- DIRECTION_MATRIX: Strategy direction mode vs market direction alignment
"""

import logging
from typing import Dict, Tuple, Optional

from config.settings import (
    REGIME_GENERAL,
    DIRECTION_GENERAL,
    DIRECTION_MATRIX
)


class PositionSizer:
    """
    Handles position sizing adjustments based on market conditions.
    
    Separates sizing logic from orchestration, making it:
    - Testable independently
    - Reusable across components (backtester, CLI tools)
    - Maintainable in isolation
    
    Usage:
        sizer = PositionSizer(logger)
        adjusted_amount, metadata = sizer.calculate_adjusted_amount(
            base_amount=40.0,
            regime_trending=1.5,
            regime_ranging=1.0,
            regime_volatile=1.0,
            direction_mode='long_only',
            market_regime='trending',
            market_direction='uptrend'
        )
    """
    
    def __init__(self, logger: logging.Logger):
        """
        Initialize position sizer.
        
        Args:
            logger: Logger instance for sizing decisions
        """
        self.logger = logger
    
    def calculate_adjusted_amount(
        self,
        base_amount: float,
        regime_trending: float = 1.0,
        regime_ranging: float = 1.0,
        regime_volatile: float = 1.0,
        direction_mode: Optional[str] = 'general',
        market_regime: str = 'ranging',
        market_direction: str = 'uptrend'
    ) -> Tuple[float, Dict]:
        """
        Calculate adjusted order amount based on regime/direction alignment.
        
        This implements the EXACT logic from orchestrator._search_signals():
        1. Get regime multiplier (regime_trending/ranging/volatile vs market_regime)
        2. Get direction multiplier (direction_mode vs market_direction)
        3. Multiply both: final_mult = regime_mult × direction_mult
        4. Apply to base: adjusted_amount = base_amount × final_mult
        
        Args:
            base_amount: Base order amount from strategy config
            regime_trending: Strategy's trending multiplier
            regime_ranging: Strategy's ranging multiplier
            regime_volatile: Strategy's volatile multiplier
            direction_mode: Strategy's direction mode ('long_only', 'short_only', 'general')
            market_regime: Current market regime from classifier
            market_direction: Current market direction from classifier
        
        Returns:
            Tuple of (adjusted_amount, metadata_dict)
            
            metadata_dict contains:
                - base_amount: Original amount
                - market_regime: Current regime
                - market_direction: Current direction
                - regime_multiplier: Applied regime multiplier
                - regime_source: Source of regime multiplier
                - direction_multiplier: Applied direction multiplier
                - direction_source: Source of direction multiplier
                - final_multiplier: Combined multiplier
                - adjusted_amount: Final adjusted amount
                - blocked: Whether strategy is blocked (mult=0)
        
        Example:
            >>> sizer = PositionSizer(logger)
            >>> amount, meta = sizer.calculate_adjusted_amount(
            ...     base_amount=40.0,
            ...     regime_trending=1.8,
            ...     regime_ranging=1.0,
            ...     regime_volatile=0.5,
            ...     direction_mode='long_only',
            ...     market_regime='trending',
            ...     market_direction='uptrend'
            ... )
            >>> print(f"Adjusted: ${amount:.2f}, Multiplier: {meta['final_multiplier']:.1f}x")
            Adjusted: $72.00, Multiplier: 1.8x
        """
        # STEP 1: Calculate REGIME multiplier (cloned from orchestrator)
        regime_mult, regime_source = self._get_regime_multiplier(
            regime_trending, regime_ranging, regime_volatile, market_regime
        )
        
        # STEP 2: Calculate DIRECTION multiplier (cloned from orchestrator)
        direction_mult, dir_source = self._get_direction_multiplier(
            direction_mode, market_direction
        )
        
        # STEP 3: MULTIPLY both multipliers (cloned from orchestrator)
        final_mult = regime_mult * direction_mult
        
        # STEP 4: Calculate adjusted amount
        adjusted_amount = base_amount * final_mult
        
        # Build metadata dict for logging/debugging
        metadata = {
            'base_amount': base_amount,
            'market_regime': market_regime,
            'market_direction': market_direction,
            'regime_multiplier': regime_mult,
            'regime_source': regime_source,
            'direction_multiplier': direction_mult,
            'direction_source': dir_source,
            'final_multiplier': final_mult,
            'adjusted_amount': adjusted_amount,
            'blocked': (final_mult == 0)
        }
        
        return adjusted_amount, metadata
    
    def _get_regime_multiplier(
        self,
        regime_trending: float,
        regime_ranging: float,
        regime_volatile: float,
        market_regime: str
    ) -> Tuple[float, str]:
        """
        Get regime multiplier for strategy.
        
        Selects the appropriate multiplier based on current market regime.
        
        Args:
            regime_trending: Strategy's trending multiplier
            regime_ranging: Strategy's ranging multiplier
            regime_volatile: Strategy's volatile multiplier
            market_regime: Current market regime
        
        Returns:
            Tuple of (multiplier, source_description)
        """
        # Select multiplier based on market regime
        if market_regime == 'trending':
            mult = regime_trending
        elif market_regime == 'ranging':
            mult = regime_ranging
        elif market_regime == 'volatile':
            mult = regime_volatile
        else:
            # Fallback to REGIME_GENERAL if unknown regime
            mult = REGIME_GENERAL.get(market_regime, 1.0)
        
        source = 'strategy-specific'
        
        return mult, source
    
    def _get_direction_multiplier(
        self,
        direction_mode: Optional[str],
        market_direction: str
    ) -> Tuple[float, str]:
        """
        Get direction multiplier for strategy.
        
        EXACT logic from orchestrator STEP 2:
        - If direction_mode == 'general' → use DIRECTION_GENERAL
        - Else if direction_mode exists → use DIRECTION_MATRIX[mode][direction]
        - Else → no adjustment (1.0)
        
        Args:
            direction_mode: Strategy's direction mode
            market_direction: Current market direction
        
        Returns:
            Tuple of (multiplier, source_description)
        """
        if direction_mode == 'general':
            mult = DIRECTION_GENERAL[market_direction]
            source = 'general'
        elif direction_mode:
            mult = DIRECTION_MATRIX[direction_mode][market_direction]
            source = 'strategy-specific'
        else:
            # No direction mode defined
            mult = 1.0
            source = 'none'
        
        return mult, source
    
    def format_log_message(
        self,
        strategy_id: str,
        metadata: Dict
    ) -> str:
        """
        Format standardized log message for sizing decision.
        
        EXACT format from orchestrator logs:
        - Blocked: "[SIZING] Skip {id}: regime=X(Nx), dir=Y(Nx), final=0x → BLOCKED"
        - Active: "[SIZING] {id}: Market=[X, Y] | Base=$N × regime(Nx) × dir(Nx) = $N"
        
        Args:
            strategy_id: Strategy identifier
            metadata: Metadata dict from calculate_adjusted_amount()
        
        Returns:
            Formatted log string
        """
        if metadata['blocked']:
            # Blocked strategy format
            return (
                f"[SIZING] Skip {strategy_id}: "
                f"regime={metadata['market_regime']}({metadata['regime_multiplier']}x), "
                f"dir={metadata['market_direction']}({metadata['direction_multiplier']}x), "
                f"final=0x → BLOCKED"
            )
        else:
            # Active strategy format
            return (
                f"[SIZING] {strategy_id}: "
                f"Market=[{metadata['market_regime']}, {metadata['market_direction']}] | "
                f"Base=${metadata['base_amount']:.0f} × "
                f"regime({metadata['regime_multiplier']:.1f}x) × "
                f"dir({metadata['direction_multiplier']:.1f}x) = "
                f"${metadata['adjusted_amount']:.0f}"
            )