# shared_batchs/runs/run_best_combinations.py
import logging
import numpy as np
import pandas as pd
from itertools import combinations as _combinations

from shared_batchs.utils.batch_metrics import compute_metrics

logger = logging.getLogger("BOT_batch.runs.run_best_combinations")


# =============================================================================
# BEST R² COMBINATION
# =============================================================================

def find_best_r2_combination_ids(
    strategy_trades_is: list,
    initial_balance: float,
    precomputed_metrics: dict = None,
) -> list:
    """Find the strategy combination with highest R² on IS trade logs."""
    named   = {sid: df for sid, df in strategy_trades_is}
    metrics = precomputed_metrics or {
        sid: compute_metrics(df, capital=initial_balance, name=sid)
        for sid, df in named.items()
    }

    best_r2    = -1.0
    best_combo = list(named.keys())

    for r in range(1, len(named) + 1):
        for combo in _combinations(named.keys(), r):
            if len(combo) == 1:
                r2 = metrics.get(combo[0], {}).get("R_Squared", -1.0)
            else:
                combo_tl = pd.concat(
                    [named[sid] for sid in combo], ignore_index=True
                ).sort_values("sell_time").reset_index(drop=True)
                r2 = compute_metrics(combo_tl, capital=initial_balance * len(combo), name="")["R_Squared"]

            if r2 > best_r2:
                best_r2    = r2
                best_combo = list(combo)

    logger.debug(f"Best R² combination on IS: {best_combo} — R²={best_r2:.3f}")
    return best_combo