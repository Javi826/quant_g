import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "BOT_trading")))

from config.strategies_E1 import STRATEGIES as PROD_STRATEGIES

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_BATCH = os.path.join(os.path.dirname(__file__), "..", "strategies_batch.py")
OUTPUT_LOOP  = os.path.join(os.path.dirname(__file__), "..", "strategies_loop.py")

# Params that are optimized by batch
PARAM_GRID_KEYS = {"lookback", "tolerance", "ma_period", "tp_pct", "sl_pct", "impulse", "flag", "ranges"}

# Default batch loop config (overridable per strategy)
DEFAULT_N_SYMBOLS          = 10
DEFAULT_ORDER_AMOUNT_BATCH = 80

# =============================================================================
# SIGNAL MAPPING — maps production 'name' to batch 'signal' key
# =============================================================================
SIGNAL_MAP = {
    "reversal_long":     "reversal_long",
    "reversal_short":    "reversal_short",
    "parity_long":       "parity_long",
    "parity_short":      "parity_short",
    "flag_long":         "flag_long",
    "flag_short":        "flag_short",
    "orderblocks_short": "orderblocks_short",
    "orderblocks_long":  "orderblocks_long",
    "ranging_long":      "ranging_long",
    "ranging_short":     "ranging_short",
}

# =============================================================================
# HELPERS
# =============================================================================
def _fmt_val(val):
    if isinstance(val, bool):
        return str(val)
    if isinstance(val, str):
        return f'"{val}"'
    return str(val)


def _write(path, lines):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# =============================================================================
# GENERATE strategies_batch.py — static + dynamic fields, no param_grid
# =============================================================================
def generate_batch():
    lines = [
        '"""',
        'strategies_batch.py — Source of truth for BOT_batch.',
        '',
        'Static fields : strategy_id, signal, side, timeframe,',
        '                direction_mode, order_amount_prod, sell_after_ncandles.',
        '',
        'Dynamic fields (updated by batch): active, regime_trending,',
        '                                   regime_ranging, regime_volatile,',
        '                                   and all optimized params.',
        '"""',
        '',
        'STRATEGIES = [',
    ]

    for s in PROD_STRATEGIES:
        sid  = s["id"]
        name = s["name"]
        tf   = s["timeframe"]
        side = s["direction"]

        signal_base = "_".join(name.split("_")[:-1])
        signal      = SIGNAL_MAP.get(signal_base, signal_base)

        lines.append("    {")
        lines.append(f'        # --- Identification ---')
        lines.append(f'        "strategy_id": "{sid}",')
        lines.append(f'        "signal": "{signal}",')
        lines.append(f'        "side": "{side}",')
        lines.append(f'        "timeframe": "{tf}",')
        lines.append(f'')
        lines.append(f'        # --- Production config (static) ---')
        lines.append(f'        "direction_mode": "{s.get("direction_mode", "general")}",')
        lines.append(f'        "order_amount_prod": {s.get("order_amount", 200)},')
        lines.append(f'        "sell_after_ncandles": {s.get("sell_after_ncandles", 0)},')
        lines.append(f'')
        lines.append(f'        # --- Updated by batch ---')
        lines.append(f'        "active": {s.get("active", False)},')
        lines.append(f'        "regime_trending": {float(s.get("regime_trending", 1.0))},')
        lines.append(f'        "regime_ranging": {float(s.get("regime_ranging", 1.0))},')
        lines.append(f'        "regime_volatile": {float(s.get("regime_volatile", 1.0))},')
        for k in PARAM_GRID_KEYS:
            if k in s:
                lines.append(f'        "{k}": {_fmt_val(s[k])},')
        lines.append("    },")

    lines.append("]")
    _write(OUTPUT_BATCH, lines)
    print(f"✅ strategies_batch.py generated → {os.path.abspath(OUTPUT_BATCH)}")


# =============================================================================
# GENERATE strategies_loop.py — param_grid + batch config, editable manually
# =============================================================================
def generate_loop():
    lines = [
        '"""',
        'strategies_loop.py — Batch loop configuration.',
        '',
        'Edit param_grid, n_symbols and order_amount before each run.',
        'This file is NOT updated by the batch automatically.',
        '"""',
        '',
        'STRATEGIES_LOOP = [',
    ]

    for s in PROD_STRATEGIES:
        sid = s["id"]

        param_grid = {"SELL_AFTER": [s.get("sell_after_ncandles", 0)]}
        for k in PARAM_GRID_KEYS:
            if k in s:
                param_grid[k.upper()] = [s[k]]

        lines.append("    {")
        lines.append(f'        "strategy_id": "{sid}",')
        lines.append(f'        "n_symbols": {DEFAULT_N_SYMBOLS},')
        lines.append(f'        "order_amount": {DEFAULT_ORDER_AMOUNT_BATCH},')
        lines.append(f'        "param_grid": {{')
        for pk, pv in param_grid.items():
            lines.append(f'            "{pk}": {pv},')
        lines.append(f'        }},')
        lines.append("    },")

    lines.append("]")
    _write(OUTPUT_LOOP, lines)
    print(f"✅ strategies_loop.py generated → {os.path.abspath(OUTPUT_LOOP)}")


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    generate_batch()
    generate_loop()
    print(f"   Strategies: {len(PROD_STRATEGIES)}")