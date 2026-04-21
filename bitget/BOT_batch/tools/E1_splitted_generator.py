#BOT_batch/strategies_params/generator_dict_strategies.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "BOT_trading")))

from config.strategies_E1 import STRATEGIES as PROD_STRATEGIES

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_BATCH = os.path.join(os.path.dirname(__file__), "..", "strategies_files", "strategies_BT_batch.py")
OUTPUT_LOOP  = os.path.join(os.path.dirname(__file__), "..", "strategies_files", "strategies_loop.py")
PARAM_GRID_KEYS   = {"lookback", "tolerance", "ma_period", "tp_pct", "sl_pct", "impulse", "flag"}
SIGNAL_PARAM_KEYS = ("lookback", "tolerance", "ma_period", "impulse", "flag")
REGIME_BIN_KEYS   = (
    "regime_trending_uptrend", "regime_trending_dwtrend",
    "regime_ranging_uptrend",  "regime_ranging_dwtrend",
    "regime_volatile_uptrend", "regime_volatile_dwtrend",
)

symbols_live_folder = os.path.join(os.path.dirname(__file__), "..", "..", "BOT_trading", "symbols_live")
DEFAULT_N_SYMBOLS    = 10
DEFAULT_ORDER_AMOUNT = 80
USE_SYMBOLS_LIVE_FOR_N = True


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
# GENERATE strategies_batch.py
# =============================================================================
def generate_batch():
    lines = [
        '"""',
        'strategies_batch.py — Input for BOT_batch. Do not edit manually.',
        'Generated from strategies_E1.py. Re-run this script after each production deploy.',
        '"""',
        '',
        'STRATEGIES = [',
    ]

    for s in PROD_STRATEGIES:
        lines.append("    {")
        lines.append(f'        "id": "{s["id"]}",')
        lines.append(f'        "name": "{s["name"]}",')
        lines.append(f'        "timeframe": "{s["timeframe"]}",')
        lines.append(f'        "active": {s.get("active", False)},')
        lines.append(f'        "direction": "{s["direction"]}",')
        for bin_key in REGIME_BIN_KEYS:
            lines.append(f'        "{bin_key}": {float(s.get(bin_key, 1.0))},')
        lines.append(f'        "sell_after_ncandles": {s.get("sell_after_ncandles", 0)},')
        lines.append(f'        "order_amount_prod": {s.get("order_amount", 200)},')
        for k in SIGNAL_PARAM_KEYS:
            if k in s:
                lines.append(f'        "{k}": {_fmt_val(s[k])},')
        for k in ("tp_pct", "sl_pct"):
            if k in s:
                lines.append(f'        "{k}": {_fmt_val(s[k])},')
        lines.append("    },")

    lines.append("]")
    _write(OUTPUT_BATCH, lines)
    print(f"✅ strategies_batch.py generated → {os.path.abspath(OUTPUT_BATCH)}")


# =============================================================================
# GENERATE strategies_loop.py
# =============================================================================

def generate_loop():
    lines = [
        '"""',
        'strategies_loop.py — Batch loop configuration.',
        'Edit param_grid, n_symbols and order_amount before each run.',
        'This file is NOT updated by the batch automatically.',
        '"""',
        '',
        'STRATEGIES_LOOP = [',
    ]

    for s in PROD_STRATEGIES:
        live_path = os.path.join(symbols_live_folder, f"symbols_live_{s['id']}_{s['timeframe']}.csv")
        if USE_SYMBOLS_LIVE_FOR_N and os.path.exists(live_path):
            import pandas as pd
            n_symbols = len(pd.read_csv(live_path, header=None))
        else:
            n_symbols = DEFAULT_N_SYMBOLS

        param_grid = {"SELL_AFTER": [s.get("sell_after_ncandles", 0)]}
        for k in SIGNAL_PARAM_KEYS:
            if k in s:
                param_grid[k.upper()] = [s[k]]
        for k in ("tp_pct", "sl_pct"):
            if k in s:
                param_grid[k.upper()] = [s[k]]

        lines.append("    {")
        lines.append(f'        "id": "{s["id"]}",')
        lines.append(f'        "n_symbols": {n_symbols},')
        lines.append(f'        "order_amount": {DEFAULT_ORDER_AMOUNT},')
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