"""
csv_to_strategies.py

Converts strategies_params.csv back to a production-ready STRATEGIES list.
Output: strategies_E1_from_csv.py (same directory as the CSV)

IMPORTANT:
- Batch columns are discarded (not part of strategy definition)
- Optional fields (tolerance, ma_period, impulse, flag, ranges) only written if non-null
- Types are enforced via explicit schema — no inference from CSV
- Full validation runs before writing any output file
"""

import os
import sys
import pandas as pd

# =============================================================================
# PATHS
# =============================================================================
CSV_PATH    = "/home/javi/projects/quant/quant_g/bitget/BOT_batch/strategies_params/strategies_params.csv"
OUTPUT_PATH = "/home/javi/projects/quant/quant_g/bitget/BOT_batch/strategies_params/strategies_E1_from_csv.py"

# =============================================================================
# SCHEMA
# Required fields: always present, type-enforced.
# Optional fields: only written if non-null in CSV.
# =============================================================================
REQUIRED_FIELDS: dict[str, type] = {
    "id":                  str,
    "name":                str,
    "timeframe":           str,
    "active":              bool,
    "direction":           str,
    "regime_trending":     float,
    "regime_ranging":      float,
    "regime_volatile":     float,
    "direction_mode":      str,
    "sell_after_ncandles": int,
    "order_amount":        int,
    "lookback":            int,
    "tp_pct":              float,
    "sl_pct":              float,
}

OPTIONAL_FIELDS: dict[str, type] = {
    "tolerance": int,
    "ma_period": int,
    "impulse":   float,
    "flag":      int,
    "ranges":    int,
}

BATCH_COLUMNS = {
    "last_run", "bt_netgain_pct", "bt_r2",
    "prob_negative", "validated",
    "last_change_active", "last_change_params", "last_change_regime",
}

FIELD_ORDER = [
    "id", "name", "timeframe", "active", "direction",
    "regime_trending", "regime_ranging", "regime_volatile",
    "direction_mode", "sell_after_ncandles", "order_amount",
    "lookback", "tolerance", "ma_period",
    "impulse", "flag", "ranges",
    "tp_pct", "sl_pct",
]

VALID_DIRECTIONS      = {"long", "short"}
VALID_DIRECTION_MODES = {"long_only", "short_only", "general"}
VALID_TIMEFRAMES      = {"1H", "4H", "6Hutc", "1D"}
VALID_REGIME_VALUES   = {0.0, 1.0}

# =============================================================================
# TYPE CASTING
# =============================================================================
def cast_value(value, target_type: type, field: str):
    """Cast a CSV value to the target type with strict validation."""
    if pd.isna(value):
        raise ValueError(f"Field '{field}' is null — required fields must have a value")

    if target_type == bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            if value.strip().lower() == "true":
                return True
            if value.strip().lower() == "false":
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        raise ValueError(f"Field '{field}': cannot cast '{value}' to bool")

    try:
        return target_type(value)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Field '{field}': cannot cast '{value}' to {target_type.__name__} — {e}") from e


def cast_optional(value, target_type: type, field: str):
    """Cast an optional field; returns None if null."""
    if pd.isna(value):
        return None
    return cast_value(value, target_type, field)


# =============================================================================
# VALIDATION
# =============================================================================
def validate_strategy(strategy: dict, row_index: int) -> list[str]:
    """Return a list of validation error messages for a single strategy."""
    errors = []
    sid = strategy.get("id", f"<row {row_index}>")

    def err(msg):
        errors.append(f"[{sid}] {msg}")

    if strategy.get("direction") not in VALID_DIRECTIONS:
        err(f"invalid direction '{strategy.get('direction')}'")

    if strategy.get("direction_mode") not in VALID_DIRECTION_MODES:
        err(f"invalid direction_mode '{strategy.get('direction_mode')}'")

    if strategy.get("timeframe") not in VALID_TIMEFRAMES:
        err(f"invalid timeframe '{strategy.get('timeframe')}'")

    for regime_field in ("regime_trending", "regime_ranging", "regime_volatile"):
        val = strategy.get(regime_field)
        if val not in VALID_REGIME_VALUES:
            err(f"{regime_field}={val} must be 0.0 or 1.0")

    for pct_field in ("tp_pct", "sl_pct"):
        val = strategy.get(pct_field)
        if val is not None and not (0 < val <= 100):
            err(f"{pct_field}={val} must be between 0 and 100")

    for positive_field in ("order_amount", "lookback", "sell_after_ncandles", "tolerance", "ma_period"):
        val = strategy.get(positive_field)
        if val is not None and val <= 0:
            err(f"{positive_field}={val} must be positive")

    return errors


# =============================================================================
# FORMATTING
# =============================================================================
def format_value(value) -> str:
    """Format a Python value as a valid Python literal string."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, str):
        return f"'{value}'"
    if isinstance(value, float):
        return repr(round(value, 10))
    return repr(value)


def strategy_to_str(strategy: dict) -> str:
    """Render a strategy dict as a formatted Python dict string, in canonical field order."""
    lines = ["    {"]
    for key in FIELD_ORDER:
        if key in strategy:
            lines.append(f"        '{key}': {format_value(strategy[key])},")
    lines.append("    }")
    return "\n".join(lines)


# =============================================================================
# MAIN
# =============================================================================
def main():
    if not os.path.exists(CSV_PATH):
        print(f"❌ CSV not found: {CSV_PATH}")
        sys.exit(1)

    df = pd.read_csv(CSV_PATH)
    print(f"✅ CSV loaded — {len(df)} rows, {len(df.columns)} columns")

    missing_cols = [f for f in REQUIRED_FIELDS if f not in df.columns]
    if missing_cols:
        print(f"❌ Missing required columns: {missing_cols}")
        sys.exit(1)

    strategies = []
    all_errors = []

    for i, row in df.iterrows():
        strategy = {}

        for field, target_type in REQUIRED_FIELDS.items():
            try:
                strategy[field] = cast_value(row[field], target_type, field)
            except ValueError as e:
                all_errors.append(f"[row {i}] {e}")

        for field, target_type in OPTIONAL_FIELDS.items():
            if field in df.columns:
                try:
                    casted = cast_optional(row[field], target_type, field)
                    if casted is not None:
                        strategy[field] = casted
                except ValueError as e:
                    all_errors.append(f"[row {i}] {e}")

        row_errors = validate_strategy(strategy, i)
        all_errors.extend(row_errors)
        strategies.append(strategy)

    if all_errors:
        print(f"\n❌ Validation failed — {len(all_errors)} error(s):\n")
        for e in all_errors:
            print(f"   {e}")
        print("\n⛔ Output file NOT written.")
        sys.exit(1)

    strategy_blocks = ",\n".join(strategy_to_str(s) for s in strategies)
    file_header = (
        '''"""\n'''
        "BOT_trading/config/strategies_E1.py Trading Strategies Configuration\n\n"
        "This file defines all trading strategies used by the bot.\n"
        "Each strategy must have all required parameters defined.\n\n"
        "IMPORTANT:\n"
        "- Strategy IDs must match those in IMPLEMENTED_STRATEGIES\n"
        "- All strategies must be listed here even if inactive\n"
        "- Parameter validation happens at bot startup\n"
        '''"""\n\n'''
    )
    output = f"{file_header}STRATEGIES = [\n{strategy_blocks},\n]\n"

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write(output)

    print(f"\n✅ strategies_E1_from_csv.py written → {OUTPUT_PATH}")
    print(f"   Strategies: {len(strategies)}")
    print(f"   Active: {sum(1 for s in strategies if s['active'])}")


if __name__ == "__main__":
    main()