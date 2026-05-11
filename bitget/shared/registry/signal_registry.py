#shared/registry/singal_registry.py
from signals.add_signals_parity      import parity_long, parity_short
from signals.add_signals_reversal    import reversal_long, reversal_short
from signals.add_signals_flag        import flag_long, flag_short
from signals.add_signals_orderblocks import orderblocks_long, orderblocks_short

SIGNAL_REGISTRY = {
    "parity_long":       {"fn": parity_long,       "params": ["lookback", "tolerance", "ma_period"]},
    "parity_short":      {"fn": parity_short,      "params": ["lookback", "tolerance", "ma_period"]},
    "reversal_long":     {"fn": reversal_long,     "params": ["lookback", "tolerance", "ma_period"]},
    "reversal_short":    {"fn": reversal_short,    "params": ["lookback", "tolerance", "ma_period"]},
    "flag_long":         {"fn": flag_long,         "params": ["lookback", "impulse", "flag", "ma_period"]},
    "flag_short":        {"fn": flag_short,        "params": ["lookback", "impulse", "flag", "ma_period"]},
    "orderblocks_long":  {"fn": orderblocks_long,  "params": ["lookback", "tolerance", "impulse"]},
    "orderblocks_short": {"fn": orderblocks_short, "params": ["lookback", "tolerance", "impulse"]},
}

SIGNAL_PARAM_KEYS = tuple({p for entry in SIGNAL_REGISTRY.values() for p in entry["params"]})
PARAM_KEYS = {"lookback", "tolerance", "ma_period", "tp_pct", "sl_pct", "impulse", "flag"}