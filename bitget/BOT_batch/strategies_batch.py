"""
strategies_batch.py — Source of truth for BOT_batch.

Static fields : strategy_id, signal, side, timeframe,
                direction_mode, order_amount_prod, sell_after_ncandles.

Dynamic fields (updated by batch): active, regime_trending,
                                   regime_ranging, regime_volatile,
                                   and all optimized params.
"""

STRATEGIES = [
    {
        # --- Identification ---
        "strategy_id": "02_reversal_long_4H",
        "signal": "reversal_long",
        "side": "long",
        "timeframe": "4H",

        # --- Production config (static) ---
        "direction_mode": "long_only",
        "order_amount_prod": 200,
        "sell_after_ncandles": 50,

        # --- Updated by batch ---
        "active": False,
        "regime_trending": 1.0,
        "regime_ranging": 1.0,
        "regime_volatile": 0.0,
        "tp_pct": 3,
        "ma_period": 50,
        "tolerance": 20,
        "sl_pct": 10,
        "lookback": 4,
    },
    {
        # --- Identification ---
        "strategy_id": "03_parity_long_4H",
        "signal": "parity_long",
        "side": "long",
        "timeframe": "4H",

        # --- Production config (static) ---
        "direction_mode": "general",
        "order_amount_prod": 200,
        "sell_after_ncandles": 50,

        # --- Updated by batch ---
        "active": False,
        "regime_trending": 0.0,
        "regime_ranging": 1.0,
        "regime_volatile": 0.0,
        "tp_pct": 3,
        "ma_period": 50,
        "tolerance": 40,
        "sl_pct": 10,
        "lookback": 150,
    },
    {
        # --- Identification ---
        "strategy_id": "04_reversal_short_4H",
        "signal": "reversal_short",
        "side": "short",
        "timeframe": "4H",

        # --- Production config (static) ---
        "direction_mode": "general",
        "order_amount_prod": 200,
        "sell_after_ncandles": 50,

        # --- Updated by batch ---
        "active": True,
        "regime_trending": 1.0,
        "regime_ranging": 1.0,
        "regime_volatile": 0.0,
        "tp_pct": 3,
        "ma_period": 50,
        "tolerance": 25,
        "sl_pct": 9,
        "lookback": 4,
    },
    {
        # --- Identification ---
        "strategy_id": "06_reversal_long_1H",
        "signal": "reversal_long",
        "side": "long",
        "timeframe": "1H",

        # --- Production config (static) ---
        "direction_mode": "long_only",
        "order_amount_prod": 200,
        "sell_after_ncandles": 100,

        # --- Updated by batch ---
        "active": False,
        "regime_trending": 0.0,
        "regime_ranging": 0.0,
        "regime_volatile": 1.0,
        "tp_pct": 2,
        "ma_period": 25,
        "tolerance": 40,
        "sl_pct": 10,
        "lookback": 7,
    },
    {
        # --- Identification ---
        "strategy_id": "07_reversal_short_1H",
        "signal": "reversal_short",
        "side": "short",
        "timeframe": "1H",

        # --- Production config (static) ---
        "direction_mode": "general",
        "order_amount_prod": 200,
        "sell_after_ncandles": 100,

        # --- Updated by batch ---
        "active": True,
        "regime_trending": 1.0,
        "regime_ranging": 1.0,
        "regime_volatile": 1.0,
        "tp_pct": 2.0,
        "ma_period": 50,
        "tolerance": 30,
        "sl_pct": 5,
        "lookback": 5,
    },
    {
        # --- Identification ---
        "strategy_id": "08_reversal_long_6Hutc",
        "signal": "reversal_long",
        "side": "long",
        "timeframe": "6Hutc",

        # --- Production config (static) ---
        "direction_mode": "long_only",
        "order_amount_prod": 200,
        "sell_after_ncandles": 50,

        # --- Updated by batch ---
        "active": False,
        "regime_trending": 1.0,
        "regime_ranging": 0.0,
        "regime_volatile": 0.0,
        "tp_pct": 4,
        "ma_period": 50,
        "tolerance": 20,
        "sl_pct": 10,
        "lookback": 3,
    },
    {
        # --- Identification ---
        "strategy_id": "09_reversal_short_6Hutc",
        "signal": "reversal_short",
        "side": "short",
        "timeframe": "6Hutc",

        # --- Production config (static) ---
        "direction_mode": "general",
        "order_amount_prod": 200,
        "sell_after_ncandles": 50,

        # --- Updated by batch ---
        "active": False,
        "regime_trending": 1.0,
        "regime_ranging": 1.0,
        "regime_volatile": 0.0,
        "tp_pct": 4,
        "ma_period": 25,
        "tolerance": 30,
        "sl_pct": 7.5,
        "lookback": 6,
    },
    {
        # --- Identification ---
        "strategy_id": "10_parity_long_1H",
        "signal": "parity_long",
        "side": "long",
        "timeframe": "1H",

        # --- Production config (static) ---
        "direction_mode": "long_only",
        "order_amount_prod": 200,
        "sell_after_ncandles": 75,

        # --- Updated by batch ---
        "active": False,
        "regime_trending": 0.0,
        "regime_ranging": 0.0,
        "regime_volatile": 1.0,
        "tp_pct": 2,
        "ma_period": 25,
        "tolerance": 15,
        "sl_pct": 10,
        "lookback": 150,
    },
    {
        # --- Identification ---
        "strategy_id": "11_parity_short_1H",
        "signal": "parity_short",
        "side": "short",
        "timeframe": "1H",

        # --- Production config (static) ---
        "direction_mode": "general",
        "order_amount_prod": 200,
        "sell_after_ncandles": 50,

        # --- Updated by batch ---
        "active": True,
        "regime_trending": 1.0,
        "regime_ranging": 1.0,
        "regime_volatile": 1.0,
        "tp_pct": 2,
        "ma_period": 50,
        "tolerance": 20,
        "sl_pct": 7.5,
        "lookback": 150,
    },
    {
        # --- Identification ---
        "strategy_id": "12_parity_long_6Hutc",
        "signal": "parity_long",
        "side": "long",
        "timeframe": "6Hutc",

        # --- Production config (static) ---
        "direction_mode": "long_only",
        "order_amount_prod": 200,
        "sell_after_ncandles": 50,

        # --- Updated by batch ---
        "active": False,
        "regime_trending": 0.0,
        "regime_ranging": 0.0,
        "regime_volatile": 1.0,
        "tp_pct": 3.5,
        "ma_period": 25,
        "tolerance": 40,
        "sl_pct": 10,
        "lookback": 50,
    },
    {
        # --- Identification ---
        "strategy_id": "13_orderblocks_short_4H",
        "signal": "orderblocks_short",
        "side": "short",
        "timeframe": "4H",

        # --- Production config (static) ---
        "direction_mode": "general",
        "order_amount_prod": 200,
        "sell_after_ncandles": 50,

        # --- Updated by batch ---
        "active": False,
        "regime_trending": 1.0,
        "regime_ranging": 1.0,
        "regime_volatile": 1.0,
        "tp_pct": 4,
        "impulse": 0.01,
        "tolerance": 35,
        "sl_pct": 11,
        "lookback": 50,
    },
    {
        # --- Identification ---
        "strategy_id": "16_ranging_short_6Hutc",
        "signal": "ranging_short",
        "side": "short",
        "timeframe": "6Hutc",

        # --- Production config (static) ---
        "direction_mode": "general",
        "order_amount_prod": 200,
        "sell_after_ncandles": 50,

        # --- Updated by batch ---
        "active": False,
        "regime_trending": 1.0,
        "regime_ranging": 1.0,
        "regime_volatile": 0.0,
        "tp_pct": 4,
        "ranges": 25,
        "tolerance": 5,
        "sl_pct": 6,
        "lookback": 10,
    },
    {
        # --- Identification ---
        "strategy_id": "17_flag_long_4H",
        "signal": "flag_long",
        "side": "long",
        "timeframe": "4H",

        # --- Production config (static) ---
        "direction_mode": "general",
        "order_amount_prod": 200,
        "sell_after_ncandles": 50,

        # --- Updated by batch ---
        "active": False,
        "regime_trending": 0.0,
        "regime_ranging": 1.0,
        "regime_volatile": 1.0,
        "tp_pct": 4,
        "impulse": 5,
        "ma_period": 50,
        "sl_pct": 10,
        "lookback": 13,
        "flag": 40,
    },
    {
        # --- Identification ---
        "strategy_id": "19_flag_short_4H",
        "signal": "flag_short",
        "side": "short",
        "timeframe": "4H",

        # --- Production config (static) ---
        "direction_mode": "general",
        "order_amount_prod": 200,
        "sell_after_ncandles": 50,

        # --- Updated by batch ---
        "active": False,
        "regime_trending": 1.0,
        "regime_ranging": 1.0,
        "regime_volatile": 1.0,
        "tp_pct": 3,
        "impulse": 3,
        "ma_period": 50,
        "sl_pct": 9,
        "lookback": 10,
        "flag": 50,
    },
    {
        # --- Identification ---
        "strategy_id": "20_flag_short_1H",
        "signal": "flag_short",
        "side": "short",
        "timeframe": "1H",

        # --- Production config (static) ---
        "direction_mode": "general",
        "order_amount_prod": 200,
        "sell_after_ncandles": 50,

        # --- Updated by batch ---
        "active": True,
        "regime_trending": 1.0,
        "regime_ranging": 1.0,
        "regime_volatile": 1.0,
        "tp_pct": 2,
        "impulse": 3,
        "ma_period": 25,
        "sl_pct": 8,
        "lookback": 20,
        "flag": 60,
    },
]
