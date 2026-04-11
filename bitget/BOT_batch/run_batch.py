from main_batch import run_batch, run_portfolio_analysis

STRATEGIES = [
    {
        "strategy_id":  "03_parity_long_4H",
        "signal":       "parity_long",
        "side":         "long",
        "timeframe":    "4H",
        "n_symbols":    9,
        "order_amount": 80,
        "param_grid": {
# =============================================================================
#             "SELL_AFTER": [0],
#             "LOOKBACK":   [100, 150],
#             "TOLERANCE":  [15, 30, 45],
#             "MA_PERIOD":  [25],
#             "TP_PCT":     [2, 3, 4],
#             "SL_PCT":     [8, 9, 10],
# =============================================================================
            "SELL_AFTER": [0],
            "LOOKBACK":   [150],
            "TOLERANCE":  [40],
            "MA_PERIOD":  [50],
            "TP_PCT":     [3],
            "SL_PCT":     [10],
        },
    },
# =============================================================================
#     {
#         "strategy_id":  "11_parity_short_1H",
#         "signal":       "parity_short",
#         "side":         "short",
#         "timeframe":    "1H",
#         "n_symbols":    6,
#         "order_amount": 80,
#         "param_grid": {
# # =============================================================================
# #             "SELL_AFTER": [0],
# #             "LOOKBACK":   [100, 150],
# #             "TOLERANCE":  [15, 30, 45],
# #             "MA_PERIOD":  [25, 50],
# #             "TP_PCT":     [2, 3],
# #             "SL_PCT":     [7, 8, 9],
# # =============================================================================
#             "SELL_AFTER": [0],
#             "LOOKBACK":   [150],
#             "TOLERANCE":  [20],
#             "MA_PERIOD":  [50],
#             "TP_PCT":     [2],
#             "SL_PCT":     [7.5],
#         },
#     },
# =============================================================================
]

if __name__ == "__main__":
    for strategy in STRATEGIES:
        print(f"\n{'#'*60}")
        print(f"#  Running: {strategy['strategy_id']}")
        print(f"{'#'*60}")
        run_batch(strategy)
    run_portfolio_analysis()