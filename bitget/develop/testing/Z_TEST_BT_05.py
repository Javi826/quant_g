"""
correlation_analysis.py
Standalone script to analyze profit and drawdown correlation between strategies.
Aligned with frontend /api/correlation-matrix logic (daily profit/DD, not cumulative).
"""

import pandas as pd

# =============================================================================
# CONFIG
# =============================================================================

TRADES_FILE  = "/home/javi/projects/quant/quant_g/bitget/BOT_trading/persistence/bot_files_00/bot_trades_00.xlsx"
ROUND        = 2

# =============================================================================
# LOAD & PREPARE
# =============================================================================

df = pd.read_excel(TRADES_FILE)
df['CLOSE_AT'] = pd.to_datetime(df['CLOSE_AT'])
df['date']     = df['CLOSE_AT'].dt.date

# Shorten strategy name to leading number (e.g. "04_reversal_short_4H" -> "04")
df['STRAT_ID'] = df['STRATEGY'].str.extract(r'^(\d+)')

strategies = sorted(df['STRAT_ID'].unique())

# =============================================================================
# BUILD DAILY SERIES (same logic as frontend endpoint)
# =============================================================================

def build_profit_pivot() -> pd.DataFrame:
    """Daily profit per strategy — discrete, not cumulative."""
    series = {}
    for strat in strategies:
        daily = df[df['STRAT_ID'] == strat].groupby('date')['PROFIT'].sum()
        series[strat] = daily
    return pd.DataFrame(series).fillna(0)

def build_dd_pivot(profit_pivot: pd.DataFrame) -> pd.DataFrame:
    """Daily DD per strategy — drop from peak of daily cumsum (same as endpoint)."""
    def _dd(col):
        cum = col.cumsum()
        return cum - cum.cummax()
    return profit_pivot.apply(_dd)

# =============================================================================
# CORRELATION MATRICES
# =============================================================================

profit_pivot = build_profit_pivot()
dd_pivot     = build_dd_pivot(profit_pivot)

profit_corr  = profit_pivot.corr().round(ROUND)
dd_corr      = dd_pivot.corr().round(ROUND)

# =============================================================================
# OUTPUT
# =============================================================================
print(build_dd_pivot(profit_pivot).head(10))
pd.set_option('display.max_columns',  None)
pd.set_option('display.max_rows',     None)
pd.set_option('display.width',        None)
pd.set_option('display.float_format', lambda x: f'{x:.2f}')

print("\n" + "=" * 80)
print("PROFIT CORRELATION MATRIX")
print("=" * 80)
print(profit_corr.to_string())

print("\n" + "=" * 80)
print("DD CORRELATION MATRIX")
print("=" * 80)
print(dd_corr.to_string())