#!/usr/bin/env python3
"""
develop/live_lab/live_lab.py vs Live-Demo Trade Comparison
Block-based validation of backtesting vs live-demo results — 1H strategies only
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
import matplotlib.pyplot as plt

# =============================================================================
# CONFIGURATION
# =============================================================================

LAB_FOLDER     = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "brief_trades")))
LIVE_DEMO_FILE = Path('/home/javi/projects/quant/quant_b/bitget/BOT_trading/persistence/bot_files_00/bot_trades_00.xlsx')

DATE_FROM      = '2026-03-25'
DATE_TO        = '2026-04-08'
BLOCK_DAYS     = 5
TIMEFRAME_FILTER = '4H'

# =============================================================================


def load_lab_trades(folder: Path, date_from: str, date_to: str, timeframe: str) -> pd.DataFrame:
    """Load and combine all lab trade files, filtered by date range and timeframe"""

    files = glob(str(folder / 'all_trades_*.csv'))

    if not files:
        print("⚠️  No lab trade files found")
        return pd.DataFrame()

    all_trades = []
    for filepath in files:
        df = pd.read_csv(filepath)
        df['sell_time'] = pd.to_datetime(df['sell_time'])
        df['buy_time']  = pd.to_datetime(df['buy_time'])
        all_trades.append(df)

    combined = pd.concat(all_trades, ignore_index=True)
    combined = combined.sort_values('buy_time').reset_index(drop=True)

    mask     = (combined['buy_time'] >= date_from) & (combined['buy_time'] <= date_to)
    filtered = combined[mask].copy()
    filtered = filtered[filtered['strategy'].str.endswith(timeframe)].copy()

    print(f"   Lab files found:  {len(files)}")
    print(f"   Lab trades total: {len(combined):,} → filtered ({timeframe}): {len(filtered):,}")

    return filtered


def load_live_trades(filepath: Path, date_from: str, date_to: str, timeframe: str) -> pd.DataFrame:
    """Load live-demo trade file, filtered by date range and timeframe"""

    if not filepath.exists():
        print(f"⚠️  Live-demo file not found: {filepath}")
        return pd.DataFrame()

    df             = pd.read_excel(filepath)
    df['CLOSE_AT'] = pd.to_datetime(df['CLOSE_AT'])
    df['OPEN_AT']  = pd.to_datetime(df['OPEN_AT'])
    df             = df.sort_values('OPEN_AT').reset_index(drop=True)

    mask     = (df['OPEN_AT'] >= date_from) & (df['OPEN_AT'] <= date_to)
    filtered = df[mask].copy()
    filtered = filtered[filtered['STRATEGY'].str.endswith(timeframe)].copy()

    print(f"   Live trades total: {len(df):,} → filtered ({timeframe}): {len(filtered):,}")

    return filtered


def calculate_metrics(df_subset: pd.DataFrame, profit_col: str) -> dict:
    """Calculate num_trades, win_rate and avg_profit"""

    total   = len(df_subset)
    winners = (df_subset[profit_col] > 0).sum() if total > 0 else 0
    wr      = (winners / total * 100) if total > 0 else 0
    avg     = df_subset[profit_col].mean() if total > 0 else 0

    return {
        'Trades':     total,
        'WR%':        round(wr, 1),
        'Avg_Profit': round(avg, 2),
    }


def generate_blocks(date_from: str, date_to: str, block_days: int) -> list[tuple]:
    """Generate list of (block_num, start, end) tuples"""

    start  = pd.Timestamp(date_from)
    end    = pd.Timestamp(date_to)
    blocks = []
    block  = 1
    cursor = start

    while cursor <= end:
        block_end = min(cursor + pd.Timedelta(days=block_days - 1), end)
        blocks.append((block, cursor, block_end))
        cursor = block_end + pd.Timedelta(days=1)
        block += 1

    return blocks


def build_system_table(df_lab: pd.DataFrame, df_live: pd.DataFrame, blocks: list[tuple]) -> pd.DataFrame:
    """Build system-level block comparison table"""

    rows = []

    for block_num, block_start, block_end in blocks:
        lab_block  = df_lab[(df_lab['buy_time'] >= block_start) & (df_lab['buy_time'] <= block_end)]
        live_block = df_live[(df_live['OPEN_AT'] >= block_start) & (df_live['OPEN_AT'] <= block_end)]

        lab_m  = calculate_metrics(lab_block, 'profit')
        live_m = calculate_metrics(live_block, 'PROFIT')

        rows.append({
            'Block':            block_num,
            'Date_From':        block_start.strftime('%Y-%m-%d'),
            'Date_To':          block_end.strftime('%Y-%m-%d'),
            'Lab_Trades':       lab_m['Trades'],
            'Live_Trades':      live_m['Trades'],
            'Delta_Trades':     live_m['Trades'] - lab_m['Trades'],
            'Lab_WR%':          lab_m['WR%'],
            'Live_WR%':         live_m['WR%'],
            'Delta_WR%':        round(live_m['WR%'] - lab_m['WR%'], 1),
            'Lab_Avg_Profit':   lab_m['Avg_Profit'],
            'Live_Avg_Profit':  live_m['Avg_Profit'],
            'Delta_Avg_Profit': round(live_m['Avg_Profit'] - lab_m['Avg_Profit'], 2),
        })

    return pd.DataFrame(rows)


def build_strategy_tables(df_lab: pd.DataFrame, df_live: pd.DataFrame, blocks: list[tuple]) -> dict[str, pd.DataFrame]:
    """Build per-strategy block comparison tables"""

    strategies = sorted(df_lab['strategy'].dropna().unique())
    tables     = {}

    for strategy in strategies:
        lab_strat  = df_lab[df_lab['strategy'] == strategy]
        live_strat = df_live[df_live['STRATEGY'] == strategy]

        rows = []

        for block_num, block_start, block_end in blocks:
            lab_block  = lab_strat[(lab_strat['buy_time'] >= block_start) & (lab_strat['buy_time'] <= block_end)]
            live_block = live_strat[(live_strat['OPEN_AT'] >= block_start) & (live_strat['OPEN_AT'] <= block_end)]

            lab_m  = calculate_metrics(lab_block, 'profit')
            live_m = calculate_metrics(live_block, 'PROFIT')

            rows.append({
                'Block':            block_num,
                'Date_From':        block_start.strftime('%Y-%m-%d'),
                'Date_To':          block_end.strftime('%Y-%m-%d'),
                'Lab_Trades':       lab_m['Trades'],
                'Live_Trades':      live_m['Trades'],
                'Delta_Trades':     live_m['Trades'] - lab_m['Trades'],
                'Lab_WR%':          lab_m['WR%'],
                'Live_WR%':         live_m['WR%'],
                'Delta_WR%':        round(live_m['WR%'] - lab_m['WR%'], 1),
                'Lab_Avg_Profit':   lab_m['Avg_Profit'],
                'Live_Avg_Profit':  live_m['Avg_Profit'],
                'Delta_Avg_Profit': round(live_m['Avg_Profit'] - lab_m['Avg_Profit'], 2),
            })

        tables[strategy] = pd.DataFrame(rows)

    return tables


def build_summary_tables(df_lab: pd.DataFrame, df_live: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build synthetic summary tables: system-level and per-strategy"""

    def _metrics(df: pd.DataFrame, profit_col: str) -> dict:
        total   = len(df)
        winners = (df[profit_col] > 0).sum() if total > 0 else 0
        return {
            'Trades':       total,
            'WR%':          round(winners / total * 100, 1) if total > 0 else 0.0,
            'Total_Profit': round(df[profit_col].sum(), 2),
            'Avg_Profit':   round(df[profit_col].mean(), 2) if total > 0 else 0.0,
        }

    def _build_row(label: str, lab_df: pd.DataFrame, live_df: pd.DataFrame) -> dict:
        lab_m  = _metrics(lab_df,  'profit')
        live_m = _metrics(live_df, 'PROFIT')
        return {
            'Strategy':          label,
            'Lab_Trades':        lab_m['Trades'],
            'Live_Trades':       live_m['Trades'],
            'Delta_Trades':      live_m['Trades'] - lab_m['Trades'],
            'Lab_WR%':           lab_m['WR%'],
            'Live_WR%':          live_m['WR%'],
            'Delta_WR%':         round(live_m['WR%'] - lab_m['WR%'], 1),
            'Lab_Total_Profit':  lab_m['Total_Profit'],
            'Live_Total_Profit': live_m['Total_Profit'],
            'Delta_Total':       round(live_m['Total_Profit'] - lab_m['Total_Profit'], 2),
            'Lab_Avg_Profit':    lab_m['Avg_Profit'],
            'Live_Avg_Profit':   live_m['Avg_Profit'],
            'Delta_Avg':         round(live_m['Avg_Profit'] - lab_m['Avg_Profit'], 2),
        }

    df_system = pd.DataFrame([_build_row('TOTAL', df_lab, df_live)])

    strategies    = sorted(df_lab['strategy'].dropna().unique())
    strat_rows    = [
        _build_row(s, df_lab[df_lab['strategy'] == s], df_live[df_live['STRATEGY'] == s])
        for s in strategies
    ]
    df_strategies = pd.DataFrame(strat_rows)

    return df_system, df_strategies


def print_table(df: pd.DataFrame, title: str) -> None:
    """Print clean formatted table"""

    print("\n" + "=" * 130)
    print(title)
    print("=" * 130)
    print()

    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)

    print(df.to_string(index=False))
    print()


def plot_wr_by_block(df_system: pd.DataFrame):
    """Plot per-block WR% and cumulative WR% for lab vs live"""

    blocks      = df_system['Block'].values
    lab_wr      = df_system['Lab_WR%'].values
    live_wr     = df_system['Live_WR%'].values
    lab_trades  = df_system['Lab_Trades'].values
    live_trades = df_system['Live_Trades'].values

    lab_cum_winners  = np.cumsum(lab_wr / 100 * lab_trades)
    lab_cum_trades   = np.cumsum(lab_trades)
    live_cum_winners = np.cumsum(live_wr / 100 * live_trades)
    live_cum_trades  = np.cumsum(live_trades)

    lab_cum_wr  = np.where(lab_cum_trades > 0,  lab_cum_winners  / lab_cum_trades  * 100, 0)
    live_cum_wr = np.where(live_cum_trades > 0, live_cum_winners / live_cum_trades * 100, 0)

    x_labels = df_system['Date_From'].values

    fig1, ax1 = plt.subplots(figsize=(14, 5))
    ax1.plot(blocks, lab_wr,  marker='o', linewidth=2, color='steelblue',  markersize=5, label='Lab WR%')
    ax1.plot(blocks, live_wr, marker='o', linewidth=2, color='darkorange', markersize=5, label='Live WR%')
    ax1.axhline(y=60, color='red', linestyle='--', linewidth=1, alpha=0.5, label='60% threshold')
    ax1.set_xticks(blocks)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    ax1.set_title(f'Win Rate per Block ({BLOCK_DAYS}d) — Lab vs Live [{TIMEFRAME_FILTER}]', fontsize=14, fontweight='bold')
    ax1.set_ylabel('WR%')
    ax1.set_ylim([0, 100])
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    plt.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.plot(blocks, lab_cum_wr,  marker='o', linewidth=2, color='steelblue',  markersize=5, label='Lab Cumulative WR%')
    ax2.plot(blocks, live_cum_wr, marker='o', linewidth=2, color='darkorange', markersize=5, label='Live Cumulative WR%')
    ax2.axhline(y=60, color='red', linestyle='--', linewidth=1, alpha=0.5, label='60% threshold')
    ax2.set_xticks(blocks)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    ax2.set_title(f'Cumulative Win Rate — Lab vs Live [{TIMEFRAME_FILTER}]', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Cumulative WR%')
    ax2.set_ylim([0, 100])
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()


def plot_equity_curve(df_lab: pd.DataFrame, df_live: pd.DataFrame) -> None:
    """Plot cumulative equity curves for lab vs live — USDT, aligned at first live data point"""

    lab_equity  = df_lab.sort_values('sell_time').copy()
    live_equity = df_live.sort_values('CLOSE_AT').copy()

    lab_equity['cum_profit']  = lab_equity['profit'].cumsum()
    live_equity['cum_profit'] = live_equity['PROFIT'].cumsum()

    live_start        = live_equity['CLOSE_AT'].iloc[0]
    lab_at_live_start = lab_equity[lab_equity['sell_time'] <= live_start]

    offset_usdt = lab_at_live_start['cum_profit'].iloc[-1] if not lab_at_live_start.empty else 0

    lab_equity['cum_profit']  -= offset_usdt
    live_equity['cum_profit'] -= live_equity['cum_profit'].iloc[0]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(lab_equity['sell_time'],  lab_equity['cum_profit'],  linewidth=2, color='steelblue',  label='Lab')
    ax.plot(live_equity['CLOSE_AT'],  live_equity['cum_profit'], linewidth=2, color='darkorange', label='Live')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5)
    ax.set_title(f'Cumulative Equity — USDT [{TIMEFRAME_FILTER}] (aligned at first live trade)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Profit (USDT)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()


def main():
    print("\n" + "=" * 130)
    print(f"LAB vs LIVE-DEMO COMPARISON — {TIMEFRAME_FILTER} strategies")
    print("=" * 130)
    print(f"\n   Period:     {DATE_FROM} → {DATE_TO}")
    print(f"   Block size: {BLOCK_DAYS} days")

    print("\n📂 Loading lab trades...")
    df_lab = load_lab_trades(LAB_FOLDER, DATE_FROM, DATE_TO, TIMEFRAME_FILTER)

    print("\n📂 Loading live-demo trades...")
    df_live = load_live_trades(LIVE_DEMO_FILE, DATE_FROM, DATE_TO, TIMEFRAME_FILTER)

    if df_lab.empty or df_live.empty:
        print("\n❌ Cannot compare: one or both datasets are empty.")
        return

    print(f"\n📅 Date ranges:")
    print(f"   Lab  buy_time:  {df_lab['buy_time'].min().date()} → {df_lab['buy_time'].max().date()}")
    print(f"   Live OPEN_AT:   {df_live['OPEN_AT'].min().date()} → {df_live['OPEN_AT'].max().date()}")

    blocks = generate_blocks(DATE_FROM, DATE_TO, BLOCK_DAYS)

    df_system = build_system_table(df_lab, df_live, blocks)
    print_table(df_system, f"SYSTEM — Lab vs Live ({BLOCK_DAYS}d blocks) [{TIMEFRAME_FILTER}]")

    strategy_tables = build_strategy_tables(df_lab, df_live, blocks)
    for strategy, df_strat in strategy_tables.items():
        print_table(df_strat, f"STRATEGY: {strategy} — Lab vs Live ({BLOCK_DAYS}d blocks)")

    df_summary_system, df_summary_strategies = build_summary_tables(df_lab, df_live)
    print_table(df_summary_system,     f"SUMMARY — TOTAL SYSTEM [{TIMEFRAME_FILTER}]")
    print_table(df_summary_strategies, f"SUMMARY — BY STRATEGY [{TIMEFRAME_FILTER}]")

    print("=" * 130)
    print("COMPARISON COMPLETE")
    print("=" * 130 + "\n")

    print("📊 Generating charts...")
    plot_wr_by_block(df_system)
    plot_equity_curve(df_lab, df_live)
    plt.show()
    print("✅ Charts displayed")


if __name__ == "__main__":
    main()