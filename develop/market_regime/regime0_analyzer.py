#!/usr/bin/env python3
"""
develop/market_regime/regime0_analyzer.py

Autonomous script that compares system performance with/without trend filtering.
Calculates BTC MAs on-the-fly - no pre-enrichment needed.

Usage:
    python regime_analyzer_STANDALONE.py
    
Parameters (edit at top of script):
    TRADES_FOLDER: Folder with all_trades_*.xlsx files
    BTC_FILE: Path to BTC 1D parquet
    MA_PERIOD: Moving average period for trend detection (5, 10, 20, 50, 200)
    INITIAL_CAPITAL: Capital per strategy (default 800)
"""
import os
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob

# =============================================================================
# CONFIGURATION - EDIT THESE PARAMETERS
# =============================================================================

TRADES_FOLDER   = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "BOT_batch", "brief_trades")
TRADES_FOLDER = os.path.join(os.path.dirname(__file__), "..", "..", "develop", "brief_trades")
BTC_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "data_pipeline", "data", "04_split", "expanding", "OOS", "crypto_2025-04_2026-04_OOS", "BTCUSDT_1Dutc.parquet")
MA_PERIOD       = 5  # Options: 5, 10, 20, 50, 200
LONG_TH         = 1.00  # Threshold for LONG: BTC > MA * LONG_TH
SHORT_TH        = 1.00  # Threshold for SHORT: BTC < MA * SHORT_TH
INITIAL_CAPITAL = 800

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_btc_1d(btc_file: str) -> pd.DataFrame:
    """Load BTC 1D data and calculate MA"""
    if not Path(btc_file).exists():
        raise FileNotFoundError(f"BTC file not found: {btc_file}")
    
    df = pd.read_parquet(btc_file)
    df.columns = df.columns.str.lower()
    
    if 'timestamp' in df.columns:
        df['ts'] = pd.to_datetime(df['timestamp'])
    else:
        df['ts'] = pd.to_datetime(df.index)
    
    df = df.sort_values('ts').reset_index(drop=True)
    
    # Calculate MA
    df[f'ma{MA_PERIOD}'] = df['close'].rolling(window=MA_PERIOD).mean()
    
    return df


def get_btc_value_at_trade(btc_df: pd.DataFrame, trade_time: pd.Timestamp) -> tuple:
    """Get BTC close and MA at trade time (only closed candles)"""
    closed_candles = btc_df[btc_df['ts'] < trade_time]
    
    if len(closed_candles) < MA_PERIOD:
        return None, None
    
    last_candle = closed_candles.iloc[-1]
    
    if pd.isna(last_candle[f'ma{MA_PERIOD}']):
        return None, None
    
    return last_candle['close'], last_candle[f'ma{MA_PERIOD}']


def detect_strategy_type(strategy_name: str) -> str:
    """Detect if strategy is LONG or SHORT based on name"""
    name_lower = strategy_name.lower()
    
    if '_long_' in name_lower or name_lower.endswith('_long'):
        return 'LONG'
    elif '_short_' in name_lower or name_lower.endswith('_short'):
        return 'SHORT'
    
    print(f"⚠️  Cannot detect type for '{strategy_name}', assuming LONG")
    return 'LONG'


def calculate_strategy_metrics(df: pd.DataFrame, initial_capital: float) -> dict:
    """Calculate key metrics for a strategy"""
    if len(df) == 0:
        return {
            'num_trades': 0,
            'total_profit': 0.0,
            'net_gain_pct': 0.0,
            'max_dd_pct': 0.0
        }
    
    df = df.sort_values('buy_time').copy()
    df['cumulative_profit'] = df['profit'].cumsum()
    df['balance'] = initial_capital + df['cumulative_profit']
    
    # Net gain
    final_balance = df['balance'].iloc[-1]
    net_gain_pct = (final_balance - initial_capital) / initial_capital * 100
    
    # Max DD
    cummax = df['balance'].cummax()
    drawdown_pct = ((df['balance'] - cummax) / cummax * 100)
    max_dd_pct = drawdown_pct.min()
    
    return {
        'num_trades': len(df),
        'total_profit': df['profit'].sum(),
        'net_gain_pct': net_gain_pct,
        'max_dd_pct': max_dd_pct
    }


def load_trades(filepath: str) -> pd.DataFrame:
    """Load trades from Excel file"""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.lower().str.strip()
    
    if 'buy_time' in df.columns:
        df['buy_time'] = pd.to_datetime(df['buy_time'])
    else:
        raise ValueError(f"File {filepath} missing 'buy_time' column")
    
    return df


def classify_trades_by_trend(df: pd.DataFrame, btc_df: pd.DataFrame, strategy_type: str) -> pd.DataFrame:
    """Add trend classification to each trade based on BTC MA with asymmetric thresholds"""
    df['trend'] = 'unknown'
    
    for idx, trade in df.iterrows():
        btc_close, ma_value = get_btc_value_at_trade(btc_df, trade['buy_time'])
        
        if btc_close is not None and ma_value is not None:
            if strategy_type == 'LONG':
                # LONG: BTC > MA * LONG_TH
                df.at[idx, 'trend'] = 'uptrend' if btc_close > ma_value * LONG_TH else 'downtrend'
            else:  # SHORT
                # SHORT: BTC < MA * SHORT_TH
                df.at[idx, 'trend'] = 'downtrend' if btc_close < ma_value * SHORT_TH else 'uptrend'
    
    return df


def analyze_strategy(filepath: str, btc_df: pd.DataFrame, initial_capital: float) -> dict:
    """Analyze single strategy with and without trend filter"""
    strategy = Path(filepath).stem.replace('all_trades_', '')
    df = load_trades(filepath)
    
    # Detect strategy type
    strategy_type = detect_strategy_type(strategy)
    
    # Classify trades by trend (with asymmetric thresholds)
    df = classify_trades_by_trend(df, btc_df, strategy_type)
    
    # SCENARIO A: WITHOUT FILTER (all trades)
    metrics_without = calculate_strategy_metrics(df, initial_capital)
    
    # SCENARIO B: WITH FILTER (only matching trend)
    if strategy_type == 'LONG':
        df_filtered = df[df['trend'] == 'uptrend'].copy()
    else:  # SHORT
        df_filtered = df[df['trend'] == 'downtrend'].copy()
    
    metrics_with = calculate_strategy_metrics(df_filtered, initial_capital)
    
    return {
        'strategy': strategy,
        'type': strategy_type,
        'filepath': filepath,
        'without_filter': metrics_without,
        'with_filter': metrics_with
    }


def calculate_global_portfolio(results: list, btc_df: pd.DataFrame, initial_capital: float, use_filter: bool = False) -> dict:
    """Calculate global portfolio metrics"""
    all_trades = []
    
    for r in results:
        df = load_trades(r['filepath'])
        df = classify_trades_by_trend(df, btc_df, r['type'])
        
        if use_filter:
            if r['type'] == 'LONG':
                df = df[df['trend'] == 'uptrend'].copy()
            else:  # SHORT
                df = df[df['trend'] == 'downtrend'].copy()
        
        all_trades.append(df[['buy_time', 'profit']].copy())
    
    if not all_trades:
        return {'num_trades': 0, 'total_profit': 0.0, 'net_gain_pct': 0.0, 'max_dd_pct': 0.0}
    
    combined_trades = pd.concat(all_trades, ignore_index=True)
    combined_trades = combined_trades.sort_values('buy_time').reset_index(drop=True)
    
    if len(combined_trades) == 0:
        return {'num_trades': 0, 'total_profit': 0.0, 'net_gain_pct': 0.0, 'max_dd_pct': 0.0}
    
    total_capital = initial_capital * len(results)
    
    # Calculate equity curve
    combined_trades['cumulative_profit'] = combined_trades['profit'].cumsum()
    combined_trades['balance'] = total_capital + combined_trades['cumulative_profit']
    
    # Net gain
    final_balance = combined_trades['balance'].iloc[-1]
    net_gain_pct = (final_balance - total_capital) / total_capital * 100
    
    # Max DD
    cummax = combined_trades['balance'].cummax()
    drawdown_pct = ((combined_trades['balance'] - cummax) / cummax * 100)
    max_dd_pct = drawdown_pct.min()
    
    return {
        'num_trades': len(combined_trades),
        'total_profit': combined_trades['profit'].sum(),
        'net_gain_pct': net_gain_pct,
        'max_dd_pct': max_dd_pct
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("REGIME ANALYZER - Trend Filtering Comparison (STANDALONE)")
    print("=" * 70)
    
    print(f"\nConfiguration:")
    print(f"  Trades folder: {TRADES_FOLDER}")
    print(f"  BTC file:      {BTC_FILE}")
    print(f"  MA period:     MA{MA_PERIOD}")
    print(f"  LONG TH:       {LONG_TH}")
    print(f"  SHORT TH:      {SHORT_TH}")
    print(f"  Capital:       ${INITIAL_CAPITAL}")
    
    print("\nComparison scenarios:")
    print("  WITHOUT FILTER: All trades")
    print(f"  WITH FILTER:    LONG when BTC > MA×{LONG_TH}, SHORT when BTC < MA×{SHORT_TH}")
    
    # Load BTC 1D
    print("\n📂 Loading BTC 1D data...")
    btc_df = load_btc_1d(BTC_FILE)
    print(f"✅ Loaded {len(btc_df)} daily bars")
    
    # Find all trades files
    pattern = str(Path(TRADES_FOLDER) / 'all_trades_*.csv')
    files = sorted(glob(pattern))
    
    if not files:
        print(f"\n❌ No trades files found in {TRADES_FOLDER}")
        return
    
    print(f"\n📂 Found {len(files)} strategy files")
    
    # Analyze each strategy
    print("\n🔍 Analyzing strategies...")
    results = []
    for filepath in files:
        result = analyze_strategy(filepath, btc_df, INITIAL_CAPITAL)
        results.append(result)
        print(f"   ✅ {result['strategy']}")
    
    # Calculate global portfolios
    global_without = calculate_global_portfolio(results, btc_df, INITIAL_CAPITAL, use_filter=False)
    global_with = calculate_global_portfolio(results, btc_df, INITIAL_CAPITAL, use_filter=True)
    # =============================================================================
    # DIAGNOSTIC BLOCK — paste at the end of regime0_analyzer.py
    # For each strategy: one plot (3 lines) + verification print
    # =============================================================================
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    def _build_equity(df, initial_capital):
        df = df.sort_values("buy_time").reset_index(drop=True)
        eq = initial_capital + df["profit"].cumsum().values
        pct = (eq - initial_capital) / initial_capital * 100
        return df["buy_time"].values, pct

    def _build_btc_trend(btc_df, start, end):
        mask = (btc_df["ts"] >= start) & (btc_df["ts"] <= end)
        sub  = btc_df[mask].copy()
        return sub["ts"].values, sub["close"].values, sub[f"ma{MA_PERIOD}"].values

    def _verify_condition(df_all, btc_df, strategy_type):
        rows = []
        for _, trade in df_all.iterrows():
            closed  = btc_df[btc_df["ts"] < trade["buy_time"]]
            if len(closed) < MA_PERIOD:
                direction = "unknown"
                btc_close = np.nan
                ma_val    = np.nan
            else:
                last      = closed.iloc[-1]
                btc_close = last["close"]
                ma_val    = last[f"ma{MA_PERIOD}"]
                if pd.isna(ma_val):
                    direction = "unknown"
                elif strategy_type == "LONG":
                    direction = "uptrend" if btc_close > ma_val * LONG_TH else "downtrend"
                else:
                    direction = "downtrend" if btc_close < ma_val * SHORT_TH else "uptrend"
            rows.append({
                "buy_time":  trade["buy_time"],
                "btc_close": round(btc_close, 2) if not np.isnan(btc_close) else np.nan,
                "ma":        round(ma_val, 2)    if not np.isnan(ma_val)    else np.nan,
                "direction": direction,
                "taken":     (strategy_type == "LONG" and direction == "uptrend") or
                             (strategy_type == "SHORT" and direction == "downtrend"),
            })
        return pd.DataFrame(rows)

    print("\n" + "=" * 80)
    print("  DIAGNOSTIC — Per-strategy plot + verification")
    print("=" * 80)

    for r in results:
        strategy_name = r["strategy"]
        strategy_type = r["type"]

        df_all  = load_trades(r["filepath"])
        df_all  = classify_trades_by_trend(df_all, btc_df, strategy_type)

        keep_trend = "uptrend" if strategy_type == "LONG" else "downtrend"
        df_filt = df_all[df_all["trend"] == keep_trend].copy()

        # --- Verification table ---
        verif = _verify_condition(df_all, btc_df, strategy_type)
        taken    = verif[verif["taken"]]
        discarded = verif[~verif["taken"] & (verif["direction"] != "unknown")]

        print(f"\n{'─'*80}")
        print(f"  {strategy_name}  [{strategy_type}]  |  MA{MA_PERIOD}  LONG_TH={LONG_TH}  SHORT_TH={SHORT_TH}")
        print(f"{'─'*80}")
        print(f"  Total trades   : {len(verif)}")
        print(f"  Taken          : {len(taken)}  "
              f"({len(taken)/len(verif)*100:.1f}%)  "
              f"date range: {taken['buy_time'].min().date() if len(taken) > 0 else 'N/A'} → "
              f"{taken['buy_time'].max().date() if len(taken) > 0 else 'N/A'}")
        print(f"  Discarded      : {len(discarded)}  "
              f"({len(discarded)/len(verif)*100:.1f}%)  "
              f"date range: {discarded['buy_time'].min().date() if len(discarded) > 0 else 'N/A'} → "
              f"{discarded['buy_time'].max().date() if len(discarded) > 0 else 'N/A'}")

        # Sample of discarded trades with BTC values
        if len(discarded) > 0:
            sample = discarded.tail(5)[["buy_time", "btc_close", "ma", "direction"]]
            print(f"\n  Last discarded trades (BTC close vs MA{MA_PERIOD}):")
            print(sample.to_string(index=False))

        # Days in uptrend vs downtrend over full BTC period
        start_d = df_all["buy_time"].min()
        end_d   = df_all["buy_time"].max()
        btc_sub = btc_df[(btc_df["ts"] >= start_d) & (btc_df["ts"] <= end_d)].dropna(subset=[f"ma{MA_PERIOD}"])
        n_up   = (btc_sub["close"] > btc_sub[f"ma{MA_PERIOD}"] * LONG_TH).sum()
        n_down = (btc_sub["close"] < btc_sub[f"ma{MA_PERIOD}"] * SHORT_TH).sum()
        print(f"\n  BTC days in period [{start_d.date()} → {end_d.date()}]:")
        print(f"    uptrend  : {n_up}  ({n_up/(n_up+n_down)*100:.1f}%)")
        print(f"    downtrend: {n_down}  ({n_down/(n_up+n_down)*100:.1f}%)")

        # --- Plot ---
        ts_all,  eq_all  = _build_equity(df_all,  INITIAL_CAPITAL)
        ts_filt, eq_filt = _build_equity(df_filt, INITIAL_CAPITAL)

        t_start = pd.Timestamp(df_all["buy_time"].min())
        t_end   = pd.Timestamp(df_all["buy_time"].max())
        btc_ts, btc_close_vals, btc_ma_vals = _build_btc_trend(btc_df, t_start, t_end)

        # Normalize BTC to start at INITIAL_CAPITAL for visual comparison
        if len(btc_close_vals) > 0 and btc_close_vals[0] != 0:
            btc_norm = (btc_close_vals / btc_close_vals[0] - 1) * 100

        fig, ax = plt.subplots(figsize=(14, 5))

        ax.plot(ts_all,  eq_all,  color="steelblue",  linewidth=1.2, label="Equity (no filter)")
        ax.plot(ts_filt, eq_filt, color="seagreen",   linewidth=1.2, label=f"Equity ({keep_trend} only)")
        if len(btc_ts) > 0:
            ax.plot(btc_ts, btc_norm, color="darkorange", linewidth=0.8, linestyle="--", label=f"BTC (normalized)")

        # Shade uptrend/downtrend regions on BTC
        if len(btc_ts) > 1:
            for i in range(len(btc_ts) - 1):
                if pd.isna(btc_ma_vals[i]):
                    continue
                is_up = btc_close_vals[i] > btc_ma_vals[i] * LONG_TH
                color = "green" if is_up else "red"
                ax.axvspan(btc_ts[i], btc_ts[i+1], alpha=0.04, color=color)

        ax.set_title(f"{strategy_name}  [{strategy_type}]  — MA{MA_PERIOD} filter  |  taken={len(taken)}  discarded={len(discarded)}")
        ax.set_ylabel("Net Gain (%)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        fig.autofmt_xdate()
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="upper left")
        plt.tight_layout()
        plt.show()
    
    # ==========================================================================
    # PRINT COMPARISON TABLE
    # ==========================================================================
    print("\n" + "=" * 80)
    print("STRATEGY-BY-STRATEGY COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Strategy':<30} {'Type':<8} {'ΔProfit%':>12} {'ΔDD%':>12}")
    print("-" * 80)
    
    for r in results:
        w = r['without_filter']
        f = r['with_filter']
        
        # Calculate % change in profit
        if w['total_profit'] != 0:
            profit_change_pct = ((f['total_profit'] - w['total_profit']) / abs(w['total_profit'])) * 100
        else:
            profit_change_pct = 0.0
        
        # Calculate % change in DD
        if w['max_dd_pct'] != 0:
            dd_change_pct = ((f['max_dd_pct'] - w['max_dd_pct']) / abs(w['max_dd_pct'])) * 100
        else:
            dd_change_pct = 0.0
        
        # Format values with fixed width
        profit_str = f"{profit_change_pct:+7.1f}%".replace('.', ',')
        dd_str = f"{dd_change_pct:+7.1f}%".replace('.', ',')
        
        # Apply colors
        if profit_change_pct > 5:
            profit_final = f"\033[92m{profit_str}\033[0m"
        elif profit_change_pct < -5:
            profit_final = f"\033[91m{profit_str}\033[0m"
        else:
            profit_final = profit_str
        
        if dd_change_pct > 5:
            dd_final = f"\033[92m{dd_str}\033[0m"
        elif dd_change_pct < -5:
            dd_final = f"\033[91m{dd_str}\033[0m"
        else:
            dd_final = dd_str
        
        # Manual padding to account for ANSI codes
        profit_padded = ' ' * (12 - len(profit_str)) + profit_final
        dd_padded = ' ' * (12 - len(dd_str)) + dd_final
        
        print(f"{r['strategy']:<30} {r['type']:<8} {profit_padded} {dd_padded}")
    
    print("-" * 80)
    
    # ==========================================================================
    # GLOBAL SUMMARY TABLE
    # ==========================================================================
    print("\n" + "=" * 100)
    print("GLOBAL PORTFOLIO SUMMARY")
    print("=" * 100)
    
    print(f"\n{'Metric':<25} {'WITHOUT FILTER':>20} {'WITH FILTER':>20} {'CHANGE':>20}")
    print("-" * 100)
    
    # Trades
    trades_change = global_with['num_trades'] - global_without['num_trades']
    trades_change_pct = (trades_change / global_without['num_trades'] * 100) if global_without['num_trades'] > 0 else 0
    trades_without_str = f"{global_without['num_trades']:,}".replace(',', '.')
    trades_with_str = f"{global_with['num_trades']:,}".replace(',', '.')
    trades_change_str = f"{trades_change_pct:+.1f}".replace('.', ',')
    print(f"{'Trades':<25} {trades_without_str:>20} {trades_with_str:>20} {trades_change_str:>19}%")
    
    # Profit
    profit_change = global_with['total_profit'] - global_without['total_profit']
    profit_change_pct = (profit_change / abs(global_without['total_profit']) * 100) if global_without['total_profit'] != 0 else 0
    profit_without_str = f"{global_without['total_profit']:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    profit_with_str = f"{global_with['total_profit']:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
    change_str = f"{profit_change_pct:+.1f}".replace('.', ',')
    print(f"{'Total Profit':<25} {profit_without_str:>20} {profit_with_str:>20} {change_str:>19}%")
    
    # Net Gain
    gain_change = global_with['net_gain_pct'] - global_without['net_gain_pct']
    gain_without_str = f"{global_without['net_gain_pct']:.2f}".replace('.', ',')
    gain_with_str = f"{global_with['net_gain_pct']:.2f}".replace('.', ',')
    gain_change_str = f"{gain_change:+.2f}".replace('.', ',')
    print(f"{'Net Gain %':<25} {gain_without_str:>19}% {gain_with_str:>19}% {gain_change_str:>19}%")
    
    # Max DD
    dd_change = global_with['max_dd_pct'] - global_without['max_dd_pct']
    dd_without_str = f"{global_without['max_dd_pct']:.2f}".replace('.', ',')
    dd_with_str = f"{global_with['max_dd_pct']:.2f}".replace('.', ',')
    dd_change_str = f"{dd_change:+.2f}".replace('.', ',')
    print(f"{'Max Drawdown %':<25} {dd_without_str:>19}% {dd_with_str:>19}% {dd_change_str:>19}%")
    
    print("-" * 100)
    
    # Improvement stats
    improvements = sum(1 for r in results if r['with_filter']['net_gain_pct'] > r['without_filter']['net_gain_pct'])
    print(f"\nStrategies improved: {improvements}/{len(results)} ({improvements/len(results)*100:.1f}%)")
    
    # ==========================================================================
    # RECOMMENDATION
    # ==========================================================================
    print("\n" + "=" * 100)
    print("RECOMMENDATION")
    print("=" * 100)
    
    delta_global_gain = global_with['net_gain_pct'] - global_without['net_gain_pct']
    delta_global_dd = global_with['max_dd_pct'] - global_without['max_dd_pct']
    
    if delta_global_gain > 2.0 and (delta_global_dd > -1.0):
        print("\n✅ RECOMMEND USING TREND FILTER")
        print(f"   • Net Gain improves by {delta_global_gain:.2f}%")
        print(f"   • Max DD similar or better")
        print(f"   • {improvements} out of {len(results)} strategies improve")
    elif delta_global_gain < -2.0:
        print("\n❌ DO NOT USE TREND FILTER")
        print(f"   • Net Gain decreases by {abs(delta_global_gain):.2f}%")
        print(f"   • System performs better without filtering")
    else:
        print("\n⚠️  MARGINAL IMPACT")
        print(f"   • Net Gain change: {delta_global_gain:+.2f}%")
        print(f"   • Consider other factors (complexity, robustness, etc.)")
    
    print("=" * 100)


if __name__ == "__main__":
    main()
    
