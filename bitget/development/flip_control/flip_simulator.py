"""
flip_control/flip_simulator.py

Main module for flip control simulation.
Detects flips, applies partial closing, and compares performance.

Usage:
    python flip_simulator.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flip_control.config import (
    ENRICHED_TRADES_FOLDER, ENRICHED_TRADES_PATTERN,
    BTC_OHLC_FOLDER, OHLC_FOLDER_15M,
    BTC_SYMBOL, MA_PERIOD,
    FLIP_CONFIRMATION_BARS, FLIP_DISTANCE_PCT, PARTIAL_CLOSE_PCT,
    INITIAL_CAPITAL, DATE_RANGE_FILTER
)
from flip_control.flip_detector import load_btc_ohlc, detect_flips, get_regime_at_time


# Cache for OHLC data
_ohlc_cache = {}


def extract_timeframe(filename: str) -> str:
    """
    Extracts timeframe from trades filename.
    
    Examples:
        trades_enriched_parity_long_4H.xlsx → 4H
        trades_enriched_reversal_short_1H_OOS.xlsx → 1H
    """
    name = Path(filename).stem.replace('trades_enriched_', '')
    parts = name.split('_')
    
    # Remove OOS/IS if present
    if parts[-1].upper() in ['IS', 'OOS']:
        parts = parts[:-1]
    
    # Timeframe should be last part
    if parts:
        timeframe = parts[-1]
        if any(c.isdigit() for c in timeframe.upper()) and 'H' in timeframe.upper():
            return timeframe
    
    return '4H'  # Default fallback


def load_symbol_ohlc_15m(symbol: str) -> pd.DataFrame:
    """
    Loads 15m OHLC data for a symbol (with caching).
    
    Args:
        symbol: Symbol name (e.g., 'ETHUSDT')
    
    Returns:
        DataFrame with columns: ts, open, high, low, close
    """
    if symbol in _ohlc_cache:
        return _ohlc_cache[symbol]
    
    filepath = Path(OHLC_FOLDER_15M) / f"{symbol}_15m.parquet"
    
    if not filepath.exists():
        print(f"        ⚠️  15m data not found for {symbol}: {filepath}")
        return None
    
    df = pd.read_parquet(filepath)
    df.columns = df.columns.str.lower()
    
    # Ensure timestamp column
    ts_columns = ['timestamp', 'ts', 'date', 'time']
    ts_col = None
    for col in ts_columns:
        if col in df.columns:
            ts_col = col
            break
    
    if ts_col:
        df['ts'] = pd.to_datetime(df[ts_col])
    else:
        df['ts'] = pd.to_datetime(df.index)
        df = df.reset_index(drop=True)
    
    df = df.sort_values('ts').reset_index(drop=True)
    
    _ohlc_cache[symbol] = df
    return df


def get_price_at_flip(symbol: str, flip_timestamp: pd.Timestamp) -> float:
    """
    Gets approximate price of symbol at flip timestamp using 15m data.
    
    Args:
        symbol: Symbol name
        flip_timestamp: Timestamp of flip
    
    Returns:
        Price (close of nearest 15m bar), or None if not available
    """
    ohlc_15m = load_symbol_ohlc_15m(symbol)
    
    if ohlc_15m is None:
        return None
    
    # Find nearest 15m bar (before or at flip_timestamp)
    available_bars = ohlc_15m[ohlc_15m['ts'] <= flip_timestamp]
    
    if len(available_bars) == 0:
        return None
    
    # Use close of last available bar
    return float(available_bars.iloc[-1]['close'])


def calculate_max_dd_pct(equity_curve: pd.Series) -> float:
    """
    Calculates Maximum Drawdown % correctly.
    DD% = max((peak - valley) / peak * 100)
    """
    if len(equity_curve) == 0:
        return 0.0
    
    cummax = equity_curve.cummax()
    
    drawdown_pct = np.where(
        cummax > 0,
        ((cummax - equity_curve) / cummax) * 100,
        0.0
    )
    
    return float(np.max(drawdown_pct))


def extract_direction(strategy_name: str) -> str:
    """Extracts direction (long/short) from strategy name."""
    name_lower = strategy_name.lower()
    if 'long' in name_lower:
        return 'LONG'
    elif 'short' in name_lower:
        return 'SHORT'
    else:
        return 'UNKNOWN'


def is_trade_against_regime(trade_direction: str, flip_type: str) -> bool:
    """
    Checks if trade is against the new regime after flip.
    
    Args:
        trade_direction: 'LONG' or 'SHORT'
        flip_type: 'UP_TO_DOWN' or 'DOWN_TO_UP'
    
    Returns:
        True if trade should be closed (against regime)
    """
    if flip_type == 'UP_TO_DOWN':
        # Market flipped to DOWN → close LONG positions
        return trade_direction == 'LONG'
    elif flip_type == 'DOWN_TO_UP':
        # Market flipped to UP → close SHORT positions
        return trade_direction == 'SHORT'
    else:
        return False


def apply_partial_close(
    trades_df: pd.DataFrame,
    flips: List[Dict],
    strategy_direction: str,
    partial_close_pct: float
) -> pd.DataFrame:
    """
    Applies partial closing to trades affected by flips.
    
    Args:
        trades_df: DataFrame with trades (must have: buy_time, sell_time, buy_price, sell_price, qty, profit, symbol)
        flips: List of flip events from flip_detector
        strategy_direction: Direction of strategy ('LONG' or 'SHORT')
        partial_close_pct: Percentage to close (0.0-1.0)
    
    Returns:
        DataFrame with added columns: flip_affected, flip_timestamp, profit_adjusted
    """
    trades_df = trades_df.copy()
    
    # Initialize columns
    trades_df['flip_affected'] = False
    trades_df['flip_timestamp'] = pd.NaT
    trades_df['price_at_flip'] = np.nan
    trades_df['profit_partial'] = 0.0
    trades_df['profit_remaining'] = 0.0
    trades_df['profit_adjusted'] = trades_df['profit']
    
    if partial_close_pct == 0.0:
        # Test mode: no changes
        return trades_df
    
    # Counters
    total_modified = 0
    better_with_flip = 0
    worse_with_flip = 0
    
    # Process each flip
    for flip in flips:
        flip_ts = flip['timestamp']
        flip_type = flip['flip_type']
        
        # Find trades active during this flip
        active_trades = trades_df[
            (trades_df['buy_time'] < flip_ts) & 
            (trades_df['sell_time'] > flip_ts)
        ]
        
        for idx in active_trades.index:
            trade = trades_df.loc[idx]
            
            # Check if trade is against the new regime
            if not is_trade_against_regime(strategy_direction, flip_type):
                continue
            
            # Get price at flip (15m approximation)
            symbol = trade['symbol']
            price_at_flip = get_price_at_flip(symbol, flip_ts)
            
            if price_at_flip is None:
                # Cannot calculate, skip this trade
                continue
            
            # Calculate partial profit (closed at flip)
            buy_price = trade['buy_price']
            sell_price = trade['sell_price']
            qty = trade['qty']
            actual_trade_profit = trade['profit']
            
            # Profit from closing partial_pct at flip (approximation without exact fees)
            profit_partial = partial_close_pct * (price_at_flip - buy_price) * qty
            
            # Profit from remaining position: use proportional share of ACTUAL profit
            # (this includes fees, slippage proportionally distributed)
            profit_remaining = (1 - partial_close_pct) * actual_trade_profit
            
            # Adjusted profit
            profit_adjusted = profit_partial + profit_remaining
            
            # Track if flip improved the trade
            if profit_adjusted > actual_trade_profit:
                better_with_flip += 1
            elif profit_adjusted < actual_trade_profit:
                worse_with_flip += 1
            
            # Update trade
            trades_df.at[idx, 'flip_affected'] = True
            trades_df.at[idx, 'flip_timestamp'] = flip_ts
            trades_df.at[idx, 'price_at_flip'] = price_at_flip
            trades_df.at[idx, 'profit_partial'] = profit_partial
            trades_df.at[idx, 'profit_remaining'] = profit_remaining
            trades_df.at[idx, 'profit_adjusted'] = profit_adjusted
            
            total_modified += 1
            
            # Only apply first flip per trade
            break
    
    # Return stats along with dataframe
    stats = {
        'total_modified': total_modified,
        'better_with_flip': better_with_flip,
        'worse_with_flip': worse_with_flip
    }
    
    return trades_df, stats


def calculate_metrics(trades_df: pd.DataFrame, initial_capital: float, profit_col: str = 'profit') -> Dict:
    """
    Calculates performance metrics from trades.
    
    Args:
        trades_df: DataFrame with trades
        initial_capital: Initial capital
        profit_col: Column name for profit ('profit' or 'profit_adjusted')
    
    Returns:
        Dict with metrics: total_profit, max_dd_pct, win_rate
    """
    if len(trades_df) == 0:
        return {
            'total_profit': 0.0,
            'max_dd_pct': 0.0,
            'win_rate': 0.0
        }
    
    # Sort by time
    df_sorted = trades_df.sort_values('buy_time').reset_index(drop=True)
    
    # Calculate equity curve
    df_sorted['equity'] = initial_capital + df_sorted[profit_col].cumsum()
    
    # Metrics
    total_profit = df_sorted[profit_col].sum()
    max_dd_pct = calculate_max_dd_pct(df_sorted['equity'])
    win_rate = (df_sorted[profit_col] > 0).mean() * 100
    
    return {
        'total_profit': total_profit,
        'max_dd_pct': max_dd_pct,
        'win_rate': win_rate
    }


def process_strategy(
    filepath: str,
    flips_by_timeframe: Dict[str, List[Dict]],
    partial_close_pct: float,
    initial_capital: float,
    date_range: tuple = None
) -> Dict:
    """
    Processes a single strategy file.
    
    Args:
        filepath: Path to enriched trades file
        flips_by_timeframe: Dict mapping timeframe to list of flips
        partial_close_pct: Percentage to close on flip
        initial_capital: Initial capital
        date_range: Optional date range filter
    
    Returns:
        Dict with strategy results
    """
    strategy_name = Path(filepath).stem.replace('trades_enriched_', '')
    timeframe = extract_timeframe(Path(filepath).name)
    
    # Load trades
    trades_df = pd.read_excel(filepath)
    trades_df.columns = trades_df.columns.str.lower().str.strip()
    trades_df['buy_time'] = pd.to_datetime(trades_df['buy_time'])
    trades_df['sell_time'] = pd.to_datetime(trades_df['sell_time'])
    
    # Apply date range filter
    if date_range is not None:
        start_date, end_date = date_range
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        trades_df = trades_df[
            (trades_df['buy_time'] >= start_date) & 
            (trades_df['buy_time'] <= end_date)
        ].copy()
    
    if len(trades_df) == 0:
        return None
    
    # Get flips for this timeframe
    flips = flips_by_timeframe.get(timeframe, [])
    
    # Extract strategy direction
    strategy_direction = extract_direction(strategy_name)
    
    # Apply partial closing
    trades_adjusted, flip_stats = apply_partial_close(
        trades_df, 
        flips, 
        strategy_direction, 
        partial_close_pct
    )
    
    # Calculate metrics
    metrics_original = calculate_metrics(trades_df, initial_capital, 'profit')
    metrics_adjusted = calculate_metrics(trades_adjusted, initial_capital, 'profit_adjusted')
    
    # Count affected trades
    num_trades = len(trades_df)
    num_affected = trades_adjusted['flip_affected'].sum()
    
    return {
        'strategy': strategy_name,
        'timeframe': timeframe,
        'num_trades': num_trades,
        'num_affected': num_affected,
        'metrics_original': metrics_original,
        'metrics_adjusted': metrics_adjusted,
        'flip_stats': flip_stats  # Add stats
    }


def print_strategy_summary(result: Dict):
    """Prints summary for a single strategy."""
    print(f"\n{'-'*80}")
    print(f"STRATEGY: {result['strategy']}")
    print(f"{'-'*80}")
    print(f"Trades affected: {result['num_affected']} / {result['num_trades']} ({result['num_affected']/result['num_trades']*100:.1f}%)")
    
    # Show flip stats if any trades were modified
    flip_stats = result.get('flip_stats', {})
    if flip_stats.get('total_modified', 0) > 0:
        total = flip_stats['total_modified']
        better = flip_stats['better_with_flip']
        worse = flip_stats['worse_with_flip']
        neutral = flip_stats.get('neutral', 0)  # Use get with default
        print(f"  Better with flip: {better} ({better/total*100:.1f}%)")
        print(f"  Worse with flip: {worse} ({worse/total*100:.1f}%)")
        if neutral > 0:
            print(f"  Neutral: {neutral}")
    
    orig = result['metrics_original']
    adj = result['metrics_adjusted']
    
    profit_delta = adj['total_profit'] - orig['total_profit']
    profit_delta_pct = (profit_delta / abs(orig['total_profit']) * 100) if orig['total_profit'] != 0 else 0
    dd_delta = adj['max_dd_pct'] - orig['max_dd_pct']
    wr_delta = adj['win_rate'] - orig['win_rate']
    
    profit_icon = "✅" if profit_delta > 0 else ("🟢" if profit_delta == 0 else "❌")
    dd_icon = "✅" if dd_delta < 0 else ("🟢" if dd_delta == 0 else "❌")
    wr_icon = "✅" if wr_delta > 0 else ("🟢" if wr_delta == 0 else "❌")
    
    print(f"\nORIGINAL:  Profit: ${orig['total_profit']:>8.2f}  DD: {orig['max_dd_pct']:>6.2f}%  WR: {orig['win_rate']:>5.1f}%")
    print(f"ADJUSTED:  Profit: ${adj['total_profit']:>8.2f}  DD: {adj['max_dd_pct']:>6.2f}%  WR: {adj['win_rate']:>5.1f}%")
    print(f"DELTA:     {profit_icon} {profit_delta:>+8.2f} ({profit_delta_pct:>+6.1f}%)  {dd_icon} {dd_delta:>+6.2f}%  {wr_icon} {wr_delta:>+5.1f}%")


def run_simulation():
    """Main simulation function."""
    
    print("=" * 80)
    print("FLIP CONTROL SIMULATION")
    print("=" * 80)
    
    print(f"\nParameters:")
    print(f"  PARTIAL_CLOSE_PCT: {PARTIAL_CLOSE_PCT}")
    print(f"  FLIP_CONFIRMATION_BARS: {FLIP_CONFIRMATION_BARS}")
    print(f"  FLIP_DISTANCE_PCT: {FLIP_DISTANCE_PCT}")
    
    if DATE_RANGE_FILTER:
        print(f"\n⚠️  DATE RANGE FILTER ACTIVE: {DATE_RANGE_FILTER[0]} → {DATE_RANGE_FILTER[1]}")
    
    # Find enriched trades files
    pattern = os.path.join(ENRICHED_TRADES_FOLDER, ENRICHED_TRADES_PATTERN)
    files = sorted(glob(pattern))
    
    if not files:
        print(f"\n❌ No enriched trades found in {ENRICHED_TRADES_FOLDER}")
        return
    
    print(f"\nFiles found: {len(files)}")
    
    # Detect flips for each timeframe
    print(f"\n{'='*80}")
    print("DETECTING FLIPS")
    print(f"{'='*80}")
    
    timeframes = set(extract_timeframe(Path(f).name) for f in files)
    flips_by_timeframe = {}
    
    for tf in sorted(timeframes):
        print(f"\nTimeframe: {tf}")
        btc_df = load_btc_ohlc(BTC_OHLC_FOLDER, BTC_SYMBOL, tf)
        print(f"  BTC data: {len(btc_df)} bars ({btc_df['ts'].min()} → {btc_df['ts'].max()})")
        
        flips = detect_flips(
            btc_df, 
            ma_period=MA_PERIOD,
            confirmation_bars=FLIP_CONFIRMATION_BARS,
            distance_pct=FLIP_DISTANCE_PCT
        )
        
        flips_by_timeframe[tf] = flips
        
        # Count flip types
        up_to_down = sum(1 for f in flips if f['flip_type'] == 'UP_TO_DOWN')
        down_to_up = sum(1 for f in flips if f['flip_type'] == 'DOWN_TO_UP')
        
        print(f"  Flips detected: {len(flips)} ({up_to_down} UP→DOWN, {down_to_up} DOWN→UP)")
    
    # Process each strategy
    print(f"\n{'='*80}")
    print("PROCESSING STRATEGIES")
    print(f"{'='*80}")
    
    results = []
    for f in files:
        result = process_strategy(
            f, 
            flips_by_timeframe, 
            PARTIAL_CLOSE_PCT, 
            INITIAL_CAPITAL,
            date_range=DATE_RANGE_FILTER
        )
        
        if result is not None:
            results.append(result)
            print_strategy_summary(result)
    
    # Portfolio summary
    print(f"\n{'='*80}")
    print("PORTFOLIO TOTAL")
    print(f"{'='*80}")
    
    total_trades = sum(r['num_trades'] for r in results)
    total_affected = sum(r['num_affected'] for r in results)
    
    total_profit_orig = sum(r['metrics_original']['total_profit'] for r in results)
    total_profit_adj = sum(r['metrics_adjusted']['total_profit'] for r in results)
    
    # Calculate portfolio-level DD and WR (need combined equity curve)
    all_trades_orig = []
    all_trades_adj = []
    
    for f in files:
        trades_df = pd.read_excel(f)
        trades_df.columns = trades_df.columns.str.lower().str.strip()
        trades_df['buy_time'] = pd.to_datetime(trades_df['buy_time'])
        
        if DATE_RANGE_FILTER:
            start_date, end_date = DATE_RANGE_FILTER
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            trades_df = trades_df[
                (trades_df['buy_time'] >= start_date) & 
                (trades_df['buy_time'] <= end_date)
            ]
        
        if len(trades_df) == 0:
            continue
        
        tf = extract_timeframe(Path(f).name)
        strategy_name = Path(f).stem.replace('trades_enriched_', '')
        strategy_direction = extract_direction(strategy_name)
        flips = flips_by_timeframe.get(tf, [])
        
        trades_adj, _ = apply_partial_close(trades_df, flips, strategy_direction, PARTIAL_CLOSE_PCT)
        
        all_trades_orig.append(trades_df[['buy_time', 'profit']])
        all_trades_adj.append(trades_adj[['buy_time', 'profit_adjusted']].rename(columns={'profit_adjusted': 'profit'}))
    
    combined_orig = pd.concat(all_trades_orig, ignore_index=True).sort_values('buy_time').reset_index(drop=True)
    combined_adj = pd.concat(all_trades_adj, ignore_index=True).sort_values('buy_time').reset_index(drop=True)
    
    portfolio_orig = calculate_metrics(combined_orig, INITIAL_CAPITAL, 'profit')
    portfolio_adj = calculate_metrics(combined_adj, INITIAL_CAPITAL, 'profit')
    
    print(f"\nTrades affected: {total_affected} / {total_trades} ({total_affected/total_trades*100:.1f}%)")
    
    profit_delta = total_profit_adj - total_profit_orig
    profit_delta_pct = (profit_delta / abs(total_profit_orig) * 100) if total_profit_orig != 0 else 0
    dd_delta = portfolio_adj['max_dd_pct'] - portfolio_orig['max_dd_pct']
    wr_delta = portfolio_adj['win_rate'] - portfolio_orig['win_rate']
    
    profit_icon = "✅" if profit_delta > 0 else ("🟢" if profit_delta == 0 else "❌")
    dd_icon = "✅" if dd_delta < 0 else ("🟢" if dd_delta == 0 else "❌")
    wr_icon = "✅" if wr_delta > 0 else ("🟢" if wr_delta == 0 else "❌")
    
    print(f"\nORIGINAL:  Profit: ${total_profit_orig:>8.2f}  DD: {portfolio_orig['max_dd_pct']:>6.2f}%  WR: {portfolio_orig['win_rate']:>5.1f}%")
    print(f"ADJUSTED:  Profit: ${total_profit_adj:>8.2f}  DD: {portfolio_adj['max_dd_pct']:>6.2f}%  WR: {portfolio_adj['win_rate']:>5.1f}%")
    print(f"DELTA:     {profit_icon} {profit_delta:>+8.2f} ({profit_delta_pct:>+6.1f}%)  {dd_icon} {dd_delta:>+6.2f}%  {wr_icon} {wr_delta:>+5.1f}%")
    
    # Test validation
    if PARTIAL_CLOSE_PCT == 0.0:
        if abs(profit_delta) < 0.01 and abs(dd_delta) < 0.01 and abs(wr_delta) < 0.01:
            print(f"\n✅ TEST PASSED: Parameters at 0 produce identical results")
        else:
            print(f"\n❌ TEST FAILED: Parameters at 0 should produce identical results")
    
    print(f"\n{'='*80}")


if __name__ == "__main__":
    run_simulation()