#!/usr/bin/env python3
"""
Compare BTC 1D Filter: Last Closed Candle vs Trade-Time Price

APPROACH A: Use BTC close from last closed 1D candle
APPROACH B: Use BTC price at exact trade time (from 1H/4H data)

Both compare against MA5 from closed 1D candles.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob


def load_btc_data():
    """Load BTC data in multiple timeframes"""
    base_path = Path('/home/javi/projects/quant/quant_g/bitget/development/defense_mode')
    
    # Load BTC 1D
    btc_1d = pd.read_parquet(base_path / 'BTCUSDT_1Dutc.parquet')
    btc_1d.columns = btc_1d.columns.str.lower()
    btc_1d['ts'] = pd.to_datetime(btc_1d['timestamp'] if 'timestamp' in btc_1d.columns else btc_1d.index)
    btc_1d = btc_1d.sort_values('ts').reset_index(drop=True)
    btc_1d['ma5'] = btc_1d['close'].rolling(window=5).mean()
    
    # Load BTC 4H
    btc_4h = pd.read_parquet(base_path / 'BTCUSDT_4H.parquet')
    btc_4h.columns = btc_4h.columns.str.lower()
    btc_4h['ts'] = pd.to_datetime(btc_4h['timestamp'] if 'timestamp' in btc_4h.columns else btc_4h.index)
    btc_4h = btc_4h.sort_values('ts').reset_index(drop=True)
    
    # Load BTC 1H (optional, fallback to 4H if not available)
    try:
        btc_1h = pd.read_parquet(base_path / 'BTCUSDT_1H.parquet')
        btc_1h.columns = btc_1h.columns.str.lower()
        btc_1h['ts'] = pd.to_datetime(btc_1h['timestamp'] if 'timestamp' in btc_1h.columns else btc_1h.index)
        btc_1h = btc_1h.sort_values('ts').reset_index(drop=True)
    except:
        print("⚠️  BTC 1H not found, using 4H as fallback")
        btc_1h = btc_4h.copy()
    
    return btc_1d, btc_4h, btc_1h


def get_btc_regime_approach_a(btc_1d, trade_time):
    """
    APPROACH A: Use close from last closed 1D candle
    """
    closed_candles = btc_1d[btc_1d['ts'] < trade_time]
    
    if len(closed_candles) < 5:
        return None
    
    last_candle = closed_candles.iloc[-1]
    
    if pd.isna(last_candle['ma5']):
        return None
    
    return {
        'btc_price': last_candle['close'],
        'ma5': last_candle['ma5']
    }


def get_btc_regime_approach_b(btc_1d, btc_intraday, trade_time):
    """
    APPROACH B: Use BTC price at trade time from intraday data
    """
    # Get MA5 from closed 1D candles (same as Approach A)
    closed_candles_1d = btc_1d[btc_1d['ts'] < trade_time]
    
    if len(closed_candles_1d) < 5:
        return None
    
    ma5 = closed_candles_1d['close'].tail(5).mean()
    
    if pd.isna(ma5):
        return None
    
    # Get BTC price at trade time from intraday data (1H or 4H)
    closed_intraday = btc_intraday[btc_intraday['ts'] <= trade_time]
    
    if len(closed_intraday) == 0:
        return None
    
    btc_price_at_trade = closed_intraday.iloc[-1]['close']
    
    return {
        'btc_price': btc_price_at_trade,
        'ma5': ma5
    }


def should_allow_trade(direction, btc_regime, threshold_long=1.02, threshold_short=1.00):
    """Check if trade should be allowed"""
    if btc_regime is None:
        return True
    
    btc_price = btc_regime['btc_price']
    ma5 = btc_regime['ma5']
    
    if direction == 'LONG':
        return btc_price > ma5 * threshold_long
    elif direction == 'SHORT':
        return btc_price < ma5 * threshold_short
    
    return True


def load_all_lab_trades():
    """Load all lab trades"""
    lab_folder = Path('/home/javi/projects/quant/quant_g/bitget/development/brief_trades')
    files = glob(str(lab_folder / 'all_trades_*.xlsx'))
    
    all_trades = []
    for filepath in files:
        df = pd.read_excel(filepath)
        df['sell_time'] = pd.to_datetime(df['sell_time'])
        df['buy_time'] = pd.to_datetime(df['buy_time'])
        all_trades.append(df)
    
    combined = pd.concat(all_trades, ignore_index=True)
    return combined.sort_values('buy_time').reset_index(drop=True)


def evaluate_approach(df_trades, btc_1d, btc_intraday, approach='A'):
    """Evaluate trades with given approach"""
    results = []
    
    for idx, trade in df_trades.iterrows():
        direction = trade['position_type']
        profit = trade['profit']
        trade_time = trade['buy_time']
        
        # Get BTC regime based on approach
        if approach == 'A':
            btc_regime = get_btc_regime_approach_a(btc_1d, trade_time)
        else:  # approach == 'B'
            btc_regime = get_btc_regime_approach_b(btc_1d, btc_intraday, trade_time)
        
        # Check if trade allowed
        allowed = should_allow_trade(direction, btc_regime)
        
        if allowed:
            results.append({
                'profit': profit,
                'is_winner': profit > 0
            })
    
    # Calculate metrics
    if len(results) == 0:
        return {
            'num_trades': 0,
            'total_profit': 0,
            'win_rate': 0
        }
    
    num_trades = len(results)
    total_profit = sum(r['profit'] for r in results)
    winners = sum(1 for r in results if r['is_winner'])
    win_rate = (winners / num_trades * 100) if num_trades > 0 else 0
    
    return {
        'num_trades': num_trades,
        'total_profit': total_profit,
        'win_rate': win_rate
    }


def main():
    print("="*100)
    print("BTC 1D FILTER COMPARISON: Last Closed Candle (A) vs Trade-Time Price (B)")
    print("="*100)
    
    # Load BTC data
    print("\n📂 Loading BTC data...")
    btc_1d, btc_4h, btc_1h = load_btc_data()
    print(f"✅ BTC 1D: {len(btc_1d)} candles")
    print(f"✅ BTC 4H: {len(btc_4h)} candles")
    print(f"✅ BTC 1H: {len(btc_1h)} candles")
    
    # Load LAB trades
    print("\n📂 Loading LAB trades...")
    df_lab = load_all_lab_trades()
    print(f"✅ Loaded {len(df_lab)} LAB trades")
    
    # Evaluate Approach A (last closed 1D candle)
    print("\n🔍 Evaluating APPROACH A: Last Closed 1D Candle...")
    results_a = evaluate_approach(df_lab, btc_1d, btc_4h, approach='A')
    
    # Evaluate Approach B (trade-time price from 4H)
    print("🔍 Evaluating APPROACH B: Trade-Time Price (4H)...")
    results_b_4h = evaluate_approach(df_lab, btc_1d, btc_4h, approach='B')
    
    # Evaluate Approach B (trade-time price from 1H)
    print("🔍 Evaluating APPROACH B: Trade-Time Price (1H)...")
    results_b_1h = evaluate_approach(df_lab, btc_1d, btc_1h, approach='B')
    
    # Display results
    print("\n" + "="*100)
    print("RESULTS COMPARISON")
    print("="*100)
    
    print(f"\n{'Approach':<40} {'Trades':<15} {'Profit':<15} {'Win Rate':<15}")
    print("-"*100)
    
    print(f"{'A: Last Closed 1D Candle':<40} "
          f"{results_a['num_trades']:<15} "
          f"${results_a['total_profit']:<14.2f} "
          f"{results_a['win_rate']:<14.1f}%")
    
    print(f"{'B: Trade-Time Price (4H)':<40} "
          f"{results_b_4h['num_trades']:<15} "
          f"${results_b_4h['total_profit']:<14.2f} "
          f"{results_b_4h['win_rate']:<14.1f}%")
    
    print(f"{'B: Trade-Time Price (1H)':<40} "
          f"{results_b_1h['num_trades']:<15} "
          f"${results_b_1h['total_profit']:<14.2f} "
          f"{results_b_1h['win_rate']:<14.1f}%")
    
    # Calculate differences
    print("\n" + "="*100)
    print("DIFFERENCES (B vs A)")
    print("="*100)
    
    # 4H vs A
    diff_trades_4h = results_b_4h['num_trades'] - results_a['num_trades']
    diff_profit_4h = results_b_4h['total_profit'] - results_a['total_profit']
    diff_wr_4h = results_b_4h['win_rate'] - results_a['win_rate']
    
    print(f"\n4H Trade-Time vs Last Closed:")
    print(f"  Trades: {diff_trades_4h:+d}")
    print(f"  Profit: ${diff_profit_4h:+.2f}")
    print(f"  Win Rate: {diff_wr_4h:+.1f}pp")
    
    # 1H vs A
    diff_trades_1h = results_b_1h['num_trades'] - results_a['num_trades']
    diff_profit_1h = results_b_1h['total_profit'] - results_a['total_profit']
    diff_wr_1h = results_b_1h['win_rate'] - results_a['win_rate']
    
    print(f"\n1H Trade-Time vs Last Closed:")
    print(f"  Trades: {diff_trades_1h:+d}")
    print(f"  Profit: ${diff_profit_1h:+.2f}")
    print(f"  Win Rate: {diff_wr_1h:+.1f}pp")
    
    # Recommendation
    print("\n" + "="*100)
    print("RECOMMENDATION")
    print("="*100)
    
    if results_a['total_profit'] >= results_b_4h['total_profit'] and results_a['total_profit'] >= results_b_1h['total_profit']:
        print("\n✅ APPROACH A (Last Closed 1D Candle) is BEST")
        print("   → Use close from last closed 1D candle (current implementation)")
    elif results_b_1h['total_profit'] > results_a['total_profit']:
        print("\n⚠️  APPROACH B with 1H data shows better profit")
        print("   → Consider using trade-time price from 1H data")
        print(f"   → Improvement: ${diff_profit_1h:+.2f}")
    else:
        print("\n⚠️  APPROACH B with 4H data shows better profit")
        print("   → Consider using trade-time price from 4H data")
        print(f"   → Improvement: ${diff_profit_4h:+.2f}")


if __name__ == "__main__":
    main()