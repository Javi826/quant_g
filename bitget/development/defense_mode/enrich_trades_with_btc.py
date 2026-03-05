#!/usr/bin/env python3
"""
defense_mode/enrich_trades_with_btc.py

Enriches bot_trades Excel files with BTC price and metrics at EXACT trade open time.
Does NOT use lookahead (uses closed candle before trade).

Usage:
    Place this script in defense_mode/ folder with bot_trades_*.xlsx files
    python enrich_trades_with_btc.py
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from glob import glob

# Add parent to path to import market_regime
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_regime.regime_metrics import calc_all_metrics


def find_btc_parquet():
    """Find BTC 4H parquet file"""
    # Try common locations
    script_dir = Path(__file__).parent
    
    possible_paths = [
        script_dir / 'BTCUSDT_4H.parquet',  # Same folder
        script_dir.parent / 'data' / 'crypto_OOS_2025' / 'BTCUSDT_4H.parquet',
        script_dir.parent / 'data' / 'BTCUSDT_4H.parquet',
        Path.home() / 'projects' / 'quant' / 'quant_g' / 'bitget' / 'development' / 'data' / 'crypto_OOS_2025' / 'BTCUSDT_4H.parquet',
    ]
    
    for path in possible_paths:
        if path.exists():
            return str(path)
    
    raise FileNotFoundError(
        "Could not find BTCUSDT_4H.parquet. Please provide path manually."
    )


def load_btc_data(btc_path: str) -> pd.DataFrame:
    """Load and prepare BTC OHLC data"""
    print(f"📂 Loading BTC data from: {btc_path}")
    
    df = pd.read_parquet(btc_path)
    df.columns = df.columns.str.lower()
    
    # Find timestamp column
    ts_col = None
    for col in ['timestamp', 'ts', 'date', 'time']:
        if col in df.columns:
            ts_col = col
            break
    
    if ts_col:
        df['ts'] = pd.to_datetime(df[ts_col])
    else:
        df['ts'] = pd.to_datetime(df.index)
        df = df.reset_index(drop=True)
    
    df = df.sort_values('ts').reset_index(drop=True)
    
    print(f"✅ Loaded {len(df)} BTC 4H bars")
    print(f"   Date range: {df['ts'].min()} → {df['ts'].max()}")
    
    return df


def get_btc_at_time(btc_df: pd.DataFrame, trade_time: pd.Timestamp) -> dict:
    """
    Get BTC price and metrics at trade time.
    Uses LAST CLOSED candle before trade_time (no lookahead).
    """
    # Only use candles that closed BEFORE trade time
    closed_candles = btc_df[btc_df['ts'] < trade_time]
    
    if len(closed_candles) == 0:
        return None
    
    # Get last closed candle
    idx = closed_candles.index[-1]
    candle = btc_df.iloc[idx]
    
    # Basic price info
    result = {
        'btc_timestamp': candle['ts'],
        'btc_open': float(candle['open']),
        'btc_high': float(candle['high']),
        'btc_low': float(candle['low']),
        'btc_close': float(candle['close']),
    }
    
    # Calculate metrics if we have enough history
    lookback = 100
    start_idx = max(0, idx - lookback + 1)
    
    if idx - start_idx >= 20:
        subset = btc_df.iloc[start_idx:idx + 1]
        
        ohlc = {
            'open': subset['open'].values.astype(np.float64),
            'high': subset['high'].values.astype(np.float64),
            'low': subset['low'].values.astype(np.float64),
            'close': subset['close'].values.astype(np.float64)
        }
        
        try:
            metrics = calc_all_metrics(
                ohlc,
                hurst_window=100,
                er_window=14,
                atr_window=14,
                pe_window=50,
                pe_order=3
            )
            
            result.update({
                'btc_hurst': metrics['hurst'],
                'btc_efficiency_ratio': metrics['efficiency_ratio'],
                'btc_atr_pct': metrics['atr_pct'],
                'btc_permutation_entropy': metrics['permutation_entropy']
            })
        except:
            pass
    
    # Calculate MAs
    if idx >= 19:
        ma20 = btc_df.iloc[idx-19:idx+1]['close'].mean()
        result['btc_ma20'] = float(ma20)
        result['btc_price_vs_ma20'] = float(candle['close'] / ma20)
    
    if idx >= 49:
        ma50 = btc_df.iloc[idx-49:idx+1]['close'].mean()
        result['btc_ma50'] = float(ma50)
        result['btc_price_vs_ma50'] = float(candle['close'] / ma50)
    
    if idx >= 199:
        ma200 = btc_df.iloc[idx-199:idx+1]['close'].mean()
        result['btc_ma200'] = float(ma200)
        result['btc_price_vs_ma200'] = float(candle['close'] / ma200)
    
    return result


def enrich_trades_file(trades_file: str, btc_df: pd.DataFrame) -> pd.DataFrame:
    """Enrich single trades file with BTC data"""
    print(f"\n📊 Processing: {Path(trades_file).name}")
    
    # Load trades
    df = pd.read_excel(trades_file)
    
    # Find timestamp column
    ts_col = None
    for col in ['OPEN_AT', 'open_at', 'buy_time', 'entry_time']:
        if col in df.columns:
            ts_col = col
            break
    
    if not ts_col:
        raise ValueError(f"No timestamp column found in {trades_file}")
    
    df[ts_col] = pd.to_datetime(df[ts_col])
    print(f"   Trades: {len(df)}")
    print(f"   Date range: {df[ts_col].min()} → {df[ts_col].max()}")
    
    # Enrich each trade
    success = 0
    for idx, row in df.iterrows():
        trade_time = row[ts_col]
        btc_data = get_btc_at_time(btc_df, trade_time)
        
        if btc_data:
            for key, value in btc_data.items():
                df.at[idx, key] = value
            success += 1
    
    print(f"   ✅ Enriched: {success}/{len(df)} trades")
    
    # Drop rows with missing critical data
    before = len(df)
    critical_cols = ['btc_close', 'btc_ma50']
    df = df.dropna(subset=critical_cols).reset_index(drop=True)
    after = len(df)
    
    if before != after:
        print(f"   ⚠️  Dropped {before - after} trades with incomplete BTC data")
    
    return df


def main():
    """Main function"""
    print("="*80)
    print("TRADE ENRICHER - Add BTC price and metrics")
    print("="*80)
    
    try:
        # Find BTC data
        btc_path = find_btc_parquet()
        btc_df = load_btc_data(btc_path)
        
        # Find trades files
        script_dir = Path(__file__).parent
        trades_files = list(script_dir.glob('bot_trades_*.xlsx'))
        
        if not trades_files:
            print("\n❌ No bot_trades_*.xlsx files found in current directory")
            return
        
        print(f"\n📁 Found {len(trades_files)} trades files:")
        for f in trades_files:
            print(f"   • {f.name}")
        
        # Process each file
        print("\n" + "="*80)
        print("PROCESSING FILES")
        print("="*80)
        
        for trades_file in trades_files:
            enriched_df = enrich_trades_file(str(trades_file), btc_df)
            
            # Save enriched file
            output_name = f"enriched_{trades_file.name}"
            output_path = script_dir / output_name
            
            enriched_df.to_excel(output_path, index=False)
            print(f"   💾 Saved: {output_name}")
        
        print("\n" + "="*80)
        print("✅ ENRICHMENT COMPLETE")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()