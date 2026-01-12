"""
market_regime/run_analysis.py

Main script to run regime analysis for strategies.

MODES:
- 'single': Analyzes a specific strategy
- 'batch': Analyzes ALL strategies of a generator

OHLC FOLDER:
- If OHLC_FOLDER_IS and OHLC_FOLDER_OOS are set, auto-detects from strategy name
- If only OHLC_FOLDER is set, uses that for all strategies

Usage:
    %runfile run_analysis.py
"""

import os
import sys
from pathlib import Path
from typing import List, Tuple, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_regime.trade_analyzer import TradeAnalyzer
from market_regime.strategy_profiler import StrategyProfiler


# =============================================================================
# CONFIGURATION - EDIT HERE
# =============================================================================
MODE = 'batch'                  # 'single' or 'batch'
GENERATOR = 'parity'            # For batch: processes all {GENERATOR}_*
STRATEGY = 'parity_long_4H_OOS'  # For single: specific strategy

# Base directory (project root)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Folders
TRADES_FOLDER = os.path.join(BASE_DIR, 'brief_trades')
OUTPUT_FOLDER = os.path.join(BASE_DIR, 'market_regime', 'output')

# OHLC folders - set both for auto-detection, or just OHLC_FOLDER for manual
OHLC_FOLDER_IS = os.path.join(BASE_DIR, 'data', 'crypto_2022_IS')
OHLC_FOLDER_OOS = os.path.join(BASE_DIR, 'data', 'crypto_OOS_2')
OHLC_FOLDER = None  # If set, overrides auto-detection and uses this for all

# =============================================================================


def get_ohlc_folder_for_strategy(strategy_name: str) -> str:
    """
    Determines the OHLC folder based on strategy name suffix.
    
    Args:
        strategy_name: Strategy name (e.g., 'parity_long_4H_IS' or 'parity_long_4H_OOS')
    
    Returns:
        Path to OHLC folder
    """
    # If manual override is set, use it
    if OHLC_FOLDER is not None:
        return OHLC_FOLDER
    
    # Auto-detect from strategy name
    if strategy_name.endswith('_IS'):
        return OHLC_FOLDER_IS
    elif strategy_name.endswith('_OOS'):
        return OHLC_FOLDER_OOS
    else:
        # Default to OOS if no suffix
        print(f"⚠️  No IS/OOS suffix in '{strategy_name}', defaulting to OOS folder")
        return OHLC_FOLDER_OOS


def full_analysis(
    strategy_name: str,
    trades_folder: str,
    ohlc_folder: str,
    output_folder: str,
    save_results: bool = True,
    verbose: bool = True
) -> Tuple:
    """
    Runs complete analysis for a strategy.
    
    Returns:
        tuple: (profile, df_enriched)
    """
    # Paths
    trades_path = os.path.join(trades_folder, f'all_trades_{strategy_name}.xlsx')
    
    # Infer timeframe from name
    # Example: parity_long_4H_OOS → timeframe = 4H
    parts = strategy_name.split('_')
    
    # If last part is IS/OOS, timeframe is second to last
    if parts[-1].upper() in ['IS', 'OOS']:
        timeframe = parts[-2] if len(parts) >= 2 else '4H'
    else:
        timeframe = parts[-1] if parts else '4H'
    
    if verbose:
        print("=" * 70)
        print(f"🔍 REGIME ANALYSIS: {strategy_name}")
        print("=" * 70)
        print(f"\n📂 Trades: {trades_path}")
        print(f"📂 OHLC: {ohlc_folder}")
        print(f"⏱️  Timeframe: {timeframe}")
    
    # Step 1: Analyze trades
    if verbose:
        print("\n" + "-" * 70)
        print("STEP 1: Calculating regime metrics per trade...")
        print("-" * 70)
    
    analyzer = TradeAnalyzer(
        trades_path=trades_path,
        ohlc_folder=ohlc_folder,
        timeframe=timeframe
    )
    
    df_enriched = analyzer.analyze(verbose=verbose)
    
    # Step 2: Generate profile
    if verbose:
        print("\n" + "-" * 70)
        print("STEP 2: Generating strategy profile...")
        print("-" * 70)
    
    profiler = StrategyProfiler(df_enriched)
    profile = profiler.generate_profile()
    
    if verbose:
        print()
        profiler.print_report(profile)
    
    # Step 3: Save results
    if save_results:
        Path(output_folder).mkdir(parents=True, exist_ok=True)
        
        # Save enriched trades
        enriched_path = os.path.join(output_folder, f'trades_enriched_{strategy_name}.xlsx')
        df_enriched.to_excel(enriched_path, index=False)
        
        # Save profile
        profile_path = os.path.join(output_folder, f'profile_{strategy_name}.xlsx')
        df_profile = profiler.to_dataframe(profile)
        df_profile.to_excel(profile_path, index=False)
        
        if verbose:
            print(f"\n💾 Results saved to: {output_folder}/")
            print(f"   - trades_enriched_{strategy_name}.xlsx")
            print(f"   - profile_{strategy_name}.xlsx")
    
    return profile, df_enriched


def find_strategies(generator: str, trades_folder: str, data_type: Optional[str] = None) -> List[str]:
    """
    Finds all strategies for a generator in the trades folder.
    
    Args:
        generator: Generator name (e.g., 'parity')
        trades_folder: Folder to search
        data_type: Optional filter - 'IS', 'OOS', or None for all
    
    Returns:
        List of strategy names
    """
    path = Path(trades_folder)
    
    if not path.exists():
        print(f"❌ Folder not found: {trades_folder}")
        return []
    
    # Search for all_trades_{generator}_*.xlsx
    pattern = f'all_trades_{generator}_*.xlsx'
    files = list(path.glob(pattern))
    
    strategies = []
    for f in files:
        # all_trades_parity_long_4H.xlsx → parity_long_4H
        name = f.stem.replace('all_trades_', '')
        
        # Filter by data_type if specified
        if data_type is not None:
            if data_type == 'IS' and not name.endswith('_IS'):
                continue
            elif data_type == 'OOS' and not name.endswith('_OOS'):
                continue
        
        strategies.append(name)
    
    return sorted(strategies)


def run_single_mode():
    """Runs analysis for a single strategy."""
    ohlc_folder = get_ohlc_folder_for_strategy(STRATEGY)
    
    print(f"📁 Base dir: {BASE_DIR}")
    print(f"📁 Trades folder: {TRADES_FOLDER}")
    print(f"📁 OHLC folder: {ohlc_folder}")
    print()
    
    profile, df = full_analysis(
        strategy_name=STRATEGY,
        trades_folder=TRADES_FOLDER,
        ohlc_folder=ohlc_folder,
        output_folder=OUTPUT_FOLDER
    )
    
    return {STRATEGY: (profile, df)}


def run_batch_mode():
    """Runs analysis for all strategies of a generator."""
    print("=" * 70)
    print(f"📊 BATCH ANALYSIS: {GENERATOR.upper()}")
    print("=" * 70)
    print(f"\n📁 Base dir: {BASE_DIR}")
    print(f"📁 Trades folder: {TRADES_FOLDER}")
    print(f"📁 OHLC IS: {OHLC_FOLDER_IS}")
    print(f"📁 OHLC OOS: {OHLC_FOLDER_OOS}")
    if OHLC_FOLDER:
        print(f"📁 OHLC Override: {OHLC_FOLDER}")
    
    # Find strategies
    strategies = find_strategies(GENERATOR, TRADES_FOLDER)
    
    if not strategies:
        print(f"\n❌ No strategies found for '{GENERATOR}'")
        print(f"   Searched: all_trades_{GENERATOR}_*.xlsx")
        print(f"   In: {TRADES_FOLDER}")
        return {}
    
    print(f"\n📋 Strategies found: {len(strategies)}")
    for s in strategies:
        suffix = "IS" if s.endswith('_IS') else ("OOS" if s.endswith('_OOS') else "?")
        print(f"   • {s} [{suffix}]")
    
    # Process each one
    results = {}
    errors = []
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n{'#' * 70}")
        print(f"# [{i}/{len(strategies)}] Processing: {strategy}")
        print(f"{'#' * 70}")
        
        # Get appropriate OHLC folder
        ohlc_folder = get_ohlc_folder_for_strategy(strategy)
        
        try:
            profile, df = full_analysis(
                strategy_name=strategy,
                trades_folder=TRADES_FOLDER,
                ohlc_folder=ohlc_folder,
                output_folder=OUTPUT_FOLDER
            )
            results[strategy] = (profile, df)
            
        except Exception as e:
            print(f"\n❌ Error processing {strategy}: {e}")
            errors.append((strategy, str(e)))
    
    # Final summary
    print("\n" + "=" * 70)
    print("📊 BATCH SUMMARY")
    print("=" * 70)
    
    print(f"\n✅ Processed successfully: {len(results)}/{len(strategies)}")
    
    if results:
        print(f"\n{'STRATEGY':<35} {'FAMILY':<15} {'TRADES':>8} {'PROFIT':>12} {'WIN%':>8}")
        print("-" * 85)
        
        for name, (profile, _) in results.items():
            print(f"{name:<35} {profile.family:<15} {profile.total_trades:>8} {profile.total_profit:>12.2f} {profile.win_rate:>7.1f}%")
    
    if errors:
        print(f"\n❌ Errors ({len(errors)}):")
        for strategy, error in errors:
            print(f"   • {strategy}: {error}")
    
    print("\n" + "=" * 70)
    print(f"💾 Results saved to: {OUTPUT_FOLDER}/")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    if MODE == 'single':
        results = run_single_mode()
    elif MODE == 'batch':
        results = run_batch_mode()
    else:
        print(f"❌ Unknown mode: {MODE}")
        print("   Use 'single' or 'batch'")