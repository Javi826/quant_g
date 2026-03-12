"""
market_regime/regime1_feature_validator.py

Validates REGIME 1 classification using ML feature importance.
Trains Random Forest per-strategy to predict trade winners.
Shows feature importance ranking to validate rule-based filters.

Usage:
    python regime1_feature_validator.py
"""

import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from glob import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import mutual_info_regression

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_regime.config import OUTPUT_FOLDER


# FAMILY metrics (all) + DIRECTION metric (only MA50)
ALL_FEATURES = [
    # FAMILY metrics
    'hurst',
    'efficiency_ratio',
    'atr_pct',
    'permutation_entropy',
    # DIRECTION metric
    'price_vs_ma_50'
]


def load_enriched_trades(filepath: str) -> pd.DataFrame:
    """Loads enriched trades from Excel file."""
    df = pd.read_excel(filepath)
    df.columns = df.columns.str.lower().str.strip()
    
    if 'buy_time' in df.columns:
        df['buy_time'] = pd.to_datetime(df['buy_time'])
    
    return df


def prepare_data(df: pd.DataFrame, features: list, train_ratio: float = 0.8):
    """
    Prepares ML data with temporal train/test split.
    
    Args:
        df: Enriched trades dataframe
        features: List of feature column names
        train_ratio: Ratio for train split (0.8 = 80%)
    
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    # Sort by time
    df = df.sort_values('buy_time').reset_index(drop=True)
    
    # Target: profit > 0
    df['is_winner'] = (df['profit'] > 0).astype(int)
    
    # Select features that exist and have data
    available_features = [f for f in features if f in df.columns]
    
    if not available_features:
        return None, None, None, None, []
    
    # Drop rows with NaN in any feature
    df_clean = df[available_features + ['is_winner']].dropna()
    
    if len(df_clean) < 50:  # Minimum trades for reliable analysis
        return None, None, None, None, []
    
    # Temporal split
    split_idx = int(len(df_clean) * train_ratio)
    
    train = df_clean.iloc[:split_idx]
    test = df_clean.iloc[split_idx:]
    
    X_train = train[available_features].values
    y_train = train['is_winner'].values
    
    X_test = test[available_features].values
    y_test = test['is_winner'].values
    
    return X_train, X_test, y_train, y_test, available_features


def calculate_feature_profit_correlation(df: pd.DataFrame, features: list) -> dict:
    """
    Calculates correlation between each feature and profit.
    
    Returns:
        Dict with feature -> correlation coefficient
    """
    correlations = {}
    
    for feature in features:
        if feature in df.columns and 'profit' in df.columns:
            # Drop NaN values for this specific feature
            valid_data = df[[feature, 'profit']].dropna()
            
            if len(valid_data) > 10:
                corr = valid_data[feature].corr(valid_data['profit'])
                correlations[feature] = corr
            else:
                correlations[feature] = np.nan
        else:
            correlations[feature] = np.nan
    
    return correlations


def calculate_mutual_information(df: pd.DataFrame, features: list) -> dict:
    """
    Calculates mutual information between each feature and profit.
    MI captures non-linear relationships.
    
    Returns:
        Dict with feature -> MI score
    """
    mi_scores = {}
    
    for feature in features:
        if feature in df.columns and 'profit' in df.columns:
            # Drop NaN values for this specific feature
            valid_data = df[[feature, 'profit']].dropna()
            
            if len(valid_data) > 30:  # Need more samples for MI
                X = valid_data[[feature]].values
                y = valid_data['profit'].values
                
                # Calculate MI
                mi = mutual_info_regression(X, y, random_state=42)
                mi_scores[feature] = mi[0]
            else:
                mi_scores[feature] = np.nan
        else:
            mi_scores[feature] = np.nan
    
    return mi_scores


def train_and_evaluate(X_train, X_test, y_train, y_test, feature_names: list, strategy_name: str, df_full: pd.DataFrame = None):
    """
    Trains Random Forest and evaluates performance.
    
    Returns:
        Dict with results including feature importance ranking and correlations
    """
    if X_train is None or len(X_train) < 20:
        return None
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=5,
        min_samples_leaf=10,
        random_state=42,
        class_weight='balanced'
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_test = model.predict(X_test)
    
    # Metrics
    test_acc = accuracy_score(y_test, y_pred_test)
    
    # Feature importance
    feature_importance = dict(zip(feature_names, model.feature_importances_))
    
    # Sort by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Baseline (always predict majority class)
    baseline_test = max(np.mean(y_test), 1 - np.mean(y_test))
    
    # Calculate correlations with profit
    correlations = {}
    mi_scores = {}
    if df_full is not None:
        correlations = calculate_feature_profit_correlation(df_full, feature_names)
        mi_scores = calculate_mutual_information(df_full, feature_names)
    
    return {
        'strategy': strategy_name,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'test_acc': test_acc,
        'baseline_test': baseline_test,
        'feature_importance': feature_importance,
        'sorted_features': sorted_features,
        'correlations': correlations,
        'mi_scores': mi_scores
    }


def print_feature_importance_table(results: list):
    """Prints feature importance ranking for each strategy."""
    print("\n" + "=" * 140)
    print("FEATURE IMPORTANCE RANKING PER STRATEGY")
    print("=" * 140)
    
    for r in results:
        print(f"\n{'─'*140}")
        print(f"STRATEGY: {r['strategy']}")
        print(f"  Trades: {r['n_train']} train / {r['n_test']} test")
        print(f"  ML Accuracy: {r['test_acc']*100:.1f}% | Baseline: {r['baseline_test']*100:.1f}% | Improvement: {(r['test_acc'] - r['baseline_test'])*100:+.1f}%")
        print(f"{'─'*140}")
        print(f"{'RANK':<6} {'FEATURE':<25} {'IMPORTANCE':>12} {'CORR':>8} {'MI':>8} {'BAR':<40}")
        print("-" * 140)
        
        for rank, (feature, importance) in enumerate(r['sorted_features'], 1):
            # Visual bar
            bar_length = int(importance * 80)  # Scale to 80 chars max
            bar = '█' * bar_length
            
            # Get correlation
            corr = r['correlations'].get(feature, np.nan)
            if np.isnan(corr):
                corr_str = "N/A"
            else:
                corr_str = f"{corr:+.3f}"
            
            # Get MI score
            mi = r['mi_scores'].get(feature, np.nan)
            if np.isnan(mi):
                mi_str = "N/A"
            else:
                mi_str = f"{mi:.3f}"
            
            print(f"{rank:<6} {feature:<25} {importance:>12.4f} {corr_str:>8} {mi_str:>8} {bar:<40}")
        
        print("-" * 140)


def print_aggregated_feature_importance(results: list):
    """Prints aggregated feature importance across all strategies."""
    print("\n" + "=" * 140)
    print("AGGREGATED FEATURE IMPORTANCE (AVERAGE ACROSS ALL STRATEGIES)")
    print("=" * 140)
    
    # Collect all features
    all_features = set()
    for r in results:
        all_features.update(r['feature_importance'].keys())
    
    # Calculate average importance
    avg_importance = {}
    for feature in all_features:
        importances = [r['feature_importance'].get(feature, 0) for r in results]
        avg_importance[feature] = np.mean(importances)
    
    # Calculate average correlation
    avg_correlation = {}
    for feature in all_features:
        correlations = [r['correlations'].get(feature, np.nan) for r in results if not np.isnan(r['correlations'].get(feature, np.nan))]
        if correlations:
            avg_correlation[feature] = np.mean(correlations)
        else:
            avg_correlation[feature] = np.nan
    
    # Calculate average MI
    avg_mi = {}
    for feature in all_features:
        mi_scores = [r['mi_scores'].get(feature, np.nan) for r in results if not np.isnan(r['mi_scores'].get(feature, np.nan))]
        if mi_scores:
            avg_mi[feature] = np.mean(mi_scores)
        else:
            avg_mi[feature] = np.nan
    
    # Sort by average importance
    sorted_avg = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n{'RANK':<6} {'FEATURE':<25} {'AVG_IMP':>10} {'AVG_CORR':>10} {'AVG_MI':>10} {'BAR':<40}")
    print("-" * 140)
    
    for rank, (feature, importance) in enumerate(sorted_avg, 1):
        # Visual bar
        bar_length = int(importance * 80)
        bar = '█' * bar_length
        
        # Get average correlation
        corr = avg_correlation.get(feature, np.nan)
        if np.isnan(corr):
            corr_str = "N/A"
        else:
            corr_str = f"{corr:+.3f}"
        
        # Get average MI
        mi = avg_mi.get(feature, np.nan)
        if np.isnan(mi):
            mi_str = "N/A"
        else:
            mi_str = f"{mi:.3f}"
        
        print(f"{rank:<6} {feature:<25} {importance:>10.4f} {corr_str:>10} {mi_str:>10} {bar:<40}")
    
    print("-" * 140)


def print_correlation_interpretation(results: list):
    """Prints interpretation of correlation and MI analysis."""
    print("\n" + "=" * 140)
    print("CORRELATION & MUTUAL INFORMATION ANALYSIS - FEATURE vs PROFIT")
    print("=" * 140)
    
    # Collect all features
    all_features = set()
    for r in results:
        all_features.update(r['correlations'].keys())
    
    # Calculate average correlation
    avg_correlation = {}
    for feature in sorted(all_features):
        correlations = [r['correlations'].get(feature, np.nan) for r in results if not np.isnan(r['correlations'].get(feature, np.nan))]
        if correlations:
            avg_correlation[feature] = np.mean(correlations)
        else:
            avg_correlation[feature] = np.nan
    
    # Calculate average MI
    avg_mi = {}
    for feature in sorted(all_features):
        mi_scores = [r['mi_scores'].get(feature, np.nan) for r in results if not np.isnan(r['mi_scores'].get(feature, np.nan))]
        if mi_scores:
            avg_mi[feature] = np.mean(mi_scores)
        else:
            avg_mi[feature] = np.nan
    
    print("\n📊 ANALYSIS (averaged across all strategies):")
    print("-" * 140)
    print(f"{'FEATURE':<25} {'CORR':>8} {'MI':>8} {'CORR_STR':>12} {'MI_STR':>12} {'INTERPRETATION':<60}")
    print("-" * 140)
    
    for feature in sorted(all_features):
        corr = avg_correlation.get(feature, np.nan)
        mi = avg_mi.get(feature, np.nan)
        
        # Format correlation
        if np.isnan(corr):
            corr_str = "N/A"
            corr_strength = "N/A"
        else:
            corr_str = f"{corr:+.3f}"
            abs_corr = abs(corr)
            if abs_corr >= 0.3:
                corr_strength = "🔥 STRONG"
            elif abs_corr >= 0.1:
                corr_strength = "⚠️  WEAK"
            else:
                corr_strength = "❌ NONE"
        
        # Format MI
        if np.isnan(mi):
            mi_str = "N/A"
            mi_strength = "N/A"
        else:
            mi_str = f"{mi:.3f}"
            if mi >= 0.20:
                mi_strength = "🔥 STRONG"
            elif mi >= 0.10:
                mi_strength = "⚠️  MODERATE"
            else:
                mi_strength = "❌ WEAK"
        
        # Interpretation
        if np.isnan(corr) or np.isnan(mi):
            interpretation = "Insufficient data"
        elif abs(corr) < 0.1 and mi >= 0.15:
            interpretation = "🎯 NON-LINEAR relationship (high MI, low corr)"
        elif abs(corr) >= 0.1 and mi >= 0.15:
            if corr > 0:
                interpretation = "↗️  Linear + non-linear (higher → MORE profit)"
            else:
                interpretation = "↘️  Linear + non-linear (higher → LESS profit)"
        elif abs(corr) >= 0.1:
            if corr > 0:
                interpretation = "→ Weak linear positive"
            else:
                interpretation = "→ Weak linear negative"
        else:
            interpretation = "→ No meaningful relationship"
        
        print(f"{feature:<25} {corr_str:>8} {mi_str:>8} {corr_strength:>12} {mi_strength:>12} {interpretation:<60}")
    
    print("-" * 140)
    
    # Key insights
    print("\n💡 KEY INSIGHTS:")
    
    # Non-linear relationships (high MI, low corr)
    nonlinear = [f for f in all_features 
                 if not np.isnan(avg_mi.get(f, np.nan)) 
                 and not np.isnan(avg_correlation.get(f, np.nan))
                 and avg_mi.get(f, 0) >= 0.15 
                 and abs(avg_correlation.get(f, 0)) < 0.1]
    
    if nonlinear:
        print(f"\n  🎯 NON-LINEAR relationships detected (high MI, low correlation):")
        for feature in nonlinear:
            print(f"     • {feature:<25} MI={avg_mi[feature]:.3f}, CORR={avg_correlation[feature]:+.3f}")
        print(f"     → These features predict profit through complex patterns, not simple linear trends")
    
    # Strong linear relationships
    strong_linear = [f for f, c in avg_correlation.items() if not np.isnan(c) and abs(c) >= 0.2]
    
    if strong_linear:
        print(f"\n  ✅ LINEAR relationships:")
        for feature in strong_linear:
            direction = "MORE" if avg_correlation[feature] > 0 else "LESS"
            print(f"     • {feature:<25} CORR={avg_correlation[feature]:+.3f} (higher → {direction} profit)")
    
    # Weak overall
    if not nonlinear and not strong_linear:
        print("\n  ⚠️  No strong relationships found")
        print("     → Check if features interact with each other")
        print("     → Consider per-strategy analysis (relationships may vary)")
    
    print("\n📖 INTERPRETATION GUIDE:")
    print("  CORRELATION (linear relationship):")
    print("     • |CORR| ≥ 0.3 = Strong linear")
    print("     • |CORR| 0.1-0.3 = Weak linear")
    print("     • |CORR| < 0.1 = No linear relationship")
    print("\n  MUTUAL INFORMATION (captures non-linear):")
    print("     • MI ≥ 0.20 = Strong predictive power (may be non-linear)")
    print("     • MI 0.10-0.20 = Moderate predictive power")
    print("     • MI < 0.10 = Weak predictive power")
    print("\n  🎯 IDEAL: Low CORR + High MI = Non-linear pattern (U-shape, threshold effects)")
    
    print("\n" + "=" * 140)


def print_inverted_analysis(results: list):
    """Prints inverted analysis: by feature, showing which strategies find it most important."""
    print("\n" + "=" * 100)
    print("INVERTED ANALYSIS - BY FEATURE (Which strategies find each feature most important)")
    print("=" * 100)
    
    # Collect all features
    all_features = set()
    for r in results:
        all_features.update(r['feature_importance'].keys())
    
    # For each feature, collect importance per strategy
    for feature in sorted(all_features):
        print(f"\n{'─'*100}")
        print(f"FEATURE: {feature}")
        print(f"{'─'*100}")
        print(f"{'RANK':<6} {'STRATEGY':<40} {'IMPORTANCE':>12} {'BAR':<40}")
        print("-" * 100)
        
        # Collect (strategy, importance) pairs for this feature
        feature_data = []
        for r in results:
            if feature in r['feature_importance']:
                feature_data.append((r['strategy'], r['feature_importance'][feature]))
        
        # Sort by importance (descending)
        feature_data.sort(key=lambda x: x[1], reverse=True)
        
        # Print top strategies for this feature
        for rank, (strategy, importance) in enumerate(feature_data, 1):
            bar_length = int(importance * 80)
            bar = '█' * bar_length
            print(f"{rank:<6} {strategy:<40} {importance:>12.4f} {bar:<40}")
        
        print("-" * 100)


def print_interpretation(results: list):
    """Prints interpretation and recommendations."""
    print("\n" + "=" * 100)
    print("INTERPRETATION & RECOMMENDATIONS")
    print("=" * 100)
    
    # Calculate average feature importance across all strategies
    all_features = set()
    for r in results:
        all_features.update(r['feature_importance'].keys())
    
    avg_importance = {}
    for feature in all_features:
        importances = [r['feature_importance'].get(feature, 0) for r in results]
        avg_importance[feature] = np.mean(importances)
    
    sorted_avg = sorted(avg_importance.items(), key=lambda x: x[1], reverse=True)
    
    print("\n🎯 TOP FEATURES (averaged across all strategies):")
    for rank, (feature, importance) in enumerate(sorted_avg, 1):
        # Classify feature type
        if feature in ['hurst', 'efficiency_ratio', 'atr_pct', 'permutation_entropy']:
            feat_type = "← FAMILY"
        elif feature == 'price_vs_ma_50':
            feat_type = "← DIRECTION"
        else:
            feat_type = ""
        
        print(f"  {rank}. {feature:<25} (importance: {importance:.3f}) {feat_type}")
    
    # Analyze FAMILY vs DIRECTION
    print("\n📊 FAMILY vs DIRECTION COMPARISON:")
    
    family_features = ['hurst', 'efficiency_ratio', 'atr_pct', 'permutation_entropy']
    family_total = sum(avg_importance.get(f, 0) for f in family_features)
    direction_total = avg_importance.get('price_vs_ma_50', 0)
    
    print(f"\n  FAMILY metrics total importance:     {family_total:.3f}")
    print(f"  DIRECTION metric importance:          {direction_total:.3f}")
    
    if family_total > direction_total * 1.5:
        print(f"\n  → FAMILY classification is MORE important than DIRECTION")
        print(f"  → Focus on regime characteristics (trending/volatile/ranging)")
    elif direction_total > family_total * 0.5:
        print(f"\n  → DIRECTION is CRITICAL for performance")
        print(f"  → Market direction (uptrend/downtrend) matters significantly")
    else:
        print(f"\n  → Both FAMILY and DIRECTION are important")
        print(f"  → Use full regime classification (family + direction)")
    
    # Top FAMILY feature
    family_sorted = [(f, avg_importance.get(f, 0)) for f in family_features]
    family_sorted.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n📈 TOP FAMILY METRICS:")
    for rank, (feature, importance) in enumerate(family_sorted, 1):
        if feature == 'hurst':
            note = "(trending detection)"
        elif feature == 'efficiency_ratio':
            note = "(trending detection)"
        elif feature == 'atr_pct':
            note = "(volatility detection)"
        elif feature == 'permutation_entropy':
            note = "(complexity detection)"
        else:
            note = ""
        
        print(f"  {rank}. {feature:<25} {importance:.3f} {note}")
    
    # Strategy-level validation
    print(f"\n✅ VALIDATION OF CURRENT RULES:")
    
    # Check if top features align with current classification logic
    top3 = sorted_avg[:3]
    top_features = [f for f, _ in top3]
    
    hurst_er_in_top = ('hurst' in top_features or 'efficiency_ratio' in top_features)
    direction_in_top = 'price_vs_ma_50' in top_features
    
    if hurst_er_in_top and direction_in_top:
        print("  ✅ Current TRENDING classification (Hurst/ER) is ML-validated")
        print("  ✅ Current DIRECTION filter (price_vs_ma_50) is ML-validated")
        print("  → No changes needed to classification logic")
    elif hurst_er_in_top:
        print("  ✅ TRENDING classification is validated")
        print("  ⚠️  DIRECTION filter may be less important than expected")
    elif direction_in_top:
        print("  ✅ DIRECTION filter is validated")
        print("  ⚠️  FAMILY classification may need review")
    else:
        print("  ⚠️  Current rule-based features not in top 3")
        print(f"  → Consider focusing on: {', '.join(top_features)}")
    
    print("\n" + "=" * 100)


def main():
    print("=" * 100)
    print("REGIME 1 FEATURE IMPORTANCE VALIDATOR - FAMILY + DIRECTION Analysis")
    print("=" * 100)
    print(f"\nFeatures:")
    print(f"  FAMILY: hurst, efficiency_ratio, atr_pct, permutation_entropy")
    print(f"  DIRECTION: price_vs_ma_50")
    print(f"\nTarget: is_winner (profit > 0)")
    print(f"Split: 80% train / 20% test (temporal)")
    print(f"Model: Random Forest (100 trees, max_depth=5)")
    
    # Find enriched files
    pattern = os.path.join(OUTPUT_FOLDER, "trades_enriched_*.xlsx")
    files = sorted(glob(pattern))
    
    if not files:
        print(f"\n❌ No enriched files found in {OUTPUT_FOLDER}")
        return
    
    print(f"\nFound {len(files)} strategies")
    
    # Analyze each strategy
    results = []
    
    print("\nProcessing strategies...")
    for filepath in files:
        strategy = Path(filepath).stem.replace('trades_enriched_', '')
        
        # Load data
        df = load_enriched_trades(filepath)
        
        # Prepare train/test
        X_train, X_test, y_train, y_test, feature_names = prepare_data(df, ALL_FEATURES, train_ratio=0.8)
        
        # Train and evaluate (pass full df for correlation analysis)
        result = train_and_evaluate(X_train, X_test, y_train, y_test, feature_names, strategy, df_full=df)
        
        if result:
            results.append(result)
            print(f"  ✅ {strategy}")
        else:
            print(f"  ⚠️  {strategy} (insufficient data)")
    
    if not results:
        print("\n❌ No valid results to display")
        return
    
    # Print results
    print_feature_importance_table(results)
    print_aggregated_feature_importance(results)
    print_correlation_interpretation(results)
    print_inverted_analysis(results)
    print_interpretation(results)
    
    print("\n" + "=" * 100)
    print("ANALYSIS COMPLETE")
    print("=" * 100)


if __name__ == "__main__":
    main()