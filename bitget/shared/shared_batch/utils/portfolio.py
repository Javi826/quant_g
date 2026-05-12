#shared_batch/pipeline/portfolio.py
import logging

import numpy as np
import pandas as pd

from shared_batch.utils.metrics import compute_metrics

logger = logging.getLogger("BOT_batch.utils.portfolio")


# =============================================================================
# PRIVATE HELPERS
# =============================================================================

def _num(sid: str) -> int:
    for part in sid.split("_"):
        if part.isdigit():
            return int(part)
    return 0


def _profit_series(df: pd.DataFrame, capital: float) -> pd.Series:
    tl          = df.copy()
    tl["_date"] = pd.to_datetime(tl["sell_time"]).dt.normalize()
    daily       = tl.groupby("_date")["profit"].sum().groupby(level=0).sum()
    date_range  = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="1D")
    return daily.reindex(date_range).fillna(0.0)


def _equity_series(strategy_trades, capital: float) -> pd.Series:
    """Convert trades to daily equity series."""
    all_tl = (
        pd.concat([df for _, df in strategy_trades], ignore_index=True)
        if isinstance(strategy_trades, list)
        else strategy_trades
    )
    all_tl          = all_tl.sort_values("sell_time").reset_index(drop=True)
    all_tl["_date"] = pd.to_datetime(all_tl["sell_time"]).dt.normalize()
    daily           = all_tl.groupby("_date")["profit"].sum()
    date_range      = pd.date_range(start=daily.index.min(), end=daily.index.max(), freq="1D")
    daily           = daily.reindex(date_range, fill_value=0.0)
    equity          = capital + daily.cumsum()
    return pd.Series(equity.values, index=date_range)


def _decorrelate(
    strategy_trades_oos1: list,
    strategy_trades_oos2: list,
    initial_balance: float,
    threshold: float,
    precomputed_metrics: dict,
    strategy_trades_oos3: list,
    series_fn,
    label: str,
) -> list:
    metrics  = precomputed_metrics or {
        sid: compute_metrics(df, capital=initial_balance, name=sid)
        for sid, df in strategy_trades_oos1
    }
    oos1_map = {sid: df for sid, df in strategy_trades_oos1}
    oos2_map = {sid: df for sid, df in strategy_trades_oos2}
    oos3_map = {sid: df for sid, df in (strategy_trades_oos3 or [])}
    all_sids = [sid for sid, _ in strategy_trades_oos1]

    series_combined = {}
    for sid in all_sids:
        parts = []
        if sid in oos1_map:
            parts.append(series_fn(oos1_map[sid], initial_balance))
        if sid in oos2_map:
            parts.append(series_fn(oos2_map[sid], initial_balance))
        if sid in oos3_map:
            parts.append(series_fn(oos3_map[sid], initial_balance))
        if parts:
            combined             = pd.concat(parts).sort_index()
            series_combined[sid] = combined.groupby(level=0).mean()

    if len(series_combined) < 2:
        logger.info("  Not enough strategies for correlation analysis.")
        return strategy_trades_oos1

    num_map = {sid: f"{_num(sid):02d}" for sid in series_combined}
    df_     = pd.DataFrame({num_map[sid]: s for sid, s in series_combined.items()}).fillna(0)
    corr_mx = df_.corr().round(2)
    logger.debug(f"\n{corr_mx.to_string()}")

    ranked    = sorted(all_sids, key=lambda s: metrics.get(s, {}).get("Net_Gain_pct", 0), reverse=True)
    selected  = []
    discarded = []
    lines     = [f"\n  {'Rank':<6} {'Strategy':<30} {'NetGain%':>10} {'Action':<20} {'Reason'}"]
    lines.append(f"  {'─'*85}")

    for sid in ranked:
        ng         = metrics.get(sid, {}).get("Net_Gain_pct", 0)
        num        = num_map.get(sid, sid)
        correlated = False
        reason     = ""
        for kept in selected:
            kept_num = num_map.get(kept, kept)
            val      = corr_mx.loc[num, kept_num] if num in corr_mx.index and kept_num in corr_mx.columns else 0.0
            if pd.notna(val) and val > threshold:
                correlated = True
                reason     = f"corr={val:.2f} with {kept}"
                discarded.append(sid)
                break
        if correlated:
            lines.append(f"  {ranked.index(sid)+1:<6} {sid:<30} {ng:>9.2f}%  {'❌ DISCARDED':<20} {reason}")
        else:
            selected.append(sid)
            lines.append(f"  {ranked.index(sid)+1:<6} {sid:<30} {ng:>9.2f}%  {'✅ SELECTED':<20}")

    lines.append(f"  {'─'*85}")
    logger.info("\n".join(lines))

    return [(sid, oos1_map[sid]) for sid in sorted(selected, key=_num) if sid in oos1_map]


# =============================================================================
# PUBLIC FUNCTIONS
# =============================================================================

def decorrelate_by_profit(
    strategy_trades_oos1: list,
    strategy_trades_oos2: list,
    initial_balance: float,
    threshold: float = 0.7,
    precomputed_metrics: dict = None,
    strategy_trades_oos3: list = None,
) -> list:
    """Greedy profit-correlation filter. Keeps best NetGain from each correlated pair."""
    return _decorrelate(
        strategy_trades_oos1, strategy_trades_oos2, initial_balance,
        threshold, precomputed_metrics, strategy_trades_oos3,
        series_fn=_profit_series, label="Profit",
    )


def find_complementary_portfolio(
    main_portfolio_ids: list,
    all_strategy_trades_oos1: list,
    all_strategy_trades_oos2: list,
    all_strategy_trades_oos3: list,
    initial_balance: float,
    max_correlation: float = 0.5,
    top_n: int = 5,
) -> list:
    """Find complementary portfolio with low correlation to main portfolio."""
    oos1_map = {sid: df for sid, df in all_strategy_trades_oos1}
    oos2_map = {sid: df for sid, df in all_strategy_trades_oos2}
    oos3_map = {sid: df for sid, df in all_strategy_trades_oos3}
    all_sids = list(oos1_map.keys())

    main_trades_combined = []
    for sid in main_portfolio_ids:
        parts = []
        if sid in oos1_map: parts.append((sid, oos1_map[sid]))
        if sid in oos2_map: parts.append((sid, oos2_map[sid]))
        if sid in oos3_map: parts.append((sid, oos3_map[sid]))
        if parts:
            combined_df = pd.concat([df for _, df in parts], ignore_index=True)
            main_trades_combined.append((sid, combined_df))

    if not main_trades_combined:
        logger.warning("Main portfolio is empty")
        return []

    main_capital = initial_balance * len(main_trades_combined)
    main_eq      = _equity_series(main_trades_combined, main_capital)

    candidates = []
    for sid in all_sids:
        if sid in main_portfolio_ids:
            continue

        parts = []
        if sid in oos1_map: parts.append((sid, oos1_map[sid]))
        if sid in oos2_map: parts.append((sid, oos2_map[sid]))
        if sid in oos3_map: parts.append((sid, oos3_map[sid]))
        if not parts:
            continue

        combined_df  = pd.concat([df for _, df in parts], ignore_index=True)
        strategy_eq  = _equity_series([(sid, combined_df)], initial_balance)
        common_dates = main_eq.index.intersection(strategy_eq.index)

        if len(common_dates) < 10:
            continue

        corr            = np.corrcoef(main_eq.loc[common_dates].values, strategy_eq.loc[common_dates].values)[0, 1]
        m               = compute_metrics(oos1_map[sid], capital=initial_balance, name=sid)
        anti_corr       = 1 - abs(corr)
        risk_adj_return = m["Net_Gain_pct"] / abs(m["Max_DD_pct"]) if m["Max_DD_pct"] != 0 else 0

        candidates.append({
            "strategy_id":    sid,
            "trades_df":      oos1_map[sid],
            "correlation":    round(corr, 2),
            "anti_corr_score": round(anti_corr * risk_adj_return, 2),
            "netgain_pct":    m["Net_Gain_pct"],
            "maxdd_pct":      m["Max_DD_pct"],
        })

    filtered          = [c for c in candidates if abs(c["correlation"]) < max_correlation]
    sorted_candidates = sorted(filtered, key=lambda x: x["anti_corr_score"], reverse=True)

    logger.info(f"\n{'─'*115}")
    logger.info(f"  COMPLEMENTARY PORTFOLIO ANALYSIS (vs Main Portfolio)")
    logger.info(f"{'─'*115}")
    logger.info(f"  Main Portfolio: {main_portfolio_ids}")
    logger.info(f"  Correlation threshold: {max_correlation}")
    logger.info(f"\n  {'Strategy':<30} {'Corr':>6} {'Score':>7} {'NetGain%':>10} {'MaxDD%':>9}")
    logger.info(f"  {'-'*80}")
    for c in sorted_candidates[:top_n * 2]:
        logger.info(
            f"  {c['strategy_id']:<30} {c['correlation']:>6.2f} {c['anti_corr_score']:>7.2f} "
            f"{c['netgain_pct']:>9.1f}% {c['maxdd_pct']:>8.2f}%"
        )
    logger.info(f"  {'-'*80}\n")

    return [(c["strategy_id"], c["trades_df"]) for c in sorted_candidates[:top_n]]