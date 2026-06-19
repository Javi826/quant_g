"""
Production vs Batch trade comparison tool.

Compares trade metrics between live production trades (xlsx) and batch backtest
trades (csv) over a configurable time window and strategy selection.
"""

import os
import glob
import logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("live_lab.compare")

# =============================================================================
# CONFIGURATION
# =============================================================================

PRODUCTION_XLSX = os.path.expanduser(
    "~/projects/quant/quant_b/bitget/BOT_trading/persistence/bot_files_E1/bot_trades_E1.xlsx"
)
BATCH_TRADES_DIR = os.path.expanduser(
    "~/projects/quant/quant_b/develop/brief_trades"
)

# Batch file pattern: trades_{OOS_PERIOD}_{BATCH_MODE}_{strategy_id}.csv
OOS_PERIOD  = "oos1"       # "oos1" | "oos2" | "oos3"
BATCH_MODE  = "baseline"   # "baseline" | "regime"

# Time window filter (None = no filter)
DATE_FROM = "2026-06-09"
DATE_TO   = "2026-06-18"

# Set to [] to compare all available strategies
SELECTED_STRATEGIES = [
    "05_reversal_long_1H",
    "20_parity_short_6Hutc",
    "22_flag_short_15m",
    "31_orderblocks_long_15m",
    "34_orderblocks_short_30m",
]

# Strategy to plot individually (None to skip)
PLOT_STRATEGY = "31_orderblocks_long_15m"

# Symbols to exclude from production trades before comparison
EXCLUDE_SYMBOLS = [
    "PIPPINUSDT",
]

# =============================================================================
# LOADERS
# =============================================================================

def load_production(path: str) -> pd.DataFrame:
    df = pd.read_excel(path)
    df.columns = [c.strip().upper() for c in df.columns]
    df["OPEN_AT"]  = pd.to_datetime(df["OPEN_AT"],  errors="coerce", utc=True)
    df["CLOSE_AT"] = pd.to_datetime(df["CLOSE_AT"], errors="coerce", utc=True)
    df = df.rename(columns={
        "OPEN_AT":   "buy_time",
        "CLOSE_AT":  "sell_time",
        "STRATEGY":  "strategy",
        "SYMBOL":    "symbol",
        "PROFIT":    "profit",
    })
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
    return df[["buy_time", "sell_time", "strategy", "symbol", "profit"]].dropna(subset=["buy_time"])


def load_batch(trades_dir: str, oos_period: str, mode: str, strategy_ids: list[str]) -> pd.DataFrame:
    frames = []
    pattern = os.path.join(trades_dir, f"trades_{oos_period}_{mode}_*.csv")
    for path in glob.glob(pattern):
        fname    = os.path.basename(path)
        prefix   = f"trades_{oos_period}_{mode}_"
        strat_id = fname.replace(prefix, "").replace(".csv", "")
        if strategy_ids and strat_id not in strategy_ids:
            continue
        try:
            df = pd.read_csv(path)
            df["strategy"] = strat_id
            frames.append(df)
        except Exception as e:
            logger.warning(f"  ⚠️  Could not read {fname}: {e}")

    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    df["buy_time"]  = pd.to_datetime(df["buy_time"],  errors="coerce", utc=True)
    df["sell_time"] = pd.to_datetime(df["sell_time"], errors="coerce", utc=True)
    df["profit"]    = pd.to_numeric(df["profit"],     errors="coerce")
    return df[["buy_time", "sell_time", "strategy", "symbol", "profit"]].dropna(subset=["buy_time"])


# =============================================================================
# FILTERS
# =============================================================================

def apply_filters(
    df:              pd.DataFrame,
    date_from:       str | None,
    date_to:         str | None,
    strategy_ids:    list[str],
    exclude_symbols: list[str] | None = None,
) -> pd.DataFrame:
    if date_from:
        df = df[df["buy_time"] >= pd.Timestamp(date_from, tz="UTC")]
    if date_to:
        df = df[df["buy_time"] <= pd.Timestamp(date_to,   tz="UTC")]
    if strategy_ids:
        df = df[df["strategy"].isin(strategy_ids)]
    if exclude_symbols:
        df = df[~df["symbol"].isin(exclude_symbols)]
    return df.copy()


# =============================================================================
# METRICS
# =============================================================================

def compute_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n_trades": 0, "win_rate": np.nan, "total_profit": np.nan}
    n    = len(df)
    wins = (df["profit"] > 0).sum()
    return {
        "n_trades":     n,
        "win_rate":     round(wins / n * 100, 1),
        "total_profit": round(df["profit"].sum(), 2),
    }


# =============================================================================
# REPORT
# =============================================================================

def print_report(
    results:    list[dict],
    oos_period: str,
    batch_mode: str,
    date_from:  str | None,
    date_to:    str | None,
) -> None:
    period_str = f"{date_from or '—'} → {date_to or '—'}"
    logger.info(f"\n{'='*110}")
    logger.info(f"  PRODUCTION vs BATCH COMPARISON")
    logger.info(f"  Period     : {period_str}")
    logger.info(f"  Batch      : {oos_period.upper()} | {batch_mode}")
    logger.info(f"  Excluded   : {EXCLUDE_SYMBOLS or '—'}")
    logger.info(f"{'='*110}")
    logger.info(
        f"  {'STRATEGY':<32} "
        f"{'N_TR prod':>9} {'N_TR btch':>9} {'Δ':>6} | "
        f"{'WR% prod':>9} {'WR% btch':>9} {'Δ':>6} | "
        f"{'PNL prod':>10} {'PNL btch':>10} {'Δ':>8}"
    )
    logger.info(f"  {'-'*105}")

    for r in results:
        p   = r["prod"]
        b   = r["batch"]
        sid = r["strategy_id"]

        dn = (b["n_trades"] - p["n_trades"]) if (p["n_trades"] and b["n_trades"]) else None
        dw = round(b["win_rate"]     - p["win_rate"],     1) if not (np.isnan(p["win_rate"])     or np.isnan(b["win_rate"]))     else None
        dp = round(b["total_profit"] - p["total_profit"], 2) if not (np.isnan(p["total_profit"]) or np.isnan(b["total_profit"])) else None

        def _fmt(val, fmt=".1f"):
            return f"{val:{fmt}}" if val is not None and not (isinstance(val, float) and np.isnan(val)) else "—"

        def _delta(val, fmt=".1f"):
            if val is None:
                return "—"
            sign = "+" if val > 0 else ""
            return f"{sign}{val:{fmt}}"

        logger.info(
            f"  {sid:<32} "
            f"{_fmt(p['n_trades'], 'd'):>9} {_fmt(b['n_trades'], 'd'):>9} {_delta(dn, 'd'):>6} | "
            f"{_fmt(p['win_rate']):>9} {_fmt(b['win_rate']):>9} {_delta(dw):>6} | "
            f"{_fmt(p['total_profit'], '.2f'):>10} {_fmt(b['total_profit'], '.2f'):>10} {_delta(dp, '.2f'):>8}"
        )

    logger.info(f"  {'='*105}\n")


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    strategy_ids = SELECTED_STRATEGIES or []

    logger.info("  Loading production trades...")
    df_prod = load_production(PRODUCTION_XLSX)
    df_prod = apply_filters(df_prod, None, DATE_TO, strategy_ids, EXCLUDE_SYMBOLS)

    # Use first production trade date as effective start for both sources
    if df_prod.empty:
        logger.warning("  ⚠️  No production trades found.")
        return
    effective_from = df_prod["buy_time"].min().strftime("%Y-%m-%d")
    if DATE_FROM and effective_from < DATE_FROM:
        effective_from = DATE_FROM
    df_prod = apply_filters(df_prod, effective_from, DATE_TO, strategy_ids, EXCLUDE_SYMBOLS)
    logger.info(f"  Effective start : {effective_from}")

    logger.info("  Loading batch trades...")
    df_batch = load_batch(BATCH_TRADES_DIR, OOS_PERIOD, BATCH_MODE, strategy_ids)
    df_batch = apply_filters(df_batch, effective_from, DATE_TO, strategy_ids)

    all_strategies = sorted(
        set(df_prod["strategy"].unique()) | set(df_batch["strategy"].unique())
    )
    if not all_strategies:
        logger.warning("  ⚠️  No trades found for the given filters.")
        return

    results = []
    for sid in all_strategies:
        results.append({
            "strategy_id": sid,
            "prod":        compute_metrics(df_prod[df_prod["strategy"]   == sid]),
            "batch":       compute_metrics(df_batch[df_batch["strategy"] == sid]),
        })

    print_report(results, OOS_PERIOD, BATCH_MODE, effective_from, DATE_TO)
    plot_portfolio(df_prod, df_batch)
    if PLOT_STRATEGY:
        plot_strategy(df_prod, df_batch, PLOT_STRATEGY)


# =============================================================================
# PLOTS — portfolio daily evolution
# =============================================================================

def _daily_portfolio(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trades by day: cumulative profit and daily win rate."""
    df = df.copy()
    df["date"] = df["buy_time"].dt.tz_localize(None).dt.normalize()
    daily = df.groupby("date").agg(
        profit=("profit", "sum"),
        wins=("profit", lambda x: (x > 0).sum()),
        n=("profit", "count"),
    ).reset_index()
    daily["cum_profit"] = daily["profit"].cumsum()
    daily["win_rate"]   = daily["wins"] / daily["n"] * 100
    return daily


def plot_portfolio(df_prod: pd.DataFrame, df_batch: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    daily_prod  = _daily_portfolio(df_prod)
    daily_batch = _daily_portfolio(df_batch)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Portfolio — Production vs Batch", fontsize=13, fontweight="bold")

    # — Cumulative profit —
    ax1.plot(daily_prod["date"],  daily_prod["cum_profit"],  label="Production", color="#2196F3", linewidth=2)
    ax1.plot(daily_batch["date"], daily_batch["cum_profit"], label="Batch",      color="#FF9800", linewidth=2, linestyle="--")
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_ylabel("Cumulative Profit (USDT)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # — Daily win rate —
    ax2.plot(daily_prod["date"],  daily_prod["win_rate"],  label="Production", color="#2196F3", linewidth=2)
    ax2.plot(daily_batch["date"], daily_batch["win_rate"], label="Batch",      color="#FF9800", linewidth=2, linestyle="--")
    ax2.axhline(50, color="gray", linewidth=0.8, linestyle=":")
    ax2.set_ylabel("Win Rate % (daily)")
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_strategy(df_prod: pd.DataFrame, df_batch: pd.DataFrame, strategy_id: str) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    prod_s  = df_prod[df_prod["strategy"]  == strategy_id]
    batch_s = df_batch[df_batch["strategy"] == strategy_id]

    if prod_s.empty and batch_s.empty:
        logger.warning(f"  ⚠️  No trades found for strategy: {strategy_id}")
        return

    daily_prod  = _daily_portfolio(prod_s)  if not prod_s.empty  else None
    daily_batch = _daily_portfolio(batch_s) if not batch_s.empty else None

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Strategy: {strategy_id} — Production vs Batch", fontsize=13, fontweight="bold")

    if daily_prod is not None:
        ax1.plot(daily_prod["date"],  daily_prod["cum_profit"],  label="Production", color="#2196F3", linewidth=2)
        ax2.plot(daily_prod["date"],  daily_prod["win_rate"],    label="Production", color="#2196F3", linewidth=2)
    if daily_batch is not None:
        ax1.plot(daily_batch["date"], daily_batch["cum_profit"], label="Batch",      color="#FF9800", linewidth=2, linestyle="--")
        ax2.plot(daily_batch["date"], daily_batch["win_rate"],   label="Batch",      color="#FF9800", linewidth=2, linestyle="--")

    ax1.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_ylabel("Cumulative Profit (USDT)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.axhline(50, color="gray", linewidth=0.8, linestyle=":")
    ax2.set_ylabel("Win Rate % (daily)")
    ax2.set_ylim(0, 100)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()