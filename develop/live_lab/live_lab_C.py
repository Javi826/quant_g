
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
    "~/projects/quant/quant_b/bitget/BOT_trading/persistence/bot_files_E1/bot_trades_E1.csv"
)
BATCH_TRADES_DIR = os.path.expanduser(
    "~/projects/quant/quant_b/develop/brief_trades"
)

# Batch file pattern: trades_{OOS_PERIOD}_{BATCH_MODE}_{strategy_id}.csv
#OOS_PERIOD  = "oos"      # "oos" | "oos2" | "oos3"
OOS_PERIOD  = "wfo_test"   
BATCH_MODE  = "regime"   # "baseline" | "regime"

# Time window filte (None = no filter)
DATE_FROM = "2026-04-08"
DATE_TO   = "2026-06-30"

# Set to [] to compare all available strategies
SELECTED_STRATEGIES = [
    "05_reversal_long_1H",
    #"20_parity_short_6Hutc",
    "22_flag_short_15m",
    "31_orderblocks_long_15m",
    "34_orderblocks_short_30m",
]

# A production trade matches a batch trade if it opens within this many minutes
# AFTER the batch trade (never before): batch_time <= prod_time <= batch_time + window
MATCH_FORWARD_MINUTES = 3

# Symbols to exclude from production trades before comparison
EXCLUDE_SYMBOLS = [
   # "PIPPINUSDT",
]

# Strategy to plot individually (None to skip)
PLOT_STRATEGY = "05_reversal_long_1H"
#PLOT_STRATEGY = "05_reversal_long_1H"

# Strategy for entry-rounds inspection (None to skip)
ENTRY_ROUNDS_STRATEGY = PLOT_STRATEGY 

# Max gap (seconds) between consecutive buy_times to consider them the same
# simultaneous-open round (signals fired together when the system was flat)
ROUND_GAP_SECONDS = 5

# The batch backtester needs this many future candles to resolve a trade
# (see DEFAULT_CANDLES in ZX_compute.py) — signals within this many candles of
# the end of the data are silently skipped, regardless of sell_after_ncandles.
EDGE_CANDLES = 50


# =============================================================================
# LOADERS
# =============================================================================

def load_production(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.strip().upper() for c in df.columns]
    df["OPEN_AT"]  = pd.to_datetime(df["OPEN_AT"],  errors="coerce", utc=True)
    df["CLOSE_AT"] = pd.to_datetime(df["CLOSE_AT"], errors="coerce", utc=True)
    df = df.rename(columns={
        "OPEN_AT":  "buy_time",
        "CLOSE_AT": "sell_time",
        "STRATEGY": "strategy",
        "SYMBOL":   "symbol",
        "PROFIT":   "profit",
    })
    # Convert European-style decimal comma ("7,25" -> "7.25") before numeric parsing
    df["profit"] = (
        df["profit"].astype(str).str.replace(",", ".", regex=False)
    )
    df["profit"] = pd.to_numeric(df["profit"], errors="coerce")
    return df[["buy_time", "sell_time", "strategy", "symbol", "profit"]].dropna(subset=["buy_time"])


def load_batch(trades_dir: str, oos_period: str, mode: str, strategy_ids: list[str]) -> pd.DataFrame:
    frames = []
    has_mode = not oos_period.startswith("wfo")
    prefix   = f"trades_{oos_period}_{mode}_" if has_mode else f"trades_{oos_period}_"
    pattern  = os.path.join(trades_dir, f"{prefix}*.csv")
    for path in glob.glob(pattern):
        fname    = os.path.basename(path)
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
        df = df[df["buy_time"] <= pd.Timestamp(date_to, tz="UTC")]
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
# TRADE MATCHING — anchor + sequential chain walk per strategy
# =============================================================================

def _is_match(prow: pd.Series, brow: pd.Series, window: pd.Timedelta) -> bool:
    """True if symbols match and prod opens within [0, window] after batch."""
    if prow["symbol"] != brow["symbol"]:
        return False
    delta = prow["buy_time"] - brow["buy_time"]
    return pd.Timedelta(0) <= delta <= window


def _match_chain(p: pd.DataFrame, b: pd.DataFrame, window: pd.Timedelta) -> dict:
    p = p.sort_values("buy_time").reset_index(drop=True)
    b = b.sort_values("buy_time").reset_index(drop=True)

    anchor_p = anchor_b = None
    for bi in range(len(b)):
        for pi in range(len(p)):
            if _is_match(p.iloc[pi], b.iloc[bi], window):
                anchor_p, anchor_b = pi, bi
                break
        if anchor_p is not None:
            break

    if anchor_p is None:
        return {
            "synced":     False,
            "anchor_ts":  None,
            "matched":    0,
            "prod_only":  len(p),
            "batch_only": len(b),
            "wr_agree":   0,
            "chain_len":  0,
        }

    p_sync    = p.iloc[anchor_p:].reset_index(drop=True)
    b_sync    = b.iloc[anchor_b:].reset_index(drop=True)
    anchor_ts = p_sync.iloc[0]["buy_time"]

    matched  = 0
    wr_agree = 0
    used_b   = set()

    for _, prow in p_sync.iterrows():
        for bj in range(len(b_sync)):
            if bj in used_b:
                continue
            if _is_match(prow, b_sync.iloc[bj], window):
                used_b.add(bj)
                matched += 1
                if (prow["profit"] > 0) == (b_sync.iloc[bj]["profit"] > 0):
                    wr_agree += 1
                break

    return {
        "synced":     True,
        "anchor_ts":  anchor_ts,
        "matched":    matched,
        "prod_only":  len(p_sync) - matched,
        "batch_only": len(b_sync) - len(used_b),
        "wr_agree":   wr_agree,
        "chain_len":  max(len(p_sync), len(b_sync)),
    }


def match_trades(df_prod: pd.DataFrame, df_batch: pd.DataFrame, forward_minutes: int) -> dict:
    window     = pd.Timedelta(minutes=forward_minutes)
    strategies = sorted(set(df_prod["strategy"].unique()) | set(df_batch["strategy"].unique()))

    per_strategy = []
    tot = {"matched": 0, "prod_only": 0, "batch_only": 0, "wr_agree": 0}

    for sid in strategies:
        p = df_prod[df_prod["strategy"]   == sid]
        b = df_batch[df_batch["strategy"] == sid]
        r = _match_chain(p, b, window)
        r["strategy_id"]  = sid
        r["wr_agree_pct"] = round(r["wr_agree"] / r["matched"] * 100, 1) if r["matched"] else None
        r["match_pct"]    = round(r["matched"] / r["chain_len"] * 100, 1) if r["chain_len"] else None
        per_strategy.append(r)
        for k in tot:
            tot[k] += r[k]

    tot["wr_agree_pct"] = round(tot["wr_agree"] / tot["matched"] * 100, 1) if tot["matched"] else None
    return {"per_strategy": per_strategy, "totals": tot}


def print_match_report(match_result: dict, forward_minutes: int) -> None:
    logger.info(f"\n{'='*120}")
    logger.info(f"  TRADE MATCHING — anchor-synced per strategy | window +{forward_minutes} min (prod after batch)")
    logger.info(f"{'='*120}")
    logger.info(
        f"  {'STRATEGY':<32} {'ANCHOR':<20} {'MATCHED':>8} {'P_ONLY':>7} {'B_ONLY':>7} "
        f"{'MATCH%':>7} {'WR_OK':>6} {'WR_OK%':>7}"
    )
    logger.info(f"  {'-'*100}")

    for r in match_result["per_strategy"]:
        anchor = str(r["anchor_ts"])[:19] if r["anchor_ts"] is not None else "✗ no sync"
        wr_pct = f"{r['wr_agree_pct']}" if r["wr_agree_pct"] is not None else "—"
        m_pct  = f"{r['match_pct']}"    if r["match_pct"]    is not None else "—"
        logger.info(
            f"  {r['strategy_id']:<32} {anchor:<20} {r['matched']:>8} {r['prod_only']:>7} {r['batch_only']:>7} "
            f"{m_pct:>7} {r['wr_agree']:>6} {wr_pct:>7}"
        )

    t      = match_result["totals"]
    wr_pct = f"{t['wr_agree_pct']}" if t["wr_agree_pct"] is not None else "—"
    logger.info(f"  {'-'*100}")
    logger.info(
        f"  {'SYSTEM TOTAL':<32} {'':<20} {t['matched']:>8} {t['prod_only']:>7} {t['batch_only']:>7} "
        f"{'':>7} {t['wr_agree']:>6} {wr_pct:>7}"
    )
    logger.info(f"  {'='*120}\n")


def print_trade_pairs(
    df_prod:     pd.DataFrame,
    df_batch:    pd.DataFrame,
    strategy_id: str,
    anchor_ts:   pd.Timestamp,
    window:      pd.Timedelta,
) -> None:
    p = (
        df_prod[(df_prod["strategy"] == strategy_id) & (df_prod["buy_time"] >= anchor_ts)]
        .sort_values("buy_time")
        .reset_index(drop=True)
    )
    b = (
        df_batch[(df_batch["strategy"] == strategy_id) & (df_batch["buy_time"] >= anchor_ts)]
        .sort_values("buy_time")
        .reset_index(drop=True)
    )

    logger.info(f"\n{'='*100}")
    logger.info(f"  TRADE-BY-TRADE COMPARISON — {strategy_id} | anchor {anchor_ts} | window +{int(window.total_seconds() / 60)} min")
    logger.info(f"{'='*100}")
    logger.info(
        f"  {'SYMBOL':<14} {'PROD_TIME':<20} {'BATCH_TIME':<20} {'PROD_P':>9} {'BATCH_P':>9} {'MATCH':>6}"
    )
    logger.info(f"  {'-'*100}")

    used_b = set()
    for _, prow in p.iterrows():
        match_idx = None
        for bj in range(len(b)):
            if bj in used_b:
                continue
            if _is_match(prow, b.iloc[bj], window):
                match_idx = bj
                used_b.add(bj)
                break

        if match_idx is not None:
            brow = b.iloc[match_idx]
            ok   = "✓" if (prow["profit"] > 0) == (brow["profit"] > 0) else "✗"
            logger.info(
                f"  {prow['symbol']:<14} {str(prow['buy_time'])[:19]:<20} {str(brow['buy_time'])[:19]:<20} "
                f"{prow['profit']:>9.2f} {brow['profit']:>9.2f} {ok:>6}"
            )
        else:
            logger.info(
                f"  {prow['symbol']:<14} {str(prow['buy_time'])[:19]:<20} {'— no match —':<20} "
                f"{prow['profit']:>9.2f} {'—':>9} {'✗':>6}"
            )

    for bi in (bi for bi in range(len(b)) if bi not in used_b):
        brow = b.iloc[bi]
        logger.info(
            f"  {brow['symbol']:<14} {'— no match —':<20} {str(brow['buy_time'])[:19]:<20} "
            f"{'—':>9} {brow['profit']:>9.2f} {'✗':>6}"
        )

    logger.info(f"  {'='*100}\n")


def print_trade_pairs_summary(
    df_prod:     pd.DataFrame,
    df_batch:    pd.DataFrame,
    strategy_id: str,
    anchor_ts:   pd.Timestamp,
    window:      pd.Timedelta,
) -> None:
    p = (
        df_prod[(df_prod["strategy"] == strategy_id) & (df_prod["buy_time"] >= anchor_ts)]
        .sort_values("buy_time").reset_index(drop=True)
    )
    b = (
        df_batch[(df_batch["strategy"] == strategy_id) & (df_batch["buy_time"] >= anchor_ts)]
        .sort_values("buy_time").reset_index(drop=True)
    )

    matched_prod, matched_batch = [], []
    used_b = set()
    for _, prow in p.iterrows():
        for bj in range(len(b)):
            if bj in used_b:
                continue
            if _is_match(prow, b.iloc[bj], window):
                used_b.add(bj)
                matched_prod.append(prow["profit"])
                matched_batch.append(b.iloc[bj]["profit"])
                break

    if not matched_prod:
        logger.info("  No matched trades to summarize.")
        return

    mp    = pd.Series(matched_prod)
    mb    = pd.Series(matched_batch)
    agree = (mp > 0) == (mb > 0)

    logger.info(f"\n{'='*60}")
    logger.info(f"  MATCHED TRADES SUMMARY — {strategy_id}")
    logger.info(f"{'='*60}")
    logger.info(f"  Matched pairs   : {len(mp)}")
    logger.info(f"  Prod  PNL       : {mp.sum():+.2f}  (WR {(mp>0).sum()/len(mp)*100:.1f}%)")
    logger.info(f"  Batch PNL       : {mb.sum():+.2f}  (WR {(mb>0).sum()/len(mb)*100:.1f}%)")
    logger.info(f"  PNL diff        : {mb.sum()-mp.sum():+.2f}")
    logger.info(f"  Direction agree : {agree.sum()}/{len(agree)} ({agree.sum()/len(agree)*100:.1f}%)")
    logger.info(f"  {'='*60}\n")


# =============================================================================
# WR DAILY SCORE
# =============================================================================

def compute_wr_daily_score(df_prod: pd.DataFrame, df_batch: pd.DataFrame) -> dict:
    """% of days where rounded daily win rate matches between prod and batch."""
    def daily_wr(df):
        df = df.copy()
        df["date"] = df["buy_time"].dt.tz_localize(None).dt.normalize()
        return df.groupby("date").apply(
            lambda x: round((x["profit"] > 0).sum() / len(x) * 100, 0),
            include_groups=False,
        ).rename("wr")

    prod_wr  = daily_wr(df_prod)
    batch_wr = daily_wr(df_batch)
    merged   = prod_wr.to_frame().join(batch_wr.to_frame(), lsuffix="_prod", rsuffix="_batch", how="inner")
    if merged.empty:
        return {"days_common": 0, "days_match": 0, "score_pct": None}
    days_match = (merged["wr_prod"] == merged["wr_batch"]).sum()
    return {
        "days_common": len(merged),
        "days_match":  int(days_match),
        "score_pct":   round(days_match / len(merged) * 100, 1),
    }


def print_wr_score_report(
    df_prod:  pd.DataFrame,
    df_batch: pd.DataFrame,
    label:    str = "GLOBAL",
) -> None:
    strategies = sorted(set(df_prod["strategy"].unique()) | set(df_batch["strategy"].unique()))

    logger.info(f"\n{'='*80}")
    logger.info(f"  WR DAILY SCORE ({label}) — % days where round(WR,0) matches prod vs batch")
    logger.info(f"{'='*80}")
    logger.info(f"  {'STRATEGY':<32} {'DAYS_COMMON':>12} {'DAYS_MATCH':>11} {'SCORE%':>8}")
    logger.info(f"  {'-'*65}")

    total_common = total_match = 0
    for sid in strategies:
        p = df_prod[df_prod["strategy"]   == sid]
        b = df_batch[df_batch["strategy"] == sid]
        r = compute_wr_daily_score(p, b)
        score_str = f"{r['score_pct']}" if r["score_pct"] is not None else "—"
        logger.info(f"  {sid:<32} {r['days_common']:>12} {r['days_match']:>11} {score_str:>8}")
        total_common += r["days_common"]
        total_match  += r["days_match"]

    total_score = round(total_match / total_common * 100, 1) if total_common else None
    score_str   = f"{total_score}" if total_score is not None else "—"
    logger.info(f"  {'-'*65}")
    logger.info(f"  {'SYSTEM TOTAL':<32} {total_common:>12} {total_match:>11} {score_str:>8}")
    logger.info(f"  {'='*80}\n")


# =============================================================================
# ENTRY ROUNDS
# =============================================================================

def _timeframe_to_offset(strategy_id: str) -> pd.Timedelta:
    """Extract candle size from strategy_id suffix (e.g. '_15m', '_1H', '_6Hutc')."""
    suffix = strategy_id.split("_")[-1].replace("utc", "")
    unit   = suffix[-1]
    value  = int(suffix[:-1])
    if unit == "m":
        return pd.Timedelta(minutes=value)
    if unit == "H":
        return pd.Timedelta(hours=value)
    if unit == "D":
        return pd.Timedelta(days=value)
    raise ValueError(f"Unknown timeframe in strategy_id: {strategy_id}")


def _group_into_rounds(df: pd.DataFrame, gap_seconds: int) -> list[dict]:
    df = df.sort_values("buy_time").reset_index(drop=True)
    if df.empty:
        return []

    rounds         = []
    round_start    = df.loc[0, "buy_time"]
    round_symbols  = [df.loc[0, "symbol"]]
    round_sell_max = df.loc[0, "sell_time"]

    for i in range(1, len(df)):
        gap = (df.loc[i, "buy_time"] - df.loc[i - 1, "buy_time"]).total_seconds()
        if gap > gap_seconds:
            rounds.append({"round_start": round_start, "round_end": round_sell_max, "symbols": round_symbols})
            round_start    = df.loc[i, "buy_time"]
            round_symbols  = []
            round_sell_max = df.loc[i, "sell_time"]
        round_symbols.append(df.loc[i, "symbol"])
        round_sell_max = max(round_sell_max, df.loc[i, "sell_time"])

    rounds.append({"round_start": round_start, "round_end": round_sell_max, "symbols": round_symbols})
    return rounds


def _nearest_round_delta(ts: pd.Timestamp, other_rounds: list[dict]) -> float | None:
    """Signed delta in minutes (ts - nearest.round_start) to the closest round."""
    if not other_rounds:
        return None
    deltas = [(ts - r["round_start"]).total_seconds() / 60.0 for r in other_rounds]
    return min(deltas, key=abs)


def _is_other_side_busy(other_df: pd.DataFrame, ts: pd.Timestamp) -> bool:
    """True if other_df has any trade open at ts (buy_time <= ts < sell_time)."""
    if other_df.empty:
        return False
    mask = (other_df["buy_time"] <= ts) & (other_df["sell_time"] > ts)
    return bool(mask.any())


def print_entry_rounds_report(
    df_prod:      pd.DataFrame,
    df_batch:     pd.DataFrame,
    strategy_id:  str,
    anchor_ts:    pd.Timestamp,
    window:       pd.Timedelta,
    gap_seconds:  int,
    data_end_ts:  pd.Timestamp,
    edge_candles: int = 50,
) -> None:
    p = df_prod[(df_prod["strategy"] == strategy_id) & (df_prod["buy_time"] >= anchor_ts)]
    b = df_batch[(df_batch["strategy"] == strategy_id) & (df_batch["buy_time"] >= anchor_ts)]

    p_rounds      = _group_into_rounds(p, gap_seconds)
    b_rounds      = _group_into_rounds(b, gap_seconds)
    timeframe     = _timeframe_to_offset(strategy_id)
    timeframe_min = timeframe.total_seconds() / 60.0
    edge_cutoff   = data_end_ts - edge_candles * timeframe

    logger.info(f"\n{'='*110}")
    logger.info(f"  ENTRY ROUNDS COMPARISON — {strategy_id} | anchor {anchor_ts} | window +{int(window.total_seconds() / 60)} min")
    logger.info(f"  Edge cutoff: rounds >= {edge_cutoff} flagged (batch needs {edge_candles} future candles to close a trade)")
    logger.info(f"{'='*110}")
    logger.info(
        f"  {'PROD_ROUND':<20} {'BATCH_ROUND':<20} {'N_PROD':>7} {'N_BATCH':>8} {'MATCH':>6} "
        f"{'NEAREST_Δ(min)':>15} {'STATUS':>9} {'CLOSE_PROD':<20} {'CLOSE_BATCH':<20} {'Δclose(min)':>12}"
    )
    logger.info(f"  {'-'*110}")

    near_timeframe_count = 0
    unmatched_count      = 0
    edge_count           = 0
    busy_count           = 0
    used_b               = set()

    for pr in p_rounds:
        match_idx = None
        for bi, br in enumerate(b_rounds):
            if bi in used_b:
                continue
            delta = pr["round_start"] - br["round_start"]
            if pd.Timedelta(0) <= delta <= window:
                match_idx = bi
                used_b.add(bi)
                break

        if match_idx is not None:
            br          = b_rounds[match_idx]
            close_delta = (pr["round_end"] - br["round_end"]).total_seconds() / 60.0
            logger.info(
                f"  {str(pr['round_start'])[:19]:<20} {str(br['round_start'])[:19]:<20} "
                f"{len(pr['symbols']):>7} {len(br['symbols']):>8} {'✓':>6} {'—':>15} {'—':>9} "
                f"{str(pr['round_end'])[:19]:<20} {str(br['round_end'])[:19]:<20} {close_delta:>+12.1f}"
            )
        elif pr["round_start"] >= edge_cutoff:
            logger.info(
                f"  {str(pr['round_start'])[:19]:<20} {'— no match —':<20} "
                f"{len(pr['symbols']):>7} {'—':>8} {'⚠️ edge':>6} {'—':>15} {'—':>9} "
                f"{'—':<20} {'—':<20} {'—':>12}"
            )
            edge_count += 1
        else:
            nearest_delta = _nearest_round_delta(pr["round_start"], b_rounds)
            delta_str     = f"{nearest_delta:+.1f}" if nearest_delta is not None else "—"
            status        = "OCUPADO" if _is_other_side_busy(b, pr["round_start"]) else "LIBRE"
            if status == "OCUPADO":
                busy_count += 1
            logger.info(
                f"  {str(pr['round_start'])[:19]:<20} {'— no match —':<20} "
                f"{len(pr['symbols']):>7} {'—':>8} {'✗':>6} {delta_str:>15} {status:>9} "
                f"{'—':<20} {'—':<20} {'—':>12}"
            )
            unmatched_count += 1
            if nearest_delta is not None and abs(abs(nearest_delta) % timeframe_min) <= 5:
                near_timeframe_count += 1

    for bi in (bi for bi in range(len(b_rounds)) if bi not in used_b):
        br            = b_rounds[bi]
        nearest_delta = _nearest_round_delta(br["round_start"], p_rounds)
        delta_str     = f"{nearest_delta:+.1f}" if nearest_delta is not None else "—"
        status        = "OCUPADO" if _is_other_side_busy(p, br["round_start"]) else "LIBRE"
        if status == "OCUPADO":
            busy_count += 1
        logger.info(
            f"  {'— no match —':<20} {str(br['round_start'])[:19]:<20} "
            f"{'—':>7} {len(br['symbols']):>8} {'✗':>6} {delta_str:>15} {status:>9} "
            f"{'—':<20} {'—':<20} {'—':>12}"
        )
        unmatched_count += 1
        if nearest_delta is not None and abs(abs(nearest_delta) % timeframe_min) <= 5:
            near_timeframe_count += 1

    n_matched = len(used_b)
    logger.info(f"  {'-'*110}")
    logger.info(f"  Rounds matched: {n_matched} | prod rounds: {len(p_rounds)} | batch rounds: {len(b_rounds)} | edge-flagged: {edge_count}")
    logger.info(f"  Unmatched rounds where the other side was busy (OCUPADO): {busy_count}/{unmatched_count}")
    logger.info(
        f"  Unmatched rounds near a multiple of the {timeframe_min:.0f}-min timeframe (±5 min): "
        f"{near_timeframe_count}/{unmatched_count}"
    )
    logger.info(f"  {'='*110}\n")




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

    ax1.plot(daily_prod["date"],  daily_prod["cum_profit"],  label="Production", color="#2196F3", linewidth=2)
    ax1.plot(daily_batch["date"], daily_batch["cum_profit"], label="Batch",      color="#FF9800", linewidth=2, linestyle="--")
    ax1.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax1.set_ylabel("Cumulative Profit (USDT)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(daily_prod["date"],  daily_prod["win_rate"],  label="Production", color="#2196F3", linewidth=2)
    ax2.plot(daily_batch["date"], daily_batch["win_rate"], label="Batch",      color="#FF9800", linewidth=2, linestyle="--")
    ax2.axhline(50, color="gray", linewidth=0.8, linestyle=":")
    ax2.set_ylabel("Win Rate % (daily)")
    ax2.set_ylim(0, 110)
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

    prod_s  = df_prod[df_prod["strategy"]   == strategy_id]
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
    ax2.set_ylim(0, 110)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax2.xaxis.set_major_locator(mdates.DayLocator())
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    strategy_ids = SELECTED_STRATEGIES or []

    logger.info("  Loading production trades...")
    df_prod = load_production(PRODUCTION_XLSX)
    df_prod = apply_filters(df_prod, None, DATE_TO, strategy_ids, EXCLUDE_SYMBOLS)

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

    match_result = match_trades(df_prod, df_batch, MATCH_FORWARD_MINUTES)
    print_match_report(match_result, MATCH_FORWARD_MINUTES)

    if ENTRY_ROUNDS_STRATEGY:
        anchor = next(
            (r["anchor_ts"] for r in match_result["per_strategy"] if r["strategy_id"] == ENTRY_ROUNDS_STRATEGY),
            None,
        )
        if anchor is not None:
            print_trade_pairs(
                df_prod, df_batch, ENTRY_ROUNDS_STRATEGY, anchor,
                pd.Timedelta(minutes=MATCH_FORWARD_MINUTES),
            )
            print_trade_pairs_summary(
                df_prod, df_batch, ENTRY_ROUNDS_STRATEGY, anchor,
                pd.Timedelta(minutes=MATCH_FORWARD_MINUTES),
            )
            print_entry_rounds_report(
                df_prod, df_batch, ENTRY_ROUNDS_STRATEGY, anchor,
                pd.Timedelta(minutes=MATCH_FORWARD_MINUTES), ROUND_GAP_SECONDS,
                data_end_ts=pd.Timestamp(DATE_TO, tz="UTC"), edge_candles=EDGE_CANDLES,
            )
        else:
            logger.warning(f"  ⚠️  No anchor found for strategy: {ENTRY_ROUNDS_STRATEGY}")

    print_wr_score_report(df_prod, df_batch, label="GLOBAL")

    plot_portfolio(df_prod, df_batch)
    if PLOT_STRATEGY:
        plot_strategy(df_prod, df_batch, PLOT_STRATEGY)


if __name__ == "__main__":
    main()