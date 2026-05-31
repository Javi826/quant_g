#develop/market_regime/corr_anlysis.py
"""
Point-biserial and Mutual Information correlation between
BTC/symbol 1D OHLCV-derived variables and trade outcomes.

For each baseline trade across all OOS periods:
  1. Load 1D candle of the day before buy_time (lookahead-safe)
  2. Compute ~33 derived variables from that candle
  3. Compute point-biserial correlation and mutual information vs (profit > 0)

Tables:
  - Global PB ranking by abs(point_biserial) — top N
  - Global MI ranking by MI score — top N
  - By-period PB with consistency score — top N
  - By-period MI with consistency score — top N
"""
import os
import glob
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import mutual_info_classif

# =============================================================================
# CONFIGURATION
# =============================================================================

TRADES_FOLDER  = os.path.expanduser("~/projects/quant/quant_b/develop/brief_trades")
BTC_FULL_PATH  = os.path.expanduser("~/projects/quant/quant_b/bitget/data_pipeline/data/04_split_OLD/expanding/IS/crypto_full_IS/BTCUSDT_1Dutc.parquet")

ER_WINDOW      = 20
VOL_MA_WINDOW  = 20
ATR_WINDOW     = 14
RANK_WINDOW    = 20
MIN_TRADES     = 30
TOP_N          = 10

ANALYSIS_MODE  = "BTC"   # "BTC" | "SYMBOL"

# =============================================================================
# LOAD OHLCV
# =============================================================================

CRYPTO_FULL_DIR = os.path.expanduser("~/projects/quant/quant_b/bitget/data_pipeline/data/04_split_OLD/expanding/IS/crypto_full_IS")


def _load_ohlcv(symbol: str) -> pd.DataFrame:
    path = os.path.join(CRYPTO_FULL_DIR, f"{symbol}_1Dutc.parquet")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_parquet(path)
    df.columns = [c.lower().strip() for c in df.columns]

    if df.index.name and df.index.name.lower() in ("timestamp", "ts", "date", "time"):
        df.index.name = "ts"
        df = df.reset_index()

    rename_map = {"timestamp": "ts", "open_time": "ts", "date": "ts", "time": "ts",
                  "volume_quote": "volume", "vol": "volume"}
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    if "volume_base" in df.columns and "volume_quote" in df.columns:
        df.drop(columns=["volume_base"], inplace=True)

    df["ts"] = pd.to_datetime(df["ts"], errors="coerce")
    df["ts"] = df["ts"].dt.tz_localize("UTC") if df["ts"].dt.tz is None else df["ts"].dt.tz_convert("UTC")
    df.dropna(subset=["ts"], inplace=True)
    df.sort_values("ts", inplace=True)
    df.drop_duplicates(subset=["ts"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)

    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.dropna(subset=["close"], inplace=True)
    return df


def _load_btc() -> pd.DataFrame:
    return _load_ohlcv("BTCUSDT")


# =============================================================================
# PRECOMPUTE INDICATORS
# =============================================================================

def _precompute_indicators(btc: pd.DataFrame) -> pd.DataFrame:
    c = btc["close"].values
    o = btc["open"].values
    h = btc["high"].values
    l = btc["low"].values
    v = btc["volume"].values if "volume" in btc.columns else np.ones(len(c))
    n = len(c)

    vars_arr = {k: np.full(n, np.nan) for k in [
        "er", "range_norm", "body_size", "upper_shadow", "lower_shadow",
        "vol_rel", "return_1d", "atr_norm", "range_expansion", "close_position",
        "gap", "vol_acceleration", "high_pct_rank", "low_pct_rank",
        "trend_strength", "mean_reversion", "vol_percentile",
        "range_percentile", "consecutive_up", "body_direction",
        "hl_ratio", "range_vs_atr", "vol_trend",
        "price_pos_52w", "drawdown_peak", "recovery_trough",
        "momentum_5d", "momentum_20d", "momentum_ratio",
        "vol_regime", "realized_vol", "amihud", "turnover_rel",
    ]}

    for i in range(1, n):
        hl    = h[i] - l[i]
        c_pos = (c[i] - l[i]) / hl if hl > 0 else np.nan

        vars_arr["range_norm"][i]     = hl / c[i] if c[i] > 0 else np.nan
        vars_arr["body_size"][i]      = abs(c[i] - o[i]) / hl if hl > 0 else np.nan
        vars_arr["upper_shadow"][i]   = (h[i] - max(o[i], c[i])) / hl if hl > 0 else np.nan
        vars_arr["lower_shadow"][i]   = (min(o[i], c[i]) - l[i]) / hl if hl > 0 else np.nan
        vars_arr["close_position"][i] = c_pos
        vars_arr["gap"][i]            = (o[i] - c[i-1]) / c[i-1] if c[i-1] > 0 else np.nan
        vars_arr["return_1d"][i]      = (c[i] - c[i-1]) / c[i-1] if c[i-1] > 0 else np.nan
        vars_arr["body_direction"][i] = 1.0 if c[i] > o[i] else -1.0
        vars_arr["hl_ratio"][i]       = hl / c[i-1] if c[i-1] > 0 else np.nan

        if i >= ER_WINDOW:
            series       = c[i - ER_WINDOW: i + 1]
            total_change = np.sum(np.abs(np.diff(series)))
            vars_arr["er"][i] = float(np.clip(abs(series[-1] - series[0]) / total_change, 0, 1)) if total_change > 0 else 0.0

        if i >= VOL_MA_WINDOW:
            ma_vol = v[i - VOL_MA_WINDOW: i].mean()
            vars_arr["vol_rel"][i] = v[i] / ma_vol if ma_vol > 0 else np.nan
            if i >= VOL_MA_WINDOW + 1:
                prev_ma = v[i - VOL_MA_WINDOW - 1: i - 1].mean()
                prev_vol_rel = v[i-1] / prev_ma if prev_ma > 0 else np.nan
                vars_arr["vol_acceleration"][i] = vars_arr["vol_rel"][i] / prev_vol_rel if prev_vol_rel and prev_vol_rel > 0 else np.nan

        if i >= ATR_WINDOW:
            trs = [max(h[j]-l[j], abs(h[j]-c[j-1]), abs(l[j]-c[j-1])) for j in range(i - ATR_WINDOW + 1, i + 1)]
            atr = np.mean(trs)
            vars_arr["atr_norm"][i]    = atr / c[i] if c[i] > 0 else np.nan
            vars_arr["range_vs_atr"][i] = hl / atr if atr > 0 else np.nan

        if i >= RANK_WINDOW:
            recent_ranges = h[i - RANK_WINDOW: i] - l[i - RANK_WINDOW: i]
            mean_range    = recent_ranges.mean()
            vars_arr["range_expansion"][i] = hl / mean_range if mean_range > 0 else np.nan

            recent_highs = h[i - RANK_WINDOW: i + 1]
            recent_lows  = l[i - RANK_WINDOW: i + 1]
            recent_vols  = v[i - RANK_WINDOW: i + 1]
            recent_rn    = (recent_highs - recent_lows) / np.where(recent_highs > 0, recent_highs, np.nan)

            vars_arr["high_pct_rank"][i]  = float(np.sum(recent_highs[:-1] < h[i])) / RANK_WINDOW
            vars_arr["low_pct_rank"][i]   = float(np.sum(recent_lows[:-1] > l[i])) / RANK_WINDOW
            vars_arr["vol_percentile"][i] = float(np.sum(recent_vols[:-1] < v[i])) / RANK_WINDOW
            valid_rn = recent_rn[~np.isnan(recent_rn)]
            if len(valid_rn) > 1:
                vars_arr["range_percentile"][i] = float(np.sum(valid_rn[:-1] < valid_rn[-1])) / (len(valid_rn) - 1)

            y     = c[i - RANK_WINDOW: i + 1]
            slope = np.polyfit(np.arange(len(y)), y, 1)[0]
            vars_arr["trend_strength"][i] = slope / c[i] if c[i] > 0 else np.nan

            ma = c[i - RANK_WINDOW: i + 1].mean()
            vars_arr["mean_reversion"][i] = (c[i] - ma) / ma if ma > 0 else np.nan

            yv     = v[i - RANK_WINDOW: i + 1].astype(float)
            mean_v = yv.mean()
            vars_arr["vol_trend"][i] = np.polyfit(np.arange(len(yv)), yv, 1)[0] / mean_v if mean_v > 0 else np.nan

            peak   = np.max(h[i - RANK_WINDOW: i + 1])
            trough = np.min(l[i - RANK_WINDOW: i + 1])
            vars_arr["drawdown_peak"][i]    = (c[i] - peak)   / peak   if peak   > 0 else np.nan
            vars_arr["recovery_trough"][i]  = (c[i] - trough) / trough if trough > 0 else np.nan

        if i >= 1:
            streak = 0
            for j in range(i, max(i - 10, 0), -1):
                if c[j] > c[j-1]:
                    streak += 1
                else:
                    break
            vars_arr["consecutive_up"][i] = float(streak)

        w52 = 252
        if i >= w52:
            high_52 = np.max(h[i - w52: i + 1])
            low_52  = np.min(l[i - w52: i + 1])
            rng_52  = high_52 - low_52
            vars_arr["price_pos_52w"][i] = (c[i] - low_52) / rng_52 if rng_52 > 0 else np.nan

        if i >= 5:
            vars_arr["momentum_5d"][i]  = (c[i] - c[i-5])  / c[i-5]  if c[i-5]  > 0 else np.nan
        if i >= 20:
            vars_arr["momentum_20d"][i] = (c[i] - c[i-20]) / c[i-20] if c[i-20] > 0 else np.nan
        if i >= 20 and not np.isnan(vars_arr["momentum_5d"][i]) and not np.isnan(vars_arr["momentum_20d"][i]):
            vars_arr["momentum_ratio"][i] = (vars_arr["momentum_5d"][i] / vars_arr["momentum_20d"][i]
                                             if vars_arr["momentum_20d"][i] != 0 else np.nan)

        if i >= 31:
            trs_short = [max(h[j]-l[j], abs(h[j]-c[j-1]), abs(l[j]-c[j-1])) for j in range(i-6,  i+1)]
            trs_long  = [max(h[j]-l[j], abs(h[j]-c[j-1]), abs(l[j]-c[j-1])) for j in range(i-29, i+1)]
            atr_s = np.mean(trs_short)
            atr_l = np.mean(trs_long)
            vars_arr["vol_regime"][i] = atr_s / atr_l if atr_l > 0 else np.nan

        if i >= RANK_WINDOW:
            rets = np.diff(np.log(c[i - RANK_WINDOW: i + 1] + 1e-10))
            vars_arr["realized_vol"][i] = float(np.std(rets)) if len(rets) > 1 else np.nan

        if i >= 1 and v[i] > 0 and c[i-1] > 0:
            vars_arr["amihud"][i] = abs(c[i] - c[i-1]) / c[i-1] / v[i]

        if i >= VOL_MA_WINDOW:
            turnover_i  = v[i] * c[i]
            turnover_ma = np.mean(v[i - VOL_MA_WINDOW: i] * c[i - VOL_MA_WINDOW: i])
            vars_arr["turnover_rel"][i] = turnover_i / turnover_ma if turnover_ma > 0 else np.nan

    btc = btc.copy()
    for k, arr in vars_arr.items():
        btc[k] = arr
    return btc.set_index("ts")


# =============================================================================
# LOOKUP
# =============================================================================

def _lookup_candle(ind: pd.DataFrame, signal_ts: pd.Timestamp) -> pd.Series | None:
    ts = signal_ts
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    ts_lookup = ts.normalize() - pd.Timedelta(days=1)
    idx = ind.index.searchsorted(ts_lookup, side="right") - 1
    if idx < 0:
        return None
    return ind.iloc[idx]


# =============================================================================
# LOAD TRADES
# =============================================================================

def _load_trades() -> pd.DataFrame:
    pattern = os.path.join(TRADES_FOLDER, "trades_*_baseline_*.csv")
    files   = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No baseline trade files found in {TRADES_FOLDER}")

    frames = []
    for f in files:
        df = pd.read_csv(f)
        df.columns = [c.lower().strip() for c in df.columns]
        period = os.path.basename(f).split("_")[1].upper()
        df["period"] = period
        frames.append(df)

    trades = pd.concat(frames, ignore_index=True)
    trades["buy_time"] = pd.to_datetime(trades["buy_time"], errors="coerce")
    trades["buy_time"] = (trades["buy_time"].dt.tz_localize("UTC")
                          if trades["buy_time"].dt.tz is None
                          else trades["buy_time"].dt.tz_convert("UTC"))
    trades.dropna(subset=["buy_time", "profit"], inplace=True)
    trades["won"] = (trades["profit"] > 0).astype(int)
    return trades


# =============================================================================
# BUILD ANALYSIS DATAFRAME
# =============================================================================

def _build_analysis(trades: pd.DataFrame, btc_ind: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    symbol_cache: dict[str, pd.DataFrame] = {}
    rows = []

    for _, row in trades.iterrows():
        if ANALYSIS_MODE == "SYMBOL":
            symbol = str(row.get("symbol", "")).strip()
            if not symbol:
                continue
            if symbol not in symbol_cache:
                df_sym = _load_ohlcv(symbol)
                symbol_cache[symbol] = _precompute_indicators(df_sym) if not df_sym.empty else pd.DataFrame()
            ind    = symbol_cache[symbol]
            if ind.empty:
                continue
            candle = _lookup_candle(ind, row["buy_time"])
        else:
            candle = _lookup_candle(btc_ind, row["buy_time"])

        if candle is None:
            continue
        r = {"strategy": row.get("strategy", "unknown"), "period": row["period"],
             "won": row["won"], "profit": row["profit"]}
        for v in variables:
            r[v] = float(candle[v]) if v in candle.index else np.nan
        rows.append(r)

    return pd.DataFrame(rows)


# =============================================================================
# CONSISTENCY SCORE
# =============================================================================

def _consistency_score(corrs: list[float]) -> float:
    valid = [c for c in corrs if not np.isnan(c)]
    if len(valid) < 2:
        return 0.0
    signs = [np.sign(c) for c in valid]
    if all(s == signs[0] for s in signs):
        sign_factor = 1.0
    elif sum(s == signs[0] for s in signs) >= len(signs) - 1:
        sign_factor = 0.5
    else:
        sign_factor = 0.0
    return float(np.mean([abs(c) for c in valid])) * sign_factor


# =============================================================================
# POINT-BISERIAL
# =============================================================================

def _point_biserial(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    results = []
    for var in variables:
        sub = df[["won", var]].dropna()
        if len(sub) < MIN_TRADES or sub["won"].nunique() < 2:
            continue
        corr, pval = stats.pointbiserialr(sub["won"], sub[var])
        results.append({"variable": var, "corr": round(corr, 4),
                         "abs_corr": round(abs(corr), 4), "pval": round(pval, 4), "n": len(sub)})
    return pd.DataFrame(results).sort_values("abs_corr", ascending=False)


# =============================================================================
# MUTUAL INFORMATION
# =============================================================================

def _mutual_information(df: pd.DataFrame, variables: list[str]) -> pd.DataFrame:
    results = []
    for var in variables:
        sub = df[["won", var]].dropna()
        if len(sub) < MIN_TRADES or sub["won"].nunique() < 2:
            continue
        X  = sub[[var]].values
        y  = sub["won"].values
        mi = mutual_info_classif(X, y, discrete_features=False, random_state=42)[0]
        results.append({"variable": var, "mi": round(mi, 6), "n": len(sub)})
    return pd.DataFrame(results).sort_values("mi", ascending=False)


# =============================================================================
# PRINT TABLES
# =============================================================================

def _print_global_pb(df_corr: pd.DataFrame) -> None:
    print(f"\n{'='*80}")
    print(f"  GLOBAL POINT-BISERIAL — TOP {TOP_N} BY ABS_CORR")
    print(f"{'='*80}")
    print(f"  {'VARIABLE':<20} {'CORR':>8} {'ABS_CORR':>9} {'P-VALUE':>9} {'N':>7}")
    print(f"  {'─'*58}")
    for _, r in df_corr.head(TOP_N).iterrows():
        sig = "***" if r["pval"] < 0.001 else "** " if r["pval"] < 0.01 else "*  " if r["pval"] < 0.05 else "   "
        cc  = "\033[92m" if r["corr"] > 0 else "\033[91m"
        rs  = "\033[0m"
        print(f"  {r['variable']:<20} {cc}{r['corr']:>+8.4f}{rs} {r['abs_corr']:>9.4f} {r['pval']:>9.4f}{sig} {r['n']:>7}")
    print(f"  {'─'*58}")
    print(f"  Significance: * p<0.05  ** p<0.01  *** p<0.001\n")


def _print_global_mi(df_mi: pd.DataFrame) -> None:
    print(f"\n{'='*80}")
    print(f"  GLOBAL MUTUAL INFORMATION — TOP {TOP_N} BY MI SCORE")
    print(f"{'='*80}")
    print(f"  {'VARIABLE':<20} {'MI':>10} {'N':>7}")
    print(f"  {'─'*42}")
    for _, r in df_mi.head(TOP_N).iterrows():
        print(f"  {r['variable']:<20} {r['mi']:>10.6f} {r['n']:>7}")
    print(f"  {'─'*42}\n")


def _print_by_period_pb(df_analysis: pd.DataFrame, variables: list[str]) -> None:
    periods = ["OOS2", "OOS3", "OOS1"]
    rows = []
    for var in variables:
        corrs, pvals = {}, {}
        for period in periods:
            sub = df_analysis[df_analysis["period"] == period][["won", var]].dropna()
            if len(sub) < MIN_TRADES or sub["won"].nunique() < 2:
                corrs[period] = np.nan; pvals[period] = np.nan
            else:
                c, p = stats.pointbiserialr(sub["won"], sub[var])
                corrs[period] = c; pvals[period] = p
        score = _consistency_score(list(corrs.values()))
        rows.append({"variable": var, "score": score,
                     **corrs, **{f"p_{k}": v for k, v in pvals.items()}})

    df = pd.DataFrame(rows).sort_values("score", ascending=False).head(TOP_N)
    print(f"\n{'='*80}")
    print(f"  POINT-BISERIAL BY PERIOD — TOP {TOP_N} BY CONSISTENCY SCORE")
    print(f"  (score = mean_abs_corr × sign_consistency: 1.0=all same, 0.5=2/3, 0.0=mixed)")
    print(f"{'='*80}")
    print(f"  {'VARIABLE':<20} {'SCORE':>6}  {'OOS2':>10} {'OOS3':>10} {'OOS1':>10}")
    print(f"  {'─'*62}")
    for _, r in df.iterrows():
        row = f"  {r['variable']:<20} {r['score']:>6.4f}  "
        for period in periods:
            val  = r[period]
            pval = r[f"p_{period}"]
            if np.isnan(val):
                row += f"  {'—':>8}  "; continue
            sig = "*" if pval < 0.05 else " "
            cc  = "\033[92m" if val > 0 else "\033[91m"
            row += f"  {cc}{val:>+7.4f}{sig}\033[0m"
        print(row)
    print(f"  {'─'*62}\n")


def _print_by_period_mi(df_analysis: pd.DataFrame, variables: list[str]) -> None:
    periods = ["OOS2", "OOS3", "OOS1"]
    rows = []
    for var in variables:
        mis = {}
        for period in periods:
            sub = df_analysis[df_analysis["period"] == period][["won", var]].dropna()
            if len(sub) < MIN_TRADES or sub["won"].nunique() < 2:
                mis[period] = np.nan; continue
            mi = mutual_info_classif(sub[[var]].values, sub["won"].values,
                                     discrete_features=False, random_state=42)[0]
            mis[period] = round(mi, 6)
        # MI is always >= 0, so consistency = mean of valid values (no sign factor)
        valid = [v for v in mis.values() if not np.isnan(v)]
        score = float(np.mean(valid)) if valid else 0.0
        rows.append({"variable": var, "score": score, **mis})

    df = pd.DataFrame(rows).sort_values("score", ascending=False).head(TOP_N)
    print(f"\n{'='*80}")
    print(f"  MUTUAL INFORMATION BY PERIOD — TOP {TOP_N} BY MEAN MI")
    print(f"{'='*80}")
    print(f"  {'VARIABLE':<20} {'SCORE':>8}  {'OOS2':>10} {'OOS3':>10} {'OOS1':>10}")
    print(f"  {'─'*65}")
    for _, r in df.iterrows():
        row = f"  {r['variable']:<20} {r['score']:>8.6f}  "
        for period in periods:
            val = r[period]
            row += f"  {'—':>10}" if np.isnan(val) else f"  {val:>10.6f}"
        print(row)
    print(f"  {'─'*65}\n")


# =============================================================================
# MAIN
# =============================================================================

def run() -> None:
    print(f"\n{'='*80}")
    print(f"  1D CANDLE CORRELATION ANALYSIS — TOP {TOP_N}")
    print(f"  MODE={ANALYSIS_MODE} | ER_WINDOW={ER_WINDOW} | VOL_MA={VOL_MA_WINDOW} | ATR={ATR_WINDOW} | RANK_WIN={RANK_WINDOW}")
    print(f"{'='*80}")

    print("  Loading BTC data...")
    btc     = _load_btc()
    btc_ind = _precompute_indicators(btc)
    print(f"  BTC candles: {len(btc_ind)} | range: {btc_ind.index[0]} → {btc_ind.index[-1]}")

    print("  Loading trades...")
    trades = _load_trades()
    print(f"  Trades: {len(trades)} | strategies: {trades['strategy'].nunique()} | periods: {sorted(trades['period'].unique())}")

    variables = [
        "er", "range_norm", "body_size", "upper_shadow", "lower_shadow",
        "vol_rel", "return_1d", "atr_norm", "range_expansion", "close_position",
        "gap", "vol_acceleration", "high_pct_rank", "low_pct_rank",
        "trend_strength", "mean_reversion", "vol_percentile",
        "range_percentile", "consecutive_up", "body_direction",
        "hl_ratio", "range_vs_atr", "vol_trend",
        "price_pos_52w", "drawdown_peak", "recovery_trough",
        "momentum_5d", "momentum_20d", "momentum_ratio",
        "vol_regime", "realized_vol", "amihud", "turnover_rel",
    ]

    print("  Building analysis dataframe...")
    df_analysis = _build_analysis(trades, btc_ind, variables)
    print(f"  Rows with indicator data: {len(df_analysis)}\n")

    df_corr = _point_biserial(df_analysis, variables)
    df_mi   = _mutual_information(df_analysis, variables)

    _print_global_pb(df_corr)
    _print_global_mi(df_mi)
    _print_by_period_pb(df_analysis, variables)
    _print_by_period_mi(df_analysis, variables)


if __name__ == "__main__":
    run()