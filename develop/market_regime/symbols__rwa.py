import os
import pandas as pd
import numpy as np

# =============================================================================
# CONFIG
# =============================================================================
DATA_FOLDER_IS = os.path.expanduser(
    "~/projects/quant/quant_b/bitget/data_pipeline/data/04_split/expanding/IS/rwa_2025-01_2026-03_IS"
)

TIMEFRAME  = "1Dutc"
BENCHMARK  = "QQQUSDT"
PRICE_COL  = "close"

SELECTED_SYMBOLS = [
    "NVDAUSDT",
    "PLTRUSDT",
    "HOODUSDT",
    "ASMLUSDT",
    "GOOGLUSDT",
    "AMZNUSDT",
    "TSLAUSDT",
    "COINUSDT",
    "MRVLUSDT",
    "METAUSDT",
    "MSFTUSDT",
]

INTRADAY_TIMEFRAMES = {"1H": 24, "30m": 48, "15m": 96}

# =============================================================================
# HELPERS
# =============================================================================

def load_parquet(folder: str, symbol: str, timeframe: str) -> pd.DataFrame | None:
    path = os.path.join(folder, f"{symbol}_{timeframe}.parquet")
    if not os.path.exists(path):
        return None
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    return df


def compute_correlation(returns_a: pd.Series, returns_b: pd.Series) -> float:
    aligned = pd.concat([returns_a, returns_b], axis=1).dropna()
    if aligned.shape[0] < 2:
        return np.nan
    return aligned.iloc[:, 0].corr(aligned.iloc[:, 1])


def list_symbols(folder: str, timeframe: str) -> list[str]:
    files = [f for f in os.listdir(folder) if f.endswith(f"_{timeframe}.parquet")]
    return [f.replace(f"_{timeframe}.parquet", "") for f in sorted(files)]

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    symbols = list_symbols(DATA_FOLDER_IS, TIMEFRAME)

    if BENCHMARK not in symbols:
        raise FileNotFoundError(f"Benchmark '{BENCHMARK}_{TIMEFRAME}.parquet' not found in IS folder.")

    benchmark_df      = load_parquet(DATA_FOLDER_IS, BENCHMARK, TIMEFRAME)
    benchmark_returns = benchmark_df[PRICE_COL].pct_change()

    # --- Full universe analysis ---
    results = []
    for symbol in symbols:
        df = load_parquet(DATA_FOLDER_IS, symbol, TIMEFRAME)
        if df is None or PRICE_COL not in df.columns:
            continue

        sym_returns = df[PRICE_COL].pct_change()
        corr        = compute_correlation(sym_returns, benchmark_returns)

        results.append({
            "symbol":            symbol,
            "start_date":        df.index[0].date(),
            "n_rows":            len(df),
            f"corr_{BENCHMARK}": round(corr, 4),
        })

    result_df = (
        pd.DataFrame(results)
        .sort_values(f"corr_{BENCHMARK}", ascending=False)
        .reset_index(drop=True)
    )

    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", "{:.4f}".format)

    print(f"\n{'='*60}")
    print(f"  Symbol analysis  |  Timeframe: {TIMEFRAME}  |  Split: IS")
    print(f"{'='*60}\n")
    print(result_df.to_string(index=True))
    print(f"\nTotal symbols: {len(result_df)}")

    # --- Common start date for selected symbols ---
    print(f"\n{'='*60}")
    print(f"  Selected symbols — common start date")
    print(f"{'='*60}\n")

    selected_start_dates = {}
    for symbol in SELECTED_SYMBOLS:
        df = load_parquet(DATA_FOLDER_IS, symbol, TIMEFRAME)
        if df is None:
            print(f"  WARNING: {symbol} not found in IS folder")
            continue
        selected_start_dates[symbol] = df.index[0]
        print(f"  {symbol:<14} start: {df.index[0].date()}")

    if not selected_start_dates:
        return

    common_start  = max(selected_start_dates.values())
    end_date      = benchmark_df.index[-1]
    trading_days  = len(benchmark_df[benchmark_df.index >= common_start])
    calendar_days = (end_date.normalize() - common_start.normalize()).days + 1
    print(f"\n  Common start date (max of all): {common_start.date()}")

    # --- Candle count from common start date ---
    print(f"\n{'='*60}")
    print(f"  Candle count from {common_start.date()} to {end_date.date()}")
    print(f"  ({trading_days} trading days | {calendar_days} calendar days in IS)")
    print(f"{'='*60}\n")

    print(f"  {'Timeframe':<8} {'Theoretical':>14} {'Actual (min / max)':>22}")
    print(f"  {'-'*46}")

    for tf, candles_per_day in INTRADAY_TIMEFRAMES.items():
        theoretical = calendar_days * candles_per_day

        actual_counts = []
        for symbol in SELECTED_SYMBOLS:
            df = load_parquet(DATA_FOLDER_IS, symbol, tf)
            if df is None:
                continue
            count = len(df[df.index >= common_start])
            actual_counts.append(count)

        if actual_counts:
            actual_str = f"{min(actual_counts):,} / {max(actual_counts):,}"
        else:
            actual_str = "N/A (no files)"

        print(f"  {tf:<8} {theoretical:>14,} {actual_str:>22}")


if __name__ == "__main__":
    main()