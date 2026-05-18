import os
import pandas as pd
import numpy as np

# =============================================================================
# CONFIG
# =============================================================================
SPLIT_MODE     = "expanding"
SPLIT_BASE     = os.path.join(os.path.dirname(__file__), "..", "..", "bitget", "data_pipeline", "data", "04_split", SPLIT_MODE)
DATA_FOLDER_IS = os.path.join(SPLIT_BASE, "IS", "crypto_full_IS")

TIMEFRAME    = "1Dutc"
BENCHMARK    = "BTCUSDT"
PRICE_COL    = "close"
VOLUME_COL   = "volume_quote"   # fallback to "volume" if not found

YEAR_FILTER  = 2026             # e.g. 2024 or 2025 — None uses full history
SORT_BY      = "volume"           # "corr" or "volume"

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


def filter_by_year(df: pd.DataFrame, year: int | None) -> pd.DataFrame:
    if year is None:
        return df
    return df[df.index.year == year]


def list_symbols(folder: str, timeframe: str) -> list[str]:
    files = [f for f in os.listdir(folder) if f.endswith(f"_{timeframe}.parquet")]
    return [f.replace(f"_{timeframe}.parquet", "") for f in sorted(files)]


def compute_correlation(returns_a: pd.Series, returns_b: pd.Series) -> float:
    aligned = pd.concat([returns_a, returns_b], axis=1).dropna()
    if aligned.shape[0] < 2:
        return np.nan
    return aligned.iloc[:, 0].corr(aligned.iloc[:, 1])


def get_avg_volume(df: pd.DataFrame) -> float:
    if VOLUME_COL in df.columns:
        return df[VOLUME_COL].mean()
    if "volume" in df.columns:
        return df["volume"].mean()
    return np.nan

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    symbols = list_symbols(DATA_FOLDER_IS, TIMEFRAME)

    if BENCHMARK not in symbols:
        raise FileNotFoundError(f"Benchmark '{BENCHMARK}_{TIMEFRAME}.parquet' not found in IS folder.")

    benchmark_df      = load_parquet(DATA_FOLDER_IS, BENCHMARK, TIMEFRAME)
    benchmark_df      = filter_by_year(benchmark_df, YEAR_FILTER)
    benchmark_returns = benchmark_df[PRICE_COL].pct_change()

    results = []
    for symbol in symbols:
        df = load_parquet(DATA_FOLDER_IS, symbol, TIMEFRAME)
        if df is None or PRICE_COL not in df.columns:
            continue

        df          = filter_by_year(df, YEAR_FILTER)
        if len(df) < 2:
            continue

        sym_returns = df[PRICE_COL].pct_change()
        corr        = compute_correlation(sym_returns, benchmark_returns)
        avg_vol     = get_avg_volume(df)

        results.append({
            "symbol":            symbol,
            "start_date":        df.index[0].date(),
            "n_rows":            len(df),
            f"corr_{BENCHMARK}": round(corr, 4),
            "avg_volume":        round(avg_vol, 0) if not np.isnan(avg_vol) else np.nan,
        })

    sort_col  = f"corr_{BENCHMARK}" if SORT_BY == "corr" else "avg_volume"
    result_df = (
        pd.DataFrame(results)
        .sort_values(sort_col, ascending=False)
        .reset_index(drop=True)
    )

    pd.set_option("display.max_rows", None)
    pd.set_option("display.float_format", "{:.4f}".format)

    year_label = str(YEAR_FILTER) if YEAR_FILTER else "full"
    print(f"\n{'='*70}")
    print(f"  Correlation & Volume  |  Timeframe: {TIMEFRAME}  |  Year: {year_label}  |  Sort: {SORT_BY}")
    print(f"{'='*70}\n")
    print(result_df.to_string(index=True))
    print(f"\nTotal symbols: {len(result_df)}")


if __name__ == "__main__":
    main()