import os
import pandas as pd
import numpy as np

# =============================================================================
# CONFIG
# =============================================================================
DATA_FOLDER_IS = os.path.expanduser(
    "~/projects/quant/quant_b/bitget/data_pipeline/data/04_split/expanding/IS/rwa_2025-01_2026-03_IS"
)

TIMEFRAME      = "1Dutc"
BENCHMARK      = "QQQUSDT"
PRICE_COL      = "close"

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


def get_start_date(df: pd.DataFrame) -> pd.Timestamp:
    return df.index[0]


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
    all_files = os.listdir(DATA_FOLDER_IS)
    print(f"Total files in folder: {len(all_files)}")
    print(f"Files NOT matching timeframe '{TIMEFRAME}':")
    for f in sorted(all_files):
        if not f.endswith(f"_{TIMEFRAME}.parquet"):
            print(f"  {f}")

    symbols = list_symbols(DATA_FOLDER_IS, TIMEFRAME)
    print(f"\nSymbols matched: {len(symbols)}\n")

    if BENCHMARK not in symbols:
        raise FileNotFoundError(f"Benchmark '{BENCHMARK}_{TIMEFRAME}.parquet' not found in IS folder.")

    benchmark_df      = load_parquet(DATA_FOLDER_IS, BENCHMARK, TIMEFRAME)
    benchmark_returns = benchmark_df[PRICE_COL].pct_change()

    results = []

    for symbol in symbols:
        df = load_parquet(DATA_FOLDER_IS, symbol, TIMEFRAME)
        if df is None or PRICE_COL not in df.columns:
            continue

        start_date  = get_start_date(df)
        sym_returns = df[PRICE_COL].pct_change()
        corr        = compute_correlation(sym_returns, benchmark_returns)

        results.append({
            "symbol":     symbol,
            "start_date": start_date.date(),
            "n_rows":     len(df),
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


if __name__ == "__main__":
    main()