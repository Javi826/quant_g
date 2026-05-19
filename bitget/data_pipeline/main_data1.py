import os
import pandas as pd

DIR_A = "/home/javi/projects/quant/quant_b/bitget/data_pipeline/data/03_highlow"
DIR_B = "/home/javi/projects/quant/quant_b/bitget/data_pipeline/data/03_highlow_v2"


def compare_highlow(symbol: str, tf: str) -> None:
    path_a = os.path.join(DIR_A, f"{symbol}_{tf}.parquet")
    path_b = os.path.join(DIR_B, f"{symbol}_{tf}.parquet")

    if not os.path.exists(path_a):
        print(f"  ❌ [{symbol}][{tf}] Missing in A: {path_a}")
        return
    if not os.path.exists(path_b):
        print(f"  ❌ [{symbol}][{tf}] Missing in B: {path_b}")
        return

    df_a = pd.read_parquet(path_a)
    df_b = pd.read_parquet(path_b)

    if "timestamp" in df_a.columns:
        df_a = df_a.set_index("timestamp")
    if "timestamp" in df_b.columns:
        df_b = df_b.set_index("timestamp")

    df_a.index = pd.to_datetime(df_a.index)
    df_b.index = pd.to_datetime(df_b.index)

    print(f"\n── {symbol} [{tf}] ──")
    print(f"  Rows A: {len(df_a)} | Rows B: {len(df_b)}")

    if len(df_a) != len(df_b):
        print(f"  ⚠ Row count mismatch!")

    common = df_a.index.intersection(df_b.index)
    print(f"  Common rows: {len(common)}")

    all_ok = True
    for col in ["high_time", "low_time"]:
        if col not in df_a.columns or col not in df_b.columns:
            print(f"  ⚠ Column '{col}' missing in one of the files")
            continue
        a = pd.to_datetime(df_a.loc[common, col])
        b = pd.to_datetime(df_b.loc[common, col])
        diff = (a != b) & ~(a.isna() & b.isna())
        n = diff.sum()
        if n > 0:
            all_ok = False
            print(f"  ❌ {col}: {n} differences")
            print(df_a.loc[common][diff][[col]].rename(columns={col: f"{col}_A"})
                  .join(df_b.loc[common][diff][[col]].rename(columns={col: f"{col}_B"}))
                  .head(5).to_string())
        else:
            print(f"  ✅ {col}: identical")

    if all_ok:
        print(f"  ✅ PERFECT MATCH")


def print_sample(symbol: str, tf: str, n: int = 5) -> None:
    path_a = os.path.join(DIR_A, f"{symbol}_{tf}.parquet")
    path_b = os.path.join(DIR_B, f"{symbol}_{tf}.parquet")

    for path, label in [(path_a, "A"), (path_b, "B")]:
        if not os.path.exists(path):
            print(f"  ❌ [{symbol}][{tf}] Missing in {label}")
            continue
        df = pd.read_parquet(path)
        if "timestamp" in df.columns:
            df = df.set_index("timestamp")
        df.index = pd.to_datetime(df.index)
        print(f"\n── {symbol} [{tf}] — {label} ──")
        print(df[["high_time", "low_time"]].head(n).to_string())


if __name__ == "__main__":
    symbols = ["BTCUSDT", "AAVEUSDT"]
    tfs     = ["1Dutc", "6Hutc", "4H", "1H"]

    for sym in symbols:
        for tf in tfs:
            compare_highlow(sym, tf)

    print("\n" + "=" * 60)
    print("  SAMPLE DATA COMPARISON")
    print("=" * 60)
    for sym in symbols:
        for tf in tfs:
            print_sample(sym, tf)