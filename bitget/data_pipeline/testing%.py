import pandas as pd

path = "/home/javi/projects/quant/quant_b/bitget/data_pipeline/data/04_split/expanding/IS/crypto_full_IS/BTCUSDT_15m.parquet"

df = pd.read_parquet(path)
if "timestamp" in df.columns:
    df = df.set_index("timestamp")

df["high_time"] = pd.to_datetime(df["high_time"])
df["low_time"]  = pd.to_datetime(df["low_time"])

total       = len(df)
coincide    = (df["high_time"] == df["low_time"]).sum()
pct         = coincide / total * 100

print(f"Total bars  : {total}")
print(f"Coincide    : {coincide} ({pct:.1f}%)")
print(f"\nEjemplos donde coinciden:")
print(df[df["high_time"] == df["low_time"]][["high_time", "low_time"]].head(10))