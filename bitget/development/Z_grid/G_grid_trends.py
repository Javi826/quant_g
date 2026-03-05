import pandas as pd
import os

DATA_FOLDER = "../data/crypto_OOS_2025"
files = [f for f in os.listdir(DATA_FOLDER) if f.endswith('_4H.parquet')]

# Revisar primer archivo
first_file = os.path.join(DATA_FOLDER, files[0])
df = pd.read_parquet(first_file)

print(f"File: {files[0]}")
print(f"Columns: {df.columns.tolist()}")
print(f"Shape: {df.shape}")
print(f"\nFirst timestamp:")
print(df['timestamp'].iloc[0] if 'timestamp' in df.columns else df['ts'].iloc[0])
print(f"\nFirst 3 rows:")
print(df.head(3))