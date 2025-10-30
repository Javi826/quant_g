import pandas as pd
from pathlib import Path
from tqdm import tqdm

# 📁 Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
input_folder = BASE_DIR / "data" / "darwinex"
output_folder = BASE_DIR / "data" / "darwinex_parquet"

# Create the output folder if it doesn't exist
output_folder.mkdir(exist_ok=True)

# Find all CSV files in the input folder
csv_files = list(input_folder.glob("*.csv"))

if not csv_files:
    print("⚠️ No CSV files found in:", input_folder)
else:
    print(f"📄 Files found: {len(csv_files)}")

# Convert each CSV file to Parquet
for csv_file in tqdm(csv_files, desc="Converting CSV to Parquet"):
    try:
        # Read CSV
        df = pd.read_csv(csv_file)

        # Rename columns if they exist
        rename_map = {}
        if "time" in df.columns:
            rename_map["time"] = "timestamp"
        if "volume" in df.columns:
            rename_map["volume"] = "volume_quote"
        if rename_map:
            df.rename(columns=rename_map, inplace=True)

        # Ensure timestamp column exists and set it as index
        if "timestamp" not in df.columns:
            raise ValueError(f"'timestamp' column not found in {csv_file.name}")
        
        # Try to convert to datetime if possible
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        except Exception as e:
            print(f"⚠️ Warning: could not convert timestamp in {csv_file.name}: {e}")

        df.set_index("timestamp", inplace=True)

        # Detect and format timeframe suffix (e.g., 1h → 1H)
        stem = csv_file.stem

        # ✅ Remove prefix "D_" if present
        if stem.startswith("D_"):
            stem = stem[2:]

        parts = stem.split("_")
        if parts[-1].lower().endswith(("h", "m", "d", "w")):  # e.g. 1h, 15m, 4h, 1d
            parts[-1] = parts[-1].upper()
        parquet_name = "_".join(parts) + ".parquet"

        # Save as Parquet (index=True so timestamp stays as index)
        parquet_file = output_folder / parquet_name
        df.to_parquet(parquet_file, index=True)

    except Exception as e:
        print(f"❌ Error processing {csv_file.name}: {e}")

print("✅ Conversion completed. Files saved to:", output_folder)
