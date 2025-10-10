from pathlib import Path
import os
import pandas as pd
import sys

# -----------------------------
# CONFIGURATION
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # sube un nivel desde la carpeta del script

SOURCE_FOLDER = BASE_DIR / "crypto_2023_highlow"
END_DATE      = "2025-06-01"
START_DATE    = "2025-06-02"

TARGET_FOLDER_UPTO    = BASE_DIR / f"{SOURCE_FOLDER.name}_UPTO"
TARGET_FOLDER_ONWARDS = BASE_DIR / f"{SOURCE_FOLDER.name}_ONWARDS"


os.makedirs(TARGET_FOLDER_UPTO, exist_ok=True)
os.makedirs(TARGET_FOLDER_ONWARDS, exist_ok=True)

# Convert dates to datetime
end_datetime = pd.to_datetime(END_DATE)
start_datetime = pd.to_datetime(START_DATE)

# List all parquet and Excel files in the source folder
files = list(Path(SOURCE_FOLDER).glob("*.parquet")) + list(Path(SOURCE_FOLDER).glob("*.xlsx"))

for file_path in files:
    try:
        # Read the file depending on its extension
        # Read the file depending on its extension
        if file_path.suffix == ".parquet":
            df = pd.read_parquet(file_path)
        elif file_path.suffix == ".xlsx":
            df = pd.read_excel(file_path, index_col=0)
        else:
            continue  # Skip other file types

        # -----------------------------
        # Asegurar índice datetime
        # -----------------------------
        if not pd.api.types.is_datetime64_any_dtype(df.index):
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp")
            else:
                df.index = pd.to_datetime(df.index, errors="coerce")


        # -----------------------------
        # Cut upto END_DATE (_UPTO)
        # -----------------------------
        df_upto = df[df.index <= end_datetime]
        target_upto = Path(TARGET_FOLDER_UPTO) / file_path.name
        if file_path.suffix == ".parquet":
            df_upto.to_parquet(target_upto, index=True)
        elif file_path.suffix == ".xlsx":
            df_upto.to_excel(target_upto, index=True)
        print(f"✅ Saved {file_path.name} upto {END_DATE} ({len(df_upto)} records) in {TARGET_FOLDER_UPTO}")

        # -----------------------------
        # Cut onwards START_DATE (_ONWARDS)
        # -----------------------------
        df_onwards = df[df.index >= start_datetime]
        target_onwards = Path(TARGET_FOLDER_ONWARDS) / file_path.name
        if file_path.suffix == ".parquet":
            df_onwards.to_parquet(target_onwards, index=True)
        elif file_path.suffix == ".xlsx":
            df_onwards.to_excel(target_onwards, index=True)
        print(f"✅ Saved {file_path.name} onwards {START_DATE} ({len(df_onwards)} records) in {TARGET_FOLDER_ONWARDS}")

    except Exception as e:
        print(f"❌ Error processing {file_path.name}: {e}")

print("\n📂 All files processed for both '_UPTO' and '_ONWARDS' folders")



