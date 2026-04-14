#STEP 2 datacleaning+integrity
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 📁 Folders with the files
BASE_DIR      = Path(__file__).resolve().parent.parent
input_folder  = BASE_DIR / "data" / "crypto_2026_short"
output_folder = BASE_DIR / "data" / "crypto_2026_short_clean"

output_folder.mkdir(exist_ok=True)  # Create output folder if it doesn't exist

def parse_filename(filename):
    """Extracts the symbol from the filename (e.g., BTCUSDT_1D.parquet → BTCUSDT)."""
    stem = filename.stem
    symbol = stem.rsplit("_", 1)[0]
    return symbol

def fix_zero_volumes():
    """Fixes rows with volume_base or volume_quote = 0 using forward fill and saves corrected files."""
    # Find Parquet and Excel files
    files = list(input_folder.glob("*.parquet")) + list(input_folder.glob("*.xlsx"))
    symbols_with_zero = set()

    for file in tqdm(files, desc="Fixing files (base and quote)"):
        try:
            if file.suffix == ".parquet":
                df = pd.read_parquet(file)
            elif file.suffix == ".xlsx":
                df = pd.read_excel(file)
            else:
                continue
        except Exception as e:
            print(f"⚠️ Error reading {file.name}: {e}")
            continue

        corrected = False  # flag to check if any modification was made

        for col in ["volume_base", "volume_quote"]:
            if col in df.columns and (df[col] == 0).any():
                corrected = True
                symbols_with_zero.add(parse_filename(file))

                # Convert to float and replace 0 with NaN
                df[col] = df[col].astype("float64").replace(0, np.nan).ffill()

                # If there are still NaN values (at the beginning), fill with the first valid value
                if df[col].isna().any():
                    first_valid = df[col].dropna().iloc[0]
                    df[col] = df[col].fillna(first_valid)

        # Always save the corrected file (even if no changes, for consistency)
        output_parquet = output_folder / file.name
        df.to_parquet(output_parquet, index=False)

        output_excel = output_folder / f"{file.stem}.xlsx"
        df.to_excel(output_excel, index=False)

    # Final message
    if symbols_with_zero:
        print("\n📊 Symbols corrected (volume_base and/or volume_quote):")
        for sym in sorted(symbols_with_zero):
            print(f"  • {sym}")
        print(f"\n✅ Corrected files saved in '{output_folder}'")
    else:
        print("\n✅ No file required correction in volume_base or volume_quote")

if __name__ == "__main__":
    fix_zero_volumes()
