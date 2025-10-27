from pathlib import Path

# -----------------------------
# CONFIGURATION
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
folder_a = BASE_DIR / "data" / "crypto_2021_copy"
folder_b = BASE_DIR / "data" / "crypto_2021_copy_clean"

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------
def parse_filename(filename: Path):
    """Extracts the symbol and timeframe from the filename: sym_timeframe.ext"""
    stem = filename.stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        symbol, timeframe = parts
        return symbol, timeframe
    return None, None

def get_symbols(folder: Path):
    """Returns a set of unique symbols in the folder"""
    files = list(folder.glob("*.parquet")) + list(folder.glob("*.xlsx"))
    symbols = set()
    for f in files:
        symbol, _ = parse_filename(f)
        if symbol:
            symbols.add(symbol)
    return symbols

def get_files(folder: Path):
    """Returns a set of filenames in the folder"""
    files = list(folder.glob("*.parquet")) + list(folder.glob("*.xlsx"))
    return set(f.name for f in files)

# -----------------------------
# PROCESSING
# -----------------------------
# By symbols
symbols_a = get_symbols(folder_a)
symbols_b = get_symbols(folder_b)

common_symbols = symbols_a & symbols_b
only_in_a = symbols_a - symbols_b
only_in_b = symbols_b - symbols_a

# By files
files_a = get_files(folder_a)
files_b = get_files(folder_b)

common_files = files_a & files_b
only_files_in_a = files_a - files_b
only_files_in_b = files_b - files_a

# -----------------------------
# RESULTS
# -----------------------------
# Symbols comparison
print(f"📁 Folder A: {folder_a} → {len(symbols_a)} unique symbols")
print(f"📁 Folder B: {folder_b} → {len(symbols_b)} unique symbols\n")

print(f"🔹 Common symbols ({len(common_symbols)}): {sorted(common_symbols)}")
print(f"🔹 Only in A ({len(only_in_a)}): {sorted(only_in_a)}")
print(f"🔹 Only in B ({len(only_in_b)}): {sorted(only_in_b)}\n")

# Files comparison
print(f"📁 Folder A: {folder_a} → {len(files_a)} files")
print(f"📁 Folder B: {folder_b} → {len(files_b)} files\n")

print(f"🔹 Common files ({len(common_files)}): {sorted(common_files)}")
print(f"🔹 Only in A ({len(only_files_in_a)}): {sorted(only_files_in_a)}")
print(f"🔹 Only in B ({len(only_files_in_b)}): {sorted(only_files_in_b)}")

print("\n✅ Comparison completed.")
