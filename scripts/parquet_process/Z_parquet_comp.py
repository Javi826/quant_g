from pathlib import Path

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
folder_a = BASE_DIR / "data" / "crypto_2023_IS"
folder_b = BASE_DIR / "data" / "crypto_2023_ISOLD"

# -----------------------------
# FUNCIÓN AUXILIAR
# -----------------------------
def parse_filename(filename: Path):
    """Extrae el símbolo y timeframe del nombre del archivo: sym_timeframe.ext"""
    stem = filename.stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        symbol, timeframe = parts
        return symbol, timeframe
    return None, None

def get_symbols(folder: Path):
    """Devuelve un set con los símbolos únicos en la carpeta"""
    files = list(folder.glob("*.parquet")) + list(folder.glob("*.xlsx"))
    symbols = set()
    for f in files:
        symbol, _ = parse_filename(f)
        if symbol:
            symbols.add(symbol)
    return symbols

# -----------------------------
# PROCESAMIENTO
# -----------------------------
symbols_a = get_symbols(folder_a)
symbols_b = get_symbols(folder_b)

# Símbolos comunes y diferencias
common_symbols = symbols_a & symbols_b
only_in_a = symbols_a - symbols_b
only_in_b = symbols_b - symbols_a

# -----------------------------
# RESULTADOS
# -----------------------------
print(f"📁 Carpeta A: {folder_a} → {len(symbols_a)} símbolos únicos")
print(f"📁 Carpeta B: {folder_b} → {len(symbols_b)} símbolos únicos\n")

print(f"🔹 Símbolos comunes ({len(common_symbols)}): {sorted(common_symbols)}")
print(f"🔹 Solo en A ({len(only_in_a)}): {sorted(only_in_a)}")
print(f"🔹 Solo en B ({len(only_in_b)}): {sorted(only_in_b)}")

print("\n✅ Comparación completada.")
