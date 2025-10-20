from pathlib import Path
import pandas as pd

# -----------------------------
# CONFIGURACIÓN
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
folder_a = BASE_DIR / "data" / "crypto_2023_IS"
folder_b = BASE_DIR / "data" / "crypto_2023_ISOLD"
symbol_to_compare = "ETHUSDT"
timeframe_to_compare = "4H"

# -----------------------------
# FUNCIÓN AUXILIAR
# -----------------------------
def parse_filename(filename: Path):
    stem = filename.stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None

def get_file_for_symbol_and_timeframe(folder: Path, symbol: str, timeframe: str):
    for f in folder.glob("*.xlsx"):
        file_symbol, file_timeframe = parse_filename(f)
        if file_symbol == symbol and file_timeframe == timeframe:
            return f
    return None

# -----------------------------
# CARGA DE ARCHIVOS
# -----------------------------
file_a = get_file_for_symbol_and_timeframe(folder_a, symbol_to_compare, timeframe_to_compare)
file_b = get_file_for_symbol_and_timeframe(folder_b, symbol_to_compare, timeframe_to_compare)

if not file_a or not file_b:
    print("❌ No se encontró el archivo del símbolo y timeframe especificados en alguna de las carpetas.")
    exit()

df_a = pd.read_excel(file_a)
df_b = pd.read_excel(file_b)

# -----------------------------
# COMPARACIÓN close
# -----------------------------
cols_close = ["timestamp", "close"]
df_a_close = df_a[cols_close].copy()
df_b_close = df_b[cols_close].copy()

merged_close = df_a_close.merge(df_b_close, on="timestamp", how="outer", suffixes=("_A", "_B"), indicator=True)
merged_close["close_differs"] = merged_close.apply(
    lambda row: True if row["_merge"] != "both" else row["close_A"] != row["close_B"],
    axis=1
)
differences_close = merged_close[merged_close["close_differs"]]

# -----------------------------
# COMPARACIÓN low_time y high_time
# -----------------------------
cols_low_high = ["timestamp", "low_time", "high_time"]
df_a_low_high = df_a[cols_low_high].copy()
df_b_low_high = df_b[cols_low_high].copy()

merged_low_high = df_a_low_high.merge(df_b_low_high, on="timestamp", how="outer", suffixes=("_A", "_B"), indicator=True)
merged_low_high["low_differs"] = merged_low_high.apply(
    lambda row: True if row["_merge"] != "both" else row["low_time_A"] != row["low_time_B"],
    axis=1
)
merged_low_high["high_differs"] = merged_low_high.apply(
    lambda row: True if row["_merge"] != "both" else row["high_time_A"] != row["high_time_B"],
    axis=1
)

differences_low_high = merged_low_high[(merged_low_high["low_differs"]) | (merged_low_high["high_differs"])]

# -----------------------------
# RESULTADOS
# -----------------------------
print(f"📁 Comparando '{symbol_to_compare}' con timeframe '{timeframe_to_compare}'\n")

print(f"🔹 Filas con diferencias en close: {len(differences_close)}")
print(differences_close)

print(f"\n🔹 Filas con diferencias en low_time o high_time: {len(differences_low_high)}")
print(differences_low_high)

print("\n✅ Comparación completada.")
