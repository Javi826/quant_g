import pandas as pd
from pathlib import Path

# ==============================
# CONFIGURACIÓN
# ==============================
FOLDER_NAME = "crypto_2023_ISOLD"
SOURCE_FOLDER = Path("/home/javi/projects/quant/quant_g/data") / FOLDER_NAME

print(f"SOURCE_FOLDER: {SOURCE_FOLDER}")

# Buscar todos los Excel en la carpeta
all_files = list(SOURCE_FOLDER.glob("*.xlsx"))
print(f"🔹 Número total de ficheros encontrados: {len(all_files)}")



results = []

total_rows_global = 0
low_before_high_global = 0
high_before_low_global = 0
equal_global = 0
processed_files = 0

# ==============================
# PROCESAR CADA FICHERO
# ==============================
for file_path in all_files:
    try:
        # Cargar Excel (acepta decimales con coma)
        df = pd.read_excel(file_path, decimal=",")
    except Exception as e:
        print(f"⚠️ Error leyendo {file_path.name}: {e}")
        continue

    # Verificar columnas necesarias
    required_cols = {"timestamp", "low_time", "high_time"}
    if not required_cols.issubset(df.columns):
        print(f"⚠️ {file_path.name} no contiene las columnas necesarias {required_cols}")
        continue

    # Convertir a datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["low_time"] = pd.to_datetime(df["low_time"], errors="coerce")
    df["high_time"] = pd.to_datetime(df["high_time"], errors="coerce")

    # Eliminar filas sin datos válidos
    df = df.dropna(subset=["low_time", "high_time"])
    if df.empty:
        print(f"⚠️ {file_path.name} no tiene datos válidos tras limpieza")
        continue

    total_rows = len(df)
    low_before_high = (df["low_time"] < df["high_time"]).sum()
    high_before_low = (df["low_time"] > df["high_time"]).sum()
    equal = (df["low_time"] == df["high_time"]).sum()

    results.append({
        "symbol_file": file_path.name,
        "total_rows": total_rows,
        "low_before_high": low_before_high,
        "high_before_low": high_before_low,
        "equal": equal,
        "low_before_high_%": low_before_high / total_rows * 100,
        "high_before_low_%": high_before_low / total_rows * 100,
        "equal_%": equal / total_rows * 100,
    })

    # Acumular totales globales
    total_rows_global += total_rows
    low_before_high_global += low_before_high
    high_before_low_global += high_before_low
    equal_global += equal
    processed_files += 1

# ==============================
# RESUMEN POR FICHERO
# ==============================
if results:
    df_summary = pd.DataFrame(results)
    df_summary = df_summary.sort_values("high_before_low_%", ascending=False)

    print("\n--- Resumen por fichero ---")
    print(df_summary.head(20))

    # Guardar CSV
    output_path = SOURCE_FOLDER / "summary_high_low_times.csv"
    df_summary.to_csv(output_path, index=False)
    print(f"\n💾 Resumen guardado en: {output_path}")
else:
    print("⚠️ No hay datos válidos para mostrar resumen por fichero")
    df_summary = pd.DataFrame()

# ==============================
# RESUMEN GLOBAL
# ==============================
if total_rows_global > 0:
    print("\n--- Total global ---")
    print(f"Ficheros procesados correctamente: {processed_files}")
    print(f"Filas totales: {total_rows_global:,}")
    print(f"🔺 low_time < high_time  (Low antes que High → SL antes que TP): {low_before_high_global:,}  → {low_before_high_global / total_rows_global * 100:.2f}%")
    print(f"🔹 low_time > high_time  (High antes que Low → TP antes que SL): {high_before_low_global:,}  → {high_before_low_global / total_rows_global * 100:.2f}%")
    print(f"⚖️ low_time = high_time  (Empate): {equal_global:,}  → {equal_global / total_rows_global * 100:.2f}%")
else:
    print("⚠️ No hay datos válidos para resumen global")
