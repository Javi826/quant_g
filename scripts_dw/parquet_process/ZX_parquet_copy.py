from pathlib import Path
import shutil
import pandas as pd

# -----------------------------
# Configuración
# -----------------------------
BASE_DIR      = Path(__file__).resolve().parent.parent
input_folder  = BASE_DIR / "data" / "crypto_2021_clean"
output_folder = BASE_DIR / "data" / "crypto_2021_utc"

# Crear carpeta de salida si no existe
output_folder.mkdir(exist_ok=True, parents=True)

# -----------------------------
# Timeframes a eliminar (PARAMETRO)
# Ejemplo: ["_15m", "_6H", "_1D"]
# -----------------------------
TIMEFRAMES_EXCLUIR = ["_12H", "_6H","_1D"]

# Extensiones que pueden existir
EXTENSIONES = [".parquet", ".xlsx"]

# -----------------------------
# Listar y procesar ficheros
# -----------------------------
all_files = list(input_folder.glob("*"))

for f in all_files:
    if not f.is_file():
        continue

    # Verificar si el archivo coincide con alguno de los timeframes a excluir
    ignorar = any(
        f.name.endswith(f"{tf}{ext}")
        for tf in TIMEFRAMES_EXCLUIR
        for ext in EXTENSIONES
    )

    if ignorar:
        print(f"❌ Ignorado: {f.name}")
        continue

    # Copiar archivo
    shutil.copy2(f, output_folder / f.name)
    print(f"💾 Copiado: {f.name}")

print("✅ Proceso completado.")
