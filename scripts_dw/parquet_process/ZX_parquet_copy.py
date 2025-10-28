from pathlib import Path
import shutil
import pandas as pd

# -----------------------------
# Configuración
# -----------------------------
BASE_DIR      = Path(__file__).resolve().parent.parent
input_folder  = BASE_DIR / "data" / "crypto_2021"
output_folder = BASE_DIR / "data" / "crypto_2021_clean"

# Crear carpeta de salida si no existe
output_folder.mkdir(exist_ok=True, parents=True)

# -----------------------------
# Listar todos los ficheros
# -----------------------------
all_files = list(input_folder.glob("*"))

for f in all_files:
    if not f.is_file():
        continue
    
    # Filtrar los que queremos eliminar
    if f.name.endswith("_15m.parquet") or f.name.endswith("_15m.xlsx"):
        print(f"❌ Ignorado: {f.name}")
        continue
    
    # Copiar todos los archivos tal cual (sin normalización)
    shutil.copy2(f, output_folder / f.name)
    print(f"💾 Copiado: {f.name}")

print("✅ Proceso completado.")