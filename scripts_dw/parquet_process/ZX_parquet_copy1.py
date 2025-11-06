from pathlib import Path
import shutil

# -----------------------------
# Configuración
# -----------------------------
BASE_DIR      = Path(__file__).resolve().parent.parent
input_folder  = BASE_DIR / "data" / "crypto_utc"
output_folder = BASE_DIR / "data" / "crypto_2021_utc"

# Crear carpeta de salida si no existe
output_folder.mkdir(exist_ok=True, parents=True)

# -----------------------------
# Listar y copiar ficheros
# -----------------------------
all_files = list(input_folder.glob("*"))

for f in all_files:
    if not f.is_file():
        continue

    destino = output_folder / f.name
    shutil.copy2(f, destino)  # 👈 Copia el archivo sin eliminar el original
    print(f"💾 Copiado: {f.name}")

print("✅ Todos los archivos han sido copiados correctamente.")
