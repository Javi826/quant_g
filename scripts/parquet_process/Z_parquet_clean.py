import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 📁 Carpetas con los archivos
BASE_DIR = Path(__file__).resolve().parent.parent
input_folder = BASE_DIR / "data" / "crypto_2021_copy"
output_folder = BASE_DIR / "data" / "crypto_2021_copy_clean"

output_folder.mkdir(exist_ok=True)  # Crear carpeta si no existe

def parse_filename(filename):
    """Extrae símbolo del nombre del archivo (ej: BTCUSDT_1D.parquet → BTCUSDT)."""
    stem = filename.stem
    symbol = stem.rsplit("_", 1)[0]
    return symbol

def fix_zero_volumes():
    """Corrige filas con volume_base o volume_quote = 0 usando forward fill y guarda archivos corregidos."""
    # Buscar archivos Parquet y Excel
    files = list(input_folder.glob("*.parquet")) + list(input_folder.glob("*.xlsx"))
    symbols_with_zero = set()

    for file in tqdm(files, desc="Corrigiendo archivos (base y quote)"):
        try:
            if file.suffix == ".parquet":
                df = pd.read_parquet(file)
            elif file.suffix == ".xlsx":
                df = pd.read_excel(file)
            else:
                continue
        except Exception as e:
            print(f"⚠️ Error leyendo {file.name}: {e}")
            continue

        corrected = False  # bandera para saber si se modificó algo

        for col in ["volume_base", "volume_quote"]:
            if col in df.columns and (df[col] == 0).any():
                corrected = True
                symbols_with_zero.add(parse_filename(file))

                # Convertir a float y reemplazar 0 con NaN
                df[col] = df[col].astype("float64").replace(0, np.nan).ffill()

                # Si aún hay NaN (al principio), reemplazar por el primer valor válido
                if df[col].isna().any():
                    first_valid = df[col].dropna().iloc[0]
                    df[col] = df[col].fillna(first_valid)

        # Guardar siempre el archivo corregido (aunque no haya cambios, para consistencia)
        output_parquet = output_folder / file.name
        df.to_parquet(output_parquet, index=False)

        output_excel = output_folder / f"{file.stem}.xlsx"
        df.to_excel(output_excel, index=False)

    # Mensaje final
    if symbols_with_zero:
        print("\n📊 Símbolos corregidos (volume_base y/o volume_quote):")
        for sym in sorted(symbols_with_zero):
            print(f"  • {sym}")
        print(f"\n✅ Archivos corregidos guardados en '{output_folder}'")
    else:
        print("\n✅ Ningún archivo necesitaba corrección en volume_base ni volume_quote")

if __name__ == "__main__":
    fix_zero_volumes()
