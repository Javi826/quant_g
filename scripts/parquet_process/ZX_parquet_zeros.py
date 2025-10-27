import pandas as pd
from pathlib import Path
from tqdm import tqdm

# 📁 Carpeta con los archivos Parquet
BASE_DIR = Path(__file__).resolve().parent.parent
input_folder = BASE_DIR / "data" / "crypto_2021_copy_clean"

def parse_filename(filename):
    """Extrae símbolo del nombre del archivo (ej: BTCUSDT_1D.parquet → BTCUSDT)."""
    stem = filename.stem
    symbol = stem.rsplit("_", 1)[0]
    return symbol

def identify_zero_volumes():
    """Identifica archivos .parquet que contienen volume_base o volume_quote = 0."""
    files = list(input_folder.glob("*.parquet"))
    symbols_with_zero = {}

    for file in tqdm(files, desc="Analizando archivos Parquet"):
        try:
            df = pd.read_parquet(file)
        except Exception as e:
            print(f"⚠️ Error leyendo {file.name}: {e}")
            continue
        
        zero_info = {}
        # 🔍 Revisa ambas columnas
        for col in ["volume_base", "volume_quote"]:
            if col in df.columns:
                zeros = int((df[col] == 0).sum())
                if zeros > 0:
                    zero_info[col] = zeros

        # Si encontró ceros en alguna de las dos columnas
        if zero_info:
            symbols_with_zero[parse_filename(file)] = {
                "archivo": file.name,
                **zero_info
            }

    # 📊 Resultados
    if symbols_with_zero:
        print("\n📊 Archivos con ceros en volume_base o volume_quote:\n")
        for sym, info in sorted(symbols_with_zero.items()):
            detalles = ", ".join([f"{col}: {n} filas con 0" for col, n in info.items() if col != "archivo"])
            print(f"  • {sym} → {info['archivo']} ({detalles})")
        print(f"\n✅ Total: {len(symbols_with_zero)} archivos con ceros en volumen")
    else:
        print("\n✅ Ningún archivo tiene ceros en volume_base ni volume_quote")

if __name__ == "__main__":
    identify_zero_volumes()
