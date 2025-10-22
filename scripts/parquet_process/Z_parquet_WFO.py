import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta

# -----------------------------
# PARÁMETROS DE CONFIGURACIÓN
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # sube un nivel
input_folder = BASE_DIR / "data" / "crypto_2021_highlow"
output_folder_is = BASE_DIR / "data" / "crypto_2021_TEST1"
output_folder_oos = BASE_DIR / "data" / "crypto_2021_TEST2"

# Rango de fechas para IN-SAMPLE (IS)
IS_START = "2025-01-01"
IS_END = "2025-10-01"

# OOS automático: desde IS_END hasta el final del archivo
# OOS_START = IS_END
# OOS_END = fin del archivo

output_folder_is.mkdir(exist_ok=True, parents=True)
output_folder_oos.mkdir(exist_ok=True, parents=True)


def read_file(filepath):
    """Lee archivo parquet o xlsx y devuelve DataFrame."""
    try:
        if filepath.suffix == ".parquet":
            df = pd.read_parquet(filepath)
        elif filepath.suffix == ".xlsx":
            df = pd.read_excel(filepath)
        else:
            return None
        return df
    except Exception as e:
        print(f"❌ Error leyendo {filepath.name}: {e}")
        return None


def write_file(df, filepath):
    """Escribe DataFrame a parquet o xlsx con timestamp como índice."""
    try:
        # Asegurarse de que hay columna 'timestamp'
        if 'timestamp' in df.columns:
            # Convertir a datetime si no lo es
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Poner timestamp como índice
            df.set_index('timestamp', inplace=True)
        
        # Guardar según extensión
        if filepath.suffix == ".parquet":
            df.to_parquet(filepath, index=True)  # guardar índice
        elif filepath.suffix == ".xlsx":
            df.to_excel(filepath, index=True)    # guardar índice
    except Exception as e:
        print(f"❌ Error escribiendo {filepath.name}: {e}")




def split_is_oos(df, is_start, is_end):
    """
    Divide DataFrame en dos:
    - IS: desde is_start hasta is_end (exclusivo)
    - OOS: desde is_end hasta el final
    """
    # Identificar columna de timestamp
    if 'timestamp' in df.columns:
        time_col = 'timestamp'
    elif df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        time_col = df.columns[0]
    else:
        # Asumir que la primera columna es timestamp
        time_col = df.columns[0]
    
    # Convertir a datetime si no lo es
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Convertir fechas de filtro
    start = pd.to_datetime(is_start)
    end = pd.to_datetime(is_end)
    
    # Dividir en IS y OOS
    mask_is = (df[time_col] >= start) & (df[time_col] < end)
    mask_oos = (df[time_col] >= end)
    
    df_is = df[mask_is].copy()
    df_oos = df[mask_oos].copy()
    
    return df_is, df_oos


def process_files():
    """Procesa todos los archivos en la carpeta input."""
    
    # Obtener todos los archivos
    files = list(input_folder.glob("*.parquet")) + list(input_folder.glob("*.xlsx"))
    
    if len(files) == 0:
        print(f"❌ No se encontraron archivos en {input_folder}")
        return
    
    # Calcular OOS_START automáticamente
    oos_start = pd.to_datetime(IS_END)
    
    print(f"\n{'='*60}")
    print(f"DIVISIÓN IS / OOS")
    print(f"{'='*60}")
    print(f"\n📅 IN-SAMPLE (IS):")
    print(f"   Inicio: {IS_START}")
    print(f"   Fin:    {IS_END}")
    print(f"\n📅 OUT-OF-SAMPLE (OOS):")
    print(f"   Inicio: {IS_END}")
    print(f"   Fin:    [fin del archivo]")
    print(f"\n📁 Archivos a procesar: {len(files)}\n")
    
    processed = 0
    skipped_is = 0
    skipped_oos = 0
    errors = 0
    
    for file in tqdm(files, desc="Procesando archivos"):
        # Leer archivo
        df = read_file(file)
        
        if df is None:
            errors += 1
            continue
        
        # Dividir en IS y OOS
        try:
            df_is, df_oos = split_is_oos(df, IS_START, IS_END)
            
            has_is = len(df_is) > 0
            has_oos = len(df_oos) > 0
            
            # Guardar IS
            if has_is:
                output_file_is = output_folder_is / file.name
                write_file(df_is, output_file_is)
            else:
                skipped_is += 1
            
            # Guardar OOS
            if has_oos:
                output_file_oos = output_folder_oos / file.name
                write_file(df_oos, output_file_oos)
            else:
                skipped_oos += 1
            
            if has_is or has_oos:
                processed += 1
                status_is = f"IS: {len(df_is)}" if has_is else "IS: 0 (omitido)"
                status_oos = f"OOS: {len(df_oos)}" if has_oos else "OOS: 0 (omitido)"
                tqdm.write(f"✅ {file.name}: {status_is}, {status_oos}")
            else:
                tqdm.write(f"⚠️  {file.name}: Sin datos en ningún rango")
            
        except Exception as e:
            errors += 1
            tqdm.write(f"❌ Error procesando {file.name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"RESUMEN")
    print(f"{'='*60}")
    print(f"✅ Archivos procesados: {processed}")
    print(f"⚠️  Archivos sin datos IS: {skipped_is}")
    print(f"⚠️  Archivos sin datos OOS: {skipped_oos}")
    print(f"❌ Errores: {errors}")
    print(f"📁 Total: {len(files)}")
    print(f"\n📂 Carpetas de salida:")
    print(f"   IS:  {output_folder_is}")
    print(f"   OOS: {output_folder_oos}")
    print(f"\n{'='*60}")
    print("✨ Proceso completado")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    process_files()