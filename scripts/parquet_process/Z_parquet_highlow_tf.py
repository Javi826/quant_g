import re
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# -----------------------------
# PARÁMETROS DE CONFIGURACIÓN
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # sube un nivel
input_folder = BASE_DIR / "data" / "crypto_2021_copy_clean"
output_folder = BASE_DIR / "data" / "crypto_2021_highlow"

# Par de timeframes a procesar: [timeframe_superior, timeframe_intrabarra]
timeframes_to_consider = ["1D", "4H"]

output_folder.mkdir(exist_ok=True, parents=True)


def parse_filename(filename):
    """
    Extrae el símbolo y timeframe del nombre del archivo.
    Formato esperado: sym_timeframe.parquet o sym_timeframe.xlsx
    """
    stem = filename.stem  # nombre sin extensión
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        symbol = parts[0]
        timeframe = parts[1]
        return symbol, timeframe
    return None, None


def read_file(filepath):
    """Lee archivo parquet o xlsx y devuelve DataFrame con índice datetime."""
    if filepath.suffix == ".parquet":
        df = pd.read_parquet(filepath)
    elif filepath.suffix == ".xlsx":
        df = pd.read_excel(filepath)
    else:
        return None
    
    # Asegurar que timestamp sea el índice
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    return df


def write_file(df, filepath):
    """Escribe DataFrame a parquet o xlsx según extensión original."""
    df_out = df.reset_index()
    df_out.rename(columns={"index": "timestamp"}, inplace=True)
    
    if filepath.suffix == ".parquet":
        df_out.to_parquet(filepath, index=False)
    elif filepath.suffix == ".xlsx":
        df_out.to_excel(filepath, index=False)


def find_timestamp_extremum(df, df_lower_timeframe):
    """
    Encuentra el timestamp exacto donde ocurren el high y low de cada barra.
    
    :param df: DataFrame del timeframe superior (ej: 4H)
    :param df_lower_timeframe: DataFrame del timeframe inferior (ej: 1H)
    :return: df con columnas adicionales low_time y high_time
    """
    df = df.copy()
    
    # Ajustar inicio al primer dato disponible en lower timeframe
    df = df.loc[df_lower_timeframe.index[0]:]
    
    # Inicializar nuevas columnas
    df["low_time"] = pd.NaT
    df["high_time"] = pd.NaT
    
    # Procesar cada barra
    for i in tqdm(range(len(df) - 1), desc="Procesando barras"):
        start = df.index[i]
        end = df.index[i + 1]
        
        # Extraer datos intrabarra del período
        # Incluimos la barra inicial (start) pero excluimos la final (end)
        intrabar_data = df_lower_timeframe.loc[start:end].iloc[:-1]
        
        if len(intrabar_data) == 0:
            continue
        
        try:
            # Encontrar timestamp del máximo y mínimo en todas las barras del período
            high_time = intrabar_data["high"].idxmax()
            low_time = intrabar_data["low"].idxmin()
            
            df.loc[start, "low_time"] = low_time
            df.loc[start, "high_time"] = high_time
            
        except Exception as e:
            print(f"Error en {start}: {e}")
            continue
    
    # Eliminar última fila (incompleta)
    df = df.iloc[:-1]
    
    # Estadísticas
    valid_rows = df[["low_time", "high_time"]].notna().all(axis=1).sum()
    total_rows = len(df)
    percentage_valid = (valid_rows / total_rows * 100) if total_rows > 0 else 0
    print(f"Filas válidas: {valid_rows}/{total_rows} ({percentage_valid:.2f}%)")
    
    return df


def process_files():
    """Procesa todos los archivos en la carpeta input."""
    
    # Obtener todos los archivos
    files = list(input_folder.glob("*.parquet")) + list(input_folder.glob("*.xlsx"))
    
    if len(files) == 0:
        print(f"❌ No se encontraron archivos en {input_folder}")
        return
    
    # Agrupar por (símbolo, timeframe, extensión) para procesar cada uno
    symbol_timeframe_files = {}
    for file in files:
        symbol, timeframe = parse_filename(file)
        if symbol and timeframe:
            key = (symbol, timeframe, file.suffix)
            symbol_timeframe_files[key] = file
    
    # Reorganizar para contar símbolos únicos
    symbols_set = set()
    for (symbol, _, _) in symbol_timeframe_files.keys():
        symbols_set.add(symbol)
    
    print(f"Encontrados {len(symbols_set)} símbolos")
    print(f"Par a procesar: {timeframes_to_consider[0]} -> {timeframes_to_consider[1]}\n")
    
    tf_high = timeframes_to_consider[0]
    tf_low = timeframes_to_consider[1]
    
    # Procesar cada combinación (símbolo, extensión)
    processed = set()
    
    for (symbol, timeframe, extension) in sorted(symbol_timeframe_files.keys()):
        # Solo procesar archivos del timeframe superior
        if timeframe != tf_high:
            continue
        
        # Evitar procesar el mismo símbolo+extensión dos veces
        combo_key = (symbol, extension)
        if combo_key in processed:
            continue
        
        print(f"\n{'='*60}")
        print(f"Procesando: {symbol} ({extension})")
        print(f"{'='*60}")
        
        # Buscar archivos con la misma extensión
        key_high = (symbol, tf_high, extension)
        key_low = (symbol, tf_low, extension)
        
        if key_high not in symbol_timeframe_files:
            print(f"⚠️  {symbol}: No existe archivo {tf_high}{extension}")
            continue
        
        if key_low not in symbol_timeframe_files:
            print(f"⚠️  {symbol}: No existe archivo {tf_low}{extension} (intrabarra)")
            continue
        
        print(f"\n📊 {symbol}_{tf_high}{extension} (usando {tf_low}{extension} como intrabarra)")
        
        # Leer archivos
        file_high = symbol_timeframe_files[key_high]
        file_low = symbol_timeframe_files[key_low]
        
        df_high = read_file(file_high)
        df_low = read_file(file_low)
        
        if df_high is None or df_low is None:
            print(f"❌ Error leyendo archivos para {symbol}")
            continue
        
        # Normalizar nombres de columnas a minúsculas
        df_high.columns = df_high.columns.str.lower()
        df_low.columns = df_low.columns.str.lower()
        
        # Procesar
        df_result = find_timestamp_extremum(df_high, df_low)
        
        # Guardar resultado con la misma extensión
        output_file = output_folder / file_high.name
        write_file(df_result, output_file)
        print(f"✅ Guardado: {output_file.name}")
        
        processed.add(combo_key)
    
    print(f"\n{'='*60}")
    print("✨ Proceso completado")
    print(f"{'='*60}")


if __name__ == "__main__":
    process_files()