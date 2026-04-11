from pathlib import Path
import pandas as pd

import warnings


DATA_FOLDER         = "data/crypto_2023_ISOLD"

warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', None)
# -----------------------------
# CONFIGURACIÓN
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
folder = BASE_DIR / "data" / "crypto_2023_ISOLD"
timeframe_to_compare = "4H"  # Timeframe a filtrar
float_tolerance = 1e-12      # Tolerancia para comparar floats
max_rows = 10                 # Máximo de filas a mostrar por símbolo

# -----------------------------
# FUNCIONES AUXILIARES
# -----------------------------
def parse_filename(filename: Path):
    stem = filename.stem
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return None, None

def get_symbols_in_folder(folder: Path, timeframe: str):
    symbols = set()
    for f in folder.glob("*.xlsx"):
        symbol, tf = parse_filename(f)
        if tf == timeframe:
            symbols.add(symbol)
    for f in folder.glob("*.parquet"):
        symbol, tf = parse_filename(f)
        if tf == timeframe:
            symbols.add(symbol)
    return symbols

def get_files_for_symbol_and_timeframe(folder: Path, symbol: str, timeframe: str):
    matched_files = []
    for ext in ["*.xlsx", "*.parquet"]:
        for f in folder.glob(ext):
            file_symbol, file_timeframe = parse_filename(f)
            if file_symbol == symbol and file_timeframe == timeframe:
                matched_files.append(f)
    return matched_files

# -----------------------------
# OBTENEMOS TODOS LOS SÍMBOLOS
# -----------------------------
symbols = get_symbols_in_folder(folder, timeframe_to_compare)

# -----------------------------
# PROCESO DE COMPARACIÓN
# -----------------------------
for symbol in sorted(symbols):
    files = get_files_for_symbol_and_timeframe(folder, symbol, timeframe_to_compare)
    if len(files) < 2:
        continue  # omitimos si falta algún archivo

    xlsx_file = next(f for f in files if f.suffix == ".xlsx")
    parquet_file = next(f for f in files if f.suffix == ".parquet")

    # Carga
    df_xlsx = pd.read_excel(xlsx_file)
    df_parquet = pd.read_parquet(parquet_file)

    # Normalizamos nombres de columnas
    df_xlsx.columns = df_xlsx.columns.str.strip().str.lower()
    df_parquet.columns = df_parquet.columns.str.strip().str.lower()

    # Si 'timestamp' no está en columnas, puede estar en el índice
    if "timestamp" not in df_xlsx.columns and isinstance(df_xlsx.index, pd.DatetimeIndex):
        df_xlsx = df_xlsx.reset_index()
        df_xlsx.rename(columns={df_xlsx.columns[0]: "timestamp"}, inplace=True)

    if "timestamp" not in df_parquet.columns:
        if isinstance(df_parquet.index, pd.DatetimeIndex):
            df_parquet = df_parquet.reset_index()
            df_parquet.rename(columns={df_parquet.columns[0]: "timestamp"}, inplace=True)
        else:
            raise KeyError(f"No se encontró la columna 'timestamp' en Parquet para {symbol}")

    # Tipos
    df_xlsx["timestamp"] = pd.to_datetime(df_xlsx["timestamp"])
    df_parquet["timestamp"] = pd.to_datetime(df_parquet["timestamp"])
    for col in ["close", "volume_quote"]:
        if col in df_xlsx.columns and col in df_parquet.columns:
            df_xlsx[col] = df_xlsx[col].astype(float)
            df_parquet[col] = df_parquet[col].astype(float)

    # Ordenar por timestamp
    df_xlsx = df_xlsx.sort_values("timestamp").reset_index(drop=True)
    df_parquet = df_parquet.sort_values("timestamp").reset_index(drop=True)

    # -----------------------------
    # COMPARACIÓN
    # -----------------------------
    merged = df_xlsx[["timestamp", "close", "volume_quote"]].merge(
        df_parquet[["timestamp", "close", "volume_quote"]],
        on="timestamp",
        how="outer",
        suffixes=("_xlsx", "_parquet"),
        indicator=True
    )

    # Comparación segura: tolerancia para close, exacta para volume_quote
    diff_flags = pd.Series(False, index=merged.index)
    diff_flags |= (merged["close_xlsx"] - merged["close_parquet"]).abs() > float_tolerance
    diff_flags |= merged["volume_quote_xlsx"] != merged["volume_quote_parquet"]
    diff_flags |= merged["_merge"] != "both"

    diff_rows = merged[diff_flags].head(max_rows)

    # -----------------------------
    # RESULTADOS
    # -----------------------------
    if not diff_rows.empty:
        print(f"❌ Diferencias para '{symbol}' (timeframe {timeframe_to_compare}):")
        print(diff_rows[["timestamp", "close_xlsx", "close_parquet", "volume_quote_xlsx", "volume_quote_parquet"]], "\n")
