import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path

# -----------------------------
# PARÁMETROS DE CONFIGURACIÓN
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # sube un nivel
input_folder = BASE_DIR / "crypto_2023"
output_folder = BASE_DIR / "crypto_2023_highlow"
timeframes_to_consider = ["15m", "1H"]

# Crear carpeta de salida si no existe
output_folder.mkdir(exist_ok=True, parents=True)

# -----------------------------
# Función principal de cálculo
# -----------------------------
def find_timestamp_extremum(df_high_tf, df_low_tf):
    df_high_tf = df_high_tf.copy()
    df_high_tf = df_high_tf.loc[df_high_tf.index >= df_low_tf.index[0]]

    # Inicializar columnas datetime
    df_high_tf["low_time"] = pd.NaT
    df_high_tf["high_time"] = pd.NaT

    for i in tqdm(range(len(df_high_tf) - 1), desc="Calculando extremos"):
        start = df_high_tf.index[i]
        end = df_high_tf.index[i + 1]

        mask = (df_low_tf.index >= start) & (df_low_tf.index < end)
        df_low_slice = df_low_tf.loc[mask]

        if df_low_slice.empty:
            df_high_tf.loc[start, ["high_time", "low_time"]] = [pd.NaT, pd.NaT]
            continue

        high_time = df_low_slice["high"].idxmax()
        low_time = df_low_slice["low"].idxmin()
        df_high_tf.loc[start, "high_time"] = high_time
        df_high_tf.loc[start, "low_time"] = low_time

    df_high_tf = df_high_tf.iloc[:-1]  # eliminar última fila incompleta

    # Asegurar tipos
    numeric_cols = ["open","high","low","close","volume_base","volume_quote"]
    df_high_tf[numeric_cols] = df_high_tf[numeric_cols].astype(float)
    df_high_tf["low_time"] = pd.to_datetime(df_high_tf["low_time"])
    df_high_tf["high_time"] = pd.to_datetime(df_high_tf["high_time"])

    # Porcentaje filas buenas
    good_mask = df_high_tf["high_time"].notna() & df_high_tf["low_time"].notna()
    percentage_good_row = good_mask.sum() / len(df_high_tf) * 100
    percentage_garbage_row = 100 - percentage_good_row
    print(f"WARNINGS: Garbage row: {'%.2f' % percentage_garbage_row} %")

    return df_high_tf

# -----------------------------
# Detectar todos los ficheros Parquet
# -----------------------------
all_files = list(input_folder.glob("*.parquet"))

file_map = {}
for f in all_files:
    name_parts = f.stem.split("_")
    symbol = "_".join(name_parts[:-1])
    tf_ext = name_parts[-1]
    if symbol not in file_map:
        file_map[symbol] = {}
    file_map[symbol][tf_ext] = f

# -----------------------------
# Procesar todos los símbolos
# -----------------------------
summary = []

for symbol, tf_files in file_map.items():
    available_tfs = [tf for tf in tf_files.keys() if tf in timeframes_to_consider]
    if len(available_tfs) < 2:
        print(f"❌ Jumping {symbol}: no at least 2 df selected")
        continue

    def tf_to_minutes(tf):
        if tf.endswith("m"):
            return int(tf[:-1])
        elif tf.endswith("h"):
            return int(tf[:-1]) * 60
        elif tf.endswith("d"):
            return int(tf[:-1]) * 1440
        else:
            return 999999

    sorted_tfs = sorted(available_tfs, key=tf_to_minutes)
    low_tf_name = sorted_tfs[0]
    high_tf_name = sorted_tfs[-1]

    print(f"\nProcesando {symbol}: low_tf={low_tf_name}, high_tf={high_tf_name}")

    # -----------------------------
    # Cargar Parquet
    # -----------------------------
    df_low_tf  = pd.read_parquet(tf_files[low_tf_name], engine='pyarrow')
    df_high_tf = pd.read_parquet(tf_files[high_tf_name], engine='pyarrow')

    # Asegurar índice datetime
    if not pd.api.types.is_datetime64_any_dtype(df_low_tf.index):
        df_low_tf = df_low_tf.set_index(pd.to_datetime(df_low_tf["timestamp"]))
    if not pd.api.types.is_datetime64_any_dtype(df_high_tf.index):
        df_high_tf = df_high_tf.set_index(pd.to_datetime(df_high_tf["timestamp"]))

    # Calcular extremos
    df_ready = find_timestamp_extremum(df_high_tf, df_low_tf)

    # Guardar Excel y Parquet
    excel_file = output_folder / f"{symbol}_{high_tf_name}.xlsx"
    parquet_file = output_folder / f"{symbol}_{high_tf_name}.parquet"

    df_ready.to_excel(excel_file, index=False, float_format="%.6f")
    df_ready.to_parquet(parquet_file, engine='pyarrow', index=False)

    print(f"💾 Saved: {excel_file} y {parquet_file}")

    summary.append({
        "symbol": symbol,
        "low_tf": low_tf_name,
        "high_tf": high_tf_name,
        "rows": len(df_ready),
        "good_rows_pct": len(df_ready.dropna())/len(df_ready)*100
    })

# -----------------------------
# Resumen final
# -----------------------------
print("\n--- Resumen final (solo filas con garbage) ---")
for s in summary:
    if s['good_rows_pct'] < 100:
        print(f"{s['symbol']}: {s['low_tf']} -> {s['high_tf']}, filas={s['rows']}, good_rows%={'%.2f' % s['good_rows_pct']}%")
        

