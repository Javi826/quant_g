import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
input_folder = BASE_DIR / "data" / "crypto_2021"

# Leer un símbolo cualquiera (elige uno que tengas)
symbol = "BTCUSDT"  # Cambia si no tienes este

df_1d = pd.read_parquet(input_folder / f"{symbol}_1D.parquet")
df_4h = pd.read_parquet(input_folder / f"{symbol}_4H.parquet")

print("="*60)
print("ANÁLISIS DE ALINEACIÓN - BITGET")
print("="*60)

print("\n📅 Primeros 3 días (1D):")
print(df_1d['timestamp'].head(3).to_list())

print("\n🕐 Primeras 12 barras de 4H:")
for i, ts in enumerate(df_4h['timestamp'].head(12)):
    print(f"  {i+1:2d}. {ts}")

print("\n🔍 Análisis de un día específico:")
# Tomar el segundo día
day_ts = df_1d['timestamp'].iloc[1]
print(f"\nDía en 1D: {day_ts}")

# Buscar barras de 4H alrededor de ese día
start = day_ts - pd.Timedelta(hours=8)
end = day_ts + pd.Timedelta(hours=28)
bars_4h = df_4h[(df_4h['timestamp'] >= start) & (df_4h['timestamp'] <= end)]

print(f"\nBarras de 4H cerca de {day_ts}:")
for ts in bars_4h['timestamp']:
    marker = " ← DÍA 1D" if ts == day_ts else ""
    print(f"  {ts}{marker}")

print("\n" + "="*60)