import os
import pandas as pd
from glob import glob

def resumen_trades_brief_trades():

    folder = "brief_trades"

    # 1) Verificar que la carpeta exista
    if not os.path.exists(folder):
        print("⚠️ La carpeta 'brief_trades' no existe.")
        return

    # 2) Buscar todos los excels
    excel_files = glob(os.path.join(folder, "*.xlsx"))

    if not excel_files:
        print("⚠️ No se encontraron archivos .xlsx en brief_trades.")
        return

    print(f"📄 Archivos encontrados: {len(excel_files)}")
    for f in excel_files:
        print(" -", f)

    # 3) Intentar cargar todos los excels
    dfs = []
    for f in excel_files:
        try:
            df = pd.read_excel(f, engine="openpyxl")
            dfs.append(df)
        except Exception as e:
            print(f"❌ Error leyendo {f}: {e}")

    if not dfs:
        print("⚠️ No se pudo cargar ningún archivo.")
        return

    # 4) Concatenar todos los trades
    all_trades = pd.concat(dfs, ignore_index=True)

    # 5) Verificar columnas obligatorias
    required_cols = ["symbol", "profit"]
    for col in required_cols:
        if col not in all_trades.columns:
            print(f"❌ Falta la columna requerida: '{col}'")
            print("Columnas disponibles:", list(all_trades.columns))
            return

    # 6) Arreglar el formato de profit si viene con coma
    if all_trades["profit"].dtype == object:
        all_trades["profit"] = (
            all_trades["profit"]
            .astype(str)
            .str.replace(",", ".", regex=False)
            .astype(float)
        )

    # 7) Calcular métricas por símbolo
    resumen = (
        all_trades
        .groupby("symbol")
        .agg(
            profit_total=("profit", "sum"),
            total_trades=("profit", "count"),
            win_ratio=("profit", lambda x: (x > 0).mean() * 100)
        )
        .reset_index()
    )

    # 👉 Orden alfabético por símbolo
    resumen = resumen.sort_values(by="win_ratio")

    # 8) Imprimir tabla
    print("\n📊 BRIEF SYMBOL\n")
    print(f"{'symbol':<15} {'profit':<12} {'total_trades':<15} {'win_ratio'}")
    print("-" * 55)

    for _, row in resumen.iterrows():
        print(
            f"{row['symbol']:<15} "
            f"{row['profit_total']:<12.2f} "
            f"{row['total_trades']:<15} "
            f"{row['win_ratio']:.2f}%"
        )


# Permite ejecutarlo directamente
if __name__ == "__main__":
    resumen_trades_brief_trades()
