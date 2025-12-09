import os
import pandas as pd
import matplotlib.pyplot as plt


def generar_equity_curve(
    filename="bot_trading_trades.xlsx",
    initial_capital=3946,
    usar_fecha="CLOSE_AT",      # Cambia a "OPEN_AT" si quieres
    resample_rule="D"
):
    """
    Genera y grafica la curva de equity desde un Excel.
    Si no encuentra el archivo en /mnt/data/, usa un ejemplo interno.
    """

    # -----------------------------------------------------
    # 1. Ruta EXACTA como el original
    # -----------------------------------------------------
    file_path = f"/mnt/data/{filename}"

    if os.path.exists(file_path):
        print(f"Usando archivo real: {file_path}")
        df = pd.read_excel(file_path)
    else:
        print("Archivo NO encontrado. Usando ejemplo interno…")

        ejemplo = {
            "OPEN_AT":  ["2025-01-01 10:00", "2025-01-02 11:00", "2025-01-03 13:00"],
            "CLOSE_AT": ["2025-01-01 11:00", "2025-01-02 12:00", "2025-01-03 14:00"],
            "PROFIT":   [12.5, -8.3, 20.1]
        }

        df = pd.DataFrame(ejemplo)

    # -----------------------------------------------------
    # 2. Asegurar formato de fechas
    # -----------------------------------------------------
    df["OPEN_AT"] = pd.to_datetime(df["OPEN_AT"])
    df["CLOSE_AT"] = pd.to_datetime(df["CLOSE_AT"])

    # Elegir fecha según tu cambio
    fecha = usar_fecha.upper()
    if fecha not in ["CLOSE_AT", "OPEN_AT"]:
        raise ValueError("usar_fecha debe ser 'CLOSE_AT' o 'OPEN_AT'")

    df = df.sort_values(fecha)

    # -----------------------------------------------------
    # 3. Equity acumulada
    # -----------------------------------------------------
    df["cum_profit"] = df["PROFIT"].cumsum()
    df["equity"] = initial_capital + df["cum_profit"]

    # -----------------------------------------------------
    # 4. Serie temporal continua
    # -----------------------------------------------------
    start = df["OPEN_AT"].min()
    end   = df["CLOSE_AT"].max()

    idx = pd.date_range(start, end, freq=resample_rule)
    equity_ts = pd.Series(index=idx, dtype=float)

    for _, row in df.iterrows():
        key = pd.to_datetime(row[fecha].floor(resample_rule))
        if key in equity_ts.index:
            equity_ts.loc[key] = row["equity"]

    equity_ts = equity_ts.ffill().bfill()

    # -----------------------------------------------------
    # 5. Gráfica
    # -----------------------------------------------------
    plt.figure(figsize=(12,5))
    plt.plot(equity_ts.index, equity_ts.values, linewidth=1.7)
    plt.scatter(df[fecha], df["equity"], s=40)
    plt.title(f"Equity Curve usando {fecha}")
    plt.xlabel("Tiempo")
    plt.ylabel("Capital")
    plt.grid(True)
    plt.show()

    return df, equity_ts



# EJEMPLOS DE USO:
generar_equity_curve()                              # Usa CLOSE_AT
# generar_equity_curve(usar_fecha="OPEN_AT")          # Usa OPEN_AT
# generar_equity_curve("mi_archivo.xlsx")             # Otro archivo
