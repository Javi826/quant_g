import requests
import pandas as pd

def download_fng_index():
    """Descarga el índice FNG y normaliza valores (0-1). Filtra desde 2025-01-01 UTC"""
    url = "https://api.alternative.me/fng/?limit=0"
    resp = requests.get(url)
    resp.raise_for_status()
    data_json = resp.json()

    df = pd.DataFrame(data_json["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df["fear_greed_norm"] = df["value"].astype(float) / 100
    df = df[["timestamp", "fear_greed_norm"]].sort_values("timestamp").reset_index(drop=True)

    # Filtrar desde 2025
    df = df[df["timestamp"] >= pd.Timestamp("2025-01-01", tz="UTC")]
    return df

def expand_to_timeframes(df_daily, timeframes=["1H", "4H", "6H"]):
    """Expande el índice diario a distintos timeframes"""
    start = df_daily["timestamp"].min().floor("D")
    end = df_daily["timestamp"].max().ceil("D")
    results = {}

    for tf in timeframes:
        idx = pd.date_range(start=start, end=end, freq=tf, tz="UTC")
        df_tf = pd.DataFrame({"timestamp": idx})

        # Merge asof para asignar último valor diario disponible
        df_tf = pd.merge_asof(
            df_tf.sort_values("timestamp"),
            df_daily.sort_values("timestamp"),
            left_on="timestamp",
            right_on="timestamp",
            direction="backward"
        )
        results[tf] = df_tf
    return results

if __name__ == "__main__":
    df_fng = download_fng_index()
    dfs = expand_to_timeframes(df_fng, ["1H", "4H", "6H"])

    for tf, df in dfs.items():
        # Definir sufijo
        if tf == "6H":
            fname_base = "fear_greed_6Hutc"
        else:
            fname_base = f"fear_greed_{tf}"

        # Guardar Parquet (mantiene UTC)
        df.to_parquet(f"{fname_base}.parquet", index=False)

        # Guardar Excel (sin timezone)
        df_excel = df.copy()
        df_excel["timestamp"] = df_excel["timestamp"].dt.tz_localize(None)
        df_excel.to_excel(f"{fname_base}.xlsx", index=False)

        print(f"✅ Guardado {tf}: {len(df)} filas → {fname_base}.parquet / .xlsx")
