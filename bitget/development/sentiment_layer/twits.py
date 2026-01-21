import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

# ========================
# CONFIG
# ========================
BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAEKl7AEAAAAAYh0CLMHBaBcn3nFSwmkngvXgFUY%3DcAyfsrA3wUOKSDDAyIfG3jCe3rum9a1JZNnT0dgoHN7oCrGwOx"  # Sustituye con tu token
KEYWORDS = ["BTC", "ETH"]
TIMEFRAMES = ["1H", "4H", "6H"]  
DAYS_BACK = 1  # Twitter free API recent search permite máximo 7 días
START_DATE = datetime.now(timezone.utc) - timedelta(days=DAYS_BACK)

# ========================
# FUNCIONES
# ========================
def count_tweets(keyword, start_time, end_time):
    """Cuenta tweets por hora con la keyword usando la API recent search."""
    url = "https://api.twitter.com/2/tweets/counts/recent"
    headers = {"Authorization": f"Bearer {BEARER_TOKEN}"}
    params = {
        "query": keyword,
        "granularity": "hour",
        "start_time": start_time.isoformat(),
        "end_time": end_time.isoformat()
    }
    resp = requests.get(url, headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data.get("data", []))
    if df.empty:
        return pd.DataFrame(columns=["timestamp", "count"])
    df["timestamp"] = pd.to_datetime(df["start"], utc=True)
    df["count"] = df["tweet_count"]
    df = df[["timestamp", "count"]].sort_values("timestamp").reset_index(drop=True)
    return df

def resample_df(df, tf):
    """Resamplea el DataFrame a timeframe tf"""
    df_res = df.set_index("timestamp").resample(tf).sum().reset_index()
    return df_res

# ========================
# SCRIPT PRINCIPAL
# ========================
if __name__ == "__main__":
    end = datetime.now(timezone.utc)
    start = START_DATE

    dfs = []
    for kw in KEYWORDS:
        df_kw = count_tweets(kw, start, end)
        df_kw.rename(columns={"count": f"{kw.lower()}_mentions"}, inplace=True)
        dfs.append(df_kw)

    # Merge por timestamp
    df_mentions = dfs[0]
    for df in dfs[1:]:
        df_mentions = pd.merge(df_mentions, df, on="timestamp", how="outer")

    # Guardar por timeframe
    for tf in TIMEFRAMES:
        df_tf = resample_df(df_mentions, tf)
        fname_base = f"mentions_{tf}utc" if tf == "6H" else f"mentions_{tf}"

        # Guardar Parquet
        df_tf.to_parquet(f"{fname_base}.parquet", index=False)

        # Guardar Excel (sin timezone)
        df_excel = df_tf.copy()
        df_excel["timestamp"] = df_excel["timestamp"].dt.tz_localize(None)
        df_excel.to_excel(f"{fname_base}.xlsx", index=False)

        print(f"✅ Guardado {tf}: {len(df_tf)} filas → {fname_base}.parquet / .xlsx")
