import os
import random
import hashlib
import numpy as np
import pandas as pd
from typing import Union
from shared_config import VOLUME_COL
import logging
logger = logging.getLogger("BOT_batch.utils")

np.random.seed(42)
random.seed(42)

symbols_to_exclude = {'XAUTUSDT', 'PAXGUSDT', 'XAGUSDT', 'XAUUSDT'}

symbols_to_include = []

def filter_symbols(symbols, min_vol_usdt, timeframe=None, data_folder=None, exchange=None,
                   min_price=None, vol_window=50, my_symbols=False, custom_symbols=None):
    ohlcv_data         = {}
    filtered_symbols   = []
    removed_symbols    = []
    removed_by_reasons = {"No data": 0, "Not enough bars": 0, "Last close too low": 0, "Avg volume too low": 0, "File missing": 0}
    
    # ---- Inclusion ----
    if my_symbols:
        inclusion_list = custom_symbols if custom_symbols is not None else symbols_to_include
        symbols = [s for s in symbols if s in inclusion_list]
    
    for sym in symbols:
        #Exclusion
        if sym in symbols_to_exclude:
            removed_symbols.append(sym)
            continue
        
        #Si my_symbols=True, solo cargar sin filtros
        if my_symbols:
            file_path = os.path.join(data_folder, f"{sym}_{timeframe}.parquet")
            if os.path.exists(file_path):
                df = pd.read_parquet(file_path)
                if not df.empty:
                    ohlcv_data[sym] = df
                    filtered_symbols.append(sym)
                else:
                    removed_symbols.append(sym)
            else:
                removed_symbols.append(sym)
            continue
        
        df        = None
        reasons   = []
        file_path = os.path.join(data_folder, f"{sym}_{timeframe}.parquet")
        
        if not os.path.exists(file_path):
            reasons.append("File missing")
        else:
            df = pd.read_parquet(file_path)
            if df.empty:
                reasons.append("No data")
            if df is not None and min_price is not None:
                last_close = df['close'].iloc[-1]
                if last_close <= min_price:
                    reasons.append("Last close too low")
                    
            if df is not None:
                avg_vol = df[VOLUME_COL].tail(vol_window).mean()
                if avg_vol < min_vol_usdt:
                    reasons.append("Avg volume too low")
                    
            if df is not None:
                n_rows = len(df)
                if timeframe == "1H":
                    min_bars = 4600
                elif timeframe == "30m":
                    min_bars = 16000
                elif timeframe == "4H":
                    min_bars = 1800
                elif timeframe == "6Hutc":
                    min_bars = 1200
                elif timeframe == "12Hutc":
                    min_bars = 700
                elif timeframe == "1Dutc":
                    min_bars = 300
                else:
                    min_bars = 999999999
                    
                if n_rows < min_bars:
                    reasons.append("Not enough bars")
        if reasons:
            removed_symbols.append(sym)
            for r in reasons:
                removed_by_reasons[r] += 1
        else:
            ohlcv_data[sym] = df
            filtered_symbols.append(sym)
            
    logger.debug(f"🔹Total symbols     : {len(symbols)}")
    logger.debug(f"🔹Symbols removed   : {len(removed_symbols)}")
    logger.debug(f"🔹Symbols remaining : {len(filtered_symbols)}")
    
    return ohlcv_data, filtered_symbols

        
def final_prints(strategy, data_folder, timeframe, min_vol_usdt, order_amount, param_names, lists_for_grid):

    def format_number(n):
        if isinstance(n, (int, float)):
            # Usa formato con separador de miles y cambia coma por punto
            return f"{n:,}".replace(",", ".")
        return str(n)

    print(f'\n== {strategy} ==\n')

    # Diccionario con todas las claves y valores a imprimir
    info = {
        "DATA_FOLDER": data_folder,
        "TIMEFRAME": timeframe,
        "ORDER_AMOUNT": format_number(order_amount),
        "MIN_VOL_USDT": format_number(min_vol_usdt),
    }

    # Añadimos los parámetros dinámicos
    for name, values_list in zip(param_names, lists_for_grid):
        info[f"{name}_LIST"] = str(values_list)

    # Calcular la longitud máxima de todas las claves
    max_key_len = max(len(k) for k in info.keys())

    # Imprimir todo alineado según la longitud máxima
    for key, value in info.items():
        print(f"{key:<{max_key_len}} : {value}")

    print()


def seed_for_symbol(symbol: Union[str, object], base_seed: int = 42, path_idx: int = 0, mod: int = 100000) -> int:

    s = str(getattr(symbol, "name", symbol))
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    
    return int(base_seed) + (int(h, 16) % mod) + int(path_idx)


def save_filtered_symbols(filtered_symbols, strategy="_", timeframe="10H", save_symbols=False, folder="live_trading/symbols_live"):
    if save_symbols:
        os.makedirs(folder, exist_ok=True)  
        df_symbols   = pd.DataFrame({"Filtered_symbols": filtered_symbols})
        path_symbols = os.path.join(folder, f"symbols_live_{strategy}_{timeframe}.csv")
        df_symbols.to_csv(path_symbols, index=False, header=False)  # Sin index, sin header
        print(f"📂 {len(filtered_symbols)} symbols saved in '{path_symbols}'")

