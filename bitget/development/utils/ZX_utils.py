import os
import random
import hashlib
import smtplib
import numpy as np
import pandas as pd
from typing import Union
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

np.random.seed(42)
random.seed(42)

symbols_to_exclude = {}
symbols_to_include = ["ZENUSDT","SEIUSDT","1000BONKUSDT","BNBUSDT","FETUSDT","ETHUSDT","PENGUUSDT","AAVEUSDT",
                      "ENAUSDT","ONDOUSDT","XRPUSDT","LTCUSDT","ARBUSDT","AVAXUSDT","TAOUSDT","HBARUSDT",
                      "DOTUSDT","DOGEUSDT","SOLUSDT","FARTCOINUSDT","WIFUSDT","WLDUSDT","NEARUSDT","LINKUSDT",
                      "BGBUSDT","SNXUSDT","BTCUSDT","SUIUSDT","APTUSDT","FORMUSDT","HYPEUSDT","ADAUSDT"]
 

def filter_symbols(symbols, min_vol_usdt, timeframe=None, data_folder=None, exchange=None, min_price=None, vol_window=50, my_symbols=False):
    ohlcv_data         = {}
    filtered_symbols   = []
    removed_symbols    = []
    removed_by_reasons = {"No data": 0, "Not enough bars": 0, "Last close too low": 0, "Avg volume too low": 0, "File missing": 0}
    
    # ---- Filtro de inclusión ----
    if my_symbols:
        symbols = [s for s in symbols if s in symbols_to_include]
    # -----------------------------
    
    for sym in symbols:
        
        # ---- Exclusión manual ----
        if sym in symbols_to_exclude:
            removed_symbols.append(sym)
            continue
        # --------------------------
        
        # ---- Si my_symbols=True, solo cargar sin filtros ----
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
        # -----------------------------------------------------
        
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
                avg_vol = df['volume_quote'].tail(vol_window).mean()
                if avg_vol < min_vol_usdt:
                    reasons.append("Avg volume too low")
                    
            if df is not None:
                n_rows = len(df)
                if timeframe == "1H":
                    min_bars = 4320
                elif timeframe == "30m":
                    min_bars = 7800
                elif timeframe == "4H":
                    min_bars = 1080
                elif timeframe == "6Hutc":
                    min_bars = 720
                elif timeframe == "12Hutc":
                    min_bars = 360
                elif timeframe == "1Dutc":
                    min_bars = 180
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
            
    print(f"\n🔹Total symbols     : {len(symbols)}")
    print(f"🔹Symbols removed   : {len(removed_symbols)}")
    print(f"🔹Symbols remaining : {len(filtered_symbols)}\n")
    
    

    #print(f"\n📊 Removal reasons breakdown:")
    #for reason, count in removed_by_reasons.items():
    #     if count > 0:
    #         print(f"   • {reason:<25}: {count:>4} symbols")
    #print()
    
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


def save_filtered_symbols(filtered_symbols, strategy="_",timeframe="10H",save_symbols=False, folder="live_trading/symbols_live"):

    if save_symbols:
        os.makedirs(folder, exist_ok=True)  
        df_symbols   = pd.DataFrame({"Filtered_symbols": filtered_symbols})
        path_symbols = os.path.join(folder, f"symbols_live_{strategy}_{timeframe}.xlsx")
        df_symbols.to_excel(path_symbols, index=False)   
        print(f"📂 {len(filtered_symbols)} symbols saved in '{path_symbols}'")

def save_equity_to_excel(grid_results_list, folder, initial_capital, strategy_name,save_file=False):
    
    if save_file:
        if not os.path.exists(folder):
            os.makedirs(folder)
    
        all_dfs = []
    
        for comb, res in grid_results_list:
            for name, r in res.items():
                equity_hist = r['sim_balance_history']
                if equity_hist is None or len(equity_hist['timestamp']) == 0:
                    continue
                df_eq = pd.DataFrame(equity_hist)
                df_eq['net_gain_pct'] = (df_eq['balance'] - initial_capital) / initial_capital * 100
                df_eq['strategy'] = strategy_name
                df_eq['params'] = str(comb)
                all_dfs.append(df_eq)
    
        if all_dfs:
            final_df = pd.concat(all_dfs, ignore_index=True)
            file_name = f"equity_{strategy_name}.xlsx"
            save_path = os.path.join(folder, file_name)
            final_df.to_excel(save_path, index=False)
            print(f"📂 Excel saved at {save_path}")
        else:
            print("⚠️ No equity data to save")

import numpy as np

def align_filter_to_symbol(symbol_timestamps, btc_timestamps, btc_filter):
    """
    Alinea un filtro de BTC con los timestamps de un símbolo.
    
    Esta función resuelve el problema de que diferentes símbolos pueden tener
    diferentes fechas de inicio (diferente número de velas), pero queremos
    aplicar el filtro de BTC basándonos en el MOMENTO temporal, no en el índice.
    
    Parameters:
    -----------
    symbol_timestamps : np.array
        Timestamps del símbolo (en formato Unix o datetime64)
    btc_timestamps : np.array
        Timestamps de BTC (mismo formato)
    btc_filter : np.array
        Filtro de BTC (valores binarios: 0 o 1, o continuos)
        Debe tener la misma longitud que btc_timestamps
    
    Returns:
    --------
    aligned_filter : np.array (int8)
        Filtro alineado con los timestamps del símbolo
        Mismo tamaño que symbol_timestamps
        
    Examples:
    ---------
    >>> # BTC tiene 1000 velas desde 2022-01-01
    >>> # ETH tiene 800 velas desde 2022-01-15 (listado después)
    >>> btc_vol_filter = detect_volatility(btc_arr)  # 1000 valores
    >>> eth_timestamps = eth_arr['ts']  # 800 timestamps
    >>> btc_timestamps = btc_arr['ts']  # 1000 timestamps
    >>> 
    >>> aligned = align_filter_to_symbol(eth_timestamps, btc_timestamps, btc_vol_filter)
    >>> # aligned tiene 800 valores, cada uno corresponde al filtro BTC del mismo momento
    
    Notes:
    ------
    - Si un timestamp del símbolo no existe en BTC, usa filtro = 1 (permite operar)
    - Usa búsqueda binaria (searchsorted) para eficiencia O(log n)
    - Timestamps deben estar ordenados cronológicamente (típico en OHLCV data)
    """
    
    # Crear filtro alineado (inicializado en 1 = permitir por defecto)
    aligned_filter = np.ones(len(symbol_timestamps), dtype=np.int8)
    
    # Para cada timestamp del símbolo, buscar el correspondiente en BTC
    for i, sym_ts in enumerate(symbol_timestamps):
        # Búsqueda binaria: encuentra índice donde sym_ts encajaría en btc_timestamps
        btc_idx = np.searchsorted(btc_timestamps, sym_ts)
        
        # Si el índice está dentro del rango Y los timestamps coinciden exactamente
        if btc_idx < len(btc_filter):
            aligned_filter[i] = btc_filter[btc_idx]
    
    return aligned_filter