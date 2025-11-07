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

BASE_URL       = "https://api.bitget.com"
PRODUCT_TYPE   = 'usdt-futures'  


def filter_symbols(symbols, min_vol_usdt, timeframe=None, data_folder=None, exchange=None, min_price=None, vol_window=50):

    ohlcv_data = {}
    filtered_symbols = []
    removed_symbols = []

    # Contadores por motivo de eliminación
    removed_by_reasons = {"No data": 0, "Not enough bars": 0, "Last close too low": 0, "Avg volume too low": 0, "File missing": 0}
    for sym in symbols:
        df = None
        reasons = []

        # -------------------
        # Load data
        # -------------------

        file_path = os.path.join(data_folder, f"{sym}_{timeframe}.parquet")

        if not os.path.exists(file_path):
            reasons.append("File missing")
        else:
            df = pd.read_parquet(file_path)
            if df.empty:
                reasons.append("No data")

            # -------------------
            # Min price
            # -------------------
            if df is not None and min_price is not None:
                last_close = df['close'].iloc[-1]
                if last_close <= min_price:
                    reasons.append("Last close too low")

            # -------------------
            # Avg volume (últimas vol_window velas)
            # -------------------
            if df is not None:
                avg_vol = df['volume_quote'].tail(vol_window).mean()
                if avg_vol < min_vol_usdt:
                    reasons.append("Avg volume too low")

            # -------------------
            # MIN BARS
            # -------------------
            if df is not None:
                n_rows = len(df)
                if timeframe == "1H":
                    min_bars = 4320
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

        # -------------------
        # Registrar resultado
        # -------------------
        if reasons:
            removed_symbols.append(sym)
            for r in reasons:
                removed_by_reasons[r] += 1
        else:
            ohlcv_data[sym] = df
            filtered_symbols.append(sym)
            
    # -------------------
    # Summary
    # -------------------
    print(f"\n🔹Total symbols BROKER   : {len(symbols)}")
    print(f"🔹Symbols removed total  : {len(removed_symbols)}")
    print(f"🔹Symbols remaining      : {len(filtered_symbols)}\n")

    return ohlcv_data, filtered_symbols

        
def final_prints(strategy, data_folder, timeframe, min_vol_usdt, order_amount, param_names, lists_for_grid):

    def format_number(n):
        if isinstance(n, (int, float)):
            # Usa formato con separador de miles y cambia coma por punto
            return f"{n:,}".replace(",", ".")
        return str(n)

    print(f'\n== {strategy} ==\n')
    print(f"DATA_FOLDER           : {data_folder}")
    print(f"TIMEFRAME             : {timeframe}")
    print(f"ORDER_AMOUNT          : {format_number(order_amount)}")
    print(f"MIN_VOL_USDT          : {format_number(min_vol_usdt)}")

    # Calcular longitud máxima de los nombres base para alinear los prints
    max_len = max(len(name) for name in param_names)

    # Imprimir las listas de parámetros alineadas
    for name, values_list in zip(param_names, lists_for_grid):
        print(f"{name + '_LIST':<{max_len + 6}} : {values_list}")
    print()

def seed_for_symbol(symbol: Union[str, object], base_seed: int = 42, path_idx: int = 0, mod: int = 100000) -> int:

    s = str(getattr(symbol, "name", symbol))
    h = hashlib.md5(s.encode("utf-8")).hexdigest()[:8]
    
    return int(base_seed) + (int(h, 16) % mod) + int(path_idx)

# EMAIL CONFIG
# -----------------------------
EMAIL_FROM     = "jlahoz.ferrandez@gmail.com"
EMAIL_PASSWORD = "tvli cxgk duwh yzdd"
EMAIL_TO       = "jlahoz.ferrandez@gmail.com"

def send_email(detected_cryptos):
    if not detected_cryptos: return
    msg = MIMEMultipart()
    msg['From'] = EMAIL_FROM
    msg['To'] = EMAIL_TO
    msg['Subject'] = f"Crypto_signals: {', '.join([d['symbol'] for d in detected_cryptos])}"
    body = "\n".join([f"{d['symbol']} | Signal: {d['signal_type']} | Close: {d['close']:.2f}" for d in detected_cryptos])
    msg.attach(MIMEText(body, 'plain'))
    try:
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(EMAIL_FROM, EMAIL_PASSWORD)
        server.send_message(msg)
        server.quit()
        print(f"📧 Email sent: {', '.join([d['symbol'] for d in detected_cryptos])}")
    except Exception as e:
        print(f"⚠️ Error sending email: {e}")

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
            file_name = f"equity_summary_{strategy_name}.xlsx"
            save_path = os.path.join(folder, file_name)
            final_df.to_excel(save_path, index=False)
            print(f"📂 Excel saved at {save_path}")
        else:
            print("⚠️ No equity data to save")

