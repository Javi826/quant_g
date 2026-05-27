#shared_batch/pipeline/universe.py
import os
import logging
import pandas as pd
from shared_config import VOLUME_COL
logger = logging.getLogger("BOT_batch.pipeline.universe")

# =============================================================================
# FILTER SYMBOLS
# =============================================================================

symbols_to_exclude = {}

# =============================================================================
# symbols_to_include = [
#     "NVDAUSDT",   # NVIDIA        - corr 0.77, 202 rows
#     "PLTRUSDT",   # Palantir      - corr 0.73, 181 rows
#     "HOODUSDT",   # Robinhood     - corr 0.71, 194 rows
#     "ASMLUSDT",   # ASML          - corr 0.70, 182 rows
#     "GOOGLUSDT",  # Alphabet      - corr 0.65, 195 rows
#     "AMZNUSDT",   # Amazon        - corr 0.65, 195 rows
#     "TSLAUSDT",   # Tesla         - corr 0.64, 202 rows
#     "COINUSDT",   # Coinbase      - corr 0.63, 194 rows
#     "MRVLUSDT",   # Marvell       - corr 0.63, 180 rows
#     "METAUSDT",   # Meta          - corr 0.59, 195 rows
#     "MSFTUSDT",   # Microsoft     - corr 0.48, 168 rows
# ]
# =============================================================================

symbols_to_include = [
    "BTCUSDT",   # Bitcoin       - corr 1.00, 133 rows
    "ETHUSDT",   # Ethereum      - corr 0.93, 133 rows
    "SOLUSDT",   # Solana        - corr 0.91, 133 rows
    "LINKUSDT",  # Chainlink     - corr 0.90, 133 rows
    "BNBUSDT",   # BNB           - corr 0.90, 133 rows
    "XRPUSDT",   # XRP           - corr 0.88, 133 rows
    "AVAXUSDT",  # Avalanche     - corr 0.87, 133 rows
    "ADAUSDT",   # Cardano       - corr 0.83, 133 rows
    "SUIUSDT",   # Sui           - corr 0.80, 133 rows
    "DOGEUSDT",  # Dogecoin      - corr 0.79, 133 rows
    "AAVEUSDT",  # Aave          - corr 0.75, 133 rows
]

def filter_symbols(symbols, min_vol_usdt, timeframe=None, data_folder=None, exchange=None,
                   min_price=None, vol_window=50, my_symbols=False, custom_symbols=None):
    ohlcv_data         = {}
    filtered_symbols   = []
    removed_symbols    = []
    removed_by_reasons = {"No data": 0, "Not enough bars": 0, "Last close too low": 0, "Avg volume too low": 0, "File missing": 0}
    
    #Inclusion
    if my_symbols:
        inclusion_list = custom_symbols if custom_symbols is not None else symbols_to_include
        symbols = [s for s in symbols if s in inclusion_list]
        
    #Exclusion
    for sym in symbols:
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
                if timeframe == "1Dutc":
                    min_bars = 300
                elif timeframe == "12Hutc":
                    min_bars = 600
                elif timeframe == "6Hutc":
                    min_bars = 1200
                elif timeframe == "4H":
                    min_bars = 1800
                elif timeframe == "1H":
                    min_bars = 7200
                elif timeframe == "30m":
                    min_bars = 14400                    
                elif timeframe == "15m":
                    min_bars = 28800
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
# =============================================================================
# UNIVERSE SELECTION
# =============================================================================

def select_universe(
    data_folder_is: str,
    data_folder_oos: str,
    timeframe: str,
    n_symbols: int,
    min_price: float,
    filter_symbols_fn: callable,
    my_symbols: bool = False,
    fix_symbols_mcis: bool = False,
    n_symbols_mcis: int = 20,
) -> tuple:
    """
    Select OOS universe (top N by volume) and match IS universe.
    If fix_symbols_mcis=True, IS universe is top n_symbols_mcis from IS by volume directly.

    Returns:
        tuple: (symbols_is_final, symbols_oos_final, ohlcv_is, ohlcv_oos)
    """
    raw_is  = sorted([f.split("_")[0] for f in os.listdir(data_folder_is)  if f.endswith(f"_{timeframe}.parquet")])
    raw_oos = sorted([f.split("_")[0] for f in os.listdir(data_folder_oos) if f.endswith(f"_{timeframe}.parquet")])

    ohlcv_oos, filtered_oos = filter_symbols_fn(raw_oos, min_vol_usdt=0, timeframe=timeframe, data_folder=data_folder_oos, min_price=min_price, vol_window=50, my_symbols=my_symbols)
    ohlcv_is,  filtered_is  = filter_symbols_fn(raw_is,  min_vol_usdt=0, timeframe=timeframe, data_folder=data_folder_is,  min_price=min_price, vol_window=50, my_symbols=my_symbols)

    def _vol_1d(sym, folder):
        path = os.path.join(folder, f"{sym}_1Dutc.parquet")
        if not os.path.exists(path):
            return 0.0
        df = pd.read_parquet(path, columns=[VOLUME_COL])
        return float(df[VOLUME_COL].tail(180).mean())

    vol_oos           = {sym: _vol_1d(sym, data_folder_oos) for sym in filtered_oos}
    oos_ranked        = sorted(filtered_oos, key=lambda s: vol_oos.get(s, 0), reverse=True)
    symbols_oos_final = oos_ranked[:n_symbols]

    if fix_symbols_mcis:
        vol_is           = {sym: _vol_1d(sym, data_folder_is) for sym in filtered_is}
        is_ranked        = sorted(filtered_is, key=lambda s: vol_is.get(s, 0), reverse=True)
        symbols_is_final = is_ranked[:n_symbols_mcis]
        logger.debug(f"FIX_SYMBOLS_MCIS_TRAINING=True — IS top {n_symbols_mcis} by volume: {symbols_is_final}")
    else:
        syms_is  = set(filtered_is)
        syms_oos = set(symbols_oos_final)
        in_both              = sorted(syms_is & syms_oos)
        only_in_oos          = sorted(syms_oos - syms_is)
        vol_is               = {sym: _vol_1d(sym, data_folder_is) for sym in syms_is}
        is_candidates_by_vol = sorted(syms_is - syms_oos, key=lambda s: vol_is.get(s, 0), reverse=True)
        needed               = max(0, n_symbols - len(in_both))
        symbols_is_final     = sorted(in_both + is_candidates_by_vol[:needed])

        logger.debug(f"OOS pool ({len(filtered_oos):>3}): {len(filtered_oos)} candidates")
        logger.debug(f"IS  pool ({len(filtered_is):>3}): {len(filtered_is)} candidates")
        logger.debug(f"In both  ({len(in_both):>3}): {in_both}")
        logger.debug(f"Only in OOS ({len(only_in_oos):>3}): {only_in_oos}")

    logger.debug(f"OOS final universe ({len(symbols_oos_final):>3}): {sorted(symbols_oos_final)}")
    logger.debug(f"IS  final universe ({len(symbols_is_final):>3}): {symbols_is_final}")

    fix_str = "FIX=True" if fix_symbols_mcis else "FIX=False"
    #logger.info(f"STAGE 0 ── Universe Selection     ── IS:{len(symbols_is_final)} symbols | OOS:{len(symbols_oos_final)} symbols | {fix_str}")

    if fix_symbols_mcis:
        if len(symbols_is_final) < n_symbols_mcis:
            logger.warning(f"⚠️  IS has only {len(symbols_is_final)} symbols — fewer than N_SYMBOLS_MCIS ({n_symbols_mcis}). Proceeding with available.")
    else:
        if len(symbols_is_final) < n_symbols:
            logger.warning(f"⚠️  IS has only {len(symbols_is_final)} symbols — fewer than N_SYMBOLS ({n_symbols}). Proceeding with available.")

    return symbols_is_final, symbols_oos_final, ohlcv_is, ohlcv_oos