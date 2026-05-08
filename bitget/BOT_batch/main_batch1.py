"""
analyze_symbol_coverage.py
---------------------------
Standalone script to diagnose symbol availability differences
across timeframes (1H, 4H) and OOS periods (OOS2, OOS3).

Run blocks independently by commenting/uncommenting sections.
"""

import os
import pandas as pd

# =============================================================================
# CONFIGURATION — edit these paths
# =============================================================================

BASE        = "/home/javi/projects/quant/quant_b/bitget/data_pipeline/data/04_split/expanding"
OOS2_FOLDER = os.path.join(BASE, "OOS", "crypto_2022-01_2023-01_OOS")
OOS3_FOLDER = os.path.join(BASE, "OOS", "crypto_2023-01_2024-01_OOS")

TIMEFRAMES  = ["1H", "4H"]
MIN_BARS    = {"1H": 7200, "4H": 1800}   # adjust as needed
VOLUME_COL  = "volume"                    # adjust if different

# =============================================================================
# BLOCK 1 — List all available parquet files per folder and timeframe
# =============================================================================

def block1_list_files():
    """Show how many files exist per timeframe in each OOS folder."""
    print("\n" + "="*80)
    print("  BLOCK 1 — File count per timeframe per folder")
    print("="*80)
    for folder_name, folder in [("OOS2", OOS2_FOLDER), ("OOS3", OOS3_FOLDER)]:
        print(f"\n  {folder_name}: {folder}")
        for tf in TIMEFRAMES:
            files = [f for f in os.listdir(folder) if f.endswith(f"_{tf}.parquet")]
            syms  = sorted([f.split(f"_{tf}")[0] for f in files])
            print(f"    {tf:>4}: {len(syms):>3} files — {syms[:5]}{'...' if len(syms) > 5 else ''}")

block1_list_files()

# =============================================================================
# BLOCK 2 — For each symbol, compare row count across timeframes
# =============================================================================

def block2_compare_row_counts():
    """For each symbol present in both timeframes, compare bar counts."""
    print("\n" + "="*80)
    print("  BLOCK 2 — Row count comparison per symbol (1H vs 4H)")
    print("="*80)

    for folder_name, folder in [("OOS2", OOS2_FOLDER), ("OOS3", OOS3_FOLDER)]:
        print(f"\n  {folder_name}")
        print(f"  {'Symbol':<20} {'1H rows':>10} {'4H rows':>10} {'1H days':>10} {'4H days':>10} {'1H pass':>8} {'4H pass':>8}")
        print("  " + "-"*80)

        syms_1h = {f.split("_1H")[0] for f in os.listdir(folder) if f.endswith("_1H.parquet")}
        syms_4h = {f.split("_4H")[0] for f in os.listdir(folder) if f.endswith("_4H.parquet")}
        common  = sorted(syms_1h & syms_4h)

        for sym in common:
            df1 = pd.read_parquet(os.path.join(folder, f"{sym}_1H.parquet"))
            df4 = pd.read_parquet(os.path.join(folder, f"{sym}_4H.parquet"))
            r1  = len(df1)
            r4  = len(df4)
            d1  = round(r1 / 24, 1)
            d4  = round(r4 / 6, 1)
            p1  = "✅" if r1 >= MIN_BARS["1H"] else "❌"
            p4  = "✅" if r4 >= MIN_BARS["4H"] else "❌"
            print(f"  {sym:<20} {r1:>10} {r4:>10} {d1:>10} {d4:>10} {p1:>8} {p4:>8}")

block2_compare_row_counts()

# =============================================================================
# BLOCK 3 — Symbols that pass 4H but fail 1H
# =============================================================================

def block3_asymmetric_symbols():
    """Show symbols that pass 4H filter but fail 1H filter."""
    print("\n" + "="*80)
    print("  BLOCK 3 — Symbols passing 4H but failing 1H")
    print("="*80)

    for folder_name, folder in [("OOS2", OOS2_FOLDER), ("OOS3", OOS3_FOLDER)]:
        print(f"\n  {folder_name}")
        syms_1h = {f.split("_1H")[0] for f in os.listdir(folder) if f.endswith("_1H.parquet")}
        syms_4h = {f.split("_4H")[0] for f in os.listdir(folder) if f.endswith("_4H.parquet")}
        common  = sorted(syms_1h & syms_4h)

        asymmetric = []
        for sym in common:
            df1 = pd.read_parquet(os.path.join(folder, f"{sym}_1H.parquet"))
            df4 = pd.read_parquet(os.path.join(folder, f"{sym}_4H.parquet"))
            pass_1h = len(df1) >= MIN_BARS["1H"]
            pass_4h = len(df4) >= MIN_BARS["4H"]
            if pass_4h and not pass_1h:
                asymmetric.append((sym, len(df1), len(df4)))

        if asymmetric:
            print(f"  {'Symbol':<20} {'1H rows':>10} {'4H rows':>10} {'1H equiv days':>15} {'4H equiv days':>15}")
            print("  " + "-"*75)
            for sym, r1, r4 in asymmetric:
                print(f"  {sym:<20} {r1:>10} {r4:>10} {round(r1/24,1):>15} {round(r4/6,1):>15}")
        else:
            print("  No asymmetric symbols found.")

block3_asymmetric_symbols()

# =============================================================================
# BLOCK 4 — Date range coverage per symbol
# =============================================================================

def block4_date_ranges():
    """Show date range covered by each symbol per timeframe."""
    print("\n" + "="*80)
    print("  BLOCK 4 — Date range coverage per symbol")
    print("="*80)

    for folder_name, folder in [("OOS2", OOS2_FOLDER), ("OOS3", OOS3_FOLDER)]:
        print(f"\n  {folder_name}")
        print(f"  {'Symbol':<20} {'TF':>4} {'First bar':<22} {'Last bar':<22} {'Rows':>8}")
        print("  " + "-"*80)

        for tf in TIMEFRAMES:
            syms = sorted([f.split(f"_{tf}")[0] for f in os.listdir(folder) if f.endswith(f"_{tf}.parquet")])
            for sym in syms[:10]:  # limit to first 10 per timeframe
                df = pd.read_parquet(os.path.join(folder, f"{sym}_{tf}.parquet"))
                ts_col = "timestamp" if "timestamp" in df.columns else df.index.name or "index"
                try:
                    first = str(pd.to_datetime(df["timestamp"].iloc[0]))[:19] if "timestamp" in df.columns else str(df.index[0])[:19]
                    last  = str(pd.to_datetime(df["timestamp"].iloc[-1]))[:19] if "timestamp" in df.columns else str(df.index[-1])[:19]
                except Exception:
                    first, last = "?", "?"
                print(f"  {sym:<20} {tf:>4} {first:<22} {last:<22} {len(df):>8}")

block4_date_ranges()

# =============================================================================
# BLOCK 5 — Summary: how many symbols pass each timeframe filter
# =============================================================================

def block5_filter_summary():
    """Summary count of passing symbols per timeframe per folder."""
    print("\n" + "="*80)
    print("  BLOCK 5 — Filter summary")
    print("="*80)

    for folder_name, folder in [("OOS2", OOS2_FOLDER), ("OOS3", OOS3_FOLDER)]:
        print(f"\n  {folder_name}")
        for tf in TIMEFRAMES:
            files   = [f for f in os.listdir(folder) if f.endswith(f"_{tf}.parquet")]
            passing = []
            failing = []
            for f in files:
                sym = f.split(f"_{tf}")[0]
                df  = pd.read_parquet(os.path.join(folder, f))
                if len(df) >= MIN_BARS[tf]:
                    passing.append(sym)
                else:
                    failing.append((sym, len(df)))
            print(f"    {tf}: {len(passing)} pass | {len(failing)} fail (min_bars={MIN_BARS[tf]})")
            if failing:
                print(f"      Failing: {[(s, r) for s, r in sorted(failing, key=lambda x: x[1], reverse=True)[:5]]}")

block5_filter_summary()

# =============================================================================
# BLOCK 6 — Exact count passing with new min_bars threshold
# =============================================================================

def block6_new_threshold(new_min_bars_1h=5500):
    """Check how many symbols pass with a new 1H min_bars threshold."""
    print("\n" + "="*80)
    print(f"  BLOCK 6 — Symbols passing 1H with min_bars={new_min_bars_1h}")
    print("="*80)

    for folder_name, folder in [("OOS2", OOS2_FOLDER), ("OOS3", OOS3_FOLDER)]:
        print(f"\n  {folder_name}")
        files   = [f for f in os.listdir(folder) if f.endswith("_1H.parquet")]
        results = []
        for f in files:
            sym = f.split("_1H")[0]
            n   = len(pd.read_parquet(os.path.join(folder, f)))
            results.append((sym, n, "✅ PASS" if n >= new_min_bars_1h else "❌ FAIL"))
        results.sort(key=lambda x: x[1], reverse=True)
        passing = sum(1 for _, _, s in results if "PASS" in s)
        print(f"  Passing: {passing}/{len(results)}")
        print(f"  {'Symbol':<20} {'Rows':>8} {'Status':>8}")
        print("  " + "-"*40)
        for sym, n, status in results:
            print(f"  {sym:<20} {n:>8} {status:>8}")

block6_new_threshold(new_min_bars_1h=5500)

# =============================================================================
# BLOCK 7 — Check 02_clean origin for asymmetric symbols
# =============================================================================

def block7_check_clean_dir():
    """Check if the data asymmetry originates in 02_clean."""
    clean_dir = "/home/javi/projects/quant/quant_b/bitget/data_pipeline/data/02_clean"
    symbols   = ["AAVEUSDT", "AVAXUSDT", "DOGEUSDT", "NEARUSDT", "SOLUSDT", "BTCUSDT", "ETHUSDT"]

    print("\n" + "="*80)
    print("  BLOCK 7 — Date range in 02_clean (origin check)")
    print("="*80)
    print(f"  {'Symbol':<20} {'TF':>4} {'First bar':<22} {'Last bar':<22} {'Rows':>8}")
    print("  " + "-"*80)

    for sym in symbols:
        for tf in ["1H", "4H"]:
            f = os.path.join(clean_dir, f"{sym}_{tf}.parquet")
            if os.path.exists(f):
                df = pd.read_parquet(f)
                if "timestamp" not in df.columns:
                    df = df.reset_index()
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                print(f"  {sym:<20} {tf:>4} {str(df['timestamp'].min())[:19]:<22} {str(df['timestamp'].max())[:19]:<22} {len(df):>8}")
            else:
                print(f"  {sym:<20} {tf:>4} {'FILE MISSING':<22}")

block7_check_clean_dir()

# =============================================================================
# BLOCK 8 — Check 03_highlow for asymmetric symbols
# =============================================================================

def block8_check_highlow():
    highlow_dir = "/home/javi/projects/quant/quant_b/bitget/data_pipeline/data/03_highlow"
    symbols     = ["AAVEUSDT", "AVAXUSDT", "DOGEUSDT", "BTCUSDT"]

    print("\n" + "="*80)
    print("  BLOCK 8 — Date range in 03_highlow")
    print("="*80)
    print(f"  {'Symbol':<20} {'TF':>4} {'First bar':<22} {'Last bar':<22} {'Rows':>8}")
    print("  " + "-"*80)

    for sym in symbols:
        for tf in ["1H", "4H"]:
            f = os.path.join(highlow_dir, f"{sym}_{tf}.parquet")
            if os.path.exists(f):
                df = pd.read_parquet(f)
                if "timestamp" not in df.columns:
                    df = df.reset_index()
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                print(f"  {sym:<20} {tf:>4} {str(df['timestamp'].min())[:19]:<22} {str(df['timestamp'].max())[:19]:<22} {len(df):>8}")
            else:
                print(f"  {sym:<20} {tf:>4} {'FILE MISSING':<22}")

block8_check_highlow()

# =============================================================================
# BLOCK 9 — Check 15m data for asymmetric symbols
# =============================================================================

def block9_check_15m():
    clean_dir = "/home/javi/projects/quant/quant_b/bitget/data_pipeline/data/02_clean"
    symbols   = ["AAVEUSDT", "AVAXUSDT", "DOGEUSDT", "BTCUSDT"]

    print("\n" + "="*80)
    print("  BLOCK 9 — Date range of 15m in 02_clean")
    print("="*80)
    print(f"  {'Symbol':<20} {'First bar':<22} {'Last bar':<22} {'Rows':>8}")
    print("  " + "-"*80)

    for sym in symbols:
        f = os.path.join(clean_dir, f"{sym}_15m.parquet")
        if os.path.exists(f):
            df = pd.read_parquet(f)
            if "timestamp" not in df.columns:
                df = df.reset_index()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            print(f"  {sym:<20} {str(df['timestamp'].min())[:19]:<22} {str(df['timestamp'].max())[:19]:<22} {len(df):>8}")
        else:
            print(f"  {sym:<20} {'FILE MISSING':<46}")

block9_check_15m()

# =============================================================================
# BLOCK 10 — First candle date per timeframe for a specific symbol
# =============================================================================

def block10_first_candle_per_tf():
    clean_dir  = "/home/javi/projects/quant/quant_b/bitget/data_pipeline/data/02_clean"
    symbol     = "AAVEUSDT"
    timeframes = ["15m", "1H", "4H", "1Dutc"]

    print("\n" + "="*80)
    print(f"  BLOCK 10 — First/last candle per timeframe for {symbol}")
    print("="*80)
    print(f"  {'TF':<10} {'First bar':<25} {'Last bar':<25} {'Rows':>8}")
    print("  " + "-"*72)

    for tf in timeframes:
        f = os.path.join(clean_dir, f"{symbol}_{tf}.parquet")
        if os.path.exists(f):
            df = pd.read_parquet(f)
            if "timestamp" not in df.columns:
                df = df.reset_index()
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            print(f"  {tf:<10} {str(df['timestamp'].min())[:22]:<25} {str(df['timestamp'].max())[:22]:<25} {len(df):>8}")
        else:
            print(f"  {tf:<10} {'FILE MISSING':<25}")

block10_first_candle_per_tf()

# =============================================================================
# BLOCK 11 — Find earliest available candle from Bitget API
# =============================================================================

# =============================================================================
# def block11_earliest_bitget():
#     import sys
#     sys.path.insert(0, "/home/javi/projects/quant/quant_b/bitget/shared/broker_api")
#     from api_client import _call_history_candles
#     import time
# 
#     symbol     = "AAVEUSDT"
#     timeframes = ["15m", "1H", "4H"]
#     gran_map   = {"15m": 15*60*1000, "1H": 3600*1000, "4H": 4*3600*1000}
# 
#     print("\n" + "="*80)
#     print(f"  BLOCK 11 — Earliest candle from Bitget API for {symbol}")
#     print("="*80)
# 
#     for tf in timeframes:
#         gran_ms = gran_map[tf]
#         end     = int(__import__('datetime').datetime.now(__import__('datetime').timezone.utc).timestamp() * 1000)
#         earliest = None
#         prev_end = None
# 
#         for _ in range(500):
#             data = _call_history_candles(symbol, tf, limit=200, endTime=end)
#             time.sleep(0.06)
#             if not data:
#                 break
#             timestamps = [int(item[0]) for item in data if item]
#             if not timestamps:
#                 break
#             min_ts   = min(timestamps)
#             new_end  = min_ts - gran_ms
#             if prev_end is not None and new_end >= prev_end:
#                 break
#             prev_end = end
#             earliest = min_ts
#             end      = new_end
# 
#         if earliest:
#             from datetime import datetime, timezone
#             dt = datetime.fromtimestamp(earliest / 1000, tz=timezone.utc)
#             print(f"  {tf:<6} earliest: {dt.strftime('%Y-%m-%d %H:%M:%S')} UTC")
#         else:
#             print(f"  {tf:<6} no data found")
# 
# block11_earliest_bitget()
# =============================================================================

# =============================================================================
# BLOCK 12 — Test direct download from specific start date
# =============================================================================

def block12_test_download_from_date():
    import sys, time
    sys.path.insert(0, "/home/javi/projects/quant/quant_b/bitget/shared/broker_api")
    from api_client import _call_history_candles
    from datetime import datetime, timezone

    symbol     = "AAVEUSDT"
    timeframes = ["15m"]
    test_dates = ["2021-03-16", "2022-01-01", "2023-01-01", "2023-06-01"]

    print("\n" + "="*80)
    print(f"  BLOCK 12 — Test direct download from specific dates for {symbol}")
    print("="*80)

    for tf in timeframes:
        print(f"\n  {tf}:")
        for date_str in test_dates:
            start_ms = int(pd.Timestamp(date_str, tz="UTC").timestamp() * 1000)
            data     = _call_history_candles(symbol, tf, limit=5, startTime=start_ms)
            time.sleep(0.1)
            if data:
                first_ts = datetime.fromtimestamp(int(data[0][0]) / 1000, tz=timezone.utc)
                print(f"    from {date_str} → first candle returned: {first_ts.strftime('%Y-%m-%d %H:%M:%S')}")
            else:
                print(f"    from {date_str} → no data returned")

block12_test_download_from_date()

# =============================================================================
# BLOCK 13 — Test download using endTime for 15m
# =============================================================================

def block13_test_endtime():
    import sys, time
    sys.path.insert(0, "/home/javi/projects/quant/quant_b/bitget/shared/broker_api")
    from api_client import _call_history_candles
    from datetime import datetime, timezone

    symbol    = "AAVEUSDT"
    test_ends = ["2021-06-14", "2021-07-01", "2022-01-01", "2023-01-01"]

    print("\n" + "="*80)
    print(f"  BLOCK 13 — Test endTime download for {symbol} 15m")
    print("="*80)

    for date_str in test_ends:
        end_ms = int(pd.Timestamp(date_str, tz="UTC").timestamp() * 1000)
        data   = _call_history_candles(symbol, "15m", limit=5, endTime=end_ms)
        time.sleep(0.1)
        if data:
            first_ts = datetime.fromtimestamp(int(data[0][0])  / 1000, tz=timezone.utc)
            last_ts  = datetime.fromtimestamp(int(data[-1][0]) / 1000, tz=timezone.utc)
            print(f"  endTime={date_str} → first={first_ts.strftime('%Y-%m-%d %H:%M')} last={last_ts.strftime('%Y-%m-%d %H:%M')} ({len(data)} candles)")
        else:
            print(f"  endTime={date_str} → no data returned")

block13_test_endtime()


