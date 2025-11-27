import pandas as pd
from pathlib import Path
from tqdm import tqdm

# -----------------------------
# CONFIGURATION PARAMETERS
# -----------------------------
BASE_DIR      = Path(__file__).resolve().parent.parent  # go up one level
input_folder  = BASE_DIR / "data" / "crypto_2025_scalping"
output_folder = BASE_DIR / "data" / "crypto_2025_scalping_hl"

# Pair of timeframes to process: [higher_timeframe, intrabar_timeframe]
timeframes_to_consider = ["15m", "5m"]

output_folder.mkdir(exist_ok=True, parents=True)


def parse_filename(filename):
    """
    Extracts the symbol and timeframe from the file name.
    Expected format: sym_timeframe.parquet or sym_timeframe.xlsx
    """
    stem = filename.stem  # name without extension
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        symbol = parts[0]
        timeframe = parts[1]
        return symbol, timeframe
    return None, None


def read_file(filepath):
    """Reads a parquet or xlsx file and returns a DataFrame with a datetime index."""
    if filepath.suffix == ".parquet":
        df = pd.read_parquet(filepath)
    elif filepath.suffix == ".xlsx":
        df = pd.read_excel(filepath)
    else:
        return None
    
    # Ensure timestamp is the index
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    elif not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    
    return df


def write_file(df, filepath):
    """Writes a DataFrame to parquet or xlsx according to the original extension."""
    df_out = df.reset_index()
    df_out.rename(columns={"index": "timestamp"}, inplace=True)
    
    if filepath.suffix == ".parquet":
        df_out.to_parquet(filepath, index=False)
    elif filepath.suffix == ".xlsx":
        df_out.to_excel(filepath, index=False)


def find_timestamp_extremum(df, df_lower_timeframe):
    """
    Finds the exact timestamp where the high and low of each bar occur.
    
    :param df: DataFrame of the higher timeframe (e.g. 4H)
    :param df_lower_timeframe: DataFrame of the lower timeframe (e.g. 1H)
    :return: df with additional columns low_time and high_time
    """
    df = df.copy()
    
    # Adjust start to first available data in lower timeframe
    df = df.loc[df_lower_timeframe.index[0]:]
    
    # Initialize new columns
    df["low_time"] = pd.NaT
    df["high_time"] = pd.NaT
    
    # Process each bar
    for i in tqdm(range(len(df) - 1), desc="Processing bars"):
        start = df.index[i]
        end = df.index[i + 1]
        
        # Extract intrabar data for the period
        # Include the initial bar (start) but exclude the final (end)
        intrabar_data = df_lower_timeframe.loc[start:end].iloc[:-1]
        
        if len(intrabar_data) == 0:
            continue
        
        try:
            # Find timestamp of max and min within the period
            high_time = intrabar_data["high"].idxmax()
            low_time = intrabar_data["low"].idxmin()
            
            df.loc[start, "low_time"] = low_time
            df.loc[start, "high_time"] = high_time
            
        except Exception as e:
            print(f"Error at {start}: {e}")
            continue
    
    # Remove last (incomplete) row
    df = df.iloc[:-1]
    
    # Stats
    valid_rows = df[["low_time", "high_time"]].notna().all(axis=1).sum()
    total_rows = len(df)
    percentage_valid = (valid_rows / total_rows * 100) if total_rows > 0 else 0
    print(f"Valid rows: {valid_rows}/{total_rows} ({percentage_valid:.2f}%)")
    
    return df


def process_files():
    """Processes all files in the input folder."""
    
    # Get all files
    files = list(input_folder.glob("*.parquet")) + list(input_folder.glob("*.xlsx"))
    
    if len(files) == 0:
        print(f"❌ No files found in {input_folder}")
        return
    
    # Group by (symbol, timeframe, extension) to process each one
    symbol_timeframe_files = {}
    for file in files:
        symbol, timeframe = parse_filename(file)
        if symbol and timeframe:
            key = (symbol, timeframe, file.suffix)
            symbol_timeframe_files[key] = file
    
    # Count unique symbols
    symbols_set = set()
    for (symbol, _, _) in symbol_timeframe_files.keys():
        symbols_set.add(symbol)
    
    print(f"Found {len(symbols_set)} symbols")
    print(f"Pair to process: {timeframes_to_consider[0]} -> {timeframes_to_consider[1]}\n")
    
    tf_high = timeframes_to_consider[0]
    tf_low = timeframes_to_consider[1]
    
    # Process each (symbol, extension) combination
    processed = set()
    
    for (symbol, timeframe, extension) in sorted(symbol_timeframe_files.keys()):
        # Only process higher timeframe files
        if timeframe != tf_high:
            continue
        
        # Avoid processing the same symbol+extension twice
        combo_key = (symbol, extension)
        if combo_key in processed:
            continue
        
        print(f"\n{'='*60}")
        print(f"Processing: {symbol} ({extension})")
        print(f"{'='*60}")
        
        # Look for files with same extension
        key_high = (symbol, tf_high, extension)
        key_low = (symbol, tf_low, extension)
        
        if key_high not in symbol_timeframe_files:
            print(f"⚠️  {symbol}: Missing file {tf_high}{extension}")
            continue
        
        if key_low not in symbol_timeframe_files:
            print(f"⚠️  {symbol}: Missing file {tf_low}{extension} (intrabar)")
            continue
        
        print(f"\n{symbol}_{tf_high}{extension} (using {tf_low}{extension} as intrabar)")
        
        # Read files
        file_high = symbol_timeframe_files[key_high]
        file_low = symbol_timeframe_files[key_low]
        
        df_high = read_file(file_high)
        df_low = read_file(file_low)
        
        if df_high is None or df_low is None:
            print(f"❌ Error reading files for {symbol}")
            continue
        
        # Normalize column names to lowercase
        df_high.columns = df_high.columns.str.lower()
        df_low.columns = df_low.columns.str.lower()
        
        # Process
        df_result = find_timestamp_extremum(df_high, df_low)
        
        # Save result with same extension
        output_file = output_folder / file_high.name
        write_file(df_result, output_file)
        print(f"✅ Saved: {output_file.name}")
        
        processed.add(combo_key)
    
    print(f"\n{'='*60}")
    print("🏁 Process completed")
    print(f"{'='*60}")


if __name__ == "__main__":
    process_files()
