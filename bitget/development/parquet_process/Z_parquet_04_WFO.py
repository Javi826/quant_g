import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime, timedelta

# -----------------------------
# CONFIGURATION PARAMETERS
# -----------------------------
BASE_DIR          = Path(__file__).resolve().parent.parent  
input_folder      = BASE_DIR / "data" / "crypto_2022_IS"
output_folder_is  = BASE_DIR / "data" / "crypto_2022_OOS"
output_folder_oos = BASE_DIR / "data" / "crypto_2022_NNN"

# Date range for IN-SAMPLE (IS)
IS_START = "2022-01-01"
IS_END   = "2022-12-31"

# Automatic OOS: from IS_END until the end of the file
# OOS_START = IS_END
# OOS_END = end of file

output_folder_is.mkdir(exist_ok=True, parents=True)
output_folder_oos.mkdir(exist_ok=True, parents=True)


def read_file(filepath):
    """Reads parquet or xlsx file and returns a DataFrame."""
    try:
        if filepath.suffix == ".parquet":
            df = pd.read_parquet(filepath)
        elif filepath.suffix == ".xlsx":
            df = pd.read_excel(filepath)
        else:
            return None
        return df
    except Exception as e:
        print(f"❌ Error reading {filepath.name}: {e}")
        return None


def write_file(df, filepath):
    """Writes DataFrame to parquet or xlsx with timestamp as index."""
    try:
        # Ensure 'timestamp' column exists
        if 'timestamp' in df.columns:
            # Convert to datetime if not already
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            # Set timestamp as index
            df.set_index('timestamp', inplace=True)
        
        # Save according to extension
        if filepath.suffix == ".parquet":
            df.to_parquet(filepath, index=True)
        elif filepath.suffix == ".xlsx":
            df.to_excel(filepath, index=True)
    except Exception as e:
        print(f"❌ Error writing {filepath.name}: {e}")


def split_is_oos(df, is_start, is_end):
    """
    Splits DataFrame into two:
    - IS: from is_start to is_end (exclusive)
    - OOS: from is_end to the end
    """
    # Identify timestamp column
    if 'timestamp' in df.columns:
        time_col = 'timestamp'
    elif df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()
        time_col = df.columns[0]
    else:
        # Assume first column is timestamp
        time_col = df.columns[0]
    
    # Convert to datetime if necessary
    df[time_col] = pd.to_datetime(df[time_col])
    
    # Convert filter dates
    start = pd.to_datetime(is_start)
    end = pd.to_datetime(is_end)
    
    # Split into IS and OOS
    mask_is = (df[time_col] >= start) & (df[time_col] < end)
    mask_oos = (df[time_col] >= end)
    
    df_is = df[mask_is].copy()
    df_oos = df[mask_oos].copy()
    
    return df_is, df_oos


def process_files():
    """Processes all files in the input folder."""
    
    # Get all files
    files = list(input_folder.glob("*.parquet")) + list(input_folder.glob("*.xlsx"))
    
    if len(files) == 0:
        print(f"❌ No files found in {input_folder}")
        return
    
    # Automatically calculate OOS_START
    oos_start = pd.to_datetime(IS_END)
    
    print(f"\n{'='*60}")
    print(f"IS / OOS SPLIT")
    print(f"{'='*60}")
    print(f"\n📅 IN-SAMPLE (IS):")
    print(f"   Start: {IS_START}")
    print(f"   End:   {IS_END}")
    print(f"\n📅 OUT-OF-SAMPLE (OOS):")
    print(f"   Start: {IS_END}")
    print(f"   End:   [end of file]")
    print(f"\n📁 Files to process: {len(files)}\n")
    
    processed = 0
    skipped_is = 0
    skipped_oos = 0
    errors = 0
    
    for file in tqdm(files, desc="Processing files"):
        # Read file
        df = read_file(file)
        
        if df is None:
            errors += 1
            continue
        
        # Split into IS and OOS
        try:
            df_is, df_oos = split_is_oos(df, IS_START, IS_END)
            
            has_is = len(df_is) > 0
            has_oos = len(df_oos) > 0
            
            # Save IS
            if has_is:
                output_file_is = output_folder_is / file.name
                write_file(df_is, output_file_is)
            else:
                skipped_is += 1
            
            # Save OOS
            if has_oos:
                output_file_oos = output_folder_oos / file.name
                write_file(df_oos, output_file_oos)
            else:
                skipped_oos += 1
            
            if has_is or has_oos:
                processed += 1
                status_is = f"IS: {len(df_is)}" if has_is else "IS: 0 (skipped)"
                status_oos = f"OOS: {len(df_oos)}" if has_oos else "OOS: 0 (skipped)"
                tqdm.write(f"✅ {file.name}: {status_is}, {status_oos}")
            else:
                tqdm.write(f"⚠️  {file.name}: No data in either range")
            
        except Exception as e:
            errors += 1
            tqdm.write(f"❌ Error processing {file.name}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"✅ Files processed: {processed}")
    print(f"⚠️ Files with no IS data: {skipped_is}")
    print(f"⚠️ Files with no OOS data: {skipped_oos}")
    print(f"❌ Errors: {errors}")
    print(f"📁 Total: {len(files)}")
    print(f"\n📂 Output folders:")
    print(f"   IS:  {output_folder_is}")
    print(f"   OOS: {output_folder_oos}")
    print(f"\n{'='*60}")
    print("🏁 Process completed")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    process_files()
