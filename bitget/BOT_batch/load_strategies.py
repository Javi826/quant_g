import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "BOT_trading")))

import pandas as pd
from config.strategies_E1 import STRATEGIES

# =============================================================================
# CONFIGURATION
# =============================================================================
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "strategies.csv")

# Batch-managed columns — added on first run, updated by each batch execution
BATCH_COLUMNS = {
    "approved":       None,
    "last_run":       None,
    "bt_netgain_pct": None,
    "bt_r2":          None,
    "prob_negative":  None,
}

# =============================================================================
# GENERATE CSV
# =============================================================================
df = pd.DataFrame(STRATEGIES)

for col, default in BATCH_COLUMNS.items():
    df[col] = default

df.to_csv(OUTPUT_PATH, index=False)
print(f"✅ strategies.csv generated → {OUTPUT_PATH}")
print(f"   Rows: {len(df)}")
print(df[["id", "name", "timeframe", "direction", "active"]].to_string(index=False))