#shared/shared_batch_regime/config_paths.py
import os

BITGET_ROOT    = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
SPLIT_BASE     = os.path.join(BITGET_ROOT, "data_pipeline", "data", "04_split", "expanding")
DATA_FOLDER_IS = os.path.join(SPLIT_BASE, "IS",  "crypto_2022-01_2026-08_IS")
#


