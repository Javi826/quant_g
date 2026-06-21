#shared/shared_batch_regime/config_paths.py
import os

#BITGET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
#SPLIT_BASE  = os.path.join(BITGET_ROOT, "data_pipeline", "data", "04_split_OLD", "expanding")

#DATA_FOLDER_IS   = os.path.join(SPLIT_BASE, "IS",  "crypto_2024-01_2025-05_IS")
#DATA_FOLDER_OOS1 = os.path.join(SPLIT_BASE, "OOS", "crypto_2025-05_2026-05_OOS")
#DATA_FOLDER_OOS2 = os.path.join(SPLIT_BASE, "OOS", "crypto_2022-01_2023-01_OOS")
#DATA_FOLDER_OOS3 = os.path.join(SPLIT_BASE, "OOS", "crypto_2023-01_2024-01_OOS")
#CRYPTO_FULL_DIR  = os.path.join(SPLIT_BASE, "IS",  "crypto_full_IS")
#DATA_FOLDER_OOS1 = os.path.join(SPLIT_BASE, "OOS", "crypto_2026-04_2026-06_OOS")


#shared/shared_batch_regime/config_paths.py
import os

BITGET_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
SPLIT_BASE  = os.path.join(BITGET_ROOT, "data_pipeline", "data", "04_split", "expanding")

DATA_FOLDER_IS   = os.path.join(SPLIT_BASE, "IS",  "crypto_2022-01_2025-06_IS")
DATA_FOLDER_OOS1 = os.path.join(SPLIT_BASE, "OOS", "crypto_2025-06_2026-06_OOS")
DATA_FOLDER_OOS2 = os.path.join(SPLIT_BASE, "OOS", "crypto_2022-01_2023-01_OOS")
DATA_FOLDER_OOS3 = os.path.join(SPLIT_BASE, "OOS", "crypto_2023-01_2024-01_OOS")
CRYPTO_FULL_DIR  = os.path.join(SPLIT_BASE, "IS",  "crypto_full_IS")
#DATA_FOLDER_OOS1 = os.path.join(SPLIT_BASE, "OOS", "crypto_2026-04_2026-06_OOS")