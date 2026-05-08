import logging
import os
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "shared", "broker_api")))

import step1_extraction

config = {
    "start_date":       "2021-01-01",
    "end_date":         None,
    "timeframe":        "15m",
    "selected_symbols": ["AAVEUSDT"],
    #"selected_symbols": ["BTCUSDT"],
    "raw_dir":          "/home/javi/projects/quant/quant_b/bitget/data_pipeline/data/01_raw_test3",
}
step1_extraction.run(config)