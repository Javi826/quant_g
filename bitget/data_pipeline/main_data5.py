import integrity
import os

_base = os.path.dirname(os.path.abspath(__file__))
config = {
    "highlow_dir":        os.path.join(_base, "data", "03_highlow"),
    "timeframes_highlow": [["1Dutc","1H"],["6Hutc","1H"],["4H","1H"],["1H","15m"]],
}
integrity.run_highlow(config)