import requests
import time
from datetime import datetime, timezone

BASE_URL    = "https://api.bitget.com"
GRANULARITY = "1Dutc"
LIMIT       = 200
SLEEP       = 0.06
MAX_ITERS   = 500
PRODUCT_TYPE = "USDT-FUTURES"


# =============================================================================
# SYMBOL FETCHERS
# =============================================================================

def fetch_rwa_futures_symbols() -> list[str]:
    url = f"{BASE_URL}/api/v2/mix/market/contracts"
    r   = requests.get(url, params={"productType": PRODUCT_TYPE}, timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])
    return sorted([
        item["symbol"] for item in data
        if item.get("isRwa") == "YES"
    ])


def fetch_rwa_spot_symbols() -> list[str]:
    url = f"{BASE_URL}/api/v2/spot/public/symbols"
    r   = requests.get(url, timeout=10)
    r.raise_for_status()
    data = r.json().get("data", [])

    rwa_prefixes = {
        "TSLAON", "NVDAON", "AAPLON", "GOOGLON", "AMZNON", "MSFTON",
        "AMDON", "SLVON", "IAUON", "QQQON", "IVVON", "SPYON", "ITOTON",
        "IWMON", "METAON",
    }

    return sorted([
        s["symbol"] for s in data
        if s.get("quoteCoin") == "USDT"
        and s.get("status") == "online"
        and (
            s["symbol"] == "XAUTUSDT"
            or any(s["symbol"].startswith(p) for p in rwa_prefixes)
        )
    ])


# =============================================================================
# EARLIEST CANDLE
# =============================================================================

def _find_earliest(url: str, params: dict) -> str | None:
    end            = int(datetime.now(tz=timezone.utc).timestamp() * 1000)
    earliest_found = None

    for _ in range(MAX_ITERS):
        try:
            r    = requests.get(url, params={**params, "endTime": str(end), "limit": LIMIT}, timeout=10)
            data = r.json().get("data", [])
        except Exception:
            return None
        time.sleep(SLEEP)

        if not data:
            break

        timestamps = [int(row[0]) for row in data if row]
        min_ts     = min(timestamps)

        if earliest_found is None or min_ts < earliest_found:
            earliest_found = min_ts

        new_end = min_ts - 1
        if new_end >= end:
            break
        end = new_end

    if earliest_found is None:
        return None
    return datetime.fromtimestamp(earliest_found / 1000, tz=timezone.utc).strftime("%Y-%m-%d")


def find_earliest_futures(symbol: str) -> str | None:
    return _find_earliest(
        f"{BASE_URL}/api/v2/mix/market/history-candles",
        {"symbol": symbol, "granularity": GRANULARITY, "productType": PRODUCT_TYPE},
    )


def find_earliest_spot(symbol: str) -> str | None:
    return _find_earliest(
        f"{BASE_URL}/api/v2/spot/market/candles",
        {"symbol": symbol, "granularity": GRANULARITY},
    )


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Fetching RWA futures symbols...")
    futures_symbols = fetch_rwa_futures_symbols()
    print(f"  {len(futures_symbols)} futures RWA symbols\n")

    print("Fetching RWA spot symbols...")
    spot_symbols = fetch_rwa_spot_symbols()
    print(f"  {len(spot_symbols)} spot RWA symbols\n")

    all_base = sorted(set(
        [s.replace("USDT", "") for s in futures_symbols] +
        [s.replace("USDT", "").replace("ON", "") for s in spot_symbols]
    ))

    futures_map = {s.replace("USDT", ""): s for s in futures_symbols}
    spot_map    = {s.replace("USDT", "").replace("ON", ""): s for s in spot_symbols}
    spot_map["XAUT"] = "XAUTUSDT"

    print(f"{'asset':<12} {'futures_symbol':<20} {'earliest_futures':<20} {'spot_symbol':<20} {'earliest_spot'}")
    print("-" * 90)

    for base in all_base:
        fut_sym  = futures_map.get(base)
        spot_sym = spot_map.get(base)

        fut_date  = find_earliest_futures(fut_sym)  if fut_sym  else "N/A"
        spot_date = find_earliest_spot(spot_sym)    if spot_sym else "N/A"

        print(f"{base:<12} {(fut_sym or 'N/A'):<20} {(fut_date or 'N/A'):<20} {(spot_sym or 'N/A'):<20} {spot_date or 'N/A'}")


if __name__ == "__main__":
    main()