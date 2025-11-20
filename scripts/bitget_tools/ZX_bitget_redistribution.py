import os
import sys
import uuid
from typing import Dict, Any, List, Tuple
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from live_trading.ZX_connect_live import  make_get_TT,make_post_TT

BASE_URL = "https://api.bitget.com"
PRODUCT_TYPE = "USDT-FUTURES"

# -----------------------------
# EXTRACT FREE MARGIN
# -----------------------------
def extract_available_from_asset(asset: Dict[str, Any]) -> float:
    """
    Detects possible field names that could represent 'free margin' in USDT.
    Returns 0.0 if none is found.
    """
    candidates = [
        "available", "availableBalance", "availableBalanceUsdt",
        "availableMargin", "availableEquity", "free", "freeMargin",
        "usdtAvailable", "usdtEquity"
    ]
    for k in candidates:
        v = asset.get(k)
        if v is None:
            continue
        try:
            return float(v or 0)
        except (TypeError, ValueError):
            continue
    return 0.0

def get_subaccounts_free_margin(product_type: str = PRODUCT_TYPE) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Calls /api/v2/mix/account/sub-account-assets and extracts free USDT margin.
    Returns (total_free_usdt, [ {userId, available_usdt}, ... ])
    """
    endpoint = "/api/v2/mix/account/sub-account-assets"
    params = {"productType": product_type}
    resp = make_get_TT(endpoint, params)
    data = resp.get("data", []) or []

    balances = []
    total = 0.0

    for sub in data:
        user_id = sub.get("userId")
        assets = sub.get("assetList", []) or []
        for asset in assets:
            if asset.get("marginCoin", "").upper() == "USDT" or asset.get("coin", "").upper() == "USDT":
                available_usdt = extract_available_from_asset(asset)
                if available_usdt == 0.0:
                    try:
                        available_usdt = float(asset.get("usdtEquity", 0) or 0)
                    except Exception:
                        available_usdt = 0.0
                total += available_usdt
                balances.append({"userId": str(user_id), "available_usdt": float(round(available_usdt, 8))})
                break

    return float(round(total, 8)), balances

# -----------------------------
# PLAN REDISTRIBUTION
# -----------------------------
def plan_redistribution(balances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Given balances = [ {userId, available_usdt}, ... ]
    Returns a list of transfers: [{fromUserId,toUserId,amount}, ...]
    """
    EPS = 1e-8
    items = [{"userId": b["userId"], "amt": float(b["available_usdt"])} for b in balances]
    n = len(items)
    if n == 0:
        return []

    total = sum(i["amt"] for i in items)
    target = total / n

    diffs = []
    for i in items:
        diff = i["amt"] - target
        if abs(diff) < EPS:
            diff = 0.0
        diffs.append({"userId": i["userId"], "diff": float(round(diff, 8))})

    deficits = []
    surpluses = []
    for d in diffs:
        if d["diff"] < 0:
            deficits.append({"userId": d["userId"], "need": -d["diff"]})
        elif d["diff"] > 0:
            surpluses.append({"userId": d["userId"], "have": d["diff"]})

    deficits.sort(key=lambda x: x["need"])
    surpluses.sort(key=lambda x: x["have"], reverse=True)

    transfers = []
    i, j = 0, 0
    while i < len(surpluses) and j < len(deficits):
        s = surpluses[i]
        d = deficits[j]
        amount = min(s["have"], d["need"])
        if amount > 1e-6:
            transfers.append({
                "fromUserId": s["userId"],
                "toUserId": d["userId"],
                "amount": float(round(amount, 8))
            })
        s["have"] = float(round(s["have"] - amount, 8))
        d["need"] = float(round(d["need"] - amount, 8))
        if s["have"] <= 1e-8:
            i += 1
        if d["need"] <= 1e-8:
            j += 1

    return transfers

# -----------------------------
# EXECUTE TRANSFERS
# -----------------------------
def execute_transfer(from_user: str, to_user: str, amount: float,
                     fromType: str = "usdt_futures", toType: str = "usdt_futures", coin: str = "USDT") -> Dict[str, Any]:
    """
    Executes a transfer between subaccounts using the parent's API.
    Calls: /api/v2/spot/wallet/subaccount-transfer
    """
    endpoint = "/api/v2/spot/wallet/subaccount-transfer"
    body = {
        "fromUserId": str(from_user),
        "toUserId": str(to_user),
        "fromType": fromType,
        "toType": toType,
        "amount": str(amount),
        "coin": coin,
        "clientOid": str(uuid.uuid4())
    }
    return make_post_TT(endpoint, body)

# -----------------------------
# MAIN REDISTRIBUTION LOGIC
# -----------------------------
def redistribute_all_equal(dry_run: bool = True):
    total, balances = get_subaccounts_free_margin(PRODUCT_TYPE)
    n = len(balances)
    print("🔎 Subaccounts detected:", n)
    for b in balances:
        print(f" - {b['userId']}: {b['available_usdt']:.8f} USDT")

    print(f"\n💼 Total free USDT margin: {total:.8f}")
    if n == 0:
        print("No subaccounts found. Aborting.")
        return

    transfers = plan_redistribution(balances)
    if not transfers:
        print("\n✅ All subaccounts are already balanced or differences are negligible.")
        return

    print("\n📋 Transfer plan to equalize capital:")
    agg = 0.0
    for t in transfers:
        print(f" - Send {t['amount']:.8f} USDT: {t['fromUserId']} -> {t['toUserId']}")
        agg += t["amount"]
    print(f"🔁 Total amount moved: {agg:.8f} USDT")

    if dry_run:
        print("\n⏸ dry_run=True -> Transfers NOT executed. Set dry_run=False to execute.")
        return transfers

    print("\n🚀 Executing transfers...")
    results = []
    for t in transfers:
        try:
            r = execute_transfer(t["fromUserId"], t["toUserId"], t["amount"])
            results.append({"transfer": t, "result": r})
            print(f"  ✅ {t['fromUserId']} -> {t['toUserId']}: {t['amount']:.8f} OK")
        except Exception as e:
            print(f"  ❌ Error sending {t['amount']:.8f} from {t['fromUserId']} to {t['toUserId']}: {e}")
            results.append({"transfer": t, "error": str(e)})

    return results

if __name__ == "__main__":
    redistribute_all_equal(dry_run=True)
