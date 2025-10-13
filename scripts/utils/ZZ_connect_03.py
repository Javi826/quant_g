import ccxt

# -----------------------------
# BITGET CONFIG
# -----------------------------
BITGET_API_KEY    = "bg_afdcb9221ad98efb3b0b7bdd4c236338"
BITGET_API_SECRET = "0c4214cbfccfb648f841b43ca5d68531c8fb44b75ab271fdd222da9a74ee413f"
BITGET_API_PASS   = "Cryptobitget86"

# -----------------------------
# CONNECTION AND SYMBOLS
# -----------------------------
def connect_bitget():
    exchange = ccxt.bitget({
        'apiKey': BITGET_API_KEY,
        'secret': BITGET_API_SECRET,
        'password': BITGET_API_PASS,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',      
            'subAccount': 'entropia'    
        }
    })
    exchange.load_markets()
    return exchange


