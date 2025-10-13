import ccxt
#ENTROPY
# -----------------------------
# BITGET CONFIG
# -----------------------------
BITGET_API_KEY_03    = "bg_afdcb9221ad98efb3b0b7bdd4c236338"
BITGET_API_SECRET_03 = "0c4214cbfccfb648f841b43ca5d68531c8fb44b75ab271fdd222da9a74ee413f"
BITGET_API_PASS_03   = "Cryptobitget86"

# -----------------------------
# CONNECTION AND SYMBOLS
# -----------------------------
def connect_bitget_03():
    exchange = ccxt.bitget({
        'apiKey': BITGET_API_KEY_03,
        'secret': BITGET_API_SECRET_03,
        'password': BITGET_API_PASS_03,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',      
            'subAccount': 'entropia'    
        }
    })
    exchange.load_markets()
    return exchange


