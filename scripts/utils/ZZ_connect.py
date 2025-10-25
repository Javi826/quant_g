import ccxt

BASE_URL = "https://api.bitget.com"
#ENTROPY
# -----------------------------
# BITGET CONFIG
# -----------------------------
BITGET_API_KEY_01    = "bg_afdcb9221ad98efb3b0b7bdd4c236338"
BITGET_API_SECRET_01 = "0c4214cbfccfb648f841b43ca5d68531c8fb44b75ab271fdd222da9a74ee413f"
BITGET_API_PASS_01   = "Cryptobitget86"

BITGET_API_KEY_03    = "bg_32bb96c54d766f315346ac7ca3933fa1"
BITGET_API_SECRET_03 = "5ac75f6e920f045f2bba886e4d22fe2376151681e93f2f0c4a29003afcb091e7"
BITGET_API_PASS_03   = "Cryptobitget863"

# -----------------------------
# CONNECTION AND SYMBOLS
# -----------------------------
def connect_bitget_01():
    exchange = ccxt.bitget({
        'apiKey': BITGET_API_KEY_01,
        'secret': BITGET_API_SECRET_01,
        'password': BITGET_API_PASS_01,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',      
            'subAccount': 'entropia'    
        }
    })
    exchange.load_markets()
    return exchange


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

