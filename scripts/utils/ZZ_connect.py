import ccxt

BASE_URL = "https://api.bitget.com"
#ENTROPY
# -----------------------------
# BITGET CONFIG
# -----------------------------


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

def connect_bitget_02():
    exchange = ccxt.bitget({
        'apiKey': BITGET_API_KEY_02,
        'secret': BITGET_API_SECRET_02,
        'password': BITGET_API_PASS_02,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',      
            'subAccount': 'entropia'    
        }
    })
    exchange.load_markets()
    return exchange

def connect_bitget_04():
    exchange = ccxt.bitget({
        'apiKey': BITGET_API_KEY_04,
        'secret': BITGET_API_SECRET_04,
        'password': BITGET_API_PASS_04,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',      
            'subAccount': 'entropia'    
        }
    })
    exchange.load_markets()
    return exchange