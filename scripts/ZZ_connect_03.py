import ccxt

# -----------------------------
# BITGET CONFIG
# -----------------------------


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


