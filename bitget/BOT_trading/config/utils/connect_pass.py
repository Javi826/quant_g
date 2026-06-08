#BOT_trading/config/connect_pass.py

import ccxt

# -----------------------------
# BITGET CONFIG
# -----------------------------
BITGET_API_KEY_E1    = "bg_3acc0d69d70f0ebdfa5f0a924de5bed4"
BITGET_API_SECRET_E1 = "9bdc56066997b4da4e9a487cff49faa00449427b567b5bb2fe9c92d064465166"
BITGET_API_PASS_E1   = "Cuentaelite861"

BITGET_API_KEY_00    = "bg_ff8f1f849c60b830017a546477ea9d65"
BITGET_API_SECRET_00 = "5bfac08cb3d1904a0b5c088cb2fba014b40bb4751fd83c173c5a735fc6745854"
BITGET_API_PASS_00   = "Cuentaprincipal86"

BITGET_API_KEY_01    = "bg_afdcb9221ad98efb3b0b7bdd4c236338"
BITGET_API_SECRET_01 = "0c4214cbfccfb648f841b43ca5d68531c8fb44b75ab271fdd222da9a74ee413f"
BITGET_API_PASS_01   = "Cryptobitget86"



# -----------------------------
# CONNECTION AND SYMBOLS
# -----------------------------
def connect_bitget_E1():
    exchange = ccxt.bitget({
        'apiKey': BITGET_API_KEY_E1,
        'secret': BITGET_API_SECRET_E1,
        'password': BITGET_API_PASS_E1,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',      
            'subAccount': 'entropia'    
        }
    })
    exchange.load_markets()
    return exchange
    
def connect_bitget_00():
    exchange = ccxt.bitget({
        'apiKey': BITGET_API_KEY_00,
        'secret': BITGET_API_SECRET_00,
        'password': BITGET_API_PASS_00,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',      
            'subAccount': 'entropia'    
        }
    })
    exchange.load_markets()
    return exchange

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

