import ccxt

BASE_URL = "https://api.bitget.com"

# -----------------------------
# BITGET CONFIG
# -----------------------------
BITGET_API_KEY_00    = "bg_ff8f1f849c60b830017a546477ea9d65"
BITGET_API_SECRET_00 = "5bfac08cb3d1904a0b5c088cb2fba014b40bb4751fd83c173c5a735fc6745854"
BITGET_API_PASS_00   = "Cuentaprincipal86"

BITGET_API_KEY_01    = "bg_afdcb9221ad98efb3b0b7bdd4c236338"
BITGET_API_SECRET_01 = "0c4214cbfccfb648f841b43ca5d68531c8fb44b75ab271fdd222da9a74ee413f"
BITGET_API_PASS_01   = "Cryptobitget86"

BITGET_API_KEY_03    = "bg_32bb96c54d766f315346ac7ca3933fa1"
BITGET_API_SECRET_03 = "5ac75f6e920f045f2bba886e4d22fe2376151681e93f2f0c4a29003afcb091e7"
BITGET_API_PASS_03   = "Cryptobitget863"

BITGET_API_KEY_02    = "bg_bf379251452ff6d79fc67bbfb9594356"
BITGET_API_SECRET_02 = "39c09a8605ad552b62bb6969d06ec27f390b2625e74b31de70f83dcd2aa1b1b3"
BITGET_API_PASS_02   = "Cryptobitget865"

BITGET_API_KEY_04    = "bg_26b5f8a419d7ae77e521a905c334c963"
BITGET_API_SECRET_04 = "2ad14979254ef56caae32646f19c31c0912639e2afb054db831f72d94a60549f"
BITGET_API_PASS_04   = "Cryptobitget864"

BITGET_API_KEY_05    = "bg_a41dab207256d0cf889f0f548bcaa843"
BITGET_API_SECRET_05 = "b7e502c13a58547b87d8072ca8ae86492cb7d0daf5fd78dabfd746c581ff792d"
BITGET_API_PASS_05   = "Cryptobitget866"

# -----------------------------
# CONNECTION AND SYMBOLS
# -----------------------------
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

def connect_bitget_05():
    exchange = ccxt.bitget({
        'apiKey': BITGET_API_KEY_05,
        'secret': BITGET_API_SECRET_05,
        'password': BITGET_API_PASS_05,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',      
            'subAccount': 'entropia'    
        }
    })
    exchange.load_markets()
    return exchange