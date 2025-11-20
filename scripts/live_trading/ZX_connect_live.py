import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import os
import time
import requests
import json
import hashlib
import base64
import hmac
from urllib.parse import urlencode
from typing import Dict, Any, List
from utils.ZZ_connect import BITGET_API_KEY_TT, BITGET_API_SECRET_TT, BITGET_API_PASS_TT
from utils.ZZ_connect import BITGET_API_KEY_01, BITGET_API_SECRET_01, BITGET_API_PASS_01
from utils.ZZ_connect import BITGET_API_KEY_03, BITGET_API_SECRET_03, BITGET_API_PASS_03
from utils.ZZ_connect import BITGET_API_KEY_02, BITGET_API_SECRET_02, BITGET_API_PASS_02
from utils.ZZ_connect import BITGET_API_KEY_04, BITGET_API_SECRET_04, BITGET_API_PASS_04
from utils.ZZ_connect import BITGET_API_KEY_05, BITGET_API_SECRET_05, BITGET_API_PASS_05


BASE_URL     = "https://api.bitget.com"
PRODUCT_TYPE = 'usdt-futures'

def _now_ms():
    return str(int(time.time() * 1000))

def _body_to_str(body):
    return json.dumps(body, separators=(",", ":"), ensure_ascii=False) if body else ""

# -----------------------------
# TT
# -----------------------------
def sign_request_TT(timestamp: str, method: str, path: str, query_string: str, body_str: str) -> str:
    to_sign = timestamp + method.upper() + path
    if query_string:
        to_sign += "?" + query_string
    to_sign += body_str
    digest = hmac.new(BITGET_API_SECRET_TT.encode("utf-8"), to_sign.encode("utf-8"), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()

def get_usdt_balance_TT(exchange):
    balance = exchange.fetch_balance()
    return balance['free']['USDT']

def get_open_positions_TT(product_type: str = "USDT-FUTURES") -> List[Dict[str, Any]]:
    endpoint = "/api/v2/mix/position/all-position"
    params   = {"productType": product_type}
    response = make_get_TT(endpoint, params)
    return response.get("data", [])

def send_request_TT(method, path, params=None, body=None):
    ts = _now_ms()
    query_string = urlencode(params) if params else ""
    body_str = _body_to_str(body)
    sign = sign_request_TT(ts, method, path, query_string, body_str)
    headers = {
        "ACCESS-KEY": BITGET_API_KEY_TT,
        "ACCESS-SIGN": sign,
        "ACCESS-TIMESTAMP": ts,
        "ACCESS-PASSPHRASE": BITGET_API_PASS_TT,
        "Content-Type": "application/json"
    }
    url = BASE_URL + path + (f"?{query_string}" if query_string else "")
    try:
        if method.upper() != "GET":
            r = requests.post(url, headers=headers, data=body_str.encode('utf-8'), timeout=15)
        else:
            r = requests.get(url, headers=headers, timeout=15)
        ct = r.headers.get("Content-Type", "")
        return r.status_code, r.json() if ct.startswith("application/json") else r.text
    except Exception as e:
        return 0, {"error": str(e)}

def make_get_TT(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    qs = "&".join(f"{k}={v}" for k, v in params.items() if v not in [None, ""])
    url = BASE_URL + endpoint + (f"?{qs}" if qs else "")
    timestamp = str(int(time.time() * 1000))
    sign = sign_request_TT(timestamp, "GET", endpoint, qs, "")
    headers = {
        "ACCESS-KEY": BITGET_API_KEY_TT,
        "ACCESS-SIGN": sign,
        "ACCESS-PASSPHRASE": BITGET_API_PASS_TT,
        "ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()

def make_post_TT(endpoint: str, body: Dict[str, Any]) -> Dict[str, Any]:
    body_str = json.dumps(body, separators=(',', ':'))
    url = BASE_URL + endpoint
    timestamp = str(int(time.time() * 1000))
    sign = sign_request_TT(timestamp, "POST", endpoint, "", body_str)
    headers = {
        "ACCESS-KEY": BITGET_API_KEY_TT,
        "ACCESS-SIGN": sign,
        "ACCESS-PASSPHRASE": BITGET_API_PASS_TT,
        "ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    resp = requests.post(url, headers=headers, data=body_str, timeout=30)
    resp.raise_for_status()
    return resp.json()

# =============================================================================
# 01 
# =============================================================================
def sign_request_01(timestamp, method, path, query_string, body_str):
    to_sign = timestamp + method.upper() + path
    if query_string:
        to_sign += "?" + query_string
    to_sign += body_str
    digest = hmac.new(BITGET_API_SECRET_01.encode('utf-8'), to_sign.encode('utf-8'), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()

def get_usdt_balance_01(exchange):
    balance = exchange.fetch_balance()
    return balance['free']['USDT']

def get_open_positions_01(product_type: str = "USDT-FUTURES") -> List[Dict[str, Any]]:
    endpoint = "/api/v2/mix/position/all-position"
    params = {"productType": product_type}
    response = make_get_01(endpoint, params)
    return response.get("data", [])

def make_get_01(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    qs = "&".join(f"{k}={v}" for k, v in params.items() if v not in [None, ""])
    url = BASE_URL + endpoint + (f"?{qs}" if qs else "")
    timestamp = str(int(time.time() * 1000))
    sign = sign_request_01(timestamp, "GET", endpoint, qs, "")
    headers = {
        "ACCESS-KEY": BITGET_API_KEY_01,
        "ACCESS-SIGN": sign,
        "ACCESS-PASSPHRASE": BITGET_API_PASS_01,
        "ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def send_request_01(method, path, params=None, body=None):
    ts = _now_ms()
    query_string = urlencode(params) if params else ""
    body_str = _body_to_str(body)
    sign = sign_request_01(ts, method, path, query_string, body_str)
    headers = {
        "ACCESS-KEY": BITGET_API_KEY_01,
        "ACCESS-SIGN": sign,
        "ACCESS-TIMESTAMP": ts,
        "ACCESS-PASSPHRASE": BITGET_API_PASS_01,
        "Content-Type": "application/json"
    }
    url = BASE_URL + path + (f"?{query_string}" if query_string else "")
    try:
        if method.upper() != "GET":
            r = requests.post(url, headers=headers, data=body_str.encode('utf-8'), timeout=15)
        else:
            r = requests.get(url, headers=headers, timeout=15)
        ct = r.headers.get("Content-Type", "")
        return r.status_code, r.json() if ct.startswith("application/json") else r.text
    except Exception as e:
        return 0, {"error": str(e)}
    
# =============================================================================
# 03 
# =============================================================================
def sign_request_03(timestamp, method, path, query_string, body_str):
    to_sign = timestamp + method.upper() + path
    if query_string:
        to_sign += "?" + query_string
    to_sign += body_str
    digest = hmac.new(BITGET_API_SECRET_03.encode('utf-8'), to_sign.encode('utf-8'), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()

def get_usdt_balance_03(exchange):
    balance = exchange.fetch_balance()
    return balance['free']['USDT']

def get_open_positions_03(product_type: str = "USDT-FUTURES") -> List[Dict[str, Any]]:
    endpoint = "/api/v2/mix/position/all-position"
    params = {"productType": product_type}
    response = make_get_03(endpoint, params)
    return response.get("data", [])

def make_get_03(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    qs = "&".join(f"{k}={v}" for k, v in params.items() if v not in [None, ""])
    url = BASE_URL + endpoint + (f"?{qs}" if qs else "")
    timestamp = str(int(time.time() * 1000))
    sign = sign_request_03(timestamp, "GET", endpoint, qs, "")
    headers = {
        "ACCESS-KEY": BITGET_API_KEY_03,
        "ACCESS-SIGN": sign,
        "ACCESS-PASSPHRASE": BITGET_API_PASS_03,
        "ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def send_request_03(method, path, params=None, body=None):
    ts = _now_ms()
    query_string = urlencode(params) if params else ""
    body_str = _body_to_str(body)
    sign = sign_request_03(ts, method, path, query_string, body_str)
    headers = {
        "ACCESS-KEY": BITGET_API_KEY_03,
        "ACCESS-SIGN": sign,
        "ACCESS-TIMESTAMP": ts,
        "ACCESS-PASSPHRASE": BITGET_API_PASS_03,
        "Content-Type": "application/json"
    }
    url = BASE_URL + path + (f"?{query_string}" if query_string else "")
    try:
        if method.upper() != "GET":
            r = requests.post(url, headers=headers, data=body_str.encode('utf-8'), timeout=15)
        else:
            r = requests.get(url, headers=headers, timeout=15)
        ct = r.headers.get("Content-Type", "")
        return r.status_code, r.json() if ct.startswith("application/json") else r.text
    except Exception as e:
        return 0, {"error": str(e)}
    
# =============================================================================
# 02 
# =============================================================================
def sign_request_02(timestamp, method, path, query_string, body_str):
    to_sign = timestamp + method.upper() + path
    if query_string:
        to_sign += "?" + query_string
    to_sign += body_str
    digest = hmac.new(BITGET_API_SECRET_02.encode('utf-8'), to_sign.encode('utf-8'), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()

def get_usdt_balance_02(exchange):
    balance = exchange.fetch_balance()
    return balance['free']['USDT']

def get_open_positions_02(product_type: str = "USDT-FUTURES") -> List[Dict[str, Any]]:
    endpoint = "/api/v2/mix/position/all-position"
    params = {"productType": product_type}
    response = make_get_02(endpoint, params)
    return response.get("data", [])

def make_get_02(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    qs = "&".join(f"{k}={v}" for k, v in params.items() if v not in [None, ""])
    url = BASE_URL + endpoint + (f"?{qs}" if qs else "")
    timestamp = str(int(time.time() * 1000))
    sign = sign_request_02(timestamp, "GET", endpoint, qs, "")
    headers = {
        "ACCESS-KEY": BITGET_API_KEY_02,
        "ACCESS-SIGN": sign,
        "ACCESS-PASSPHRASE": BITGET_API_PASS_02,
        "ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def send_request_02(method, path, params=None, body=None):
    ts = _now_ms()
    query_string = urlencode(params) if params else ""
    body_str = _body_to_str(body)
    sign = sign_request_02(ts, method, path, query_string, body_str)
    headers = {
        "ACCESS-KEY": BITGET_API_KEY_02,
        "ACCESS-SIGN": sign,
        "ACCESS-TIMESTAMP": ts,
        "ACCESS-PASSPHRASE": BITGET_API_PASS_02,
        "Content-Type": "application/json"
    }
    url = BASE_URL + path + (f"?{query_string}" if query_string else "")
    try:
        if method.upper() != "GET":
            r = requests.post(url, headers=headers, data=body_str.encode('utf-8'), timeout=15)
        else:
            r = requests.get(url, headers=headers, timeout=15)
        ct = r.headers.get("Content-Type", "")
        return r.status_code, r.json() if ct.startswith("application/json") else r.text
    except Exception as e:
        return 0, {"error": str(e)}
    
# =============================================================================
# 04 
# =============================================================================
def sign_request_04(timestamp, method, path, query_string, body_str):
    to_sign = timestamp + method.upper() + path
    if query_string:
        to_sign += "?" + query_string
    to_sign += body_str
    digest = hmac.new(BITGET_API_SECRET_04.encode('utf-8'), to_sign.encode('utf-8'), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()

def get_usdt_balance_04(exchange):
    balance = exchange.fetch_balance()
    return balance['free']['USDT']

def get_open_positions_04(product_type: str = "USDT-FUTURES") -> List[Dict[str, Any]]:
    endpoint = "/api/v2/mix/position/all-position"
    params = {"productType": product_type}
    response = make_get_04(endpoint, params)
    return response.get("data", [])

def make_get_04(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    qs = "&".join(f"{k}={v}" for k, v in params.items() if v not in [None, ""])
    url = BASE_URL + endpoint + (f"?{qs}" if qs else "")
    timestamp = str(int(time.time() * 1000))
    sign = sign_request_04(timestamp, "GET", endpoint, qs, "")
    headers = {
        "ACCESS-KEY": BITGET_API_KEY_04,
        "ACCESS-SIGN": sign,
        "ACCESS-PASSPHRASE": BITGET_API_PASS_04,
        "ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def send_request_04(method, path, params=None, body=None):
    ts = _now_ms()
    query_string = urlencode(params) if params else ""
    body_str = _body_to_str(body)
    sign = sign_request_04(ts, method, path, query_string, body_str)
    headers = {
        "ACCESS-KEY": BITGET_API_KEY_04,
        "ACCESS-SIGN": sign,
        "ACCESS-TIMESTAMP": ts,
        "ACCESS-PASSPHRASE": BITGET_API_PASS_04,
        "Content-Type": "application/json"
    }
    url = BASE_URL + path + (f"?{query_string}" if query_string else "")
    try:
        if method.upper() != "GET":
            r = requests.post(url, headers=headers, data=body_str.encode('utf-8'), timeout=15)
        else:
            r = requests.get(url, headers=headers, timeout=15)
        ct = r.headers.get("Content-Type", "")
        return r.status_code, r.json() if ct.startswith("application/json") else r.text
    except Exception as e:
        return 0, {"error": str(e)}
    
    
# =============================================================================
# 05
# =============================================================================
def sign_request_05(timestamp, method, path, query_string, body_str):
    to_sign = timestamp + method.upper() + path
    if query_string:
        to_sign += "?" + query_string
    to_sign += body_str
    digest = hmac.new(BITGET_API_SECRET_05.encode('utf-8'), to_sign.encode('utf-8'), hashlib.sha256).digest()
    return base64.b64encode(digest).decode()

def get_usdt_balance_05(exchange):
    balance = exchange.fetch_balance()
    return balance['free']['USDT']

def get_open_positions_05(product_type: str = "USDT-FUTURES") -> List[Dict[str, Any]]:
    endpoint = "/api/v2/mix/position/all-position"
    params = {"productType": product_type}
    response = make_get_05(endpoint, params)
    return response.get("data", [])

def make_get_05(endpoint: str, params: Dict[str, Any]) -> Dict[str, Any]:
    qs = "&".join(f"{k}={v}" for k, v in params.items() if v not in [None, ""])
    url = BASE_URL + endpoint + (f"?{qs}" if qs else "")
    timestamp = str(int(time.time() * 1000))
    sign = sign_request_05(timestamp, "GET", endpoint, qs, "")
    headers = {
        "ACCESS-KEY": BITGET_API_KEY_05,
        "ACCESS-SIGN": sign,
        "ACCESS-PASSPHRASE": BITGET_API_PASS_05,
        "ACCESS-TIMESTAMP": timestamp,
        "Content-Type": "application/json"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()
    return resp.json()


def send_request_05(method, path, params=None, body=None):
    ts = _now_ms()
    query_string = urlencode(params) if params else ""
    body_str = _body_to_str(body)
    sign = sign_request_05(ts, method, path, query_string, body_str)
    headers = {
        "ACCESS-KEY": BITGET_API_KEY_05,
        "ACCESS-SIGN": sign,
        "ACCESS-TIMESTAMP": ts,
        "ACCESS-PASSPHRASE": BITGET_API_PASS_05,
        "Content-Type": "application/json"
    }
    url = BASE_URL + path + (f"?{query_string}" if query_string else "")
    try:
        if method.upper() != "GET":
            r = requests.post(url, headers=headers, data=body_str.encode('utf-8'), timeout=15)
        else:
            r = requests.get(url, headers=headers, timeout=15)
        ct = r.headers.get("Content-Type", "")
        return r.status_code, r.json() if ct.startswith("application/json") else r.text
    except Exception as e:
        return 0, {"error": str(e)}