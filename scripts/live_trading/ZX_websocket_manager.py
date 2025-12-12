import websocket
import threading
import json
import time
import hmac
import base64
from decimal import Decimal
from collections import defaultdict
from utils.ZZ_connect import BITGET_API_KEY_02, BITGET_API_SECRET_02, BITGET_API_PASS_02

class BitgetWSManager:
    """Manager WebSocket para Bitget con keep-alive y reconexión robusta"""
    
    def __init__(self, api_key, secret_key, passphrase):
        self.api_key = api_key
        self.secret_key = secret_key
        self.passphrase = passphrase
        self.ws_public_url = "wss://ws.bitget.com/v2/ws/public"
        self.ws_private_url = "wss://ws.bitget.com/v2/ws/private"
        
        # Cache de datos
        self.ticker_cache = {}
        self.positions_cache = {}
        self.fills_cache = defaultdict(list)
        
        self.ws_public = None
        self.ws_private = None
        self.running = False
        self.subscribed_symbols = set()
        
        # Control de reconexión
        self.public_connected = False
        self.private_connected = False
        self.private_authenticated = False
        
    def start(self):
        """Inicia conexiones WebSocket"""
        self.running = True
        threading.Thread(target=self._run_public_ws, daemon=True).start()
        threading.Thread(target=self._run_private_ws, daemon=True).start()
        threading.Thread(target=self._ping_loop, daemon=True).start()
        time.sleep(3)
        print("✅ WebSocket manager started")
        
    def stop(self):
        self.running = False
        if self.ws_public:
            self.ws_public.close()
        if self.ws_private:
            self.ws_private.close()
    
    def _ping_loop(self):
        """Envía ping cada 20 segundos para mantener la conexión viva"""
        while self.running:
            try:
                # Ping público
                if self.ws_public and self.ws_public.sock and self.public_connected:
                    try:
                        self.ws_public.send('ping')
                    except Exception as e:
                        print(f"🔔 Ping public failed: {e}")
                
                # Ping privado
                if self.ws_private and self.ws_private.sock and self.private_connected:
                    try:
                        self.ws_private.send('ping')
                    except Exception as e:
                        print(f"🔔 Ping private failed: {e}")
                
                time.sleep(20)
            except Exception as e:
                print(f"🔔 Ping loop error: {e}")
                time.sleep(5)
            
    def _generate_signature(self, timestamp):
        message = f"{timestamp}GET/user/verify"
        mac = hmac.new(
            self.secret_key.encode(),
            message.encode(),
            digestmod='sha256'
        )
        return base64.b64encode(mac.digest()).decode()
        
    def _run_public_ws(self):
        def on_message(ws, message):
            try:
                # Ignorar pong
                if message == 'pong':
                    return
                
                data = json.loads(message)
                
                # Manejar suscripción exitosa
                if data.get('event') == 'subscribe':
                    return
                
                arg = data.get('arg', {})
                
                if arg.get('channel') == 'ticker':
                    for item in data.get('data', []):
                        symbol = item.get('instId')
                        if symbol:
                            self.ticker_cache[symbol] = item
            except json.JSONDecodeError:
                pass
            except Exception as e:
                print(f"❌ WS Public message error: {e}")
                
        def on_error(ws, error):
            if self.running and str(error) != "Connection to remote host was lost.":
                print(f"🔔 WS Public error: {error}")
            
        def on_close(ws, close_status_code, close_msg):
            self.public_connected = False
            if self.running:
                print("🔄 Reconnecting WS Public in 5s...")
                time.sleep(5)
                self._run_public_ws()
                
        def on_open(ws):
            self.public_connected = True
            print("🔌 WS Public connected")
            
            # Re-suscribir símbolos previos
            if self.subscribed_symbols:
                for symbol in list(self.subscribed_symbols):
                    self._subscribe_ticker_internal(symbol)
            
        self.ws_public = websocket.WebSocketApp(
            self.ws_public_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        self.ws_public.run_forever(
            ping_interval=20,
            ping_timeout=10
        )
        
    def _run_private_ws(self):
        def on_message(ws, message):
            try:
                # Ignorar pong
                if message == 'pong':
                    return
                
                data = json.loads(message)
                
                # Manejar login
                if data.get('event') == 'login':
                    if data.get('code') == '0':
                        self.private_authenticated = True
                        print("🔐 WS Private authenticated")
                        
                        # Suscribirse tras login exitoso
                        subscribe_msg = {
                            "op": "subscribe",
                            "args": [
                                {"instType": "USDT-FUTURES", "channel": "positions", "instId": "default"},
                                {"instType": "USDT-FUTURES", "channel": "fill", "instId": "default"}
                            ]
                        }
                        ws.send(json.dumps(subscribe_msg))
                    else:
                        print(f"❌ Authentication failed: {data}")
                    return
                
                # Manejar suscripción exitosa
                if data.get('event') == 'subscribe':
                    return
                    
                arg = data.get('arg', {})
                channel = arg.get('channel')
                
                if channel == 'positions':
                    for item in data.get('data', []):
                        symbol = item.get('instId')
                        if symbol:
                            total = float(item.get('total', 0))
                            if total == 0:
                                self.positions_cache.pop(symbol, None)
                            else:
                                self.positions_cache[symbol] = item
                                
                elif channel == 'fill':
                    for item in data.get('data', []):
                        order_id = item.get('orderId')
                        if order_id:
                            self.fills_cache[order_id].append(item)
                            
            except json.JSONDecodeError:
                pass
            except Exception as e:
                print(f"❌ WS Private message error: {e}")
                
        def on_error(ws, error):
            if self.running and str(error) != "Connection to remote host was lost.":
                print(f"🔔 WS Private error: {error}")
            
        def on_close(ws, close_status_code, close_msg):
            self.private_connected = False
            self.private_authenticated = False
            if self.running:
                print("🔄 Reconnecting WS Private in 5s...")
                time.sleep(5)
                self._run_private_ws()
                
        def on_open(ws):
            self.private_connected = True
            print("🔌 WS Private connected")
            
            # Autenticar
            timestamp = str(int(time.time()))
            sign = self._generate_signature(timestamp)
            
            auth_msg = {
                "op": "login",
                "args": [{
                    "apiKey": self.api_key,
                    "passphrase": self.passphrase,
                    "timestamp": timestamp,
                    "sign": sign
                }]
            }
            ws.send(json.dumps(auth_msg))
            
        self.ws_private = websocket.WebSocketApp(
            self.ws_private_url,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        self.ws_private.run_forever(
            ping_interval=20,
            ping_timeout=10
        )
    
    def _subscribe_ticker_internal(self, symbol):
        """Suscripción interna sin validaciones"""
        if self.ws_public and self.ws_public.sock and self.public_connected:
            try:
                msg = {
                    "op": "subscribe",
                    "args": [{
                        "instType": "USDT-FUTURES",
                        "channel": "ticker",
                        "instId": symbol
                    }]
                }
                self.ws_public.send(json.dumps(msg))
            except Exception as e:
                print(f"🔔 Subscribe error for {symbol}: {e}")
        
    def subscribe_ticker(self, symbol):
        """Suscribe a ticker de un símbolo"""
        if symbol in self.subscribed_symbols:
            return
        
        self.subscribed_symbols.add(symbol)
        self._subscribe_ticker_internal(symbol)
            
    def get_ticker(self, symbol):
        """Obtiene ticker desde cache WS"""
        if symbol not in self.subscribed_symbols:
            self.subscribe_ticker(symbol)
            time.sleep(0.5)
        return self.ticker_cache.get(symbol)
        
    def get_position(self, symbol):
        """Obtiene posición desde cache WS"""
        return self.positions_cache.get(symbol)
        
    def get_fills(self, order_id):
        """Obtiene fills desde cache WS"""
        return self.fills_cache.get(order_id, [])
    
    def is_connected(self):
        """Verifica si ambas conexiones están activas"""
        return self.public_connected and self.private_connected

# Instancia global
_ws_manager = None

def init_ws_manager(api_key, secret_key, passphrase):
    global _ws_manager
    _ws_manager = BitgetWSManager(api_key, secret_key, passphrase)
    _ws_manager.start()
    return _ws_manager

def get_ws_manager():
    return _ws_manager