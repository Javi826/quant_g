"""
Módulo de dashboard web para el bot de trading.
Se ejecuta en un thread separado y proporciona visualización en tiempo real.
"""

import os
import json
import re
import threading
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, jsonify

class DashboardServer:
    """Servidor web del dashboard para monitoreo en tiempo real del bot"""
    
    def __init__(self, account_number, base_dir, get_current_price_func, 
                 get_balance_func, strategies_config, color_code=None,
                 initial_capital=0, implemented_strategies=None, symbols_by_strategy=None):
        """
        Inicializa el servidor del dashboard.
        
        Args:
            account_number: Número de cuenta (ej: "01", "E1")
            base_dir: Directorio base de los archivos del bot
            get_current_price_func: Función para obtener precio actual de un símbolo
            get_balance_func: Función para obtener balance USDT
            strategies_config: Lista de configuraciones de estrategias (STRATEGIES)
            color_code: Código ANSI de color para logs (opcional)
            initial_capital: Capital inicial de la cuenta
            implemented_strategies: Set de estrategias implementadas
            symbols_by_strategy: Dict con símbolos por estrategia
        """
        self.account_number = account_number
        self.base_dir = base_dir
        self.get_current_price = get_current_price_func
        self.get_balance = get_balance_func
        self.strategies = strategies_config
        self.color_code = color_code or ""
        self.initial_capital = initial_capital
        self.implemented_strategies = implemented_strategies or set()
        self.symbols_by_strategy = symbols_by_strategy or {}
        
        # Rutas de archivos
        self.state_file = os.path.join(base_dir, f'bot_state_{account_number}.json')
        self.trades_file = os.path.join(base_dir, f'bot_trades_{account_number}.xlsx')
        self.log_file = os.path.join(base_dir, f'BOT_all_strategies_{account_number}.log')
        
        # ⭐ Templates directory COMÚN (no dentro de bot_files_XX)
        self.templates_dir = os.path.join(os.path.dirname(base_dir), 'templates')
        os.makedirs(self.templates_dir, exist_ok=True)
        
        # Flask app
        self.app = Flask(__name__, template_folder=self.templates_dir)
        self.app.last_log_position = 0
        
        # Registrar rutas
        self._register_routes()
        
        # Thread del servidor
        self.server_thread = None
        self.running = False
    
    def _register_routes(self):
        """Registra todas las rutas de la API del dashboard"""
        
        @self.app.route('/')
        def index():
            """Página principal del dashboard - Pasar account_number al template"""
            return render_template('dashboard.html', account=self.account_number)
        
        @self.app.route('/api/logs/stream')
        def stream_logs():
            """
            Devuelve las líneas nuevas del log desde la última lectura.
            Limpia códigos ANSI para mejor visualización en web.
            """
            try:
                if not os.path.exists(self.log_file):
                    return jsonify({'logs': [], 'timestamp': None})
                
                with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    # Ir a la última posición leída
                    f.seek(self.app.last_log_position)
                    new_lines = f.readlines()
                    # Actualizar posición para próxima lectura
                    self.app.last_log_position = f.tell()
                
                # Limpiar códigos ANSI (colores de terminal)
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                
                # Limpiar emojis
                emoji_pattern = re.compile(
                    "["
                    u"\U0001F600-\U0001F64F"
                    u"\U0001F300-\U0001F5FF"
                    u"\U0001F680-\U0001F6FF"
                    u"\U0001F1E0-\U0001F1FF"
                    u"\U00002702-\U000027B0"
                    u"\U000024C2-\U0001F251"
                    "]+", flags=re.UNICODE)
                
                clean_lines = []
                for line in new_lines:
                    if line.strip():
                        clean = ansi_escape.sub('', line.strip())
                        clean = emoji_pattern.sub('', clean)
                        clean = ' '.join(clean.split())
                        if clean:
                            clean_lines.append(clean)
                
                return jsonify({
                    'logs': clean_lines,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({'error': str(e), 'logs': []}), 500
        
        @self.app.route('/api/status')
        def get_status():
            """Estado general del bot"""
            try:
                if not os.path.exists(self.state_file):
                    return jsonify({'error': 'State file not found'}), 404
                
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                # Contar posiciones abiertas
                total_positions = sum(len(positions) 
                                    for positions in state.get('positions', {}).values())
                
                # Obtener balance actual del WebSocket
                try:
                    balance = self.get_balance(None)
                except Exception as e:
                    print(f"⚠️  Error getting balance: {e}")
                    balance = 0.0
                
                # Calcular profit cerrado desde Excel
                total_profit = 0
                num_trades = 0
                positive_trades = 0
                profit_pct = 0
                trades_pct = 0
                
                if os.path.exists(self.trades_file):
                    try:
                        df = pd.read_excel(self.trades_file)
                        total_profit = df['PROFIT'].sum()
                        num_trades = len(df)
                        positive_trades = len(df[df['PROFIT'] > 0])
                        
                        if num_trades > 0:
                            trades_pct = (positive_trades / num_trades) * 100
                        
                        if self.initial_capital > 0:
                            profit_pct = (total_profit / self.initial_capital) * 100
                    except Exception as e:
                        print(f"⚠️  Error reading trades file: {e}")
                
                # Calcular PnL abierto
                open_pnl = 0
                for strat_id, positions in state.get('positions', {}).items():
                    for pos in positions:
                        try:
                            symbol = pos['symbol']
                            current_price = self.get_current_price(symbol)
                            entry_price = float(pos['entry_price'])
                            size = float(pos['size'])
                            direction = pos['direction'].lower()
                            
                            if direction == 'long':
                                pnl = (float(current_price) - entry_price) * size
                            else:
                                pnl = (entry_price - float(current_price)) * size
                            
                            open_pnl += pnl
                        except Exception as e:
                            print(f"⚠️No PnL for {pos.get('symbol')}: {e}")
                
                # Obtener precio de BTC
                btc_price = 0
                try:
                    btc_price = float(self.get_current_price('BTCUSDT'))
                except:
                    pass
                
                return jsonify({
                    'status': 'running',
                    'account': self.account_number,
                    'total_positions': total_positions,
                    'strategies': state.get('positions', {}),
                    'candles': state.get('strategy_candles', {}),
                    'total_profit': float(total_profit),
                    'open_pnl': float(open_pnl),
                    'balance': float(balance),
                    'profit_pct': float(profit_pct),
                    'num_trades': num_trades,
                    'trades_pct': float(trades_pct),
                    'btc_price': float(btc_price),
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/positions')
        def get_positions():
            """Posiciones con precios actuales"""
            try:
                if not os.path.exists(self.state_file):
                    return jsonify([])
                
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                positions_data = []
                for strategy_id, positions in state.get('positions', {}).items():
                    max_candles = 50
                    for strat in self.strategies:
                        if strat['id'] == strategy_id:
                            max_candles = strat.get('sell_after_ncandles', 50)
                            break
                    
                    for pos in positions:
                        try:
                            symbol = pos['symbol']
                            current_price = self.get_current_price(symbol)
                            entry_price = float(pos['entry_price'])
                            size = float(pos['size'])
                            direction = pos['direction'].lower()
                            
                            if direction == 'long':
                                pnl = (float(current_price) - entry_price) * size
                            else:
                                pnl = (entry_price - float(current_price)) * size
                            
                            candles = state.get('strategy_candles', {}).get(strategy_id, 0)
                            
                            positions_data.append({
                                'strategy': strategy_id,
                                'symbol': symbol,
                                'direction': pos['direction'],
                                'entry_price': entry_price,
                                'current_price': float(current_price),
                                'size': size,
                                'tp': float(pos['tp']),
                                'sl': float(pos['sl']),
                                'current_pnl': float(pnl),
                                'candles': candles,
                                'max_candles': max_candles,
                                'opened_at': pos['opened_at']
                            })
                        except Exception as e:
                            print(f"⚠️  Error processing position {pos.get('symbol')}: {e}")
                
                return jsonify(positions_data)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/trades/recent')
        def get_recent_trades():
            """Últimos 15 trades cerrados"""
            try:
                if not os.path.exists(self.trades_file):
                    return jsonify([])
                
                df = pd.read_excel(self.trades_file)
                recent = df.tail(15).to_dict('records')
                return jsonify(recent)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/strategy-analysis')
        def get_strategy_analysis():
            """Análisis detallado por estrategia"""
            try:
                if not os.path.exists(self.trades_file):
                    return jsonify([])
                
                df = pd.read_excel(self.trades_file)
                
                if df.empty:
                    return jsonify([])
                
                df['OPEN_AT'] = pd.to_datetime(df['OPEN_AT'])
                df['CLOSE_AT'] = pd.to_datetime(df['CLOSE_AT'])
                df['DURATION'] = (df['CLOSE_AT'] - df['OPEN_AT']).dt.total_seconds() / 86400
                
                results = []
                
                num_strategies = df['STRATEGY'].nunique()
                capital_per_strategy = self.initial_capital / num_strategies if num_strategies > 0 else 0
                
                for strategy in sorted(df['STRATEGY'].unique()):
                    df_strategy = df[df['STRATEGY'] == strategy]
                    
                    num_trades = len(df_strategy)
                    positive_trades = len(df_strategy[df_strategy['PROFIT'] > 0])
                    pct_positive = (positive_trades / num_trades * 100) if num_trades > 0 else 0
                    total_profit = df_strategy['PROFIT'].sum()
                    profit_pct = (total_profit / capital_per_strategy * 100) if capital_per_strategy > 0 else 0
                    avg_duration = round(df_strategy['DURATION'].mean(), 2)
                    date_fo = df_strategy['OPEN_AT'].min()
                    
                    total_reasons = len(df_strategy)
                    tp_count = len(df_strategy[df_strategy['REASON_OUT'].str.contains('TP', na=False)])
                    sl_count = len(df_strategy[df_strategy['REASON_OUT'].str.contains('SL', na=False)])
                    oom_count = len(df_strategy[df_strategy['REASON_OUT'].str.contains('OUT_OF_MARGIN', na=False)])
                    
                    pct_tp = (tp_count / total_reasons * 100) if total_reasons > 0 else 0
                    pct_sl = (sl_count / total_reasons * 100) if total_reasons > 0 else 0
                    pct_oom = (oom_count / total_reasons * 100) if total_reasons > 0 else 0
                    
                    results.append({
                        'Strategy': strategy,
                        'date_fo': date_fo.strftime('%Y-%m-%d'),
                        'Trades_num': num_trades,
                        'Trades_pct': round(pct_positive, 2),
                        'Total_profit': round(total_profit, 2),
                        'Profit_pct': round(profit_pct, 2),
                        'TP_pct': round(pct_tp, 2),
                        'SL_pct': round(pct_sl, 2),
                        'OOM_pct': round(pct_oom, 2),
                        'Avg_days': avg_duration
                    })
                
                return jsonify(results)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/bot-config')
        def get_bot_config():
            """Configuración y estado del bot con detección de 3 estados"""
            try:
                # WebSocket status
                ws_status = {
                    'public_connected': False,
                    'private_connected': False,
                    'authenticated': False
                }
                
                try:
                    import ZX_BOT_ws_manager
                    if ZX_BOT_ws_manager._ws_manager:
                        ws = ZX_BOT_ws_manager._ws_manager
                        ws_status['public_connected'] = (
                            ws.public_ws and 
                            ws.public_ws.sock and 
                            ws.public_ws.sock.connected
                        )
                        ws_status['private_connected'] = (
                            ws.private_ws and 
                            ws.private_ws.sock and 
                            ws.private_ws.sock.connected
                        )
                        ws_status['authenticated'] = ws.authenticated
                except:
                    pass
                
                # Agrupar estrategias por timeframe
                timeframes_grouped = {}
                for strat in self.strategies:
                    tf = strat.get('timeframe', 'Unknown')
                    if tf not in timeframes_grouped:
                        timeframes_grouped[tf] = []
                    timeframes_grouped[tf].append(strat['id'])
                
                # Detectar 3 estados: ACTIVE, DEPRECATING, NOT IMPLEMENTED
                declared_names = {s['name'] for s in self.strategies}
                
                active_count = 0
                deprecating_count = 0
                not_implemented_count = len(self.implemented_strategies - declared_names)
                
                # Preparar lista completa de estrategias
                strategies_list = []
                
                # 1. Estrategias declaradas en STRATEGIES
                for strat in self.strategies:
                    is_active = strat.get('active', True)
                    
                    if is_active:
                        status = 'ACTIVE'
                        active_count += 1
                    else:
                        status = 'DEPRECATING'
                        deprecating_count += 1
                    
                    # Obtener número de símbolos
                    symbols_count = len(self.symbols_by_strategy.get(strat['id'], []))
                    
                    strategies_list.append({
                        'id': strat['id'],
                        'name': strat.get('name', strat['id']),
                        'timeframe': strat.get('timeframe', 'N/A'),
                        'direction': strat.get('direction', 'N/A'),
                        'status': status,
                        'symbols_count': symbols_count,
                        'tp_pct': strat.get('tp_pct', 'N/A'),
                        'sl_pct': strat.get('sl_pct', 'N/A'),
                        'order_amount': strat.get('order_amount', 'N/A'),
                        'sell_after_ncandles': strat.get('sell_after_ncandles', 'N/A'),
                        'lookback': strat.get('lookback', 'N/A'),
                        'tolerance': strat.get('tolerance', 'N/A'),
                        'ma_period': strat.get('ma_period', 'N/A'),
                        'impulse': strat.get('impulse', 'N/A'),
                        'trend_th': strat.get('trend_th', 'N/A')
                    })
                
                # 2. Estrategias implementadas pero NO declaradas
                not_declared = self.implemented_strategies - declared_names
                for name in sorted(not_declared):
                    strategies_list.append({
                        'id': name,
                        'name': name,
                        'timeframe': 'N/A',
                        'direction': 'N/A',
                        'status': 'NOT IMPLEMENTED',
                        'symbols_count': 0,
                        'tp_pct': 'N/A',
                        'sl_pct': 'N/A',
                        'order_amount': 'N/A',
                        'sell_after_ncandles': 'N/A',
                        'lookback': 'N/A',
                        'tolerance': 'N/A',
                        'ma_period': 'N/A',
                        'impulse': 'N/A',
                        'trend_th': 'N/A'
                    })
                
                return jsonify({
                    'account': self.account_number,
                    'initial_capital': self.initial_capital,
                    'websocket_status': ws_status,
                    'strategies': strategies_list,
                    'timeframes': timeframes_grouped,
                    'stats': {
                        'total': len(self.implemented_strategies),
                        'active': active_count,
                        'deprecating': deprecating_count,
                        'not_implemented': not_implemented_count
                    }
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/config')
        def get_config():
            """Configuración legacy (mantener por compatibilidad)"""
            try:
                strategies_info = []
                for strat in self.strategies:
                    strategies_info.append({
                        'id': strat['id'],
                        'name': strat['name'],
                        'timeframe': strat['timeframe'],
                        'active': strat.get('active', True),
                        'direction': strat['direction'],
                        'tp_pct': strat['tp_pct'],
                        'sl_pct': strat['sl_pct'],
                        'order_amount': strat.get('order_amount', 0)
                    })
                
                return jsonify({
                    'account': self.account_number,
                    'strategies': strategies_info
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def start(self, host='0.0.0.0', port=5000):
        """Inicia el servidor del dashboard"""
        if self.running:
            print("⚠️  Dashboard already running")
            return
        
        def run_server():
            import logging
            log = logging.getLogger('werkzeug')
            log.setLevel(logging.ERROR)
            
            try:
                self.app.run(
                    host=host, 
                    port=port, 
                    debug=False, 
                    use_reloader=False, 
                    threaded=True
                )
            except Exception as e:
                print(f"❌ Dashboard server error: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.running = True
        
        import time
        time.sleep(0.5)
        
        color = self.color_code
        reset = "\033[0m" if color else ""
        
        print(f"\n{color}🌐 Dashboard Web Started{reset}")
        print(f"{'─' * 45}")
        print(f"📊 Local:   http://localhost:{port}")
        print(f"🔗 Network: http://127.0.0.1:{port}")
        print(f"🌍 LAN:     http://<your-ip>:{port}")
        print(f"{'─' * 45}\n")
    
    def stop(self):
        """Detiene el servidor"""
        self.running = False
        print("🛑 Dashboard server stopped")


def create_dashboard_template(base_dir):
    """Crea el archivo HTML del dashboard si no existe en ruta común"""
    # ⭐ Ruta común (no dentro de bot_files_XX)
    templates_dir = os.path.join(os.path.dirname(base_dir), 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    template_path = os.path.join(templates_dir, 'dashboard.html')
    
    if os.path.exists(template_path):
        return
    
    print(f"⚠️  Creating template at {template_path}")
    print(f"   Please ensure the complete dashboard.html is in place")


if __name__ == '__main__':
    print("⚠️  This module should be imported, not run directly")
    print("Usage:")
    print("  from ZX_BOT_dashboard import DashboardServer")
    print("  dashboard = DashboardServer(...)")
    print("  dashboard.start()")