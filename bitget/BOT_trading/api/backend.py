"""
Módulo de dashboard web para el bot de trading.
Se ejecuta en un thread separado y proporciona visualización en tiempo real.
"""

import os
import json
import re
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, jsonify, send_from_directory, request
import logging
from market_data.websocket_manager import get_ws_manager
logger = logging.getLogger('BOT_trading.api.backend')

from analytics.metrics import MetricsCalculator

# ═══════════════════════════════════════════════════════════════════════════
# MARKET REGIME IMPORT
# ═══════════════════════════════════════════════════════════════════════════
try:
    from market_regime.regime_classifier import get_regime_info
    from config.settings import REGIME_FAMILIES, REGIME_FAMILY_SIZING
    REGIME_MODULE_AVAILABLE = True
except ImportError as e:
    logger.warning(f"WAR-Market regime module not available: {e}")
    REGIME_MODULE_AVAILABLE = False
    REGIME_FAMILIES = {}
    REGIME_FAMILY_SIZING = {}


class DashboardServer:
    """Servidor web del dashboard para monitoreo en tiempo real del bot"""
    
    def __init__(self, account_number, base_dir, get_current_price_func, 
                 get_balance_func, strategies_config,
                 initial_capital=0, implemented_strategies=None, symbols_by_strategy=None):
        """
        Inicializa el servidor del dashboard.
        
        Args:
            account_number: Número de cuenta (ej: "01", "E1")
            base_dir: Directorio base de los archivos del bot
            get_current_price_func: Función para obtener precio actual de un símbolo
            get_balance_func: Función para obtener balance USDT
            initial_capital: Capital inicial de la cuenta
            implemented_strategies: Set de estrategias implementadas
            symbols_by_strategy: Dict con símbolos por estrategia
        """
        self.account_number = account_number
        self.base_dir = base_dir
        self.get_current_price = get_current_price_func
        self.get_balance = get_balance_func
        self.strategies = strategies_config
        self.initial_capital = initial_capital
        self.implemented_strategies = implemented_strategies or set()
        self.symbols_by_strategy = symbols_by_strategy or {}
        
        self.state_file = os.path.join(base_dir, f'bot_state_{account_number}.json')
        self.trades_file = os.path.join(base_dir, f'bot_trades_{account_number}.xlsx')
        self.log_file = os.path.join(base_dir, f'BOT_orchestator_{account_number}.log')
        
        self.templates_dir = os.path.join(os.path.dirname(__file__), 'templates')
        os.makedirs(self.templates_dir, exist_ok=True)
        
        self.app = Flask(__name__, template_folder=self.templates_dir)
        self.app.last_log_position = 0
        
        self._register_routes()
        
        self.server_thread = None
        self.running = False
        
    def get_precision_for_price(self, price):
        """
        Determina el número de decimales a mostrar según la magnitud del precio.
        """
        price = abs(float(price))
        
        if price >= 10000:
            return 1
        elif price >= 1000:
            return 2
        elif price >= 100:
            return 2
        elif price >= 10:
            return 3
        elif price >= 1:
            return 4
        elif price >= 0.01:
            return 5
        else:
            return 5
    
    def _load_trades_dataframe(self):
        """
        Carga y valida el DataFrame de trades desde el archivo Excel.
        """
        if not os.path.exists(self.trades_file):
            return None
        
        try:
            df = pd.read_excel(self.trades_file, engine='openpyxl')
            if df.empty:
                return None
            return df
        except Exception as e:
            logger.error(f"Error-loading trades file: {e}")
            return None
    
    def _prepare_trades_dataframe(self, df):
        """
        Prepara el DataFrame de trades con columnas de fechas procesadas.
        """
        df = df.copy()
        df['OPEN_AT'] = pd.to_datetime(df['OPEN_AT'])
        df['CLOSE_AT'] = pd.to_datetime(df['CLOSE_AT'])
        df['CLOSE_DATE'] = pd.to_datetime(df['CLOSE_AT'])
        df['DURATION'] = (df['CLOSE_AT'] - df['OPEN_AT']).dt.total_seconds() / 86400
        return df
    
    def _filter_df_by_dates(self, df, date_from=None, date_to=None):
        """
        Filtra DataFrame por rango de fechas.
        
        Args:
            df: DataFrame con columna CLOSE_AT
            date_from: Fecha inicio (string YYYY-MM-DD o None)
            date_to: Fecha fin (string YYYY-MM-DD o None)
        
        Returns:
            DataFrame filtrado
        """
        if df is None or df.empty:
            return df
        
        df = df.copy()
        
        if 'CLOSE_AT' not in df.columns:
            return df
        
        if date_from:
            try:
                date_from_dt = pd.to_datetime(date_from)
                df = df[df['CLOSE_AT'] >= date_from_dt]
            except:
                pass
        
        if date_to:
            try:
                date_to_dt = pd.to_datetime(date_to) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
                df = df[df['CLOSE_AT'] <= date_to_dt]
            except:
                pass
        
        return df
    
    @staticmethod
    def _extract_number_from_id(strategy_id):
        """
        Extrae el número prefijo del ID de estrategia.
        """
        if not strategy_id or not isinstance(strategy_id, str):
            return '??'
        
        match = re.match(r'^(\d{2})_', strategy_id)
        if match:
            return match.group(1)
        
        match = re.match(r'^(\d+)', strategy_id)
        if match:
            return match.group(1).zfill(2)
        
        return '??'
    
    @staticmethod
    def _calculate_r_squared(equity_values):
        """
        Calcula R² de la equity curve vs línea recta.
        Mide consistencia del crecimiento.
        
        R² = 1.0 → Línea recta perfecta (ideal)
        R² > 0.9 → Muy consistente
        R² = 0.7-0.9 → Buena consistencia
        R² < 0.7 → Equity errática
        
        Args:
            equity_values: Lista o array de valores de equity
        
        Returns:
            float: R² entre 0 y 1
        """
        if len(equity_values) < 2:
            return 0.0
        
        try:
            y = np.array(equity_values).reshape(-1, 1)
            X = np.arange(len(y)).reshape(-1, 1)
            
            # Calcular R² manualmente (sin sklearn para evitar dependencia)
            y_mean = np.mean(y)
            
            # Regresión lineal simple: y = mx + b
            X_mean = np.mean(X)
            numerator = np.sum((X - X_mean) * (y - y_mean))
            denominator = np.sum((X - X_mean) ** 2)
            
            if denominator == 0:
                return 0.0
            
            slope = numerator / denominator
            intercept = y_mean - slope * X_mean
            
            # Predicciones
            y_pred = slope * X + intercept
            
            # R²
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - y_mean) ** 2)
            
            if ss_tot == 0:
                return 1.0 if ss_res == 0 else 0.0
            
            r_squared = 1 - (ss_res / ss_tot)
            
            return round(float(max(0, min(1, r_squared))), 3)
        except Exception as e:
            logger.error(f"Error calculating R²: {e}")
            return 0.0
    
    def _calculate_capital_allocation(self, num_strategies=None):
        """
        Calcula el capital asignado por estrategia.
        """
        if num_strategies is None:
            num_strategies = len(self.implemented_strategies)
        
        if num_strategies == 0:
            return 0.0
        
        return self.initial_capital / num_strategies
    
    def _get_full_strategies_list_with_numbers(self):
        """
        Genera la lista completa de estrategias con numeración extraída de los IDs.
        """
        COMMON_PARAMS = {
            'id', 'name', 'timeframe', 'direction', 'active',
            'tp_pct', 'sl_pct', 'order_amount', 'sell_after_ncandles'
        }
        
        EXCLUDE_PARAMS = {'active'}
        
        declared_ids = {s['id'] for s in self.strategies}
        strategies_list = []
        
        for strat in self.strategies:
            is_active = strat.get('active', True)
            status = 'ACTIVE' if is_active else 'DEPRECATING'
            symbols_count = len(self.symbols_by_strategy.get(strat['id'], []))
            
            strategy_dict = {
                'id': strat['id'],
                'name': strat.get('name', strat['id']),
                'timeframe': strat.get('timeframe', 'N/A'),
                'direction': strat.get('direction', 'N/A'),
                'status': status,
                'symbols_count': symbols_count
            }
            
            for key, value in strat.items():
                if key not in strategy_dict and key not in EXCLUDE_PARAMS:
                    strategy_dict[key] = value if value is not None else 'N/A'
            
            for param in COMMON_PARAMS - EXCLUDE_PARAMS - {'id', 'name'}:
                if param not in strategy_dict:
                    strategy_dict[param] = 'N/A'
            
            strategies_list.append(strategy_dict)
        
        not_declared = self.implemented_strategies - declared_ids
        for id in sorted(not_declared):
            strategies_list.append({
                'id': id,
                'name': id,
                'timeframe': 'N/A',
                'direction': 'N/A',
                'status': 'NOT IMPLE.',
                'symbols_count': 0,
                'tp_pct': 'N/A',
                'sl_pct': 'N/A',
                'order_amount': 'N/A',
                'sell_after_ncandles': 'N/A'
            })
        
        strategies_list.sort(key=lambda x: x['id'])
        for strat in strategies_list:
            strat['number'] = self._extract_number_from_id(strat['id'])
        
        return strategies_list
    
    def _register_routes(self):
        """Registra todas las rutas de la API del dashboard"""
        
        @self.app.route('/')
        def index():
            return render_template('dashboard.html', account=self.account_number)
        
        @self.app.route('/favicon.jpg')
        def favicon():
            return send_from_directory(self.base_dir, 'favicon.jpg', mimetype='image/jpeg')
        
        @self.app.route('/api/health')
        def health_check():
            return jsonify({
                'status': 'ready',
                'account': self.account_number,
                'timestamp': datetime.now().isoformat()
            })
        

        @self.app.route('/api/logs/stream')
        def stream_logs():
            try:
                if not os.path.exists(self.log_file):
                    return jsonify({'logs': [], 'timestamp': None})
                
                with open(self.log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    f.seek(self.app.last_log_position)
                    new_lines = f.readlines()
                    self.app.last_log_position = f.tell()
                
                import re
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                
                clean_lines = []
                for line in new_lines:
                    line = ansi_escape.sub('', line)
                    line = line.strip()
                    
                    if line:
                        if ' - ' in line:
                            message = line.split(' - ')[-1]
                            clean_lines.append(message)
                        else:
                            clean_lines.append(line)
                
                return jsonify({
                    'logs': clean_lines,
                    'timestamp': datetime.now().isoformat()
                })
            except Exception as e:
                return jsonify({'error': str(e), 'logs': []}), 500
            
        @self.app.route('/api/status')
        def get_status():
            try:
                if not os.path.exists(self.state_file):
                    return jsonify({'error': 'Error-State file not found'}), 404
                
                with open(self.state_file, 'r') as f:
                    state = json.load(f)
                
                total_positions = sum(len(positions) 
                                    for positions in state.get('positions', {}).values())
                
                try:
                    balance = self.get_balance(None)
                except Exception as e:
                    print(f"Error-getting balance: {e}")
                    balance = 0.0
                
                total_profit = 0
                num_trades = 0
                positive_trades = 0
                profit_pct = 0
                trades_pct = 0
                
                df = self._load_trades_dataframe()
                if df is not None:
                    total_profit = df['PROFIT'].sum()
                    num_trades = len(df)
                    positive_trades = len(df[df['PROFIT'] > 0])
                    
                    if num_trades > 0:
                        trades_pct = (positive_trades / num_trades) * 100
                    
                    if self.initial_capital > 0:
                        profit_pct = (total_profit / self.initial_capital) * 100
                
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
                            print(f"No PnL - {pos.get('symbol')}: {e}")
                
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
                            symbol        = pos['symbol']
                            current_price = self.get_current_price(symbol)
                            entry_price   = float(pos['entry_price'])
                            size          = float(pos['size'])
                            direction     = pos['direction'].lower()
                            tp_price      = float(pos['tp'])
                            sl_price      = float(pos['sl'])
                            
                            if direction == 'long':
                                pnl = (float(current_price) - entry_price) * size
                            else:
                                pnl = (entry_price - float(current_price)) * size
                            
                            candles = state.get('strategy_candles', {}).get(strategy_id, 0)
                            
                            precision = self.get_precision_for_price(current_price)
                            
                            current_price_rounded = round(float(current_price), precision)
                            tp_rounded = round(tp_price, precision)
                            sl_rounded = round(sl_price, precision)
                            entry_rounded = round(entry_price, precision)
                            
                            if direction == 'long':
                                distance_to_tp = ((tp_price - float(current_price)) / float(current_price)) * 100
                                distance_to_sl = ((float(current_price) - sl_price) / float(current_price)) * 100
                            else:
                                distance_to_tp = ((float(current_price) - tp_price) / float(current_price)) * 100
                                distance_to_sl = ((sl_price - float(current_price)) / float(current_price)) * 100
                            
                            positions_data.append({
                                'strategy': strategy_id,
                                'symbol': symbol,
                                'direction': pos['direction'],
                                'entry_price': entry_rounded,
                                'current_price': current_price_rounded,
                                'size': size,
                                'tp': tp_rounded,
                                'sl': sl_rounded,
                                'current_pnl': float(pnl),
                                'candles': candles,
                                'max_candles': max_candles,
                                'opened_at': pos['opened_at'],
                                'distance_to_tp_pct': round(distance_to_tp, 2),
                                'distance_to_sl_pct': round(distance_to_sl, 2),
                                'precision': precision
                            })
                        except Exception as e:
                            print(f"⚠️  Error processing position {pos.get('symbol')}: {e}")
                
                return jsonify(positions_data)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
            
        @self.app.route('/api/bot/stop', methods=['POST'])
        def stop_bot():
            try:
                import subprocess
                import os
                import signal
                
                result = subprocess.run(
                    ['pgrep', '-f', f'main.py --account {self.account_number}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    pid = int(result.stdout.strip())
                    
                    os.kill(pid, signal.SIGTERM)
                    
                    logger.info(f"Stop signal sent to PID {pid}")
                    
                    return jsonify({
                        'status': 'stopped',
                        'pid': pid,
                        'message': f'Bot process {pid} terminated'
                    })
                else:
                    return jsonify({
                        'status': 'not_found',
                        'message': 'Bot process not found'
                    }), 404
                    
            except subprocess.TimeoutExpired:
                logger.error("Error-pgrep command timed out")
                return jsonify({'error': 'Error-Command timeout'}), 500
            except Exception as e:
                logger.error(f"Error stopping bot: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/bot/verify-stopped')
        def verify_stopped():
            try:
                import subprocess
                
                result = subprocess.run(
                    ['pgrep', '-f', f'main.py --account {self.account_number}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                running = result.returncode == 0 and result.stdout.strip()
                pid = int(result.stdout.strip()) if running else None
                
                return jsonify({'pid': pid, 'running': running})
                
            except Exception as e:
                logger.error(f"Error-verifying stop: {e}")
                return jsonify({'running': False, 'error': str(e)}), 200
                
        @self.app.route('/api/trades/recent')
        def get_recent_trades():
            try:
                df = self._load_trades_dataframe()
                if df is None:
                    return jsonify([])
                
                recent = df.tail(15).to_dict('records')
                return jsonify(recent)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/strategy-analysis')
        def get_strategy_analysis():
            try:
                date_from = request.args.get('date_from', None)
                date_to = request.args.get('date_to', None)
                
                df = self._load_trades_dataframe()
                if df is None:
                    return jsonify([])
                
                df = self._prepare_trades_dataframe(df)
                df = self._filter_df_by_dates(df, date_from, date_to)
                
                if df.empty:
                    return jsonify([])
                
                results = []
                
                num_strategies = df['STRATEGY'].nunique()
                capital_per_strategy = self._calculate_capital_allocation(num_strategies)
                
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
                    timeout_count = len(df_strategy[df_strategy['REASON_OUT'] == 'TIMEOUT'])
                    
                    pct_tp = (tp_count / total_reasons * 100) if total_reasons > 0 else 0
                    pct_sl = (sl_count / total_reasons * 100) if total_reasons > 0 else 0
                    pct_oom = (oom_count / total_reasons * 100) if total_reasons > 0 else 0
                    pct_timeout = (timeout_count / total_reasons * 100) if total_reasons > 0 else 0
                    
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
                        'TIMEOUT_pct': round(pct_timeout, 2),
                        'Avg_days': avg_duration
                    })
                
                return jsonify(results)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/bot-config')
        def get_bot_config():
            try:
                ws_status = {
                    'public_connected': False,
                    'private_connected': False,
                    'authenticated': False
                }
                
                try:
                    ws = get_ws_manager()
                    
                    if ws:
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
                except Exception as e:
                    logger.warning(f"WAR-Could not get WS status: {e}")
                
                timeframes_grouped = {}
                for strat in self.strategies:
                    tf = strat.get('timeframe', 'Unknown')
                    if tf not in timeframes_grouped:
                        timeframes_grouped[tf] = []
                    timeframes_grouped[tf].append(strat['id'])
                
                strategies_list = self._get_full_strategies_list_with_numbers()
                
                active_count = sum(1 for s in strategies_list if s['status'] == 'ACTIVE')
                deprecating_count = sum(1 for s in strategies_list if s['status'] == 'DEPRECATING')
                not_implemented_count = sum(1 for s in strategies_list if s['status'] == 'NOT IMPLEMENTED')
                
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
        
        @self.app.route('/api/compose-analysis')
        def get_compose_analysis():
            try:
                from itertools import combinations
                
                metric = request.args.get('metric', 'profit_factor')
                date_from = request.args.get('date_from', None)
                date_to = request.args.get('date_to', None)
                
                df = self._load_trades_dataframe()
                if df is None:
                    return jsonify([])
                
                df = self._prepare_trades_dataframe(df)
                df = self._filter_df_by_dates(df, date_from, date_to)
                
                if df.empty:
                    return jsonify([])
                
                strategies_list = self._get_full_strategies_list_with_numbers()
                
                active_deprecating = [s for s in strategies_list 
                                     if s['status'] in ('ACTIVE', 'DEPRECATING')]
                
                strategies_in_excel = df['STRATEGY'].unique().tolist()
                
                strategies_with_trades = []
                for strat in active_deprecating:
                    if strat['id'] in strategies_in_excel:
                        strategies_with_trades.append(strat['id'])
                
                if len(strategies_with_trades) == 0:
                    return jsonify([])
                
                results = []
                
                num_strategies_with_trades = len(df['STRATEGY'].unique())
                capital_per_strat = self._calculate_capital_allocation(num_strategies_with_trades)
                
                for r in range(1, len(strategies_with_trades) + 1):
                    for combo in combinations(strategies_with_trades, r):
                        df_combo = df[df['STRATEGY'].isin(combo)]
                        
                        if len(df_combo) == 0:
                            continue
                        
                        combo_capital = capital_per_strat * len(combo)
                        
                        metrics = MetricsCalculator.calculate_all_metrics(
                            df=df_combo,
                            capital_assigned=combo_capital,
                            include_profit_pct=True
                        )
                        
                        # Calcular R² de la equity curve
                        daily_profit = metrics.get('daily_profit')
                        r_squared = 0.0
                        if daily_profit is not None and not daily_profit.empty and 'equity_usd' in daily_profit.columns:
                            r_squared = self._calculate_r_squared(daily_profit['equity_usd'].values)
                        
                        combo_numbers = [self._extract_number_from_id(s) for s in combo]
                        combo_str = '+'.join(combo_numbers)
                        
                        results.append({
                            'combination': combo_str,
                            'num_trades': metrics['num_trades'],
                            'total_profit_pct': metrics['total_profit_pct'],
                            'total_profit_usd': metrics['total_profit_usd'],
                            'profit_factor': metrics['profit_factor'],
                            'weekly_win_pct': metrics['weekly_win_pct'],
                            'win_rate': metrics['win_rate'],
                            'max_dd': metrics['max_dd'],
                            'r_squared': r_squared,
                            'sharpe_ratio': metrics['sharpe_ratio']
                        })
                
                if metric == 'max_dd':
                    results_sorted = sorted(results, key=lambda x: x[metric], reverse=True)
                elif metric == 'r_squared':
                    results_sorted = sorted(results, key=lambda x: x[metric], reverse=True)
                else:
                    results_sorted = sorted(results, key=lambda x: x[metric], reverse=True)
                
                return jsonify(results_sorted[:10])
                
            except Exception as e:
                print(f"Error-in compose: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/monthly-analysis')
        def get_monthly_analysis():
            """
            Devuelve Profit % por mes para las estrategias seleccionadas.
            """
            try:
                strategies_param = request.args.get('strategies', '')
                selected_strategies = [s.strip() for s in strategies_param.split(',') if s.strip()]
                
                if not selected_strategies:
                    return jsonify({'error': 'No strategies selected'}), 400
                
                df = self._load_trades_dataframe()
                if df is None:
                    return jsonify([])
                
                df = self._prepare_trades_dataframe(df)
                df = df[df['STRATEGY'].isin(selected_strategies)]
                
                if df.empty:
                    return jsonify([])
                
                # Calcular capital asignado - usar solo estrategias CON trades (igual que header)
                strategies_with_trades = df['STRATEGY'].unique()
                num_strategies_with_trades = len(strategies_with_trades)
                capital_per_strat = self._calculate_capital_allocation(num_strategies_with_trades)
                capital_assigned = capital_per_strat * num_strategies_with_trades
                
                # Agrupar por mes
                df['month'] = df['CLOSE_AT'].dt.to_period('M')
                
                results = []
                
                for month in sorted(df['month'].unique()):
                    df_month = df[df['month'] == month]
                    
                    num_trades = len(df_month)
                    total_profit = df_month['PROFIT'].sum()
                    profit_pct = (total_profit / capital_assigned * 100) if capital_assigned > 0 else 0
                    
                    positive_trades = len(df_month[df_month['PROFIT'] > 0])
                    win_rate = (positive_trades / num_trades * 100) if num_trades > 0 else 0
                    
                    results.append({
                        'month': str(month),
                        'month_name': month.strftime('%b %Y'),
                        'num_trades': num_trades,
                        'profit_usd': round(total_profit, 2),
                        'profit_pct': round(profit_pct, 2),
                        'win_rate': round(win_rate, 1)
                    })
                
                return jsonify(results)
                
            except Exception as e:
                print(f"Error in monthly analysis: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/symbols-analysis')
        def get_symbols_analysis():
            try:
                df = self._load_trades_dataframe()
                if df is None:
                    return jsonify([])
                
                results = []
                
                for symbol in sorted(df['SYMBOL'].unique()):
                    df_symbol = df[df['SYMBOL'] == symbol]
                    
                    total_trades = len(df_symbol)
                    positive_trades = len(df_symbol[df_symbol['PROFIT'] > 0])
                    win_pct = (positive_trades / total_trades * 100) if total_trades > 0 else 0
                    total_profit = df_symbol['PROFIT'].sum()
                    avg_profit = total_profit / total_trades if total_trades > 0 else 0
                    
                    results.append({
                        'Symbol': symbol,
                        'Total_Trades': total_trades,
                        'Win_Pct': round(win_pct, 2),
                        'Total_Profit': round(total_profit, 2),
                        'Avg_Profit': round(avg_profit, 2)
                    })
                
                return jsonify(results)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/weekday-analysis')
        def get_weekday_analysis():
            try:
                df = self._load_trades_dataframe()
                if df is None:
                    return jsonify([])
                
                df = self._prepare_trades_dataframe(df)
                
                df['weekday'] = df['OPEN_AT'].dt.day_name()
                
                weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                
                results = []
                
                for day in weekday_order:
                    df_day = df[df['weekday'] == day]
                    
                    if len(df_day) == 0:
                        continue
                    
                    total_trades = len(df_day)
                    positive_trades = len(df_day[df_day['PROFIT'] > 0])
                    win_pct = (positive_trades / total_trades * 100) if total_trades > 0 else 0
                    total_profit = df_day['PROFIT'].sum()
                    avg_profit = total_profit / total_trades if total_trades > 0 else 0
                    
                    results.append({
                        'Day': day,
                        'Total_Trades': total_trades,
                        'Win_Pct': round(win_pct, 2),
                        'Total_Profit': round(total_profit, 2),
                        'Avg_Profit': round(avg_profit, 2)
                    })
                
                return jsonify(results)
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/equity-data')
        def get_equity_data():
            try:
                df = self._load_trades_dataframe()
                if df is None:
                    return jsonify({'error': 'Error-No trades file found'}), 404
                
                num_strategies_with_trades = len(df['STRATEGY'].unique())
                
                strategies_param = request.args.get('strategies', '')
                selected_strategies = [s.strip() for s in strategies_param.split(',') if s.strip()]
                date_from = request.args.get('date_from', None)
                date_to = request.args.get('date_to', None)
                
                if not selected_strategies:
                    return jsonify({'error': 'Error-No strategies selected'}), 400
                
                df = df[df['STRATEGY'].isin(selected_strategies)]
                
                # Preparar y filtrar por fechas
                df = self._prepare_trades_dataframe(df)
                df = self._filter_df_by_dates(df, date_from, date_to)
                
                if df.empty:
                    num_selected = len(selected_strategies)
                    return jsonify({
                        'dates': [],
                        'equity_pct': [],
                        'drawdown_pct': [],
                        'capital_assigned': 0,
                        'num_selected': num_selected,
                        'total_strategies': num_strategies_with_trades,
                        'num_trades': 0,
                        'total_profit_usd': 0,
                        'profit_factor': 0,
                        'weekly_win_pct': 0,
                        'win_rate': 0,
                        'max_dd': 0,
                        'r_squared': 0,
                        'sharpe_ratio': 0,
                        'message': 'No trades found for selected strategies'
                    })
                
                df = df.sort_values('CLOSE_AT')
                df['date_str'] = df['CLOSE_AT'].dt.strftime('%Y-%m-%d')
                
                num_selected = len(selected_strategies)
                
                capital_per_strategy = self._calculate_capital_allocation(num_strategies_with_trades)
                capital_assigned = capital_per_strategy * num_selected
                
                metrics_data = MetricsCalculator.calculate_all_metrics(
                    df=df,
                    capital_assigned=capital_assigned,
                    include_profit_pct=False
                )
                
                daily_profit = metrics_data['daily_profit']
                
                r_squared = 0.0
                
                if not daily_profit.empty:
                    daily_profit['date_str'] = daily_profit['date'].astype(str)
                    
                    if capital_assigned > 0:
                        daily_profit['equity_pct'] = ((daily_profit['equity_usd'] / capital_assigned) - 1) * 100
                    else:
                        daily_profit['equity_pct'] = 0
                    
                    daily_profit['peak_usd'] = daily_profit['equity_usd'].cummax()
                    daily_profit['drawdown_pct'] = ((daily_profit['peak_usd'] - daily_profit['equity_usd']) / daily_profit['peak_usd']) * 100
                    
                    # Calcular R²
                    r_squared = self._calculate_r_squared(daily_profit['equity_usd'].values)
                    
                    dates = daily_profit['date_str'].tolist()
                    equity_pct = [round(val, 2) for val in daily_profit['equity_pct'].tolist()]
                    drawdown_pct = [round(val, 2) for val in daily_profit['drawdown_pct'].tolist()]
                else:
                    dates = []
                    equity_pct = []
                    drawdown_pct = []
                
                return jsonify({
                    'dates': dates,
                    'equity_pct': equity_pct,
                    'drawdown_pct': drawdown_pct,
                    'capital_assigned': round(capital_assigned, 2),
                    'num_selected': num_selected,
                    'total_strategies': num_strategies_with_trades,
                    'num_trades': metrics_data['num_trades'],
                    'total_profit_usd': metrics_data['total_profit_usd'],
                    'profit_factor': metrics_data['profit_factor'],
                    'weekly_win_pct': metrics_data['weekly_win_pct'],
                    'win_rate': metrics_data['win_rate'],
                    'max_dd': metrics_data['max_dd'],
                    'r_squared': r_squared,
                    'sharpe_ratio': metrics_data['sharpe_ratio']
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
        ##ddd
        @self.app.route('/api/correlation-matrix', methods=['POST'])
        def get_correlation_matrix():
            """Calculate correlation matrix between selected strategies."""
            try:
                import pandas as pd
                
                # Get selected strategies from request
                data = request.get_json()
                selected_strategies = data.get('strategies', [])
                
                if not selected_strategies:
                    return jsonify({'error': 'No strategies selected'}), 400
                
                # Load trades from Excel
                if not os.path.exists(self.trades_file):
                    return jsonify({'error': 'No trades data available'}), 404
                
                df = pd.read_excel(self.trades_file)
                
                if df.empty:
                    return jsonify({'error': 'Trades file is empty'}), 404
                
                # Filter by selected strategies
                df = df[df['STRATEGY'].isin(selected_strategies)]
                
                if df.empty:
                    return jsonify({'error': 'No trades found for selected strategies'}), 404
                
                # Group by strategy and date, calculate daily returns
                returns_by_strategy = {}
                for strat_id in selected_strategies:
                    strat_df = df[df['STRATEGY'] == strat_id].copy()
                    
                    if strat_df.empty:
                        continue
                    
                    # Convert to datetime and get date
                    strat_df['date'] = pd.to_datetime(strat_df['CLOSE_AT']).dt.date
                    
                    # Sum daily profits
                    daily_returns = strat_df.groupby('date')['PROFIT'].sum()
                    returns_by_strategy[strat_id] = daily_returns
                
                if len(returns_by_strategy) < 2:
                    return jsonify({'error': 'Need at least 2 strategies with trades'}), 400
                
                # Create DataFrame and calculate correlation
                returns_df = pd.DataFrame(returns_by_strategy)
                
                # Fill NaN with 0 (days with no trades)
                returns_df = returns_df.fillna(0)
                
                # Calculate correlation matrix
                corr_matrix = returns_df.corr()
                
                # Find high correlation pairs (>0.7)
                high_corr_pairs = []
                strategies_list = list(corr_matrix.columns)
                
                for i in range(len(strategies_list)):
                    for j in range(i + 1, len(strategies_list)):
                        corr_value = corr_matrix.iloc[i, j]
                        
                        if pd.notna(corr_value) and corr_value > 0.7:
                            high_corr_pairs.append({
                                'strat1': strategies_list[i],
                                'strat2': strategies_list[j],
                                'correlation': round(float(corr_value), 3)
                            })
                
                # Sort by correlation (highest first)
                high_corr_pairs.sort(key=lambda x: x['correlation'], reverse=True)
                
                # Convert matrix to dict (handle NaN values)
                matrix_dict = {}
                for col in corr_matrix.columns:
                    matrix_dict[col] = {}
                    for idx in corr_matrix.index:
                        val = corr_matrix.loc[idx, col]
                        matrix_dict[col][idx] = round(float(val), 3) if pd.notna(val) else 0
                
                return jsonify({
                    'success': True,
                    'matrix': matrix_dict,
                    'strategies': strategies_list,
                    'high_corr_pairs': high_corr_pairs
                })
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500
        # ===========================================================================
        # MARKET REGIME ENDPOINTS
        # ===========================================================================
        
        @self.app.route('/api/regime/current')
        def get_regime_current():
            """
            Obtiene el régimen de mercado actual para un timeframe específico.
            
            Query params:
                timeframe: Timeframe a analizar (ej: '4H', '1H', '6Hutc')
            
            Returns:
                JSON con family, multiplier, metrics, thresholds, all_families, all_thresholds
            """
            if not REGIME_MODULE_AVAILABLE:
                return jsonify({
                    'success': False,
                    'error': 'Market regime module not available',
                    'family': 'ranging',
                    'multiplier': 1.0,
                    'metrics': {},
                    'timeframe': request.args.get('timeframe', '4H'),
                    'all_families': {},
                    'all_thresholds': {}
                }), 200
            
            try:
                timeframe = request.args.get('timeframe', '4H')
                
                # Validar timeframe
                valid_timeframes = ['1H', '4H', '6Hutc', '2m', '5m', '15m', '30m']
                if timeframe not in valid_timeframes:
                    return jsonify({
                        'success': False,
                        'error': f'Invalid timeframe. Valid: {valid_timeframes}',
                        'family': 'ranging',
                        'multiplier': 1.0,
                        'metrics': {},
                        'timeframe': timeframe,
                        'all_families': {},
                        'all_thresholds': {}
                    }), 400
                
                # Obtener régimen actual
                regime_info = get_regime_info(timeframe)
                
                # Retornar info completa incluyendo todas las familias
                return jsonify({
                    'success': True,
                    'timeframe': timeframe,
                    'family': regime_info['family'],
                    'multiplier': regime_info['multiplier'],
                    'metrics': regime_info['metrics'],
                    'thresholds': regime_info.get('thresholds', {}),
                    'all_families': REGIME_FAMILY_SIZING,
                    'all_thresholds': REGIME_FAMILIES
                })
                
            except Exception as e:
                logger.error(f"Error getting regime: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'family': 'ranging',
                    'multiplier': 1.0,
                    'metrics': {},
                    'timeframe': request.args.get('timeframe', '4H'),
                    'all_families': {},
                    'all_thresholds': {}
                }), 500
        
        @self.app.route('/api/regime/history')
        def get_regime_history():
            """
            Obtiene el historial de régimen de mercado (PLACEHOLDER).
            
            Query params:
                timeframe: Timeframe a analizar (ej: '4H', '1H')
                bars: Número de barras históricas (default: 24)
            
            Returns:
                JSON con historial de regímenes
            """
            if not REGIME_MODULE_AVAILABLE:
                return jsonify({
                    'success': False,
                    'error': 'Market regime module not available',
                    'history': []
                }), 200
            
            try:
                timeframe = request.args.get('timeframe', '4H')
                bars = int(request.args.get('bars', 24))
                
                # PLACEHOLDER: Por ahora retornar estructura vacía
                # TODO: Implementar tracking histórico de regímenes
                return jsonify({
                    'success': True,
                    'timeframe': timeframe,
                    'bars': bars,
                    'history': [],
                    'message': 'Historical tracking not yet implemented'
                })
                
            except Exception as e:
                logger.error(f"Error getting regime history: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'history': []
                }), 500
    
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
                logger.error(f"Error-Dashboard server error: {e}")
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        self.running = True
        
        import time
        time.sleep(0.5)
        
        logger.info(f"\nDashboard Web Started")
        logger.info(f"{'─' * 45}")
        logger.info(f"Local:   http://localhost:{port}")
        logger.info(f"Network: http://127.0.0.1:{port}")
        logger.info(f"LAN:     http://<your-ip>:{port}")
        logger.info(f"{'─' * 45}\n")
    
    def stop(self):
        """Detiene el servidor"""
        self.running = False
        logger.info("Dashboard server stopped")


def create_dashboard_template(base_dir):
    """Crea el archivo HTML del dashboard si no existe en ruta común"""
    api_dir = os.path.join(os.path.dirname(__file__))
    templates_dir = os.path.join(api_dir, 'templates')
    os.makedirs(templates_dir, exist_ok=True)
    
    template_path = os.path.join(templates_dir, 'dashboard.html')
    
    if os.path.exists(template_path):
        return
    
    logger.warning(f"WAR-Creating template at {template_path}")
    logger.warning(f"WAR-Please ensure the complete dashboard.html is in place")


if __name__ == '__main__':
    logger.info("This module should be imported, not run directly")
    logger.info("Usage:")
    logger.info("from ZX_BOT_backend import DashboardServer")
    logger.info("dashboard = DashboardServer(...)")
    logger.info("dashboard.start()")