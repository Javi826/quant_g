"""
backend.py
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
import schedule
import time as time_module
import requests
from market_data.websocket_manager import get_ws_manager
logger = logging.getLogger('BOT_trading.api.backend')
from config.settings import SLIPPAGE_WARNING_PCT, SLIPPAGE_CRITICAL_PCT

import psycopg2
from api.metrics import MetricsCalculator
from market_regime.regime_classifier import get_regime_info
from config.settings import REGIME_FAMILIES, REGIME_GENERAL
from config.settings import POSTGRES_CONFIG, RISK_LIMITS, LEVERAGE


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
        # PostgreSQL configuration
        self.postgres_config = POSTGRES_CONFIG
        
        self._register_routes()
        
        self.server_thread = None
        self.running = False
        
        self.snapshot_thread = None
        self.snapshot_running = False
        self.dashboard_port = None
        
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
            Load and validate trades DataFrame from PostgreSQL.
            """
            try:
                conn = psycopg2.connect(**self.postgres_config)
                query = f"SELECT * FROM trades WHERE account = '{self.account_number}'"
                df = pd.read_sql(query, conn)
                conn.close()
                
                if df.empty:
                    return None
                
                # Rename columns to match Excel format (for compatibility)
                df.rename(columns={
                    'open_at': 'OPEN_AT',
                    'close_at': 'CLOSE_AT',
                    'duration_days': 'DURATION_DAYS',
                    'strategy': 'STRATEGY',
                    'symbol': 'SYMBOL',
                    'direction': 'DIRECTION',
                    'usdt_amount': 'USDT_AMOUNT',
                    'size': 'SIZE',
                    'price_entry': 'PRICE_ENTRY',
                    'price_close': 'PRICE_CLOSE',
                    'profit': 'PROFIT',
                    'fee': 'FEE',
                    'profit_pct': 'PROFIT_PCT',
                    'reason_out': 'REASON_OUT',
                    'regime_family': 'REGIME_FAMILY',
                    'regime_multiplier': 'REGIME_MULTIPLIER',
                    'market_direction': 'MARKET_DIRECTION',
                    'direction_multiplier': 'DIRECTION_MULTIPLIER',
                    'tp_target': 'TP_TARGET',      # ← AÑADIR
                    'sl_target': 'SL_TARGET'
                }, inplace=True)
                
                return df
            except Exception as e:
                logger.error(f"Error loading trades from PostgreSQL: {e}")
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
    
    
    def _load_state(self):
            """
            Load bot state from PostgreSQL.
            
            Returns:
                Dictionary with 'positions' and 'strategy_candles'
            """
            try:
                conn = psycopg2.connect(**self.postgres_config)
                cursor = conn.cursor()
                
                cursor.execute(
                    "SELECT state_data FROM bot_state WHERE account = %s",
                    (self.account_number,)
                )
                
                result = cursor.fetchone()
                cursor.close()
                conn.close()
                
                if result:
                    return result[0]  # JSONB data
                else:
                    return {'positions': {}, 'strategy_candles': {}}
                    
            except Exception as e:
                logger.error(f"Error loading state from PostgreSQL: {e}")
                return {'positions': {}, 'strategy_candles': {}}
    
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
        
        @self.app.route('/api/quality/thresholds')
        def get_quality_thresholds():
            return jsonify({
                'success': True,
                'thresholds': {
                    'slippage_warning_pct': SLIPPAGE_WARNING_PCT,
                    'slippage_critical_pct': SLIPPAGE_CRITICAL_PCT
                }
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
                state = self._load_state()
                
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
                state = self._load_state()
                
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
                                'usdt_amount': pos.get('usdt_amount', 0),
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
                
                df_sorted = df.sort_values('CLOSE_AT', ascending=False)
                
                # Take first 15 (most recent)
                recent = df_sorted.head(15).replace({np.nan: None}).to_dict('records')
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
                
                # Calculate profit per strategy first
                strategy_profits = {}
                for strategy in df['STRATEGY'].unique():
                    df_strategy = df[df['STRATEGY'] == strategy]
                    strategy_profits[strategy] = df_strategy['PROFIT'].sum()
                
                # Separate positive and negative STRATEGY profits
                total_profit_positive = sum(p for p in strategy_profits.values() if p > 0)
                total_profit_negative = sum(p for p in strategy_profits.values() if p < 0)
                
                for strategy in sorted(df['STRATEGY'].unique()):
                    df_strategy = df[df['STRATEGY'] == strategy]
                    
                    num_trades = len(df_strategy)
                    positive_trades = len(df_strategy[df_strategy['PROFIT'] > 0])
                    pct_positive = (positive_trades / num_trades * 100) if num_trades > 0 else 0
                    total_profit = strategy_profits[strategy]
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
                    
                    # Calculate Total % (contribution to respective group)
                    if total_profit > 0:
                        # Winning strategy: % of total positive strategies profit
                        total_pct = (total_profit / total_profit_positive * 100) if total_profit_positive > 0 else 0
                    elif total_profit < 0:
                        # Losing strategy: % of total negative strategies profit
                        total_pct = (total_profit / abs(total_profit_negative) * 100) if total_profit_negative < 0 else 0
                    else:
                        # Break-even strategy
                        total_pct = 0.0
                    
                    results.append({
                        'Strategy': strategy,
                        'date_fo': date_fo.strftime('%Y-%m-%d'),
                        'Trades_num': num_trades,
                        'Trades_pct': round(pct_positive, 2),
                        'Total_profit': round(total_profit, 2),
                        'Profit_pct': round(profit_pct, 2),
                        'Total_pct': round(total_pct, 2),
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
                        'order_amount': strat.get('order_amount', 0),
                        'family_sizing': strat.get('family_sizing', None)
                    })
                
                return jsonify({
                    'account': self.account_number,
                    'strategies': strategies_info
                })
            except Exception as e:
                return jsonify({'error': str(e)}), 500
            
        @self.app.route('/api/regime/strategies')
        def get_regime_strategies():
            """
            Returns strategies with their regime multipliers and direction_mode for the matrix table.
            """
            try:
                from config.settings import DIRECTION_MATRIX, DIRECTION_GENERAL
                
                strategies_info = []
                
                for idx, strat in enumerate(self.strategies, 1):
                    direction_mode = strat.get('direction_mode', 'general')
                    
                    # Get regime multipliers directly from strategy config
                    regime_trending = strat.get('regime_trending', 1.0)
                    regime_ranging = strat.get('regime_ranging', 1.0)
                    regime_volatile = strat.get('regime_volatile', 1.0)
                    
                    strategies_info.append({
                        'number': idx,
                        'id': strat['id'],
                        'direction_mode': direction_mode,
                        'regime_trending': regime_trending,
                        'regime_ranging': regime_ranging,
                        'regime_volatile': regime_volatile,
                        'active': strat.get('active', True)
                    })
                
                return jsonify({
                    'success': True,
                    'strategies': strategies_info,
                    'regime_general': REGIME_GENERAL,
                    'direction_matrix': DIRECTION_MATRIX,
                    'direction_general': DIRECTION_GENERAL
                })
                
            except Exception as e:
                logger.error(f"Error getting regime strategies: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'strategies': []
                }), 500
        
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
                            'r_squared': metrics['r_squared'],
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
                    logger.error("ERROR: DataFrame is None")
                    return jsonify([])
                
                # Check if slippage columns exist (CLOSE execution data) - LOWERCASE
                has_slippage_data = 'order_price_close' in df.columns and 'PRICE_CLOSE' in df.columns
                
                results = []
                
                for symbol in sorted(df['SYMBOL'].unique()):
                    df_symbol = df[df['SYMBOL'] == symbol]
                    
                    # Existing metrics
                    total_trades = len(df_symbol)
                    positive_trades = len(df_symbol[df_symbol['PROFIT'] > 0])
                    win_pct = (positive_trades / total_trades * 100) if total_trades > 0 else 0
                    total_profit = df_symbol['PROFIT'].sum()
                    avg_profit = total_profit / total_trades if total_trades > 0 else 0
                    
                    # Calculate slippage metrics from CLOSE execution data
                    slippage_total = None
                    slippage_l30 = None
                    
                    if has_slippage_data:
                        df_with_slippage = df_symbol[
                            df_symbol['order_price_close'].notna() & 
                            df_symbol['PRICE_CLOSE'].notna()
                        ].copy()
                        
                        # Total slippage
                        if len(df_with_slippage) > 0:
                            df_with_slippage['slippage_pct'] = (
                                (df_with_slippage['PRICE_CLOSE'] - df_with_slippage['order_price_close']) 
                                / df_with_slippage['order_price_close'] 
                                * 100
                            )
                            slippage_total = df_with_slippage['slippage_pct'].mean()
                        
                        # Last 30 trades slippage
                        if len(df_with_slippage) > 0:
                            df_last30 = df_with_slippage.tail(30)
                            if len(df_last30) > 0:
                                df_last30['slippage_pct'] = (
                                    (df_last30['PRICE_CLOSE'] - df_last30['order_price_close']) 
                                    / df_last30['order_price_close'] 
                                    * 100
                                )
                                slippage_l30 = df_last30['slippage_pct'].mean()
                    
                    results.append({
                        'Symbol': symbol,
                        'Total_Trades': total_trades,
                        'Win_Pct': round(win_pct, 2),
                        'Total_Profit': round(total_profit, 2),
                        'Avg_Profit': round(avg_profit, 2),
                        'Slippage_Total': round(slippage_total, 2) if slippage_total is not None else None,
                        'Slippage_L30': round(slippage_l30, 2) if slippage_l30 is not None else None
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
                
                # DESPUÉS:
                metrics_data = MetricsCalculator.calculate_all_metrics(
                    df=df,
                    capital_assigned=capital_assigned,
                    include_profit_pct=False
                )
                
                daily_profit = metrics_data['daily_profit']
                
                if not daily_profit.empty:
                    daily_profit['date_str'] = daily_profit['date'].astype(str)
                    
                    if capital_assigned > 0:
                        daily_profit['equity_pct'] = ((daily_profit['equity_usd'] / capital_assigned) - 1) * 100
                    else:
                        daily_profit['equity_pct'] = 0
                    
                    daily_profit['peak_usd'] = daily_profit['equity_usd'].cummax()
                    daily_profit['drawdown_pct'] = ((daily_profit['peak_usd'] - daily_profit['equity_usd']) / daily_profit['peak_usd']) * 100
                    
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
                    'r_squared': metrics_data['r_squared'],
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
                
                # Get selected strategies and metric from request
                data = request.get_json()
                selected_strategies = data.get('strategies', [])
                metric = data.get('metric', 'profit')  # 'profit' or 'drawdown'
                
                if not selected_strategies:
                    return jsonify({'error': 'No strategies selected'}), 400
                
                # Load trades from PostgreSQL
                df = self._load_trades_dataframe()
                
                if df is None or df.empty:
                    return jsonify({'error': 'No trades data available'}), 404
                
                # Filter by selected strategies
                df = df[df['STRATEGY'].isin(selected_strategies)]
                
                if df.empty:
                    return jsonify({'error': 'No trades found for selected strategies'}), 404
                
                # Group by strategy and date
                returns_by_strategy = {}
                for strat_id in selected_strategies:
                    strat_df = df[df['STRATEGY'] == strat_id].copy()
                    
                    if strat_df.empty:
                        continue
                    
                    # Convert to datetime and get date
                    strat_df['date'] = pd.to_datetime(strat_df['CLOSE_AT']).dt.date
                    
                    if metric == 'profit':
                        # Sum daily profits
                        daily_returns = strat_df.groupby('date')['PROFIT'].sum()
                    else:  # drawdown
                        # Calculate daily drawdown
                        daily_returns = strat_df.groupby('date')['PROFIT'].sum()
                        cumulative = daily_returns.cumsum()
                        running_max = cumulative.cummax()
                        daily_drawdown = cumulative - running_max
                        daily_returns = daily_drawdown
                    
                    returns_by_strategy[strat_id] = daily_returns
                
                if len(returns_by_strategy) < 2:
                    return jsonify({'error': 'Need at least 2 strategies with trades'}), 400
                
                # Create DataFrame and calculate correlation
                returns_df = pd.DataFrame(returns_by_strategy)
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
                
                # Convert matrix to dict
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
                    'high_corr_pairs': high_corr_pairs,
                    'metric': metric
                })
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/api/analytics/regime')
        def get_regime_analytics():
            """
            Analytics by market regime: trades, P&L, and win rate per regime family.
            
            Returns:
                JSON with stats per regime (trending, ranging, volatile, unknown)
                
            Example response:
            {
                "success": true,
                "data": {
                    "trending": {
                        "trades": 45,
                        "total_trades": 120,
                        "pnl": 156.80,
                        "winrate": 68.9
                    },
                    "ranging": {
                        "trades": 32,
                        "total_trades": 120,
                        "pnl": 45.20,
                        "winrate": 53.1
                    },
                    "volatile": {
                        "trades": 12,
                        "total_trades": 120,
                        "pnl": -23.40,
                        "winrate": 41.7
                    },
                    "unknown": {
                        "trades": 5,
                        "total_trades": 120,
                        "pnl": 12.10,
                        "winrate": 60.0
                    }
                }
            }
            """
            try:
                df = self._load_trades_dataframe()
                if df is None:
                    return jsonify({'error': 'No trades data available'}), 404
                
                # Check if regime columns exist
                if 'REGIME_FAMILY' not in df.columns:
                    return jsonify({
                        'error': 'No regime data in trades file (old format)',
                        'data': {}
                    }), 200
                
                # Fill NaN with 'unknown'
                df['REGIME_FAMILY'] = df['REGIME_FAMILY'].fillna('unknown')
                
                # Calculate total trades
                total_trades = len(df)
                
                # Group by regime
                results = {}
                for regime in df['REGIME_FAMILY'].unique():
                    df_regime = df[df['REGIME_FAMILY'] == regime]
                    
                    trades_count = len(df_regime)
                    pnl = df_regime['PROFIT'].sum()
                    positive_trades = len(df_regime[df_regime['PROFIT'] > 0])
                    winrate = (positive_trades / trades_count * 100) if trades_count > 0 else 0
                    
                    results[regime] = {
                        'trades': int(trades_count),
                        'total_trades': int(total_trades),
                        'pnl': round(float(pnl), 2),
                        'winrate': round(float(winrate), 1)
                    }
                
                return jsonify({
                    'success': True,
                    'data': results
                })
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500
            
        @self.app.route('/api/regime/strategy-breakdown')
        def get_regime_strategy_breakdown():
            """
            Performance breakdown by strategy across market regimes and directions.
            
            Query params:
                date_from: YYYY-MM-DD (optional)
                date_to: YYYY-MM-DD (optional)
            
            Returns:
                JSON with per-strategy stats across regimes and directions
            """
            try:
                date_from = request.args.get('date_from', None)
                date_to = request.args.get('date_to', None)
                
                # Load trades
                df = self._load_trades_dataframe()
                if df is None or df.empty:
                    return jsonify({
                        'success': False,
                        'error': 'No trades data available',
                        'data': []
                    })
                
                # Check if regime columns exist
                if 'REGIME_FAMILY' not in df.columns or 'MARKET_DIRECTION' not in df.columns:
                    return jsonify({
                        'success': False,
                        'error': 'No regime/direction data in trades file',
                        'data': []
                    })
                
                # Prepare and filter by dates
                df = self._prepare_trades_dataframe(df)
                df = self._filter_df_by_dates(df, date_from, date_to)
                
                if df.empty:
                    return jsonify({
                        'success': True,
                        'data': []
                    })
                
                # Fill NaN values
                df['REGIME_FAMILY'] = df['REGIME_FAMILY'].fillna('unknown')
                df['MARKET_DIRECTION'] = df['MARKET_DIRECTION'].fillna('unknown')
                
                # Get unique strategies
                strategies = sorted(df['STRATEGY'].unique())
                
                results = []
                
                for idx, strategy in enumerate(strategies, 1):
                    df_strat = df[df['STRATEGY'] == strategy]
                    
                    # Global stats
                    total_trades = len(df_strat)
                    positive_trades = len(df_strat[df_strat['PROFIT'] > 0])
                    win_rate = (positive_trades / total_trades * 100) if total_trades > 0 else 0
                    total_profit = df_strat['PROFIT'].sum()
                    
                    # Helper function to calculate regime/direction stats
                    def get_stats(filtered_df):
                        if len(filtered_df) == 0:
                            return {'trades': 0, 'win_pct': 0}
                        
                        trades = len(filtered_df)
                        wins = len(filtered_df[filtered_df['PROFIT'] > 0])
                        win_pct = (wins / trades * 100) if trades > 0 else 0
                        
                        return {
                            'trades': int(trades),
                            'win_pct': round(float(win_pct), 1)
                        }
                    
                    # Calculate stats for each regime
                    trending_stats = get_stats(df_strat[df_strat['REGIME_FAMILY'] == 'trending'])
                    ranging_stats = get_stats(df_strat[df_strat['REGIME_FAMILY'] == 'ranging'])
                    volatile_stats = get_stats(df_strat[df_strat['REGIME_FAMILY'] == 'volatile'])
                    
                    # Calculate stats for each direction
                    uptrend_stats = get_stats(df_strat[df_strat['MARKET_DIRECTION'] == 'uptrend'])
                    downtrend_stats = get_stats(df_strat[df_strat['MARKET_DIRECTION'] == 'dwtrend'])
                    
                    results.append({
                        'number': idx,
                        'strategy': strategy,
                        'total_trades': int(total_trades),
                        'win_rate': round(float(win_rate), 1),
                        'profit': round(float(total_profit), 2),
                        'trending': trending_stats,
                        'ranging': ranging_stats,
                        'volatile': volatile_stats,
                        'uptrend': uptrend_stats,
                        'downtrend': downtrend_stats
                    })
                
                return jsonify({
                    'success': True,
                    'data': results
                })
                
            except Exception as e:
                logger.error(f"Error in regime strategy breakdown: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'data': []
                }), 500
            
        @self.app.route('/api/analytics/market-direction')
        def get_market_direction_analytics():
            """
            Analytics by market direction: trades, P&L, and win rate per direction.
            """
            try:
                df = self._load_trades_dataframe()
                if df is None:
                    return jsonify({'error': 'No trades data available'}), 404
                
                # Check if column exists
                if 'MARKET_DIRECTION' not in df.columns:
                    return jsonify({
                        'error': 'No market direction data in trades file',
                        'data': {}
                    }), 200
                
                # Fill NaN with 'unknown'
                df['MARKET_DIRECTION'] = df['MARKET_DIRECTION'].fillna('unknown')
                
                # Calculate total trades
                total_trades = len(df)
                
                # Group by direction
                results = {}
                for direction in df['MARKET_DIRECTION'].unique():
                    df_dir = df[df['MARKET_DIRECTION'] == direction]
                    
                    trades_count = len(df_dir)
                    pnl = df_dir['PROFIT'].sum()
                    positive_trades = len(df_dir[df_dir['PROFIT'] > 0])
                    winrate = (positive_trades / trades_count * 100) if trades_count > 0 else 0
                    
                    results[direction] = {
                        'trades': int(trades_count),
                        'total_trades': int(total_trades),
                        'pnl': round(float(pnl), 2),
                        'winrate': round(float(winrate), 1)
                    }
                
                return jsonify({
                    'success': True,
                    'data': results
                })
                
            except Exception as e:
                import traceback
                traceback.print_exc()
                return jsonify({'error': str(e)}), 500

        # ==============================================================================
        # EXACT LOCATION IN FILE
        # ==============================================================================
        
        @self.app.route('/api/regime/current')
        def get_regime_current():
            """
            Obtiene el régimen de mercado actual para un timeframe específico.
            
            Query params:
                timeframe: Timeframe a analizar (ej: '4H', '1H', '6Hutc')
            
            Returns:
                JSON con family, multiplier, metrics, thresholds, all_families, all_thresholds
            """
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
                    'btc_price': regime_info.get('btc_price'),           
                    'btc_ma50': regime_info.get('btc_ma50'),             
                    'btc_trend': regime_info.get('btc_trend'),           
                    'all_families': REGIME_GENERAL,
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
        
        # ==============================================================================
        # BTC DATA ENDPOINTS
        # ==============================================================================
        
        @self.app.route('/api/btc/history')
        def get_btc_history():
            """
            Get BTC price history for chart overlay.
            
            Query params:
                date_from: YYYY-MM-DD (optional)
                date_to: YYYY-MM-DD (optional)
            
            Returns:
                JSON with dates and prices arrays
            """
            try:
                date_from = request.args.get('date_from')
                date_to = request.args.get('date_to')
                
                # Build query
                query = """
                    SELECT date, price
                    FROM btc_history
                    WHERE 1=1
                """
                params = []
                
                if date_from:
                    query += " AND date >= %s"
                    params.append(date_from)
                
                if date_to:
                    query += " AND date <= %s"
                    params.append(date_to)
                
                query += " ORDER BY date"
                
                # Execute query
                conn = psycopg2.connect(**self.postgres_config)
                cursor = conn.cursor()
                cursor.execute(query, params)
                rows = cursor.fetchall()
                cursor.close()
                conn.close()
                
                if not rows:
                    return jsonify({
                        'success': True,
                        'dates': [],
                        'prices': []
                    })
                
                # Format response
                dates = [row[0].strftime('%Y-%m-%d') for row in rows]
                prices = [float(row[1]) if row[1] else None for row in rows]
                
                return jsonify({
                    'success': True,
                    'dates': dates,
                    'prices': prices
                })
                
            except Exception as e:
                logger.error(f"Error getting BTC history: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
            
        @self.app.route('/api/risk/exposure')
        def get_risk_exposure():
            """
            Get current risk exposure snapshot.
            
            Returns:
                JSON with gross/net exposure metrics and per-strategy breakdown
            """
            try:
                state = self._load_state()
                
                # Calculate available capital (initial + closed P&L)
                df = self._load_trades_dataframe()
                closed_pnl = df['PROFIT'].sum() if df is not None and not df.empty else 0
                available_capital = self.initial_capital + closed_pnl
                
                # Calculate exposure from open positions
                total_long_usdt = 0
                total_short_usdt = 0
                positions_by_strategy = []
                
                for strategy_id, positions in state.get('positions', {}).items():
                    for pos in positions:
                        usdt_amount = float(pos.get('usdt_amount', 0))
                        real_exposure = usdt_amount / LEVERAGE  # Divide by leverage
                        direction = pos['direction'].lower()
                        
                        if direction == 'long':
                            total_long_usdt += real_exposure
                        else:
                            total_short_usdt += real_exposure
                        
                        positions_by_strategy.append({
                            'strategy': strategy_id,
                            'symbol': pos['symbol'],
                            'side': direction.upper(),
                            'usdt': real_exposure 
                        })
                
                # Calculate exposure percentages
                gross_exposure_pct = ((total_long_usdt + total_short_usdt) / available_capital * 100) if available_capital > 0 else 0
                net_exposure_pct = ((total_long_usdt - total_short_usdt) / available_capital * 100) if available_capital > 0 else 0
                long_exposure_pct = (total_long_usdt / available_capital * 100) if available_capital > 0 else 0
                short_exposure_pct = (total_short_usdt / available_capital * 100) if available_capital > 0 else 0
                
                # Aggregate by strategy
                strategy_exposure = {}
                for pos in positions_by_strategy:
                    strat = pos['strategy']
                    if strat not in strategy_exposure:
                        strategy_exposure[strat] = {
                            'side': pos['side'],
                            'usdt': 0,
                            'pct': 0
                        }
                    strategy_exposure[strat]['usdt'] += pos['usdt']
                
                # Calculate percentages
                for strat, data in strategy_exposure.items():
                    data['pct'] = (data['usdt'] / available_capital * 100) if available_capital > 0 else 0
                
                # Sort by exposure DESC
                strategy_list = [
                    {'strategy': k, **v} 
                    for k, v in sorted(strategy_exposure.items(), key=lambda x: x[1]['usdt'], reverse=True)
                ]
                
                return jsonify({
                    'success': True,
                    'metrics': {
                        'gross_exposure_pct': round(gross_exposure_pct, 2),
                        'net_exposure_pct': round(net_exposure_pct, 2),
                        'long_exposure_pct': round(long_exposure_pct, 2),
                        'short_exposure_pct': round(short_exposure_pct, 2),
                        'num_positions': sum(len(positions) for positions in state.get('positions', {}).values()),
                        'available_capital': round(available_capital, 2)
                    },
                    'strategies': strategy_list,
                    'limits': {
                        'max_gross': RISK_LIMITS['max_gross_exposure_pct'],
                        'max_net': RISK_LIMITS['max_net_exposure_pct']
                    }
                })
                
            except Exception as e:
                logger.error(f"Error getting risk exposure: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
            
        @self.app.route('/api/risk/exposure-history')
        def get_risk_exposure_history():
            """
            Get historical risk exposure data.
            
            Query params:
                days: Number of days to retrieve (default: 30)
                date_from: Start date (YYYY-MM-DD, optional)
                date_to: End date (YYYY-MM-DD, optional)
            
            Returns:
                JSON with historical exposure data
            """
            try:
                from datetime import date, datetime, timedelta
                
                # Get filter parameters
                days = int(request.args.get('days', 30))
                date_from_str = request.args.get('date_from')
                date_to_str = request.args.get('date_to')
                
                # Check if we need to capture today's snapshot
                conn = psycopg2.connect(**self.postgres_config)
                cursor = conn.cursor()
                
                today = date.today()
                
                cursor.execute("""
                    SELECT date FROM exposure_history 
                    WHERE account = %s AND date = %s
                """, [self.account_number, today])
                
                exists = cursor.fetchone()
                
                if not exists:
                    # Capture today's snapshot
                    state = self._load_state()
                    df = self._load_trades_dataframe()
                    closed_pnl = df['PROFIT'].sum() if df is not None and not df.empty else 0
                    available_capital = self.initial_capital + closed_pnl
                    
                    total_long_usdt = 0
                    total_short_usdt = 0
                    num_positions = 0
                    
                    for strategy_id, positions in state.get('positions', {}).items():
                        num_positions += len(positions)
                        for pos in positions:
                            usdt_amount = float(pos.get('usdt_amount', 0))
                            real_exposure = usdt_amount / LEVERAGE  # Divide by leverage
                            if pos['direction'].lower() == 'long':
                                total_long_usdt += real_exposure
                            else:
                                total_short_usdt += real_exposure
                    
                    gross_pct = ((total_long_usdt + total_short_usdt) / available_capital * 100) if available_capital > 0 else 0
                    net_pct = ((total_long_usdt - total_short_usdt) / available_capital * 100) if available_capital > 0 else 0
                    long_pct = (total_long_usdt / available_capital * 100) if available_capital > 0 else 0
                    short_pct = (total_short_usdt / available_capital * 100) if available_capital > 0 else 0
                    
                    cursor.execute("""
                        INSERT INTO exposure_history 
                        (date, account, gross_exposure_pct, net_exposure_pct, long_exposure_pct, short_exposure_pct, num_positions)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (date, account) DO NOTHING
                    """, [today, self.account_number, 
                          float(gross_pct), float(net_pct), float(long_pct), float(short_pct), int(num_positions)])
                    
                    conn.commit()
                
                # Build query based on filters
                if date_from_str and date_to_str:
                    # Use explicit date range
                    query = """
                        SELECT date, gross_exposure_pct, net_exposure_pct, 
                               long_exposure_pct, short_exposure_pct, num_positions
                        FROM exposure_history
                        WHERE account = %s AND date >= %s AND date <= %s
                        ORDER BY date ASC
                    """
                    params = [self.account_number, date_from_str, date_to_str]
                elif date_from_str:
                    # From date to today
                    query = """
                        SELECT date, gross_exposure_pct, net_exposure_pct, 
                               long_exposure_pct, short_exposure_pct, num_positions
                        FROM exposure_history
                        WHERE account = %s AND date >= %s
                        ORDER BY date ASC
                    """
                    params = [self.account_number, date_from_str]
                elif date_to_str:
                    # From 30 days ago to date_to
                    query = """
                        SELECT date, gross_exposure_pct, net_exposure_pct, 
                               long_exposure_pct, short_exposure_pct, num_positions
                        FROM exposure_history
                        WHERE account = %s AND date <= %s AND date >= %s
                        ORDER BY date ASC
                    """
                    date_to = datetime.strptime(date_to_str, '%Y-%m-%d').date()
                    date_from = date_to - timedelta(days=days)
                    params = [self.account_number, date_to_str, date_from]
                else:
                    # Default: last N days
                    query = """
                        SELECT date, gross_exposure_pct, net_exposure_pct, 
                               long_exposure_pct, short_exposure_pct, num_positions
                        FROM exposure_history
                        WHERE account = %s AND date >= CURRENT_DATE - INTERVAL '%s days'
                        ORDER BY date ASC
                    """
                    params = [self.account_number, days]
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                cursor.close()
                conn.close()
                
                if not rows:
                    return jsonify({
                        'success': True,
                        'history': {
                            'dates': [],
                            'gross': [],
                            'net': [],
                            'long': [],
                            'short': [],
                            'positions': []
                        }
                    })
                
                dates = [row[0].strftime('%Y-%m-%d') for row in rows]
                gross = [float(row[1]) if row[1] else 0 for row in rows]
                net = [float(row[2]) if row[2] else 0 for row in rows]
                long_exp = [float(row[3]) if row[3] else 0 for row in rows]
                short_exp = [float(row[4]) if row[4] else 0 for row in rows]
                positions = [int(row[5]) if row[5] else 0 for row in rows]
                
                return jsonify({
                    'success': True,
                    'history': {
                        'dates': dates,
                        'gross': gross,
                        'net': net,
                        'long': long_exp,
                        'short': short_exp,
                        'positions': positions
                    }
                })
                
            except Exception as e:
                logger.error(f"Error getting exposure history: {e}")
                return jsonify({'success': False, 'error': str(e)}), 500
            
        @self.app.route('/api/quality/drift')
        def get_quality_drift():
            """
            Get drift analysis for all strategies.
            
            Returns:
                JSON with drift status per strategy
            """
            try:
                from quality_control.analyzer import analyze_drift_status
                
                # Load trades from PostgreSQL
                df = self._load_trades_dataframe()
                if df is None or df.empty:
                    return jsonify({
                        'success': False,
                        'error': 'No trades data available',
                        'data': {}
                    })
                
                # Get strategies config
                strategies_list = self._get_full_strategies_list_with_numbers()
                
                # Filter only ACTIVE and DEPRECATING strategies
                active_strategies = [
                    s for s in strategies_list 
                    if s['status'] in ('ACTIVE', 'DEPRECATING')
                ]
                
                # Analyze drift
                drift_results = analyze_drift_status(df, active_strategies)
                
                return jsonify({
                    'success': True,
                    'data': drift_results
                })
                
            except Exception as e:
                logger.error(f"Error in drift analysis: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'data': {}
                }), 500

        @self.app.route('/api/quality/execution')
        def get_quality_execution():
            try:
                from quality_control.analyzer import analyze_execution_quality
                
                df = self._load_trades_dataframe()
                if df is None or df.empty:
                    return jsonify({
                        'success': False,
                        'error': 'No trades data available',
                        'data': {}
                    })
                
                strategies_list = self._get_full_strategies_list_with_numbers()
                active_strategies = [
                    s for s in strategies_list 
                    if s['status'] in ('ACTIVE', 'DEPRECATING')
                ]
                
                execution_results = analyze_execution_quality(df, active_strategies)
                
                return jsonify({
                    'success': True,
                    'data': execution_results
                })
                
            except Exception as e:
                logger.error(f"Error in execution quality analysis: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'data': {}
                }), 500
            
        @self.app.route('/api/quality/target-deviation')
        def get_quality_target_deviation():
            """
            Get target deviation analysis (TP/SL real vs configured).
            """
            try:
                from quality_control.analyzer import analyze_target_deviation
                
                # Load trades from PostgreSQL
                df = self._load_trades_dataframe()
                if df is None or df.empty:
                    return jsonify({
                        'success': False,
                        'error': 'No trades data available',
                        'data': {}
                    })
                
                # Get strategies config
                strategies_list = self._get_full_strategies_list_with_numbers()
                
                # Filter only ACTIVE and DEPRECATING strategies
                active_strategies = [
                    s for s in strategies_list 
                    if s['status'] in ('ACTIVE', 'DEPRECATING')
                ]
                
                # Analyze target deviation
                deviation_results = analyze_target_deviation(df, active_strategies)
                
                return jsonify({
                    'success': True,
                    'data': deviation_results
                })
                
            except Exception as e:
                logger.error(f"Error in target deviation analysis: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'error': str(e),
                    'data': {}
                }), 500
            
        @self.app.route('/api/quality/winrate-evolution')
        def get_winrate_evolution():
            """
            Get cumulative win rate evolution over time for selected strategies.
            
            Query params:
                strategies: Comma-separated strategy IDs
                date_from: Start date (YYYY-MM-DD, optional)
                date_to: End date (YYYY-MM-DD, optional)
            
            Returns:
                JSON with dates and cumulative win rate percentages
            """
            try:
                # Get parameters
                strategies_param = request.args.get('strategies', '')
                selected_strategies = [s.strip() for s in strategies_param.split(',') if s.strip()]
                date_from = request.args.get('date_from', None)
                date_to = request.args.get('date_to', None)
                
                if not selected_strategies:
                    return jsonify({
                        'success': False,
                        'error': 'No strategies selected'
                    }), 400
                
                # Load trades
                df = self._load_trades_dataframe()
                if df is None or df.empty:
                    return jsonify({
                        'success': True,
                        'dates': [],
                        'winrate': []
                    })
                
                # Filter by strategies
                df = df[df['STRATEGY'].isin(selected_strategies)].copy()
                
                if df.empty:
                    return jsonify({
                        'success': True,
                        'dates': [],
                        'winrate': []
                    })
                
                # Prepare and filter by dates
                df = self._prepare_trades_dataframe(df)
                df = self._filter_df_by_dates(df, date_from, date_to)
                
                if df.empty:
                    return jsonify({
                        'success': True,
                        'dates': [],
                        'winrate': []
                    })
                
                # Sort by close date
                df = df.sort_values('CLOSE_AT')
                
                # Group by date and calculate cumulative win rate
                df['date'] = df['CLOSE_AT'].dt.date
                df['is_winner'] = (df['PROFIT'] > 0).astype(int)
                
                # Calculate cumulative stats
                df['cumulative_wins'] = df['is_winner'].cumsum()
                df['cumulative_trades'] = range(1, len(df) + 1)
                df['cumulative_winrate'] = (df['cumulative_wins'] / df['cumulative_trades']) * 100
                
                # Get one value per day (last trade of the day)
                daily = df.groupby('date').last().reset_index()
                
                # Format output
                dates = [d.strftime('%Y-%m-%d') for d in daily['date']]
                winrate = [round(wr, 2) for wr in daily['cumulative_winrate']]
                
                return jsonify({
                    'success': True,
                    'dates': dates,
                    'winrate': winrate,
                    'total_trades': int(len(df))
                })
                
            except Exception as e:
                logger.error(f"Error in winrate evolution: {e}")
                import traceback
                traceback.print_exc()
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
        
        @self.app.route('/api/btc/snapshot')
        def capture_btc_snapshot():
            """
            Capture today's BTC price snapshot.
            Auto-called by scheduler at 23:55 UTC daily.
            
            Returns:
                JSON with success status and captured price
            """
            try:
                from datetime import date
                
                today = date.today()
                
                # Check if today's snapshot already exists
                conn = psycopg2.connect(**self.postgres_config)
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT price FROM btc_history 
                    WHERE date = %s
                """, [today])
                
                existing = cursor.fetchone()
                
                if existing:
                    cursor.close()
                    conn.close()
                    return jsonify({
                        'success': True,
                        'message': 'Snapshot already exists for today',
                        'price': float(existing[0]),
                        'date': today.isoformat()
                    })
                
                # Get current BTC price
                try:
                    btc_price = float(self.get_current_price('BTCUSDT'))
                except Exception as e:
                    cursor.close()
                    conn.close()
                    logger.error(f"[BTC SNAPSHOT] Error getting BTC price: {e}")
                    return jsonify({
                        'success': False,
                        'error': f'Failed to get BTC price: {str(e)}'
                    }), 500
                
                # Insert today's snapshot
                cursor.execute("""
                    INSERT INTO btc_history (date, price)
                    VALUES (%s, %s)
                    ON CONFLICT (date) DO NOTHING
                """, [today, btc_price])
                
                conn.commit()
                cursor.close()
                conn.close()
                
                logger.info(f"[BTC SNAPSHOT] ✓ Captured: {today} -> ${btc_price:.2f}")
                
                return jsonify({
                    'success': True,
                    'message': 'BTC snapshot captured successfully',
                    'price': btc_price,
                    'date': today.isoformat()
                })
                
            except Exception as e:
                logger.error(f"[BTC SNAPSHOT] Error: {e}")
                return jsonify({
                    'success': False,
                    'error': str(e)
                }), 500
    
    # ==========================================================================
    # DAILY SNAPSHOT SCHEDULER METHODS (CLASS LEVEL)
    # ==========================================================================
    
    def _capture_snapshot(self):
        """
        Capture daily exposure snapshot by calling internal endpoint.
        Triggered by scheduler at 23:55 UTC daily.
        """
        try:
            if not self.dashboard_port:
                logger.warning("[SNAPSHOT] Port not set, skipping")
                return
            
            url = f'http://localhost:{self.dashboard_port}/api/risk/exposure-history?days=1'
            response = requests.get(url, timeout=10)
            
            if response.ok:
                data = response.json()
                if data.get('success'):
                    logger.info(f"[SNAPSHOT] ✓ Daily exposure captured for {self.account_number}")
                else:
                    logger.warning(f"[SNAPSHOT] Endpoint returned error: {data.get('error')}")
            else:
                logger.warning(f"[SNAPSHOT] HTTP {response.status_code}: {response.text[:100]}")
                
        except requests.exceptions.Timeout:
            logger.error("[SNAPSHOT] Request timeout (>10s)")
        except Exception as e:
            logger.error(f"[SNAPSHOT] Error capturing snapshot: {e}")
    
    def _capture_btc_snapshot(self):
        """
        Capture daily BTC price snapshot by calling internal endpoint.
        Triggered by scheduler at 23:55 UTC daily.
        """
        try:
            if not self.dashboard_port:
                logger.warning("[BTC SNAPSHOT] Port not set, skipping")
                return
            
            url = f'http://localhost:{self.dashboard_port}/api/btc/snapshot'
            response = requests.get(url, timeout=10)
            
            if response.ok:
                data = response.json()
                if data.get('success'):
                    logger.info(f"[BTC SNAPSHOT] ✓ Daily BTC price captured: ${data.get('price')}")
                else:
                    logger.warning(f"[BTC SNAPSHOT] Endpoint returned error: {data.get('error')}")
            else:
                logger.warning(f"[BTC SNAPSHOT] HTTP {response.status_code}: {response.text[:100]}")
                
        except requests.exceptions.Timeout:
            logger.error("[BTC SNAPSHOT] Request timeout (>10s)")
        except Exception as e:
            logger.error(f"[BTC SNAPSHOT] Error capturing snapshot: {e}")
    
    def _schedule_daily_snapshot(self):
        """
        Scheduler loop that runs in separate thread.
        Captures exposure snapshot daily at 23:55 UTC.
        Only account 00 captures BTC price (shared resource).
        """
        schedule.every().day.at("23:55").do(self._capture_snapshot)
        
        # Only account 00 captures BTC (shared across all accounts)
        if self.account_number == '00':
            schedule.every().day.at("23:55").do(self._capture_btc_snapshot)
            logger.info("[SNAPSHOT] Scheduler started - captures exposure + BTC daily at 23:55 UTC")
        else:
            logger.info("[SNAPSHOT] Scheduler started - captures exposure daily at 23:55 UTC")
        
        while self.snapshot_running:
            schedule.run_pending()
            time_module.sleep(60)  # Check every minute
        
        logger.info("[SNAPSHOT] Scheduler stopped")
    
    def _start_snapshot_scheduler(self):
        """Start snapshot scheduler thread"""
        if self.snapshot_thread and self.snapshot_thread.is_alive():
            logger.warning("[SNAPSHOT] Scheduler already running")
            return
        
        self.snapshot_running = True
        self.snapshot_thread = threading.Thread(
            target=self._schedule_daily_snapshot,
            daemon=True,
            name=f'SnapshotScheduler-{self.account_number}'
        )
        self.snapshot_thread.start()
    
    def _stop_snapshot_scheduler(self):
        """Stop snapshot scheduler thread"""
        if self.snapshot_running:
            self.snapshot_running = False
            if self.snapshot_thread:
                self.snapshot_thread.join(timeout=2)
    
    def start(self, host='0.0.0.0', port=5000):
        """Inicia el servidor del dashboard"""
        if self.running:
            print("⚠️  Dashboard already running")
            return
        
        # Store port for snapshot scheduler
        self.dashboard_port = port
        
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
        
        # Start snapshot scheduler AFTER Flask is running
        self._start_snapshot_scheduler()
    
    def stop(self):
        """Detiene el servidor"""
        self._stop_snapshot_scheduler()  # Stop scheduler first
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