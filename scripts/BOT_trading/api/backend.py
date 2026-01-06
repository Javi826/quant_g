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
from flask import Flask, render_template, jsonify, send_from_directory, request
import logging

logger = logging.getLogger('BOT_trading.api.backend')

from analytics.metrics import MetricsCalculator


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
        
        Args:
            price: Precio del activo
        
        Returns:
            int: Número de decimales recomendados
        """
        price = abs(float(price))
        
        if price >= 10000:      # BTC, BNB (50000.0)
            return 1
        elif price >= 1000:     # ETH alto (3000.0)
            return 2
        elif price >= 100:      # ETH, SOL (300.0)
            return 2
        elif price >= 10:       # ADA, DOT (15.0)
            return 3
        elif price >= 1:        # DOGE, XRP (1.5)
            return 4
        elif price >= 0.01:     # Coins medianos (0.05)
            return 5
        else:                   # SHIB, PEPE (0.00001)
            return 5
    
    def _load_trades_dataframe(self):
        """
        Carga y valida el DataFrame de trades desde el archivo Excel.
        
        Returns:
            pd.DataFrame: DataFrame de trades si existe y tiene datos
            None: Si el archivo no existe o está vacío
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
        
        Args:
            df: DataFrame crudo de trades
        
        Returns:
            pd.DataFrame: DataFrame con columnas adicionales
        """
        df = df.copy()
        df['OPEN_AT'] = pd.to_datetime(df['OPEN_AT'])
        df['CLOSE_AT'] = pd.to_datetime(df['CLOSE_AT'])
        df['CLOSE_DATE'] = pd.to_datetime(df['CLOSE_AT'])
        df['DURATION'] = (df['CLOSE_AT'] - df['OPEN_AT']).dt.total_seconds() / 86400
        return df
    
    @staticmethod
    def _extract_number_from_id(strategy_id):
        """
        Extrae el número prefijo del ID de estrategia.
        
        Formato esperado: 'NN_nombre_estrategia' donde NN es un número de 2 dígitos.
        Ejemplo: '02_reversal_long_4H' → '02'
        
        Args:
            strategy_id: ID de la estrategia
        
        Returns:
            str: Número extraído o '??' si no se puede extraer
        """
        if not strategy_id or not isinstance(strategy_id, str):
            return '??'
        
        # Buscar patrón NN_ al inicio (2 dígitos seguidos de guión bajo)
        match = re.match(r'^(\d{2})_', strategy_id)
        if match:
            return match.group(1)
        
        # Fallback: buscar cualquier número al inicio
        match = re.match(r'^(\d+)', strategy_id)
        if match:
            return match.group(1).zfill(2)
        
        return '??'
    
    def _calculate_capital_allocation(self, num_strategies=None):
        """
        Calcula el capital asignado por estrategia.
        
        Args:
            num_strategies: Número de estrategias para dividir el capital
        
        Returns:
            float: Capital asignado por estrategia
        """
        if num_strategies is None:
            num_strategies = len(self.implemented_strategies)
        
        if num_strategies == 0:
            return 0.0
        
        return self.initial_capital / num_strategies
    
    def _get_full_strategies_list_with_numbers(self):
        """
        Genera la lista completa de estrategias con numeración extraída de los IDs.
        
        Los números se extraen directamente de los IDs con prefijo numérico (ej: '02_reversal_long_4H').
        
        Returns:
            list: Lista de estrategias con información completa y campo 'number'
        """
        declared_names = {s['name'] for s in self.strategies}
        
        strategies_list = []
        
        # Procesar estrategias declaradas
        for strat in self.strategies:
            is_active = strat.get('active', True)
            
            if is_active:
                status = 'ACTIVE'
            else:
                status = 'DEPRECATING'
            
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
        
        # Procesar estrategias no declaradas (implementadas pero no en config)
        not_declared = self.implemented_strategies - declared_names
        for name in sorted(not_declared):
            strategies_list.append({
                'id': name,
                'name': name,
                'timeframe': 'N/A',
                'direction': 'N/A',
                'status': 'NOT IMPLE.',
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
        
        # Ordenar por ID (orden alfabético, que respeta el prefijo numérico)
        strategies_list.sort(key=lambda x: x['id'])
        
        # Extraer número del ID para cada estrategia
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
                
                # Filter logs: remove ANSI codes and extract message
                import re
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                
                clean_lines = []
                for line in new_lines:
                    # Remove ANSI color codes
                    line = ansi_escape.sub('', line)
                    line = line.strip()
                    
                    if line:
                        if ' - ' in line:
                            # Get only the message (last part after ' - ')
                            message = line.split(' - ')[-1]
                            clean_lines.append(message)
                        else:
                            # If no ' - ' separator, keep the whole line
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
                            
                            # Calcular PnL
                            if direction == 'long':
                                pnl = (float(current_price) - entry_price) * size
                            else:
                                pnl = (entry_price - float(current_price)) * size
                            
                            candles = state.get('strategy_candles', {}).get(strategy_id, 0)
                            
                            # PRECISIÓN DINÁMICA
                            precision = self.get_precision_for_price(current_price)
                            
                            current_price_rounded = round(float(current_price), precision)
                            tp_rounded = round(tp_price, precision)
                            sl_rounded = round(sl_price, precision)
                            entry_rounded = round(entry_price, precision)
                            
                            # CALCULAR DISTANCIAS (Δ%)
                            if direction == 'long':
                                # LONG: TP arriba, SL abajo
                                distance_to_tp = ((tp_price - float(current_price)) / float(current_price)) * 100
                                distance_to_sl = ((float(current_price) - sl_price) / float(current_price)) * 100
                            else:
                                # SHORT: TP abajo, SL arriba
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
                
                # ⭐ CORREGIDO: Buscar main.py en vez de BOT_orchestator_WS.py
                result = subprocess.run(
                    ['pgrep', '-f', f'main.py --account {self.account_number}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    pid = int(result.stdout.strip())
                    
                    # Enviar SIGTERM (señal de terminación limpia)
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
                
                # ⭐ CORREGIDO: Buscar main.py en vez de BOT_orchestator_WS.py
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
                df = self._load_trades_dataframe()
                if df is None:
                    return jsonify([])
                
                df = self._prepare_trades_dataframe(df)
                
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
                    import ZX_BOT_websocket
                    if ZX_BOT_websocket._ws_manager:
                        ws = ZX_BOT_websocket._ws_manager
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
                
                df = self._load_trades_dataframe()
                if df is None:
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
                        
                        # Extraer números directamente de los IDs
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
                            'ulcer_index': metrics['ulcer_index'],
                            'sharpe_ratio': metrics['sharpe_ratio']
                        })
                
                if metric == 'max_dd':
                    results_sorted = sorted(results, key=lambda x: x[metric], reverse=True)
                elif metric == 'ulcer_index':
                    results_sorted = sorted(results, key=lambda x: x[metric], reverse=False)
                else:
                    results_sorted = sorted(results, key=lambda x: x[metric], reverse=True)
                
                return jsonify(results_sorted[:10])
                
            except Exception as e:
                print(f"Error-in compose: {e}")
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
                
                if not selected_strategies:
                    return jsonify({'error': 'Error-No strategies selected'}), 400
                
                df = df[df['STRATEGY'].isin(selected_strategies)]
                
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
                        'ulcer_index': 0,
                        'sharpe_ratio': 0,
                        'message': 'No trades found for selected strategies'
                    })
                
                df = self._prepare_trades_dataframe(df)
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
                    'ulcer_index': metrics_data['ulcer_index'],
                    'sharpe_ratio': metrics_data['sharpe_ratio']
                })
                
            except Exception as e:
                return jsonify({'error': str(e)}), 500
    
    def start(self, host='0.0.0.0', port=5000):
        """Inicia el servidor del dashboard"""
        if self.running:
            print("⚠️  Dashboard already running")
            return
        
        def run_server():
            import logging  # ← Añadir este import
            log = logging.getLogger('werkzeug')  # ← CORRECTO
            log.setLevel(logging.ERROR)  # ← Cambiar logger.ERROR a logging.ERROR
            
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
    # Templates should be in api/ directory
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