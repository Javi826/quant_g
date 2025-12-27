<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Cache-Control" content="no-cache, no-store, must-revalidate">
    <meta http-equiv="Pragma" content="no-cache">
    <meta http-equiv="Expires" content="0">
    <title>BOT_trading - {{ account }}</title>
    <link rel="icon" type="image/jpeg" href="/favicon.jpg">
    <style>
        * { 
            margin: 0; 
            padding: 0; 
            box-sizing: border-box; 
        }
        
        body {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', 'Roboto', sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            height: 100vh;
            overflow: hidden;
            font-size: 15px;
            -webkit-font-smoothing: antialiased;
            -moz-osx-font-smoothing: grayscale;
        }
        
        /* ═══════════════════════════════════════════════════════════════
           LOADING SPLASH SCREEN
           ═══════════════════════════════════════════════════════════════ */
        
        #loading-splash {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            {% if account == '00' %}
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            {% elif account == 'E1' %}
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f0f0f 100%);
            {% elif account == '01' %}
            background: linear-gradient(135deg, #4b5563 0%, #6b7280 100%);
            {% else %}
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            {% endif %}
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            z-index: 99999;
            color: white;
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        }
        
        #loading-splash.hidden {
            display: none;
        }
        
        .loading-content {
            text-align: center;
            max-width: 500px;
            padding: 40px;
        }
        
        .loading-title {
            font-size: 32px;
            font-weight: 600;
            margin: 0 0 10px 0;
        }
        
        .loading-subtitle {
            font-size: 18px;
            opacity: 0.9;
            margin: 0 0 40px 0;
        }
        
        .loading-spinner {
            width: 50px;
            height: 50px;
            border: 4px solid rgba(255, 255, 255, 0.3);
            border-top-color: white;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: 0 auto 20px auto;
        }
        
        .loading-status {
            font-size: 14px;
            opacity: 0.7;
            margin: 0;
        }
        
        .loading-error {
            background: rgba(239, 68, 68, 0.2);
            border: 2px solid #ef4444;
            border-radius: 8px;
            padding: 20px;
            margin-top: 20px;
            display: none;
        }
        
        .loading-error.visible {
            display: block;
        }
        
        .loading-error-title {
            font-size: 18px;
            font-weight: 600;
            color: #fca5a5;
            margin-bottom: 10px;
        }
        
        .loading-error-text {
            font-size: 14px;
            color: #fecaca;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(0.95); }
        }
        
        /* ═══════════════════════════════════════════════════════════════
           DASHBOARD STYLES (ORIGINAL)
           ═══════════════════════════════════════════════════════════════ */
        
        .dashboard-container {
            display: grid;
            grid-template-columns: 420px 1fr;
            grid-template-rows: auto 1fr;
            height: 100vh;
            gap: 0;
        }
        
        .header {
            grid-column: 1 / -1;
            {% if account == '00' %}
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            {% elif account == 'E1' %}
            background: linear-gradient(135deg, #0a0a0a 0%, #1a1a1a 50%, #0f0f0f 100%);
            {% elif account == '01' %}
            background: linear-gradient(135deg, #4b5563 0%, #6b7280 100%);
            {% else %}
            background: linear-gradient(135deg, #1e3a8a 0%, #1e40af 100%);
            {% endif %}
            padding: 20px 32px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
            box-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        }
        
        .header h1 { 
            font-size: 22px;
            display: flex;
            align-items: center;
            gap: 12px;
            font-weight: 600;
            color: #ffffff;
            letter-spacing: -0.3px;
        }
        
        .account-badge {
            font-size: 13px;
            {% if account == '00' %}
            background: rgba(37, 99, 235, 0.25);
            border: 1px solid rgba(59, 130, 246, 0.4);
            color: #93c5fd;
            {% elif account == 'E1' %}
            font-size: 13px;
            background: rgba(255, 255, 255, 0.15);
            border: 1px solid rgba(255, 255, 255, 0.3);
            color: #e0e0e0;
            {% elif account == '01' %}
            background: rgba(156, 163, 175, 0.25);
            border: 1px solid rgba(209, 213, 219, 0.4);
            color: #d1d5db;
            {% else %}
            background: rgba(37, 99, 235, 0.25);
            border: 1px solid rgba(59, 130, 246, 0.4);
            color: #93c5fd;
            {% endif %}
            padding: 5px 12px;
            border-radius: 6px;
            margin-left: 10px;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .header-right {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        
        .status-badge {
            background: rgba(16, 185, 129, 0.15);
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 7px 16px;
            border-radius: 6px;
            font-weight: 500;
            font-size: 13px;
            color: #34d399;
            letter-spacing: 0.3px;
        }
        
        .status-badge::before {
            content: '●';
            margin-right: 6px;
        }
        
        .btn-stop {
            background: rgba(248, 81, 73, 0.15);
            color: #ff7b72;
            border: 1px solid rgba(248, 81, 73, 0.3);
            padding: 7px 16px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 13px;
            font-weight: 600;
            transition: all 0.2s ease;
            letter-spacing: 0.3px;
        }
        
        .btn-stop:hover {
            background: rgba(248, 81, 73, 0.25);
            border-color: rgba(248, 81, 73, 0.5);
            transform: translateY(-1px);
        }
        
        .stop-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.95);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 10000;
        }
        
        .stop-overlay.active {
            display: flex;
        }
        
        .stop-box {
            background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
            border: 2px solid #ff7b72;
            border-radius: 12px;
            padding: 40px;
            min-width: 600px;
            max-width: 700px;
            text-align: center;
        }
        
        .stop-box h2 {
            color: #ff7b72;
            font-size: 24px;
            margin-bottom: 20px;
        }
        
        .stop-status {
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.8;
            color: #c9d1d9;
            text-align: left;
            background: #0d1117;
            padding: 20px;
            border-radius: 8px;
            margin-top: 20px;
            max-height: 400px;
            overflow-y: auto;
        }
        
        .stop-status-line {
            margin-bottom: 4px;
        }
        
        .stop-status-line.success { color: #3fb950; }
        .stop-status-line.error { color: #f85149; }
        .stop-status-line.warning { color: #d29922; }
        
        .stop-status::-webkit-scrollbar {
            width: 8px;
        }
        
        .stop-status::-webkit-scrollbar-track {
            background: #0d1117;
        }
        
        .stop-status::-webkit-scrollbar-thumb {
            background: #30363d;
            border-radius: 4px;
        }
        
        .logs-panel {
            background: #161b22;
            border-right: 1px solid #21262d;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }
        
        .logs-header {
            background: #1c2128;
            padding: 14px 20px;
            border-bottom: 1px solid #21262d;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .logs-header h2 {
            font-size: 14px;
            display: flex;
            align-items: center;
            gap: 8px;
            font-weight: 600;
            color: #8b949e;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        
        .logs-controls {
            display: flex;
            gap: 6px;
        }
        
        .btn-small {
            background: rgba(31, 111, 235, 0.15);
            color: #58a6ff;
            border: 1px solid rgba(31, 111, 235, 0.3);
            padding: 5px 12px;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .btn-small:hover { 
            background: rgba(31, 111, 235, 0.25);
            border-color: rgba(31, 111, 235, 0.5);
        }
        
        .btn-small.danger { 
            background: rgba(248, 81, 73, 0.15);
            color: #ff7b72;
            border-color: rgba(248, 81, 73, 0.3);
        }
        
        .btn-small.danger:hover { 
            background: rgba(248, 81, 73, 0.25);
            border-color: rgba(248, 81, 73, 0.5);
        }
        
        .logs-content {
            flex: 1;
            overflow-y: auto;
            padding: 12px;
            font-family: 'SF Mono', 'Consolas', 'Monaco', 'Courier New', monospace;
            font-size: 12px;
            line-height: 0.9;
            background: #0d1117;
        }
        
        .log-line {
            padding: 4px 6px;
            margin-bottom: 1px;
            border-radius: 3px;
            word-wrap: break-word;
        }
        
        .log-line.info { color: #58a6ff; }
        .log-line.success { color: #3fb950; }
        .log-line.warning { color: #d29922; }
        .log-line.error { color: #f85149; }
        .log-line.default { color: #8b949e; }
        
        .logs-content::-webkit-scrollbar,
        .main-panel::-webkit-scrollbar {
            width: 8px;
        }
        
        .logs-content::-webkit-scrollbar-track,
        .main-panel::-webkit-scrollbar-track {
            background: #0d1117;
        }
        
        .logs-content::-webkit-scrollbar-thumb,
        .main-panel::-webkit-scrollbar-thumb {
            background: #30363d;
            border-radius: 4px;
        }
        
        .logs-content::-webkit-scrollbar-thumb:hover,
        .main-panel::-webkit-scrollbar-thumb:hover {
            background: #484f58;
        }
        
        .main-panel {
            background: #0d1117;
            overflow-y: auto;
            padding: 24px;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 14px;
            margin-bottom: 24px;
        }
        
        .stat-card {
            background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
            padding: 18px;
            border-radius: 8px;
            border: 1px solid #21262d;
            transition: all 0.2s ease;
        }
        
        .stat-card:hover {
            border-color: #30363d;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.3);
        }
        
        .stat-label {
            color: #8b949e;
            font-size: 13px;
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.8px;
            font-weight: 600;
        }
        
        .stat-value {
            font-size: 28px;
            font-weight: 700;
            letter-spacing: -0.8px;
        }
        
        .stat-value.positive { color: #3fb950; }
        .stat-value.negative { color: #f85149; }
        .stat-value.neutral { color: #58a6ff; }
        .stat-value.warning { color: #d29922; }
        
        .content-section {
            background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
            border-radius: 8px;
            padding: 24px;
            border: 1px solid #21262d;
            margin-bottom: 20px;
        }
        
        .content-section h2 {
            margin-bottom: 20px;
            color: #c9d1d9;
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 20px;
            font-weight: 600;
            letter-spacing: -0.3px;
        }
        
        .tabs-container {
            display: flex;
            gap: 4px;
            margin-bottom: 20px;
            border-bottom: 1px solid #21262d;
            padding-bottom: 0;
        }
        
        .tab-btn {
            background: transparent;
            color: #8b949e;
            border: none;
            border-bottom: 2px solid transparent;
            padding: 10px 16px;
            cursor: pointer;
            font-size: 16px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .tab-btn:hover {
            color: #c9d1d9;
            background: rgba(255, 255, 255, 0.03);
        }
        
        .tab-btn.active {
            color: #58a6ff;
            border-bottom-color: #58a6ff;
            background: transparent;
        }
        
        .tab-content {
            display: none;
        }
        
        .tab-content.active {
            display: block;
        }
        
        .view-selector {
            display: flex;
            gap: 0;
            background: #1c2128;
            padding: 3px;
            border-radius: 6px;
            border: 1px solid #21262d;
        }
        
        .view-btn {
            background: transparent;
            color: #8b949e;
            border: none;
            padding: 6px 14px;
            border-radius: 4px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 500;
            transition: all 0.2s ease;
        }
        
        .view-btn:hover {
            color: #c9d1d9;
        }
        
        .view-btn.active {
            background: #388bfd;
            color: white;
        }
        
        table {
            width: 100%;
            border-collapse: separate;
            border-spacing: 0;
        }
        
        th {
            text-align: left;
            padding: 12px 14px;
            background: #1c2128;
            color: #ffffff;
            font-weight: 600;
            font-size: 13px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            border-bottom: 1px solid #30363d;
        }
        
        td {
            padding: 12px 14px;
            border-bottom: 1px solid #21262d;
            font-size: 14px;
            color: #c9d1d9;
        }
        
        tbody tr {
            transition: background 0.15s ease;
        }
        
        tbody tr:hover { 
            background: rgba(56, 139, 253, 0.05);
        }
        
        .direction-long { 
            color: #3fb950;
            font-weight: 600;
        }
        
        .direction-short { 
            color: #f85149;
            font-weight: 600;
        }
        
        .delta-tp {
            color: #22d3ee;
            font-weight: 500;
        }
        
        .delta-sl {
            color: #e879f9;
            font-weight: 500;
        }
        
        .delta-tp-close {
            color: #10ff10;
            font-weight: 700;
            text-shadow: 0 0 8px rgba(16, 255, 16, 0.6);
        }
        
        .delta-sl-close {
            color: #ff1010;
            font-weight: 700;
            text-shadow: 0 0 8px rgba(255, 16, 16, 0.6);
        }
        
        .badge {
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
            display: inline-block;
            letter-spacing: 0.3px;
        }
        
        .badge-active { 
            background: #10b981;
            color: white;
        }
        
        .badge-deprecating { 
            background: #64748b;
            color: white;
        }
        
        .badge-not-implemented { 
            background: #dc2626;
            color: white;
        }
        
        .badge-tp { 
            background: #10b981;
            color: white;
        }
        
        .badge-sl { 
            background: #ef4444;
            color: white;
        }
        
        .badge-timeout { 
            background: #64748b;
            color: white;
        }
        
        .ws-indicator {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            font-size: 13px;
        }
        
        .ws-dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }
        
        .ws-dot.connected {
            background: #3fb950;
            box-shadow: 0 0 8px rgba(63, 185, 80, 0.5);
        }
        
        .ws-dot.disconnected {
            background: #f85149;
        }
        
        .config-card {
            background: #1c2128;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #21262d;
            margin-bottom: 16px;
        }
        
        .config-card h3 {
            color: #c9d1d9;
            font-size: 18px;
            font-weight: 600;
            margin-bottom: 16px;
            display: flex;
            align-items: center;
            gap: 8px;
        }
        
        .config-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid #21262d;
        }
        
        .config-row:last-child {
            border-bottom: none;
        }
        
        .config-label {
            color: #8b949e;
            font-size: 14px;
        }
        
        .config-value {
            color: #c9d1d9;
            font-size: 14px;
            font-weight: 600;
        }
        
        .timeframe-item {
            background: #1c2128;
            padding: 12px 16px;
            border-radius: 6px;
            margin-bottom: 8px;
            border: 1px solid #21262d;
        }
        
        .timeframe-header {
            color: #58a6ff;
            font-weight: 600;
            margin-bottom: 6px;
            font-size: 14px;
        }
        
        .timeframe-strategies {
            color: #8b949e;
            font-size: 12px;
        }
    </style>
</head>
<body>
    <!-- ═══════════════════════════════════════════════════════════════
         LOADING SPLASH SCREEN
         ═══════════════════════════════════════════════════════════════ -->
    <div id="loading-splash">
        <div class="loading-content">
            <h1 class="loading-title">Trading Bot Dashboard</h1>
            <p class="loading-subtitle">Connecting to backend...</p>
            <div class="loading-spinner"></div>
            <p class="loading-status" id="loading-status">Attempt <span id="attempt-count">1</span> of 30</p>
            <div class="loading-error" id="loading-error">
                <div class="loading-error-title">⚠️ Connection Timeout</div>
                <div class="loading-error-text">
                    Backend is taking longer than expected to respond.<br>
                    The bot might still be starting up. Please wait...
                </div>
            </div>
        </div>
    </div>

    <!-- ═══════════════════════════════════════════════════════════════
         DASHBOARD (ORIGINAL)
         ═══════════════════════════════════════════════════════════════ -->
    <div class="dashboard-container">
        <div class="header">
            <h1>
                BOT Dashboard
                <span class="account-badge">ACC: {{ account }}</span>
            </h1>
            <div class="header-right">
                <div class="status-badge">Running</div>
                <button class="btn-stop" onclick="stopBot()">⛔ Stop Bot</button>
            </div>
        </div>
        
        <div class="logs-panel">
            <div class="logs-header">
                <h2>Execution Logs</h2>
                <div class="logs-controls">
                    <button class="btn-small" onclick="toggleAutoScroll()" id="auto-scroll-btn">Auto</button>
                    <button class="btn-small danger" onclick="clearLogs()">Clear</button>
                </div>
            </div>
            <div class="logs-content" id="logs-content">
                <div class="log-line default">Waiting for bot output...</div>
            </div>
        </div>
        
        <div class="main-panel">
            <div class="stats-grid">
                <div class="stat-card">
                    <div class="stat-label">💵 Closed P/L</div>
                    <div class="stat-value positive" id="total-profit">$-</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">📊 Profit %</div>
                    <div class="stat-value positive" id="profit-pct">-%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">📋 Trades</div>
                    <div class="stat-value neutral" id="trades-num">-</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">🎯 Win Rate</div>
                    <div class="stat-value neutral" id="trades-pct">-%</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">📊 Positions</div>
                    <div class="stat-value neutral" id="total-positions">-</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">📈 Open P/L</div>
                    <div class="stat-value positive" id="open-pnl">$-</div>
                </div>
                <div class="stat-card">
                    <div class="stat-label">₿ BTC Price</div>
                    <div class="stat-value warning" id="btc-price">$-</div>
                </div>
            </div>
            
            <div class="tabs-container">
                <button class="tab-btn active" onclick="switchTab('positions')">Positions</button>
                <button class="tab-btn" onclick="switchTab('analysis')">Strategy Analysis</button>
                <button class="tab-btn" onclick="switchTab('trades')">Recent Trades</button>
                <button class="tab-btn" onclick="switchTab('config')">Config & Connections</button>
            </div>
            
            <div id="tab-positions" class="tab-content active">
                <div class="content-section">
                    <h2>
                        <span>Active Positions</span>
                        <div class="view-selector">
                            <button class="view-btn active" onclick="setPositionsView('compact')">Compact</button>
                            <button class="view-btn" onclick="setPositionsView('detailed')">Detailed</button>
                        </div>
                    </h2>
                    <div id="positions-container"></div>
                </div>
            </div>
            
            <div id="tab-analysis" class="tab-content">
                <div class="content-section">
                    <h2>Strategy Performance Analysis</h2>
                    <div id="analysis-container">Loading...</div>
                </div>
            </div>
            
            <div id="tab-trades" class="tab-content">
                <div class="content-section">
                    <h2>Recent Closed Trades</h2>
                    <table>
                        <thead>
                            <tr>
                                <th>Closed At</th>
                                <th>Strategy</th>
                                <th>Symbol</th>
                                <th>Side</th>
                                <th>Profit</th>
                                <th>Profit %</th>
                                <th>Exit</th>
                            </tr>
                        </thead>
                        <tbody id="trades-body">
                            <tr><td colspan="7" style="text-align: center;">Loading...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
            
            <div id="tab-config" class="tab-content">
                <div class="content-section">
                    <h2>Bot Configuration & Status</h2>
                    
                    <div class="config-card">
                        <h3>🌐 WebSocket Connections</h3>
                        <div class="config-row">
                            <span class="config-label">Public Channel:</span>
                            <span class="ws-indicator" id="ws-public">
                                <span class="ws-dot disconnected"></span>
                                <span>Connecting...</span>
                            </span>
                        </div>
                        <div class="config-row">
                            <span class="config-label">Private Channel:</span>
                            <span class="ws-indicator" id="ws-private">
                                <span class="ws-dot disconnected"></span>
                                <span>Connecting...</span>
                            </span>
                        </div>
                        <div class="config-row">
                            <span class="config-label">Authentication:</span>
                            <span class="ws-indicator" id="ws-auth">
                                <span class="ws-dot disconnected"></span>
                                <span>Pending...</span>
                            </span>
                        </div>
                    </div>
                    
                    <div class="config-card">
                        <h3>⚙️ Configuration</h3>
                        <div class="config-row">
                            <span class="config-label">Account:</span>
                            <span class="config-value" id="config-account">-</span>
                        </div>
                        <div class="config-row">
                            <span class="config-label">Initial Capital:</span>
                            <span class="config-value" id="config-capital">-</span>
                        </div>
                        <div class="config-row">
                            <span class="config-label">Total Strategies:</span>
                            <span class="config-value" id="config-total-strat">-</span>
                        </div>
                        <div class="config-row">
                            <span class="config-label">Active:</span>
                            <span class="config-value" id="config-active-strat">-</span>
                        </div>
                        <div class="config-row">
                            <span class="config-label">Deprecating:</span>
                            <span class="config-value" id="config-deprecating-strat">-</span>
                        </div>
                        <div class="config-row">
                            <span class="config-label">Not Implemented:</span>
                            <span class="config-value" id="config-not-implemented-strat">-</span>
                        </div>
                    </div>
                    
                    <div class="config-card">
                        <h3>📋 Strategies List</h3>
                        <div style="overflow-x: auto;">
                            <table id="strategies-table">
                                <thead>
                                    <tr>
                                        <th>ID</th>
                                        <th>TF</th>
                                        <th>Side</th>
                                        <th>Symbols</th>
                                        <th>TP%</th>
                                        <th>SL%</th>
                                        <th>Amount</th>
                                        <th>Candles</th>
                                        <th>Lookback</th>
                                        <th>Tolerance</th>
                                        <th>MA</th>
                                        <th>Impulse</th>
                                        <th>Trend</th>
                                        <th>Status</th>
                                    </tr>
                                </thead>
                                <tbody id="strategies-body">
                                    <tr><td colspan="14" style="text-align: center;">Loading...</td></tr>
                                </tbody>
                            </table>
                        </div>
                    </div>
                    
                    <div class="config-card">
                        <h3>⏰ Timeframes</h3>
                        <div id="timeframes-container">Loading...</div>
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <div class="stop-overlay" id="stop-overlay">
        <div class="stop-box">
            <h2 id="stop-title">⛔ STOPPING BOT</h2>
            <div class="stop-status" id="stop-status">
                <div class="stop-status-line">Initializing stop sequence...</div>
            </div>
        </div>
    </div>
    
    <script>
// ═══════════════════════════════════════════════════════════════════
// ✅ NEW: Wait for Backend Function
// ═══════════════════════════════════════════════════════════════════

async function waitForBackend() {
    const maxAttempts = 30; // 30 seconds maximum
    let attempts = 0;
    
    const splash = document.getElementById('loading-splash');
    const statusText = document.getElementById('loading-status');
    const attemptCount = document.getElementById('attempt-count');
    const errorBox = document.getElementById('loading-error');
    
    // ✅ FORZAR que el splash esté visible y reseteado
    splash.classList.remove('hidden');
    splash.style.opacity = '1';
    splash.style.display = 'flex';
    errorBox.classList.remove('visible');
    
    console.log('🔍 Waiting for backend to be ready...');
    console.log('🔄 Splash screen reset and shown');
    
    while (attempts < maxAttempts) {
        attempts++;
        attemptCount.textContent = attempts;
        
        console.log(`Attempt ${attempts}/${maxAttempts}: Checking backend...`);
        
        try {
            // Try /api/status endpoint
            const response = await fetch('/api/status', { 
                method: 'GET',
                cache: 'no-cache',
                headers: {
                    'Accept': 'application/json',
                    'Cache-Control': 'no-cache'
                }
            });
            
            if (response.ok) {
                const data = await response.json();
                
                // Verify backend is actually ready
                if (data.status === 'running' || data.account) {
                    console.log('✅ Backend ready!', data);
                    
                    // Update status to show success
                    statusText.innerHTML = '✅ Connected successfully!';
                    statusText.style.color = '#4ade80';
                    
                    // Wait a bit to show success message
                    await new Promise(r => setTimeout(r, 500));
                    
                    // Hide splash screen with fade out
                    splash.style.transition = 'opacity 0.3s ease';
                    splash.style.opacity = '0';
                    
                    await new Promise(r => setTimeout(r, 300));
                    splash.classList.add('hidden');
                    splash.style.display = 'none';
                    
                    console.log('✅ Splash hidden, dashboard ready');
                    return true;
                }
            }
        } catch (error) {
            // Backend not responding yet, continue waiting
            console.log(`Attempt ${attempts}/${maxAttempts}: Backend not ready yet... (${error.message})`);
        }
        
        // Wait 1 second before next attempt
        await new Promise(r => setTimeout(r, 1000));
    }
    
    // Timeout reached
    console.warn('⚠️ Backend connection timeout after 30 seconds');
    errorBox.classList.add('visible');
    
    // Keep trying in background but allow user to see the dashboard
    setTimeout(() => {
        splash.style.transition = 'opacity 0.5s ease';
        splash.style.opacity = '0';
        setTimeout(() => {
            splash.classList.add('hidden');
            splash.style.display = 'none';
        }, 500);
    }, 5000);
    
    return false;
}

// ═══════════════════════════════════════════════════════════════════
// ORIGINAL DASHBOARD CODE
// ═══════════════════════════════════════════════════════════════════

let autoScroll = true;
let isLoadingData = false;
let isLoadingLogs = false;
let currentPositionsView = 'compact';
let cachedPositions = [];
let currentTab = 'positions';

async function stopBot() {
    if (!confirm('WARNING: STOP TRADING BOT?\n\nThis will terminate the bot process.\n\nAre you sure?')) {
        return;
    }
    
    const overlay = document.getElementById('stop-overlay');
    const statusDiv = document.getElementById('stop-status');
    overlay.classList.add('active');
    
    addStopLog('Initializing stop sequence...');
    addStopLog('Stop signal requested');
    
    try {
        const stopResponse = await fetch('/api/bot/stop', { method: 'POST' });
        if (!stopResponse.ok) throw new Error('Failed to send stop signal');
        
        const stopData = await stopResponse.json();
        addStopLog('Stop signal sent (SIGTERM) to PID ' + stopData.pid, 'success');
        
        let attempts = 0;
        const maxAttempts = 30;
        let botStoppedConfirmed = false;
        
        const verifyInterval = setInterval(async () => {
            attempts++;
            addStopLog('Verifying shutdown... (' + attempts + '/' + maxAttempts + ')');
            
            try {
                const verifyResponse = await fetch('/api/bot/verify-stopped', {
                    method: 'GET',
                    cache: 'no-cache'
                });
                
                if (verifyResponse.ok) {
                    const status = await verifyResponse.json();
                    if (!status.running) {
                        // ✅ Bot confirmado detenido
                        clearInterval(verifyInterval);
                        botStoppedConfirmed = true;
                        addStopLog('Process verified as terminated', 'success');
                        addStopLog('✅ BOT STOPPED SUCCESSFULLY', 'success');
                        document.getElementById('stop-title').textContent = '✅ BOT STOPPED';
                        document.getElementById('stop-title').style.color = '#3fb950';
                        return;
                    }
                }
            } catch (error) {
                // ✅ Si Flask no responde = bot se cerró completamente (éxito)
                clearInterval(verifyInterval);
                botStoppedConfirmed = true;
                
                addStopLog('Backend stopped responding', 'success');
                addStopLog('Flask server terminated (expected)', 'success');
                addStopLog('✅ BOT STOPPED SUCCESSFULLY', 'success');
                document.getElementById('stop-title').textContent = '✅ BOT STOPPED';
                document.getElementById('stop-title').style.color = '#3fb950';
                return;
            }
            
            if (attempts >= maxAttempts && !botStoppedConfirmed) {
                clearInterval(verifyInterval);
                addStopLog('⚠️ VERIFICATION TIMEOUT', 'warning');
                addStopLog('Bot may still be running. Check manually.', 'warning');
                document.getElementById('stop-title').textContent = '⚠️ TIMEOUT';
                document.getElementById('stop-title').style.color = '#d29922';
            }
        }, 1000);
        
    } catch (error) {
        // Error enviando señal de stop inicial
        if (error.message.includes('fetch') || error.name === 'TypeError') {
            addStopLog('⚠️ Backend not responding', 'warning');
            addStopLog('Bot may have already stopped', 'warning');
        } else {
            addStopLog('❌ ERROR: ' + error.message, 'error');
        }
    }
}

function addStopLog(message, className = '') {
    const statusDiv = document.getElementById('stop-status');
    const line = document.createElement('div');
    line.className = 'stop-status-line' + (className ? ' ' + className : '');
    line.textContent = message;
    statusDiv.appendChild(line);
    statusDiv.scrollTop = statusDiv.scrollHeight;
}

function switchTab(tabName) {
    currentTab = tabName;
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    document.querySelectorAll('.tab-content').forEach(content => content.classList.remove('active'));
    document.getElementById('tab-' + tabName).classList.add('active');
    
    if (tabName === 'analysis') loadStrategyAnalysis();
    if (tabName === 'config') loadBotConfig();
}

function getLogClass(line) {
    if (line.includes('Position opened') || line.includes('TP REACHED')) return 'success';
    if (line.includes('Error') || line.includes('SL REACHED')) return 'error';
    if (line.includes('Warning')) return 'warning';
    if (line.includes('Checking')) return 'info';
    return 'default';
}

async function loadLogs() {
    if (isLoadingLogs) return;
    isLoadingLogs = true;
    try {
        const res = await fetch('/api/logs/stream');
        if (!res.ok) throw new Error('HTTP ' + res.status);
        const data = await res.json();
        
        if (data.logs && data.logs.length > 0) {
            const logsContent = document.getElementById('logs-content');
            if (logsContent.children.length === 1 && logsContent.children[0].textContent.includes('Waiting')) {
                logsContent.innerHTML = '';
            }
            
            const fragment = document.createDocumentFragment();
            data.logs.forEach(log => {
                const logLine = document.createElement('div');
                logLine.className = 'log-line ' + getLogClass(log);
                logLine.textContent = log;
                fragment.appendChild(logLine);
            });
            logsContent.appendChild(fragment);
            
            while (logsContent.children.length > 500) {
                logsContent.removeChild(logsContent.firstChild);
            }
            
            if (autoScroll) {
                requestAnimationFrame(() => {
                    logsContent.scrollTop = logsContent.scrollHeight;
                });
            }
        }
    } catch (error) {
        console.error('Error loading logs:', error);
    } finally {
        isLoadingLogs = false;
    }
}

function setPositionsView(view) {
    currentPositionsView = view;
    document.querySelectorAll('.view-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');
    renderPositions(cachedPositions);
    localStorage.setItem('positionsView', view);
}

function renderPositions(positions) {
    cachedPositions = positions;
    const container = document.getElementById('positions-container');
    
    if (!positions || positions.length === 0) {
        container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">No active positions</div>';
        return;
    }
    
    if (currentPositionsView === 'compact') {
        renderCompactView(container, positions);
    } else {
        renderDetailedView(container, positions);
    }
}

function renderCompactView(container, positions) {
    const groupedByStrategy = {};
    positions.forEach(pos => {
        if (!groupedByStrategy[pos.strategy]) {
            groupedByStrategy[pos.strategy] = {
                positions: [],
                totalPnl: 0,
                direction: pos.direction,
                opened_at: pos.opened_at,
                candles: pos.candles,
                max_candles: pos.max_candles
            };
        }
        groupedByStrategy[pos.strategy].positions.push(pos);
        groupedByStrategy[pos.strategy].totalPnl += (pos.current_pnl || 0);
    });
    
    const html = '<table><thead><tr><th>Strategy</th><th>Side</th><th>Opened</th><th style="text-align: right;">Candles</th><th style="text-align: center;">#pos</th><th>PnL</th></tr></thead><tbody>' +
        Object.entries(groupedByStrategy).map(([strategyId, data]) => {
            const pnl = data.totalPnl;
            const pnlClass = pnl >= 0 ? 'direction-long' : 'direction-short';
            let openedDateStr = '';
            if (data.opened_at) {
                try {
                    const date = new Date(data.opened_at);
                    openedDateStr = date.toISOString().split('T')[0];
                } catch {
                    openedDateStr = String(data.opened_at).substring(0, 10);
                }
            }
            return '<tr><td>' + strategyId + '</td><td class="direction-' + data.direction.toLowerCase() + '">' + data.direction.toUpperCase() + '</td><td>' + openedDateStr + '</td><td style="text-align: right;">' + (data.candles || 0) + '/' + (data.max_candles || 50) + '</td><td style="text-align: center; color: #58a6ff; font-weight: 600;">' + data.positions.length + '</td><td class="' + pnlClass + '">' + (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2) + '</td></tr>';
        }).join('') +
        '</tbody></table>';
    container.innerHTML = html;
}

function renderDetailedView(container, positions) {
    const html = '<table><thead><tr><th>Strategy</th><th>Symbol</th><th>Side</th><th>Entry</th><th>Current</th><th>Size</th><th>TP</th><th>SL</th><th>P/L</th><th>Delta TP</th><th>Delta SL</th><th style="text-align: right;">Candles</th></tr></thead><tbody>' +
        positions.map(pos => {
            const pnl = pos.current_pnl || 0;
            const pnlClass = pnl >= 0 ? 'direction-long' : 'direction-short';
            const currentPrice = parseFloat(pos.current_price || pos.entry_price);
            const tp = parseFloat(pos.tp);
            const sl = parseFloat(pos.sl);
            
            let deltaTp, deltaSl;
            if (pos.direction.toLowerCase() === 'long') {
                deltaTp = ((tp - currentPrice) / currentPrice * 100);
                deltaSl = ((currentPrice - sl) / currentPrice * 100);
            } else {
                deltaTp = ((currentPrice - tp) / currentPrice * 100);
                deltaSl = ((sl - currentPrice) / currentPrice * 100);
            }
            
            const deltaTpClass = Math.abs(deltaTp) < 1 ? 'delta-tp-close' : 'delta-tp';
            const deltaSlClass = Math.abs(deltaSl) < 1 ? 'delta-sl-close' : 'delta-sl';
            
            return '<tr><td>' + pos.strategy + '</td><td>' + pos.symbol + '</td><td class="direction-' + pos.direction.toLowerCase() + '">' + pos.direction.toUpperCase() + '</td><td>$' + parseFloat(pos.entry_price).toFixed(2) + '</td><td>$' + currentPrice.toFixed(2) + '</td><td>' + parseFloat(pos.size).toFixed(4) + '</td><td>$' + tp.toFixed(2) + '</td><td>$' + sl.toFixed(2) + '</td><td class="' + pnlClass + '">' + (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2) + '</td><td class="' + deltaTpClass + '">' + (deltaTp >= 0 ? '+' : '') + deltaTp.toFixed(2) + '%</td><td class="' + deltaSlClass + '">' + (deltaSl >= 0 ? '+' : '') + deltaSl.toFixed(2) + '%</td><td style="text-align: right;">' + (pos.candles || 0) + '/' + (pos.max_candles || 50) + '</td></tr>';
        }).join('') +
        '</tbody></table>';
    container.innerHTML = html;
}

async function loadStrategyAnalysis() {
    try {
        const res = await fetch('/api/strategy-analysis');
        const data = await res.json();
        const container = document.getElementById('analysis-container');
        
        if (!data || data.length === 0) {
            container.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 40px;">No data</div>';
            return;
        }
        
        const html = '<table><thead><tr><th>Strategy</th><th>First</th><th>Trades</th><th>Win %</th><th>Profit</th><th>Profit %</th><th>TP %</th><th>SL %</th><th>OOM %</th><th>Avg Days</th></tr></thead><tbody>' +
            data.map(s => {
                const profitClass = s.Total_profit >= 0 ? 'direction-long' : 'direction-short';
                return '<tr><td>' + s.Strategy + '</td><td>' + s.date_fo + '</td><td>' + s.Trades_num + '</td><td>' + s.Trades_pct.toFixed(1) + '%</td><td class="' + profitClass + '">' + (s.Total_profit >= 0 ? '+' : '') + '$' + s.Total_profit.toFixed(2) + '</td><td class="' + profitClass + '">' + (s.Profit_pct >= 0 ? '+' : '') + s.Profit_pct.toFixed(1) + '%</td><td>' + s.TP_pct.toFixed(1) + '%</td><td>' + s.SL_pct.toFixed(1) + '%</td><td>' + s.OOM_pct.toFixed(1) + '%</td><td>' + s.Avg_days.toFixed(2) + '</td></tr>';
            }).join('') +
            '</tbody></table>';
        container.innerHTML = html;
    } catch (error) {
        console.error('Error:', error);
    }
}

async function loadBotConfig() {
    try {
        const res = await fetch('/api/bot-config');
        const data = await res.json();
        
        document.getElementById('config-account').textContent = data.account || '-';
        document.getElementById('config-capital').textContent = '$' + (data.initial_capital || 0).toLocaleString();
        document.getElementById('config-total-strat').textContent = data.stats.total || 0;
        document.getElementById('config-active-strat').textContent = data.stats.active || 0;
        document.getElementById('config-deprecating-strat').textContent = data.stats.deprecating || 0;
        document.getElementById('config-not-implemented-strat').textContent = data.stats.not_implemented || 0;
        
        const wsPublic = document.getElementById('ws-public');
        const wsPrivate = document.getElementById('ws-private');
        const wsAuth = document.getElementById('ws-auth');
        
        if (data.websocket_status.public_connected) {
            wsPublic.innerHTML = '<span class="ws-dot connected"></span><span>Connected</span>';
        } else {
            wsPublic.innerHTML = '<span class="ws-dot disconnected"></span><span>Disconnected</span>';
        }
        
        if (data.websocket_status.private_connected) {
            wsPrivate.innerHTML = '<span class="ws-dot connected"></span><span>Connected</span>';
        } else {
            wsPrivate.innerHTML = '<span class="ws-dot disconnected"></span><span>Disconnected</span>';
        }
        
        if (data.websocket_status.authenticated) {
            wsAuth.innerHTML = '<span class="ws-dot connected"></span><span>Authenticated</span>';
        } else {
            wsAuth.innerHTML = '<span class="ws-dot disconnected"></span><span>Not Authenticated</span>';
        }
        
        const strategiesBody = document.getElementById('strategies-body');
        if (!data.strategies || data.strategies.length === 0) {
            strategiesBody.innerHTML = '<tr><td colspan="14" style="text-align: center;">No strategies</td></tr>';
        } else {
            strategiesBody.innerHTML = data.strategies.map(strat => {
                let statusBadge = '';
                if (strat.status === 'ACTIVE') {
                    statusBadge = '<span class="badge badge-active">Active</span>';
                } else if (strat.status === 'DEPRECATING') {
                    statusBadge = '<span class="badge badge-deprecating">Deprecating</span>';
                } else {
                    statusBadge = '<span class="badge badge-not-implemented">Not Impl.</span>';
                }
                
                return '<tr><td>' + strat.id + '</td><td>' + strat.timeframe + '</td><td class="direction-' + strat.direction.toLowerCase() + '">' + strat.direction.toUpperCase() + '</td><td style="text-align: center; color: #58a6ff;">' + strat.symbols_count + '</td><td>' + strat.tp_pct + '</td><td>' + strat.sl_pct + '</td><td>$' + strat.order_amount + '</td><td>' + strat.sell_after_ncandles + '</td><td>' + strat.lookback + '</td><td>' + strat.tolerance + '</td><td>' + strat.ma_period + '</td><td>' + strat.impulse + '</td><td>' + strat.trend_th + '</td><td>' + statusBadge + '</td></tr>';
            }).join('');
        }
        
        const timeframesContainer = document.getElementById('timeframes-container');
        if (!data.timeframes || Object.keys(data.timeframes).length === 0) {
            timeframesContainer.innerHTML = '<div style="text-align: center; color: #8b949e; padding: 20px;">No timeframes</div>';
        } else {
            timeframesContainer.innerHTML = Object.entries(data.timeframes)
                .sort(([a], [b]) => a.localeCompare(b))
                .map(([tf, strategies]) => '<div class="timeframe-item"><div class="timeframe-header">' + tf + ' (' + strategies.length + ' strategies)</div><div class="timeframe-strategies">' + strategies.join(', ') + '</div></div>')
                .join('');
        }
    } catch (error) {
        console.error('Error:', error);
    }
}

async function loadData() {
    if (isLoadingData) return;
    isLoadingData = true;
    try {
        const statusRes = await fetch('/api/status');
        if (!statusRes.ok) throw new Error('HTTP ' + statusRes.status);
        const status = await statusRes.json();
        
        requestAnimationFrame(() => {
            document.getElementById('total-positions').textContent = status.total_positions || 0;
            
            const totalProfit = status.total_profit || 0;
            const profitEl = document.getElementById('total-profit');
            profitEl.textContent = '$' + totalProfit.toFixed(2);
            profitEl.className = 'stat-value ' + (totalProfit >= 0 ? 'positive' : 'negative');
            
            const openPnl = status.open_pnl || 0;
            const openPnlEl = document.getElementById('open-pnl');
            openPnlEl.textContent = '$' + openPnl.toFixed(2);
            openPnlEl.className = 'stat-value ' + (openPnl >= 0 ? 'positive' : 'negative');
            
            const profitPct = status.profit_pct || 0;
            const profitPctEl = document.getElementById('profit-pct');
            profitPctEl.textContent = (profitPct >= 0 ? '+' : '') + profitPct.toFixed(2) + '%';
            profitPctEl.className = 'stat-value ' + (profitPct >= 0 ? 'positive' : 'negative');
            
            document.getElementById('trades-num').textContent = status.num_trades || 0;
            const tradesPct = status.trades_pct || 0;
            document.getElementById('trades-pct').textContent = tradesPct.toFixed(1) + '%';
            const btcPrice = status.btc_price || 0;
            document.getElementById('btc-price').textContent = '$' + btcPrice.toLocaleString();
        });
        
        const posRes = await fetch('/api/positions');
        if (!posRes.ok) throw new Error('HTTP ' + posRes.status);
        const positions = await posRes.json();
        
        requestAnimationFrame(() => {
            if (currentTab === 'positions') renderPositions(positions);
        });
        
        const tradesRes = await fetch('/api/trades/recent');
        if (!tradesRes.ok) throw new Error('HTTP ' + tradesRes.status);
        const trades = await tradesRes.json();
        
        requestAnimationFrame(() => {
            const tradesBody = document.getElementById('trades-body');
            if (!trades || trades.length === 0) {
                tradesBody.innerHTML = '<tr><td colspan="7" style="text-align: center;">No trades</td></tr>';
            } else {
                tradesBody.innerHTML = trades.reverse().map(trade => {
                    const profitClass = trade.PROFIT >= 0 ? 'direction-long' : 'direction-short';
                    let reasonBadge = '';
                    if (trade.REASON_OUT === 'TP') {
                        reasonBadge = '<span class="badge badge-tp">TP</span>';
                    } else if (trade.REASON_OUT === 'SL') {
                        reasonBadge = '<span class="badge badge-sl">SL</span>';
                    } else {
                        reasonBadge = '<span class="badge badge-timeout">TIMEOUT</span>';
                    }
                    return '<tr><td>' + new Date(trade.CLOSE_AT).toLocaleString() + '</td><td>' + trade.STRATEGY + '</td><td>' + trade.SYMBOL + '</td><td class="direction-' + trade.DIRECTION.toLowerCase() + '">' + trade.DIRECTION + '</td><td class="' + profitClass + '">' + (trade.PROFIT >= 0 ? '+' : '') + '$' + trade.PROFIT.toFixed(2) + '</td><td class="' + profitClass + '">' + (trade.PROFIT_PCT >= 0 ? '+' : '') + trade.PROFIT_PCT.toFixed(2) + '%</td><td>' + reasonBadge + '</td></tr>';
                }).join('');
            }
        });
    } catch (error) {
        console.error('Error:', error);
    } finally {
        isLoadingData = false;
    }
}

function toggleAutoScroll() {
    autoScroll = !autoScroll;
    document.getElementById('auto-scroll-btn').textContent = autoScroll ? 'Auto' : 'Manual';
}

function clearLogs() {
    if (confirm('Clear all logs?')) {
        document.getElementById('logs-content').innerHTML = '<div class="log-line default">Logs cleared</div>';
    }
}

let dataInterval, logsInterval;

function startPolling() {
    loadData();
    loadLogs();
    dataInterval = setInterval(() => loadData().catch(console.error), 3000);
    logsInterval = setInterval(() => loadLogs().catch(console.error), 1500);
}

function stopPolling() {
    if (dataInterval) clearInterval(dataInterval);
    if (logsInterval) clearInterval(logsInterval);
}

document.addEventListener('visibilitychange', () => {
    if (document.hidden) stopPolling();
    else startPolling();
});

// ═══════════════════════════════════════════════════════════════════
// ✅ MODIFIED: Initialize with waitForBackend - ALWAYS RUN
// ═══════════════════════════════════════════════════════════════════

const savedView = localStorage.getItem('positionsView') || 'compact';
currentPositionsView = savedView;

// ✅ FUNCIÓN DE INICIALIZACIÓN - se ejecuta siempre
async function initializeDashboard() {
    console.log('🚀 Dashboard initialization started');
    console.log('📍 Document ready state:', document.readyState);
    
    // Setup view buttons
    document.querySelectorAll('.view-btn').forEach(btn => {
        if (btn.textContent.toLowerCase().includes(savedView)) btn.classList.add('active');
        else btn.classList.remove('active');
    });
    
    // ✅ ALWAYS wait for backend before starting
    console.log('⏳ Calling waitForBackend()...');
    const backendReady = await waitForBackend();
    
    if (backendReady) {
        console.log('✅ Backend confirmed ready, starting polling');
    } else {
        console.warn('⚠️ Backend timeout, starting polling anyway');
    }
    
    startPolling();
    console.log('✅ Dashboard fully initialized');
}

// ✅ EJECUTAR SIEMPRE, sin importar el estado del documento
console.log('📄 Script loaded, scheduling initialization...');

if (document.readyState === 'loading') {
    console.log('📌 Document still loading, waiting for DOMContentLoaded');
    document.addEventListener('DOMContentLoaded', initializeDashboard);
} else {
    console.log('📌 Document already loaded, initializing immediately');
    initializeDashboard();
}

// ✅ TAMBIÉN ejecutar en pageshow (por si el navegador usa caché bfcache)
window.addEventListener('pageshow', function(event) {
    if (event.persisted) {
        console.log('🔄 Page restored from bfcache, re-initializing...');
        initializeDashboard();
    }
});
    </script>
</body>
</html>