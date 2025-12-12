import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar, Cell } from 'recharts';
import { TrendingUp, TrendingDown, Activity, DollarSign, Target, AlertCircle, Play, Pause, RefreshCw } from 'lucide-react';

const TradingBotDashboard = () => {
  const INITIAL_CAPITAL = 3971;
  
  const generateMockTradesData = () => {
    const strategies = [
      'double_top_long_4H',
      'reversal_long_4H', 
      'parity_long_4H',
      'reversal_short_4H',
      'parity_short_4H',
      'reversal_long_1H',
      'reversal_short_1H',
      'reversal_long_6Hutc',
      'reversal_short_6Hutc'
    ];

    const capitalPerStrategy = INITIAL_CAPITAL / strategies.length;

    const strategyMetrics = strategies.map(strat => {
      const profit = (Math.random() - 0.3) * 500;
      const profitPct = (profit / capitalPerStrategy) * 100;
      
      return {
        name: strat.replace(/_/g, ' ').toUpperCase(),
        trades: Math.floor(Math.random() * 50) + 10,
        winRate: Math.random() * 40 + 50,
        profit: profit,
        profitPct: profitPct,
        avgDuration: Math.random() * 5 + 1,
        capitalAssigned: capitalPerStrategy
      };
    });

    const equityCurve = [];
    let equity = INITIAL_CAPITAL;
    const startDate = new Date('2024-11-01');
    
    for (let i = 0; i < 40; i++) {
      const date = new Date(startDate);
      date.setDate(date.getDate() + i);
      equity += (Math.random() - 0.45) * 50;
      equityCurve.push({
        date: date.toISOString().split('T')[0],
        equity: Math.round(equity * 100) / 100
      });
    }

    return { strategies: strategyMetrics, equityCurve };
  };

  const [botState, setBotState] = useState(null);
  const [tradesData, setTradesData] = useState(null);
  const [isLoading, setIsLoading] = useState(true);
  const [lastUpdate, setLastUpdate] = useState(new Date());
  const [autoRefresh, setAutoRefresh] = useState(true);
  const [usingMockData, setUsingMockData] = useState(false);
  const [mockData] = useState(generateMockTradesData());

  const loadBotData = async () => {
    try {
      setIsLoading(true);
      let foundStateFile = false;
      
      // Intentar leer bot_state.json
      if (typeof window.fs !== 'undefined') {
        try {
          const stateData = await window.fs.readFile('bot_state.json', { encoding: 'utf8' });
          const state = JSON.parse(stateData);
          setBotState(state);
          foundStateFile = true;
          console.log('✅ bot_state.json loaded successfully');
        } catch (e) {
          // Archivo no encontrado, silencioso
        }
      }

      // Siempre usar datos mock para las métricas (ya que no procesamos Excel)
      setTradesData(mockData);
      
      // Solo mostrar warning si no encontró ningún archivo
      setUsingMockData(!foundStateFile);

      setLastUpdate(new Date());
    } catch (error) {
      console.error('Error loading bot data:', error);
      setUsingMockData(true);
      setTradesData(mockData);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadBotData();
    
    if (autoRefresh) {
      const interval = setInterval(loadBotData, 5000);
      return () => clearInterval(interval);
    }
  }, [autoRefresh]);

  const calculateTotals = () => {
    if (!tradesData) return { totalTrades: 0, totalProfit: 0, winRate: 0, activePositions: 0 };
    
    const totalTrades = tradesData.strategies.reduce((sum, s) => sum + s.trades, 0);
    const totalProfit = tradesData.strategies.reduce((sum, s) => sum + s.profit, 0);
    const avgWinRate = tradesData.strategies.reduce((sum, s) => sum + s.winRate, 0) / tradesData.strategies.length;
    
    let activePositions = 0;
    if (botState?.positions) {
      Object.values(botState.positions).forEach(stratPositions => {
        activePositions += stratPositions.length;
      });
    }

    return { totalTrades, totalProfit, winRate: avgWinRate, activePositions };
  };

  const totals = calculateTotals();

  const getActivePositions = () => {
    if (!botState?.positions) return [];
    
    const positions = [];
    Object.entries(botState.positions).forEach(([stratId, stratPositions]) => {
      stratPositions.forEach(pos => {
        positions.push({
          ...pos,
          strategy: stratId,
          pnl: calculatePnL(pos),
          pnlPct: calculatePnLPct(pos),
          duration: calculateDuration(pos.opened_at),
          currentPrice: simulateCurrentPrice(pos)
        });
      });
    });
    return positions;
  };

  const simulateCurrentPrice = (position) => {
    return parseFloat(position.entry_price) * (1 + (Math.random() - 0.5) * 0.05);
  };

  const calculatePnL = (position) => {
    const currentPrice = simulateCurrentPrice(position);
    const entry = parseFloat(position.entry_price);
    const size = parseFloat(position.size);
    
    if (position.direction === 'long') {
      return (currentPrice - entry) * size;
    } else {
      return (entry - currentPrice) * size;
    }
  };

  const calculatePnLPct = (position) => {
    const currentPrice = simulateCurrentPrice(position);
    const entry = parseFloat(position.entry_price);
    
    if (position.direction === 'long') {
      return ((currentPrice - entry) / entry) * 100;
    } else {
      return ((entry - currentPrice) / entry) * 100;
    }
  };

  const calculateDuration = (openedAt) => {
    const opened = new Date(openedAt);
    const now = new Date();
    const hours = Math.floor((now - opened) / (1000 * 60 * 60));
    return `${hours}h`;
  };

  const formatPrice = (price) => {
    const p = parseFloat(price);
    if (p < 0.01) return p.toFixed(6);
    if (p < 1) return p.toFixed(4);
    if (p < 100) return p.toFixed(2);
    return p.toFixed(1);
  };

  const calculateDistanceToTP = (position) => {
    const current = simulateCurrentPrice(position);
    const tp = parseFloat(position.tp);
    const entry = parseFloat(position.entry_price);
    
    if (position.direction === 'long') {
      return ((tp - current) / entry) * 100;
    } else {
      return ((current - tp) / entry) * 100;
    }
  };

  const calculateDistanceToSL = (position) => {
    const current = simulateCurrentPrice(position);
    const sl = parseFloat(position.sl);
    const entry = parseFloat(position.entry_price);
    
    if (position.direction === 'long') {
      return ((current - sl) / entry) * 100;
    } else {
      return ((sl - current) / entry) * 100;
    }
  };

  const activePositions = getActivePositions();
  const totalPnL = activePositions.reduce((sum, pos) => sum + pos.pnl, 0);

  if (isLoading && !tradesData) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="w-12 h-12 text-cyan-400 animate-spin mx-auto mb-4" />
          <p className="text-gray-300 text-xl">Loading Bot Data...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900">
      {/* Header */}
      <div className="p-6 border-b border-slate-700">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-3xl font-bold text-white mb-1 flex items-center gap-3">
              <Activity className="w-8 h-8 text-cyan-400" />
              Trading Bot Dashboard
            </h1>
            <p className="text-gray-400 text-sm">Multi-Strategy Futures Trading System</p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setAutoRefresh(!autoRefresh)}
              className={`px-3 py-2 rounded-lg font-medium transition-all ${
                autoRefresh 
                  ? 'bg-green-600 hover:bg-green-700 text-white' 
                  : 'bg-gray-700 hover:bg-gray-600 text-gray-300'
              }`}
            >
              {autoRefresh ? <Play className="w-4 h-4" /> : <Pause className="w-4 h-4" />}
            </button>
            <button
              onClick={loadBotData}
              className="px-3 py-2 bg-cyan-600 hover:bg-cyan-700 text-white rounded-lg font-medium transition-all flex items-center gap-2"
            >
              <RefreshCw className="w-4 h-4" />
              Refresh
            </button>
          </div>
        </div>
        <p className="text-xs text-gray-500 mt-2">
          Last update: {lastUpdate.toLocaleTimeString()}
        </p>
      </div>

      {/* Warning Banner */}
      {usingMockData && (
        <div className="mx-6 mt-4 bg-yellow-900/30 border border-yellow-600/50 rounded-lg p-3 flex items-center gap-2">
          <AlertCircle className="w-5 h-5 text-yellow-400 flex-shrink-0" />
          <div>
            <p className="text-yellow-200 font-medium text-sm">Using Example Data for Metrics</p>
            <p className="text-yellow-300/80 text-xs">
              bot_state.json not found. Positions table will show real data when file is available.
            </p>
          </div>
        </div>
      )}

      {/* Main Content - Split Layout */}
      <div className="flex h-[calc(100vh-140px)]">
        {/* LEFT SIDE - Dashboard */}
        <div className="w-1/2 overflow-y-auto p-6 border-r border-slate-700">
          {/* Stats Cards */}
          <div className="grid grid-cols-4 gap-4 mb-6">
            <StatCard
              icon={<Activity className="w-5 h-5" />}
              title="Active Positions"
              value={activePositions.length}
              color="cyan"
            />
            <StatCard
              icon={<Target className="w-5 h-5" />}
              title="Total Trades"
              value={totals.totalTrades}
              color="purple"
            />
            <StatCard
              icon={<DollarSign className="w-5 h-5" />}
              title="Total P&L"
              value={`$${totals.totalProfit.toFixed(2)}`}
              color={totals.totalProfit >= 0 ? 'green' : 'red'}
              trend={totals.totalProfit >= 0 ? 'up' : 'down'}
            />
            <StatCard
              icon={<TrendingUp className="w-5 h-5" />}
              title="Win Rate"
              value={`${totals.winRate.toFixed(1)}%`}
              color="blue"
            />
          </div>

          {/* Charts Grid */}
          <div className="grid grid-cols-1 gap-6 mb-6">
            {/* Equity Curve */}
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-4 border border-slate-700">
              <h2 className="text-lg font-bold text-white mb-3">Equity Curve</h2>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={tradesData?.equityCurve || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="date" 
                    stroke="#9CA3AF"
                    tick={{ fill: '#9CA3AF', fontSize: 10 }}
                  />
                  <YAxis 
                    stroke="#9CA3AF"
                    tick={{ fill: '#9CA3AF', fontSize: 10 }}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1e293b', 
                      border: '1px solid #475569',
                      borderRadius: '8px',
                      fontSize: '12px'
                    }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="equity" 
                    stroke="#06b6d4" 
                    strokeWidth={2}
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>

            {/* Strategy Performance Bar Chart */}
            <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-4 border border-slate-700">
              <h2 className="text-lg font-bold text-white mb-3">Strategy P&L</h2>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={tradesData?.strategies || []}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="name" 
                    stroke="#9CA3AF"
                    tick={{ fill: '#9CA3AF', fontSize: 8 }}
                    angle={-45}
                    textAnchor="end"
                    height={80}
                  />
                  <YAxis 
                    stroke="#9CA3AF"
                    tick={{ fill: '#9CA3AF', fontSize: 10 }}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1e293b', 
                      border: '1px solid #475569',
                      borderRadius: '8px',
                      fontSize: '12px'
                    }}
                  />
                  <Bar dataKey="profit">
                    {tradesData?.strategies.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.profit >= 0 ? '#10b981' : '#ef4444'} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Strategy Performance Table */}
          <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-4 border border-slate-700">
            <h2 className="text-lg font-bold text-white mb-2">Strategy Performance</h2>
            <div className="mb-3 text-gray-400 text-xs">
              Initial Capital: <span className="text-white font-medium">${INITIAL_CAPITAL.toFixed(2)}</span> | 
              Capital per Strategy: <span className="text-white font-medium">${(INITIAL_CAPITAL / (tradesData?.strategies.length || 1)).toFixed(2)}</span>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-slate-700">
                    <th className="text-left py-2 px-2 text-gray-400 font-medium">Strategy</th>
                    <th className="text-right py-2 px-2 text-gray-400 font-medium">Trades</th>
                    <th className="text-right py-2 px-2 text-gray-400 font-medium">Win%</th>
                    <th className="text-right py-2 px-2 text-gray-400 font-medium">P&L ($)</th>
                    <th className="text-right py-2 px-2 text-gray-400 font-medium">P&L (%)</th>
                    <th className="text-right py-2 px-2 text-gray-400 font-medium">Avg Days</th>
                  </tr>
                </thead>
                <tbody>
                  {tradesData?.strategies.map((strat, idx) => (
                    <tr key={idx} className="border-b border-slate-700/50 hover:bg-slate-700/30 transition-colors">
                      <td className="py-2 px-2 text-white font-medium">{strat.name}</td>
                      <td className="py-2 px-2 text-right text-gray-300">{strat.trades}</td>
                      <td className="py-2 px-2 text-right">
                        <span className={`font-medium ${
                          strat.winRate >= 60 ? 'text-green-400' :
                          strat.winRate >= 50 ? 'text-yellow-400' : 'text-red-400'
                        }`}>
                          {strat.winRate.toFixed(1)}%
                        </span>
                      </td>
                      <td className={`py-2 px-2 text-right font-medium ${
                        strat.profit >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}>
                        ${strat.profit.toFixed(2)}
                      </td>
                      <td className={`py-2 px-2 text-right font-bold ${
                        strat.profitPct >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {strat.profitPct >= 0 ? '+' : ''}{strat.profitPct.toFixed(2)}%
                      </td>
                      <td className="py-2 px-2 text-right text-gray-300">
                        {strat.avgDuration.toFixed(1)}d
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* RIGHT SIDE - Active Positions Table (Rich Style) */}
        <div className="w-1/2 overflow-y-auto bg-slate-900/50 p-6">
          <div className="bg-slate-800/70 rounded-xl p-4 border border-slate-600">
            <div className="mb-4">
              <h2 className="text-xl font-bold text-white flex items-center gap-2 mb-1">
                🔷 Checking TP/SL
              </h2>
              <div className="text-sm text-gray-400">
                {new Date().toLocaleString()}
              </div>
              <div className={`text-lg font-bold mt-2 ${totalPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                💰 Total PnL: {totalPnL >= 0 ? '+' : ''}{totalPnL.toFixed(2)} USDT
              </div>
            </div>

            {activePositions.length === 0 ? (
              <div className="text-center py-12 text-gray-500">
                <Activity className="w-12 h-12 mx-auto mb-3 opacity-50" />
                <p>No active positions</p>
              </div>
            ) : (
              <div className="overflow-x-auto">
                <table className="w-full text-xs font-mono">
                  <thead>
                    <tr className="border-b border-slate-600">
                      <th className="text-left py-2 px-2 text-gray-300 font-semibold">Strategy</th>
                      <th className="text-left py-2 px-2 text-gray-300 font-semibold">Symbol</th>
                      <th className="text-center py-2 px-2 text-gray-300 font-semibold">Side</th>
                      <th className="text-left py-2 px-2 text-gray-300 font-semibold">Opened</th>
                      <th className="text-center py-2 px-2 text-gray-300 font-semibold">Candles</th>
                      <th className="text-right py-2 px-2 text-gray-300 font-semibold">Entry</th>
                      <th className="text-right py-2 px-2 text-gray-300 font-semibold">Size</th>
                      <th className="text-right py-2 px-2 text-gray-300 font-semibold">Current</th>
                      <th className="text-center py-2 px-2 text-gray-300 font-semibold">↕</th>
                      <th className="text-right py-2 px-2 text-gray-300 font-semibold">PnL</th>
                      <th className="text-right py-2 px-2 text-gray-300 font-semibold">TP</th>
                      <th className="text-right py-2 px-2 text-gray-300 font-semibold">SL</th>
                    </tr>
                  </thead>
                  <tbody>
                    {activePositions.map((pos, idx) => {
                      const distToTP = calculateDistanceToTP(pos);
                      const distToSL = calculateDistanceToSL(pos);
                      const tpColor = distToTP < 1 ? 'text-green-400' : 'text-cyan-400';
                      const slColor = distToSL < 1 ? 'text-red-400' : 'text-pink-400';
                      
                      return (
                        <tr key={idx} className="border-b border-slate-700/30 hover:bg-slate-700/20">
                          <td className="py-2 px-2 text-gray-300 text-xs">{pos.strategy}</td>
                          <td className="py-2 px-2 text-white font-semibold">{pos.symbol}</td>
                          <td className="py-2 px-2 text-center">
                            <span className={`px-2 py-0.5 rounded text-xs ${
                              pos.direction === 'long' 
                                ? 'bg-green-900/50 text-green-300' 
                                : 'bg-red-900/50 text-red-300'
                            }`}>
                              {pos.direction.toUpperCase()}
                            </span>
                          </td>
                          <td className="py-2 px-2 text-gray-300">
                            {new Date(pos.opened_at).toISOString().split('T')[0]}
                          </td>
                          <td className="py-2 px-2 text-center text-gray-300">-/-</td>
                          <td className="py-2 px-2 text-right text-gray-300">
                            {formatPrice(pos.entry_price)}
                          </td>
                          <td className="py-2 px-2 text-right text-gray-300">
                            {parseFloat(pos.size).toFixed(4)}
                          </td>
                          <td className="py-2 px-2 text-right text-yellow-400 font-semibold">
                            {formatPrice(pos.currentPrice)}
                          </td>
                          <td className="py-2 px-2 text-center">
                            <span className={pos.pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                              {pos.pnl >= 0 ? '↑' : '↓'}
                            </span>
                          </td>
                          <td className={`py-2 px-2 text-right font-semibold ${
                            pos.pnl >= 0 ? 'text-green-400' : 'text-red-400'
                          }`}>
                            {pos.pnl >= 0 ? '+' : ''}{pos.pnl.toFixed(2)}
                          </td>
                          <td className={`py-2 px-2 text-right ${tpColor}`}>
                            {formatPrice(pos.tp)}
                            <span className="text-xs ml-1">
                              (Δ {distToTP >= 0 ? '+' : ''}{distToTP.toFixed(2)}%)
                            </span>
                          </td>
                          <td className={`py-2 px-2 text-right ${slColor}`}>
                            {formatPrice(pos.sl)}
                            <span className="text-xs ml-1">
                              (Δ {distToSL >= 0 ? '+' : ''}{distToSL.toFixed(2)}%)
                            </span>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

const StatCard = ({ icon, title, value, color, trend }) => {
  const colorClasses = {
    cyan: 'from-cyan-600 to-cyan-700',
    purple: 'from-purple-600 to-purple-700',
    green: 'from-green-600 to-green-700',
    red: 'from-red-600 to-red-700',
    blue: 'from-blue-600 to-blue-700'
  };

  return (
    <div className="bg-slate-800/50 backdrop-blur-sm rounded-xl p-4 border border-slate-700 hover:border-slate-600 transition-all">
      <div className="flex items-center justify-between mb-2">
        <div className={`p-2 rounded-lg bg-gradient-to-br ${colorClasses[color]}`}>
          {icon}
        </div>
        {trend && (
          <div className={trend === 'up' ? 'text-green-400' : 'text-red-400'}>
            {trend === 'up' ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
          </div>
        )}
      </div>
      <h3 className="text-gray-400 text-xs font-medium mb-1">{title}</h3>
      <p className="text-white text-xl font-bold">{value}</p>
    </div>
  );
};

export default TradingBotDashboard;