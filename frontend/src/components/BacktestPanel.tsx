import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Play, Download, Settings } from 'lucide-react';

interface BacktestPanelProps {
  service: any;
}

export const BacktestPanel: React.FC<BacktestPanelProps> = ({ service }) => {
  const [backtestConfig, setBacktestConfig] = useState({
    symbol: 'BTCUSDT',
    startDate: '2024-01-01',
    endDate: '2024-12-31',
    initialBalance: 10000,
    leverage: 3,
    enableML: true,
    enableTechnical: true,
    enableSentiment: true,
    enableFearGreed: true
  });

  const [backtestResults, setBacktestResults] = useState(null);
  const [isRunning, setIsRunning] = useState(false);

  const handleRunBacktest = async () => {
    setIsRunning(true);
    try {
      const results = await service.runBacktest(backtestConfig);
      setBacktestResults(results);
    } catch (error) {
      console.error('Backtest failed:', error);
    } finally {
      setIsRunning(false);
    }
  };

  const formatCurrency = (value: number) => `$${value.toFixed(2)}`;
  const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">Backtesting Engine</h2>
        <button
          onClick={handleRunBacktest}
          disabled={isRunning}
          className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors disabled:opacity-50"
        >
          <Play className="h-4 w-4" />
          <span>{isRunning ? 'Running...' : 'Run Backtest'}</span>
        </button>
      </div>

      {/* Configuration */}
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center space-x-2 mb-4">
          <Settings className="h-5 w-5 text-blue-400" />
          <h3 className="text-lg font-semibold text-white">Configuration</h3>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Symbol</label>
            <select
              value={backtestConfig.symbol}
              onChange={(e) => setBacktestConfig({...backtestConfig, symbol: e.target.value})}
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="BTCUSDT">BTCUSDT</option>
              <option value="ETHUSDT">ETHUSDT</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Start Date</label>
            <input
              type="date"
              value={backtestConfig.startDate}
              onChange={(e) => setBacktestConfig({...backtestConfig, startDate: e.target.value})}
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">End Date</label>
            <input
              type="date"
              value={backtestConfig.endDate}
              onChange={(e) => setBacktestConfig({...backtestConfig, endDate: e.target.value})}
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Initial Balance</label>
            <input
              type="number"
              value={backtestConfig.initialBalance}
              onChange={(e) => setBacktestConfig({...backtestConfig, initialBalance: Number(e.target.value)})}
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Max Leverage</label>
            <input
              type="number"
              min="1"
              max="10"
              value={backtestConfig.leverage}
              onChange={(e) => setBacktestConfig({...backtestConfig, leverage: Number(e.target.value)})}
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>

        <div className="mt-4">
          <h4 className="text-sm font-medium text-gray-400 mb-2">Signal Sources</h4>
          <div className="flex flex-wrap gap-4">
            {['enableML', 'enableTechnical', 'enableSentiment', 'enableFearGreed'].map((key) => (
              <label key={key} className="flex items-center space-x-2">
                <input
                  type="checkbox"
                  checked={backtestConfig[key]}
                  onChange={(e) => setBacktestConfig({...backtestConfig, [key]: e.target.checked})}
                  className="rounded focus:ring-2 focus:ring-blue-500"
                />
                <span className="text-white text-sm">
                  {key.replace('enable', '').replace(/([A-Z])/g, ' $1').trim()}
                </span>
              </label>
            ))}
          </div>
        </div>
      </div>

      {/* Results */}
      {backtestResults && (
        <>
          {/* Performance Metrics */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4 text-white">Performance Metrics</h3>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
              <div>
                <p className="text-sm text-gray-400">Total Return</p>
                <p className={`text-xl font-bold ${backtestResults.totalReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {formatPercent(backtestResults.totalReturn)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Sharpe Ratio</p>
                <p className="text-xl font-bold text-white">{backtestResults.sharpeRatio?.toFixed(2) || 'N/A'}</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Max Drawdown</p>
                <p className="text-xl font-bold text-red-400">{formatPercent(backtestResults.maxDrawdown)}</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Win Rate</p>
                <p className="text-xl font-bold text-blue-400">{formatPercent(backtestResults.winRate)}</p>
              </div>
            </div>
          </div>

          {/* Equity Curve */}
          <div className="bg-gray-800 rounded-lg p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-white">Backtest Equity Curve</h3>
              <button className="flex items-center space-x-2 px-3 py-2 bg-gray-700 text-white rounded hover:bg-gray-600 transition-colors">
                <Download className="h-4 w-4" />
                <span>Export</span>
              </button>
            </div>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={backtestResults.equityCurve}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="date" 
                    stroke="#9CA3AF"
                    fontSize={12}
                    tickFormatter={(date) => new Date(date).toLocaleDateString()}
                  />
                  <YAxis 
                    stroke="#9CA3AF"
                    fontSize={12}
                    tickFormatter={formatCurrency}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '6px'
                    }}
                    formatter={(value: number) => [formatCurrency(value), 'Portfolio Value']}
                    labelFormatter={(date) => new Date(date).toLocaleDateString()}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="portfolio" 
                    stroke="#3B82F6" 
                    strokeWidth={2}
                    dot={false}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="benchmark" 
                    stroke="#6B7280" 
                    strokeWidth={1}
                    strokeDasharray="5 5"
                    dot={false}
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Trade Analysis */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4 text-white">Trade Analysis</h3>
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div>
                <p className="text-sm text-gray-400">Total Trades</p>
                <p className="text-2xl font-bold text-white">{backtestResults.totalTrades}</p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Avg Trade Return</p>
                <p className={`text-2xl font-bold ${backtestResults.avgTradeReturn >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {formatPercent(backtestResults.avgTradeReturn)}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-400">Profit Factor</p>
                <p className="text-2xl font-bold text-blue-400">{backtestResults.profitFactor?.toFixed(2) || 'N/A'}</p>
              </div>
            </div>
          </div>
        </>
      )}

      {isRunning && (
        <div className="bg-gray-800 rounded-lg p-8 text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-400 mx-auto mb-4"></div>
          <p className="text-gray-400">Running backtest simulation...</p>
          <p className="text-sm text-gray-500 mt-2">This may take several minutes depending on the date range.</p>
        </div>
      )}
    </div>
  );
};