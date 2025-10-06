import React, { useState, useEffect, useCallback } from 'react';
import { TrendingUp, TrendingDown, DollarSign, Activity, RefreshCw } from 'lucide-react';
import { HealthService } from '../services/HealthService';

interface DashboardProps {
  service: any;
  data?: {
    tradingStats?: any;
    systemStatus?: any;
    lastUpdate?: string;
    isLoading?: boolean;
    error?: string;
  };
  healthService?: HealthService;
}

export const TradingDashboardSmooth: React.FC<DashboardProps> = ({ 
  service, 
  data, 
  healthService 
}) => {
  const [equityCurve, setEquityCurve] = useState([]);
  const [stats, setStats] = useState({
    totalPnL: 0,
    unrealizedPnL: 0,
    realizedPnL: 0,
    dailyPnL: 0,
    winRate: 0,
    totalTrades: 0,
    maxDrawdown: 0,
    sharpeRatio: 0,
    currentPositions: 0,
    marginRatio: 0
  });
  const [isRefreshing, setIsRefreshing] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<string | null>(null);

  // Smooth update function that doesn't cause full page reload
  const updateStats = useCallback((newStats: any) => {
    if (newStats) {
      setStats(prevStats => ({
        ...prevStats,
        ...newStats
      }));
      setLastUpdate(new Date().toISOString());
    }
  }, []);

  // Update stats when external data changes (smooth update)
  useEffect(() => {
    if (data?.tradingStats) {
      updateStats(data.tradingStats);
    }
  }, [data?.tradingStats, updateStats]);

  // Load equity curve once (static data)
  useEffect(() => {
    const loadEquityCurve = async () => {
      try {
        const equityData = await service.getEquityCurve();
        setEquityCurve(equityData || []);
      } catch (error) {
        console.error('Failed to load equity curve:', error);
      }
    };

    loadEquityCurve();
  }, [service]);

  // Manual refresh function (smooth)
  const handleManualRefresh = useCallback(async () => {
    setIsRefreshing(true);
    try {
      const [equityData, statsData] = await Promise.all([
        service.getEquityCurve(),
        service.getTradingStats()
      ]);
      
      setEquityCurve(equityData || []);
      updateStats(statsData);
    } catch (error) {
      console.error('Failed to refresh dashboard data:', error);
    } finally {
      setIsRefreshing(false);
    }
  }, [service, updateStats]);

  const formatCurrency = (value: number | undefined) => {
    if (value === undefined || value === null || isNaN(value)) {
      return '$0.00';
    }
    return `$${value.toFixed(2)}`;
  };
  
  const formatPercent = (value: number | undefined) => {
    if (value === undefined || value === null || isNaN(value)) {
      return '0.00%';
    }
    return `${(value * 100).toFixed(2)}%`;
  };

  return (
    <div className="space-y-6">
      {/* Status Bar with Smooth Updates */}
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-4 text-sm text-gray-400">
            <span>Dashboard Status: <span className="text-green-400">Live</span></span>
            {lastUpdate && (
              <span>Last updated: {new Date(lastUpdate).toLocaleTimeString()}</span>
            )}
            {data?.isLoading && (
              <div className="flex items-center space-x-2 text-blue-400">
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-400"></div>
                <span>Updating...</span>
              </div>
            )}
          </div>
          <button
            onClick={handleManualRefresh}
            disabled={isRefreshing}
            className="flex items-center space-x-2 px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
          >
            <RefreshCw className={`h-4 w-4 ${isRefreshing ? 'animate-spin' : ''}`} />
            <span>{isRefreshing ? 'Refreshing...' : 'Refresh'}</span>
          </button>
        </div>
      </div>

      {/* Error Display */}
      {data?.error && (
        <div className="bg-red-900 border border-red-700 rounded-lg p-4">
          <div className="flex items-center">
            <div className="text-red-400 mr-2">⚠️</div>
            <div>
              <h3 className="text-red-400 font-medium">Data Error</h3>
              <p className="text-red-300 text-sm">{data.error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Stats Cards with Smooth Transitions */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-gray-800 rounded-lg p-6 transition-all duration-300 hover:bg-gray-750">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Total P&L</p>
              <p className={`text-2xl font-bold transition-colors duration-500 ${
                (stats.totalPnL || 0) >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {formatCurrency(stats.totalPnL)}
              </p>
              <div className="text-xs text-gray-500 mt-1">
                <span className="text-blue-400">Unrealized: {formatCurrency(stats.unrealizedPnL)}</span>
                <span className="mx-2">•</span>
                <span className="text-purple-400">Realized: {formatCurrency(stats.realizedPnL)}</span>
                <div className="mt-1">
                  <span className="text-yellow-400">Daily: {formatCurrency(stats.dailyPnL)}</span>
                </div>
              </div>
            </div>
            {(stats.totalPnL || 0) >= 0 ? (
              <TrendingUp className="h-8 w-8 text-green-400 transition-colors duration-500" />
            ) : (
              <TrendingDown className="h-8 w-8 text-red-400 transition-colors duration-500" />
            )}
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 transition-all duration-300 hover:bg-gray-750">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Unrealized P&L</p>
              <p className={`text-2xl font-bold transition-colors duration-500 ${
                (stats.unrealizedPnL || 0) >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {formatCurrency(stats.unrealizedPnL)}
              </p>
              <div className="text-xs text-gray-500 mt-1">
                Open positions
              </div>
            </div>
            <TrendingUp className="h-8 w-8 text-blue-400" />
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 transition-all duration-300 hover:bg-gray-750">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Realized P&L</p>
              <p className={`text-2xl font-bold transition-colors duration-500 ${
                (stats.realizedPnL || 0) >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {formatCurrency(stats.realizedPnL)}
              </p>
              <div className="text-xs text-gray-500 mt-1">
                Closed positions
              </div>
            </div>
            <DollarSign className="h-8 w-8 text-purple-400" />
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 transition-all duration-300 hover:bg-gray-750">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Daily P&L</p>
              <p className={`text-2xl font-bold transition-colors duration-500 ${
                (stats.dailyPnL || 0) >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {formatCurrency(stats.dailyPnL)}
              </p>
              <div className="text-xs text-gray-500 mt-1">
                Today's performance
              </div>
            </div>
            <Activity className="h-8 w-8 text-orange-400" />
          </div>
        </div>
      </div>

      {/* Additional Stats with Smooth Updates */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gray-800 rounded-lg p-6 transition-all duration-300 hover:bg-gray-750">
          <h3 className="text-lg font-semibold text-white mb-4">Performance</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Win Rate</span>
              <span className="text-white font-medium transition-colors duration-500">
                {formatPercent(stats.winRate)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Total Trades</span>
              <span className="text-white font-medium transition-all duration-500">
                {stats.totalTrades}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Max Drawdown</span>
              <span className="text-red-400 font-medium transition-colors duration-500">
                {formatPercent(stats.maxDrawdown)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Sharpe Ratio</span>
              <span className="text-white font-medium transition-all duration-500">
                {stats.sharpeRatio.toFixed(2)}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 transition-all duration-300 hover:bg-gray-750">
          <h3 className="text-lg font-semibold text-white mb-4">Positions</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Active Positions</span>
              <span className="text-white font-medium transition-all duration-500">
                {stats.currentPositions}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Margin Ratio</span>
              <span className="text-white font-medium transition-colors duration-500">
                {formatPercent(stats.marginRatio)}
              </span>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6 transition-all duration-300 hover:bg-gray-750">
          <h3 className="text-lg font-semibold text-white mb-4">System Status</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Connection</span>
              <span className="text-green-400 font-medium">Connected</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Data Feed</span>
              <span className="text-blue-400 font-medium">Live</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Last Update</span>
              <span className="text-white font-medium">
                {lastUpdate ? new Date(lastUpdate).toLocaleTimeString() : 'N/A'}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Equity Curve Chart Placeholder */}
      <div className="bg-gray-800 rounded-lg p-6 transition-all duration-300 hover:bg-gray-750">
        <h3 className="text-lg font-semibold text-white mb-4">Equity Curve</h3>
        <div className="h-64 bg-gray-700 rounded flex items-center justify-center">
          <p className="text-gray-400">Equity curve chart would be displayed here</p>
        </div>
      </div>
    </div>
  );
};
