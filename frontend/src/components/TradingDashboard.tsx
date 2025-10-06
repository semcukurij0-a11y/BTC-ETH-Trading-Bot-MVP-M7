import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Activity, AlertTriangle, Target, Heart } from 'lucide-react';
import { HealthService } from '../services/HealthService';
import { HeartbeatIndicator } from './HeartbeatIndicator';
import { SkeletonStats, SkeletonChart, SkeletonCard } from './LoadingSkeleton';

interface DashboardProps {
  service: any;
  data: any;
  healthService?: HealthService;
}

export const TradingDashboard: React.FC<DashboardProps> = ({ service, data, healthService }) => {
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
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const loadDashboardData = async () => {
      try {
        setIsLoading(true);
        const [equityData, statsData] = await Promise.all([
          service.getEquityCurve(),
          service.getTradingStats()
        ]);
        setEquityCurve(equityData || []);
        setStats(statsData || {
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
      } catch (error) {
        console.error('Failed to load dashboard data:', error);
        // Set default values on error
        setStats({
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
      } finally {
        setIsLoading(false);
      }
    };

    loadDashboardData();
    const interval = setInterval(loadDashboardData, 10000); // Reduced from 5s to 10s
    return () => clearInterval(interval);
  }, [service]);

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

  if (isLoading) {
    return (
      <div className="space-y-6">
        <SkeletonStats />
        <SkeletonChart />
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <SkeletonCard />
          <SkeletonCard />
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Stats Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Total P&L</p>
              <p className={`text-2xl font-bold ${(stats.totalPnL || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
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
              <TrendingUp className="h-8 w-8 text-green-400" />
            ) : (
              <TrendingDown className="h-8 w-8 text-red-400" />
            )}
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Unrealized P&L</p>
              <p className={`text-2xl font-bold ${(stats.unrealizedPnL || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {formatCurrency(stats.unrealizedPnL)}
              </p>
              <div className="text-xs text-gray-500 mt-1">
                Open positions
              </div>
            </div>
            <TrendingUp className="h-8 w-8 text-blue-400" />
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Realized P&L</p>
              <p className={`text-2xl font-bold ${(stats.realizedPnL || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {formatCurrency(stats.realizedPnL)}
              </p>
              <div className="text-xs text-gray-500 mt-1">
                Closed positions
              </div>
            </div>
            <DollarSign className="h-8 w-8 text-purple-400" />
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Daily P&L</p>
              <p className={`text-2xl font-bold ${(stats.dailyPnL || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
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

      {/* Additional PnL Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Win Rate</p>
              <p className="text-2xl font-bold text-white">{formatPercent(stats.winRate)}</p>
            </div>
            <Target className="h-8 w-8 text-purple-400" />
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Positions</p>
              <p className="text-2xl font-bold text-white">{stats.currentPositions || 0}</p>
            </div>
            <Activity className="h-8 w-8 text-orange-400" />
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm font-medium text-gray-400">Margin Ratio</p>
              <p className="text-2xl font-bold text-white">{formatPercent(stats.marginRatio)}</p>
            </div>
            <AlertTriangle className="h-8 w-8 text-yellow-400" />
          </div>
        </div>
      </div>


      {/* Equity Curve Chart */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Equity Curve</h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={equityCurve}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="timestamp" 
                stroke="#9CA3AF"
                fontSize={12}
                tickFormatter={(time) => new Date(time).toLocaleDateString()}
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
                formatter={(value: number) => [formatCurrency(value), 'Equity']}
                labelFormatter={(time) => new Date(time).toLocaleString()}
              />
              <Area 
                type="monotone" 
                dataKey="equity" 
                stroke="#3B82F6" 
                fill="url(#equityGradient)"
                strokeWidth={2}
              />
              <defs>
                <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3}/>
                  <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                </linearGradient>
              </defs>
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Additional Stats Row */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gray-800 rounded-lg p-6">
          <h4 className="text-sm font-medium text-gray-400 mb-2">Total Trades</h4>
          <p className="text-xl font-bold text-white">{stats.totalTrades || 0}</p>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-6">
          <h4 className="text-sm font-medium text-gray-400 mb-2">Max Drawdown</h4>
          <p className="text-xl font-bold text-red-400">{formatPercent(stats.maxDrawdown)}</p>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-6">
          <h4 className="text-sm font-medium text-gray-400 mb-2">Sharpe Ratio</h4>
          <p className="text-xl font-bold text-white">{(stats.sharpeRatio || 0).toFixed(2)}</p>
        </div>
      </div>

      {/* System Status */}
      {data && (
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold mb-4">System Status</h3>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="text-sm">
              <span className="text-gray-400">Mode:</span>
              <span className={`ml-2 px-2 py-1 rounded text-xs font-medium ${
                data.mode === 'PAPER' ? 'bg-blue-100 text-blue-800' : 'bg-red-100 text-red-800'
              }`}>
                {data.mode}
              </span>
            </div>
            <div className="text-sm">
              <span className="text-gray-400">API Status:</span>
              <span className={`ml-2 px-2 py-1 rounded text-xs font-medium ${
                data.apiStatus === 'connected' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
              }`}>
                {data.apiStatus}
              </span>
            </div>
            <div className="text-sm">
              <span className="text-gray-400">Data Feed:</span>
              <span className={`ml-2 px-2 py-1 rounded text-xs font-medium ${
                data.dataFeed === 'live' ? 'bg-green-100 text-green-800' : 'bg-yellow-100 text-yellow-800'
              }`}>
                {data.dataFeed}
              </span>
            </div>
            <div className="text-sm">
              <span className="text-gray-400">Last Update:</span>
              <span className="ml-2 text-white">
                {data.lastUpdate ? new Date(data.lastUpdate).toLocaleTimeString() : 'N/A'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Health Status */}
      {healthService && (
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold flex items-center space-x-2">
              <Heart className="h-5 w-5 text-red-400" />
              <span>System Health</span>
            </h3>
            <HeartbeatIndicator healthService={healthService} showDetails={true} />
          </div>
          <div className="text-sm text-gray-400">
            Real-time health monitoring is active. Click the Health tab for detailed system information.
          </div>
        </div>
      )}
    </div>
  );
};