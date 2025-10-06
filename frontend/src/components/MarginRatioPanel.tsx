import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, AlertTriangle, Shield, DollarSign } from 'lucide-react';

interface MarginData {
  total_balance: number;
  available_balance: number;
  used_margin: number;
  margin_ratio: number;
  maintenance_margin: number;
  initial_margin: number;
  unrealized_pnl: number;
  realized_pnl: number;
  positions_count: number;
  leverage: number;
  risk_level: 'low' | 'medium' | 'high' | 'critical';
}

interface MarginRatioPanelProps {
  service: any;
}

export const MarginRatioPanel: React.FC<MarginRatioPanelProps> = ({ service }) => {
  const [marginData, setMarginData] = useState<MarginData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<string | null>(null);

  const fetchMarginData = async () => {
    try {
      setLoading(true);
      const data = await service.getAccountInfo();
      if (data.success) {
        // Calculate margin ratio and risk level
        const totalBalance = data.wallet.balance || 0;
        const availableBalance = data.wallet.available || 0;
        const usedMargin = totalBalance - availableBalance;
        const marginRatio = totalBalance > 0 ? (usedMargin / totalBalance) * 100 : 0;
        
        // Determine risk level
        let riskLevel: 'low' | 'medium' | 'high' | 'critical' = 'low';
        if (marginRatio >= 80) riskLevel = 'critical';
        else if (marginRatio >= 60) riskLevel = 'high';
        else if (marginRatio >= 40) riskLevel = 'medium';

        const marginData: MarginData = {
          total_balance: totalBalance,
          available_balance: availableBalance,
          used_margin: usedMargin,
          margin_ratio: marginRatio,
          maintenance_margin: totalBalance * 0.1, // 10% maintenance margin
          initial_margin: totalBalance * 0.2, // 20% initial margin
          unrealized_pnl: data.positions.total_pnl || 0,
          realized_pnl: 0, // Would need historical data
          positions_count: data.positions.active_count || 0,
          leverage: 5.0, // Default leverage
          risk_level: riskLevel
        };

        setMarginData(marginData);
        setLastUpdate(new Date().toISOString());
        setError(null);
      } else {
        setError('Failed to fetch margin data');
      }
    } catch (err) {
      console.error('Error fetching margin data:', err);
      setError('Error fetching margin data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMarginData();
    const interval = setInterval(fetchMarginData, 3000); // Refresh every 3 seconds
    return () => clearInterval(interval);
  }, []);

  const getRiskColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low':
        return 'text-green-400';
      case 'medium':
        return 'text-yellow-400';
      case 'high':
        return 'text-orange-400';
      case 'critical':
        return 'text-red-400';
      default:
        return 'text-gray-400';
    }
  };

  const getRiskBgColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'low':
        return 'bg-green-900/20 border-green-500';
      case 'medium':
        return 'bg-yellow-900/20 border-yellow-500';
      case 'high':
        return 'bg-orange-900/20 border-orange-500';
      case 'critical':
        return 'bg-red-900/20 border-red-500';
      default:
        return 'bg-gray-900/20 border-gray-500';
    }
  };

  const formatCurrency = (amount: number) => {
    return amount.toLocaleString('en-US', { 
      style: 'currency', 
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    });
  };

  const formatPercentage = (value: number) => {
    return `${value.toFixed(2)}%`;
  };

  if (loading && !marginData) {
    return (
      <div className="p-6 flex items-center justify-center">
        <div className="text-gray-400">Loading margin data...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="p-6">
        <div className="bg-red-900/50 border border-red-500 text-red-200 px-4 py-3 rounded-lg">
          {error}
        </div>
      </div>
    );
  }

  if (!marginData) {
    return (
      <div className="p-6">
        <div className="text-gray-400">No margin data available</div>
      </div>
    );
  }

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">Margin Ratio & Risk</h2>
        <div className="text-sm text-gray-400">
          {lastUpdate && `Last updated: ${new Date(lastUpdate).toLocaleString()}`}
        </div>
      </div>

      {/* Risk Level Alert */}
      <div className={`p-4 rounded-lg border ${getRiskBgColor(marginData.risk_level)}`}>
        <div className="flex items-center space-x-3">
          {marginData.risk_level === 'critical' ? (
            <AlertTriangle className="h-6 w-6 text-red-400" />
          ) : marginData.risk_level === 'high' ? (
            <TrendingUp className="h-6 w-6 text-orange-400" />
          ) : (
            <Shield className="h-6 w-6 text-green-400" />
          )}
          <div>
            <h3 className={`text-lg font-semibold ${getRiskColor(marginData.risk_level)}`}>
              Risk Level: {marginData.risk_level.toUpperCase()}
            </h3>
            <p className="text-sm text-gray-300">
              {marginData.risk_level === 'critical' && 'Immediate action required - margin call risk'}
              {marginData.risk_level === 'high' && 'High risk - consider reducing positions'}
              {marginData.risk_level === 'medium' && 'Moderate risk - monitor closely'}
              {marginData.risk_level === 'low' && 'Low risk - healthy margin levels'}
            </p>
          </div>
        </div>
      </div>

      {/* Key Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Total Balance */}
        <div className="bg-gray-800 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <DollarSign className="h-5 w-5 text-blue-400" />
            <h3 className="text-sm font-medium text-gray-300">Total Balance</h3>
          </div>
          <p className="text-2xl font-bold text-white mt-2">
            {formatCurrency(marginData.total_balance)}
          </p>
        </div>

        {/* Available Balance */}
        <div className="bg-gray-800 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <TrendingUp className="h-5 w-5 text-green-400" />
            <h3 className="text-sm font-medium text-gray-300">Available</h3>
          </div>
          <p className="text-2xl font-bold text-white mt-2">
            {formatCurrency(marginData.available_balance)}
          </p>
        </div>

        {/* Used Margin */}
        <div className="bg-gray-800 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <TrendingDown className="h-5 w-5 text-orange-400" />
            <h3 className="text-sm font-medium text-gray-300">Used Margin</h3>
          </div>
          <p className="text-2xl font-bold text-white mt-2">
            {formatCurrency(marginData.used_margin)}
          </p>
        </div>

        {/* Margin Ratio */}
        <div className="bg-gray-800 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <Shield className="h-5 w-5 text-purple-400" />
            <h3 className="text-sm font-medium text-gray-300">Margin Ratio</h3>
          </div>
          <p className={`text-2xl font-bold mt-2 ${getRiskColor(marginData.risk_level)}`}>
            {formatPercentage(marginData.margin_ratio)}
          </p>
        </div>
      </div>

      {/* Detailed Metrics */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Balance Breakdown */}
        <div className="bg-gray-800 p-6 rounded-lg">
          <h3 className="text-lg font-semibold text-white mb-4">Balance Breakdown</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Total Balance</span>
              <span className="text-white font-medium">{formatCurrency(marginData.total_balance)}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Available Balance</span>
              <span className="text-green-400 font-medium">{formatCurrency(marginData.available_balance)}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Used Margin</span>
              <span className="text-orange-400 font-medium">{formatCurrency(marginData.used_margin)}</span>
            </div>
            <div className="border-t border-gray-700 pt-3">
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Margin Ratio</span>
                <span className={`font-medium ${getRiskColor(marginData.risk_level)}`}>
                  {formatPercentage(marginData.margin_ratio)}
                </span>
              </div>
            </div>
          </div>
        </div>

        {/* PnL & Positions */}
        <div className="bg-gray-800 p-6 rounded-lg">
          <h3 className="text-lg font-semibold text-white mb-4">PnL & Positions</h3>
          <div className="space-y-3">
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Unrealized PnL</span>
              <span className={`font-medium ${marginData.unrealized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {formatCurrency(marginData.unrealized_pnl)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Realized PnL</span>
              <span className={`font-medium ${marginData.realized_pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {formatCurrency(marginData.realized_pnl)}
              </span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Active Positions</span>
              <span className="text-white font-medium">{marginData.positions_count}</span>
            </div>
            <div className="flex justify-between items-center">
              <span className="text-gray-300">Leverage</span>
              <span className="text-white font-medium">{marginData.leverage}x</span>
            </div>
          </div>
        </div>
      </div>

      {/* Margin Requirements */}
      <div className="bg-gray-800 p-6 rounded-lg">
        <h3 className="text-lg font-semibold text-white mb-4">Margin Requirements</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-400">
              {formatPercentage(10)}
            </div>
            <div className="text-sm text-gray-300">Maintenance Margin</div>
            <div className="text-xs text-gray-400">
              {formatCurrency(marginData.maintenance_margin)}
            </div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-yellow-400">
              {formatPercentage(20)}
            </div>
            <div className="text-sm text-gray-300">Initial Margin</div>
            <div className="text-xs text-gray-400">
              {formatCurrency(marginData.initial_margin)}
            </div>
          </div>
          <div className="text-center">
            <div className={`text-2xl font-bold ${getRiskColor(marginData.risk_level)}`}>
              {formatPercentage(marginData.margin_ratio)}
            </div>
            <div className="text-sm text-gray-300">Current Ratio</div>
            <div className="text-xs text-gray-400">
              {formatCurrency(marginData.used_margin)}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
