import React, { useState, useEffect } from 'react';
import { Shield, AlertTriangle, TrendingDown, DollarSign } from 'lucide-react';
import { CircularProgressbar, buildStyles } from 'react-circular-progressbar';

interface RiskMetrics {
  currentDrawdown: number;
  maxDrawdown: number;
  dailyPnL: number;
  dailyLossLimit: number;
  dailyProfitLimit: number;
  tradesCount: number;
  maxTradesPerDay: number;
  consecutiveLosses: number;
  maxConsecutiveLosses: number;
  marginRatio: number;
  exposureRatio: number;
  leverageRatio: number;
  riskScore: number;
  killSwitchStatus: boolean;
}

interface RiskPanelProps {
  service: any;
  data?: any;
}

export const RiskPanel: React.FC<RiskPanelProps> = ({ service, data }) => {
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics>({
    currentDrawdown: 0,
    maxDrawdown: 0,
    dailyPnL: 0,
    dailyLossLimit: -3000,
    dailyProfitLimit: 2000,
    tradesCount: 0,
    maxTradesPerDay: 15,
    consecutiveLosses: 0,
    maxConsecutiveLosses: 4,
    marginRatio: 0,
    exposureRatio: 0,
    leverageRatio: 0,
    riskScore: 0,
    killSwitchStatus: false
  });

  useEffect(() => {
    const loadRiskMetrics = async () => {
      try {
        const metrics = await service.getRiskMetrics();
        setRiskMetrics(metrics);
      } catch (error) {
        console.error('Failed to load risk metrics:', error);
      }
    };

    loadRiskMetrics();
    const interval = setInterval(loadRiskMetrics, 3000);
    return () => clearInterval(interval);
  }, [service]);

  const getRiskLevel = (score: number) => {
    if (score >= 80) return { level: 'Critical', color: 'text-red-400', bgColor: 'bg-red-100' };
    if (score >= 60) return { level: 'High', color: 'text-orange-400', bgColor: 'bg-orange-100' };
    if (score >= 40) return { level: 'Medium', color: 'text-yellow-400', bgColor: 'bg-yellow-100' };
    return { level: 'Low', color: 'text-green-400', bgColor: 'bg-green-100' };
  };

  const formatCurrency = (value: number) => `$${value.toFixed(2)}`;
  const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`;

  const riskLevel = getRiskLevel(riskMetrics.riskScore);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">Risk Management</h2>
        <div className={`flex items-center space-x-2 px-4 py-2 rounded-full ${riskLevel.bgColor}`}>
          <Shield className="h-5 w-5" />
          <span className="font-medium">Risk Level: {riskLevel.level}</span>
        </div>
      </div>

      {/* Kill Switch Status */}
      {riskMetrics.killSwitchStatus && (
        <div className="bg-red-800 border border-red-600 rounded-lg p-4">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="h-6 w-6 text-red-400" />
            <h3 className="text-lg font-bold text-red-400">Kill Switch Activated</h3>
          </div>
          <p className="text-red-200 mt-2">
            Trading has been halted due to risk limits being exceeded. Manual intervention required.
          </p>
        </div>
      )}

      {/* Risk Score */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Overall Risk Score</h3>
        <div className="flex items-center justify-center">
          <div className="w-32 h-32">
            <CircularProgressbar
              value={riskMetrics.riskScore}
              text={`${riskMetrics.riskScore}%`}
              styles={buildStyles({
                textColor: '#ffffff',
                pathColor: riskMetrics.riskScore >= 80 ? '#ef4444' : 
                         riskMetrics.riskScore >= 60 ? '#f97316' : 
                         riskMetrics.riskScore >= 40 ? '#eab308' : '#10b981',
                trailColor: '#374151'
              })}
            />
          </div>
        </div>
      </div>

      {/* Daily Limits */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center space-x-2 mb-4">
            <TrendingDown className="h-5 w-5 text-red-400" />
            <h4 className="font-semibold text-white">Daily Loss Tracking</h4>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">Current P&L:</span>
              <span className={riskMetrics.dailyPnL >= 0 ? 'text-green-400' : 'text-red-400'}>
                {formatCurrency(riskMetrics.dailyPnL)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Loss Limit:</span>
              <span className="text-red-400">{formatCurrency(riskMetrics.dailyLossLimit)}</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="h-2 bg-red-400 rounded-full"
                style={{ 
                  width: `${Math.min(100, Math.abs(riskMetrics.dailyPnL / riskMetrics.dailyLossLimit) * 100)}%`
                }}
              />
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center space-x-2 mb-4">
            <DollarSign className="h-5 w-5 text-green-400" />
            <h4 className="font-semibold text-white">Daily Profit Tracking</h4>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">Current P&L:</span>
              <span className={riskMetrics.dailyPnL >= 0 ? 'text-green-400' : 'text-red-400'}>
                {formatCurrency(riskMetrics.dailyPnL)}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Profit Limit:</span>
              <span className="text-green-400">{formatCurrency(riskMetrics.dailyProfitLimit)}</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="h-2 bg-green-400 rounded-full"
                style={{ 
                  width: `${Math.min(100, Math.max(0, riskMetrics.dailyPnL / riskMetrics.dailyProfitLimit) * 100)}%`
                }}
              />
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center space-x-2 mb-4">
            <Shield className="h-5 w-5 text-blue-400" />
            <h4 className="font-semibold text-white">Trade Count</h4>
          </div>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">Today's Trades:</span>
              <span className="text-white">{riskMetrics.tradesCount}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Daily Limit:</span>
              <span className="text-blue-400">{riskMetrics.maxTradesPerDay}</span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div 
                className="h-2 bg-blue-400 rounded-full"
                style={{ 
                  width: `${Math.min(100, (riskMetrics.tradesCount / riskMetrics.maxTradesPerDay) * 100)}%`
                }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* Risk Metrics Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <div className="bg-gray-800 rounded-lg p-6">
          <h4 className="text-sm font-medium text-gray-400 mb-2">Current Drawdown</h4>
          <p className="text-2xl font-bold text-red-400">{formatPercent(riskMetrics.currentDrawdown)}</p>
          <p className="text-xs text-gray-400 mt-1">Max: {formatPercent(riskMetrics.maxDrawdown)}</p>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h4 className="text-sm font-medium text-gray-400 mb-2">Margin Ratio</h4>
          <p className="text-2xl font-bold text-blue-400">{formatPercent(riskMetrics.marginRatio)}</p>
          <p className="text-xs text-gray-400 mt-1">Available margin</p>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h4 className="text-sm font-medium text-gray-400 mb-2">Exposure Ratio</h4>
          <p className="text-2xl font-bold text-orange-400">{formatPercent(riskMetrics.exposureRatio)}</p>
          <p className="text-xs text-gray-400 mt-1">Portfolio exposure</p>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h4 className="text-sm font-medium text-gray-400 mb-2">Consecutive Losses</h4>
          <p className="text-2xl font-bold text-red-400">{riskMetrics.consecutiveLosses}</p>
          <p className="text-xs text-gray-400 mt-1">Max: {riskMetrics.maxConsecutiveLosses}</p>
        </div>
      </div>
    </div>
  );
};