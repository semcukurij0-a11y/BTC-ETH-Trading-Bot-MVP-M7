import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, DollarSign, Clock, AlertCircle, RefreshCw, CheckCircle } from 'lucide-react';

interface FundingRate {
  symbol: string;
  funding_rate: number;
  funding_rate_8h: number;
  next_funding_time: string;
  predicted_funding_rate: number;
  index_price: number;
  mark_price: number;
  last_funding_rate: number;
}

interface FundingPanelProps {
  service: any;
}

export const FundingPanel: React.FC<FundingPanelProps> = ({ service }) => {
  const [fundingRates, setFundingRates] = useState<FundingRate[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<string | null>(null);

  const fetchFundingRates = async () => {
    try {
      setLoading(true);
      
      // Try to get real funding data from service first
      try {
        const realData = await service.getFundingRates();
        if (realData.success && realData.funding_rates) {
          setFundingRates(realData.funding_rates);
          setLastUpdate(new Date().toISOString());
          setError(null);
          return;
        }
      } catch (err) {
        console.log('Real funding data not available, using mock data');
      }
      
      // Fallback to mock data if real data is not available
      const mockFundingRates: FundingRate[] = [
        {
          symbol: 'BTCUSDT',
          funding_rate: 0.0001, // 0.01%
          funding_rate_8h: 0.0008, // 0.08%
          next_funding_time: new Date(Date.now() + 8 * 60 * 60 * 1000).toISOString(),
          predicted_funding_rate: 0.0002,
          index_price: 50000,
          mark_price: 50025,
          last_funding_rate: 0.00005
        },
        {
          symbol: 'ETHUSDT',
          funding_rate: -0.0002, // -0.02%
          funding_rate_8h: -0.0016, // -0.16%
          next_funding_time: new Date(Date.now() + 8 * 60 * 60 * 1000).toISOString(),
          predicted_funding_rate: -0.0001,
          index_price: 3000,
          mark_price: 2995,
          last_funding_rate: -0.0003
        },
        {
          symbol: 'SOLUSDT',
          funding_rate: 0.0003, // 0.03%
          funding_rate_8h: 0.0024, // 0.24%
          next_funding_time: new Date(Date.now() + 8 * 60 * 60 * 1000).toISOString(),
          predicted_funding_rate: 0.0004,
          index_price: 100,
          mark_price: 100.3,
          last_funding_rate: 0.0002
        },
        {
          symbol: 'ADAUSDT',
          funding_rate: -0.0001, // -0.01%
          funding_rate_8h: -0.0008, // -0.08%
          next_funding_time: new Date(Date.now() + 8 * 60 * 60 * 1000).toISOString(),
          predicted_funding_rate: 0.0000,
          index_price: 0.5,
          mark_price: 0.4995,
          last_funding_rate: -0.0002
        }
      ];

      setFundingRates(mockFundingRates);
      setLastUpdate(new Date().toISOString());
      setError(null);
    } catch (err) {
      console.error('Error fetching funding rates:', err);
      setError('Error fetching funding rates');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFundingRates();
    const interval = setInterval(fetchFundingRates, 30000); // Refresh every 30 seconds
    return () => clearInterval(interval);
  }, []);

  const formatFundingRate = (rate: number) => {
    const percentage = (rate * 100).toFixed(4);
    return `${rate >= 0 ? '+' : ''}${percentage}%`;
  };

  const formatCurrency = (amount: number) => {
    return amount.toLocaleString('en-US', { 
      style: 'currency', 
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    });
  };

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const getFundingRateColor = (rate: number) => {
    if (rate > 0.0005) return 'text-red-400'; // High positive funding
    if (rate > 0.0001) return 'text-orange-400'; // Moderate positive funding
    if (rate > -0.0001) return 'text-gray-400'; // Neutral
    if (rate > -0.0005) return 'text-blue-400'; // Moderate negative funding
    return 'text-green-400'; // High negative funding
  };

  const getFundingRateBgColor = (rate: number) => {
    if (rate > 0.0005) return 'bg-red-900/20 border-red-500';
    if (rate > 0.0001) return 'bg-orange-900/20 border-orange-500';
    if (rate > -0.0001) return 'bg-gray-900/20 border-gray-500';
    if (rate > -0.0005) return 'bg-blue-900/20 border-blue-500';
    return 'bg-green-900/20 border-green-500';
  };

  const getFundingRateIcon = (rate: number) => {
    if (rate > 0) return <TrendingUp className="h-4 w-4" />;
    if (rate < 0) return <TrendingDown className="h-4 w-4" />;
    return <DollarSign className="h-4 w-4" />;
  };

  const getFundingRateLevel = (rate: number) => {
    if (rate > 0.001) return 'Very High';
    if (rate > 0.0005) return 'High';
    if (rate > 0.0001) return 'Moderate';
    if (rate > -0.0001) return 'Low';
    if (rate > -0.0005) return 'Moderate';
    if (rate > -0.001) return 'High';
    return 'Very High';
  };

  const getTotalFundingCost = () => {
    return fundingRates.reduce((total, rate) => {
      // Calculate estimated funding cost based on position size and funding rate
      const positionSize = 1000; // Assume $1000 position size for calculation
      const fundingCost = positionSize * rate.funding_rate;
      return total + fundingCost;
    }, 0);
  };

  if (loading && fundingRates.length === 0) {
    return (
      <div className="p-6 flex items-center justify-center">
        <div className="text-gray-400">Loading funding rates...</div>
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

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">Funding Rates</h2>
        <div className="flex items-center space-x-4">
          <button
            onClick={fetchFundingRates}
            disabled={loading}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>
      </div>

      {/* Status Info */}
      <div className="flex items-center justify-between text-sm text-gray-400">
        <div>
          {lastUpdate && (
            <span>Last updated: {new Date(lastUpdate).toLocaleString()}</span>
          )}
        </div>
        <div>
          Next funding in: {fundingRates.length > 0 ? 
            Math.floor((new Date(fundingRates[0].next_funding_time).getTime() - Date.now()) / (1000 * 60 * 60)) + 'h' : 
            'Unknown'
          }
        </div>
      </div>

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-gray-800 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <DollarSign className="h-5 w-5 text-blue-400" />
            <h3 className="text-sm font-medium text-gray-300">Total Funding Cost</h3>
          </div>
          <p className="text-2xl font-bold text-white mt-2">
            {formatCurrency(getTotalFundingCost())}
          </p>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <TrendingUp className="h-5 w-5 text-green-400" />
            <h3 className="text-sm font-medium text-gray-300">Positive Rates</h3>
          </div>
          <p className="text-2xl font-bold text-white mt-2">
            {fundingRates.filter(r => r.funding_rate > 0).length}
          </p>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <TrendingDown className="h-5 w-5 text-red-400" />
            <h3 className="text-sm font-medium text-gray-300">Negative Rates</h3>
          </div>
          <p className="text-2xl font-bold text-white mt-2">
            {fundingRates.filter(r => r.funding_rate < 0).length}
          </p>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <Clock className="h-5 w-5 text-yellow-400" />
            <h3 className="text-sm font-medium text-gray-300">Next Funding</h3>
          </div>
          <p className="text-2xl font-bold text-white mt-2">
            {fundingRates.length > 0 ? 
              Math.floor((new Date(fundingRates[0].next_funding_time).getTime() - Date.now()) / (1000 * 60 * 60)) + 'h' : 
              'N/A'
            }
          </p>
        </div>
      </div>

      {/* Funding Rates Table */}
      <div className="bg-gray-800 rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-700">
          <h3 className="text-lg font-semibold text-white">Current Funding Rates</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full">
            <thead className="bg-gray-700">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Symbol
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Current Rate
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  8h Rate
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Predicted
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Level
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Index Price
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Mark Price
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Next Funding
                </th>
              </tr>
            </thead>
            <tbody className="bg-gray-800 divide-y divide-gray-700">
              {fundingRates.length === 0 ? (
                <tr>
                  <td colSpan={8} className="px-6 py-8 text-center text-gray-400">
                    No funding rate data available
                  </td>
                </tr>
              ) : (
                fundingRates.map((rate, index) => (
                  <tr key={index} className="hover:bg-gray-700">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-white">
                      {rate.symbol}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <div className="flex items-center space-x-2">
                        {getFundingRateIcon(rate.funding_rate)}
                        <span className={getFundingRateColor(rate.funding_rate)}>
                          {formatFundingRate(rate.funding_rate)}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <span className={getFundingRateColor(rate.funding_rate_8h)}>
                        {formatFundingRate(rate.funding_rate_8h)}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <span className={getFundingRateColor(rate.predicted_funding_rate)}>
                        {formatFundingRate(rate.predicted_funding_rate)}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        getFundingRateBgColor(rate.funding_rate)
                      }`}>
                        {getFundingRateLevel(rate.funding_rate)}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                      {formatCurrency(rate.index_price)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                      {formatCurrency(rate.mark_price)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                      {formatTime(rate.next_funding_time)}
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Funding Rate Trends */}
      <div className="bg-gray-800 p-6 rounded-lg">
        <h3 className="text-lg font-semibold text-white mb-4">Funding Rate Trends</h3>
        <div className="text-center text-gray-400 py-8">
          <TrendingUp className="h-12 w-12 mx-auto mb-4 text-gray-600" />
          <p>Funding rate trends chart would be displayed here</p>
          <p className="text-sm">Showing funding rate changes over time</p>
        </div>
      </div>

      {/* Funding Rate Alerts */}
      <div className="bg-gray-800 p-6 rounded-lg">
        <h3 className="text-lg font-semibold text-white mb-4">Funding Rate Alerts</h3>
        <div className="space-y-3">
          {fundingRates.filter(rate => Math.abs(rate.funding_rate) > 0.0005).map((rate, index) => (
            <div key={index} className={`p-3 rounded-lg border ${getFundingRateBgColor(rate.funding_rate)}`}>
              <div className="flex items-center space-x-3">
                <AlertCircle className="h-5 w-5 text-yellow-400" />
                <div>
                  <p className="text-sm font-medium text-white">
                    High funding rate detected for {rate.symbol}
                  </p>
                  <p className="text-xs text-gray-300">
                    Current rate: {formatFundingRate(rate.funding_rate)} - 
                    {rate.funding_rate > 0 ? ' Consider shorting' : ' Consider longing'}
                  </p>
                </div>
              </div>
            </div>
          ))}
          {fundingRates.filter(rate => Math.abs(rate.funding_rate) > 0.0005).length === 0 && (
            <div className="text-center text-gray-400 py-4">
              <CheckCircle className="h-8 w-8 mx-auto mb-2 text-green-400" />
              <p>No high funding rate alerts</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
