import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, X, DollarSign } from 'lucide-react';

interface Position {
  id: string;
  symbol: string;
  side: 'Buy' | 'Sell';
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  leverage: number;
  margin: number;
  timestamp: string;
}

interface PositionsPanelProps {
  service: any;
  data?: Position[];
  isLoading?: boolean;
  error?: string;
  lastUpdate?: string;
  hasChanges?: boolean;
  changeCount?: number;
  changeDetection?: any;
}

export const PositionsPanelOptimized: React.FC<PositionsPanelProps> = ({ 
  service, 
  data = [],
  isLoading = false,
  error,
  lastUpdate,
  hasChanges = false,
  changeCount = 0,
  changeDetection
}) => {
  // Ensure data is an array and has proper structure
  const positions = Array.isArray(data) ? data : [];
  const [closingPositions, setClosingPositions] = useState<Set<string>>(new Set());
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Debug logging
  useEffect(() => {
    console.log('🔍 PositionsPanelOptimized - Data received:', {
      data,
      positions,
      positionsLength: positions.length,
      isLoading,
      error,
      lastUpdate
    });
    
    if (positions.length > 0) {
      console.log('📊 First position details:', positions[0]);
      console.log('📊 Position properties:', {
        symbol: positions[0].symbol,
        side: positions[0].side,
        size: positions[0].size,
        entry_price: positions[0].entry_price,
        current_price: positions[0].current_price,
        unrealized_pnl: positions[0].unrealized_pnl,
        leverage: positions[0].leverage
      });
    } else {
      console.log('⚠️ No positions found - checking data structure:', {
        dataType: typeof data,
        isArray: Array.isArray(data),
        dataKeys: data ? Object.keys(data) : 'null',
        dataValue: data
      });
    }
  }, [data, positions, isLoading, error, lastUpdate]);

  const handleClosePosition = async (symbol: string) => {
    if (closingPositions.has(symbol)) return;

    setClosingPositions(prev => new Set(prev).add(symbol));
    setMessage(null);

    try {
      const result = await service.closePosition(symbol);
      if (result.success) {
        setMessage({ type: 'success', text: `Position ${symbol} closed successfully` });
      } else {
        setMessage({ type: 'error', text: result.message || 'Failed to close position' });
      }
    } catch (error) {
      setMessage({ type: 'error', text: 'Error closing position' });
    } finally {
      setClosingPositions(prev => {
        const newSet = new Set(prev);
        newSet.delete(symbol);
        return newSet;
      });
    }
  };

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  const formatPercent = (value: number) => {
    return `${(value * 100).toFixed(2)}%`;
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="animate-pulse">
            <div className="h-6 bg-gray-700 rounded mb-4"></div>
            <div className="space-y-3">
              {[...Array(3)].map((_, i) => (
                <div key={i} className="h-16 bg-gray-700 rounded"></div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Status Bar */}
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between text-sm text-gray-400">
            <div className="flex items-center space-x-4">
              <span>Active Positions: {positions.length}</span>
              {hasChanges && (
                <span className="flex items-center text-green-400">
                  <div className="w-2 h-2 bg-green-400 rounded-full mr-1 animate-pulse"></div>
                  Updated ({changeCount} changes)
                </span>
              )}
            </div>
            {lastUpdate && (
              <span>Last updated: {new Date(lastUpdate).toLocaleString()}</span>
            )}
          </div>
        </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900 border border-red-700 rounded-lg p-4">
          <div className="flex items-center">
            <div className="text-red-400 mr-2">⚠️</div>
            <div>
              <h3 className="text-red-400 font-medium">Data Error</h3>
              <p className="text-red-300 text-sm">{error}</p>
            </div>
          </div>
        </div>
      )}

      {/* Success/Error Messages */}
      {message && (
        <div className={`rounded-lg p-4 ${
          message.type === 'success' 
            ? 'bg-green-900 border border-green-700 text-green-300' 
            : 'bg-red-900 border border-red-700 text-red-300'
        }`}>
          {message.text}
        </div>
      )}

      {/* Positions Table */}
      <div className="bg-gray-800 rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-700">
          <h2 className="text-xl font-semibold text-white">Active Positions</h2>
        </div>
        
        {positions.length === 0 ? (
          <div className="px-6 py-12 text-center text-gray-400">
            <TrendingUp className="h-12 w-12 mx-auto mb-4 text-gray-600" />
            <p className="text-lg">No active positions</p>
            <p className="text-sm">Open positions will appear here</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-700">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                    Symbol
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                    Side
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                    Quantity
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                    Entry Price
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                    Current Price
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                    P&L
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                    Leverage
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="bg-gray-800 divide-y divide-gray-700">
                {positions.map((position) => (
                  <tr key={position.id} className="hover:bg-gray-700">
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-white">
                      {position.symbol}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        position.side === 'Buy' 
                          ? 'bg-green-900 text-green-200' 
                          : 'bg-red-900 text-red-200'
                      }`}>
                        {position.side}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                      {(position.size || position.quantity || 0).toFixed(6)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                      {formatCurrency(position.entry_price || 0)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                      {formatCurrency(position.current_price || 0)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <span className={`font-medium ${
                        (position.unrealized_pnl || 0) >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {formatCurrency(position.unrealized_pnl || 0)}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                      {position.leverage || 0}x
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <button
                        onClick={() => handleClosePosition(position.symbol)}
                        disabled={closingPositions.has(position.symbol)}
                        className="inline-flex items-center px-3 py-1 border border-red-600 text-red-400 rounded hover:bg-red-900 disabled:opacity-50 disabled:cursor-not-allowed"
                      >
                        {closingPositions.has(position.symbol) ? (
                          <>
                            <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-red-400 mr-2"></div>
                            Closing...
                          </>
                        ) : (
                          <>
                            <X className="h-3 w-3 mr-1" />
                            Close
                          </>
                        )}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Summary Stats */}
      {positions.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Total P&L</h3>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Unrealized P&L</span>
                <span className={`font-medium ${
                  positions.reduce((sum, pos) => sum + (pos.unrealized_pnl || 0), 0) >= 0 
                    ? 'text-green-400' 
                    : 'text-red-400'
                }`}>
                  {formatCurrency(positions.reduce((sum, pos) => sum + (pos.unrealized_pnl || 0), 0))}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Total Margin</span>
                <span className="text-white font-medium">
                  {formatCurrency(positions.reduce((sum, pos) => sum + (pos.margin_mode || pos.margin || 0), 0))}
                </span>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Position Summary</h3>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Total Positions</span>
                <span className="text-white font-medium">{positions.length}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Long Positions</span>
                <span className="text-green-400 font-medium">
                  {positions.filter(pos => pos.side === 'Buy').length}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Short Positions</span>
                <span className="text-red-400 font-medium">
                  {positions.filter(pos => pos.side === 'Sell').length}
                </span>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Risk Metrics</h3>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Avg Leverage</span>
                <span className="text-white font-medium">
                  {positions.length > 0 ? (positions.reduce((sum, pos) => sum + (pos.leverage || 0), 0) / positions.length).toFixed(1) : '0.0'}x
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Max Leverage</span>
                <span className="text-yellow-400 font-medium">
                  {positions.length > 0 ? Math.max(...positions.map(pos => pos.leverage || 0)) : 0}x
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
