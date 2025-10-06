import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, X, Target, Shield } from 'lucide-react';

interface Position {
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  leverage: number;
  marginMode: string;
  stopLoss?: number;
  takeProfit?: number;
  liquidationPrice: number;
  timestamp: string;
}

interface PositionsPanelProps {
  service: any;
}

export const PositionsPanel: React.FC<PositionsPanelProps> = ({ service }) => {
  const [positions, setPositions] = useState<Position[]>([]);
  const [loading, setLoading] = useState(true);
  const [closingPositions, setClosingPositions] = useState<Set<string>>(new Set());
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  useEffect(() => {
    const loadPositions = async () => {
      try {
        const data = await service.getPositions();
        setPositions(data);
        setLoading(false);
      } catch (error) {
        console.error('Failed to load positions:', error);
        setLoading(false);
      }
    };

    loadPositions();
    const interval = setInterval(loadPositions, 5000); // Reduced from 2s to 5s for better performance
    return () => clearInterval(interval);
  }, [service]);

  const handleClosePosition = async (symbol: string) => {
    try {
      // Add to closing positions set
      setClosingPositions(prev => new Set(prev).add(symbol));
      setMessage(null);

      // Find the position details to pass to the service
      const position = positions.find(p => p.symbol === symbol);
      const positionDetails = position ? {
        side: position.side === 'long' ? 'Buy' : 'Sell',
        size: position.size,
        entryPrice: position.entryPrice
      } : undefined;

      const result = await service.closePosition(symbol, positionDetails);
      
      if (result.success) {
        setMessage({ type: 'success', text: result.message || `Position closed successfully for ${symbol}` });
        // Refresh positions after successful close
        setTimeout(() => {
          const loadPositions = async () => {
            try {
              const data = await service.getPositions();
              setPositions(data);
            } catch (error) {
              console.error('Failed to refresh positions:', error);
            }
          };
          loadPositions();
        }, 1000);
      } else {
        setMessage({ type: 'error', text: result.error || `Failed to close position for ${symbol}` });
      }
    } catch (error) {
      console.error('Failed to close position:', error);
      setMessage({ type: 'error', text: `Failed to close position for ${symbol}` });
    } finally {
      // Remove from closing positions set
      setClosingPositions(prev => {
        const newSet = new Set(prev);
        newSet.delete(symbol);
        return newSet;
      });
    }
  };

  const formatCurrency = (value: number) => `$${value.toFixed(2)}`;
  const formatPercent = (value: number) => `${(value * 100).toFixed(2)}%`;

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-blue-400"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">Open Positions</h2>
        <div className="text-sm text-gray-400">
          {positions.length} position{positions.length !== 1 ? 's' : ''}
        </div>
      </div>

      {/* Message Display */}
      {message && (
        <div className={`p-4 rounded-lg ${
          message.type === 'success' 
            ? 'bg-green-900 text-green-100 border border-green-700' 
            : 'bg-red-900 text-red-100 border border-red-700'
        }`}>
          <div className="flex items-center justify-between">
            <span>{message.text}</span>
            <button
              onClick={() => setMessage(null)}
              className="ml-4 text-lg font-bold hover:opacity-70"
            >
              ×
            </button>
          </div>
        </div>
      )}

      {positions.length === 0 ? (
        <div className="bg-gray-800 rounded-lg p-8 text-center">
          <Target className="h-12 w-12 text-gray-600 mx-auto mb-4" />
          <p className="text-gray-400">No open positions</p>
        </div>
      ) : (
        <div className="grid gap-4">
          {positions.map((position, index) => (
            <div key={`${position.symbol}-${index}`} className="bg-gray-800 rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-3">
                  <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm font-medium ${
                    position.side === 'long' 
                      ? 'bg-green-100 text-green-800' 
                      : 'bg-red-100 text-red-800'
                  }`}>
                    {position.side === 'long' ? (
                      <TrendingUp className="h-4 w-4" />
                    ) : (
                      <TrendingDown className="h-4 w-4" />
                    )}
                    <span>{position.side.toUpperCase()}</span>
                  </div>
                  <h3 className="text-lg font-bold text-white">{position.symbol}</h3>
                  <span className="text-sm text-gray-400">
                    {position.leverage}x {position.marginMode}
                  </span>
                </div>
                <button
                  onClick={() => handleClosePosition(position.symbol)}
                  disabled={closingPositions.has(position.symbol)}
                  className={`flex items-center space-x-1 px-3 py-2 rounded transition-colors ${
                    closingPositions.has(position.symbol)
                      ? 'bg-gray-600 text-gray-400 cursor-not-allowed'
                      : 'bg-red-600 text-white hover:bg-red-700'
                  }`}
                >
                  {closingPositions.has(position.symbol) ? (
                    <>
                      <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                      <span>Closing...</span>
                    </>
                  ) : (
                    <>
                      <X className="h-4 w-4" />
                      <span>Close</span>
                    </>
                  )}
                </button>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                <div>
                  <p className="text-sm text-gray-400">Size</p>
                  <p className="text-white font-medium">{position.size.toFixed(6)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-400">Entry Price</p>
                  <p className="text-white font-medium">{formatCurrency(position.entryPrice)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-400">Current Price</p>
                  <p className="text-white font-medium">{formatCurrency(position.currentPrice)}</p>
                </div>
                <div>
                  <p className="text-sm text-gray-400">Unrealized P&L</p>
                  <p className={`font-medium ${
                    position.unrealizedPnL >= 0 ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {formatCurrency(position.unrealizedPnL)}
                  </p>
                </div>
              </div>

              <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <p className="text-sm text-gray-400">Stop Loss</p>
                  <p className="text-white font-medium">
                    {position.stopLoss ? formatCurrency(position.stopLoss) : 'N/A'}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-400">Take Profit</p>
                  <p className="text-white font-medium">
                    {position.takeProfit ? formatCurrency(position.takeProfit) : 'N/A'}
                  </p>
                </div>
                <div>
                  <p className="text-sm text-gray-400">Liquidation Price</p>
                  <p className="text-red-400 font-medium">
                    {formatCurrency(position.liquidationPrice)}
                  </p>
                </div>
              </div>

              <div className="mt-4 text-xs text-gray-400">
                Opened: {new Date(position.timestamp).toLocaleString()}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};