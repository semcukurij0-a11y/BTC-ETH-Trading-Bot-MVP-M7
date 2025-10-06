import React from 'react';
import { useRealTimeData } from '../hooks/useRealTimeData';

export const RealTimeDataTest: React.FC = () => {
  const {
    positions,
    accountInfo,
    tradingStats,
    systemStatus,
    isLoading,
    error,
    lastUpdate,
    startPolling,
    stopPolling,
    refresh,
    isPolling
  } = useRealTimeData({ pollingInterval: 3000, autoStart: true });

  return (
    <div className="p-6 bg-white rounded-lg shadow-lg">
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-2xl font-bold text-gray-800">Real-Time Data Test</h2>
        <div className="flex gap-2">
          <button
            onClick={startPolling}
            disabled={isPolling}
            className="px-4 py-2 bg-green-500 text-white rounded disabled:bg-gray-400"
          >
            Start Polling
          </button>
          <button
            onClick={stopPolling}
            disabled={!isPolling}
            className="px-4 py-2 bg-red-500 text-white rounded disabled:bg-gray-400"
          >
            Stop Polling
          </button>
          <button
            onClick={refresh}
            className="px-4 py-2 bg-blue-500 text-white rounded"
          >
            Refresh Now
          </button>
        </div>
      </div>

      {isLoading && (
        <div className="mb-4 p-3 bg-blue-100 text-blue-800 rounded">
          🔄 Loading live data...
        </div>
      )}

      {error && (
        <div className="mb-4 p-3 bg-red-100 text-red-800 rounded">
          ❌ Error: {error}
        </div>
      )}

      {lastUpdate && (
        <div className="mb-4 p-3 bg-green-100 text-green-800 rounded">
          ✅ Last updated: {new Date(lastUpdate).toLocaleTimeString()}
        </div>
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Account Info */}
        <div className="bg-gray-50 p-4 rounded">
          <h3 className="text-lg font-semibold mb-3">Account Info</h3>
          {accountInfo ? (
            <div className="space-y-2">
              <div className="flex justify-between">
                <span>Balance:</span>
                <span className="font-mono">
                  ${accountInfo.wallet?.balance?.toFixed(2) || '0.00'}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Available:</span>
                <span className="font-mono">
                  ${accountInfo.wallet?.available?.toFixed(2) || '0.00'}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Active Positions:</span>
                <span className="font-mono">
                  {accountInfo.positions?.activeCount || 0}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Total PnL:</span>
                <span className={`font-mono ${(accountInfo.positions?.totalPnL || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  ${accountInfo.positions?.totalPnL?.toFixed(2) || '0.00'}
                </span>
              </div>
              <div className="text-sm text-gray-600">
                API Status: {accountInfo.wallet?.success ? '✅ Connected' : '❌ Disconnected'}
              </div>
            </div>
          ) : (
            <div className="text-gray-500">No account data available</div>
          )}
        </div>

        {/* Positions */}
        <div className="bg-gray-50 p-4 rounded">
          <h3 className="text-lg font-semibold mb-3">Positions ({positions.length})</h3>
          {positions.length > 0 ? (
            <div className="space-y-2 max-h-48 overflow-y-auto">
              {positions.map((pos, index) => (
                <div key={index} className="bg-white p-3 rounded border">
                  <div className="flex justify-between items-center">
                    <span className="font-semibold">{pos.symbol}</span>
                    <span className={`px-2 py-1 rounded text-sm ${
                      pos.side === 'Buy' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                    }`}>
                      {pos.side}
                    </span>
                  </div>
                  <div className="text-sm text-gray-600 mt-1">
                    Size: {pos.size} | Entry: ${pos.entryPrice?.toFixed(2)} | Current: ${pos.currentPrice?.toFixed(2)}
                  </div>
                  <div className={`text-sm font-semibold ${
                    pos.unrealizedPnL >= 0 ? 'text-green-600' : 'text-red-600'
                  }`}>
                    PnL: ${pos.unrealizedPnL?.toFixed(2)}
                  </div>
                  <div className="text-xs text-gray-500">
                    {pos.isActive ? '🟢 Active' : '⚪ Inactive'}
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-gray-500">No positions found</div>
          )}
        </div>

        {/* Trading Stats */}
        <div className="bg-gray-50 p-4 rounded">
          <h3 className="text-lg font-semibold mb-3">Trading Stats</h3>
          {tradingStats ? (
            <div className="space-y-2">
              <div className="flex justify-between">
                <span>Total PnL:</span>
                <span className={`font-mono ${(tradingStats.totalPnL || 0) >= 0 ? 'text-green-600' : 'text-red-600'}`}>
                  ${tradingStats.totalPnL?.toFixed(2) || '0.00'}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Account Balance:</span>
                <span className="font-mono">
                  ${tradingStats.accountBalance?.toFixed(2) || '0.00'}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Current Positions:</span>
                <span className="font-mono">
                  {tradingStats.currentPositions || 0}
                </span>
              </div>
            </div>
          ) : (
            <div className="text-gray-500">No trading stats available</div>
          )}
        </div>

        {/* System Status */}
        <div className="bg-gray-50 p-4 rounded">
          <h3 className="text-lg font-semibold mb-3">System Status</h3>
          {systemStatus ? (
            <div className="space-y-2">
              <div className="flex justify-between">
                <span>API Status:</span>
                <span className={`px-2 py-1 rounded text-sm ${
                  systemStatus.apiStatus === 'connected' ? 'bg-green-100 text-green-800' : 'bg-red-100 text-red-800'
                }`}>
                  {systemStatus.apiStatus}
                </span>
              </div>
              <div className="flex justify-between">
                <span>Data Feed:</span>
                <span className="font-mono">{systemStatus.dataFeed}</span>
              </div>
              <div className="flex justify-between">
                <span>Uptime:</span>
                <span className="font-mono">{systemStatus.uptime}</span>
              </div>
            </div>
          ) : (
            <div className="text-gray-500">No system status available</div>
          )}
        </div>
      </div>
    </div>
  );
};
