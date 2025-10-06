import React, { useState } from 'react';
import { Clock, CheckCircle, XCircle, AlertCircle, RefreshCw } from 'lucide-react';

interface Order {
  order_id: string;
  symbol: string;
  side: string;
  order_type: string;
  quantity: number;
  price: number;
  status: string;
  filled_quantity: number;
  average_price: number;
  created_time: string;
  updated_time: string;
  timestamp: string;
}

interface OrdersPanelProps {
  service: any;
  data?: {
    orders?: Order[];
    success?: boolean;
    count?: number;
  };
  isLoading?: boolean;
  error?: string;
  lastUpdate?: string;
}

export const OrdersPanelOptimized: React.FC<OrdersPanelProps> = ({ 
  service, 
  data,
  isLoading = false,
  error,
  lastUpdate
}) => {
  const [filter, setFilter] = useState<'all' | 'open' | 'filled' | 'cancelled'>('all');

  const orders = data?.orders || [];

  const getStatusIcon = (status: string) => {
    switch (status.toLowerCase()) {
      case 'filled':
        return <CheckCircle className="h-4 w-4 text-green-400" />;
      case 'cancelled':
        return <XCircle className="h-4 w-4 text-red-400" />;
      case 'pending':
      case 'new':
        return <Clock className="h-4 w-4 text-yellow-400" />;
      default:
        return <AlertCircle className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status.toLowerCase()) {
      case 'filled':
        return 'text-green-400';
      case 'cancelled':
        return 'text-red-400';
      case 'pending':
      case 'new':
        return 'text-yellow-400';
      default:
        return 'text-gray-400';
    }
  };

  const filteredOrders = orders.filter(order => {
    if (filter === 'all') return true;
    if (filter === 'open') return ['pending', 'new', 'partially_filled'].includes(order.status.toLowerCase());
    if (filter === 'filled') return order.status.toLowerCase() === 'filled';
    if (filter === 'cancelled') return order.status.toLowerCase() === 'cancelled';
    return true;
  });

  const formatTime = (timestamp: string) => {
    return new Date(timestamp).toLocaleString();
  };

  const formatPrice = (price: number) => {
    return price.toLocaleString('en-US', { 
      style: 'currency', 
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    });
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="animate-pulse">
            <div className="h-6 bg-gray-700 rounded mb-4"></div>
            <div className="space-y-3">
              {[...Array(5)].map((_, i) => (
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
          <span>Total Orders: {orders.length}</span>
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

      {/* Filter Controls */}
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="flex items-center space-x-4">
          <span className="text-sm font-medium text-gray-300">Filter:</span>
          <div className="flex space-x-2">
            {[
              { key: 'all', label: 'All', count: orders.length },
              { key: 'open', label: 'Open', count: orders.filter(o => ['pending', 'new', 'partially_filled'].includes(o.status.toLowerCase())).length },
              { key: 'filled', label: 'Filled', count: orders.filter(o => o.status.toLowerCase() === 'filled').length },
              { key: 'cancelled', label: 'Cancelled', count: orders.filter(o => o.status.toLowerCase() === 'cancelled').length }
            ].map(({ key, label, count }) => (
              <button
                key={key}
                onClick={() => setFilter(key as any)}
                className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                  filter === key
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                }`}
              >
                {label} ({count})
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* Orders Table */}
      <div className="bg-gray-800 rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-700">
          <h2 className="text-xl font-semibold text-white">Order History</h2>
        </div>
        
        {filteredOrders.length === 0 ? (
          <div className="px-6 py-12 text-center text-gray-400">
            <Clock className="h-12 w-12 mx-auto mb-4 text-gray-600" />
            <p className="text-lg">No orders found</p>
            <p className="text-sm">Orders matching your filter will appear here</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-700">
                <tr>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                    Order ID
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                    Symbol
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                    Side
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                    Type
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                    Quantity
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                    Price
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                    Status
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                    Filled
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                    Avg Price
                  </th>
                  <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                    Created
                  </th>
                </tr>
              </thead>
              <tbody className="bg-gray-800 divide-y divide-gray-700">
                {filteredOrders.map((order) => (
                  <tr key={order.order_id} className="hover:bg-gray-700">
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                      {order.order_id.slice(0, 8)}...
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-white">
                      {order.symbol}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        order.side === 'Buy' 
                          ? 'bg-green-900 text-green-200' 
                          : 'bg-red-900 text-red-200'
                      }`}>
                        {order.side}
                      </span>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                      {order.order_type}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                      {order.quantity.toFixed(6)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                      {formatPrice(order.price)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm">
                      <div className="flex items-center">
                        {getStatusIcon(order.status)}
                        <span className={`ml-2 ${getStatusColor(order.status)}`}>
                          {order.status}
                        </span>
                      </div>
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                      {order.filled_quantity.toFixed(6)}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                      {order.average_price > 0 ? formatPrice(order.average_price) : '-'}
                    </td>
                    <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                      {formatTime(order.created_time)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Summary Stats */}
      {orders.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Order Summary</h3>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Total Orders</span>
                <span className="text-white font-medium">{orders.length}</span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Filled Orders</span>
                <span className="text-green-400 font-medium">
                  {orders.filter(o => o.status.toLowerCase() === 'filled').length}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Open Orders</span>
                <span className="text-yellow-400 font-medium">
                  {orders.filter(o => ['pending', 'new', 'partially_filled'].includes(o.status.toLowerCase())).length}
                </span>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Trading Activity</h3>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Buy Orders</span>
                <span className="text-green-400 font-medium">
                  {orders.filter(o => o.side === 'Buy').length}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Sell Orders</span>
                <span className="text-red-400 font-medium">
                  {orders.filter(o => o.side === 'Sell').length}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Market Orders</span>
                <span className="text-blue-400 font-medium">
                  {orders.filter(o => o.order_type === 'Market').length}
                </span>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Volume</h3>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Total Volume</span>
                <span className="text-white font-medium">
                  {orders.reduce((sum, order) => sum + order.quantity, 0).toFixed(2)}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Filled Volume</span>
                <span className="text-green-400 font-medium">
                  {orders.reduce((sum, order) => sum + order.filled_quantity, 0).toFixed(2)}
                </span>
              </div>
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-6">
            <h3 className="text-lg font-semibold text-white mb-4">Recent Activity</h3>
            <div className="space-y-2">
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Last 24h</span>
                <span className="text-white font-medium">
                  {orders.filter(o => {
                    const orderTime = new Date(o.created_time);
                    const dayAgo = new Date(Date.now() - 24 * 60 * 60 * 1000);
                    return orderTime > dayAgo;
                  }).length}
                </span>
              </div>
              <div className="flex justify-between items-center">
                <span className="text-gray-300">Last 7d</span>
                <span className="text-white font-medium">
                  {orders.filter(o => {
                    const orderTime = new Date(o.created_time);
                    const weekAgo = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
                    return orderTime > weekAgo;
                  }).length}
                </span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
