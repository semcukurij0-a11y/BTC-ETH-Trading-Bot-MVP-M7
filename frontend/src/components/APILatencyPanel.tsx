import React, { useState, useEffect } from 'react';
import { Clock, Wifi, WifiOff, AlertTriangle, CheckCircle, XCircle, RefreshCw } from 'lucide-react';

interface LatencyData {
  endpoint: string;
  latency: number;
  status: 'success' | 'error' | 'timeout';
  timestamp: string;
  response_size?: number;
}

interface APILatencyPanelProps {
  service: any;
}

export const APILatencyPanel: React.FC<APILatencyPanelProps> = ({ service }) => {
  const [latencyData, setLatencyData] = useState<LatencyData[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<string | null>(null);
  const [isMonitoring, setIsMonitoring] = useState(false);

  const endpoints = [
    { name: 'Health Check', path: '/health', critical: true },
    { name: 'Trading Stats', path: '/trading/stats', critical: true },
    { name: 'Positions', path: '/trading/positions', critical: true },
    { name: 'Account Info', path: '/trading/account', critical: true },
    { name: 'Market Data', path: '/trading/market', critical: false },
    { name: 'Signals', path: '/trading/signals', critical: false },
    { name: 'Orders', path: '/trading/orders', critical: true },
    { name: 'Dashboard', path: '/trading/dashboard', critical: false }
  ];

  const testEndpoint = async (endpoint: { name: string; path: string; critical: boolean }): Promise<LatencyData> => {
    const startTime = Date.now();
    try {
      const response = await fetch(`http://localhost:8000${endpoint.path}`, {
        method: 'GET',
        headers: {
          'Authorization': `Bearer ${service.authToken}`,
          'Content-Type': 'application/json'
        },
        signal: AbortSignal.timeout(10000) // 10 second timeout
      });

      const endTime = Date.now();
      const latency = endTime - startTime;

      return {
        endpoint: endpoint.name,
        latency,
        status: response.ok ? 'success' : 'error',
        timestamp: new Date().toISOString(),
        response_size: response.headers.get('content-length') ? parseInt(response.headers.get('content-length')!) : undefined
      };
    } catch (err) {
      const endTime = Date.now();
      const latency = endTime - startTime;
      
      return {
        endpoint: endpoint.name,
        latency,
        status: 'timeout',
        timestamp: new Date().toISOString()
      };
    }
  };

  const testAllEndpoints = async () => {
    setLoading(true);
    try {
      const results = await Promise.all(
        endpoints.map(endpoint => testEndpoint(endpoint))
      );
      
      setLatencyData(results);
      setLastUpdate(new Date().toISOString());
      setError(null);
    } catch (err) {
      console.error('Error testing endpoints:', err);
      setError('Error testing API endpoints');
    } finally {
      setLoading(false);
    }
  };

  const startMonitoring = () => {
    setIsMonitoring(true);
    const interval = setInterval(testAllEndpoints, 5000); // Test every 5 seconds
    return () => {
      clearInterval(interval);
      setIsMonitoring(false);
    };
  };

  const stopMonitoring = () => {
    setIsMonitoring(false);
  };

  useEffect(() => {
    testAllEndpoints();
  }, []);

  const getLatencyColor = (latency: number, critical: boolean) => {
    if (critical) {
      if (latency < 1000) return 'text-green-400';
      if (latency < 3000) return 'text-yellow-400';
      return 'text-red-400';
    } else {
      if (latency < 2000) return 'text-green-400';
      if (latency < 5000) return 'text-yellow-400';
      return 'text-red-400';
    }
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="h-4 w-4 text-green-400" />;
      case 'error':
        return <XCircle className="h-4 w-4 text-red-400" />;
      case 'timeout':
        return <WifiOff className="h-4 w-4 text-red-400" />;
      default:
        return <AlertTriangle className="h-4 w-4 text-yellow-400" />;
    }
  };

  const getLatencyLevel = (latency: number, critical: boolean) => {
    if (critical) {
      if (latency < 1000) return 'Excellent';
      if (latency < 3000) return 'Good';
      if (latency < 5000) return 'Fair';
      return 'Poor';
    } else {
      if (latency < 2000) return 'Excellent';
      if (latency < 5000) return 'Good';
      if (latency < 10000) return 'Fair';
      return 'Poor';
    }
  };

  const formatLatency = (latency: number) => {
    if (latency < 1000) return `${latency}ms`;
    return `${(latency / 1000).toFixed(2)}s`;
  };

  const getAverageLatency = () => {
    const successfulTests = latencyData.filter(d => d.status === 'success');
    if (successfulTests.length === 0) return 0;
    return successfulTests.reduce((sum, d) => sum + d.latency, 0) / successfulTests.length;
  };

  const getSuccessRate = () => {
    if (latencyData.length === 0) return 0;
    const successfulTests = latencyData.filter(d => d.status === 'success');
    return (successfulTests.length / latencyData.length) * 100;
  };

  const criticalEndpoints = latencyData.filter(d => 
    endpoints.find(e => e.name === d.endpoint)?.critical
  );

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">API Latency Monitor</h2>
        <div className="flex items-center space-x-4">
          <button
            onClick={isMonitoring ? stopMonitoring : startMonitoring}
            className={`flex items-center space-x-2 px-4 py-2 rounded-lg font-medium ${
              isMonitoring 
                ? 'bg-red-600 text-white hover:bg-red-700' 
                : 'bg-green-600 text-white hover:bg-green-700'
            }`}
          >
            {isMonitoring ? <WifiOff className="h-4 w-4" /> : <Wifi className="h-4 w-4" />}
            <span>{isMonitoring ? 'Stop' : 'Start'} Monitoring</span>
          </button>
          <button
            onClick={testAllEndpoints}
            disabled={loading}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:opacity-50"
          >
            <RefreshCw className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
            <span>Test Now</span>
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
        <div className="flex items-center space-x-4">
          <span>Average Latency: {formatLatency(getAverageLatency())}</span>
          <span>Success Rate: {getSuccessRate().toFixed(1)}%</span>
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-900/50 border border-red-500 text-red-200 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      {/* Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gray-800 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <Clock className="h-5 w-5 text-blue-400" />
            <h3 className="text-sm font-medium text-gray-300">Average Latency</h3>
          </div>
          <p className="text-2xl font-bold text-white mt-2">
            {formatLatency(getAverageLatency())}
          </p>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <CheckCircle className="h-5 w-5 text-green-400" />
            <h3 className="text-sm font-medium text-gray-300">Success Rate</h3>
          </div>
          <p className="text-2xl font-bold text-white mt-2">
            {getSuccessRate().toFixed(1)}%
          </p>
        </div>

        <div className="bg-gray-800 p-4 rounded-lg">
          <div className="flex items-center space-x-2">
            <AlertTriangle className="h-5 w-5 text-orange-400" />
            <h3 className="text-sm font-medium text-gray-300">Critical Endpoints</h3>
          </div>
          <p className="text-2xl font-bold text-white mt-2">
            {criticalEndpoints.filter(d => d.status === 'success').length}/{criticalEndpoints.length}
          </p>
        </div>
      </div>

      {/* Endpoints Table */}
      <div className="bg-gray-800 rounded-lg overflow-hidden">
        <div className="px-6 py-4 border-b border-gray-700">
          <h3 className="text-lg font-semibold text-white">Endpoint Performance</h3>
        </div>
        <div className="overflow-x-auto">
          <table className="min-w-full">
            <thead className="bg-gray-700">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Endpoint
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Latency
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Level
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Priority
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-300 uppercase tracking-wider">
                  Last Test
                </th>
              </tr>
            </thead>
            <tbody className="bg-gray-800 divide-y divide-gray-700">
              {latencyData.length === 0 ? (
                <tr>
                  <td colSpan={6} className="px-6 py-8 text-center text-gray-400">
                    {loading ? 'Testing endpoints...' : 'No data available'}
                  </td>
                </tr>
              ) : (
                latencyData.map((data, index) => {
                  const endpoint = endpoints.find(e => e.name === data.endpoint);
                  const isCritical = endpoint?.critical || false;
                  
                  return (
                    <tr key={index} className="hover:bg-gray-700">
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-white">
                        {data.endpoint}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        <div className="flex items-center space-x-2">
                          {getStatusIcon(data.status)}
                          <span className={data.status === 'success' ? 'text-green-400' : 'text-red-400'}>
                            {data.status}
                          </span>
                        </div>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        <span className={getLatencyColor(data.latency, isCritical)}>
                          {formatLatency(data.latency)}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                        {getLatencyLevel(data.latency, isCritical)}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm">
                        <span className={`px-2 py-1 rounded text-xs font-medium ${
                          isCritical 
                            ? 'bg-red-900 text-red-200' 
                            : 'bg-gray-700 text-gray-300'
                        }`}>
                          {isCritical ? 'Critical' : 'Normal'}
                        </span>
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-300">
                        {new Date(data.timestamp).toLocaleTimeString()}
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* Performance Chart Placeholder */}
      <div className="bg-gray-800 p-6 rounded-lg">
        <h3 className="text-lg font-semibold text-white mb-4">Performance Trends</h3>
        <div className="text-center text-gray-400 py-8">
          <Clock className="h-12 w-12 mx-auto mb-4 text-gray-600" />
          <p>Performance chart would be displayed here</p>
          <p className="text-sm">Showing latency trends over time</p>
        </div>
      </div>
    </div>
  );
};
