import React, { useState, useEffect, useRef } from 'react';
import { Clock, Wifi, AlertTriangle, CheckCircle, XCircle, RefreshCw } from 'lucide-react';

interface LagMeasurement {
  endpoint: string;
  timestamp: string;
  responseTime: number;
  status: 'success' | 'error' | 'timeout';
  dataSize?: number;
  error?: string;
}

interface DataLagMonitorProps {
  service: any;
  refreshInterval?: number;
}

export const DataLagMonitor: React.FC<DataLagMonitorProps> = ({ 
  service, 
  refreshInterval = 5000 
}) => {
  const [measurements, setMeasurements] = useState<LagMeasurement[]>([]);
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [stats, setStats] = useState({
    averageLatency: 0,
    maxLatency: 0,
    minLatency: 0,
    successRate: 0,
    totalRequests: 0,
    errorCount: 0
  });
  const [currentTest, setCurrentTest] = useState<string | null>(null);
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const startTimeRef = useRef<number>(0);

  // Endpoints to test
  const endpoints = [
    { name: 'Positions', url: '/trading/positions', method: 'GET' },
    { name: 'Orders', url: '/trading/orders', method: 'GET' },
    { name: 'Signals', url: '/trading/signals', method: 'GET' },
    { name: 'Stats', url: '/trading/stats', method: 'GET' },
    { name: 'Account', url: '/trading/account', method: 'GET' },
    { name: 'Health', url: '/health', method: 'GET' }
  ];

  // Measure latency for a single endpoint
  const measureEndpoint = async (endpoint: any): Promise<LagMeasurement> => {
    const startTime = performance.now();
    const timestamp = new Date().toISOString();
    
    try {
      setCurrentTest(endpoint.name);
      
      const response = await fetch(`${service.baseUrl || 'http://localhost:8000'}${endpoint.url}`, {
        method: endpoint.method,
        headers: {
          'Authorization': `Bearer ${service.authToken}`,
          'Content-Type': 'application/json',
        }
      });
      
      const endTime = performance.now();
      const responseTime = endTime - startTime;
      
      let dataSize = 0;
      if (response.ok) {
        const data = await response.json();
        dataSize = JSON.stringify(data).length;
      }
      
      return {
        endpoint: endpoint.name,
        timestamp,
        responseTime,
        status: response.ok ? 'success' : 'error',
        dataSize,
        error: response.ok ? undefined : `HTTP ${response.status}`
      };
      
    } catch (error) {
      const endTime = performance.now();
      const responseTime = endTime - startTime;
      
      return {
        endpoint: endpoint.name,
        timestamp,
        responseTime,
        status: 'error',
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    } finally {
      setCurrentTest(null);
    }
  };

  // Run latency test for all endpoints
  const runLatencyTest = async () => {
    const newMeasurements: LagMeasurement[] = [];
    
    for (const endpoint of endpoints) {
      const measurement = await measureEndpoint(endpoint);
      newMeasurements.push(measurement);
      
      // Update measurements state
      setMeasurements(prev => {
        const updated = [...prev, measurement];
        // Keep only last 50 measurements
        return updated.slice(-50);
      });
      
      // Small delay between requests
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    
    // Update stats
    updateStats(newMeasurements);
  };

  // Update statistics
  const updateStats = (newMeasurements: LagMeasurement[]) => {
    const allMeasurements = [...measurements, ...newMeasurements];
    const successfulMeasurements = allMeasurements.filter(m => m.status === 'success');
    
    if (successfulMeasurements.length > 0) {
      const latencies = successfulMeasurements.map(m => m.responseTime);
      const averageLatency = latencies.reduce((sum, lat) => sum + lat, 0) / latencies.length;
      const maxLatency = Math.max(...latencies);
      const minLatency = Math.min(...latencies);
      const successRate = (successfulMeasurements.length / allMeasurements.length) * 100;
      
      setStats({
        averageLatency,
        maxLatency,
        minLatency,
        successRate,
        totalRequests: allMeasurements.length,
        errorCount: allMeasurements.filter(m => m.status === 'error').length
      });
    }
  };

  // Start monitoring
  const startMonitoring = () => {
    if (isMonitoring) return;
    
    setIsMonitoring(true);
    setMeasurements([]);
    setStats({
      averageLatency: 0,
      maxLatency: 0,
      minLatency: 0,
      successRate: 0,
      totalRequests: 0,
      errorCount: 0
    });
    
    // Run initial test
    runLatencyTest();
    
    // Set up interval
    intervalRef.current = setInterval(runLatencyTest, refreshInterval);
  };

  // Stop monitoring
  const stopMonitoring = () => {
    setIsMonitoring(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  };

  // Manual test
  const runManualTest = async () => {
    if (isMonitoring) return;
    await runLatencyTest();
  };

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, []);

  // Get latency color based on response time
  const getLatencyColor = (responseTime: number) => {
    if (responseTime < 500) return 'text-green-400';
    if (responseTime < 1000) return 'text-yellow-400';
    if (responseTime < 2000) return 'text-orange-400';
    return 'text-red-400';
  };

  // Get status icon
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'success':
        return <CheckCircle className="h-4 w-4 text-green-400" />;
      case 'error':
        return <XCircle className="h-4 w-4 text-red-400" />;
      case 'timeout':
        return <AlertTriangle className="h-4 w-4 text-orange-400" />;
      default:
        return <Clock className="h-4 w-4 text-gray-400" />;
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Wifi className="h-8 w-8 text-blue-400" />
          <div>
            <h2 className="text-2xl font-bold text-white">Data Lag Monitor</h2>
            <p className="text-sm text-gray-400">Monitor API response times and data freshness</p>
          </div>
        </div>
        
        <div className="flex items-center space-x-3">
          {currentTest && (
            <div className="flex items-center space-x-2 text-blue-400">
              <RefreshCw className="h-4 w-4 animate-spin" />
              <span className="text-sm">Testing {currentTest}...</span>
            </div>
          )}
          
          <button
            onClick={isMonitoring ? stopMonitoring : startMonitoring}
            className={`px-4 py-2 rounded-lg font-medium ${
              isMonitoring 
                ? 'bg-red-600 hover:bg-red-700 text-white' 
                : 'bg-green-600 hover:bg-green-700 text-white'
            }`}
          >
            {isMonitoring ? 'Stop Monitoring' : 'Start Monitoring'}
          </button>
          
          <button
            onClick={runManualTest}
            disabled={isMonitoring}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg font-medium disabled:bg-gray-600"
          >
            Test Now
          </button>
        </div>
      </div>

      {/* Statistics */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-400">Average Latency</p>
              <p className="text-2xl font-bold text-white">
                {stats.averageLatency.toFixed(0)}ms
              </p>
            </div>
            <Clock className="h-8 w-8 text-blue-400" />
          </div>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-400">Success Rate</p>
              <p className="text-2xl font-bold text-white">
                {stats.successRate.toFixed(1)}%
              </p>
            </div>
            <CheckCircle className="h-8 w-8 text-green-400" />
          </div>
        </div>
        
        <div className="bg-gray-800 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-400">Total Requests</p>
              <p className="text-2xl font-bold text-white">
                {stats.totalRequests}
              </p>
            </div>
            <Wifi className="h-8 w-8 text-purple-400" />
          </div>
        </div>
      </div>

      {/* Latency Range */}
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-4">Latency Range</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="text-center">
            <p className="text-sm text-gray-400">Min Latency</p>
            <p className="text-xl font-bold text-green-400">
              {stats.minLatency.toFixed(0)}ms
            </p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-400">Max Latency</p>
            <p className="text-xl font-bold text-red-400">
              {stats.maxLatency.toFixed(0)}ms
            </p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-400">Error Count</p>
            <p className="text-xl font-bold text-orange-400">
              {stats.errorCount}
            </p>
          </div>
        </div>
      </div>

      {/* Recent Measurements */}
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-4">Recent Measurements</h3>
        
        {measurements.length === 0 ? (
          <div className="text-center py-8">
            <Clock className="h-12 w-12 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400">No measurements yet. Start monitoring to see data.</p>
          </div>
        ) : (
          <div className="space-y-2 max-h-96 overflow-y-auto">
            {measurements.slice(-20).reverse().map((measurement, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                <div className="flex items-center space-x-3">
                  {getStatusIcon(measurement.status)}
                  <div>
                    <p className="font-medium text-white">{measurement.endpoint}</p>
                    <p className="text-xs text-gray-400">
                      {new Date(measurement.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-4">
                  <div className="text-right">
                    <p className={`font-bold ${getLatencyColor(measurement.responseTime)}`}>
                      {measurement.responseTime.toFixed(0)}ms
                    </p>
                    {measurement.dataSize && (
                      <p className="text-xs text-gray-400">
                        {measurement.dataSize} bytes
                      </p>
                    )}
                  </div>
                  
                  {measurement.error && (
                    <div className="text-red-400 text-xs max-w-32 truncate">
                      {measurement.error}
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Endpoint Status */}
      <div className="bg-gray-800 rounded-lg p-4">
        <h3 className="text-lg font-semibold text-white mb-4">Endpoint Status</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {endpoints.map((endpoint, index) => {
            const recentMeasurements = measurements
              .filter(m => m.endpoint === endpoint.name)
              .slice(-5);
            
            const avgLatency = recentMeasurements.length > 0 
              ? recentMeasurements.reduce((sum, m) => sum + m.responseTime, 0) / recentMeasurements.length
              : 0;
            
            const successRate = recentMeasurements.length > 0
              ? (recentMeasurements.filter(m => m.status === 'success').length / recentMeasurements.length) * 100
              : 0;
            
            return (
              <div key={index} className="bg-gray-700 rounded-lg p-3">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-white">{endpoint.name}</h4>
                  <span className={`text-xs px-2 py-1 rounded ${
                    successRate >= 90 ? 'bg-green-900 text-green-300' :
                    successRate >= 70 ? 'bg-yellow-900 text-yellow-300' :
                    'bg-red-900 text-red-300'
                  }`}>
                    {successRate.toFixed(0)}%
                  </span>
                </div>
                
                <div className="text-sm text-gray-400">
                  <p>Avg: <span className={getLatencyColor(avgLatency)}>
                    {avgLatency.toFixed(0)}ms
                  </span></p>
                  <p>Tests: {recentMeasurements.length}</p>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};
