import React, { useState, useEffect, useCallback } from 'react';
import { 
  Activity, 
  AlertTriangle, 
  CheckCircle, 
  Clock, 
  Database, 
  HardDrive, 
  Server, 
  TrendingUp,
  RefreshCw,
  AlertCircle,
  Info
} from 'lucide-react';
import { HealthService, HealthStatus, DetailedHealthStatus } from '../services/HealthService';

interface HealthMonitorProps {
  healthService: HealthService;
  className?: string;
  showDetailed?: boolean;
  autoRefresh?: boolean;
  refreshInterval?: number;
}

export const HealthMonitor: React.FC<HealthMonitorProps> = ({
  healthService,
  className = '',
  showDetailed = false,
  autoRefresh = true,
  refreshInterval = 5000
}) => {
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [detailedStatus, setDetailedStatus] = useState<DetailedHealthStatus | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [lastUpdate, setLastUpdate] = useState<Date | null>(null);
  const [error, setError] = useState<string | null>(null);

  const loadHealthStatus = useCallback(async () => {
    try {
      setIsLoading(true);
      setError(null);
      
      const status = await healthService.checkHealth();
      setHealthStatus(status);
      setLastUpdate(new Date());

      if (showDetailed) {
        try {
          const detailed = await healthService.getDetailedHealth();
          setDetailedStatus(detailed);
        } catch (err) {
          console.warn('Failed to load detailed health status:', err);
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load health status');
      console.error('Health status load failed:', err);
    } finally {
      setIsLoading(false);
    }
  }, [healthService, showDetailed]);

  const handleRefresh = () => {
    loadHealthStatus();
  };

  useEffect(() => {
    // Initial load
    loadHealthStatus();

    // Set up auto-refresh
    if (autoRefresh) {
      const interval = setInterval(loadHealthStatus, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [loadHealthStatus, autoRefresh, refreshInterval]);

  // Subscribe to health updates
  useEffect(() => {
    const unsubscribe = healthService.onHealthUpdate((status) => {
      setHealthStatus(status);
      setLastUpdate(new Date());
      setError(null);
    });

    return unsubscribe;
  }, [healthService]);

  if (!healthStatus) {
    return (
      <div className={`bg-gray-800 rounded-lg p-6 ${className}`}>
        <div className="flex items-center justify-center space-x-2">
          <RefreshCw className="h-5 w-5 animate-spin text-blue-400" />
          <span className="text-gray-400">Loading health status...</span>
        </div>
      </div>
    );
  }

  // Add safety check for health status structure
  if (!healthStatus.system || !healthStatus.system.disk_usage) {
    console.warn('Incomplete health status received:', healthStatus);
    return (
      <div className={`bg-gray-800 rounded-lg p-6 ${className}`}>
        <div className="flex items-center justify-center space-x-2">
          <AlertTriangle className="h-5 w-5 text-yellow-400" />
          <span className="text-yellow-400">Incomplete health data received</span>
        </div>
      </div>
    );
  }

  const summary = healthService.getHealthSummary(healthStatus);

  return (
    <div className={`bg-gray-800 rounded-lg p-6 ${className}`}>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <Server className="h-6 w-6 text-blue-400" />
          <h3 className="text-lg font-semibold">System Health</h3>
        </div>
        <div className="flex items-center space-x-2">
          <button
            onClick={handleRefresh}
            disabled={isLoading}
            className="p-2 text-gray-400 hover:text-white transition-colors disabled:opacity-50"
            title="Refresh health status"
          >
            <RefreshCw className={`h-4 w-4 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
          {lastUpdate && (
            <span className="text-xs text-gray-500">
              {lastUpdate.toLocaleTimeString()}
            </span>
          )}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="mb-4 p-3 bg-red-900/20 border border-red-500/30 rounded-lg flex items-center space-x-2">
          <AlertCircle className="h-4 w-4 text-red-400" />
          <span className="text-red-400 text-sm">{error}</span>
        </div>
      )}

      {/* Main Status */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
        {/* Overall Status */}
        <div className="bg-gray-700 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            {summary.isHealthy ? (
              <CheckCircle className="h-5 w-5 text-green-400" />
            ) : (
              <AlertTriangle className="h-5 w-5 text-red-400" />
            )}
            <span className="font-medium">Status</span>
          </div>
          <p className={`text-lg font-bold ${
            summary.isHealthy ? 'text-green-400' : 'text-red-400'
          }`}>
            {summary.statusText}
          </p>
        </div>

        {/* Uptime */}
        <div className="bg-gray-700 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Clock className="h-5 w-5 text-blue-400" />
            <span className="font-medium">Uptime</span>
          </div>
          <p className="text-lg font-bold text-white">{summary.uptime}</p>
        </div>

        {/* Last Heartbeat */}
        <div className="bg-gray-700 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <Activity className="h-5 w-5 text-purple-400" />
            <span className="font-medium">Last Heartbeat</span>
          </div>
          <p className="text-sm text-gray-300">{summary.lastHeartbeat}</p>
        </div>

        {/* Disk Usage */}
        <div className="bg-gray-700 rounded-lg p-4">
          <div className="flex items-center space-x-2 mb-2">
            <HardDrive className="h-5 w-5 text-orange-400" />
            <span className="font-medium">Disk Usage</span>
          </div>
          <div className="flex items-center space-x-2">
            <div className="flex-1 bg-gray-600 rounded-full h-2">
              <div 
                className={`h-2 rounded-full ${
                  summary.diskUsage > 90 ? 'bg-red-400' : 
                  summary.diskUsage > 75 ? 'bg-yellow-400' : 'bg-green-400'
                }`}
                style={{ width: `${summary.diskUsage}%` }}
              />
            </div>
            <span className="text-sm font-medium text-white">{summary.diskUsage.toFixed(1)}%</span>
          </div>
        </div>
      </div>

      {/* System Details */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* System Information */}
        <div className="space-y-4">
          <h4 className="text-md font-semibold text-gray-300 flex items-center space-x-2">
            <Server className="h-4 w-4" />
            <span>System Information</span>
          </h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Data Directory:</span>
              <span className={healthStatus.system.data_directory_exists ? 'text-green-400' : 'text-red-400'}>
                {healthStatus.system.data_directory_exists ? 'Available' : 'Missing'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Database:</span>
              <span className={healthStatus.system.database_file_exists ? 'text-green-400' : 'text-red-400'}>
                {healthStatus.system.database_file_exists ? 'Connected' : 'Missing'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Python Version:</span>
              <span className="text-white text-xs">
                {healthStatus.system.python_version.split(' ')[0]}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Working Directory:</span>
              <span className="text-white text-xs truncate ml-2">
                {healthStatus.system.working_directory.split('/').pop() || 'Unknown'}
              </span>
            </div>
          </div>
        </div>

        {/* Disk Information */}
        <div className="space-y-4">
          <h4 className="text-md font-semibold text-gray-300 flex items-center space-x-2">
            <HardDrive className="h-4 w-4" />
            <span>Disk Information</span>
          </h4>
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">Total Space:</span>
              <span className="text-white">{healthStatus.system.disk_usage.total_gb.toFixed(1)} GB</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Used Space:</span>
              <span className="text-white">{healthStatus.system.disk_usage.used_gb.toFixed(1)} GB</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Free Space:</span>
              <span className="text-white">{healthStatus.system.disk_usage.free_gb.toFixed(1)} GB</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Usage:</span>
              <span className={`font-medium ${
                healthStatus.system.disk_usage.usage_percent > 90 ? 'text-red-400' : 
                healthStatus.system.disk_usage.usage_percent > 75 ? 'text-yellow-400' : 'text-green-400'
              }`}>
                {healthStatus.system.disk_usage.usage_percent.toFixed(1)}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Detailed Information */}
      {showDetailed && detailedStatus && (
        <div className="mt-6 space-y-4">
          <h4 className="text-md font-semibold text-gray-300 flex items-center space-x-2">
            <Database className="h-4 w-4" />
            <span>Data Status</span>
          </h4>
          
          {detailedStatus.data && (
            <div className="bg-gray-700 rounded-lg p-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
                <div>
                  <span className="text-gray-400">Total Records:</span>
                  <span className="ml-2 text-white font-medium">
                    {detailedStatus.data.total_records.toLocaleString()}
                  </span>
                </div>
                <div>
                  <span className="text-gray-400">Parquet Files:</span>
                  <span className="ml-2 text-white font-medium">
                    {detailedStatus.data.parquet_files.length}
                  </span>
                </div>
                <div>
                  <span className="text-gray-400">Last Update:</span>
                  <span className="ml-2 text-white">
                    {detailedStatus.data.last_update ? 
                      new Date(detailedStatus.data.last_update).toLocaleString() : 
                      'Never'
                    }
                  </span>
                </div>
              </div>

              {/* Recent Files */}
              {detailedStatus.data.parquet_files.length > 0 && (
                <div className="mt-4">
                  <h5 className="text-sm font-medium text-gray-300 mb-2">Recent Files</h5>
                  <div className="space-y-1 max-h-32 overflow-y-auto">
                    {detailedStatus.data.parquet_files.slice(0, 5).map((file, index) => (
                      <div key={index} className="flex justify-between items-center text-xs">
                        <span className="text-gray-400 truncate">{file.filename}</span>
                        <div className="flex space-x-2 text-gray-300">
                          <span>{file.size_mb.toFixed(1)}MB</span>
                          <span>{file.records.toLocaleString()} records</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}

          {/* Environment Information */}
          {detailedStatus.environment && (
            <div className="bg-gray-700 rounded-lg p-4">
              <h5 className="text-sm font-medium text-gray-300 mb-2">Environment</h5>
              <div className="space-y-1 text-xs">
                <div className="flex justify-between">
                  <span className="text-gray-400">Python Path:</span>
                  <span className="text-white truncate ml-2">
                    {detailedStatus.environment.python_path}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-400">Log Level:</span>
                  <span className="text-white">
                    {detailedStatus.environment.environment_variables.LOG_LEVEL}
                  </span>
                </div>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Health Status Indicator */}
      <div className="mt-4 flex items-center justify-center">
        <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm font-medium ${
          summary.isHealthy 
            ? 'bg-green-900/30 text-green-400 border border-green-500/30' 
            : 'bg-red-900/30 text-red-400 border border-red-500/30'
        }`}>
          <div className={`w-2 h-2 rounded-full ${
            summary.isHealthy ? 'bg-green-400' : 'bg-red-400'
          }`} />
          <span>
            {summary.isHealthy ? 'All Systems Operational' : 'System Issues Detected'}
          </span>
        </div>
      </div>
    </div>
  );
};
