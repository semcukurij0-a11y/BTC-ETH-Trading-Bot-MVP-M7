import React, { useState, useEffect } from 'react';
import { Activity, AlertCircle, CheckCircle, Clock, Server } from 'lucide-react';
import { HealthService, HealthStatus } from '../services/HealthService';
import { HeartbeatIndicator } from './HeartbeatIndicator';

interface SystemStatusProps {
  isConnected: boolean;
  data?: any;
  healthService?: HealthService;
}

export const SystemStatus: React.FC<SystemStatusProps> = ({ 
  isConnected, 
  data, 
  healthService 
}) => {
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [showHealthDetails, setShowHealthDetails] = useState(false);

  useEffect(() => {
    if (healthService) {
      // Subscribe to health updates
      const unsubscribe = healthService.onHealthUpdate((status) => {
        setHealthStatus(status);
      });

      // Load initial health status
      const loadHealthStatus = async () => {
        try {
          const status = await healthService.checkHealth();
          setHealthStatus(status);
        } catch (error) {
          console.error('Failed to load health status:', error);
        }
      };

      loadHealthStatus();

      return unsubscribe;
    }
  }, [healthService]);

  const getStatusColor = () => {
    if (!isConnected) return 'text-red-400';
    if (data?.mode === 'PAPER') return 'text-blue-400';
    return 'text-green-400';
  };

  const getStatusIcon = () => {
    if (!isConnected) return <AlertCircle className="h-5 w-5" />;
    if (data?.mode === 'PAPER') return <Clock className="h-5 w-5" />;
    return <CheckCircle className="h-5 w-5" />;
  };

  const getStatusText = () => {
    if (!isConnected) return 'Disconnected';
    if (data?.mode === 'PAPER') return 'Paper Trading';
    return 'Live Trading';
  };

  const getHealthStatusColor = () => {
    if (!healthStatus) return 'text-gray-400';
    if (healthStatus.status === 'healthy') return 'text-green-400';
    if (healthStatus.status === 'degraded') return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="flex items-center space-x-4">
      {/* Trading Status */}
      <div className={`flex items-center space-x-2 ${getStatusColor()}`}>
        {getStatusIcon()}
        <span className="font-medium">{getStatusText()}</span>
      </div>
      
      {/* Health Status */}
      {healthService && (
        <div className="flex items-center space-x-2">
          <button
            onClick={() => setShowHealthDetails(!showHealthDetails)}
            className="flex items-center space-x-2 text-sm hover:text-white transition-colors"
            title="Click to toggle health details"
          >
            <Server className="h-4 w-4" />
            <span className={getHealthStatusColor()}>
              Health: {healthStatus?.status || 'Unknown'}
            </span>
          </button>
        </div>
      )}

      {/* Heartbeat Indicator */}
      {healthService && (
        <HeartbeatIndicator 
          healthService={healthService} 
          showDetails={true}
        />
      )}
      
      {/* Legacy Data Display */}
      {data && (
        <div className="flex items-center space-x-4 text-sm text-gray-400">
          <div>
            <span>Latency: </span>
            <span className={data.latency < 100 ? 'text-green-400' : 'text-yellow-400'}>
              {data.latency}ms
            </span>
          </div>
          <div>
            <span>Uptime: </span>
            <span className="text-white">{data.uptime || '00:00:00'}</span>
          </div>
        </div>
      )}

      {/* Health Details Toggle */}
      {healthService && showHealthDetails && healthStatus && (
        <div className="absolute top-full right-0 mt-2 bg-gray-800 border border-gray-700 rounded-lg p-3 shadow-lg z-50 min-w-64">
          <div className="space-y-2 text-sm">
            <div className="flex justify-between">
              <span className="text-gray-400">System Status:</span>
              <span className={getHealthStatusColor()}>
                {healthStatus.status.toUpperCase()}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Last Heartbeat:</span>
              <span className="text-white">
                {new Date(healthStatus.last_heartbeat).toLocaleTimeString()}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Uptime:</span>
              <span className="text-white">
                {Math.floor(healthStatus.uptime_seconds / 3600)}h {Math.floor((healthStatus.uptime_seconds % 3600) / 60)}m
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Disk Usage:</span>
              <span className={healthStatus.system.disk_usage.usage_percent > 90 ? 'text-red-400' : 'text-white'}>
                {healthStatus.system.disk_usage.usage_percent.toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Data Directory:</span>
              <span className={healthStatus.system.data_directory_exists ? 'text-green-400' : 'text-red-400'}>
                {healthStatus.system.data_directory_exists ? 'OK' : 'Missing'}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Database:</span>
              <span className={healthStatus.system.database_file_exists ? 'text-green-400' : 'text-red-400'}>
                {healthStatus.system.database_file_exists ? 'OK' : 'Missing'}
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};