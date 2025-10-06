import React, { useState, useEffect } from 'react';
import { Activity, AlertTriangle, CheckCircle, Clock } from 'lucide-react';
import { HealthService, HealthStatus } from '../services/HealthService';

interface HeartbeatIndicatorProps {
  healthService: HealthService;
  className?: string;
  showDetails?: boolean;
}

export const HeartbeatIndicator: React.FC<HeartbeatIndicatorProps> = ({
  healthService,
  className = '',
  showDetails = false
}) => {
  const [healthStatus, setHealthStatus] = useState<HealthStatus | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [lastHeartbeat, setLastHeartbeat] = useState<Date | null>(null);
  const [heartbeatAge, setHeartbeatAge] = useState<number>(0);

  useEffect(() => {
    // Subscribe to health updates
    const unsubscribe = healthService.onHealthUpdate((status) => {
      setHealthStatus(status);
      setIsConnected(true);
      setLastHeartbeat(new Date(status.last_heartbeat));
    });

    // Initial load
    const loadInitialStatus = async () => {
      try {
        const status = await healthService.checkHealth();
        setHealthStatus(status);
        setIsConnected(true);
        setLastHeartbeat(new Date(status.last_heartbeat));
      } catch (error) {
        setIsConnected(false);
        console.error('Failed to load initial health status:', error);
      }
    };

    loadInitialStatus();

    return unsubscribe;
  }, [healthService]);

  // Update heartbeat age every second
  useEffect(() => {
    const interval = setInterval(() => {
      if (lastHeartbeat) {
        const age = Math.floor((Date.now() - lastHeartbeat.getTime()) / 1000);
        setHeartbeatAge(age);
      }
    }, 1000);

    return () => clearInterval(interval);
  }, [lastHeartbeat]);

  const getHeartbeatStatus = () => {
    if (!isConnected || !healthStatus) {
      return {
        status: 'disconnected',
        color: 'text-red-400',
        bgColor: 'bg-red-900/30',
        borderColor: 'border-red-500/30',
        icon: AlertTriangle,
        text: 'Disconnected'
      };
    }

    const age = heartbeatAge;
    if (age < 30) {
      return {
        status: 'healthy',
        color: 'text-green-400',
        bgColor: 'bg-green-900/30',
        borderColor: 'border-green-500/30',
        icon: CheckCircle,
        text: 'Healthy'
      };
    } else if (age < 120) {
      return {
        status: 'warning',
        color: 'text-yellow-400',
        bgColor: 'bg-yellow-900/30',
        borderColor: 'border-yellow-500/30',
        icon: Clock,
        text: 'Delayed'
      };
    } else {
      return {
        status: 'critical',
        color: 'text-red-400',
        bgColor: 'bg-red-900/30',
        borderColor: 'border-red-500/30',
        icon: AlertTriangle,
        text: 'Critical'
      };
    }
  };

  const formatHeartbeatAge = (seconds: number): string => {
    if (seconds < 60) {
      return `${seconds}s ago`;
    } else if (seconds < 3600) {
      const minutes = Math.floor(seconds / 60);
      return `${minutes}m ago`;
    } else {
      const hours = Math.floor(seconds / 3600);
      return `${hours}h ago`;
    }
  };

  const heartbeatStatus = getHeartbeatStatus();
  const Icon = heartbeatStatus.icon;

  return (
    <div className={`flex items-center space-x-3 ${className}`}>
      {/* Heartbeat Pulse Animation */}
      <div className="relative">
        <div className={`w-3 h-3 rounded-full ${
          heartbeatStatus.status === 'healthy' ? 'bg-green-400' : 
          heartbeatStatus.status === 'warning' ? 'bg-yellow-400' : 'bg-red-400'
        }`}>
          {heartbeatStatus.status === 'healthy' && (
            <div className="absolute inset-0 w-3 h-3 rounded-full bg-green-400 animate-ping opacity-75" />
          )}
        </div>
      </div>

      {/* Status Icon and Text */}
      <div className={`flex items-center space-x-2 px-3 py-1 rounded-full text-sm font-medium ${heartbeatStatus.bgColor} ${heartbeatStatus.borderColor} border`}>
        <Icon className={`h-4 w-4 ${heartbeatStatus.color}`} />
        <span className={heartbeatStatus.color}>{heartbeatStatus.text}</span>
      </div>

      {/* Heartbeat Details */}
      {showDetails && (
        <div className="flex items-center space-x-4 text-sm text-gray-400">
          {lastHeartbeat && (
            <div className="flex items-center space-x-1">
              <Activity className="h-4 w-4" />
              <span>{formatHeartbeatAge(heartbeatAge)}</span>
            </div>
          )}
          
          {healthStatus && (
            <div className="flex items-center space-x-1">
              <Clock className="h-4 w-4" />
              <span>
                Uptime: {Math.floor(healthStatus.uptime_seconds / 3600)}h {Math.floor((healthStatus.uptime_seconds % 3600) / 60)}m
              </span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
