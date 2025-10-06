/**
 * Health Monitoring Demo
 * 
 * This file demonstrates how to use the health monitoring system
 * and provides example configurations for different scenarios.
 */

import { HealthService } from '../services/HealthService';

// Example: Basic health monitoring setup
export const setupBasicHealthMonitoring = () => {
  const healthService = new HealthService('http://localhost:8000');
  
  // Start monitoring with 5-second intervals
  healthService.startMonitoring(5000);
  
  // Subscribe to health updates
  const unsubscribe = healthService.onHealthUpdate((status) => {
    console.log('Health Status Update:', {
      status: status.status,
      uptime: Math.floor(status.uptime_seconds / 3600) + 'h',
      diskUsage: status.system.disk_usage.usage_percent + '%',
      lastHeartbeat: new Date(status.last_heartbeat).toLocaleString()
    });
  });
  
  return { healthService, unsubscribe };
};

// Example: Advanced health monitoring with error handling
export const setupAdvancedHealthMonitoring = () => {
  const healthService = new HealthService('http://localhost:8000');
  
  // Start monitoring
  healthService.startMonitoring(3000);
  
  // Subscribe to health updates with error handling
  const unsubscribe = healthService.onHealthUpdate((status) => {
    const summary = healthService.getHealthSummary(status);
    
    if (!summary.isHealthy) {
      console.warn('⚠️ System Health Alert:', {
        status: summary.statusText,
        diskUsage: summary.diskUsage + '%',
        lastHeartbeat: summary.lastHeartbeat
      });
      
      // You could trigger alerts, notifications, or automatic responses here
      if (summary.diskUsage > 90) {
        console.error('🚨 Critical: Disk usage above 90%');
      }
    } else {
      console.log('✅ System Healthy:', {
        uptime: summary.uptime,
        diskUsage: summary.diskUsage + '%'
      });
    }
  });
  
  // Test connectivity
  const testConnectivity = async () => {
    const isConnected = await healthService.testConnectivity();
    console.log('Backend Connectivity:', isConnected ? '✅ Connected' : '❌ Disconnected');
  };
  
  testConnectivity();
  
  return { healthService, unsubscribe };
};

// Example: Health monitoring with custom intervals and callbacks
export const setupCustomHealthMonitoring = (options: {
  baseUrl?: string;
  intervalMs?: number;
  onHealthy?: (status: any) => void;
  onUnhealthy?: (status: any) => void;
  onError?: (error: Error) => void;
}) => {
  const {
    baseUrl = 'http://localhost:8000',
    intervalMs = 5000,
    onHealthy,
    onUnhealthy,
    onError
  } = options;
  
  const healthService = new HealthService(baseUrl);
  
  // Start monitoring
  healthService.startMonitoring(intervalMs);
  
  // Subscribe to health updates with custom callbacks
  const unsubscribe = healthService.onHealthUpdate((status) => {
    try {
      const summary = healthService.getHealthSummary(status);
      
      if (summary.isHealthy) {
        onHealthy?.(status);
      } else {
        onUnhealthy?.(status);
      }
    } catch (error) {
      onError?.(error as Error);
    }
  });
  
  return { healthService, unsubscribe };
};

// Example: Health monitoring for different environments
export const getHealthServiceForEnvironment = (environment: 'development' | 'staging' | 'production') => {
  const configs = {
    development: {
      baseUrl: 'http://localhost:8000',
      intervalMs: 5000,
      timeout: 5000
    },
    staging: {
      baseUrl: 'https://staging-api.example.com',
      intervalMs: 10000,
      timeout: 10000
    },
    production: {
      baseUrl: 'https://api.example.com',
      intervalMs: 30000,
      timeout: 15000
    }
  };
  
  const config = configs[environment];
  const healthService = new HealthService(config.baseUrl);
  
  return { healthService, config };
};

// Example: Health monitoring with React hooks
export const useHealthMonitoring = (baseUrl?: string, intervalMs?: number) => {
  const healthService = new HealthService(baseUrl);
  
  // This would typically be used in a React component
  // const [healthStatus, setHealthStatus] = useState(null);
  // const [isMonitoring, setIsMonitoring] = useState(false);
  
  const startMonitoring = () => {
    healthService.startMonitoring(intervalMs || 5000);
    // setIsMonitoring(true);
  };
  
  const stopMonitoring = () => {
    healthService.stopMonitoring();
    // setIsMonitoring(false);
  };
  
  const subscribeToUpdates = (callback: (status: any) => void) => {
    return healthService.onHealthUpdate(callback);
  };
  
  return {
    healthService,
    startMonitoring,
    stopMonitoring,
    subscribeToUpdates
  };
};

// Example: Health monitoring with WebSocket fallback
export const setupHealthMonitoringWithFallback = () => {
  const healthService = new HealthService('http://localhost:8000');
  
  // Start HTTP polling
  healthService.startMonitoring(5000);
  
  // You could also set up WebSocket connection for real-time updates
  // const ws = new WebSocket('ws://localhost:8000/health/ws');
  // ws.onmessage = (event) => {
  //   const healthStatus = JSON.parse(event.data);
  //   // Handle real-time health updates
  // };
  
  return { healthService };
};

// Example usage in a React component:
/*
import { useEffect, useState } from 'react';
import { HealthService } from '../services/HealthService';

const MyComponent = () => {
  const [healthStatus, setHealthStatus] = useState(null);
  const [isMonitoring, setIsMonitoring] = useState(false);
  
  useEffect(() => {
    const healthService = new HealthService();
    
    // Start monitoring
    healthService.startMonitoring(5000);
    setIsMonitoring(true);
    
    // Subscribe to updates
    const unsubscribe = healthService.onHealthUpdate((status) => {
      setHealthStatus(status);
    });
    
    return () => {
      unsubscribe();
      healthService.stopMonitoring();
      setIsMonitoring(false);
    };
  }, []);
  
  return (
    <div>
      {isMonitoring ? (
        <div>Health monitoring active</div>
      ) : (
        <div>Health monitoring stopped</div>
      )}
      {healthStatus && (
        <div>
          Status: {healthStatus.status}
          Uptime: {Math.floor(healthStatus.uptime_seconds / 3600)}h
        </div>
      )}
    </div>
  );
};
*/
