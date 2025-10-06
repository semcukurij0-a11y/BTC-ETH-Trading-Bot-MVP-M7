export interface HealthStatus {
  status: 'healthy' | 'unhealthy' | 'degraded';
  timestamp: string;
  uptime_seconds: number;
  last_heartbeat: string;
  system: {
    data_directory_exists: boolean;
    database_file_exists: boolean;
    disk_usage: {
      total_gb: number;
      used_gb: number;
      free_gb: number;
      usage_percent: number;
    };
    python_version: string;
    working_directory: string;
  };
}

export interface DetailedHealthStatus extends HealthStatus {
  data: {
    parquet_files: Array<{
      filename: string;
      size_mb: number;
      records: number;
      last_modified: string;
      date_range?: {
        start: string;
        end: string;
      };
    }>;
    total_records: number;
    last_update: string | null;
  };
  environment: {
    python_path: string;
    environment_variables: {
      PYTHONPATH: string;
      LOG_LEVEL: string;
    };
  };
}

export interface SystemMetrics {
  cpu_usage?: number;
  memory_usage?: number;
  disk_io?: number;
  network_io?: number;
}

export class HealthService {
  private baseUrl = 'http://localhost:8000';
  private healthCheckInterval: NodeJS.Timeout | null = null;
  private isMonitoring = false;
  private lastHealthStatus: HealthStatus | null = null;
  private healthCallbacks: Array<(status: HealthStatus) => void> = [];

  constructor(baseUrl?: string) {
    if (baseUrl) {
      this.baseUrl = baseUrl;
    }
  }

  /**
   * Start real-time health monitoring
   */
  startMonitoring(intervalMs: number = 5000): void {
    if (this.isMonitoring) {
      console.warn('Health monitoring is already running');
      return;
    }

    this.isMonitoring = true;
    console.log('Starting health monitoring...');

    // Initial health check
    this.checkHealth();

    // Set up interval
    this.healthCheckInterval = setInterval(() => {
      this.checkHealth();
    }, intervalMs);
  }

  /**
   * Stop real-time health monitoring
   */
  stopMonitoring(): void {
    if (this.healthCheckInterval) {
      clearInterval(this.healthCheckInterval);
      this.healthCheckInterval = null;
    }
    this.isMonitoring = false;
    console.log('Health monitoring stopped');
  }

  /**
   * Check basic health status
   */
  async checkHealth(): Promise<HealthStatus> {
    try {
      const response = await fetch(`${this.baseUrl}/health`);
      
      if (!response.ok) {
        throw new Error(`Health check failed: ${response.status} ${response.statusText}`);
      }

      const healthStatus: HealthStatus = await response.json();
      this.lastHealthStatus = healthStatus;

      // Notify callbacks
      this.healthCallbacks.forEach(callback => {
        try {
          callback(healthStatus);
        } catch (error) {
          console.error('Error in health callback:', error);
        }
      });

      return healthStatus;
    } catch (error) {
      console.error('Health check failed:', error);
      
      // Create error status
      const errorStatus: HealthStatus = {
        status: 'unhealthy',
        timestamp: new Date().toISOString(),
        uptime_seconds: 0,
        last_heartbeat: new Date().toISOString(),
        system: {
          data_directory_exists: false,
          database_file_exists: false,
          disk_usage: {
            total_gb: 0,
            used_gb: 0,
            free_gb: 0,
            usage_percent: 0
          },
          python_version: 'Unknown',
          working_directory: 'Unknown'
        }
      };

      this.lastHealthStatus = errorStatus;
      this.healthCallbacks.forEach(callback => {
        try {
          callback(errorStatus);
        } catch (error) {
          console.error('Error in health callback:', error);
        }
      });

      return errorStatus;
    }
  }

  /**
   * Get detailed health status
   */
  async getDetailedHealth(): Promise<DetailedHealthStatus> {
    try {
      const response = await fetch(`${this.baseUrl}/health/detailed`);
      
      if (!response.ok) {
        // Fallback to basic health endpoint if detailed fails
        console.warn('Detailed health endpoint not available, falling back to basic health');
        const basicResponse = await fetch(`${this.baseUrl}/health`);
        
        if (!basicResponse.ok) {
          throw new Error(`Health check failed: ${basicResponse.status} ${basicResponse.statusText}`);
        }
        
        const basicHealth = await basicResponse.json();
        // Convert basic health to detailed format
        return {
          status: basicHealth.status || 'unknown',
          timestamp: basicHealth.timestamp || new Date().toISOString(),
          uptime_seconds: basicHealth.uptime_seconds || 0,
          last_heartbeat: basicHealth.last_heartbeat || new Date().toISOString(),
          system: basicHealth.system || {},
          data: {},
          environment: {},
          bybit_connection: basicHealth.bybit_connection || {}
        };
      }

      return await response.json();
    } catch (error) {
      console.error('Detailed health check failed:', error);
      throw error;
    }
  }

  /**
   * Get system status
   */
  async getSystemStatus(): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/status`);
      
      if (!response.ok) {
        throw new Error(`System status check failed: ${response.status} ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('System status check failed:', error);
      throw error;
    }
  }

  /**
   * Get data status
   */
  async getDataStatus(): Promise<any> {
    try {
      const response = await fetch(`${this.baseUrl}/data/status`);
      
      if (!response.ok) {
        throw new Error(`Data status check failed: ${response.status} ${response.statusText}`);
      }

      return await response.json();
    } catch (error) {
      console.error('Data status check failed:', error);
      throw error;
    }
  }

  /**
   * Subscribe to health status updates
   */
  onHealthUpdate(callback: (status: HealthStatus) => void): () => void {
    this.healthCallbacks.push(callback);

    // Return unsubscribe function
    return () => {
      const index = this.healthCallbacks.indexOf(callback);
      if (index > -1) {
        this.healthCallbacks.splice(index, 1);
      }
    };
  }

  /**
   * Get the last known health status
   */
  getLastHealthStatus(): HealthStatus | null {
    return this.lastHealthStatus;
  }

  /**
   * Check if monitoring is active
   */
  isMonitoringActive(): boolean {
    return this.isMonitoring;
  }

  /**
   * Get health status summary for display
   */
  getHealthSummary(status: HealthStatus): {
    isHealthy: boolean;
    statusText: string;
    statusColor: string;
    lastHeartbeat: string;
    uptime: string;
    diskUsage: number;
  } {
    const isHealthy = status.status === 'healthy';
    const statusText = status.status.charAt(0).toUpperCase() + status.status.slice(1);
    const statusColor = isHealthy ? 'green' : status.status === 'degraded' ? 'yellow' : 'red';
    
    const lastHeartbeat = new Date(status.last_heartbeat).toLocaleString();
    const uptime = this.formatUptime(status.uptime_seconds);
    const diskUsage = status.system?.disk_usage?.usage_percent || 0;

    return {
      isHealthy,
      statusText,
      statusColor,
      lastHeartbeat,
      uptime,
      diskUsage
    };
  }

  /**
   * Format uptime in human-readable format
   */
  private formatUptime(seconds: number): string {
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const secs = Math.floor(seconds % 60);

    if (days > 0) {
      return `${days}d ${hours}h ${minutes}m`;
    } else if (hours > 0) {
      return `${hours}h ${minutes}m ${secs}s`;
    } else if (minutes > 0) {
      return `${minutes}m ${secs}s`;
    } else {
      return `${secs}s`;
    }
  }

  /**
   * Test backend connectivity
   */
  async testConnectivity(): Promise<boolean> {
    try {
      const response = await fetch(`${this.baseUrl}/health`, {
        method: 'HEAD',
        timeout: 5000
      });
      return response.ok;
    } catch (error) {
      console.error('Connectivity test failed:', error);
      return false;
    }
  }
}
