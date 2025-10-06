# Health Monitoring System

This document describes the comprehensive health monitoring system implemented for the crypto trading bot frontend.

## Overview

The health monitoring system provides real-time visibility into the backend system's health status, including:

- **System Health Status**: Overall system health (healthy, degraded, unhealthy)
- **Heartbeat Monitoring**: Real-time heartbeat tracking with age indicators
- **System Metrics**: Disk usage, uptime, database status, data directory status
- **Data Status**: Parquet file information, record counts, last update times
- **Environment Information**: Python version, working directory, environment variables

## Architecture

### Backend Health Endpoints

The backend provides several health endpoints:

- `GET /health` - Basic health check with system status
- `GET /health/detailed` - Detailed health information including data status
- `GET /status` - System status information
- `GET /data/status` - Data file status and metrics

### Frontend Components

#### 1. HealthService (`src/services/HealthService.ts`)

Core service for communicating with backend health endpoints:

```typescript
const healthService = new HealthService('http://localhost:8080');

// Start monitoring with 5-second intervals
healthService.startMonitoring(5000);

// Subscribe to health updates
const unsubscribe = healthService.onHealthUpdate((status) => {
  console.log('Health status:', status);
});

// Get detailed health information
const detailedStatus = await healthService.getDetailedHealth();
```

**Key Features:**
- Real-time health monitoring with configurable intervals
- Automatic error handling and fallback status
- Health status summary generation
- Connectivity testing
- Event-driven updates via callbacks

#### 2. HealthMonitor Component (`src/components/HealthMonitor.tsx`)

Comprehensive health monitoring dashboard:

```typescript
<HealthMonitor 
  healthService={healthService}
  showDetailed={true}
  autoRefresh={true}
  refreshInterval={5000}
/>
```

**Features:**
- Real-time health status display
- System metrics visualization
- Disk usage indicators with color coding
- Data status information
- Environment details
- Manual refresh capability

#### 3. HeartbeatIndicator Component (`src/components/HeartbeatIndicator.tsx`)

Real-time heartbeat status indicator:

```typescript
<HeartbeatIndicator 
  healthService={healthService}
  showDetails={true}
/>
```

**Features:**
- Animated heartbeat pulse
- Heartbeat age tracking
- Status color coding (green/yellow/red)
- Uptime display
- Real-time updates

#### 4. Enhanced SystemStatus Component

Updated system status component with health integration:

```typescript
<SystemStatus 
  isConnected={isConnected}
  data={systemData}
  healthService={healthService}
/>
```

**Features:**
- Integrated health status display
- Clickable health details popup
- Heartbeat indicator
- Legacy system status preservation

## Usage Examples

### Basic Setup

```typescript
import { HealthService } from './services/HealthService';

const healthService = new HealthService('http://localhost:8080');

// Start monitoring
healthService.startMonitoring(5000);

// Subscribe to updates
const unsubscribe = healthService.onHealthUpdate((status) => {
  console.log('Health update:', status);
});
```

### React Component Integration

```typescript
import React, { useEffect, useState } from 'react';
import { HealthMonitor } from './components/HealthMonitor';
import { HealthService } from './services/HealthService';

const MyComponent = () => {
  const [healthService] = useState(() => new HealthService());
  
  useEffect(() => {
    healthService.startMonitoring(5000);
    return () => healthService.stopMonitoring();
  }, [healthService]);
  
  return (
    <HealthMonitor 
      healthService={healthService}
      showDetailed={true}
    />
  );
};
```

### Advanced Configuration

```typescript
// Custom health monitoring with error handling
const healthService = new HealthService('http://localhost:8080');

healthService.onHealthUpdate((status) => {
  const summary = healthService.getHealthSummary(status);
  
  if (!summary.isHealthy) {
    // Handle unhealthy status
    console.warn('System health alert:', summary);
  }
  
  if (summary.diskUsage > 90) {
    // Handle critical disk usage
    console.error('Critical: Disk usage above 90%');
  }
});
```

## Health Status Indicators

### Status Colors
- **Green**: System healthy, all components operational
- **Yellow**: System degraded, some issues detected
- **Red**: System unhealthy, critical issues present

### Heartbeat Status
- **Healthy**: Last heartbeat within 30 seconds
- **Delayed**: Last heartbeat 30-120 seconds ago
- **Critical**: Last heartbeat over 2 minutes ago

### Disk Usage Indicators
- **Green**: Usage below 75%
- **Yellow**: Usage 75-90%
- **Red**: Usage above 90%

## Real-time Updates

The system provides real-time updates through:

1. **HTTP Polling**: Regular health checks at configurable intervals
2. **Event Callbacks**: Subscribe to health status changes
3. **Visual Indicators**: Animated heartbeat pulse and status colors
4. **Automatic Refresh**: Components automatically update when health status changes

## Error Handling

The system includes comprehensive error handling:

- **Connection Failures**: Graceful fallback to error status
- **Timeout Handling**: Configurable timeouts for health checks
- **Retry Logic**: Automatic retry on failed requests
- **Fallback Status**: Default error status when backend is unavailable

## Configuration Options

### HealthService Configuration

```typescript
const healthService = new HealthService(
  'http://localhost:8080',  // Base URL
  5000,                      // Polling interval (ms)
  10000                      // Timeout (ms)
);
```

### Component Configuration

```typescript
<HealthMonitor 
  healthService={healthService}
  showDetailed={true}        // Show detailed information
  autoRefresh={true}         // Enable automatic refresh
  refreshInterval={5000}     // Refresh interval (ms)
  className="custom-class"   // Custom CSS classes
/>
```

## Integration with Existing System

The health monitoring system integrates seamlessly with the existing trading bot interface:

1. **Dashboard Integration**: Health status displayed on main dashboard
2. **Navigation**: Dedicated Health tab for detailed monitoring
3. **Header Integration**: Real-time health status in header
4. **Service Integration**: Works alongside existing TradingBotService

## Monitoring Best Practices

1. **Regular Monitoring**: Use 5-10 second intervals for real-time monitoring
2. **Error Handling**: Always implement error callbacks for health updates
3. **Resource Management**: Stop monitoring when components unmount
4. **User Feedback**: Provide visual feedback for health status changes
5. **Alerting**: Implement alerts for critical health issues

## Troubleshooting

### Common Issues

1. **Connection Refused**: Check backend health service is running
2. **CORS Errors**: Ensure backend allows frontend origin
3. **Timeout Issues**: Increase timeout values for slow networks
4. **Memory Leaks**: Always unsubscribe from health updates

### Debug Mode

Enable debug logging:

```typescript
const healthService = new HealthService('http://localhost:8080');
// Enable console logging for debugging
healthService.onHealthUpdate((status) => {
  console.log('Health update:', status);
});
```

## Future Enhancements

Potential future improvements:

1. **WebSocket Support**: Real-time updates via WebSocket
2. **Historical Data**: Health status history and trends
3. **Alerting System**: Email/SMS alerts for critical issues
4. **Performance Metrics**: CPU, memory, and network monitoring
5. **Health Trends**: Long-term health status analysis
6. **Custom Dashboards**: Configurable health monitoring dashboards

## API Reference

### HealthService Methods

- `startMonitoring(intervalMs: number)`: Start health monitoring
- `stopMonitoring()`: Stop health monitoring
- `checkHealth()`: Perform single health check
- `getDetailedHealth()`: Get detailed health information
- `onHealthUpdate(callback)`: Subscribe to health updates
- `getHealthSummary(status)`: Get formatted health summary
- `testConnectivity()`: Test backend connectivity

### Health Status Interface

```typescript
interface HealthStatus {
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
```

This comprehensive health monitoring system provides real-time visibility into the trading bot's backend health, enabling proactive monitoring and quick issue detection.
