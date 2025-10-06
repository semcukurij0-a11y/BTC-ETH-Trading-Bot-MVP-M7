import React, { useState, useEffect } from 'react';
import { Play, Pause, Square, AlertTriangle, Shield, Activity, Power, PowerOff } from 'lucide-react';

interface ControlsPanelProps {
  service: any;
}

interface SystemStatus {
  isRunning: boolean;
  isPaused: boolean;
  isKillSwitchActive: boolean;
  lastUpdate: string;
  positionsCount: number;
  activeOrders: number;
  systemHealth: 'healthy' | 'warning' | 'critical';
}

export const ControlsPanel: React.FC<ControlsPanelProps> = ({ service }) => {
  const [status, setStatus] = useState<SystemStatus>({
    isRunning: false,
    isPaused: false,
    isKillSwitchActive: false,
    lastUpdate: new Date().toISOString(),
    positionsCount: 0,
    activeOrders: 0,
    systemHealth: 'healthy'
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchSystemStatus = async () => {
    try {
      const data = await service.getSystemStatus();
      if (data) {
        setStatus({
          isRunning: data.status === 'running',
          isPaused: data.status === 'paused',
          isKillSwitchActive: data.killSwitch || false,
          lastUpdate: data.timestamp || new Date().toISOString(),
          positionsCount: data.positionsCount || 0,
          activeOrders: data.activeOrders || 0,
          systemHealth: data.health || 'healthy'
        });
      }
    } catch (err) {
      console.error('Error fetching system status:', err);
    }
  };

  useEffect(() => {
    fetchSystemStatus();
    const interval = setInterval(fetchSystemStatus, 2000);
    return () => clearInterval(interval);
  }, []);

  const handlePauseResume = async () => {
    setLoading(true);
    try {
      if (status.isPaused) {
        await service.resumeTrading();
        setStatus(prev => ({ ...prev, isPaused: false, isRunning: true }));
      } else {
        await service.pauseTrading();
        setStatus(prev => ({ ...prev, isPaused: true, isRunning: false }));
      }
      setError(null);
    } catch (err) {
      setError(`Failed to ${status.isPaused ? 'resume' : 'pause'} trading: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  const handleKillSwitch = async () => {
    if (!status.isKillSwitchActive) {
      const confirmed = window.confirm(
        '⚠️ KILL SWITCH ACTIVATION ⚠️\n\n' +
        'This will:\n' +
        '• Flatten ALL positions immediately\n' +
        '• Cancel ALL pending orders\n' +
        '• Block new entries until manually reset\n\n' +
        'Are you sure you want to activate the kill switch?'
      );
      
      if (!confirmed) return;
    }

    setLoading(true);
    try {
      if (status.isKillSwitchActive) {
        await service.resetKillSwitch();
        setStatus(prev => ({ ...prev, isKillSwitchActive: false }));
      } else {
        await service.activateKillSwitch();
        setStatus(prev => ({ 
          ...prev, 
          isKillSwitchActive: true, 
          isRunning: false, 
          isPaused: false 
        }));
      }
      setError(null);
    } catch (err) {
      setError(`Failed to ${status.isKillSwitchActive ? 'reset' : 'activate'} kill switch: ${err}`);
    } finally {
      setLoading(false);
    }
  };

  const getHealthColor = (health: string) => {
    switch (health) {
      case 'healthy': return 'text-green-400';
      case 'warning': return 'text-yellow-400';
      case 'critical': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getHealthBgColor = (health: string) => {
    switch (health) {
      case 'healthy': return 'bg-green-900/20 border-green-500';
      case 'warning': return 'bg-yellow-900/20 border-yellow-500';
      case 'critical': return 'bg-red-900/20 border-red-500';
      default: return 'bg-gray-900/20 border-gray-500';
    }
  };

  return (
    <div className="p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">Trading Controls</h2>
        <div className="text-sm text-gray-400">
          Last updated: {new Date(status.lastUpdate).toLocaleString()}
        </div>
      </div>

      {/* Error Message */}
      {error && (
        <div className="bg-red-900/50 border border-red-500 text-red-200 px-4 py-3 rounded-lg">
          {error}
        </div>
      )}

      {/* System Status Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center space-x-3">
            <Activity className="h-6 w-6 text-blue-400" />
            <div>
              <h3 className="text-lg font-semibold text-white">System Status</h3>
              <p className={`text-sm ${status.isRunning ? 'text-green-400' : status.isPaused ? 'text-yellow-400' : 'text-red-400'}`}>
                {status.isKillSwitchActive ? 'KILL SWITCH ACTIVE' : 
                 status.isRunning ? 'RUNNING' : 
                 status.isPaused ? 'PAUSED' : 'STOPPED'}
              </p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center space-x-3">
            <Shield className="h-6 w-6 text-purple-400" />
            <div>
              <h3 className="text-lg font-semibold text-white">Positions</h3>
              <p className="text-sm text-white">{status.positionsCount} active</p>
            </div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <div className="flex items-center space-x-3">
            <AlertTriangle className={`h-6 w-6 ${getHealthColor(status.systemHealth)}`} />
            <div>
              <h3 className="text-lg font-semibold text-white">System Health</h3>
              <p className={`text-sm ${getHealthColor(status.systemHealth)}`}>
                {status.systemHealth.toUpperCase()}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Control Buttons */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Pause/Resume Control */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Trading Control</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Current Status</p>
                <p className={`text-lg font-medium ${status.isRunning ? 'text-green-400' : 'text-yellow-400'}`}>
                  {status.isRunning ? 'Running' : 'Paused'}
                </p>
              </div>
              <button
                onClick={handlePauseResume}
                disabled={loading || status.isKillSwitchActive}
                className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-colors ${
                  status.isPaused
                    ? 'bg-green-600 text-white hover:bg-green-700'
                    : 'bg-yellow-600 text-white hover:bg-yellow-700'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                {status.isPaused ? (
                  <>
                    <Play className="h-5 w-5" />
                    <span>Resume Trading</span>
                  </>
                ) : (
                  <>
                    <Pause className="h-5 w-5" />
                    <span>Pause Trading</span>
                  </>
                )}
              </button>
            </div>
            <div className="text-xs text-gray-500">
              {status.isKillSwitchActive 
                ? 'Kill switch is active - trading controls disabled'
                : 'Pause trading to stop new entries while keeping existing positions'
              }
            </div>
          </div>
        </div>

        {/* Kill Switch Control */}
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Emergency Kill Switch</h3>
          <div className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-gray-400">Kill Switch Status</p>
                <p className={`text-lg font-medium ${status.isKillSwitchActive ? 'text-red-400' : 'text-green-400'}`}>
                  {status.isKillSwitchActive ? 'ACTIVE' : 'Inactive'}
                </p>
              </div>
              <button
                onClick={handleKillSwitch}
                disabled={loading}
                className={`flex items-center space-x-2 px-6 py-3 rounded-lg font-medium transition-colors ${
                  status.isKillSwitchActive
                    ? 'bg-green-600 text-white hover:bg-green-700'
                    : 'bg-red-600 text-white hover:bg-red-700'
                } disabled:opacity-50`}
              >
                {status.isKillSwitchActive ? (
                  <>
                    <PowerOff className="h-5 w-5" />
                    <span>Reset Kill Switch</span>
                  </>
                ) : (
                  <>
                    <Power className="h-5 w-5" />
                    <span>Activate Kill Switch</span>
                  </>
                )}
              </button>
            </div>
            <div className="text-xs text-gray-500">
              {status.isKillSwitchActive 
                ? 'Kill switch is active - all trading is halted'
                : 'Emergency stop - flattens all positions and blocks new entries'
              }
            </div>
          </div>
        </div>
      </div>

      {/* System Health Status */}
      <div className={`rounded-lg p-6 border ${getHealthBgColor(status.systemHealth)}`}>
        <div className="flex items-center space-x-3">
          <AlertTriangle className={`h-6 w-6 ${getHealthColor(status.systemHealth)}`} />
          <div>
            <h3 className="text-lg font-semibold text-white">System Health Status</h3>
            <p className={`text-sm ${getHealthColor(status.systemHealth)}`}>
              {status.systemHealth === 'healthy' && 'All systems operational'}
              {status.systemHealth === 'warning' && 'Some systems showing warnings'}
              {status.systemHealth === 'critical' && 'Critical issues detected - immediate attention required'}
            </p>
          </div>
        </div>
      </div>

      {/* Active Positions and Orders */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Active Positions</h3>
          <div className="text-center">
            <div className="text-3xl font-bold text-white mb-2">{status.positionsCount}</div>
            <div className="text-sm text-gray-400">Open positions</div>
          </div>
        </div>

        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Active Orders</h3>
          <div className="text-center">
            <div className="text-3xl font-bold text-white mb-2">{status.activeOrders}</div>
            <div className="text-sm text-gray-400">Pending orders</div>
          </div>
        </div>
      </div>

      {/* Control Actions Log */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold text-white mb-4">Recent Actions</h3>
        <div className="space-y-2">
          <div className="text-sm text-gray-400">
            • System status: {status.isKillSwitchActive ? 'Kill switch active' : status.isRunning ? 'Running' : 'Paused'}
          </div>
          <div className="text-sm text-gray-400">
            • Last update: {new Date(status.lastUpdate).toLocaleString()}
          </div>
          <div className="text-sm text-gray-400">
            • Health status: {status.systemHealth.toUpperCase()}
          </div>
        </div>
      </div>
    </div>
  );
};
