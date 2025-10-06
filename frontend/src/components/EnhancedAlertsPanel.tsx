import React, { useState, useEffect } from 'react';
import { Bell, AlertTriangle, CheckCircle, Info, X, Send, TestTube, Settings, Mail, MessageSquare } from 'lucide-react';
import { AlertsService } from '../services/AlertsService';

interface Alert {
  id: string;
  type: 'error' | 'warning' | 'info' | 'success';
  title: string;
  message: string;
  timestamp: string;
  read: boolean;
  source: 'system' | 'trading' | 'risk' | 'api';
}

interface AlertsPanelProps {
  service: any;
}

export const EnhancedAlertsPanel: React.FC<AlertsPanelProps> = ({ service }) => {
  const [alerts, setAlerts] = useState<Alert[]>([]);
  const [filter, setFilter] = useState<'all' | 'unread' | 'error' | 'warning'>('all');
  const [alertsService] = useState(new AlertsService());
  const [testing, setTesting] = useState<{ telegram: boolean; email: boolean }>({ telegram: false, email: false });
  const [showSettings, setShowSettings] = useState(false);

  useEffect(() => {
    const loadAlerts = async () => {
      try {
        const data = await service.getAlerts();
        setAlerts(data);
      } catch (error) {
        console.error('Failed to load alerts:', error);
      }
    };

    loadAlerts();
    const interval = setInterval(loadAlerts, 5000);
    return () => clearInterval(interval);
  }, [service]);

  const markAsRead = async (alertId: string) => {
    try {
      await service.markAlertAsRead(alertId);
      setAlerts(alerts.map(alert => 
        alert.id === alertId ? { ...alert, read: true } : alert
      ));
    } catch (error) {
      console.error('Failed to mark alert as read:', error);
    }
  };

  const dismissAlert = async (alertId: string) => {
    try {
      await service.dismissAlert(alertId);
      setAlerts(alerts.filter(alert => alert.id !== alertId));
    } catch (error) {
      console.error('Failed to dismiss alert:', error);
    }
  };

  const testTelegramAlert = async () => {
    setTesting(prev => ({ ...prev, telegram: true }));
    try {
      const result = await alertsService.testTelegramAlert();
      if (result.success) {
        alert('✅ Telegram test alert sent successfully!');
      } else {
        alert(`❌ Telegram test failed: ${result.error}`);
      }
    } catch (error) {
      alert(`❌ Telegram test error: ${error}`);
    } finally {
      setTesting(prev => ({ ...prev, telegram: false }));
    }
  };

  const testEmailAlert = async () => {
    setTesting(prev => ({ ...prev, email: true }));
    try {
      const result = await alertsService.testEmailAlert();
      if (result.success) {
        alert('✅ Email test alert sent successfully!');
      } else {
        alert(`❌ Email test failed: ${result.error}`);
      }
    } catch (error) {
      alert(`❌ Email test error: ${error}`);
    } finally {
      setTesting(prev => ({ ...prev, email: false }));
    }
  };

  const sendMockOrderFillAlert = async () => {
    try {
      await alertsService.sendOrderFillAlert({
        side: 'Buy',
        quantity: 0.1,
        symbol: 'BTCUSDT',
        price: 50000,
        orderId: 'TEST123'
      });
      alert('✅ Mock order fill alert sent!');
    } catch (error) {
      alert(`❌ Failed to send order fill alert: ${error}`);
    }
  };

  const sendMockErrorSpikeAlert = async () => {
    try {
      await alertsService.sendErrorSpikeAlert(5, '60 seconds');
      alert('✅ Mock error spike alert sent!');
    } catch (error) {
      alert(`❌ Failed to send error spike alert: ${error}`);
    }
  };

  const sendMockDailyLimitAlert = async () => {
    try {
      await alertsService.sendDailyLimitAlert('Max Loss', 0.035, 0.03);
      alert('✅ Mock daily limit alert sent!');
    } catch (error) {
      alert(`❌ Failed to send daily limit alert: ${error}`);
    }
  };

  const sendMockDrawdownAlert = async () => {
    try {
      await alertsService.sendDrawdownAlert(0.12, 0.10);
      alert('✅ Mock drawdown alert sent!');
    } catch (error) {
      alert(`❌ Failed to send drawdown alert: ${error}`);
    }
  };

  const getAlertIcon = (type: string) => {
    switch (type) {
      case 'error':
        return <AlertTriangle className="h-5 w-5 text-red-400" />;
      case 'warning':
        return <AlertTriangle className="h-5 w-5 text-yellow-400" />;
      case 'success':
        return <CheckCircle className="h-5 w-5 text-green-400" />;
      default:
        return <Info className="h-5 w-5 text-blue-400" />;
    }
  };

  const getAlertBorderColor = (type: string) => {
    switch (type) {
      case 'error':
        return 'border-l-red-400';
      case 'warning':
        return 'border-l-yellow-400';
      case 'success':
        return 'border-l-green-400';
      default:
        return 'border-l-blue-400';
    }
  };

  const filteredAlerts = alerts.filter(alert => {
    if (filter === 'all') return true;
    if (filter === 'unread') return !alert.read;
    return alert.type === filter;
  });

  const unreadCount = alerts.filter(alert => !alert.read).length;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <h2 className="text-2xl font-bold text-white">System Alerts</h2>
          {unreadCount > 0 && (
            <span className="bg-red-500 text-white px-2 py-1 rounded-full text-sm font-medium">
              {unreadCount} unread
            </span>
          )}
        </div>
        
        <div className="flex space-x-2">
          <button
            onClick={() => setShowSettings(!showSettings)}
            className="flex items-center space-x-2 px-3 py-2 bg-gray-700 text-white rounded hover:bg-gray-600"
          >
            <Settings className="h-4 w-4" />
            <span>Settings</span>
          </button>
          {['all', 'unread', 'error', 'warning'].map((filterType) => (
            <button
              key={filterType}
              onClick={() => setFilter(filterType as any)}
              className={`px-3 py-2 rounded text-sm font-medium transition-colors ${
                filter === filterType
                  ? 'bg-blue-600 text-white'
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
              }`}
            >
              {filterType.charAt(0).toUpperCase() + filterType.slice(1)}
            </button>
          ))}
        </div>
      </div>

      {/* Alert Integration Settings */}
      {showSettings && (
        <div className="bg-gray-800 rounded-lg p-6">
          <h3 className="text-lg font-semibold text-white mb-4">Alert Integration Settings</h3>
          
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {/* Telegram Settings */}
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <MessageSquare className="h-5 w-5 text-blue-400" />
                <h4 className="text-lg font-medium text-white">Telegram Integration</h4>
              </div>
              <div className="space-y-2">
                <div className="text-sm text-gray-400">
                  Bot Token: 7552181112:AAHMSNJfbF2xfZ9tWPis4NddJGCfBPnFjvY
                </div>
                <div className="text-sm text-gray-400">
                  Chat ID: 8490393553
                </div>
                <button
                  onClick={testTelegramAlert}
                  disabled={testing.telegram}
                  className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 disabled:opacity-50"
                >
                  <TestTube className="h-4 w-4" />
                  <span>{testing.telegram ? 'Testing...' : 'Test Telegram'}</span>
                </button>
              </div>
            </div>

            {/* Email Settings */}
            <div className="space-y-4">
              <div className="flex items-center space-x-3">
                <Mail className="h-5 w-5 text-green-400" />
                <h4 className="text-lg font-medium text-white">Email Integration</h4>
              </div>
              <div className="space-y-2">
                <div className="text-sm text-gray-400">
                  Recipient: zombiewins23@gmail.com
                </div>
                <button
                  onClick={testEmailAlert}
                  disabled={testing.email}
                  className="flex items-center space-x-2 px-4 py-2 bg-green-600 text-white rounded hover:bg-green-700 disabled:opacity-50"
                >
                  <TestTube className="h-4 w-4" />
                  <span>{testing.email ? 'Testing...' : 'Test Email'}</span>
                </button>
              </div>
            </div>
          </div>

          {/* Mock Alert Testing */}
          <div className="mt-6 pt-6 border-t border-gray-700">
            <h4 className="text-lg font-medium text-white mb-4">Mock Alert Testing</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
              <button
                onClick={sendMockOrderFillAlert}
                className="flex items-center space-x-2 px-3 py-2 bg-purple-600 text-white rounded hover:bg-purple-700 text-sm"
              >
                <Send className="h-4 w-4" />
                <span>Order Fill</span>
              </button>
              <button
                onClick={sendMockErrorSpikeAlert}
                className="flex items-center space-x-2 px-3 py-2 bg-red-600 text-white rounded hover:bg-red-700 text-sm"
              >
                <Send className="h-4 w-4" />
                <span>Error Spike</span>
              </button>
              <button
                onClick={sendMockDailyLimitAlert}
                className="flex items-center space-x-2 px-3 py-2 bg-yellow-600 text-white rounded hover:bg-yellow-700 text-sm"
              >
                <Send className="h-4 w-4" />
                <span>Daily Limit</span>
              </button>
              <button
                onClick={sendMockDrawdownAlert}
                className="flex items-center space-x-2 px-3 py-2 bg-orange-600 text-white rounded hover:bg-orange-700 text-sm"
              >
                <Send className="h-4 w-4" />
                <span>Drawdown</span>
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="space-y-4">
        {filteredAlerts.length === 0 ? (
          <div className="bg-gray-800 rounded-lg p-8 text-center">
            <Bell className="h-12 w-12 text-gray-600 mx-auto mb-4" />
            <p className="text-gray-400">No alerts to display</p>
          </div>
        ) : (
          filteredAlerts.map((alert) => (
            <div
              key={alert.id}
              className={`bg-gray-800 rounded-lg p-4 border-l-4 ${getAlertBorderColor(alert.type)} ${
                !alert.read ? 'bg-gray-750' : ''
              }`}
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start space-x-3">
                  {getAlertIcon(alert.type)}
                  <div className="flex-1">
                    <div className="flex items-center space-x-2">
                      <h4 className="font-semibold text-white">{alert.title}</h4>
                      {!alert.read && (
                        <span className="w-2 h-2 bg-blue-500 rounded-full"></span>
                      )}
                    </div>
                    <p className="text-gray-300 mt-1">{alert.message}</p>
                    <p className="text-xs text-gray-400 mt-2">
                      {new Date(alert.timestamp).toLocaleString()}
                    </p>
                  </div>
                </div>
                
                <div className="flex items-center space-x-2 ml-4">
                  {!alert.read && (
                    <button
                      onClick={() => markAsRead(alert.id)}
                      className="text-blue-400 hover:text-blue-300 text-sm"
                    >
                      Mark as read
                    </button>
                  )}
                  <button
                    onClick={() => dismissAlert(alert.id)}
                    className="text-gray-400 hover:text-white"
                  >
                    <X className="h-4 w-4" />
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Alert Statistics */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Alert Summary</h3>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="text-center">
            <p className="text-2xl font-bold text-white">{alerts.length}</p>
            <p className="text-sm text-gray-400">Total</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-red-400">
              {alerts.filter(a => a.type === 'error').length}
            </p>
            <p className="text-sm text-gray-400">Errors</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-yellow-400">
              {alerts.filter(a => a.type === 'warning').length}
            </p>
            <p className="text-sm text-gray-400">Warnings</p>
          </div>
          <div className="text-center">
            <p className="text-2xl font-bold text-blue-400">{unreadCount}</p>
            <p className="text-sm text-gray-400">Unread</p>
          </div>
        </div>
      </div>
    </div>
  );
};
