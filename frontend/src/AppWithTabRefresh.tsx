import React, { useState, useEffect, useRef } from 'react';
import { TradingBotService } from './services/TradingBotService';
import { AuthService } from './services/AuthService';
import { HealthService } from './services/HealthService';
import { useTabSpecificRefresh } from './hooks/useTabSpecificRefresh';

// Import optimized components
import { TradingDashboardOptimized } from './components/TradingDashboardOptimized';
import { PositionsPanelOptimized } from './components/PositionsPanelOptimized';
import { OrdersPanelOptimized } from './components/OrdersPanelOptimized';
import { SignalsPanel } from './components/SignalsPanel';
import { RiskPanel } from './components/RiskPanel';
import { MarginRatioPanel } from './components/MarginRatioPanel';
import { APILatencyPanel } from './components/APILatencyPanel';
import { FundingPanel } from './components/FundingPanel';
import { ControlsPanel } from './components/ControlsPanel';
import { EnhancedAlertsPanel } from './components/EnhancedAlertsPanel';
import { BacktestPanel } from './components/BacktestPanel';
import { RealTimeDataTest } from './components/RealTimeDataTest';
import { HealthMonitor } from './components/HealthMonitor';
import { ConfigPanel } from './components/ConfigPanel';

// Import icons
import { 
  BarChart3, 
  TrendingUp, 
  FileText, 
  Activity, 
  Shield, 
  CreditCard, 
  Clock, 
  DollarSign, 
  Settings, 
  Bell, 
  TestTube, 
  Monitor, 
  Heart,
  Cog
} from 'lucide-react';

// Services
const service = new TradingBotService();
const authService = new AuthService();
const healthService = new HealthService();

// Tab configuration with refresh intervals
const TAB_CONFIG = {
  dashboard: { interval: 10000, icon: BarChart3 },
  positions: { interval: 5000, icon: TrendingUp },
  orders: { interval: 10000, icon: FileText },
  signals: { interval: 15000, icon: Activity },
  risk: { interval: 10000, icon: Shield },
  margin: { interval: 10000, icon: CreditCard },
  latency: { interval: 5000, icon: Clock },
  funding: { interval: 30000, icon: DollarSign },
  controls: { interval: 10000, icon: Settings },
  alerts: { interval: 10000, icon: Bell },
  backtest: { interval: 30000, icon: TestTube },
  realtime: { interval: 3000, icon: Monitor },
  health: { interval: 5000, icon: Heart },
  config: { interval: 0, icon: Cog } // No auto-refresh for config
};

interface AuthUser {
  id: string;
  username: string;
  role: string;
}

function AppWithTabRefresh() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [isConnected, setIsConnected] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const navRef = useRef<HTMLDivElement>(null);

  // Get current tab configuration
  const currentTabConfig = TAB_CONFIG[activeTab as keyof typeof TAB_CONFIG] || TAB_CONFIG.dashboard;

  // Use tab-specific refresh hook
  const {
    tradingStats,
    positions,
    orders,
    signals,
    riskData,
    marginData,
    latencyData,
    fundingData,
    systemStatus,
    alerts,
    backtestData,
    healthData,
    lastUpdate,
    isLoading: tabLoading,
    error: tabError,
    refresh: refreshTab
  } = useTabSpecificRefresh(
    activeTab,
    service,
    currentTabConfig.interval
  );

  // Initialize system
  useEffect(() => {
    const initializeSystem = async () => {
      try {
        // Check authentication status
        if (authService.isAuthenticated()) {
          setIsAuthenticated(true);
          setUser(authService.getUser());
        }

        await service.initialize();
        setIsConnected(true);
        
        // Start health monitoring (only once, not per tab)
        healthService.startMonitoring(10000);
        
        console.log('✅ System initialized with tab-specific refresh');
      } catch (error) {
        console.error('Failed to initialize trading system:', error);
      }
    };

    initializeSystem();
  }, []);

  const handleLogin = async (username: string, password: string): Promise<boolean> => {
    setIsLoading(true);
    try {
      const result = await authService.login(username, password);
      if (result.success) {
        setIsAuthenticated(true);
        setUser(result.user || null);
        return true;
      }
      return false;
    } catch (error) {
      console.error('Login error:', error);
      return false;
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = async () => {
    try {
      await authService.logout();
      setIsAuthenticated(false);
      setUser(null);
    } catch (error) {
      console.error('Logout error:', error);
    }
  };

  const handleTabChange = (tabId: string) => {
    setActiveTab(tabId);
    console.log(`🔄 Switched to tab: ${tabId} (refresh interval: ${TAB_CONFIG[tabId as keyof typeof TAB_CONFIG]?.interval || 10000}ms)`);
  };

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
    { id: 'positions', label: 'Positions', icon: TrendingUp },
    { id: 'orders', label: 'Orders', icon: FileText },
    { id: 'signals', label: 'Signals', icon: Activity },
    { id: 'risk', label: 'Risk', icon: Shield },
    { id: 'margin', label: 'Margin', icon: CreditCard },
    { id: 'latency', label: 'Latency', icon: Clock },
    { id: 'funding', label: 'Funding', icon: DollarSign },
    { id: 'controls', label: 'Controls', icon: Settings },
    { id: 'alerts', label: 'Alerts', icon: Bell },
    { id: 'backtest', label: 'Backtest', icon: TestTube },
    { id: 'realtime', label: 'Real-time', icon: Monitor },
    { id: 'health', label: 'Health', icon: Heart },
    { id: 'config', label: 'Config', icon: Cog }
  ];

  const renderActivePanel = () => {
    // Pass only the data needed for each tab
    switch (activeTab) {
      case 'dashboard':
        return (
          <TradingDashboardOptimized 
            service={service} 
            data={{ 
              tradingStats, 
              systemStatus,
              lastUpdate,
              isLoading: tabLoading,
              error: tabError
            }} 
            healthService={healthService} 
          />
        );
      case 'positions':
        return (
          <PositionsPanelOptimized 
            service={service} 
            data={positions} 
            isLoading={tabLoading}
            error={tabError}
            lastUpdate={lastUpdate}
          />
        );
      case 'orders':
        return (
          <OrdersPanelOptimized 
            service={service} 
            data={orders} 
            isLoading={tabLoading}
            error={tabError}
            lastUpdate={lastUpdate}
          />
        );
      case 'signals':
        return <SignalsPanel service={service} />;
      case 'risk':
        return <RiskPanel service={service} />;
      case 'margin':
        return <MarginRatioPanel service={service} />;
      case 'latency':
        return <APILatencyPanel service={service} />;
      case 'funding':
        return <FundingPanel service={service} />;
      case 'controls':
        return <ControlsPanel service={service} />;
      case 'alerts':
        return <EnhancedAlertsPanel service={service} />;
      case 'backtest':
        return <BacktestPanel service={service} />;
      case 'realtime':
        return <RealTimeDataTest />;
      case 'health':
        return <HealthMonitor healthService={healthService} showDetailed={true} />;
      case 'config':
        return <ConfigPanel service={service} />;
      default:
        return <TradingDashboardOptimized service={service} data={{ systemStatus }} />;
    }
  };

  if (!isAuthenticated) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <div className="bg-gray-800 p-8 rounded-lg shadow-lg w-96">
          <h2 className="text-2xl font-bold text-white mb-6 text-center">Login</h2>
          <form onSubmit={(e) => {
            e.preventDefault();
            const formData = new FormData(e.currentTarget);
            const username = formData.get('username') as string;
            const password = formData.get('password') as string;
            handleLogin(username, password);
          }}>
            <div className="mb-4">
              <label className="block text-gray-300 mb-2">Username</label>
              <input
                type="text"
                name="username"
                className="w-full px-3 py-2 bg-gray-700 text-white rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                defaultValue="admin"
                required
              />
            </div>
            <div className="mb-6">
              <label className="block text-gray-300 mb-2">Password</label>
              <input
                type="password"
                name="password"
                className="w-full px-3 py-2 bg-gray-700 text-white rounded focus:outline-none focus:ring-2 focus:ring-blue-500"
                defaultValue="admin123"
                required
              />
            </div>
            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-blue-600 text-white py-2 px-4 rounded hover:bg-blue-700 disabled:opacity-50"
            >
              {isLoading ? 'Logging in...' : 'Login'}
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <h1 className="text-xl font-bold">Trading Bot Dashboard</h1>
              <div className="flex items-center space-x-2 text-sm text-gray-400">
                <span>User: {user?.username}</span>
                <span>•</span>
                <span>Tab: {activeTab}</span>
                <span>•</span>
                <span>Refresh: {currentTabConfig.interval}ms</span>
                {lastUpdate && (
                  <>
                    <span>•</span>
                    <span>Updated: {new Date(lastUpdate).toLocaleTimeString()}</span>
                  </>
                )}
              </div>
            </div>
            <div className="flex items-center space-x-4">
              {tabLoading && (
                <div className="flex items-center space-x-2 text-blue-400">
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-400"></div>
                  <span className="text-sm">Refreshing...</span>
                </div>
              )}
              <button
                onClick={refreshTab}
                className="px-3 py-1 bg-blue-600 text-white rounded text-sm hover:bg-blue-700"
              >
                Refresh
              </button>
              <button
                onClick={handleLogout}
                className="px-3 py-1 bg-red-600 text-white rounded text-sm hover:bg-red-700"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="flex">
        {/* Sidebar Navigation */}
        <nav ref={navRef} className="w-64 bg-gray-800 border-r border-gray-700 min-h-screen">
          <div className="p-4">
            <div className="space-y-2">
              {tabs.map((tab) => {
                const Icon = tab.icon;
                const isActive = activeTab === tab.id;
                const tabConfig = TAB_CONFIG[tab.id as keyof typeof TAB_CONFIG];
                
                return (
                  <button
                    key={tab.id}
                    onClick={() => handleTabChange(tab.id)}
                    className={`w-full flex items-center space-x-3 px-3 py-2 rounded-lg text-left transition-colors ${
                      isActive
                        ? 'bg-blue-600 text-white'
                        : 'text-gray-300 hover:bg-gray-700 hover:text-white'
                    }`}
                  >
                    <Icon className="h-5 w-5" />
                    <span>{tab.label}</span>
                    {tabConfig.interval > 0 && (
                      <span className="ml-auto text-xs text-gray-400">
                        {tabConfig.interval / 1000}s
                      </span>
                    )}
                  </button>
                );
              })}
            </div>
          </div>
        </nav>

        {/* Main Content */}
        <main className="flex-1 p-6">
          {tabError && (
            <div className="mb-4 p-4 bg-red-900 border border-red-700 rounded-lg">
              <div className="flex items-center">
                <div className="text-red-400 mr-2">⚠️</div>
                <div>
                  <h3 className="text-red-400 font-medium">Data Error</h3>
                  <p className="text-red-300 text-sm">{tabError}</p>
                </div>
              </div>
            </div>
          )}
          
          {renderActivePanel()}
        </main>
      </div>
    </div>
  );
}

export default AppWithTabRefresh;
