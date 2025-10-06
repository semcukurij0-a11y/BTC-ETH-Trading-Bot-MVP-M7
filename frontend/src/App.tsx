import React, { useState, useEffect, useRef } from 'react';
import { TradingDashboard } from './components/TradingDashboard';
import { ConfigPanel } from './components/ConfigPanel';
import { SystemStatus } from './components/SystemStatus';
import { PositionsPanel } from './components/PositionsPanel';
import { SignalsPanel } from './components/SignalsPanel';
import { RiskPanel } from './components/RiskPanel';
import { AlertsPanel } from './components/AlertsPanel';
import { BacktestPanel } from './components/BacktestPanel';
import { HealthMonitor } from './components/HealthMonitor';
import { RealTimeDataTest } from './components/RealTimeDataTest';
import { OrdersPanel } from './components/OrdersPanel';
import { MarginRatioPanel } from './components/MarginRatioPanel';
import { APILatencyPanel } from './components/APILatencyPanel';
import { FundingPanel } from './components/FundingPanel';
import { ControlsPanel } from './components/ControlsPanel';
import { EnhancedAlertsPanel } from './components/EnhancedAlertsPanel';
import { Login } from './components/Login';
import { TradingBotService } from './services/TradingBotService';
import { HealthService } from './services/HealthService';
import { authService, AuthUser } from './services/AuthService';
import { 
  Activity, 
  Settings, 
  BarChart3, 
  Shield, 
  Bell, 
  TestTube,
  TrendingUp,
  DollarSign,
  Heart,
  LogOut,
  User,
  FileText,
  Gauge,
  Wifi,
  TrendingDown,
  Settings2
} from 'lucide-react';

const service = new TradingBotService();
const healthService = new HealthService();

function App() {
  const [activeTab, setActiveTab] = useState('dashboard');
  const [systemData, setSystemData] = useState(null);
  const [isConnected, setIsConnected] = useState(false);
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const navRef = useRef<HTMLDivElement>(null);

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
        
        // Start health monitoring
        healthService.startMonitoring(10000); // Reduced from 5s to 10s
        
        // Start consolidated real-time updates
        const interval = setInterval(async () => {
          const data = await service.getSystemStatus();
          setSystemData(data);
        }, 5000); // Reduced from 1s to 5s

        return () => {
          clearInterval(interval);
          healthService.stopMonitoring();
        };
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

  const handleTabClick = (tabId: string) => {
    setActiveTab(tabId);
    // Scroll to the active tab if it's not visible
    setTimeout(() => {
      if (navRef.current) {
        const activeButton = navRef.current.querySelector(`[data-tab-id="${tabId}"]`) as HTMLElement;
        if (activeButton) {
          activeButton.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
        }
      }
    }, 100);
  };

  // Show login screen if not authenticated
  if (!isAuthenticated) {
    return <Login onLogin={handleLogin} isLoading={isLoading} />;
  }

  const tabs = [
    { id: 'dashboard', label: 'Dashboard', icon: BarChart3 },
    { id: 'positions', label: 'Positions', icon: DollarSign },
    { id: 'orders', label: 'Orders', icon: FileText },
    { id: 'signals', label: 'Signals', icon: TrendingUp },
    { id: 'risk', label: 'Risk', icon: Shield },
    { id: 'margin', label: 'Margin', icon: Gauge },
    { id: 'latency', label: 'API Latency', icon: Wifi },
    { id: 'funding', label: 'Funding', icon: TrendingDown },
    { id: 'controls', label: 'Controls', icon: Settings2 },
    { id: 'alerts', label: 'Alerts', icon: Bell },
    { id: 'backtest', label: 'Backtest', icon: TestTube },
    { id: 'realtime', label: 'Live Data', icon: Activity },
    { id: 'health', label: 'Health', icon: Heart },
    { id: 'config', label: 'Config', icon: Settings },
  ];

  const renderActivePanel = () => {
    switch (activeTab) {
      case 'dashboard':
        return <TradingDashboard service={service} data={systemData} healthService={healthService} />;
      case 'positions':
        return <PositionsPanel service={service} />;
      case 'orders':
        return <OrdersPanel service={service} />;
      case 'signals':
        return <SignalsPanel service={service} />;
      case 'risk':
        return <RiskPanel service={service} data={systemData} />;
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
        return <TradingDashboard service={service} data={systemData} />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Activity className="h-8 w-8 text-blue-400" />
              <div>
                <h1 className="text-xl font-bold">Crypto Trading Bot MVP</h1>
                <p className="text-sm text-gray-400">Futures Perpetuals Trading System</p>
              </div>
            </div>
            <div className="flex items-center space-x-4">
              <SystemStatus isConnected={isConnected} data={systemData} healthService={healthService} />
              <div className="flex items-center space-x-2">
                <User className="h-5 w-5 text-gray-400" />
                <span className="text-sm text-gray-300">{user?.username}</span>
                <span className="text-xs text-gray-500">({user?.role})</span>
              </div>
              <button
                onClick={handleLogout}
                className="flex items-center space-x-1 px-3 py-2 text-sm text-gray-300 hover:text-white transition-colors"
              >
                <LogOut className="h-4 w-4" />
                <span>Logout</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation */}
      <nav className="bg-gray-800 border-b border-gray-700">
        <div className="px-6">
          <div ref={navRef} className="flex space-x-2 overflow-x-auto scrollbar-hide">
            {tabs.map((tab) => {
              const Icon = tab.icon;
              return (
                <button
                  key={tab.id}
                  data-tab-id={tab.id}
                  onClick={() => handleTabClick(tab.id)}
                  className={`flex items-center space-x-1 px-2 py-3 border-b-2 font-medium text-xs transition-colors whitespace-nowrap flex-shrink-0 ${
                    activeTab === tab.id
                      ? 'border-blue-400 text-blue-400'
                      : 'border-transparent text-gray-300 hover:text-white hover:border-gray-300'
                  }`}
                >
                  <Icon className="h-3 w-3" />
                  <span>{tab.label}</span>
                </button>
              );
            })}
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="flex-1">
        {renderActivePanel()}
      </main>
    </div>
  );
}

export default App;