import { useState, useEffect, useCallback, useRef } from 'react';
import { TradingBotService } from '../services/TradingBotService';

interface TabRefreshConfig {
  tabId: string;
  refreshInterval: number;
  dataFetchers: {
    [key: string]: () => Promise<any>;
  };
}

interface TabData {
  [key: string]: any;
  lastUpdate: string | null;
  isLoading: boolean;
  error: string | null;
}

export const useTabSpecificRefresh = (
  activeTab: string,
  service: TradingBotService,
  refreshInterval: number = 10000
) => {
  const [tabData, setTabData] = useState<TabData>({
    lastUpdate: null,
    isLoading: false,
    error: null
  });
  
  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const isActiveRef = useRef(true);

  // Define data fetchers for each tab
  const getTabDataFetchers = useCallback((tab: string) => {
    const fetchers: { [key: string]: () => Promise<any> } = {};

    switch (tab) {
      case 'dashboard':
        fetchers.tradingStats = () => service.getTradingStats();
        fetchers.equityCurve = () => service.getEquityCurve();
        break;
      
      case 'positions':
        fetchers.positions = () => service.getPositions();
        break;
      
      case 'orders':
        fetchers.orders = () => service.getOrderHistory();
        break;
      
      case 'signals':
        fetchers.signals = () => service.getCurrentSignals();
        fetchers.signalHistory = () => service.getSignalHistory();
        break;
      
      case 'risk':
        fetchers.riskData = () => service.getRiskParameters();
        fetchers.accountInfo = () => service.getAccountInfo();
        break;
      
      case 'margin':
        fetchers.marginData = () => service.getMarginData();
        fetchers.accountInfo = () => service.getAccountInfo();
        break;
      
      case 'latency':
        fetchers.latencyData = () => service.getLatencyData();
        break;
      
      case 'funding':
        fetchers.fundingData = () => service.getFundingRates();
        break;
      
      case 'controls':
        fetchers.systemStatus = () => service.getSystemStatus();
        break;
      
      case 'alerts':
        fetchers.alerts = () => service.getAlerts();
        break;
      
      case 'backtest':
        fetchers.backtestData = () => service.getBacktestResults();
        break;
      
      case 'health':
        fetchers.healthData = () => service.getSystemStatus();
        break;
      
      default:
        // For unknown tabs, fetch minimal data
        fetchers.systemStatus = () => service.getSystemStatus();
    }

    return fetchers;
  }, [service]);

  // Fetch data for the current tab
  const fetchTabData = useCallback(async () => {
    if (!isActiveRef.current) return;

    setTabData(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const fetchers = getTabDataFetchers(activeTab);
      const dataEntries = await Promise.all(
        Object.entries(fetchers).map(async ([key, fetcher]) => {
          try {
            const data = await fetcher();
            return [key, data];
          } catch (error) {
            console.error(`Error fetching ${key} for tab ${activeTab}:`, error);
            return [key, null];
          }
        })
      );

      const newData = Object.fromEntries(dataEntries);
      
      if (isActiveRef.current) {
        setTabData(prev => ({
          ...prev,
          ...newData,
          lastUpdate: new Date().toISOString(),
          isLoading: false,
          error: null
        }));
      }
    } catch (error) {
      if (isActiveRef.current) {
        setTabData(prev => ({
          ...prev,
          isLoading: false,
          error: error instanceof Error ? error.message : 'Failed to fetch data'
        }));
      }
    }
  }, [activeTab, getTabDataFetchers]);

  // Manual refresh function
  const refresh = useCallback(() => {
    fetchTabData();
  }, [fetchTabData]);

  // Start/stop polling based on active tab
  useEffect(() => {
    // Clear existing interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Set active flag
    isActiveRef.current = true;

    // Initial fetch
    fetchTabData();

    // Set up polling for the active tab
    if (refreshInterval > 0) {
      intervalRef.current = setInterval(fetchTabData, refreshInterval);
    }

    // Cleanup function
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
        intervalRef.current = null;
      }
      isActiveRef.current = false;
    };
  }, [activeTab, fetchTabData, refreshInterval]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
      isActiveRef.current = false;
    };
  }, []);

  return {
    ...tabData,
    refresh,
    isActive: isActiveRef.current
  };
};

// Hook for individual components to get their specific data
export const useComponentData = (
  activeTab: string,
  componentKey: string,
  service: TradingBotService,
  refreshInterval: number = 10000
) => {
  const [data, setData] = useState<any>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    if (activeTab !== componentKey) return;

    setIsLoading(true);
    setError(null);

    try {
      let result;
      switch (componentKey) {
        case 'tradingStats':
          result = await service.getTradingStats();
          break;
        case 'positions':
          result = await service.getPositions();
          break;
        case 'orders':
          result = await service.getOrderHistory();
          break;
        case 'signals':
          result = await service.getCurrentSignals();
          break;
        case 'accountInfo':
          result = await service.getAccountInfo();
          break;
        case 'systemStatus':
          result = await service.getSystemStatus();
          break;
        default:
          result = null;
      }

      setData(result);
      setLastUpdate(new Date().toISOString());
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to fetch data');
    } finally {
      setIsLoading(false);
    }
  }, [activeTab, componentKey, service]);

  useEffect(() => {
    if (activeTab === componentKey) {
      fetchData();
      const interval = setInterval(fetchData, refreshInterval);
      return () => clearInterval(interval);
    }
  }, [activeTab, componentKey, fetchData, refreshInterval]);

  return {
    data,
    isLoading,
    error,
    lastUpdate,
    refresh: fetchData
  };
};
