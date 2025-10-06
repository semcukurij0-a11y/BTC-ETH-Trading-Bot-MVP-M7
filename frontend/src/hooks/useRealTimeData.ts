import { useState, useEffect, useCallback, useRef } from 'react';
import { TradingBotService } from '../services/TradingBotService';

interface UseRealTimeDataOptions {
  pollingInterval?: number;
  autoStart?: boolean;
}

interface RealTimeData {
  positions: any[];
  accountInfo: any;
  tradingStats: any;
  systemStatus: any;
  isLoading: boolean;
  error: string | null;
  lastUpdate: string | null;
}

export const useRealTimeData = (options: UseRealTimeDataOptions = {}) => {
  const {
    pollingInterval = 15000, // Further reduced polling frequency for better performance
    autoStart = true
  } = options;

  const [data, setData] = useState<RealTimeData>({
    positions: [],
    accountInfo: null,
    tradingStats: null,
    systemStatus: null,
    isLoading: false,
    error: null,
    lastUpdate: null
  });

  const serviceRef = useRef<TradingBotService | null>(null);
  const pollingRef = useRef<NodeJS.Timeout | null>(null);

  // Initialize service
  useEffect(() => {
    if (!serviceRef.current) {
      serviceRef.current = new TradingBotService();
    }
  }, []);

  // Fetch all live data
  const fetchLiveData = useCallback(async () => {
    if (!serviceRef.current) return;

    setData(prev => ({ ...prev, isLoading: true, error: null }));

    try {
      const [positions, accountInfo, tradingStats, systemStatus] = await Promise.all([
        serviceRef.current.getPositions(),
        serviceRef.current.getAccountInfo(),
        serviceRef.current.getTradingStats(),
        serviceRef.current.getSystemStatus()
      ]);

      setData({
        positions,
        accountInfo,
        tradingStats,
        systemStatus,
        isLoading: false,
        error: null,
        lastUpdate: new Date().toISOString()
      });

      console.log('✅ Live data updated:', {
        positions: positions.length,
        accountBalance: accountInfo?.wallet?.balance,
        activePositions: accountInfo?.positions?.activeCount
      });

    } catch (error) {
      console.error('❌ Error fetching live data:', error);
      setData(prev => ({
        ...prev,
        isLoading: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      }));
    }
  }, []);

  // Start polling
  const startPolling = useCallback(() => {
    if (pollingRef.current) return; // Already polling

    console.log(`🔄 Starting real-time polling every ${pollingInterval}ms`);
    
    // Initial fetch
    fetchLiveData();
    
    // Set up polling
    pollingRef.current = setInterval(fetchLiveData, pollingInterval);
  }, [fetchLiveData, pollingInterval]);

  // Stop polling
  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      clearInterval(pollingRef.current);
      pollingRef.current = null;
      console.log('⏹️ Stopped real-time polling');
    }
  }, []);

  // Manual refresh
  const refresh = useCallback(() => {
    fetchLiveData();
  }, [fetchLiveData]);

  // Auto-start polling
  useEffect(() => {
    if (autoStart) {
      startPolling();
    }

    return () => {
      stopPolling();
    };
  }, [autoStart, startPolling, stopPolling]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      stopPolling();
    };
  }, [stopPolling]);

  return {
    ...data,
    startPolling,
    stopPolling,
    refresh,
    isPolling: pollingRef.current !== null
  };
};
