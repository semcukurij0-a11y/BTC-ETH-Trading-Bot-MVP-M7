import { useState, useEffect, useCallback, useRef } from 'react';
import { TradingBotService } from '../services/TradingBotService';

interface SmoothRefreshOptions {
  refreshInterval: number;
  autoStart?: boolean;
  onDataUpdate?: (data: any) => void;
  onError?: (error: string) => void;
  enableSmartRefresh?: boolean;
}

interface SmoothRefreshData {
  data: any;
  isLoading: boolean;
  isRefreshing: boolean;
  error: string | null;
  lastUpdate: string | null;
  hasChanges: boolean;
  changeCount: number;
}

interface ChangeDetection {
  hasChanges: boolean;
  changedFields: string[];
  changeDetails: any;
}

export const useSmoothRefresh = (
  service: TradingBotService,
  dataFetcher: () => Promise<any>,
  options: SmoothRefreshOptions
) => {
  const {
    refreshInterval,
    autoStart = true,
    onDataUpdate,
    onError,
    enableSmartRefresh = true
  } = options;

  const [refreshData, setRefreshData] = useState<SmoothRefreshData>({
    data: null,
    isLoading: false,
    isRefreshing: false,
    error: null,
    lastUpdate: null,
    hasChanges: false,
    changeCount: 0
  });

  const intervalRef = useRef<NodeJS.Timeout | null>(null);
  const isActiveRef = useRef(true);
  const previousDataRef = useRef<any>(null);
  const changeDetectionRef = useRef<ChangeDetection>({
    hasChanges: false,
    changedFields: [],
    changeDetails: null
  });

  // Smart change detection function
  const detectChanges = useCallback((newData: any, previousData: any): ChangeDetection => {
    if (!enableSmartRefresh || !previousData) {
      return { hasChanges: true, changedFields: ['initial'], changeDetails: null };
    }

    const changes: string[] = [];
    const changeDetails: any = {};

    // Deep comparison for arrays (positions, orders, etc.)
    if (Array.isArray(newData) && Array.isArray(previousData)) {
      if (newData.length !== previousData.length) {
        changes.push('count');
        changeDetails.count = { from: previousData.length, to: newData.length };
      }

      // Compare each item in the array
      newData.forEach((newItem, index) => {
        const prevItem = previousData[index];
        if (prevItem) {
          Object.keys(newItem).forEach(key => {
            if (newItem[key] !== prevItem[key]) {
              changes.push(`${index}.${key}`);
              if (!changeDetails[index]) changeDetails[index] = {};
              changeDetails[index][key] = { from: prevItem[key], to: newItem[key] };
            }
          });
        }
      });
    } else if (typeof newData === 'object' && typeof previousData === 'object') {
      // Deep comparison for objects
      const allKeys = new Set([...Object.keys(newData), ...Object.keys(previousData)]);
      allKeys.forEach(key => {
        if (newData[key] !== previousData[key]) {
          changes.push(key);
          changeDetails[key] = { from: previousData[key], to: newData[key] };
        }
      });
    } else {
      // Simple comparison for primitives
      if (newData !== previousData) {
        changes.push('value');
        changeDetails.value = { from: previousData, to: newData };
      }
    }

    return {
      hasChanges: changes.length > 0,
      changedFields: changes,
      changeDetails
    };
  }, [enableSmartRefresh]);

  // Smooth data update function with change detection
  const updateData = useCallback((newData: any) => {
    if (!isActiveRef.current) return;

    // Detect changes
    const changeDetection = detectChanges(newData, previousDataRef.current);
    const hasChanges = changeDetection.hasChanges;

    console.log('🔄 Smooth refresh - Data updated:', {
      newData,
      dataType: typeof newData,
      isArray: Array.isArray(newData),
      hasPositions: newData?.positions ? 'Yes' : 'No',
      positionsLength: newData?.positions?.length || 0,
      hasChanges,
      changedFields: changeDetection.changedFields,
      changeDetails: changeDetection.changeDetails
    });

    // Update change detection ref
    changeDetectionRef.current = changeDetection;

    setRefreshData(prev => ({
      ...prev,
      data: newData,
      isLoading: false,
      isRefreshing: false,
      error: null,
      lastUpdate: new Date().toISOString(),
      hasChanges,
      changeCount: prev.changeCount + (hasChanges ? 1 : 0)
    }));

    // Store current data as previous for next comparison
    previousDataRef.current = newData;

    if (onDataUpdate) {
      onDataUpdate(newData);
    }
  }, [onDataUpdate, detectChanges]);

  // Smooth error handling
  const handleError = useCallback((error: string) => {
    if (!isActiveRef.current) return;

    setRefreshData(prev => ({
      ...prev,
      isLoading: false,
      isRefreshing: false,
      error
    }));

    if (onError) {
      onError(error);
    }
  }, [onError]);

  // Fetch data smoothly with smart change detection
  const fetchData = useCallback(async (isManualRefresh = false) => {
    if (!isActiveRef.current) return;

    console.log('🔄 Smooth refresh - Fetching data...', { isManualRefresh, enableSmartRefresh });

    // Set refreshing state only for manual refreshes
    if (isManualRefresh) {
      setRefreshData(prev => ({ ...prev, isRefreshing: true, error: null }));
    } else {
      setRefreshData(prev => ({ ...prev, isLoading: true, error: null }));
    }

    try {
      const data = await dataFetcher();
      console.log('🔄 Smooth refresh - Data fetched:', {
        data,
        dataType: typeof data,
        isArray: Array.isArray(data),
        hasPositions: data?.positions ? 'Yes' : 'No',
        positionsLength: data?.positions?.length || 0
      });

      // Smart update: only update if changes detected or manual refresh
      if (isManualRefresh || !enableSmartRefresh) {
        updateData(data);
      } else {
        // Check for changes before updating
        const changeDetection = detectChanges(data, previousDataRef.current);
        if (changeDetection.hasChanges) {
          console.log('🔄 Smart refresh - Changes detected, updating UI:', {
            changedFields: changeDetection.changedFields,
            changeDetails: changeDetection.changeDetails
          });
          updateData(data);
        } else {
          console.log('🔄 Smart refresh - No changes detected, skipping UI update');
          // Still update the lastUpdate timestamp but don't trigger UI updates
          setRefreshData(prev => ({
            ...prev,
            isLoading: false,
            isRefreshing: false,
            error: null,
            lastUpdate: new Date().toISOString(),
            hasChanges: false
          }));
        }
      }
    } catch (error) {
      console.error('🔄 Smooth refresh - Fetch error:', error);
      const errorMessage = error instanceof Error ? error.message : 'Failed to fetch data';
      handleError(errorMessage);
    }
  }, [dataFetcher, updateData, handleError, enableSmartRefresh, detectChanges]);

  // Manual refresh function
  const refresh = useCallback(() => {
    fetchData(true);
  }, [fetchData]);

  // Start smooth polling
  const startPolling = useCallback(() => {
    if (intervalRef.current) return; // Already polling

    isActiveRef.current = true;
    
    // Initial fetch
    fetchData();
    
    // Set up smooth polling
    if (refreshInterval > 0) {
      intervalRef.current = setInterval(() => {
        if (isActiveRef.current) {
          fetchData();
        }
      }, refreshInterval);
    }
  }, [fetchData, refreshInterval]);

  // Stop polling
  const stopPolling = useCallback(() => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    isActiveRef.current = false;
  }, []);

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
    ...refreshData,
    refresh,
    startPolling,
    stopPolling,
    isPolling: intervalRef.current !== null,
    changeDetection: changeDetectionRef.current,
    previousData: previousDataRef.current
  };
};

// Specialized hook for dashboard data
export const useDashboardSmoothRefresh = (
  service: TradingBotService,
  refreshInterval: number = 10000
) => {
  const dataFetcher = useCallback(async () => {
    const [tradingStats, equityCurve] = await Promise.all([
      service.getTradingStats(),
      service.getEquityCurve()
    ]);
    
    return {
      tradingStats,
      equityCurve
    };
  }, [service]);

  return useSmoothRefresh(service, dataFetcher, {
    refreshInterval,
    autoStart: true,
    enableSmartRefresh: true
  });
};

// Specialized hook for positions data
export const usePositionsSmoothRefresh = (
  service: TradingBotService,
  refreshInterval: number = 5000
) => {
  const dataFetcher = useCallback(async () => {
    console.log('🔄 usePositionsSmoothRefresh - Fetching positions...');
    const positions = await service.getPositions();
    console.log('🔄 usePositionsSmoothRefresh - Positions fetched:', {
      positions,
      length: positions?.length || 0,
      type: typeof positions,
      isArray: Array.isArray(positions)
    });
    return positions;
  }, [service]);

  return useSmoothRefresh(service, dataFetcher, {
    refreshInterval,
    autoStart: true,
    enableSmartRefresh: true
  });
};

// Specialized hook for orders data
export const useOrdersSmoothRefresh = (
  service: TradingBotService,
  refreshInterval: number = 10000
) => {
  const dataFetcher = useCallback(async () => {
    return await service.getOrderHistory();
  }, [service]);

  return useSmoothRefresh(service, dataFetcher, {
    refreshInterval,
    autoStart: true,
    enableSmartRefresh: true
  });
};

// Specialized hook for signals data
export const useSignalsSmoothRefresh = (
  service: TradingBotService,
  refreshInterval: number = 15000
) => {
  const dataFetcher = useCallback(async () => {
    console.log('[INFO] useSignalsSmoothRefresh - Fetching signals...');
    try {
      const signals = await service.getCurrentSignals();
      console.log('[INFO] useSignalsSmoothRefresh - Signals fetched:', {
        signals,
        length: signals?.length || 0,
        type: typeof signals,
        isArray: Array.isArray(signals),
        firstSignal: signals?.[0] || null
      });
      
      // Additional debugging for signal structure
      if (signals && signals.length > 0) {
        const firstSignal = signals[0];
        console.log('[INFO] useSignalsSmoothRefresh - First signal details:', {
          symbol: firstSignal.symbol,
          signal: firstSignal.signal,
          confidence: firstSignal.confidence,
          components: firstSignal.components,
          hasSignal: 'signal' in firstSignal,
          hasComponents: 'components' in firstSignal,
          componentsKeys: firstSignal.components ? Object.keys(firstSignal.components) : []
        });
      } else {
        console.log('[WARNING] useSignalsSmoothRefresh - No signals received or empty array');
        console.log('[WARNING] Signals data:', {
          signals,
          isNull: signals === null,
          isUndefined: signals === undefined,
          isEmpty: Array.isArray(signals) && signals.length === 0
        });
      }
      
      return signals;
    } catch (error) {
      console.error('[ERROR] useSignalsSmoothRefresh - Error fetching signals:', error);
      console.error('[ERROR] useSignalsSmoothRefresh - Error details:', {
        message: error.message,
        stack: error.stack,
        name: error.name
      });
      throw error;
    }
  }, [service]);

  return useSmoothRefresh(service, dataFetcher, {
    refreshInterval,
    autoStart: true,
    enableSmartRefresh: true
  });
};
