// Smooth Refresh Configuration
// Customize refresh intervals and behavior here

export const SMOOTH_REFRESH_CONFIG = {
  // Dashboard refresh settings
  dashboard: {
    interval: 10000, // 10 seconds
    enableSmoothTransitions: true,
    showLoadingIndicator: true,
    enableManualRefresh: true
  },
  
  // Positions refresh settings
  positions: {
    interval: 5000, // 5 seconds
    enableSmoothTransitions: true,
    showLoadingIndicator: true,
    enableManualRefresh: true
  },
  
  // Orders refresh settings
  orders: {
    interval: 10000, // 10 seconds
    enableSmoothTransitions: true,
    showLoadingIndicator: true,
    enableManualRefresh: true
  },
  
  // Signals refresh settings
  signals: {
    interval: 15000, // 15 seconds
    enableSmoothTransitions: true,
    showLoadingIndicator: true,
    enableManualRefresh: true
  },
  
  // Risk refresh settings
  risk: {
    interval: 10000, // 10 seconds
    enableSmoothTransitions: true,
    showLoadingIndicator: true,
    enableManualRefresh: true
  },
  
  // Margin refresh settings
  margin: {
    interval: 10000, // 10 seconds
    enableSmoothTransitions: true,
    showLoadingIndicator: true,
    enableManualRefresh: true
  },
  
  // Latency refresh settings
  latency: {
    interval: 5000, // 5 seconds
    enableSmoothTransitions: true,
    showLoadingIndicator: true,
    enableManualRefresh: true
  },
  
  // Funding refresh settings
  funding: {
    interval: 30000, // 30 seconds
    enableSmoothTransitions: true,
    showLoadingIndicator: true,
    enableManualRefresh: true
  },
  
  // Controls refresh settings
  controls: {
    interval: 10000, // 10 seconds
    enableSmoothTransitions: true,
    showLoadingIndicator: true,
    enableManualRefresh: true
  },
  
  // Alerts refresh settings
  alerts: {
    interval: 10000, // 10 seconds
    enableSmoothTransitions: true,
    showLoadingIndicator: true,
    enableManualRefresh: true
  },
  
  // Backtest refresh settings
  backtest: {
    interval: 30000, // 30 seconds
    enableSmoothTransitions: true,
    showLoadingIndicator: true,
    enableManualRefresh: true
  },
  
  // Real-time refresh settings
  realtime: {
    interval: 3000, // 3 seconds
    enableSmoothTransitions: true,
    showLoadingIndicator: true,
    enableManualRefresh: true
  },
  
  // Health refresh settings
  health: {
    interval: 5000, // 5 seconds
    enableSmoothTransitions: true,
    showLoadingIndicator: true,
    enableManualRefresh: true
  },
  
  // Config refresh settings (no auto-refresh)
  config: {
    interval: 0, // No auto-refresh
    enableSmoothTransitions: false,
    showLoadingIndicator: false,
    enableManualRefresh: true
  }
};

// Global smooth refresh settings
export const GLOBAL_SMOOTH_REFRESH_SETTINGS = {
  // Enable smooth transitions globally
  enableSmoothTransitions: true,
  
  // Default transition duration (in milliseconds)
  transitionDuration: 500,
  
  // Enable loading indicators globally
  showLoadingIndicators: true,
  
  // Enable manual refresh buttons globally
  showManualRefreshButtons: true,
  
  // Enable error handling
  enableErrorHandling: true,
  
  // Enable console logging for debugging
  enableConsoleLogging: true,
  
  // Maximum retry attempts for failed requests
  maxRetryAttempts: 3,
  
  // Retry delay (in milliseconds)
  retryDelay: 1000
};

// Helper function to get config for a specific tab
export const getTabConfig = (tabId: string) => {
  return SMOOTH_REFRESH_CONFIG[tabId as keyof typeof SMOOTH_REFRESH_CONFIG] || SMOOTH_REFRESH_CONFIG.dashboard;
};

// Helper function to check if smooth transitions are enabled
export const isSmoothTransitionsEnabled = (tabId: string) => {
  const config = getTabConfig(tabId);
  return config.enableSmoothTransitions && GLOBAL_SMOOTH_REFRESH_SETTINGS.enableSmoothTransitions;
};

// Helper function to check if loading indicators should be shown
export const shouldShowLoadingIndicator = (tabId: string) => {
  const config = getTabConfig(tabId);
  return config.showLoadingIndicator && GLOBAL_SMOOTH_REFRESH_SETTINGS.showLoadingIndicators;
};
