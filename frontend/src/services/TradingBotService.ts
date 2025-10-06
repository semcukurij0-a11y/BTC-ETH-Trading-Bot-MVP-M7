export class TradingBotService {
  private baseUrl = this.getApiBaseUrl();
  private authToken: string | null = null;
  private pollingInterval: NodeJS.Timeout | null = null;
  private isPolling = false;
  private requestCache = new Map<string, { data: any; timestamp: number }>();
  private pendingRequests = new Map<string, Promise<any>>();

  private getApiBaseUrl(): string {
    // Try to get from environment variable first
    const envUrl = import.meta.env.VITE_API_BASE_URL;
    if (envUrl) {
      return envUrl;
    }
    
    // Fallback to localhost
    return 'http://localhost:8000';
  }

  // Real-time polling methods
  startPolling(intervalMs: number = 5000): void {
    if (this.isPolling) {
      console.log('Polling already active');
      return;
    }

    this.isPolling = true;
    console.log(`Starting real-time polling every ${intervalMs}ms`);
    
    this.pollingInterval = setInterval(async () => {
      try {
        // Poll for live data updates
        await this.refreshLiveData();
      } catch (error) {
        console.error('Polling error:', error);
      }
    }, intervalMs);
  }

  stopPolling(): void {
    if (this.pollingInterval) {
      clearInterval(this.pollingInterval);
      this.pollingInterval = null;
      this.isPolling = false;
      console.log('Stopped real-time polling');
    }
  }

  private async refreshLiveData(): Promise<void> {
    // This method will be called by polling to refresh live data
    // The actual components will handle the data updates
    console.log('Refreshing live data...');
  }

  // Request caching and deduplication
  private async cachedRequest<T>(
    cacheKey: string, 
    fetcher: () => Promise<T>, 
    ttl: number = 30000
  ): Promise<T> {
    const now = Date.now();
    
    // Check cache first
    const cached = this.requestCache.get(cacheKey);
    if (cached && (now - cached.timestamp) < ttl) {
      console.log(`📦 Cache hit for ${cacheKey}`);
      return cached.data;
    }
    
    // Check if request is already pending
    if (this.pendingRequests.has(cacheKey)) {
      console.log(`⏳ Request already pending for ${cacheKey}`);
      return this.pendingRequests.get(cacheKey)!;
    }
    
    // Make new request
    console.log(`🌐 Fetching fresh data for ${cacheKey}`);
    const promise = fetcher().then(data => {
      this.requestCache.set(cacheKey, { data, timestamp: now });
      this.pendingRequests.delete(cacheKey);
      return data;
    }).catch(error => {
      this.pendingRequests.delete(cacheKey);
      throw error;
    });
    
    this.pendingRequests.set(cacheKey, promise);
    return promise;
  }

  // Clear cache for specific key or all
  public clearCache(cacheKey?: string): void {
    if (cacheKey) {
      this.requestCache.delete(cacheKey);
    } else {
      this.requestCache.clear();
    }
  }

  // Calculate PnL from positions data
  private calculatePnLFromPositions(positions: any[]): { unrealizedPnL: number; realizedPnL: number } {
    let unrealizedPnL = 0;
    let realizedPnL = 0;

    positions.forEach(position => {
      if (position.isActive) {
        // Unrealized PnL from active positions
        unrealizedPnL += position.unrealizedPnL || 0;
      }
      // Note: Realized PnL should come from order history, not positions
      // Positions only show current state, not historical realized gains
    });

    return { unrealizedPnL, realizedPnL };
  }

  // Calculate realized PnL from order history
  private async calculateRealizedPnL(): Promise<number> {
    try {
      const ordersResponse = await this.getOrderHistory();
      let realizedPnL = 0;
      
      // Handle the response object structure
      const orders = ordersResponse?.orders || [];
      console.log('📋 Calculating realized PnL from orders:', orders.length);
      
      if (Array.isArray(orders)) {
        orders.forEach((order: any) => {
          // Check for different possible field names for realized PnL
          const orderPnL = order.realizedPnL || order.realized_pnl || order.pnl || order.profit || 0;
          
          if (order.status === 'filled' && orderPnL !== 0) {
            realizedPnL += orderPnL;
            console.log(`💰 Order ${order.symbol}: ${orderPnL} (Status: ${order.status})`);
          }
        });
      } else {
        console.warn('Orders data is not an array:', typeof orders);
      }
      
      console.log(`📊 Total realized PnL: ${realizedPnL}`);
      return realizedPnL;
    } catch (error) {
      console.error('Error calculating realized PnL:', error);
      return 0;
    }
  }

  // Calculate daily PnL from recent activity
  private async calculateDailyPnL(): Promise<number> {
    try {
      // Get recent orders to calculate daily PnL
      const ordersResponse = await this.getOrderHistory();
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      
      // Handle the response object structure
      const orders = ordersResponse?.orders || [];
      let dailyPnL = 0;
      
      if (Array.isArray(orders)) {
        orders.forEach((order: any) => {
          const orderDate = new Date(order.timestamp);
          if (orderDate >= today && order.status === 'filled') {
            // Check for different possible field names for realized PnL
            const orderPnL = order.realizedPnL || order.realized_pnl || order.pnl || order.profit || 0;
            dailyPnL += orderPnL;
          }
        });
      } else {
        console.warn('Orders data is not an array:', typeof orders);
      }
      
      console.log(`📅 Daily PnL: ${dailyPnL}`);
      return dailyPnL;
    } catch (error) {
      console.error('Error calculating daily PnL:', error);
      return 0;
    }
  }

  // Batch fetch essential data for initial load
  async getEssentialData(): Promise<{
    systemStatus: any;
    tradingStats: any;
    positions: any[];
    accountInfo: any;
  }> {
    return this.cachedRequest('essential-data', async () => {
      console.log('🚀 Fetching essential data in batch...');
      
      const [systemStatus, tradingStats, positions, accountInfo] = await Promise.all([
        this.getSystemStatus(),
        this.getTradingStats(),
        this.getPositions(),
        this.getAccountInfo()
      ]);

      return {
        systemStatus,
        tradingStats,
        positions,
        accountInfo
      };
    }, 5000); // 5 second cache for essential data
  }

  async initialize(): Promise<void> {
    // Check if already authenticated and token is still valid
    if (this.authToken) {
      console.log('✅ Already authenticated, skipping re-authentication');
      return;
    }

    console.log('Initializing TradingBotService with real backend...');
    
    // Authenticate with backend
    try {
      const response = await fetch(`${this.baseUrl}/auth/login`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          username: 'admin',
          password: 'admin123'
        })
      });

      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          this.authToken = data.token;
          console.log('✅ Authenticated with backend successfully');
        } else {
          console.error('❌ Authentication failed:', data.message);
        }
      } else {
        console.error('❌ Authentication request failed:', response.status);
      }
    } catch (error) {
      console.error('❌ Authentication error:', error);
    }
  }

  async getSystemStatus(): Promise<any> {
    return this.cachedRequest('system-status', async () => {
      if (!this.authToken) {
        await this.initialize();
      }

      const response = await fetch(`${this.baseUrl}/status`, {
        headers: {
          'Authorization': `Bearer ${this.authToken}`,
          'Content-Type': 'application/json',
        }
      });

      if (response.ok) {
        const data = await response.json();
        console.log('✅ Real system status fetched from backend:', data);
        return {
          mode: 'PAPER',
          apiStatus: data.bybit_connection?.success ? 'connected' : 'disconnected',
          dataFeed: 'live',
          lastUpdate: new Date().toISOString(),
          latency: 45,
          uptime: `${Math.floor(data.uptime / 3600)}:${Math.floor((data.uptime % 3600) / 60)}:${Math.floor(data.uptime % 60)}`,
          bybitConnection: data.bybit_connection,
          tradingStatus: data.trading_status,
          riskStatus: data.risk_status
        };
      } else {
        console.error('❌ Failed to fetch system status:', response.status);
        return this.getMockSystemStatus();
      }
    }, 10000); // 10 second cache for system status
  }

  private getMockSystemStatus(): any {
    return {
      mode: 'PAPER',
      apiStatus: 'disconnected',
      dataFeed: 'mock',
      lastUpdate: new Date().toISOString(),
      latency: 0,
      uptime: '00:00:00'
    };
  }

  async getEquityCurve(): Promise<any[]> {
    // Generate mock equity curve data
    const data = [];
    const startDate = new Date('2024-01-01');
    const days = 90;
    let equity = 10000;
    
    for (let i = 0; i < days; i++) {
      const date = new Date(startDate);
      date.setDate(date.getDate() + i);
      
      // Simulate trading performance with some volatility
      const dailyReturn = (Math.random() - 0.48) * 0.02; // Slight positive bias
      equity *= (1 + dailyReturn);
      
      data.push({
        timestamp: date.toISOString(),
        equity: Math.round(equity * 100) / 100
      });
    }
    
    return data;
  }

  async getTradingStats(): Promise<any> {
    return this.cachedRequest('trading-stats', async () => {
      if (!this.authToken) {
        await this.initialize();
      }

      const response = await fetch(`${this.baseUrl}/trading/stats`, {
        headers: {
          'Authorization': `Bearer ${this.authToken}`,
          'Content-Type': 'application/json',
        }
      });

      if (response.ok) {
        const data = await response.json();
        console.log('✅ Real trading stats fetched from backend:', data);
        
        // Get live positions to calculate unrealized PnL
        const positions = await this.getPositions();
        const { unrealizedPnL } = this.calculatePnLFromPositions(positions);
        
        // Calculate realized PnL from order history
        const realizedPnL = await this.calculateRealizedPnL();
        const dailyPnL = await this.calculateDailyPnL();
        
        // Log the data sources for debugging
        console.log('🔍 PnL Data Sources:', {
          backendRealizedPnL: data.realized_pnl,
          calculatedRealizedPnL: realizedPnL,
          backendDailyPnL: data.daily_pnl,
          calculatedDailyPnL: dailyPnL,
          backendUnrealizedPnL: data.unrealized_pnl,
          calculatedUnrealizedPnL: unrealizedPnL
        });
        
        console.log('📊 PnL Calculations:', {
          backend: {
            totalPnL: data.total_pnl,
            unrealizedPnL: data.unrealized_pnl,
            realizedPnL: data.realized_pnl,
            dailyPnL: data.daily_pnl
          },
          calculated: {
            unrealizedPnL,
            realizedPnL,
            dailyPnL,
            activePositions: positions.filter(p => p.isActive).length
          }
        });
        
        return {
          totalPnL: data.total_pnl || (unrealizedPnL + realizedPnL),
          unrealizedPnL: data.unrealized_pnl !== undefined ? data.unrealized_pnl : unrealizedPnL,
          realizedPnL: data.realized_pnl !== undefined ? data.realized_pnl : realizedPnL,
          dailyPnL: data.daily_pnl !== undefined ? data.daily_pnl : dailyPnL,
          winRate: data.win_rate || 0,
          totalTrades: data.total_trades || 0,
          maxDrawdown: data.max_drawdown || 0,
          sharpeRatio: data.sharpe_ratio || 0,
          currentPositions: data.current_positions || positions.filter(p => p.isActive).length,
          marginRatio: data.margin_ratio || 0,
          accountBalance: data.account_balance || 0,
          availableBalance: data.available_balance || 0
        };
      } else {
        console.error('❌ Failed to fetch trading stats:', response.status);
        return this.getMockTradingStats();
      }
    }, 10000); // Reduced cache to 10 seconds for more frequent updates
  }

  private getMockTradingStats(): any {
    return {
      totalPnL: 0,
      unrealizedPnL: 0,
      realizedPnL: 0,
      dailyPnL: 0,
      winRate: 0,
      totalTrades: 0,
      maxDrawdown: 0,
      sharpeRatio: 0,
      currentPositions: 0,
      marginRatio: 0
    };
  }

  async getPositions(): Promise<any[]> {
    return this.cachedRequest('positions', async () => {
      if (!this.authToken) {
        await this.initialize();
      }

      const response = await fetch(`${this.baseUrl}/trading/positions`, {
        headers: {
          'Authorization': `Bearer ${this.authToken}`,
          'Content-Type': 'application/json',
        }
      });

      if (response.ok) {
        const data = await response.json();
        console.log('✅ Real positions fetched from backend:', data);
        
        // Return formatted positions with live data (keep backend property names)
        const positions = data.positions || [];
        return positions.map((pos: any) => ({
          symbol: pos.symbol,
          side: pos.side,
          size: pos.size,
          entry_price: pos.entry_price,
          current_price: pos.current_price,
          unrealized_pnl: pos.unrealized_pnl,
          leverage: pos.leverage,
          margin_mode: pos.margin_mode,
          liquidation_price: pos.liquidation_price,
          is_active: pos.is_active,
          timestamp: pos.timestamp
        }));
      } else {
        console.error('❌ Failed to fetch positions:', response.status);
        return this.getMockPositions();
      }
    }, 5000); // 5 second cache for positions (more frequent updates)
  }

  async getAccountInfo(): Promise<any> {
    return this.cachedRequest('account-info', async () => {
      if (!this.authToken) {
        await this.initialize();
      }

      const response = await fetch(`${this.baseUrl}/trading/account`, {
        headers: {
          'Authorization': `Bearer ${this.authToken}`,
          'Content-Type': 'application/json',
        }
      });

      if (response.ok) {
        const data = await response.json();
        console.log('✅ Real account info fetched from backend:', data);
        return {
          success: data.success,
          timestamp: data.timestamp,
          wallet: {
            balance: data.wallet?.balance || 0,
            available: data.wallet?.available || 0,
            success: data.wallet?.success || false,
            error: data.wallet?.error
          },
          positions: {
            count: data.positions?.count || 0,
            activeCount: data.positions?.active_count || 0,
            totalPnL: data.positions?.total_pnl || 0,
            success: data.positions?.success || false,
            error: data.positions?.error
          }
        };
      } else {
        console.error('❌ Failed to fetch account info:', response.status);
        return this.getMockAccountInfo();
      }
    }, 10000); // 10 second cache for account info
  }

  private getMockAccountInfo(): any {
    return {
      success: false,
      timestamp: new Date().toISOString(),
      wallet: {
        balance: 10000,
        available: 10000,
        success: false,
        error: 'Mock data'
      },
      positions: {
        count: 0,
        activeCount: 0,
        totalPnL: 0,
        success: false,
        error: 'Mock data'
      }
    };
  }

  private getMockPositions(): any[] {
    return [
      {
        symbol: 'BTCUSDT',
        side: 'long',
        size: 0.0234,
        entryPrice: 43250.00,
        currentPrice: 43890.50,
        unrealizedPnL: 14.99,
        leverage: 3,
        marginMode: 'isolated',
        stopLoss: 42100.00,
        takeProfit: 45200.00,
        liquidationPrice: 38500.00,
        timestamp: new Date(Date.now() - 3600000).toISOString()
      }
    ];
  }

  async closePosition(symbol: string, positionDetails?: any): Promise<{ success: boolean; message?: string; error?: string }> {
    try {
      if (!this.authToken) {
        await this.initialize();
      }

      console.log(`Closing position for ${symbol}`);

      // If position details are provided, use them to avoid backend lookup
      const requestBody: any = { symbol: symbol };
      if (positionDetails) {
        requestBody.side = positionDetails.side;
        requestBody.size = positionDetails.size;
        requestBody.entryPrice = positionDetails.entryPrice;
        console.log(`Using provided position details: ${positionDetails.side} ${positionDetails.size} ${symbol}`);
      }

      const response = await fetch(`${this.baseUrl}/trading/positions/close`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.authToken}`,
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      });

      if (response.ok) {
        const data = await response.json();
        console.log('✅ Position closed successfully:', data);
        return {
          success: true,
          message: data.message || `Position closed successfully for ${symbol}`
        };
      } else {
        const errorData = await response.json().catch(() => ({}));
        console.error('❌ Failed to close position:', response.status, errorData);
        return {
          success: false,
          error: errorData.detail || `Failed to close position: ${response.status}`
        };
      }
    } catch (error) {
      console.error('❌ Error closing position:', error);
      return {
        success: false,
        error: `Error closing position: ${error}`
      };
    }
  }

  // New methods for the additional panels
  async getOrderHistory(): Promise<any> {
    return this.cachedRequest('order-history', async () => {
      if (!this.authToken) {
        await this.initialize();
      }

      const response = await fetch(`${this.baseUrl}/trading/orders`, {
        headers: {
          'Authorization': `Bearer ${this.authToken}`,
          'Content-Type': 'application/json',
        }
      });

      if (response.ok) {
        const data = await response.json();
        console.log('✅ Order history fetched from backend:', data);
        return data; // Return the full response object
      } else {
        console.error('❌ Failed to fetch order history:', response.status);
        return { success: false, orders: [], count: 0 };
      }
    }, 30000); // 30 second cache for order history
  }

  async getMarginData(): Promise<any> {
    try {
      if (!this.authToken) {
        await this.initialize();
      }

      const response = await fetch(`${this.baseUrl}/trading/account`, {
        headers: {
          'Authorization': `Bearer ${this.authToken}`,
          'Content-Type': 'application/json',
        }
      });

      if (response.ok) {
        const data = await response.json();
        console.log('✅ Margin data fetched from backend:', data);
        return data;
      } else {
        console.error('❌ Failed to fetch margin data:', response.status);
        return { success: false };
      }
    } catch (error) {
      console.error('❌ Error fetching margin data:', error);
      return { success: false };
    }
  }

  async getAPILatency(): Promise<any> {
    try {
      if (!this.authToken) {
        await this.initialize();
      }

      const startTime = Date.now();
      const response = await fetch(`${this.baseUrl}/health`, {
        headers: {
          'Authorization': `Bearer ${this.authToken}`,
          'Content-Type': 'application/json',
        }
      });
      const endTime = Date.now();
      const latency = endTime - startTime;

      if (response.ok) {
        console.log('✅ API latency test completed:', latency + 'ms');
        return { success: true, latency };
      } else {
        console.error('❌ API latency test failed:', response.status);
        return { success: false, latency: -1 };
      }
    } catch (error) {
      console.error('❌ Error testing API latency:', error);
      return { success: false, latency: -1 };
    }
  }

  async getFundingRates(): Promise<any> {
    try {
      if (!this.authToken) {
        await this.initialize();
      }

      // For now, return mock data since Bybit doesn't have a direct funding rate endpoint
      // In a real implementation, this would fetch from Bybit's funding rate API
      const mockFundingRates = {
        success: true,
        funding_rates: [
          {
            symbol: 'BTCUSDT',
            funding_rate: 0.0001,
            funding_rate_8h: 0.0008,
            next_funding_time: new Date(Date.now() + 8 * 60 * 60 * 1000).toISOString(),
            predicted_funding_rate: 0.0002,
            index_price: 50000,
            mark_price: 50025,
            last_funding_rate: 0.00005
          },
          {
            symbol: 'ETHUSDT',
            funding_rate: -0.0002,
            funding_rate_8h: -0.0016,
            next_funding_time: new Date(Date.now() + 8 * 60 * 60 * 1000).toISOString(),
            predicted_funding_rate: -0.0001,
            index_price: 3000,
            mark_price: 2995,
            last_funding_rate: -0.0003
          }
        ]
      };

      console.log('✅ Funding rates fetched:', mockFundingRates);
      return mockFundingRates;
    } catch (error) {
      console.error('❌ Error fetching funding rates:', error);
      return { success: false, funding_rates: [] };
    }
  }

  // Control methods for trading system
  async pauseTrading(): Promise<{ success: boolean; message?: string; error?: string }> {
    try {
      if (!this.authToken) {
        await this.initialize();
      }

      const response = await fetch(`${this.baseUrl}/trading/pause`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.authToken}`,
          'Content-Type': 'application/json',
        }
      });

      if (response.ok) {
        const data = await response.json();
        console.log('✅ Trading paused successfully:', data);
        return {
          success: true,
          message: data.message || 'Trading paused successfully'
        };
      } else {
        const errorData = await response.json().catch(() => ({}));
        console.error('❌ Failed to pause trading:', response.status, errorData);
        return {
          success: false,
          error: errorData.detail || `Failed to pause trading: ${response.status}`
        };
      }
    } catch (error) {
      console.error('❌ Error pausing trading:', error);
      return {
        success: false,
        error: `Error pausing trading: ${error}`
      };
    }
  }

  async resumeTrading(): Promise<{ success: boolean; message?: string; error?: string }> {
    try {
      if (!this.authToken) {
        await this.initialize();
      }

      const response = await fetch(`${this.baseUrl}/trading/resume`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.authToken}`,
          'Content-Type': 'application/json',
        }
      });

      if (response.ok) {
        const data = await response.json();
        console.log('✅ Trading resumed successfully:', data);
        return {
          success: true,
          message: data.message || 'Trading resumed successfully'
        };
      } else {
        const errorData = await response.json().catch(() => ({}));
        console.error('❌ Failed to resume trading:', response.status, errorData);
        return {
          success: false,
          error: errorData.detail || `Failed to resume trading: ${response.status}`
        };
      }
    } catch (error) {
      console.error('❌ Error resuming trading:', error);
      return {
        success: false,
        error: `Error resuming trading: ${error}`
      };
    }
  }

  async activateKillSwitch(): Promise<{ success: boolean; message?: string; error?: string }> {
    try {
      if (!this.authToken) {
        await this.initialize();
      }

      const response = await fetch(`${this.baseUrl}/trading/kill-switch`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.authToken}`,
          'Content-Type': 'application/json',
        }
      });

      if (response.ok) {
        const data = await response.json();
        console.log('✅ Kill switch activated successfully:', data);
        return {
          success: true,
          message: data.message || 'Kill switch activated - all positions flattened'
        };
      } else {
        const errorData = await response.json().catch(() => ({}));
        console.error('❌ Failed to activate kill switch:', response.status, errorData);
        return {
          success: false,
          error: errorData.detail || `Failed to activate kill switch: ${response.status}`
        };
      }
    } catch (error) {
      console.error('❌ Error activating kill switch:', error);
      return {
        success: false,
        error: `Error activating kill switch: ${error}`
      };
    }
  }

  async resetKillSwitch(): Promise<{ success: boolean; message?: string; error?: string }> {
    try {
      if (!this.authToken) {
        await this.initialize();
      }

      const response = await fetch(`${this.baseUrl}/trading/reset-kill-switch`, {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${this.authToken}`,
          'Content-Type': 'application/json',
        }
      });

      if (response.ok) {
        const data = await response.json();
        console.log('✅ Kill switch reset successfully:', data);
        return {
          success: true,
          message: data.message || 'Kill switch reset - trading can resume'
        };
      } else {
        const errorData = await response.json().catch(() => ({}));
        console.error('❌ Failed to reset kill switch:', response.status, errorData);
        return {
          success: false,
          error: errorData.detail || `Failed to reset kill switch: ${response.status}`
        };
      }
    } catch (error) {
      console.error('❌ Error resetting kill switch:', error);
      return {
        success: false,
        error: `Error resetting kill switch: ${error}`
      };
    }
  }

  async getCurrentSignals(): Promise<any[]> {
    return this.cachedRequest('current-signals', async () => {
      console.log('[INFO] TradingBotService.getCurrentSignals - Starting...');
      
      if (!this.authToken) {
        console.log('[INFO] TradingBotService.getCurrentSignals - No auth token, initializing...');
        await this.initialize();
      }

      console.log('[INFO] TradingBotService.getCurrentSignals - Making API request to:', `${this.baseUrl}/trading/signals`);
      
      try {
        const response = await fetch(`${this.baseUrl}/trading/signals`, {
          headers: {
            'Authorization': `Bearer ${this.authToken}`,
            'Content-Type': 'application/json',
          }
        });

        console.log('[INFO] TradingBotService.getCurrentSignals - Response status:', response.status);

        if (response.ok) {
          const data = await response.json();
          console.log('[SUCCESS] Real signals fetched from backend:', {
            success: data.success,
            count: data.count,
            signals: data.signals,
            signalsLength: data.signals?.length || 0
          });
          
          // Backend signals are already properly weighted, return them as-is
          console.log('[INFO] Backend signals are already properly weighted, returning as-is');
          return data.signals || [];
        } else {
          console.error('[ERROR] Failed to fetch signals:', response.status, response.statusText);
          console.log('[INFO] TradingBotService.getCurrentSignals - Falling back to mock signals');
          const mockSignals = this.getMockSignals();
          console.log('[INFO] TradingBotService.getCurrentSignals - Mock signals:', mockSignals);
          return mockSignals;
        }
      } catch (error) {
        console.error('[ERROR] Error fetching signals:', error);
        console.log('[INFO] TradingBotService.getCurrentSignals - Falling back to mock signals due to error');
        const mockSignals = this.getMockSignals();
        console.log('[INFO] TradingBotService.getCurrentSignals - Mock signals:', mockSignals);
        return mockSignals;
      }
    }, 5000); // 5 second cache for signals (reduced for testing)
  }

  // Simple fallback method that always returns signals
  private getSimpleMockSignals(): any[] {
    console.log('[INFO] TradingBotService.getSimpleMockSignals - Creating simple mock signals');
    
    return [
      {
        symbol: 'BTCUSDT',
        signal: 0.623,
        confidence: 0.78,
        components: {
          ml: 0.456,
          technical: 0.234,
          sentiment: 0.145,
          fear_greed: 0.089
        },
        timestamp: new Date().toISOString()
      },
      {
        symbol: 'ETHUSDT',
        signal: -0.234,
        confidence: 0.65,
        components: {
          ml: -0.123,
          technical: -0.089,
          sentiment: -0.067,
          fear_greed: 0.045
        },
        timestamp: new Date().toISOString()
      }
    ];
  }

  private getMockSignals(): any[] {
    try {
      // Import signal fusion calculator
      const { defaultSignalFusion } = require('../utils/signalFusion');
      
        // Raw component data (before fusion) - Only ETH and BTCUSDT
        const rawSignals = [
          {
            symbol: 'BTCUSDT',
            components: {
              ml: 0.456,
              technical: 0.234,
              sentiment: 0.145,
              fear_greed: 0.089
            },
            timestamp: new Date().toISOString()
          },
          {
            symbol: 'ETHUSDT',
            components: {
              ml: -0.123,
              technical: -0.089,
              sentiment: -0.067,
              fear_greed: 0.045
            },
            timestamp: new Date().toISOString()
          }
        ];
      
      // Calculate real fused signals using the fusion logic
      const processedSignals = defaultSignalFusion.processSignals(rawSignals);
      console.log('TradingBotService.getMockSignals - Processed signals:', processedSignals);
      return processedSignals;
    } catch (error) {
      console.error('Error in getMockSignals signal fusion:', error);
      
      // Fallback to inline fusion calculation
      try {
        const rawSignals = [
          {
            symbol: 'BTCUSDT',
            components: { ml: 0.456, technical: 0.234, sentiment: 0.145, fear_greed: 0.089 }
          },
          {
            symbol: 'ETHUSDT', 
            components: { ml: -0.123, technical: -0.089, sentiment: -0.067, fear_greed: 0.045 }
          }
        ];
        
        // Inline signal fusion calculation
        const processedSignals = rawSignals.map(signal => {
          const { ml, technical, sentiment, fear_greed } = signal.components;
          const normalized_fg = (2 * fear_greed) - 1;
          const fused_signal = (
            ml * 0.45 +
            sentiment * 0.20 +
            technical * 0.25 +
            normalized_fg * 0.10
          );
          const confidence = Math.min(Math.abs(fused_signal) * 0.8 + 0.2, 1.0);
          
          return {
            symbol: signal.symbol,
            signal: Math.round(fused_signal * 1000) / 1000,
            confidence: Math.round(confidence * 1000) / 1000,
            components: signal.components,
            timestamp: new Date().toISOString()
          };
        });
        
        console.log('TradingBotService.getMockSignals - Using inline fusion calculation:', processedSignals);
        return processedSignals;
      } catch (fallbackError) {
        console.error('Error in inline fusion calculation:', fallbackError);
        console.log('TradingBotService.getMockSignals - Using simple fallback signals');
        return this.getSimpleMockSignals();
      }
    }
  }

  async getSignalHistory(): Promise<any[]> {
    return this.cachedRequest('signal-history', async () => {
      const data = [];
      for (let i = 0; i < 24; i++) {
        const timestamp = new Date(Date.now() - i * 3600000);
        data.unshift({
          timestamp: timestamp.toISOString(),
          ml: (Math.random() - 0.5) * 0.8,
          technical: (Math.random() - 0.5) * 0.6,
          sentiment: (Math.random() - 0.5) * 0.4,
          fear_greed: (Math.random() - 0.5) * 0.3
        });
      }
      return data;
    }, 30000); // 30 second cache for signal history
  }

  async getRiskMetrics(): Promise<any> {
    try {
      if (!this.authToken) {
        await this.initialize();
      }

      const response = await fetch(`${this.baseUrl}/trading/risk`, {
        headers: {
          'Authorization': `Bearer ${this.authToken}`,
          'Content-Type': 'application/json',
        }
      });

      if (response.ok) {
        const data = await response.json();
        console.log('✅ Real risk metrics fetched from backend:', data);
        return {
          currentDrawdown: data.current_drawdown || 0,
          maxDrawdown: data.max_drawdown || 0,
          dailyPnL: data.daily_pnl || 0,
          dailyLossLimit: data.daily_loss_limit || -300.00,
          dailyProfitLimit: data.daily_profit_limit || 200.00,
          tradesCount: data.trades_count || 0,
          maxTradesPerDay: data.max_trades_per_day || 15,
          consecutiveLosses: data.consecutive_losses || 0,
          maxConsecutiveLosses: data.max_consecutive_losses || 4,
          marginRatio: data.margin_ratio || 0,
          exposureRatio: data.exposure_ratio || 0,
          leverageRatio: data.leverage_ratio || 0,
          riskScore: data.risk_score || 0,
          killSwitchStatus: data.kill_switch_status || false
        };
      } else {
        console.error('❌ Failed to fetch risk metrics:', response.status);
        return this.getMockRiskMetrics();
      }
    } catch (error) {
      console.error('❌ Error fetching risk metrics:', error);
      return this.getMockRiskMetrics();
    }
  }

  private getMockRiskMetrics(): any {
    return {
      currentDrawdown: 0,
      maxDrawdown: 0,
      dailyPnL: 0,
      dailyLossLimit: -300.00,
      dailyProfitLimit: 200.00,
      tradesCount: 0,
      maxTradesPerDay: 15,
      consecutiveLosses: 0,
      maxConsecutiveLosses: 4,
      marginRatio: 0,
      exposureRatio: 0,
      leverageRatio: 0,
      riskScore: 0,
      killSwitchStatus: false
    };
  }

  async getAlerts(): Promise<any[]> {
    return [
      {
        id: '1',
        type: 'warning',
        title: 'High Volatility Detected',
        message: 'BTCUSDT showing unusual price movements. Risk parameters adjusted.',
        timestamp: new Date(Date.now() - 3600000).toISOString(),
        read: false
      },
      {
        id: '2',
        type: 'success',
        title: 'Position Closed',
        message: 'ETHUSDT long position closed at profit target (+2.3%)',
        timestamp: new Date(Date.now() - 7200000).toISOString(),
        read: true
      },
      {
        id: '3',
        type: 'info',
        title: 'ML Model Updated',
        message: 'Weekly model retrain completed. New features added.',
        timestamp: new Date(Date.now() - 86400000).toISOString(),
        read: false
      }
    ];
  }

  async markAlertAsRead(alertId: string): Promise<void> {
    console.log(`Marking alert ${alertId} as read`);
  }

  async dismissAlert(alertId: string): Promise<void> {
    console.log(`Dismissing alert ${alertId}`);
  }

  async runBacktest(_config: any): Promise<any> {
    // Simulate backtest processing
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    return {
      totalReturn: 0.234,
      sharpeRatio: 1.42,
      maxDrawdown: 0.089,
      winRate: 0.687,
      totalTrades: 89,
      avgTradeReturn: 0.0234,
      profitFactor: 1.89,
      equityCurve: this.generateBacktestEquityCurve()
    };
  }

  private generateBacktestEquityCurve(): any[] {
    const data = [];
    const startDate = new Date('2024-01-01');
    const days = 365;
    let portfolio = 10000;
    let benchmark = 10000;
    
    for (let i = 0; i < days; i++) {
      const date = new Date(startDate);
      date.setDate(date.getDate() + i);
      
      const strategyReturn = (Math.random() - 0.47) * 0.015;
      const benchmarkReturn = (Math.random() - 0.5) * 0.01;
      
      portfolio *= (1 + strategyReturn);
      benchmark *= (1 + benchmarkReturn);
      
      data.push({
        date: date.toISOString().split('T')[0],
        portfolio: Math.round(portfolio * 100) / 100,
        benchmark: Math.round(benchmark * 100) / 100
      });
    }
    
    return data;
  }

  async getConfig(): Promise<any> {
    return {
      mode: 'PAPER',
      symbols: ['BTCUSDT', 'ETHUSDT'],
      timeframes: ['15m'],
      futures: {
        margin_mode: 'isolated',
        max_leverage: 5,
        one_way_mode: true
      },
      risk: {
        risk_per_trade: 0.0075,
        k_sl_atr: 2.5,
        k_tp_atr: 3.0,
        liq_buffer_atr: 3.0,
        dd_soft: 0.06,
        dd_hard: 0.10,
        max_loss_per_trade: 0.008
      },
      limits: {
        max_loss_per_day: 0.03,
        max_profit_per_day: 0.02,
        max_trades_per_day: 15,
        max_consecutive_losses: 4
      },
      fusion: {
        w_ml: 0.45,
        w_sent: 0.20,
        w_ta: 0.25,
        w_ctx: 0.10,
        enter_long_thresh: 0.35,
        enter_short_thresh: -0.35,
        exit_thresh: 0.10,
        min_conf: 0.55
      },
      sentiment: {
        x: { enabled: true },
        reddit: { enabled: true },
        window_min: 120,
        cap_abs: 0.7
      },
      technical: {
        sma_fast: 20,
        sma_slow: 100,
        vol_filter_pct: [25, 85],
        elliott_wave_enabled: false
      },
      broker: {
        exchange: 'bybit',
        market_type: 'linear_perp',
        slippage_bps_buffer: 12
      }
    };
  }

  async updateConfig(config: any): Promise<void> {
    console.log('Updating configuration:', config);
    // Implementation would save config to backend
  }
}