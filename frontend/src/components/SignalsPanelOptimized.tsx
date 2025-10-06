import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Brain, TrendingUp, MessageCircle, Activity, AlertCircle } from 'lucide-react';

interface Signal {
  type: 'ml' | 'technical' | 'sentiment' | 'fear_greed';
  value: number;
  confidence: number;
  timestamp: string;
}

interface FusedSignal {
  symbol: string;
  signal: number;
  confidence: number;
  components?: {
    momentum?: number;
    volume?: number;
    noise?: number;
    fear_greed?: number;
    ml?: number;
    technical?: number;
    sentiment?: number;
  };
  timestamp: string;
}

interface SignalsPanelProps {
  service: any;
  data?: FusedSignal[];
  isLoading?: boolean;
  error?: string | null;
  lastUpdate?: string | null;
  hasChanges?: boolean;
  changeCount?: number;
  changeDetection?: any;
}

export const SignalsPanelOptimized: React.FC<SignalsPanelProps> = ({ 
  service, 
  data = [],
  isLoading = false,
  error,
  lastUpdate,
  hasChanges = false,
  changeCount = 0,
  changeDetection
}) => {
  const [signalHistory, setSignalHistory] = useState([]);
  const [message, setMessage] = useState<{ type: 'success' | 'error'; text: string } | null>(null);

  // Ensure data is an array and has proper structure
  const signals = Array.isArray(data) ? data : [];

  // Debug logging
  useEffect(() => {
    console.log('🔍 SignalsPanelOptimized - Data received:', {
      data,
      signals,
      signalsLength: signals.length,
      isLoading,
      error,
      lastUpdate,
      hasChanges,
      changeCount
    });
    
    if (signals.length > 0) {
      console.log('📊 First signal details:', signals[0]);
      console.log('📊 Signal properties:', {
        symbol: signals[0].symbol,
        signal: signals[0].signal,
        confidence: signals[0].confidence,
        components: signals[0].components
      });
      
      // Check if signal values are properly formatted
      const firstSignal = signals[0];
      console.log('📊 Signal value check:', {
        signalValue: firstSignal.signal,
        signalType: typeof firstSignal.signal,
        signalIsNumber: typeof firstSignal.signal === 'number',
        signalIsNaN: isNaN(firstSignal.signal),
        signalFormatted: firstSignal.signal?.toFixed(3)
      });
    } else {
      console.log('⚠️ No signals found - checking data structure:', {
        dataType: typeof data,
        isArray: Array.isArray(data),
        dataKeys: data ? Object.keys(data) : 'null',
        dataValue: data
      });
      
      // Check if data is being passed but not as expected
      if (data && !Array.isArray(data)) {
        console.log('⚠️ Data is not an array:', {
          dataType: typeof data,
          dataKeys: Object.keys(data),
          dataValue: data
        });
      }
    }
  }, [data, signals, isLoading, error, lastUpdate, hasChanges, changeCount]);

  const getSignalColor = (value: number | undefined) => {
    if (value === undefined || value === null || isNaN(value)) return 'text-gray-400';
    if (value > 0.3) return 'text-green-400';
    if (value < -0.3) return 'text-red-400';
    return 'text-gray-400';
  };

  const getSignalStrength = (value: number | undefined) => {
    if (value === undefined || value === null || isNaN(value)) return 'Weak';
    const abs = Math.abs(value);
    if (abs > 0.6) return 'Strong';
    if (abs > 0.3) return 'Moderate';
    return 'Weak';
  };

  const formatPercent = (value: number | undefined) => {
    if (value === undefined || value === null || isNaN(value)) {
      return '0.0%';
    }
    return `${(value * 100).toFixed(1)}%`;
  };
  
  const safeToFixed = (value: number | undefined, decimals: number = 3): string => {
    if (value === undefined || value === null || isNaN(value)) {
      return '0.000';
    }
    return value.toFixed(decimals);
  };

  if (isLoading) {
    return (
      <div className="space-y-6">
        <div className="bg-gray-800 rounded-lg p-6">
          <div className="animate-pulse">
            <div className="h-6 bg-gray-700 rounded mb-4"></div>
            <div className="space-y-3">
              {[...Array(3)].map((_, i) => (
                <div key={i} className="h-16 bg-gray-700 rounded"></div>
              ))}
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Status Bar */}
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="flex items-center justify-between text-sm text-gray-400">
          <div className="flex items-center space-x-4">
            <span>Active Signals: {signals.length}</span>
            {hasChanges && (
              <span className="flex items-center text-green-400">
                <div className="w-2 h-2 bg-green-400 rounded-full mr-1 animate-pulse"></div>
                Updated ({changeCount} changes)
              </span>
            )}
          </div>
          {lastUpdate && (
            <span>Last updated: {new Date(lastUpdate).toLocaleString()}</span>
          )}
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900 border border-red-700 rounded-lg p-4">
          <div className="flex items-center">
            <div className="text-red-400 mr-2">⚠️</div>
            <div>
              <h3 className="text-red-400 font-medium">Data Error</h3>
              <p className="text-red-300 text-sm">{error}</p>
            </div>
          </div>
        </div>
      )}

      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">Trading Signals</h2>
        <div className="text-sm text-gray-400">
          Last updated: {signals[0]?.timestamp ? new Date(signals[0].timestamp).toLocaleTimeString() : 'N/A'}
        </div>
      </div>

      {/* Current Signals */}
      {signals.length === 0 ? (
        <div className="bg-gray-800 rounded-lg p-8 text-center">
          <Brain className="h-12 w-12 text-gray-600 mx-auto mb-4" />
          <h3 className="text-lg font-semibold text-gray-400 mb-2">No Signals Available</h3>
          <p className="text-gray-500">Trading signals will appear here when available.</p>
        </div>
      ) : (
        <div className="grid gap-6">
          {signals.map((signal, index) => (
            <div key={`${signal.symbol}-${index}`} className="bg-gray-800 rounded-lg p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-bold text-white">{signal.symbol}</h3>
                <div className="flex items-center space-x-4">
                  <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                    (signal.signal || 0) > 0.3 
                      ? 'bg-green-100 text-green-800' 
                      : (signal.signal || 0) < -0.3 
                      ? 'bg-red-100 text-red-800'
                      : 'bg-gray-100 text-gray-800'
                  }`}>
                    {(signal.signal || 0) > 0 ? 'BULLISH' : 'BEARISH'} - {getSignalStrength(signal.signal)}
                  </div>
                  <div className="text-sm text-gray-400">
                    Confidence: {formatPercent(signal.confidence)}
                  </div>
                </div>
              </div>

              {/* Main Fused Signal Card */}
              <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-lg p-6 mb-6 border border-gray-600">
                <div className="flex items-center justify-between mb-4">
                  <div className="flex items-center space-x-3">
                    <Brain className="h-6 w-6 text-blue-400" />
                    <div>
                      <h4 className="text-lg font-semibold text-white">Fused Signal</h4>
                      <p className="text-sm text-gray-400">Combined AI Trading Signal</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-gray-400">Confidence</div>
                    <div className="text-lg font-medium text-blue-400">
                      {formatPercent(signal.confidence)}
                    </div>
                  </div>
                </div>
                
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <div className={`text-4xl font-bold ${getSignalColor(signal.signal)} mb-2`}>
                      {safeToFixed(signal.signal, 3)}
                    </div>
                    <div className="flex items-center space-x-4">
                      <div className={`px-3 py-1 rounded-full text-sm font-medium ${
                        (signal.signal || 0) > 0.3 
                          ? 'bg-green-100 text-green-800' 
                          : (signal.signal || 0) < -0.3 
                          ? 'bg-red-100 text-red-800'
                          : 'bg-gray-100 text-gray-800'
                      }`}>
                        {getSignalStrength(signal.signal)}
                      </div>
                      <div className="text-sm text-gray-400">
                        {(signal.signal || 0) > 0 ? 'BULLISH' : 'BEARISH'}
                      </div>
                    </div>
                  </div>
                  
                  {/* Signal Strength Bar */}
                  <div className="w-32">
                    <div className="w-full bg-gray-600 rounded-full h-3 mb-2">
                      <div 
                        className={`h-3 rounded-full transition-all duration-300 ${
                          (signal.signal || 0) > 0 ? 'bg-green-400' : 'bg-red-400'
                        }`}
                        style={{ 
                          width: `${Math.abs(signal.signal || 0) * 100}%`,
                          marginLeft: (signal.signal || 0) < 0 ? `${100 - Math.abs(signal.signal || 0) * 100}%` : '0'
                        }}
                      />
                    </div>
                    <div className="text-xs text-gray-400 text-center">
                      Signal Strength
                    </div>
                  </div>
                </div>
              </div>

              {/* Signal Components Breakdown */}
              {signal.components && (
                <div className="bg-gray-700 rounded-lg p-4">
                  <h4 className="text-sm font-medium text-gray-300 mb-3">Signal Components</h4>
                  <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                    {Object.entries(signal.components).map(([key, value]) => (
                      <div key={key} className="text-center">
                        <div className="text-xs text-gray-400 capitalize">
                          {key.replace('_', ' ')}
                        </div>
                        <div className={`text-sm font-medium ${getSignalColor(value)}`}>
                          {safeToFixed(value, 3)}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Timestamp */}
              <div className="text-xs text-gray-500 mt-4">
                Generated: {signal.timestamp ? new Date(signal.timestamp).toLocaleString() : 'N/A'}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Message Display */}
      {message && (
        <div className={`rounded-lg p-4 ${
          message.type === 'success' 
            ? 'bg-green-900 border border-green-700 text-green-300' 
            : 'bg-red-900 border border-red-700 text-red-300'
        }`}>
          <div className="flex items-center">
            <div className="mr-2">
              {message.type === 'success' ? '✅' : '❌'}
            </div>
            <div>
              <h3 className="font-medium">{message.type === 'success' ? 'Success' : 'Error'}</h3>
              <p className="text-sm">{message.text}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
