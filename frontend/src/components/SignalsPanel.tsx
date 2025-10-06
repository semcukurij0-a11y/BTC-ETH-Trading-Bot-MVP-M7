import React, { useState, useEffect } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import { Brain, TrendingUp, MessageCircle, Activity, AlertCircle } from 'lucide-react';
import { SkeletonCard, SkeletonList } from './LoadingSkeleton';

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
  components: {
    ml: number;
    technical: number;
    sentiment: number;
    fear_greed: number;
  };
  timestamp: string;
}

interface SignalsPanelProps {
  service: any;
}

export const SignalsPanel: React.FC<SignalsPanelProps> = ({ service }) => {
  const [signals, setSignals] = useState<FusedSignal[]>([]);
  const [signalHistory, setSignalHistory] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadSignals = async () => {
      try {
        setError(null);
        const [currentSignals, history] = await Promise.all([
          service.getCurrentSignals(),
          service.getSignalHistory()
        ]);
        setSignals(currentSignals || []);
        setSignalHistory(history || []);
        setLoading(false);
      } catch (error) {
        console.error('Failed to load signals:', error);
        setError('Failed to load signals data');
        setLoading(false);
      }
    };

    loadSignals();
    const interval = setInterval(loadSignals, 10000); // Reduced from 5s to 10s
    return () => clearInterval(interval);
  }, [service]);

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

  if (loading) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold text-white">Trading Signals</h2>
          <div className="text-sm text-gray-400">Loading...</div>
        </div>
        <SkeletonList items={2} />
        <SkeletonCard />
      </div>
    );
  }

  if (error) {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-2xl font-bold text-white">Trading Signals</h2>
        </div>
        <div className="bg-red-900/20 border border-red-500/50 rounded-lg p-6">
          <div className="flex items-center space-x-3">
            <AlertCircle className="h-6 w-6 text-red-400" />
            <div>
              <h3 className="text-lg font-semibold text-red-400">Error Loading Signals</h3>
              <p className="text-red-300">{error}</p>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
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

            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <Brain className="h-4 w-4 text-blue-400" />
                  <span className="text-sm font-medium text-gray-300">ML Signal</span>
                </div>
                <p className={`text-xl font-bold ${getSignalColor(signal.components.ml)}`}>
                  {safeToFixed(signal.components.ml)}
                </p>
              </div>

              <div className="bg-gray-700 rounded-lg p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <TrendingUp className="h-4 w-4 text-green-400" />
                  <span className="text-sm font-medium text-gray-300">Technical</span>
                </div>
                <p className={`text-xl font-bold ${getSignalColor(signal.components.technical)}`}>
                  {safeToFixed(signal.components.technical)}
                </p>
              </div>

              <div className="bg-gray-700 rounded-lg p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <MessageCircle className="h-4 w-4 text-purple-400" />
                  <span className="text-sm font-medium text-gray-300">Sentiment</span>
                </div>
                <p className={`text-xl font-bold ${getSignalColor(signal.components.sentiment)}`}>
                  {safeToFixed(signal.components.sentiment)}
                </p>
              </div>

              <div className="bg-gray-700 rounded-lg p-4">
                <div className="flex items-center space-x-2 mb-2">
                  <Activity className="h-4 w-4 text-orange-400" />
                  <span className="text-sm font-medium text-gray-300">Fear & Greed</span>
                </div>
                <p className={`text-xl font-bold ${getSignalColor(signal.components.fear_greed)}`}>
                  {safeToFixed(signal.components.fear_greed)}
                </p>
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
          </div>
        ))}
        </div>
      )}

      {/* Signal History Chart */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4">Signal History</h3>
        <div className="h-80">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={signalHistory}>
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
              <XAxis 
                dataKey="timestamp" 
                stroke="#9CA3AF"
                fontSize={12}
                tickFormatter={(time) => new Date(time).toLocaleTimeString()}
              />
              <YAxis 
                stroke="#9CA3AF"
                fontSize={12}
                domain={[-1, 1]}
              />
              <Tooltip 
                contentStyle={{ 
                  backgroundColor: '#1F2937', 
                  border: '1px solid #374151',
                  borderRadius: '6px'
                }}
                labelFormatter={(time) => new Date(time).toLocaleString()}
              />
              <Bar dataKey="ml" fill="#3B82F6" name="ML" />
              <Bar dataKey="technical" fill="#10B981" name="Technical" />
              <Bar dataKey="sentiment" fill="#8B5CF6" name="Sentiment" />
              <Bar dataKey="fear_greed" fill="#F59E0B" name="Fear & Greed" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};