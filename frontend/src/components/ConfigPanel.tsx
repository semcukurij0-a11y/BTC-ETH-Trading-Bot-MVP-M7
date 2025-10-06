import React, { useState, useEffect } from 'react';
import { Save, Download, Upload, RotateCcw } from 'lucide-react';

interface ConfigPanelProps {
  service: any;
}

export const ConfigPanel: React.FC<ConfigPanelProps> = ({ service }) => {
  const [config, setConfig] = useState({
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
  });

  const [hasChanges, setHasChanges] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    loadConfig();
  }, []);

  const loadConfig = async () => {
    try {
      const data = await service.getConfig();
      setConfig(data);
    } catch (error) {
      console.error('Failed to load config:', error);
    }
  };

  const saveConfig = async () => {
    setSaving(true);
    try {
      await service.updateConfig(config);
      setHasChanges(false);
    } catch (error) {
      console.error('Failed to save config:', error);
    } finally {
      setSaving(false);
    }
  };

  const resetToDefaults = () => {
    // Reset to default configuration
    setConfig({
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
    });
    setHasChanges(true);
  };

  const updateConfig = (path: string, value: any) => {
    const keys = path.split('.');
    const newConfig = { ...config };
    let current = newConfig;
    
    for (let i = 0; i < keys.length - 1; i++) {
      current[keys[i]] = { ...current[keys[i]] };
      current = current[keys[i]];
    }
    
    current[keys[keys.length - 1]] = value;
    setConfig(newConfig);
    setHasChanges(true);
  };

  const formatPercent = (value: number) => (value * 100).toFixed(2);
  const parsePercent = (value: string) => parseFloat(value) / 100;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">Configuration</h2>
        <div className="flex space-x-2">
          <button
            onClick={resetToDefaults}
            className="flex items-center space-x-2 px-4 py-2 bg-gray-700 text-white rounded hover:bg-gray-600 transition-colors"
          >
            <RotateCcw className="h-4 w-4" />
            <span>Reset</span>
          </button>
          <button
            onClick={saveConfig}
            disabled={!hasChanges || saving}
            className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors disabled:opacity-50"
          >
            <Save className="h-4 w-4" />
            <span>{saving ? 'Saving...' : 'Save Changes'}</span>
          </button>
        </div>
      </div>

      {/* General Settings */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 text-white">General Settings</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Trading Mode</label>
            <select
              value={config.mode}
              onChange={(e) => updateConfig('mode', e.target.value)}
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="PAPER">Paper Trading</option>
              <option value="LIVE">Live Trading</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Exchange</label>
            <select
              value={config.broker.exchange}
              onChange={(e) => updateConfig('broker.exchange', e.target.value)}
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="bybit">Bybit</option>
              <option value="binance">Binance</option>
              <option value="okx">OKX</option>
            </select>
          </div>
        </div>
      </div>

      {/* Futures Settings */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 text-white">Futures Settings</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Margin Mode</label>
            <select
              value={config.futures.margin_mode}
              onChange={(e) => updateConfig('futures.margin_mode', e.target.value)}
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="isolated">Isolated</option>
              <option value="cross">Cross</option>
            </select>
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Max Leverage</label>
            <input
              type="number"
              min="1"
              max="10"
              value={config.futures.max_leverage}
              onChange={(e) => updateConfig('futures.max_leverage', Number(e.target.value))}
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div className="flex items-center">
            <label className="flex items-center space-x-2">
              <input
                type="checkbox"
                checked={config.futures.one_way_mode}
                onChange={(e) => updateConfig('futures.one_way_mode', e.target.checked)}
                className="rounded focus:ring-2 focus:ring-blue-500"
              />
              <span className="text-white text-sm">One-way Mode</span>
            </label>
          </div>
        </div>
      </div>

      {/* Risk Management */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 text-white">Risk Management</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Risk per Trade (%)</label>
            <input
              type="number"
              step="0.1"
              min="0.1"
              max="5"
              value={formatPercent(config.risk.risk_per_trade)}
              onChange={(e) => updateConfig('risk.risk_per_trade', parsePercent(e.target.value))}
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Stop Loss ATR Multiplier</label>
            <input
              type="number"
              step="0.1"
              min="1"
              max="5"
              value={config.risk.k_sl_atr}
              onChange={(e) => updateConfig('risk.k_sl_atr', Number(e.target.value))}
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Take Profit ATR Multiplier</label>
            <input
              type="number"
              step="0.1"
              min="1"
              max="10"
              value={config.risk.k_tp_atr}
              onChange={(e) => updateConfig('risk.k_tp_atr', Number(e.target.value))}
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>
      </div>

      {/* Daily Limits */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 text-white">Daily Limits</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Max Loss per Day (%)</label>
            <input
              type="number"
              step="0.1"
              min="1"
              max="10"
              value={formatPercent(config.limits.max_loss_per_day)}
              onChange={(e) => updateConfig('limits.max_loss_per_day', parsePercent(e.target.value))}
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Max Profit per Day (%)</label>
            <input
              type="number"
              step="0.1"
              min="1"
              max="10"
              value={formatPercent(config.limits.max_profit_per_day)}
              onChange={(e) => updateConfig('limits.max_profit_per_day', parsePercent(e.target.value))}
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Max Trades per Day</label>
            <input
              type="number"
              min="1"
              max="50"
              value={config.limits.max_trades_per_day}
              onChange={(e) => updateConfig('limits.max_trades_per_day', Number(e.target.value))}
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Max Consecutive Losses</label>
            <input
              type="number"
              min="1"
              max="10"
              value={config.limits.max_consecutive_losses}
              onChange={(e) => updateConfig('limits.max_consecutive_losses', Number(e.target.value))}
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>
      </div>

      {/* Signal Fusion Weights */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h3 className="text-lg font-semibold mb-4 text-white">Signal Fusion Weights</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">ML Weight</label>
            <input
              type="number"
              step="0.05"
              min="0"
              max="1"
              value={config.fusion.w_ml}
              onChange={(e) => updateConfig('fusion.w_ml', Number(e.target.value))}
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Sentiment Weight</label>
            <input
              type="number"
              step="0.05"
              min="0"
              max="1"
              value={config.fusion.w_sent}
              onChange={(e) => updateConfig('fusion.w_sent', Number(e.target.value))}
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Technical Weight</label>
            <input
              type="number"
              step="0.05"
              min="0"
              max="1"
              value={config.fusion.w_ta}
              onChange={(e) => updateConfig('fusion.w_ta', Number(e.target.value))}
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-2">Context Weight</label>
            <input
              type="number"
              step="0.05"
              min="0"
              max="1"
              value={config.fusion.w_ctx}
              onChange={(e) => updateConfig('fusion.w_ctx', Number(e.target.value))}
              className="w-full bg-gray-700 text-white rounded px-3 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        </div>
      </div>

      {hasChanges && (
        <div className="bg-yellow-800 border border-yellow-600 rounded-lg p-4">
          <p className="text-yellow-200">
            You have unsaved changes. Click "Save Changes" to apply your configuration updates.
          </p>
        </div>
      )}
    </div>
  );
};