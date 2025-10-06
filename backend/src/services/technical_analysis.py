"""
Technical Analysis Module for Crypto Trading Bot

This module implements technical analysis rules and generates s_ta signals in [-1, +1] range.
Features:
- SMA(20) vs SMA(100) trend analysis
- ATR(14)/Price volatility filtering with percentile thresholds
- Optional Elliott Wave analysis (feature-flag)
- Output: s_ta in [-1, +1] range
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class TechnicalAnalysisModule:
    """
    Technical Analysis Module for crypto trading signals.
    
    Implements:
    - SMA trend analysis (SMA20 vs SMA100)
    - ATR volatility filtering with percentile thresholds
    - Optional Elliott Wave analysis
    - s_ta signal generation in [-1, +1] range
    """
    
    def __init__(self, 
                 config: Optional[Dict] = None,
                 enable_elliott_wave: bool = False):
        """
        Initialize Technical Analysis Module.
        
        Args:
            config: Configuration dictionary
            enable_elliott_wave: Enable Elliott Wave analysis (feature flag)
        """
        self.config = config or {}
        self.enable_elliott_wave = enable_elliott_wave
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Technical analysis parameters
        self.sma_short = self.config.get('sma_short', 20)
        self.sma_long = self.config.get('sma_long', 100)
        self.atr_period = self.config.get('atr_period', 14)
        self.volatility_window = self.config.get('volatility_window', 90)  # days
        self.volatility_percentile_low = self.config.get('volatility_percentile_low', 25)
        self.volatility_percentile_high = self.config.get('volatility_percentile_high', 85)
        
        # Signal parameters
        self.trend_strength_threshold = self.config.get('trend_strength_threshold', 0.02)  # 2%
        self.volatility_filter_enabled = self.config.get('volatility_filter_enabled', True)
        
        # Elliott Wave parameters (if enabled)
        if self.enable_elliott_wave:
            self.elliott_wave_periods = self.config.get('elliott_wave_periods', [5, 13, 21, 34, 55])
            self.elliott_wave_threshold = self.config.get('elliott_wave_threshold', 0.618)
    
    def calculate_sma_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate SMA-based trend signals.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with SMA signals
        """
        # Determine which price column to use
        price_column = None
        if 'close' in df.columns:
            price_column = 'close'
        elif 'mark_close' in df.columns:
            price_column = 'mark_close'
        elif 'index_close' in df.columns:
            price_column = 'index_close'
        else:
            self.logger.warning("No price column found for SMA calculation, using neutral signals")
            df['sma_trend'] = 0
            df['trend_strength'] = 0
            return df
        
        # Calculate SMAs
        df[f'sma_{self.sma_short}'] = df[price_column].rolling(window=self.sma_short).mean()
        df[f'sma_{self.sma_long}'] = df[price_column].rolling(window=self.sma_long).mean()
        
        # SMA trend signal
        df['sma_trend'] = np.where(
            df[f'sma_{self.sma_short}'] > df[f'sma_{self.sma_long}'], 1, -1
        )
        
        # Trend strength (percentage difference)
        df['trend_strength'] = abs(
            (df[f'sma_{self.sma_short}'] - df[f'sma_{self.sma_long}']) / df[f'sma_{self.sma_long}']
        )
        
        # Strong trend filter
        df['strong_trend'] = df['trend_strength'] > self.trend_strength_threshold
        
        return df
    
    def calculate_atr_volatility_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate ATR-based volatility filter using percentile thresholds.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with volatility filter
        """
        # Determine which price columns to use
        high_column = None
        low_column = None
        close_column = None
        
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            high_column, low_column, close_column = 'high', 'low', 'close'
        elif 'mark_high' in df.columns and 'mark_low' in df.columns and 'mark_close' in df.columns:
            high_column, low_column, close_column = 'mark_high', 'mark_low', 'mark_close'
        elif 'index_high' in df.columns and 'index_low' in df.columns and 'index_close' in df.columns:
            high_column, low_column, close_column = 'index_high', 'index_low', 'index_close'
        else:
            self.logger.warning("No OHLC columns found for ATR calculation, using neutral filter")
            df['volatility_filter'] = 1  # Allow all trades
            return df
        
        # Calculate ATR
        high_low = df[high_column] - df[low_column]
        high_close = np.abs(df[high_column] - df[close_column].shift(1))
        low_close = np.abs(df[low_column] - df[close_column].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(window=self.atr_period).mean()
        
        # Calculate ATR/Price ratio
        df['atr_price_ratio'] = df['atr'] / df[close_column]
        
        # Calculate rolling percentiles over volatility window
        df['atr_price_p25'] = df['atr_price_ratio'].rolling(
            window=self.volatility_window * 24,  # Assuming hourly data
            min_periods=self.volatility_window * 4  # Minimum 4 days
        ).quantile(self.volatility_percentile_low / 100)
        
        df['atr_price_p85'] = df['atr_price_ratio'].rolling(
            window=self.volatility_window * 24,
            min_periods=self.volatility_window * 4
        ).quantile(self.volatility_percentile_high / 100)
        
        # Volatility filter: ATR/Price in [p25, p85] range
        df['volatility_filter'] = (
            (df['atr_price_ratio'] >= df['atr_price_p25']) &
            (df['atr_price_ratio'] <= df['atr_price_p85'])
        )
        
        return df
    
    def calculate_elliott_wave_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Elliott Wave-based signals (optional feature).
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Elliott Wave signals
        """
        if not self.enable_elliott_wave:
            df['elliott_wave_signal'] = 0
            return df
        
        # Simple Elliott Wave implementation using Fibonacci retracements
        df['elliott_wave_signal'] = 0
        
        # Determine which price columns to use
        high_column = None
        low_column = None
        close_column = None
        
        if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
            high_column, low_column, close_column = 'high', 'low', 'close'
        elif 'mark_high' in df.columns and 'mark_low' in df.columns and 'mark_close' in df.columns:
            high_column, low_column, close_column = 'mark_high', 'mark_low', 'mark_close'
        elif 'index_high' in df.columns and 'index_low' in df.columns and 'index_close' in df.columns:
            high_column, low_column, close_column = 'index_high', 'index_low', 'index_close'
        else:
            self.logger.warning("No OHLC columns found for Elliott Wave calculation, using neutral signal")
            return df
        
        for period in self.elliott_wave_periods:
            # Calculate rolling high and low
            rolling_high = df[high_column].rolling(window=period).max()
            rolling_low = df[low_column].rolling(window=period).min()
            
            # Calculate Fibonacci levels
            fib_range = rolling_high - rolling_low
            fib_618 = rolling_low + (fib_range * self.elliott_wave_threshold)
            fib_382 = rolling_high - (fib_range * self.elliott_wave_threshold)
            
            # Elliott Wave signals
            bullish_wave = df[close_column] > fib_618
            bearish_wave = df[close_column] < fib_382
            
            df['elliott_wave_signal'] += np.where(bullish_wave, 1, 0) - np.where(bearish_wave, 1, 0)
        
        # Normalize Elliott Wave signal
        df['elliott_wave_signal'] = np.clip(df['elliott_wave_signal'] / len(self.elliott_wave_periods), -1, 1)
        
        return df
    
    def calculate_rsi_signals(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Calculate RSI-based signals for additional confirmation.
        
        Args:
            df: DataFrame with OHLCV data
            period: RSI period
            
        Returns:
            DataFrame with RSI signals
        """
        # Determine which price column to use
        price_column = None
        if 'close' in df.columns:
            price_column = 'close'
        elif 'mark_close' in df.columns:
            price_column = 'mark_close'
        elif 'index_close' in df.columns:
            price_column = 'index_close'
        else:
            self.logger.warning("No price column found for RSI calculation, using neutral signals")
            df['rsi'] = 50  # Neutral RSI
            df['rsi_oversold'] = False
            df['rsi_overbought'] = False
            return df
        
        # Calculate RSI
        delta = df[price_column].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # RSI signals
        df['rsi_oversold'] = df['rsi'] < 30
        df['rsi_overbought'] = df['rsi'] > 70
        
        # RSI signal: -1 for oversold (bullish), +1 for overbought (bearish)
        df['rsi_signal'] = np.where(df['rsi_oversold'], -1, 
                                  np.where(df['rsi_overbought'], 1, 0))
        
        return df
    
    def generate_s_ta_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate s_ta signal in [-1, +1] range combining all technical indicators.
        
        Args:
            df: DataFrame with technical indicators
            
        Returns:
            DataFrame with s_ta signal
        """
        # Initialize s_ta signal
        df['s_ta'] = 0.0
        
        # SMA trend signal (primary) - check if column exists
        if 'sma_trend' in df.columns:
            df['s_ta'] += df['sma_trend'] * 0.4
        else:
            self.logger.warning("sma_trend column not found, skipping SMA signal")
        
        # RSI confirmation signal - check if column exists
        if 'rsi_signal' in df.columns:
            df['s_ta'] += df['rsi_signal'] * 0.2
        else:
            self.logger.warning("rsi_signal column not found, skipping RSI signal")
        
        # Elliott Wave signal (if enabled) - check if column exists
        if self.enable_elliott_wave and 'elliott_wave_signal' in df.columns:
            df['s_ta'] += df['elliott_wave_signal'] * 0.2
        elif self.enable_elliott_wave:
            self.logger.warning("elliott_wave_signal column not found, skipping Elliott Wave signal")
        
        # Apply volatility filter - check if column exists
        if self.volatility_filter_enabled and 'volatility_filter' in df.columns:
            df['s_ta'] = np.where(df['volatility_filter'], df['s_ta'], 0)
        elif self.volatility_filter_enabled:
            self.logger.warning("volatility_filter column not found, skipping volatility filter")
        
        # Normalize to [-1, +1] range
        df['s_ta'] = np.clip(df['s_ta'], -1, 1)
        
        # Add signal strength
        df['s_ta_strength'] = abs(df['s_ta'])
        
        # Add signal confidence based on multiple confirmations
        confirmations = 0
        if 'sma_trend' in df.columns:
            confirmations += 1
        if 'rsi_signal' in df.columns:
            confirmations += 1
        if self.enable_elliott_wave and 'elliott_wave_signal' in df.columns:
            confirmations += 1
        
        df['s_ta_confidence'] = df['s_ta_strength'] * (confirmations / 3.0)
        
        return df
    
    def analyze_technical_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Complete technical analysis pipeline.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with all technical analysis signals
        """
        self.logger.info("Starting technical analysis...")
        
        # Calculate all technical indicators
        df = self.calculate_sma_signals(df)
        df = self.calculate_atr_volatility_filter(df)
        df = self.calculate_rsi_signals(df)
        
        if self.enable_elliott_wave:
            df = self.calculate_elliott_wave_signals(df)
        
        # Generate final s_ta signal
        df = self.generate_s_ta_signal(df)
        
        self.logger.info(f"Technical analysis completed. Generated s_ta signals: {df['s_ta'].describe()}")
        
        return df
    
    def get_latest_s_ta_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get the latest technical analysis signal.
        
        Args:
            df: DataFrame with technical analysis data
            
        Returns:
            Dictionary with latest signal information
        """
        if df.empty:
            return {
                's_ta': 0.0,
                's_ta_strength': 0.0,
                's_ta_confidence': 0.0,
                'signal': 'HOLD',
                'timestamp': datetime.now().isoformat()
            }
        
        latest = df.iloc[-1]
        
        # Determine signal direction
        if latest['s_ta'] > 0.1:
            signal = 'SELL'
        elif latest['s_ta'] < -0.1:
            signal = 'BUY'
        else:
            signal = 'HOLD'
        
        return {
            's_ta': float(latest['s_ta']),
            's_ta_strength': float(latest['s_ta_strength']),
            's_ta_confidence': float(latest['s_ta_confidence']),
            'signal': signal,
            'sma_trend': float(latest.get('sma_trend', 0)),
            'rsi': float(latest.get('rsi', 50)),
            'volatility_filter': bool(latest.get('volatility_filter', True)),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_technical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get technical analysis summary statistics.
        
        Args:
            df: DataFrame with technical analysis data
            
        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {}
        
        return {
            'total_signals': len(df),
            'buy_signals': len(df[df['s_ta'] < -0.1]),
            'sell_signals': len(df[df['s_ta'] > 0.1]),
            'hold_signals': len(df[(df['s_ta'] >= -0.1) & (df['s_ta'] <= 0.1)]),
            'avg_s_ta': float(df['s_ta'].mean()),
            'avg_confidence': float(df['s_ta_confidence'].mean()),
            'volatility_filter_active': float(df['volatility_filter'].mean()),
            'elliott_wave_enabled': self.enable_elliott_wave
        }


def main():
    """Test the Technical Analysis Module."""
    import sys
    sys.path.append('src')
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = {
        'sma_short': 20,
        'sma_long': 100,
        'atr_period': 14,
        'volatility_window': 90,
        'volatility_percentile_low': 25,
        'volatility_percentile_high': 85,
        'trend_strength_threshold': 0.02,
        'volatility_filter_enabled': True,
        'elliott_wave_periods': [5, 13, 21, 34, 55],
        'elliott_wave_threshold': 0.618
    }
    
    # Initialize technical analysis module
    ta_module = TechnicalAnalysisModule(config=config, enable_elliott_wave=True)
    
    # Create sample data for testing
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'open_time': dates,
        'open': 50000 + np.cumsum(np.random.randn(len(dates)) * 100),
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    # Generate realistic OHLCV data
    for i in range(len(sample_data)):
        base_price = sample_data.iloc[i]['open']
        volatility = np.random.uniform(0.01, 0.03)
        
        sample_data.iloc[i, sample_data.columns.get_loc('high')] = base_price * (1 + volatility)
        sample_data.iloc[i, sample_data.columns.get_loc('low')] = base_price * (1 - volatility)
        sample_data.iloc[i, sample_data.columns.get_loc('close')] = base_price * (1 + np.random.uniform(-volatility, volatility))
    
    # Run technical analysis
    result_df = ta_module.analyze_technical_signals(sample_data)
    
    # Get latest signal
    latest_signal = ta_module.get_latest_s_ta_signal(result_df)
    
    print("Technical Analysis Test Results:")
    print(f"Latest s_ta signal: {latest_signal['s_ta']:.4f}")
    print(f"Signal: {latest_signal['signal']}")
    print(f"Confidence: {latest_signal['s_ta_confidence']:.4f}")
    
    # Get summary
    summary = ta_module.get_technical_summary(result_df)
    print(f"\nSummary: {summary}")


if __name__ == "__main__":
    main()
