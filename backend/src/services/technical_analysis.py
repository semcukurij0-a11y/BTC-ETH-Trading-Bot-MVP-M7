#!/usr/bin/env python3
"""
Technical Analysis Module for Crypto Trading Bot

This module provides technical analysis functionality including:
- Moving averages (SMA, EMA)
- Oscillators (RSI, MACD, Stochastic)
- Volatility indicators (ATR, Bollinger Bands)
- Trend indicators (ADX, Parabolic SAR)
- Volume indicators (Volume Z-Score, OBV)
- Support/Resistance levels
- Elliott Wave analysis (optional)
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class TechnicalAnalysisModule:
    """
    Technical Analysis Module for generating trading signals.
    
    Provides comprehensive technical analysis including:
    - Trend analysis
    - Momentum indicators
    - Volatility analysis
    - Volume analysis
    - Support/resistance levels
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Technical Analysis Module.
        
        Args:
            config: Configuration dictionary with technical analysis parameters
        """
        self.config = config or {}
        
        # Technical analysis parameters
        self.sma_short = self.config.get('sma_short', 20)
        self.sma_long = self.config.get('sma_long', 100)
        self.atr_period = self.config.get('atr_period', 14)
        self.volatility_window = self.config.get('volatility_window', 90)
        self.volatility_percentile_low = self.config.get('volatility_percentile_low', 25)
        self.volatility_percentile_high = self.config.get('volatility_percentile_high', 85)
        self.trend_strength_threshold = self.config.get('trend_strength_threshold', 0.02)
        self.volatility_filter_enabled = self.config.get('volatility_filter_enabled', True)
        self.enable_elliott_wave = self.config.get('enable_elliott_wave', False)
        
        # RSI parameters
        self.rsi_period = 14
        self.rsi_overbought = 70
        self.rsi_oversold = 30
        
        # MACD parameters
        self.macd_fast = 12
        self.macd_slow = 26
        self.macd_signal = 9
        
        # Bollinger Bands parameters
        self.bb_period = 20
        self.bb_std = 2
        
        logger.info(f"Technical Analysis Module initialized with config: {self.config}")
    
    def calculate_sma(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Simple Moving Average"""
        return prices.rolling(window=period).mean()
    
    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Exponential Moving Average"""
        return prices.ewm(span=period).mean()
    
    def calculate_atr(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = None) -> pd.Series:
        """Calculate Average True Range (ATR)"""
        if period is None:
            period = self.atr_period
            
        high_low = high - low
        high_close = np.abs(high - close.shift(1))
        low_close = np.abs(low - close.shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()
        return atr
    
    def calculate_rsi(self, prices: pd.Series, period: int = None) -> pd.Series:
        """Calculate Relative Strength Index (RSI)"""
        if period is None:
            period = self.rsi_period
            
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def calculate_macd(self, prices: pd.Series, fast: int = None, slow: int = None, signal: int = None) -> Dict[str, pd.Series]:
        """Calculate MACD (Moving Average Convergence Divergence)"""
        if fast is None:
            fast = self.macd_fast
        if slow is None:
            slow = self.macd_slow
        if signal is None:
            signal = self.macd_signal
            
        ema_fast = self.calculate_ema(prices, fast)
        ema_slow = self.calculate_ema(prices, slow)
        macd_line = ema_fast - ema_slow
        signal_line = self.calculate_ema(macd_line, signal)
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    def calculate_bollinger_bands(self, prices: pd.Series, period: int = None, std: float = None) -> Dict[str, pd.Series]:
        """Calculate Bollinger Bands"""
        if period is None:
            period = self.bb_period
        if std is None:
            std = self.bb_std
            
        sma = self.calculate_sma(prices, period)
        std_dev = prices.rolling(window=period).std()
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }
    
    def calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Dict[str, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_period).min()
        highest_high = high.rolling(window=k_period).max()
        
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }
    
    def calculate_volume_zscore(self, volume: pd.Series, window: int = 20) -> pd.Series:
        """Calculate Volume Z-Score"""
        volume_mean = volume.rolling(window=window).mean()
        volume_std = volume.rolling(window=window).std()
        volume_zscore = (volume - volume_mean) / volume_std
        return volume_zscore
    
    def calculate_adx(self, high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (ADX)"""
        # Calculate True Range
        tr = self.calculate_atr(high, low, close, 1)
        
        # Calculate Directional Movement
        dm_plus = high.diff()
        dm_minus = -low.diff()
        
        dm_plus = dm_plus.where((dm_plus > dm_minus) & (dm_plus > 0), 0)
        dm_minus = dm_minus.where((dm_minus > dm_plus) & (dm_minus > 0), 0)
        
        # Calculate smoothed values
        tr_smooth = tr.rolling(window=period).mean()
        dm_plus_smooth = dm_plus.rolling(window=period).mean()
        dm_minus_smooth = dm_minus.rolling(window=period).mean()
        
        # Calculate DI+ and DI-
        di_plus = 100 * (dm_plus_smooth / tr_smooth)
        di_minus = 100 * (dm_minus_smooth / tr_smooth)
        
        # Calculate DX
        dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus)
        
        # Calculate ADX
        adx = dx.rolling(window=period).mean()
        
        return adx
    
    def calculate_parabolic_sar(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                               acceleration: float = 0.02, maximum: float = 0.2) -> pd.Series:
        """Calculate Parabolic SAR"""
        psar = pd.Series(index=close.index, dtype=float)
        trend = pd.Series(index=close.index, dtype=int)
        af = pd.Series(index=close.index, dtype=float)
        ep = pd.Series(index=close.index, dtype=float)
        
        # Initialize first values
        psar.iloc[0] = low.iloc[0]
        trend.iloc[0] = 1
        af.iloc[0] = acceleration
        ep.iloc[0] = high.iloc[0]
        
        for i in range(1, len(close)):
            # Update trend
            if trend.iloc[i-1] == 1:  # Uptrend
                if low.iloc[i] <= psar.iloc[i-1]:
                    trend.iloc[i] = -1
                    psar.iloc[i] = ep.iloc[i-1]
                    af.iloc[i] = acceleration
                    ep.iloc[i] = low.iloc[i]
                else:
                    trend.iloc[i] = 1
                    if high.iloc[i] > ep.iloc[i-1]:
                        ep.iloc[i] = high.iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + acceleration, maximum)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
                    psar.iloc[i] = psar.iloc[i-1] + af.iloc[i] * (ep.iloc[i] - psar.iloc[i-1])
            else:  # Downtrend
                if high.iloc[i] >= psar.iloc[i-1]:
                    trend.iloc[i] = 1
                    psar.iloc[i] = ep.iloc[i-1]
                    af.iloc[i] = acceleration
                    ep.iloc[i] = high.iloc[i]
                else:
                    trend.iloc[i] = -1
                    if low.iloc[i] < ep.iloc[i-1]:
                        ep.iloc[i] = low.iloc[i]
                        af.iloc[i] = min(af.iloc[i-1] + acceleration, maximum)
                    else:
                        ep.iloc[i] = ep.iloc[i-1]
                        af.iloc[i] = af.iloc[i-1]
                    psar.iloc[i] = psar.iloc[i-1] + af.iloc[i] * (ep.iloc[i] - psar.iloc[i-1])
        
        return psar
    
    def calculate_support_resistance(self, high: pd.Series, low: pd.Series, close: pd.Series, 
                                   window: int = 20, threshold: float = 0.02) -> Dict[str, List[float]]:
        """Calculate Support and Resistance levels"""
        # Find local maxima and minima
        highs = high.rolling(window=window, center=True).max()
        lows = low.rolling(window=window, center=True).min()
        
        # Identify pivot points
        resistance_levels = []
        support_levels = []
        
        for i in range(window, len(high) - window):
            if high.iloc[i] == highs.iloc[i]:
                resistance_levels.append(high.iloc[i])
            if low.iloc[i] == lows.iloc[i]:
                support_levels.append(low.iloc[i])
        
        # Filter levels by frequency and proximity
        resistance_levels = self._filter_levels(resistance_levels, threshold)
        support_levels = self._filter_levels(support_levels, threshold)
        
        return {
            'resistance': resistance_levels,
            'support': support_levels
        }
    
    def _filter_levels(self, levels: List[float], threshold: float) -> List[float]:
        """Filter support/resistance levels by frequency and proximity"""
        if not levels:
            return []
        
        # Group nearby levels
        filtered_levels = []
        levels_sorted = sorted(levels)
        
        current_group = [levels_sorted[0]]
        
        for level in levels_sorted[1:]:
            if abs(level - current_group[-1]) / current_group[-1] <= threshold:
                current_group.append(level)
            else:
                # Average the group and add to filtered levels
                filtered_levels.append(np.mean(current_group))
                current_group = [level]
        
        # Add the last group
        if current_group:
            filtered_levels.append(np.mean(current_group))
        
        return filtered_levels
    
    def calculate_elliott_wave(self, prices: pd.Series, periods: List[int] = None) -> Dict[str, Any]:
        """Calculate Elliott Wave analysis (simplified implementation)"""
        if not self.enable_elliott_wave:
            return {'enabled': False}
        
        if periods is None:
            periods = [5, 13, 21, 34, 55]
        
        # Simplified Elliott Wave implementation
        # This is a basic version - full implementation would be much more complex
        wave_analysis = {
            'enabled': True,
            'current_wave': 'unknown',
            'wave_count': 0,
            'fibonacci_levels': [],
            'trend_direction': 'neutral'
        }
        
        # Calculate Fibonacci retracement levels
        recent_high = prices.rolling(window=20).max().iloc[-1]
        recent_low = prices.rolling(window=20).min().iloc[-1]
        price_range = recent_high - recent_low
        
        fibonacci_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        wave_analysis['fibonacci_levels'] = [
            recent_high - (level * price_range) for level in fibonacci_levels
        ]
        
        # Simple trend detection
        sma_short = self.calculate_sma(prices, 20)
        sma_long = self.calculate_sma(prices, 50)
        
        if sma_short.iloc[-1] > sma_long.iloc[-1]:
            wave_analysis['trend_direction'] = 'bullish'
        elif sma_short.iloc[-1] < sma_long.iloc[-1]:
            wave_analysis['trend_direction'] = 'bearish'
        
        return wave_analysis
    
    def analyze_technical_signals(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Analyze technical indicators and generate trading signals.
        
        Args:
            df: DataFrame with OHLCV data (columns: open, high, low, close, volume)
            
        Returns:
            List of technical analysis signals
        """
        try:
            if df.empty or len(df) < max(self.sma_long, self.atr_period, self.rsi_period):
                logger.warning("Insufficient data for technical analysis")
                return []
            
            # Ensure we have the required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"Missing required columns. Available: {df.columns.tolist()}")
                return []
            
            # Extract OHLCV data
            open_prices = df['open']
            high_prices = df['high']
            low_prices = df['low']
            close_prices = df['close']
            volume = df['volume']
            
            # Calculate technical indicators
            sma_short = self.calculate_sma(close_prices, self.sma_short)
            sma_long = self.calculate_sma(close_prices, self.sma_long)
            atr = self.calculate_atr(high_prices, low_prices, close_prices)
            rsi = self.calculate_rsi(close_prices)
            macd_data = self.calculate_macd(close_prices)
            bb_data = self.calculate_bollinger_bands(close_prices)
            stoch_data = self.calculate_stochastic(high_prices, low_prices, close_prices)
            volume_zscore = self.calculate_volume_zscore(volume)
            adx = self.calculate_adx(high_prices, low_prices, close_prices)
            psar = self.calculate_parabolic_sar(high_prices, low_prices, close_prices)
            
            # Calculate support/resistance levels
            sr_levels = self.calculate_support_resistance(high_prices, low_prices, close_prices)
            
            # Calculate Elliott Wave analysis
            elliott_wave = self.calculate_elliott_wave(close_prices)
            
            # Generate signals
            signals = []
            
            # Get the latest values
            latest_idx = -1
            current_price = close_prices.iloc[latest_idx]
            current_sma_short = sma_short.iloc[latest_idx]
            current_sma_long = sma_long.iloc[latest_idx]
            current_rsi = rsi.iloc[latest_idx]
            current_atr = atr.iloc[latest_idx]
            current_macd = macd_data['macd'].iloc[latest_idx]
            current_macd_signal = macd_data['signal'].iloc[latest_idx]
            current_bb_upper = bb_data['upper'].iloc[latest_idx]
            current_bb_lower = bb_data['lower'].iloc[latest_idx]
            current_stoch_k = stoch_data['k'].iloc[latest_idx]
            current_stoch_d = stoch_data['d'].iloc[latest_idx]
            current_adx = adx.iloc[latest_idx]
            current_psar = psar.iloc[latest_idx]
            current_volume_zscore = volume_zscore.iloc[latest_idx]
            
            # Skip if any values are NaN
            if pd.isna(current_sma_short) or pd.isna(current_sma_long) or pd.isna(current_rsi):
                logger.warning("NaN values in technical indicators, skipping signal generation")
                return []
            
            # Trend Analysis
            trend_signal = 0.0
            trend_strength = 0.0
            
            if current_sma_short > current_sma_long:
                trend_signal = 1.0  # Bullish trend
                trend_strength = (current_sma_short - current_sma_long) / current_sma_long
            elif current_sma_short < current_sma_long:
                trend_signal = -1.0  # Bearish trend
                trend_strength = (current_sma_long - current_sma_short) / current_sma_long
            
            # RSI Analysis
            rsi_signal = 0.0
            if current_rsi > self.rsi_overbought:
                rsi_signal = -1.0  # Overbought - potential sell
            elif current_rsi < self.rsi_oversold:
                rsi_signal = 1.0   # Oversold - potential buy
            
            # MACD Analysis
            macd_signal = 0.0
            if current_macd > current_macd_signal:
                macd_signal = 1.0  # Bullish MACD
            elif current_macd < current_macd_signal:
                macd_signal = -1.0  # Bearish MACD
            
            # Bollinger Bands Analysis
            bb_signal = 0.0
            if current_price > current_bb_upper:
                bb_signal = -1.0  # Price above upper band - potential sell
            elif current_price < current_bb_lower:
                bb_signal = 1.0   # Price below lower band - potential buy
            
            # Stochastic Analysis
            stoch_signal = 0.0
            if current_stoch_k > 80 and current_stoch_d > 80:
                stoch_signal = -1.0  # Overbought
            elif current_stoch_k < 20 and current_stoch_d < 20:
                stoch_signal = 1.0   # Oversold
            
            # ADX Analysis (trend strength)
            trend_strength_adx = 0.0
            if not pd.isna(current_adx):
                if current_adx > 25:
                    trend_strength_adx = min(current_adx / 50, 1.0)  # Normalize to 0-1
            
            # Parabolic SAR Analysis
            psar_signal = 0.0
            if not pd.isna(current_psar):
                if current_price > current_psar:
                    psar_signal = 1.0  # Bullish SAR
                elif current_price < current_psar:
                    psar_signal = -1.0  # Bearish SAR
            
            # Volume Analysis
            volume_signal = 0.0
            if not pd.isna(current_volume_zscore):
                if current_volume_zscore > 2:
                    volume_signal = 1.0  # High volume - confirm signal
                elif current_volume_zscore < -2:
                    volume_signal = -1.0  # Low volume - weak signal
            
            # Volatility Filter
            volatility_signal = 1.0  # Default: allow trading
            if self.volatility_filter_enabled and not pd.isna(current_atr):
                # Calculate volatility percentile
                atr_percentile = (atr.rolling(window=self.volatility_window).rank(pct=True)).iloc[latest_idx]
                if not pd.isna(atr_percentile):
                    if atr_percentile < self.volatility_percentile_low / 100:
                        volatility_signal = 0.5  # Low volatility - reduce signal strength
                    elif atr_percentile > self.volatility_percentile_high / 100:
                        volatility_signal = 0.5  # High volatility - reduce signal strength
            
            # Combine signals with weights
            signal_components = {
                'trend': trend_signal * 0.3,
                'rsi': rsi_signal * 0.2,
                'macd': macd_signal * 0.2,
                'bollinger': bb_signal * 0.15,
                'stochastic': stoch_signal * 0.1,
                'psar': psar_signal * 0.05
            }
            
            # Calculate combined signal
            combined_signal = sum(signal_components.values()) * volatility_signal
            
            # Calculate confidence based on signal agreement
            non_zero_signals = [abs(s) for s in signal_components.values() if s != 0]
            signal_agreement = len(non_zero_signals) / len(signal_components) if signal_components else 0
            
            # Calculate confidence
            confidence = min(signal_agreement * trend_strength_adx, 1.0)
            
            # Create signal object
            signal = {
                'timestamp': datetime.now().isoformat(),
                'signal': combined_signal,
                'confidence': confidence,
                'trend_strength': trend_strength,
                'components': signal_components,
                'indicators': {
                    'sma_short': current_sma_short,
                    'sma_long': current_sma_long,
                    'rsi': current_rsi,
                    'atr': current_atr,
                    'macd': current_macd,
                    'macd_signal': current_macd_signal,
                    'bb_upper': current_bb_upper,
                    'bb_lower': current_bb_lower,
                    'stoch_k': current_stoch_k,
                    'stoch_d': current_stoch_d,
                    'adx': current_adx,
                    'psar': current_psar,
                    'volume_zscore': current_volume_zscore
                },
                'support_resistance': sr_levels,
                'elliott_wave': elliott_wave,
                'volatility_filter': volatility_signal
            }
            
            signals.append(signal)
            
            logger.info(f"Generated technical analysis signal: {combined_signal:.3f} (confidence: {confidence:.3f})")
            
            return signals
            
        except Exception as e:
            logger.error(f"Error in technical analysis: {e}")
            return []
    
    def get_technical_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get a summary of technical analysis for the given data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary with technical analysis summary
        """
        try:
            signals = self.analyze_technical_signals(df)
            
            if not signals:
                return {
                    'status': 'error',
                    'message': 'No signals generated',
                    'timestamp': datetime.now().isoformat()
                }
            
            latest_signal = signals[-1]
            
            return {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'signal': latest_signal['signal'],
                'confidence': latest_signal['confidence'],
                'trend_strength': latest_signal['trend_strength'],
                'indicators': latest_signal['indicators'],
                'components': latest_signal['components'],
                'volatility_filter': latest_signal['volatility_filter']
            }
            
        except Exception as e:
            logger.error(f"Error getting technical summary: {e}")
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Example usage and testing
if __name__ == "__main__":
    # Create sample data for testing
    np.random.seed(42)
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')
    
    # Generate sample OHLCV data
    base_price = 50000
    returns = np.random.normal(0, 0.02, 200)
    prices = [base_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    df = pd.DataFrame({
        'timestamp': dates,
        'open': prices,
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'close': prices,
        'volume': np.random.uniform(1000, 10000, 200)
    })
    
    # Initialize technical analysis module
    config = {
        'sma_short': 20,
        'sma_long': 50,
        'atr_period': 14,
        'volatility_filter_enabled': True
    }
    
    ta_module = TechnicalAnalysisModule(config)
    
    # Generate signals
    signals = ta_module.analyze_technical_signals(df)
    
    if signals:
        print("Technical Analysis Results:")
        print(f"Signal: {signals[0]['signal']:.3f}")
        print(f"Confidence: {signals[0]['confidence']:.3f}")
        print(f"Trend Strength: {signals[0]['trend_strength']:.3f}")
        print(f"Components: {signals[0]['components']}")
    else:
        print("No signals generated")
