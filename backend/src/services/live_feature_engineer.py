#!/usr/bin/env python3
"""
Live Feature Engineering Service for Real-Time Trading Signals.

This service generates all required features from live market data
to ensure ML models have sufficient features for training and inference.
"""

import pandas as pd
import numpy as np
import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import talib

class LiveFeatureEngineer:
    """
    Live feature engineering service that generates all required features
    from real market data for ML model training and inference.
    """
    
    def __init__(self, bybit_api, config: Optional[Dict] = None):
        """
        Initialize the live feature engineer.
        
        Args:
            bybit_api: Bybit API service instance
            config: Configuration dictionary
        """
        self.bybit_api = bybit_api
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Feature engineering parameters
        self.lookback_periods = self.config.get('lookback_periods', 100)
        self.feature_columns = [
            'atr', 'sma_20', 'sma_100', 'rsi', 'volume_zscore',
            'target_class_60m', 'target_reg_60m', 'target_reg_30m', 'target_reg_90m',
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'returns_1h', 'returns_4h', 'returns_24h',
            'volatility_1h', 'volatility_4h', 'volatility_24h',
            'price_position', 'volume_position'
        ]
        
        self.logger.info("Live Feature Engineer initialized")
    
    async def fetch_live_data(self, symbol: str, interval: str = '15m', limit: int = 200) -> pd.DataFrame:
        """
        Fetch live market data for feature engineering.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            limit: Number of candles to fetch
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            # Fetch kline data
            kline_result = await self.bybit_api.get_kline(symbol, interval, limit)
            
            if not kline_result.get('success'):
                self.logger.error(f"Failed to fetch kline data for {symbol}: {kline_result.get('error')}")
                return pd.DataFrame()
            
            kline_data = kline_result.get('data', {}).get('list', [])
            if not kline_data:
                self.logger.error(f"No kline data for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(kline_data, columns=[
                'start_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # Convert data types
            df['start_time'] = pd.to_datetime(df['start_time'], unit='ms')
            df['open'] = df['open'].astype(float)
            df['high'] = df['high'].astype(float)
            df['low'] = df['low'].astype(float)
            df['close'] = df['close'].astype(float)
            df['volume'] = df['volume'].astype(float)
            df['turnover'] = df['turnover'].astype(float)
            
            # Sort by time
            df = df.sort_values('start_time').reset_index(drop=True)
            
            self.logger.info(f"Fetched {len(df)} candles for {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching live data for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators from OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            DataFrame with technical indicators
        """
        try:
            # Ensure we have enough data
            if len(df) < 100:
                self.logger.warning(f"Insufficient data for technical indicators: {len(df)} rows")
                return df
            
            # Calculate ATR (Average True Range)
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # Calculate SMAs
            df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
            df['sma_100'] = talib.SMA(df['close'], timeperiod=100)
            
            # Calculate RSI
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            
            # Calculate volume z-score
            df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
            df['volume_std'] = df['volume'].rolling(window=20).std()
            df['volume_zscore'] = (df['volume'] - df['volume_sma']) / df['volume_std']
            df['volume_zscore'] = df['volume_zscore'].fillna(0)
            
            # Calculate price position (current price relative to recent range)
            df['price_high_20'] = df['high'].rolling(window=20).max()
            df['price_low_20'] = df['low'].rolling(window=20).min()
            df['price_position'] = (df['close'] - df['price_low_20']) / (df['price_high_20'] - df['price_low_20'])
            df['price_position'] = df['price_position'].fillna(0.5)
            
            # Calculate volume position
            df['volume_high_20'] = df['volume'].rolling(window=20).max()
            df['volume_low_20'] = df['volume'].rolling(window=20).min()
            df['volume_position'] = (df['volume'] - df['volume_low_20']) / (df['volume_high_20'] - df['volume_low_20'])
            df['volume_position'] = df['volume_position'].fillna(0.5)
            
            self.logger.info("Technical indicators calculated successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    def calculate_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate time-based features.
        
        Args:
            df: DataFrame with timestamp data
            
        Returns:
            DataFrame with time features
        """
        try:
            # Extract time components
            df['hour'] = df['start_time'].dt.hour
            df['day_of_week'] = df['start_time'].dt.dayofweek
            
            # Create cyclical time features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
            
            self.logger.info("Time features calculated successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating time features: {e}")
            return df
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate return features.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with return features
        """
        try:
            # Calculate returns for different periods
            df['returns_1h'] = df['close'].pct_change(4)  # 4 * 15min = 1h
            df['returns_4h'] = df['close'].pct_change(16)  # 16 * 15min = 4h
            df['returns_24h'] = df['close'].pct_change(96)  # 96 * 15min = 24h
            
            # Calculate volatility
            df['volatility_1h'] = df['returns_1h'].rolling(window=20).std()
            df['volatility_4h'] = df['returns_4h'].rolling(window=20).std()
            df['volatility_24h'] = df['returns_24h'].rolling(window=20).std()
            
            # Fill NaN values
            df['returns_1h'] = df['returns_1h'].fillna(0)
            df['returns_4h'] = df['returns_4h'].fillna(0)
            df['returns_24h'] = df['returns_24h'].fillna(0)
            df['volatility_1h'] = df['volatility_1h'].fillna(0)
            df['volatility_4h'] = df['volatility_4h'].fillna(0)
            df['volatility_24h'] = df['volatility_24h'].fillna(0)
            
            self.logger.info("Return features calculated successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating return features: {e}")
            return df
    
    def calculate_target_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate target features for ML training.
        
        Args:
            df: DataFrame with price data
            
        Returns:
            DataFrame with target features
        """
        try:
            # Calculate future returns for different horizons
            df['future_return_30m'] = df['close'].shift(-2) / df['close'] - 1  # 2 * 15min = 30min
            df['future_return_60m'] = df['close'].shift(-4) / df['close'] - 1  # 4 * 15min = 60min
            df['future_return_90m'] = df['close'].shift(-6) / df['close'] - 1  # 6 * 15min = 90min
            
            # Create binary classification targets
            threshold = 0.001  # 0.1% threshold
            df['target_class_60m'] = (df['future_return_60m'] > threshold).astype(int)
            df['target_reg_30m'] = df['future_return_30m']
            df['target_reg_60m'] = df['future_return_60m']
            df['target_reg_90m'] = df['future_return_90m']
            
            # Fill NaN values
            df['target_class_60m'] = df['target_class_60m'].fillna(0)
            df['target_reg_30m'] = df['target_reg_30m'].fillna(0)
            df['target_reg_60m'] = df['target_reg_60m'].fillna(0)
            df['target_reg_90m'] = df['target_reg_90m'].fillna(0)
            
            self.logger.info("Target features calculated successfully")
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating target features: {e}")
            return df
    
    async def engineer_features(self, symbol: str, interval: str = '15m') -> pd.DataFrame:
        """
        Engineer all features from live market data.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            
        Returns:
            DataFrame with all engineered features
        """
        try:
            self.logger.info(f"Engineering features for {symbol}")
            
            # Fetch live data
            df = await self.fetch_live_data(symbol, interval, self.lookback_periods)
            if df.empty:
                self.logger.error(f"No data available for {symbol}")
                return pd.DataFrame()
            
            # Calculate all features
            df = self.calculate_technical_indicators(df)
            df = self.calculate_time_features(df)
            df = self.calculate_returns(df)
            df = self.calculate_target_features(df)
            
            # Select only the required feature columns
            available_features = [col for col in self.feature_columns if col in df.columns]
            missing_features = [col for col in self.feature_columns if col not in df.columns]
            
            if missing_features:
                self.logger.warning(f"Missing features: {missing_features}")
            
            # Create final feature DataFrame
            feature_df = df[available_features].copy()
            
            # Fill any remaining NaN values
            feature_df = feature_df.fillna(0)
            
            # Remove infinite values
            feature_df = feature_df.replace([np.inf, -np.inf], 0)
            
            self.logger.info(f"Feature engineering completed for {symbol}: {len(available_features)}/{len(self.feature_columns)} features")
            return feature_df
            
        except Exception as e:
            self.logger.error(f"Error engineering features for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_feature_coverage(self, df: pd.DataFrame) -> float:
        """
        Calculate feature coverage percentage.
        
        Args:
            df: DataFrame with features
            
        Returns:
            Feature coverage percentage
        """
        available_features = [col for col in self.feature_columns if col in df.columns]
        return len(available_features) / len(self.feature_columns)
    
    async def validate_features(self, symbol: str) -> Dict:
        """
        Validate that all required features are available.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Validation results
        """
        try:
            # Engineer features
            feature_df = await self.engineer_features(symbol)
            
            if feature_df.empty:
                return {
                    'valid': False,
                    'coverage': 0.0,
                    'missing_features': self.feature_columns,
                    'message': 'No data available'
                }
            
            # Calculate coverage
            coverage = self.get_feature_coverage(feature_df)
            missing_features = [col for col in self.feature_columns if col not in feature_df.columns]
            
            return {
                'valid': coverage >= 0.8,
                'coverage': coverage,
                'missing_features': missing_features,
                'available_features': list(feature_df.columns),
                'message': f'Feature coverage: {coverage:.1%}'
            }
            
        except Exception as e:
            self.logger.error(f"Error validating features for {symbol}: {e}")
            return {
                'valid': False,
                'coverage': 0.0,
                'missing_features': self.feature_columns,
                'message': f'Validation error: {e}'
            }
