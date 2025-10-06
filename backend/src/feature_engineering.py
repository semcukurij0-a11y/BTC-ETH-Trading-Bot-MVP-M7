#!/usr/bin/env python3
"""
Comprehensive Feature Engineering Script for Crypto Trading ML Models.

This script fetches historical and real-time data from Bybit and external APIs,
calculates technical indicators, and generates feature matrices for ML training.
"""

import asyncio
import pandas as pd
import numpy as np
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import requests
import websockets
import talib
from dataclasses import dataclass

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.bybit_api_service import BybitAPIService

@dataclass
class FeatureConfig:
    """Configuration for feature engineering."""
    symbols: List[str] = None
    intervals: List[str] = None
    start_date: str = "2025-08-15"
    end_date: str = None
    data_folder: str = "data/features"
    lookback_periods: int = 200
    technical_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.symbols is None:
            self.symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT']
        if self.intervals is None:
            self.intervals = ['1m', '5m', '15m', '1h']
        if self.end_date is None:
            self.end_date = datetime.now().strftime('%Y-%m-%d')
        if self.technical_weights is None:
            self.technical_weights = {
                'returns': 0.2,
                'atr': 0.1,
                'sma_cross': 0.2,
                'rsi': 0.2,
                'volume_zscore': 0.1
            }

class BybitDataFetcher:
    """Fetches data from real Bybit WebSocket for live market data."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    async def fetch_historical_klines(self, symbol: str, interval: str, start_time: str, end_time: str) -> pd.DataFrame:
        """Fetch historical kline data from Bybit WebSocket."""
        try:
            import websockets
            import json
            import asyncio
            
            # Convert interval to Bybit format
            interval_map = {
                '1m': '1',
                '5m': '5', 
                '15m': '15',
                '1h': '60',
                '4h': '240',
                '1d': 'D'
            }
            
            bybit_interval = interval_map.get(interval, '15')
            
            # WebSocket URL for real Bybit
            ws_url = "wss://stream.bybit.com/v5/public/linear"
            
            kline_data = []
            
            async with websockets.connect(ws_url) as websocket:
                # Subscribe to kline data
                subscribe_msg = {
                    "op": "subscribe",
                    "args": [f"kline.{bybit_interval}.{symbol}"]
                }
                
                await websocket.send(json.dumps(subscribe_msg))
                self.logger.info(f"Subscribed to kline data for {symbol} {interval}")
                
                # Collect data for a reasonable time period
                timeout = 30  # seconds
                start_time_collection = asyncio.get_event_loop().time()
                
                while (asyncio.get_event_loop().time() - start_time_collection) < timeout:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        data = json.loads(message)
                        
                        if 'data' in data and 'kline' in data['data']:
                            kline = data['data']['kline']
                            if kline['symbol'] == symbol:
                                kline_data.append([
                                    kline['start'],
                                    kline['open'],
                                    kline['high'],
                                    kline['low'],
                                    kline['close'],
                                    kline['volume'],
                                    kline['turnover']
                                ])
                                
                                # Stop when we have enough data
                                if len(kline_data) >= 200:
                                    break
                                    
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        self.logger.warning(f"Error processing WebSocket message: {e}")
                        continue
                
                # Unsubscribe
                unsubscribe_msg = {
                    "op": "unsubscribe",
                    "args": [f"kline.{bybit_interval}.{symbol}"]
                }
                await websocket.send(json.dumps(unsubscribe_msg))
            
            if kline_data:
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
                
                self.logger.info(f"Fetched {len(df)} real klines via WebSocket for {symbol} {interval}")
                return df
            else:
                self.logger.error(f"No kline data received via WebSocket for {symbol}")
                return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error fetching klines via WebSocket for {symbol}: {e}")
            return pd.DataFrame()
    
    async def fetch_funding_rates(self, symbol: str, start_time: str, end_time: str) -> pd.DataFrame:
        """Fetch funding rate data from Bybit WebSocket."""
        try:
            import websockets
            import json
            import asyncio
            
            # WebSocket URL for real Bybit
            ws_url = "wss://stream.bybit.com/v5/public/linear"
            
            funding_data = []
            
            async with websockets.connect(ws_url) as websocket:
                # Subscribe to funding rate data
                subscribe_msg = {
                    "op": "subscribe",
                    "args": [f"funding.{symbol}"]
                }
                
                await websocket.send(json.dumps(subscribe_msg))
                self.logger.info(f"Subscribed to funding rate data for {symbol}")
                
                # Collect data for a reasonable time period
                timeout = 20  # seconds
                start_time_collection = asyncio.get_event_loop().time()
                
                while (asyncio.get_event_loop().time() - start_time_collection) < timeout:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                        data = json.loads(message)
                        
                        if 'data' in data and 'funding' in data['data']:
                            funding = data['data']['funding']
                            if funding['symbol'] == symbol:
                                funding_data.append({
                                    'fundingTime': funding['fundingTime'],
                                    'fundingRate': funding['fundingRate']
                                })
                                
                                # Stop when we have enough data
                                if len(funding_data) >= 50:
                                    break
                                    
                    except asyncio.TimeoutError:
                        continue
                    except Exception as e:
                        self.logger.warning(f"Error processing WebSocket message: {e}")
                        continue
                
                # Unsubscribe
                unsubscribe_msg = {
                    "op": "unsubscribe",
                    "args": [f"funding.{symbol}"]
                }
                await websocket.send(json.dumps(unsubscribe_msg))
            
            if funding_data:
                df = pd.DataFrame(funding_data)
                df['fundingTime'] = pd.to_datetime(df['fundingTime'], unit='ms')
                df['fundingRate'] = df['fundingRate'].astype(float)
                
                self.logger.info(f"Fetched {len(df)} funding rate records via WebSocket for {symbol}")
                return df
            else:
                self.logger.warning(f"No funding rate data received via WebSocket for {symbol}")
                return pd.DataFrame()
            
        except Exception as e:
            self.logger.error(f"Error fetching funding rates via WebSocket for {symbol}: {e}")
            return pd.DataFrame()
    

class ExternalDataFetcher:
    """Fetches data from external APIs."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def fetch_fear_greed_index(self, start_date: str, end_date: str) -> pd.DataFrame:
        """Fetch Fear and Greed Index from Alternative.me API."""
        try:
            # Alternative.me API endpoint
            url = "https://api.alternative.me/fng/"
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('name') != 'Fear and Greed Index':
                self.logger.error("Invalid response from Fear and Greed API")
                return pd.DataFrame()
            
            # Extract historical data
            fgi_data = []
            for item in data.get('data', []):
                fgi_data.append({
                    'timestamp': pd.to_datetime(item['timestamp'], unit='s'),
                    'value': int(item['value']),
                    'value_classification': item['value_classification']
                })
            
            df = pd.DataFrame(fgi_data)
            df = df.sort_values('timestamp').reset_index(drop=True)
            
            self.logger.info(f"Fetched {len(df)} Fear and Greed Index records")
            return df
            
        except Exception as e:
            self.logger.error(f"Error fetching Fear and Greed Index: {e}")
            return pd.DataFrame()

class TechnicalAnalyzer:
    """Calculates technical indicators and features."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def calculate_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate return features."""
        try:
            # Calculate returns for different periods
            df['returns_1h'] = df['close'].pct_change(4)  # 4 * 15min = 1h
            df['returns_4h'] = df['close'].pct_change(16)  # 16 * 15min = 4h
            df['returns_24h'] = df['close'].pct_change(96)  # 96 * 15min = 24h
            
            # Fill NaN values
            df['returns_1h'] = df['returns_1h'].fillna(0)
            df['returns_4h'] = df['returns_4h'].fillna(0)
            df['returns_24h'] = df['returns_24h'].fillna(0)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating returns: {e}")
            return df
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate technical indicators."""
        try:
            # ATR(14)
            df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            
            # SMA(20) and SMA(100)
            df['sma_20'] = talib.SMA(df['close'], timeperiod=20)
            df['sma_100'] = talib.SMA(df['close'], timeperiod=100)
            
            # SMA Cross signal
            df['sma_cross'] = np.where(df['sma_20'] > df['sma_100'], 1, -1)
            
            # RSI(14)
            df['rsi'] = talib.RSI(df['close'], timeperiod=14)
            
            # Volume z-score
            df['volume_sma'] = talib.SMA(df['volume'], timeperiod=20)
            df['volume_std'] = df['volume'].rolling(window=20).std()
            df['volume_zscore'] = (df['volume'] - df['volume_sma']) / df['volume_std']
            df['volume_zscore'] = df['volume_zscore'].fillna(0)
            
            # Fill NaN values
            df['atr'] = df['atr'].fillna(0)
            df['sma_20'] = df['sma_20'].fillna(df['close'])
            df['sma_100'] = df['sma_100'].fillna(df['close'])
            df['rsi'] = df['rsi'].fillna(50)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating technical indicators: {e}")
            return df
    
    def calculate_normalized_technical_analysis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate normalized technical analysis with proper weighting."""
        try:
            # Normalize returns (already in percentage form)
            df['returns_norm'] = df['returns_1h'].clip(-0.1, 0.1) / 0.1  # Clip to ±10% and normalize
            
            # Normalize ATR (relative to price)
            df['atr_norm'] = (df['atr'] / df['close']).clip(0, 0.1) / 0.1  # Clip to 10% max
            
            # SMA Cross is already normalized (-1 to 1)
            df['sma_cross_norm'] = df['sma_cross']
            
            # Normalize RSI (0 to 100 -> -1 to 1)
            df['rsi_norm'] = (df['rsi'] - 50) / 50
            
            # Volume z-score is already normalized
            df['volume_zscore_norm'] = df['volume_zscore'].clip(-3, 3) / 3
            
            # Calculate weighted technical analysis
            weights = self.config.technical_weights
            df['technical_analysis'] = (
                weights['returns'] * df['returns_norm'] +
                weights['atr'] * df['atr_norm'] +
                weights['sma_cross'] * df['sma_cross_norm'] +
                weights['rsi'] * df['rsi_norm'] +
                weights['volume_zscore'] * df['volume_zscore_norm']
            )
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating normalized technical analysis: {e}")
            return df
    
    def calculate_fear_greed_features(self, df: pd.DataFrame, fgi_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Fear and Greed Index features."""
        try:
            if fgi_df.empty:
                # Create synthetic FGI if no data available
                df['fgi_value'] = 50  # Neutral
                df['fgi_classification'] = 'Neutral'
                df['fgi_scaled'] = 0.0
                return df
            
            # Merge FGI data with OHLCV data
            df['date'] = df['start_time'].dt.date
            fgi_df['date'] = fgi_df['timestamp'].dt.date
            
            # Forward fill FGI data to match OHLCV timestamps
            merged_df = df.merge(fgi_df[['date', 'value', 'value_classification']], 
                               on='date', how='left')
            
            # Forward fill missing values
            merged_df['value'] = merged_df['value'].fillna(method='ffill')
            merged_df['value_classification'] = merged_df['value_classification'].fillna(method='ffill')
            
            # Scale FGI to -1 to 1 range
            merged_df['fgi_scaled'] = (merged_df['value'] - 50) / 50
            
            # Update original DataFrame
            df['fgi_value'] = merged_df['value']
            df['fgi_classification'] = merged_df['value_classification']
            df['fgi_scaled'] = merged_df['fgi_scaled']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error calculating Fear and Greed features: {e}")
            return df
    
    def align_features_with_shift(self, df: pd.DataFrame) -> pd.DataFrame:
        """Align all features with shift(1) for proper ML training."""
        try:
            # List of feature columns to shift
            feature_columns = [
                'returns_1h', 'returns_4h', 'returns_24h',
                'atr', 'sma_20', 'sma_100', 'sma_cross', 'rsi', 'volume_zscore',
                'technical_analysis', 'fgi_scaled'
            ]
            
            # Shift all features by 1 period
            for col in feature_columns:
                if col in df.columns:
                    df[f'{col}_lag1'] = df[col].shift(1)
            
            # Remove original feature columns (keep only lagged versions)
            for col in feature_columns:
                if col in df.columns and f'{col}_lag1' in df.columns:
                    df = df.drop(columns=[col])
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error aligning features with shift: {e}")
            return df

class FeatureEngineer:
    """Main feature engineering orchestrator."""
    
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.data_fetcher = BybitDataFetcher()
        self.external_fetcher = ExternalDataFetcher()
        self.technical_analyzer = TechnicalAnalyzer(config)
        
        # Create data directories
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories."""
        try:
            base_path = Path(self.config.data_folder)
            base_path.mkdir(parents=True, exist_ok=True)
            
            for symbol in self.config.symbols:
                symbol_path = base_path / symbol.lower()
                symbol_path.mkdir(parents=True, exist_ok=True)
                
            self.logger.info(f"Created data directories in {base_path}")
            
        except Exception as e:
            self.logger.error(f"Error setting up directories: {e}")
    
    async def process_symbol_interval(self, symbol: str, interval: str) -> bool:
        """Process a single symbol and interval combination."""
        try:
            self.logger.info(f"Processing {symbol} {interval}")
            
            # Fetch historical data
            df = await self.data_fetcher.fetch_historical_klines(
                symbol, interval, self.config.start_date, self.config.end_date
            )
            
            if df.empty:
                self.logger.error(f"No real data available for {symbol} {interval}")
                return False
            
            # Fetch Fear and Greed Index
            fgi_df = self.external_fetcher.fetch_fear_greed_index(
                self.config.start_date, self.config.end_date
            )
            
            # Fetch funding rates
            funding_df = await self.data_fetcher.fetch_funding_rates(
                symbol, self.config.start_date, self.config.end_date
            )
            
            # Calculate technical indicators
            df = self.technical_analyzer.calculate_returns(df)
            df = self.technical_analyzer.calculate_technical_indicators(df)
            df = self.technical_analyzer.calculate_normalized_technical_analysis(df)
            df = self.technical_analyzer.calculate_fear_greed_features(df, fgi_df)
            
            # Add funding rate features if available
            if not funding_df.empty:
                df = self._merge_funding_data(df, funding_df)
            
            # Align features with shift(1)
            df = self.technical_analyzer.align_features_with_shift(df)
            
            # Save to Parquet file
            output_path = Path(self.config.data_folder) / symbol.lower() / f"features_{interval}.parquet"
            df.to_parquet(output_path, index=False)
            
            self.logger.info(f"Saved {len(df)} rows to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing {symbol} {interval}: {e}")
            return False
    
    def _merge_funding_data(self, df: pd.DataFrame, funding_df: pd.DataFrame) -> pd.DataFrame:
        """Merge funding rate data with OHLCV data."""
        try:
            # Convert funding time to match OHLCV time format
            funding_df['date'] = funding_df['fundingTime'].dt.date
            df['date'] = df['start_time'].dt.date
            
            # Merge funding data
            merged_df = df.merge(
                funding_df[['date', 'fundingRate']], 
                on='date', 
                how='left'
            )
            
            # Forward fill funding rates
            merged_df['fundingRate'] = merged_df['fundingRate'].fillna(method='ffill')
            
            # Remove temporary date column
            merged_df = merged_df.drop(columns=['date'])
            
            self.logger.info("Merged funding rate data")
            return merged_df
            
        except Exception as e:
            self.logger.error(f"Error merging funding data: {e}")
            return df
    
    
    async def run_feature_engineering(self):
        """Run the complete feature engineering pipeline."""
        try:
            self.logger.info("Starting feature engineering pipeline")
            
            total_tasks = len(self.config.symbols) * len(self.config.intervals)
            completed_tasks = 0
            
            for symbol in self.config.symbols:
                for interval in self.config.intervals:
                    success = await self.process_symbol_interval(symbol, interval)
                    if success:
                        completed_tasks += 1
                    
                    # Small delay to avoid rate limiting
                    await asyncio.sleep(0.1)
            
            self.logger.info(f"Feature engineering completed: {completed_tasks}/{total_tasks} tasks successful")
            
            # Generate summary report
            await self.generate_summary_report()
            
        except Exception as e:
            self.logger.error(f"Error in feature engineering pipeline: {e}")
    
    async def generate_summary_report(self):
        """Generate a summary report of the feature engineering results."""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'symbols': self.config.symbols,
                    'intervals': self.config.intervals,
                    'start_date': self.config.start_date,
                    'end_date': self.config.end_date
                },
                'results': {}
            }
            
            for symbol in self.config.symbols:
                report['results'][symbol] = {}
                
                for interval in self.config.intervals:
                    file_path = Path(self.config.data_folder) / symbol.lower() / f"features_{interval}.parquet"
                    
                    if file_path.exists():
                        df = pd.read_parquet(file_path)
                        report['results'][symbol][interval] = {
                            'rows': len(df),
                            'columns': len(df.columns),
                            'file_size_mb': file_path.stat().st_size / (1024 * 1024),
                            'date_range': {
                                'start': df['start_time'].min().isoformat() if 'start_time' in df.columns else None,
                                'end': df['start_time'].max().isoformat() if 'start_time' in df.columns else None
                            }
                        }
                    else:
                        report['results'][symbol][interval] = {'status': 'not_found'}
            
            # Save report
            report_path = Path(self.config.data_folder) / 'feature_engineering_report.json'
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            self.logger.info(f"Summary report saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}")

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('feature_engineering.log'),
            logging.StreamHandler()
        ]
    )

async def main():
    """Main function to run feature engineering."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("Starting Feature Engineering Pipeline")
        
        # Configuration
        config = FeatureConfig(
            symbols=['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT'],
            intervals=['1m', '5m', '15m', '1h'],
            start_date='2025-08-15',
            data_folder='data/features'
        )
        
        # Initialize and run feature engineering
        engineer = FeatureEngineer(config)
        await engineer.run_feature_engineering()
        
        logger.info("Feature Engineering Pipeline completed successfully")
        
    except Exception as e:
        logger.error(f"Feature Engineering Pipeline failed: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
