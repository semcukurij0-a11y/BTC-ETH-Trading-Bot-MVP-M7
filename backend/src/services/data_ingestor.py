#!/usr/bin/env python3
"""
Data Ingestor Service for Crypto Trading Bot

This service handles fetching OHLCV and funding data from Bybit API
and storing it in parquet files with incremental update capabilities.
"""

import os
import sys
import logging
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import time
import sqlite3
from pathlib import Path

# Import data validation services
try:
    from .data_validator import DataValidator
    from .data_quality_monitor import DataQualityMonitor
except ImportError:
    # Fallback for when validation services are not available
    DataValidator = None
    DataQualityMonitor = None

class DataIngestor:
    """
    Data ingestor service for fetching and storing crypto market data.
    
    Features:
    - Fetches OHLCV data from Bybit API
    - Fetches funding rate data
    - Stores data in parquet files for efficient access
    - Supports incremental updates to avoid re-fetching existing data
    - Tracks fetch history in SQLite database
    """
    
    def __init__(self, 
                 base_url: str = "https://api.bybit.com",
                 ws_url: str = "wss://stream.bybit.com",
                 parquet_folder: str = "data/parquet",
                 db_file: str = "data/sqlite/runs.db",
                 log_level: str = "INFO",
                 revalidation_bars: int = 5,
                 enable_validation: bool = True,
                 enable_quality_monitoring: bool = True):
        """
        Initialize the data ingestor.
        
        Args:
            base_url: Bybit API base URL
            ws_url: Bybit WebSocket URL (for future use)
            parquet_folder: Directory to store parquet files
            db_file: SQLite database file for tracking fetch history
            log_level: Logging level
            revalidation_bars: Number of trailing bars to revalidate for late edits
            enable_validation: Enable real-time data validation
            enable_quality_monitoring: Enable data quality monitoring
        """
        self.base_url = base_url
        self.ws_url = ws_url
        self.parquet_folder = Path(parquet_folder)
        self.db_file = Path(db_file)
        self.revalidation_bars = revalidation_bars
        self.enable_validation = enable_validation
        self.enable_quality_monitoring = enable_quality_monitoring
        
        # Create directories
        self.parquet_folder.mkdir(parents=True, exist_ok=True)
        self.db_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize validation services
        if self.enable_validation and DataValidator is not None:
            self.validator = DataValidator(db_file=str(self.db_file), log_level=log_level)
        else:
            self.validator = None
            
        if self.enable_quality_monitoring and DataQualityMonitor is not None:
            self.quality_monitor = DataQualityMonitor(
                parquet_folder=str(self.parquet_folder),
                db_file=str(self.db_file),
                log_level=log_level
            )
        else:
            self.quality_monitor = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Initialize database
        self._init_database()
        
        self.logger.info(f"DataIngestor initialized with parquet folder: {self.parquet_folder}")
    
    def _calculate_revalidation_window(self, symbol: str, interval: str) -> Optional[datetime]:
        """
        Calculate the start time for revalidation window based on the last N bars from partitioned files.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            
        Returns:
            Start time for revalidation window, or None if no existing data
        """
        try:
            # Look for partitioned OHLCV files
            base_path = self.parquet_folder / "exchange=bybit" / "market=linear_perp" / f"symbol={symbol}" / f"timeframe={interval}"
            
            if not base_path.exists():
                return None
            
            # Find all date directories and get the latest one
            date_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('date=')]
            if not date_dirs:
                return None
            
            # Sort by date and get the latest
            date_dirs.sort(key=lambda x: x.name)
            latest_date_dir = date_dirs[-1]
            
            # Read the parquet file in the latest date directory
            file_path = latest_date_dir / "bars.parquet"
            
            if not file_path.exists():
                return None
            
            # Read the last few records to determine revalidation window
            df = pd.read_parquet(file_path)
            if df.empty:
                return None
            
            # Get the last N bars for revalidation
            last_bars = df.tail(self.revalidation_bars)
            if last_bars.empty:
                return None
            
            # Return the earliest time in the revalidation window
            revalidation_start = last_bars['open_time'].min()
            
            self.logger.info(f"Revalidation window for {symbol} {interval}: {revalidation_start} (last {self.revalidation_bars} bars)")
            return revalidation_start
            
        except Exception as e:
            self.logger.error(f"Error calculating revalidation window: {e}")
            return None
    
    def _init_database(self):
        """Initialize SQLite database for tracking fetch history."""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                # Create tables for tracking fetch history
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS fetch_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        data_type TEXT NOT NULL,  -- 'ohlcv' or 'funding'
                        last_fetch_time TIMESTAMP,
                        last_record_time TIMESTAMP,
                        records_count INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(symbol, interval, data_type)
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS fetch_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_start TIMESTAMP,
                        session_end TIMESTAMP,
                        symbols TEXT,  -- JSON array of symbols
                        intervals TEXT,  -- JSON array of intervals
                        total_records INTEGER,
                        success BOOLEAN,
                        error_message TEXT
                    )
                """)
                
                conn.commit()
                self.logger.info("Database initialized successfully")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def fetch_and_save_data(self, 
                           symbol: str, 
                           interval: str,
                           ohlcv_limit: int = 1000,
                           funding_limit: int = 500,
                           incremental: bool = True,
                           days_back: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch and save OHLCV and funding data for a symbol and interval.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            interval: Time interval (e.g., '1m', '5m', '15m')
            ohlcv_limit: Maximum number of OHLCV records to fetch per request
            funding_limit: Maximum number of funding records to fetch per request
            incremental: Whether to use incremental fetching
            days_back: Number of days to fetch (if None, uses limit-based fetching)
            
        Returns:
            Dictionary with fetch results
        """
        result = {
            "success": True,
            "symbol": symbol,
            "interval": interval,
            "ohlcv_records": 0,
            "funding_records": 0,
            "ohlcv_new_records": 0,  # Track only new records appended
            "funding_new_records": 0,  # Track only new records appended
            "ohlcv_file": None,
            "funding_file": None,
            "error": None
        }
        
        try:
            self.logger.info(f"Starting data fetch for {symbol} {interval}")
            
            # Fetch OHLCV data
            ohlcv_result = self._fetch_ohlcv_data(
                symbol=symbol,
                interval=interval,
                limit=ohlcv_limit,
                incremental=incremental,
                days_back=days_back
            )
            
            if ohlcv_result["success"]:
                result["ohlcv_records"] = ohlcv_result["records_count"]
                result["ohlcv_new_records"] = ohlcv_result.get("new_records_count", 0)
                result["ohlcv_file"] = ohlcv_result["file_path"]
                self.logger.info(f"Fetched {ohlcv_result['records_count']} OHLCV records for {symbol} {interval}")
            else:
                self.logger.error(f"Failed to fetch OHLCV data: {ohlcv_result['error']}")
                result["error"] = ohlcv_result["error"]
                result["success"] = False
            
            # Fetch funding data
            funding_result = self._fetch_funding_data(
                symbol=symbol,
                limit=funding_limit,
                incremental=incremental,
                days_back=days_back
            )
            
            if funding_result["success"]:
                result["funding_records"] = funding_result["records_count"]
                result["funding_new_records"] = funding_result.get("new_records_count", 0)
                result["funding_file"] = funding_result["file_path"]
                self.logger.info(f"Fetched {funding_result['records_count']} funding records for {symbol}")
            else:
                self.logger.warning(f"Failed to fetch funding data: {funding_result['error']}")
                # Don't fail the entire operation for funding data issues
            
            # Log total summary for this symbol/timeframe (only new records)
            total_new_fetched = result["ohlcv_new_records"] + result["funding_new_records"]
            total_new_appended = result["ohlcv_new_records"] + result["funding_new_records"]
            self.logger.info(f"SUMMARY {symbol} {interval}: fetched {total_new_fetched} / appended {total_new_appended}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error in fetch_and_save_data: {e}"
            self.logger.error(error_msg)
            result["success"] = False
            result["error"] = error_msg
            return result
    
    def _fetch_ohlcv_data(self, 
                         symbol: str, 
                         interval: str,
                         limit: int = 1000,
                         incremental: bool = True,
                         days_back: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch OHLCV data from Bybit API and save to parquet.
        
        Args:
            symbol: Trading symbol
            interval: Time interval
            limit: Maximum records per API call
            incremental: Whether to use incremental fetching
            days_back: Number of days to fetch
            
        Returns:
            Dictionary with fetch results
        """
        result = {
            "success": False,
            "records_count": 0,
            "new_records_count": 0,  # Track only new records appended
            "file_path": None,
            "error": None
        }
        
        try:
            # Check existing parquet file first to get the last date
            last_parquet_time = self._get_last_parquet_time(symbol, interval, "ohlcv")
            
            # Get last fetch info from database if incremental
            last_db_fetch_time = None
            if incremental:
                last_db_fetch_time = self._get_last_fetch_time(symbol, interval, "ohlcv")
            
            # Determine fetch strategy with revalidation window
            if last_parquet_time:
                # Calculate revalidation window (last N bars)
                revalidation_start = self._calculate_revalidation_window(symbol, interval)
                
                if revalidation_start and revalidation_start < last_parquet_time:
                    # Use revalidation window if it goes further back than last parquet time
                    start_time = revalidation_start
                    self.logger.info(f"Incremental fetch with revalidation for {symbol} {interval} from: {start_time}")
                else:
                    # Use the last date from parquet file for incremental fetch
                    start_time = last_parquet_time
                    self.logger.info(f"Incremental fetch for {symbol} {interval} from parquet last date: {start_time}")
                
                end_time = datetime.now()
                result = self._fetch_ohlcv_by_date_range(symbol, interval, start_time, end_time, limit)
            elif days_back and not last_db_fetch_time:
                # First time fetch - get data from days_back ago
                end_time = datetime.now()
                start_time = end_time - timedelta(days=days_back)
                self.logger.info(f"First time fetch for {symbol} {interval} - getting data from {days_back} days ago")
                result = self._fetch_ohlcv_by_date_range(symbol, interval, start_time, end_time, limit)
            elif last_db_fetch_time:
                # Fallback to database last fetch time
                start_time = last_db_fetch_time
                end_time = datetime.now()
                self.logger.info(f"Incremental fetch for {symbol} {interval} from database: {start_time}")
                result = self._fetch_ohlcv_by_date_range(symbol, interval, start_time, end_time, limit)
            else:
                # Fetch by limit
                result = self._fetch_ohlcv_by_limit(symbol, interval, limit)
            
            if result["success"]:
                # Update fetch history
                self._update_fetch_history(symbol, interval, "ohlcv", result["last_record_time"], result["records_count"])
            
            return result
            
        except Exception as e:
            error_msg = f"Error fetching OHLCV data: {e}"
            self.logger.error(error_msg)
            result["error"] = error_msg
            return result
    
    def _fetch_ohlcv_by_date_range(self, 
                                  symbol: str, 
                                  interval: str,
                                  start_time: datetime,
                                  end_time: datetime,
                                  limit: int = 2000) -> Dict[str, Any]:
        """Fetch OHLCV data by date range using multiple API calls."""
        result = {
            "success": False,
            "records_count": 0,
            "file_path": None,
            "last_record_time": None,
            "error": None
        }
        
        try:
            all_data = []
            current_start = start_time
            
            while current_start < end_time:
                # Calculate batch size based on interval and limit
                if interval == '1m':
                    # For 1-minute data, limit to ensure we don't exceed API limits
                    batch_hours = min(24, limit // 60)  # Ensure we don't exceed limit
                elif interval == '5m':
                    batch_hours = min(24, limit // 12)  # 5-minute intervals: 12 per hour
                elif interval == '15m':
                    batch_hours = min(24, limit // 4)   # 15-minute intervals: 4 per hour
                else:
                    batch_hours = 24  # Default to 24 hours for other intervals
                
                current_end = min(current_start + timedelta(hours=batch_hours), end_time)
                
                # Convert to milliseconds
                start_ms = int(current_start.timestamp() * 1000)
                end_ms = int(current_end.timestamp() * 1000)
                
                # Fetch data for this batch
                batch_data = self._fetch_ohlcv_batch(symbol, interval, start_ms, end_ms, limit)
                
                if batch_data:
                    all_data.extend(batch_data)
                    self.logger.info(f"Fetched {len(batch_data)} records from {current_start} to {current_end}")
                    
                    # Move to the next batch by advancing start time by the batch duration
                    # This ensures no overlap and efficient coverage
                    current_start = current_end
                else:
                    # If no data returned, advance by the batch duration
                    current_start = current_end
                
                # Rate limiting
                time.sleep(0.1)
            
            if all_data:
                # Fetch historical mark and index price data for the same time range
                mark_data = []
                index_data = []
                
                # Fetch mark and index price data in batches to match OHLCV data
                current_start = start_time
                while current_start < end_time:
                    # Calculate batch size based on interval and limit
                    if interval == '1m':
                        batch_hours = min(24, limit // 60)
                    elif interval == '5m':
                        batch_hours = min(24, limit // 12)
                    elif interval == '15m':
                        batch_hours = min(24, limit // 4)
                    else:
                        batch_hours = 24
                    
                    current_end = min(current_start + timedelta(hours=batch_hours), end_time)
                    
                    # Convert to milliseconds
                    start_ms = int(current_start.timestamp() * 1000)
                    end_ms = int(current_end.timestamp() * 1000)
                    
                    # Fetch mark price data
                    mark_batch = self._fetch_mark_price_kline(symbol, interval, start_ms, end_ms, limit)
                    if mark_batch:
                        mark_data.extend(mark_batch)
                    
                    # Fetch index price data
                    index_batch = self._fetch_index_price_kline(symbol, interval, start_ms, end_ms, limit)
                    if index_batch:
                        index_data.extend(index_batch)
                    
                    current_start = current_end
                    time.sleep(0.1)  # Rate limiting
                
                # Convert to DataFrame with Bybit data format
                df = pd.DataFrame(all_data, columns=[
                    'startTime', 'openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume',
                    'turnover'
                ])
                
                # Convert time column to readable datetime format
                df['open_time'] = pd.to_datetime(pd.to_numeric(df['startTime']), unit='ms')
                df['startTime'] = df['open_time'].dt.strftime('%Y.%m.%d %H:%M:%S')  # Convert to readable string format
                
                # Rename price columns to match expected format
                df['open'] = pd.to_numeric(df['openPrice'], errors='coerce')
                df['high'] = pd.to_numeric(df['highPrice'], errors='coerce')
                df['low'] = pd.to_numeric(df['lowPrice'], errors='coerce')
                df['close'] = pd.to_numeric(df['closePrice'], errors='coerce')
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                
                # Add quote volume (turnover from Bybit)
                df['quote_volume'] = pd.to_numeric(df['turnover'], errors='coerce')
                
                # Process mark price data
                if mark_data:
                    mark_df = pd.DataFrame(mark_data, columns=['startTime', 'mark_open', 'mark_high', 'mark_low', 'mark_close'])
                    mark_df['mark_time'] = pd.to_datetime(pd.to_numeric(mark_df['startTime']), unit='ms')
                    mark_df['mark_open'] = pd.to_numeric(mark_df['mark_open'], errors='coerce')
                    mark_df['mark_high'] = pd.to_numeric(mark_df['mark_high'], errors='coerce')
                    mark_df['mark_low'] = pd.to_numeric(mark_df['mark_low'], errors='coerce')
                    mark_df['mark_close'] = pd.to_numeric(mark_df['mark_close'], errors='coerce')
                    
                    # Merge mark price data with main dataframe
                    df = df.merge(mark_df[['mark_time', 'mark_open', 'mark_high', 'mark_low', 'mark_close']], 
                                 left_on='open_time', right_on='mark_time', how='left')
                    df = df.drop(columns=['mark_time'])
                else:
                    # Fallback to regular OHLCV data if no mark price data
                    df['mark_open'] = df['open']
                    df['mark_high'] = df['high']
                    df['mark_low'] = df['low']
                    df['mark_close'] = df['close']
                
                # Process index price data
                if index_data:
                    index_df = pd.DataFrame(index_data, columns=['startTime', 'index_open', 'index_high', 'index_low', 'index_close'])
                    index_df['index_time'] = pd.to_datetime(pd.to_numeric(index_df['startTime']), unit='ms')
                    index_df['index_open'] = pd.to_numeric(index_df['index_open'], errors='coerce')
                    index_df['index_high'] = pd.to_numeric(index_df['index_high'], errors='coerce')
                    index_df['index_low'] = pd.to_numeric(index_df['index_low'], errors='coerce')
                    index_df['index_close'] = pd.to_numeric(index_df['index_close'], errors='coerce')
                    
                    # Merge index price data with main dataframe
                    df = df.merge(index_df[['index_time', 'index_open', 'index_high', 'index_low', 'index_close']], 
                                 left_on='open_time', right_on='index_time', how='left')
                    df = df.drop(columns=['index_time'])
                else:
                    # Fallback to regular OHLCV data if no index price data
                    df['index_open'] = df['open']
                    df['index_high'] = df['high']
                    df['index_low'] = df['low']
                    df['index_close'] = df['close']
                
                # Remove raw Bybit columns and redundant time columns
                df = df.drop(columns=['openPrice', 'highPrice', 'lowPrice', 'closePrice', 'turnover', 'startTime'])
                
                # Remove duplicates and sort
                df = df.drop_duplicates(subset=['open_time']).sort_values('open_time')
                
                # Save to parquet
                file_path, new_records_count = self._save_ohlcv_to_parquet(df, symbol, interval)
                
                result["success"] = True
                result["records_count"] = len(df)
                result["new_records_count"] = new_records_count
                result["file_path"] = file_path
                result["last_record_time"] = df['open_time'].max()
                
                self.logger.info(f"Successfully saved {len(df)} OHLCV records to {file_path}")
            else:
                result["error"] = "No data fetched"
            
            return result
            
        except Exception as e:
            error_msg = f"Error in _fetch_ohlcv_by_date_range: {e}"
            self.logger.error(error_msg)
            result["error"] = error_msg
            return result
    
    def _fetch_ohlcv_batch(self, 
                          symbol: str, 
                          interval: str,
                          start_time: int,
                          end_time: int,
                          limit: int = 2000) -> List[List]:
        """Fetch a batch of OHLCV data from Bybit API."""
        try:
            # Convert Binance interval format to Bybit format
            interval_map = {
                '1m': '1', '3m': '3', '5m': '5', '15m': '15', 
                '30m': '30', '1h': '60', '2h': '120', '4h': '240',
                '6h': '360', '12h': '720', '1d': 'D', '1w': 'W'
            }
            bybit_interval = interval_map.get(interval, '1')
            
            url = f"{self.base_url}/v5/market/kline"
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': bybit_interval,
                'start': start_time,
                'end': end_time,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for Bybit API errors
            if data.get("retCode") != 0:
                raise Exception(f"Bybit API Error: {data.get('retMsg', 'Unknown error')}")
            
            # Extract the kline data from Bybit response
            if "result" in data and "list" in data["result"]:
                return data["result"]["list"]
            else:
                return []
            
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV batch: {e}")
            return []
    
    def _fetch_mark_price_kline(self, symbol: str, interval: str, start_time: int, end_time: int, limit: int = 2000) -> List[List]:
        """Fetch historical mark price OHLC data from Bybit API."""
        try:
            # Convert interval to Bybit format
            interval_map = {
                '1m': '1', '3m': '3', '5m': '5', '15m': '15', 
                '30m': '30', '1h': '60', '2h': '120', '4h': '240',
                '6h': '360', '12h': '720', '1d': 'D', '1w': 'W'
            }
            bybit_interval = interval_map.get(interval, '1')
            
            url = f"{self.base_url}/v5/market/mark-price-kline"
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': bybit_interval,
                'start': start_time,
                'end': end_time,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("retCode") != 0:
                raise Exception(f"Bybit mark price API Error: {data.get('retMsg', 'Unknown error')}")
            
            if "result" in data and "list" in data["result"]:
                return data["result"]["list"]
            else:
                return []
            
        except Exception as e:
            self.logger.error(f"Error fetching mark price kline for {symbol}: {e}")
            return []
    
    def _fetch_index_price_kline(self, symbol: str, interval: str, start_time: int, end_time: int, limit: int = 2000) -> List[List]:
        """Fetch historical index price OHLC data from Bybit API."""
        try:
            # Convert interval to Bybit format
            interval_map = {
                '1m': '1', '3m': '3', '5m': '5', '15m': '15', 
                '30m': '30', '1h': '60', '2h': '120', '4h': '240',
                '6h': '360', '12h': '720', '1d': 'D', '1w': 'W'
            }
            bybit_interval = interval_map.get(interval, '1')
            
            url = f"{self.base_url}/v5/market/index-price-kline"
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': bybit_interval,
                'start': start_time,
                'end': end_time,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("retCode") != 0:
                raise Exception(f"Bybit index price API Error: {data.get('retMsg', 'Unknown error')}")
            
            if "result" in data and "list" in data["result"]:
                return data["result"]["list"]
            else:
                return []
            
        except Exception as e:
            self.logger.error(f"Error fetching index price kline for {symbol}: {e}")
            return []
    
    def _fetch_ohlcv_by_limit(self, 
                             symbol: str, 
                             interval: str,
                             limit: int = 2000) -> Dict[str, Any]:
        """Fetch OHLCV data by limit (most recent records)."""
        result = {
            "success": False,
            "records_count": 0,
            "file_path": None,
            "last_record_time": None,
            "error": None
        }
        
        try:
            # Convert Binance interval format to Bybit format
            interval_map = {
                '1m': '1', '3m': '3', '5m': '5', '15m': '15', 
                '30m': '30', '1h': '60', '2h': '120', '4h': '240',
                '6h': '360', '12h': '720', '1d': 'D', '1w': 'W'
            }
            bybit_interval = interval_map.get(interval, '1')
            
            url = f"{self.base_url}/v5/market/kline"
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': bybit_interval,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for Bybit API errors
            if data.get("retCode") != 0:
                raise Exception(f"Bybit API Error: {data.get('retMsg', 'Unknown error')}")
            
            if "result" in data and "list" in data["result"] and data["result"]["list"]:
                # Get the time range from the OHLCV data to fetch corresponding mark and index prices
                ohlcv_data = data["result"]["list"]
                if ohlcv_data:
                    # Get start and end times from OHLCV data
                    start_ms = int(ohlcv_data[0][0])  # First record timestamp
                    end_ms = int(ohlcv_data[-1][0])   # Last record timestamp
                    
                    # Fetch mark and index price data for the same time range
                    mark_data = self._fetch_mark_price_kline(symbol, interval, start_ms, end_ms, limit)
                    index_data = self._fetch_index_price_kline(symbol, interval, start_ms, end_ms, limit)
                else:
                    mark_data = []
                    index_data = []
                
                # Convert to DataFrame with Bybit data format
                df = pd.DataFrame(ohlcv_data, columns=[
                    'startTime', 'openPrice', 'highPrice', 'lowPrice', 'closePrice', 'volume',
                    'turnover'
                ])
                
                # Convert time column to readable datetime format
                df['open_time'] = pd.to_datetime(pd.to_numeric(df['startTime']), unit='ms')
                df['startTime'] = df['open_time'].dt.strftime('%Y.%m.%d %H:%M:%S')  # Convert to readable string format
                
                # Rename price columns to match expected format
                df['open'] = pd.to_numeric(df['openPrice'], errors='coerce')
                df['high'] = pd.to_numeric(df['highPrice'], errors='coerce')
                df['low'] = pd.to_numeric(df['lowPrice'], errors='coerce')
                df['close'] = pd.to_numeric(df['closePrice'], errors='coerce')
                df['volume'] = pd.to_numeric(df['volume'], errors='coerce')
                
                # Add quote volume (turnover from Bybit)
                df['quote_volume'] = pd.to_numeric(df['turnover'], errors='coerce')
                
                # Process mark price data
                if mark_data:
                    mark_df = pd.DataFrame(mark_data, columns=['startTime', 'mark_open', 'mark_high', 'mark_low', 'mark_close'])
                    mark_df['mark_time'] = pd.to_datetime(pd.to_numeric(mark_df['startTime']), unit='ms')
                    mark_df['mark_open'] = pd.to_numeric(mark_df['mark_open'], errors='coerce')
                    mark_df['mark_high'] = pd.to_numeric(mark_df['mark_high'], errors='coerce')
                    mark_df['mark_low'] = pd.to_numeric(mark_df['mark_low'], errors='coerce')
                    mark_df['mark_close'] = pd.to_numeric(mark_df['mark_close'], errors='coerce')
                    
                    # Merge mark price data with main dataframe
                    df = df.merge(mark_df[['mark_time', 'mark_open', 'mark_high', 'mark_low', 'mark_close']], 
                                 left_on='open_time', right_on='mark_time', how='left')
                    df = df.drop(columns=['mark_time'])
                else:
                    # Fallback to regular OHLCV data if no mark price data
                    df['mark_open'] = df['open']
                    df['mark_high'] = df['high']
                    df['mark_low'] = df['low']
                    df['mark_close'] = df['close']
                
                # Process index price data
                if index_data:
                    index_df = pd.DataFrame(index_data, columns=['startTime', 'index_open', 'index_high', 'index_low', 'index_close'])
                    index_df['index_time'] = pd.to_datetime(pd.to_numeric(index_df['startTime']), unit='ms')
                    index_df['index_open'] = pd.to_numeric(index_df['index_open'], errors='coerce')
                    index_df['index_high'] = pd.to_numeric(index_df['index_high'], errors='coerce')
                    index_df['index_low'] = pd.to_numeric(index_df['index_low'], errors='coerce')
                    index_df['index_close'] = pd.to_numeric(index_df['index_close'], errors='coerce')
                    
                    # Merge index price data with main dataframe
                    df = df.merge(index_df[['index_time', 'index_open', 'index_high', 'index_low', 'index_close']], 
                                 left_on='open_time', right_on='index_time', how='left')
                    df = df.drop(columns=['index_time'])
                else:
                    # Fallback to regular OHLCV data if no index price data
                    df['index_open'] = df['open']
                    df['index_high'] = df['high']
                    df['index_low'] = df['low']
                    df['index_close'] = df['close']
                
                # Remove raw Bybit columns and redundant time columns
                df = df.drop(columns=['openPrice', 'highPrice', 'lowPrice', 'closePrice', 'turnover', 'startTime'])
                
                # Save to parquet
                file_path, new_records_count = self._save_ohlcv_to_parquet(df, symbol, interval)
                
                result["success"] = True
                result["records_count"] = len(df)
                result["new_records_count"] = new_records_count
                result["file_path"] = file_path
                result["last_record_time"] = df['open_time'].max()
                
                self.logger.info(f"Successfully saved {len(df)} OHLCV records to {file_path}")
            else:
                result["error"] = "No data received from API"
            
            return result
            
        except Exception as e:
            error_msg = f"Error in _fetch_ohlcv_by_limit: {e}"
            self.logger.error(error_msg)
            result["error"] = error_msg
            return result
    
    def _fetch_funding_data(self, 
                           symbol: str,
                           limit: int = 500,
                           incremental: bool = True,
                           days_back: Optional[int] = None) -> Dict[str, Any]:
        """
        Fetch funding rate data from Bybit API and save to parquet.
        
        Args:
            symbol: Trading symbol
            limit: Maximum records per API call
            incremental: Whether to use incremental fetching
            days_back: Number of days to fetch
            
        Returns:
            Dictionary with fetch results
        """
        result = {
            "success": False,
            "records_count": 0,
            "new_records_count": 0,  # Track only new records appended
            "file_path": None,
            "error": None
        }
        
        try:
            # Check existing parquet file first to get the last date
            last_parquet_time = self._get_last_parquet_time(symbol, "1h", "funding")
            
            # Get last fetch info from database if incremental
            last_db_fetch_time = None
            if incremental:
                last_db_fetch_time = self._get_last_fetch_time(symbol, "1h", "funding")  # Funding is typically hourly
            
            # Determine fetch strategy
            if last_parquet_time:
                # Use the last date from parquet file for incremental fetch
                start_time = last_parquet_time
                end_time = datetime.now()
                self.logger.info(f"Incremental funding fetch for {symbol} from parquet last date: {start_time}")
                result = self._fetch_funding_by_date_range(symbol, start_time, end_time, limit)
            elif days_back and not last_db_fetch_time:
                # First time fetch - get data from days_back ago
                end_time = datetime.now()
                start_time = end_time - timedelta(days=days_back)
                self.logger.info(f"First time funding fetch for {symbol} - getting data from {days_back} days ago")
                result = self._fetch_funding_by_date_range(symbol, start_time, end_time, limit)
            elif last_db_fetch_time:
                # Fallback to database last fetch time
                start_time = last_db_fetch_time
                end_time = datetime.now()
                self.logger.info(f"Incremental funding fetch for {symbol} from database: {start_time}")
                result = self._fetch_funding_by_date_range(symbol, start_time, end_time, limit)
            else:
                # Fetch by limit
                result = self._fetch_funding_by_limit(symbol, limit)
            
            if result["success"]:
                # Update fetch history
                self._update_fetch_history(symbol, "1h", "funding", result.get("last_record_time"), result["records_count"])
            
            return result
            
        except Exception as e:
            error_msg = f"Error fetching funding data: {e}"
            self.logger.error(error_msg)
            result["error"] = error_msg
            return result
    
    def _fetch_funding_by_date_range(self, 
                                    symbol: str,
                                    start_time: datetime,
                                    end_time: datetime,
                                    limit: int = 500) -> Dict[str, Any]:
        """Fetch funding data by date range."""
        result = {
            "success": False,
            "records_count": 0,
            "file_path": None,
            "last_record_time": None,
            "error": None
        }
        
        try:
            # Convert to milliseconds
            start_ms = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)
            
            url = f"{self.base_url}/v5/market/funding/history"
            params = {
                'category': 'linear',
                'symbol': symbol,
                'startTime': start_ms,
                'endTime': end_ms,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for Bybit API errors
            if data.get("retCode") != 0:
                raise Exception(f"Bybit API Error: {data.get('retMsg', 'Unknown error')}")
            
            if "result" in data and "list" in data["result"] and data["result"]["list"]:
                # Convert to DataFrame
                df = pd.DataFrame(data["result"]["list"])
                
                # Convert time column
                df['fundingTime'] = pd.to_datetime(pd.to_numeric(df['fundingRateTimestamp']), unit='ms')
                
                # Convert numeric columns
                df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
                
                # Remove raw Bybit columns that are no longer needed
                df = df.drop(columns=['fundingRateTimestamp'])
                
                # Sort by time
                df = df.sort_values('fundingTime')
                
                # Save to parquet
                file_path, new_records_count = self._save_funding_to_parquet(df, symbol)
                
                result["success"] = True
                result["records_count"] = len(df)
                result["new_records_count"] = new_records_count
                result["file_path"] = file_path
                result["last_record_time"] = df['fundingTime'].max()
                
                self.logger.info(f"Successfully saved {len(df)} funding records to {file_path}")
            else:
                result["error"] = "No funding data received from API"
            
            return result
            
        except Exception as e:
            error_msg = f"Error in _fetch_funding_by_date_range: {e}"
            self.logger.error(error_msg)
            result["error"] = error_msg
            return result
    
    def _fetch_funding_by_limit(self, 
                               symbol: str,
                               limit: int = 500) -> Dict[str, Any]:
        """Fetch funding data by limit (most recent records)."""
        result = {
            "success": False,
            "records_count": 0,
            "file_path": None,
            "last_record_time": None,
            "error": None
        }
        
        try:
            url = f"{self.base_url}/v5/market/funding/history"
            params = {
                'category': 'linear',
                'symbol': symbol,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            # Check for Bybit API errors
            if data.get("retCode") != 0:
                raise Exception(f"Bybit API Error: {data.get('retMsg', 'Unknown error')}")
            
            if "result" in data and "list" in data["result"] and data["result"]["list"]:
                # Convert to DataFrame
                df = pd.DataFrame(data["result"]["list"])
                
                # Convert time column
                df['fundingTime'] = pd.to_datetime(pd.to_numeric(df['fundingRateTimestamp']), unit='ms')
                
                # Convert numeric columns
                df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
                
                # Remove raw Bybit columns that are no longer needed
                df = df.drop(columns=['fundingRateTimestamp'])
                
                # Sort by time
                df = df.sort_values('fundingTime')
                
                # Save to parquet
                file_path, new_records_count = self._save_funding_to_parquet(df, symbol)
                
                result["success"] = True
                result["records_count"] = len(df)
                result["new_records_count"] = new_records_count
                result["file_path"] = file_path
                result["last_record_time"] = df['fundingTime'].max()
                
                self.logger.info(f"Successfully saved {len(df)} funding records to {file_path}")
            else:
                result["error"] = "No funding data received from API"
            
            return result
            
        except Exception as e:
            error_msg = f"Error in _fetch_funding_by_limit: {e}"
            self.logger.error(error_msg)
            result["error"] = error_msg
            return result
    
    def _save_ohlcv_to_parquet(self, df: pd.DataFrame, symbol: str, interval: str) -> Tuple[str, int]:
        """Save OHLCV DataFrame to parquet file with daily partitioning and uniqueness enforcement.
        
        Returns:
            Tuple of (file_path, new_records_count)
        """
        try:
            # Validate data before saving
            if self.enable_validation and self.validator:
                validation_result = self.validator.validate_ohlcv_data(df, symbol, interval)
                if not validation_result['valid']:
                    critical_issues = [issue for issue in validation_result['issues'] 
                                     if issue.severity.value == 'critical']
                    if critical_issues:
                        self.logger.warning(f"Critical validation issues found for {symbol} {interval}: {len(critical_issues)} issues")
                        # Log critical issues but continue saving
                        for issue in critical_issues:
                            self.logger.warning(f"  - {issue.rule_name}: {issue.message}")
                    else:
                        self.logger.info(f"Data validation passed for {symbol} {interval} (score: {validation_result['score']:.2f})")
                else:
                    self.logger.info(f"Data validation passed for {symbol} {interval} (score: {validation_result['score']:.2f})")
            # Create partitioned directory structure
            # data/parquet/exchange=bybit/market=linear_perp/symbol=BTCUSDT/timeframe=15m/date=YYYY-MM-DD/bars.parquet
            base_path = self.parquet_folder / "exchange=bybit" / "market=linear_perp" / f"symbol={symbol}" / f"timeframe={interval}"
            
            # Group data by date and save each day separately
            df['date'] = df['open_time'].dt.date
            saved_files = []
            total_new_records = 0
            
            for date, day_df in df.groupby('date'):
                date_str = date.strftime('%Y-%m-%d')
                day_path = base_path / f"date={date_str}"
                day_path.mkdir(parents=True, exist_ok=True)
                
                file_path = day_path / "bars.parquet"
                
                # Remove date column before saving
                day_df = day_df.drop(columns=['date'])
                
                # Check if file exists for appending
                if file_path.exists():
                    # Load existing data
                    existing_df = pd.read_parquet(file_path)
                    
                    # Find truly new records by checking which open_time values don't exist in existing data
                    existing_times = set(existing_df['open_time'])
                    new_records_mask = ~day_df['open_time'].isin(existing_times)
                    new_records_df = day_df[new_records_mask]
                    new_records_count = len(new_records_df)
                    
                    # Combine dataframes
                    combined_df = pd.concat([existing_df, day_df], ignore_index=True)
                    
                    # Enforce uniqueness constraint: (symbol, timeframe, open_time)
                    initial_count = len(combined_df)
                    combined_df = combined_df.drop_duplicates(subset=['open_time'], keep='last').sort_values('open_time')
                    final_count = len(combined_df)
                    
                    # Log deduplication results
                    duplicates_removed = initial_count - final_count
                    if duplicates_removed > 0:
                        self.logger.info(f"Removed {duplicates_removed} duplicate OHLCV records (symbol={symbol}, interval={interval}, date={date_str})")
                    
                    # Save combined data
                    combined_df.to_parquet(file_path, index=False)
                    total_new_records += new_records_count
                    self.logger.info(f"Saved OHLCV data: {new_records_count} new records, {final_count} total records (symbol={symbol}, interval={interval}, date={date_str})")
                else:
                    # Create new file - ensure uniqueness
                    initial_count = len(day_df)
                    day_df_unique = day_df.drop_duplicates(subset=['open_time'], keep='last').sort_values('open_time')
                    final_count = len(day_df_unique)
                    
                    duplicates_removed = initial_count - final_count
                    if duplicates_removed > 0:
                        self.logger.info(f"Removed {duplicates_removed} duplicate OHLCV records in new file (symbol={symbol}, interval={interval}, date={date_str})")
                    
                    day_df_unique.to_parquet(file_path, index=False)
                    new_records = final_count  # All records in new file are new
                    total_new_records += new_records
                    self.logger.info(f"Created new OHLCV parquet file with {final_count} records (symbol={symbol}, interval={interval}, date={date_str})")
                
                saved_files.append(str(file_path))
            
            return (saved_files[0] if saved_files else None, total_new_records)
            
        except Exception as e:
            self.logger.error(f"Error saving OHLCV to parquet: {e}")
            raise
    
    def _save_funding_to_parquet(self, df: pd.DataFrame, symbol: str) -> Tuple[str, int]:
        """Save funding DataFrame to parquet file with daily partitioning and uniqueness enforcement.
        
        Returns:
            Tuple of (file_path, new_records_count)
        """
        try:
            # Validate funding data before saving
            if self.enable_validation and self.validator:
                validation_result = self.validator.validate_funding_data(df, symbol)
                if not validation_result['valid']:
                    critical_issues = [issue for issue in validation_result['issues'] 
                                     if issue.severity.value == 'critical']
                    if critical_issues:
                        self.logger.warning(f"Critical validation issues found for {symbol} funding: {len(critical_issues)} issues")
                        # Log critical issues but continue saving
                        for issue in critical_issues:
                            self.logger.warning(f"  - {issue.rule_name}: {issue.message}")
                    else:
                        self.logger.info(f"Funding data validation passed for {symbol} (score: {validation_result['score']:.2f})")
                else:
                    self.logger.info(f"Funding data validation passed for {symbol} (score: {validation_result['score']:.2f})")
            # Create partitioned directory structure
            # data/parquet/exchange=bybit/market=linear_perp/symbol=BTCUSDT/funding/date=YYYY-MM-DD/funding.parquet
            base_path = self.parquet_folder / "exchange=bybit" / "market=linear_perp" / f"symbol={symbol}" / "funding"
            
            # Group data by date and save each day separately
            df['date'] = df['fundingTime'].dt.date
            saved_files = []
            total_new_records = 0
            
            for date, day_df in df.groupby('date'):
                date_str = date.strftime('%Y-%m-%d')
                day_path = base_path / f"date={date_str}"
                day_path.mkdir(parents=True, exist_ok=True)
                
                file_path = day_path / "funding.parquet"
                
                # Remove date column before saving
                day_df = day_df.drop(columns=['date'])
                
                # Check if file exists for appending
                if file_path.exists():
                    # Load existing data
                    existing_df = pd.read_parquet(file_path)
                    
                    # Find truly new records by checking which fundingTime values don't exist in existing data
                    existing_times = set(existing_df['fundingTime'])
                    new_records_mask = ~day_df['fundingTime'].isin(existing_times)
                    new_records_df = day_df[new_records_mask]
                    new_records_count = len(new_records_df)
                    
                    # Combine dataframes
                    combined_df = pd.concat([existing_df, day_df], ignore_index=True)
                    
                    # Enforce uniqueness constraint: (symbol, funding_time)
                    initial_count = len(combined_df)
                    combined_df = combined_df.drop_duplicates(subset=['fundingTime'], keep='last').sort_values('fundingTime')
                    final_count = len(combined_df)
                    
                    # Log deduplication results
                    duplicates_removed = initial_count - final_count
                    if duplicates_removed > 0:
                        self.logger.info(f"Removed {duplicates_removed} duplicate funding records (symbol={symbol}, date={date_str})")
                    
                    # Save combined data
                    combined_df.to_parquet(file_path, index=False)
                    total_new_records += new_records_count
                    self.logger.info(f"Saved funding data: {new_records_count} new records, {final_count} total records (symbol={symbol}, date={date_str})")
                else:
                    # Create new file - ensure uniqueness
                    initial_count = len(day_df)
                    day_df_unique = day_df.drop_duplicates(subset=['fundingTime'], keep='last').sort_values('fundingTime')
                    final_count = len(day_df_unique)
                    
                    duplicates_removed = initial_count - final_count
                    if duplicates_removed > 0:
                        self.logger.info(f"Removed {duplicates_removed} duplicate funding records in new file (symbol={symbol}, date={date_str})")
                    
                    day_df_unique.to_parquet(file_path, index=False)
                    new_records = final_count  # All records in new file are new
                    total_new_records += new_records
                    self.logger.info(f"Created new funding parquet file with {final_count} records (symbol={symbol}, date={date_str})")
                
                saved_files.append(str(file_path))
            
            return (saved_files[0] if saved_files else None, total_new_records)
            
        except Exception as e:
            self.logger.error(f"Error saving funding to parquet: {e}")
            raise
    
    def _get_last_parquet_time(self, symbol: str, interval: str, data_type: str) -> Optional[datetime]:
        """Get the last record time from existing partitioned parquet files."""
        try:
            if data_type == "ohlcv":
                # Look for partitioned OHLCV files
                base_path = self.parquet_folder / "exchange=bybit" / "market=linear_perp" / f"symbol={symbol}" / f"timeframe={interval}"
            elif data_type == "funding":
                # Look for partitioned funding files
                base_path = self.parquet_folder / "exchange=bybit" / "market=linear_perp" / f"symbol={symbol}" / "funding"
            else:
                return None
            
            if not base_path.exists():
                return None
            
            # Find all date directories and get the latest one
            date_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('date=')]
            if not date_dirs:
                return None
            
            # Sort by date and get the latest
            date_dirs.sort(key=lambda x: x.name)
            latest_date_dir = date_dirs[-1]
            
            # Read the parquet file in the latest date directory
            if data_type == "ohlcv":
                file_path = latest_date_dir / "bars.parquet"
            else:  # funding
                file_path = latest_date_dir / "funding.parquet"
            
            if file_path.exists():
                df = pd.read_parquet(file_path)
                if not df.empty:
                    if data_type == "ohlcv":
                        last_time = df['open_time'].max()
                    else:  # funding
                        last_time = df['fundingTime'].max()
                    
                    self.logger.info(f"Found existing partitioned parquet file for {symbol} {interval} {data_type} with last record at {last_time}")
                    return last_time
            
            return None
                
        except Exception as e:
            self.logger.error(f"Error getting last parquet time: {e}")
            return None
    
    def _get_last_fetch_time(self, symbol: str, interval: str, data_type: str) -> Optional[datetime]:
        """Get the last fetch time from database."""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT last_record_time FROM fetch_history 
                    WHERE symbol = ? AND interval = ? AND data_type = ?
                """, (symbol, interval, data_type))
                
                result = cursor.fetchone()
                if result and result[0]:
                    return datetime.fromisoformat(result[0])
                return None
                
        except Exception as e:
            self.logger.error(f"Error getting last fetch time: {e}")
            return None
    
    def _update_fetch_history(self, 
                             symbol: str, 
                             interval: str, 
                             data_type: str,
                             last_record_time: Optional[datetime],
                             records_count: int):
        """Update fetch history in database."""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                # Insert or update fetch history
                cursor.execute("""
                    INSERT OR REPLACE INTO fetch_history 
                    (symbol, interval, data_type, last_fetch_time, last_record_time, records_count)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    symbol, 
                    interval, 
                    data_type, 
                    datetime.now().isoformat(),
                    last_record_time.isoformat() if last_record_time else None,
                    records_count
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error updating fetch history: {e}")
    
    def get_data_info(self, symbol: str, interval: str) -> Dict[str, Any]:
        """Get information about stored data for a symbol and interval."""
        try:
            info = {
                "symbol": symbol,
                "interval": interval,
                "ohlcv_file": None,
                "funding_file": None,
                "ohlcv_records": 0,
                "funding_records": 0,
                "ohlcv_date_range": None,
                "funding_date_range": None
            }
            
            # Check OHLCV file
            ohlcv_file = self.parquet_folder / f"{symbol}_{interval}_ohlcv.parquet"
            if ohlcv_file.exists():
                df = pd.read_parquet(ohlcv_file)
                info["ohlcv_file"] = str(ohlcv_file)
                info["ohlcv_records"] = len(df)
                if len(df) > 0:
                    info["ohlcv_date_range"] = {
                        "start": df['open_time'].min().isoformat(),
                        "end": df['open_time'].max().isoformat()
                    }
            
            # Check funding file
            funding_file = self.parquet_folder / f"{symbol}_funding.parquet"
            if funding_file.exists():
                df = pd.read_parquet(funding_file)
                info["funding_file"] = str(funding_file)
                info["funding_records"] = len(df)
                if len(df) > 0:
                    info["funding_date_range"] = {
                        "start": df['fundingTime'].min().isoformat(),
                        "end": df['fundingTime'].max().isoformat()
                    }
            
            return info
            
        except Exception as e:
            self.logger.error(f"Error getting data info: {e}")
            return {"error": str(e)}
    
    def close(self):
        """Close the data ingestor and cleanup resources."""
        self.logger.info("Closing DataIngestor")
        # Add any cleanup code here if needed

def main():
    """Test the DataIngestor class."""
    # Initialize data ingestor
    ingestor = DataIngestor()
    
    # Test fetching data
    result = ingestor.fetch_and_save_data(
        symbol="BTCUSDT",
        interval="1m",
        ohlcv_limit=1000,
        funding_limit=500,
        incremental=True,
        days_back=7  # Fetch last 7 days
    )
    
    print("Fetch result:", result)
    
    # Get data info
    info = ingestor.get_data_info("BTCUSDT", "1m")
    print("Data info:", info)
    
    # Close ingestor
    ingestor.close()

if __name__ == "__main__":
    main()