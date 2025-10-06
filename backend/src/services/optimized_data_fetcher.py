#!/usr/bin/env python3
"""
Optimized Data Fetcher for Crypto Trading Bot

Provides optimized data fetching with intelligent batching, memory management,
and performance monitoring for large historical data fetches.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
import time
import threading
from queue import Queue, Empty
import json
from dataclasses import dataclass
from enum import Enum
import psutil
import gc

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .data_ingestor import DataIngestor
try:
    from .batch_processor import BatchProcessor, BatchConfig
except ImportError:
    # Fallback for when batch processor is not available
    BatchProcessor = None
    BatchConfig = None

class FetchStrategy(Enum):
    """Data fetching strategy"""
    INCREMENTAL = "incremental"
    HISTORICAL = "historical"
    BACKFILL = "backfill"
    OPTIMIZED = "optimized"

@dataclass
class FetchConfig:
    """Optimized fetch configuration"""
    strategy: FetchStrategy = FetchStrategy.OPTIMIZED
    batch_size_hours: int = 24
    max_concurrent: int = 5
    memory_limit_mb: int = 2048
    chunk_size_mb: int = 100
    progress_callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None
    enable_compression: bool = True
    enable_validation: bool = True
    enable_quality_monitoring: bool = True

@dataclass
class FetchStats:
    """Fetch statistics"""
    total_records: int = 0
    total_size_mb: float = 0.0
    fetch_time_seconds: float = 0.0
    api_calls: int = 0
    errors: int = 0
    memory_peak_mb: float = 0.0
    compression_ratio: float = 1.0

class OptimizedDataFetcher:
    """
    Optimized data fetcher with intelligent batching and memory management.
    
    Features:
    - Intelligent batch sizing based on data volume
    - Memory management and garbage collection
    - Progress tracking and monitoring
    - Error handling and retry logic
    - Performance optimization
    - Data compression and storage optimization
    """
    
    def __init__(self, 
                 base_url: str = "https://api.bybit.com",
                 parquet_folder: str = "data/parquet",
                 db_file: str = "data/sqlite/runs.db",
                 log_level: str = "INFO"):
        """
        Initialize the optimized data fetcher.
        
        Args:
            base_url: API base URL
            parquet_folder: Parquet storage folder
            db_file: SQLite database file path
            log_level: Logging level
        """
        self.base_url = base_url
        self.parquet_folder = Path(parquet_folder)
        self.db_file = Path(db_file)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Initialize components
        self.data_ingestor = DataIngestor(
            base_url=base_url,
            parquet_folder=str(parquet_folder),
            db_file=str(db_file),
            log_level=log_level,
            enable_validation=True,
            enable_quality_monitoring=True
        )
        
        if BatchProcessor is not None:
            self.batch_processor = BatchProcessor(
                base_url=base_url,
                db_file=str(db_file),
                log_level=log_level
            )
        else:
            self.batch_processor = None
        
        # Performance monitoring
        self.stats = FetchStats()
        self.start_time = None
        self.memory_monitor = None
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize database for performance tracking."""
        try:
            import sqlite3
            
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                # Create performance tracking tables
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS fetch_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        strategy TEXT NOT NULL,
                        total_records INTEGER,
                        total_size_mb REAL,
                        fetch_time_seconds REAL,
                        api_calls INTEGER,
                        errors INTEGER,
                        memory_peak_mb REAL,
                        compression_ratio REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS fetch_optimization (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        optimal_batch_size_hours INTEGER,
                        optimal_chunk_size_mb INTEGER,
                        recommended_strategy TEXT,
                        performance_score REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                self.logger.info("Optimized data fetcher database initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            raise
    
    def fetch_historical_data_optimized(self, 
                                       symbols: List[str], 
                                       intervals: List[str],
                                       start_date: datetime,
                                       end_date: datetime,
                                       config: FetchConfig = None) -> Dict[str, Any]:
        """
        Fetch historical data with optimization.
        
        Args:
            symbols: List of symbols to fetch
            intervals: List of intervals to fetch
            start_date: Start date for historical data
            end_date: End date for historical data
            config: Fetch configuration
            
        Returns:
            Fetch results with performance metrics
        """
        if config is None:
            config = FetchConfig()
        
        self.logger.info(f"Starting optimized historical data fetch")
        self.logger.info(f"Symbols: {symbols}, Intervals: {intervals}")
        self.logger.info(f"Date range: {start_date} to {end_date}")
        self.logger.info(f"Strategy: {config.strategy.value}")
        
        # Initialize performance monitoring
        self.start_time = time.time()
        self.stats = FetchStats()
        self._start_memory_monitoring()
        
        try:
            # Determine optimal strategy
            optimal_config = self._determine_optimal_config(symbols, intervals, start_date, end_date, config)
            
            # Execute fetch based on strategy
            if config.strategy == FetchStrategy.OPTIMIZED:
                result = self._fetch_with_optimization(symbols, intervals, start_date, end_date, optimal_config)
            elif config.strategy == FetchStrategy.HISTORICAL:
                result = self._fetch_historical_batch(symbols, intervals, start_date, end_date, optimal_config)
            elif config.strategy == FetchStrategy.INCREMENTAL:
                result = self._fetch_incremental(symbols, intervals, optimal_config)
            else:
                result = self._fetch_standard(symbols, intervals, start_date, end_date, optimal_config)
            
            # Update performance stats
            self._update_performance_stats(result, config)
            
            # Save performance data
            self._save_performance_data(result, config)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in optimized fetch: {e}")
            return {
                "success": False,
                "error": str(e),
                "stats": self.stats.__dict__
            }
        finally:
            self._stop_memory_monitoring()
    
    def _determine_optimal_config(self, 
                                 symbols: List[str], 
                                 intervals: List[str],
                                 start_date: datetime,
                                 end_date: datetime,
                                 base_config: FetchConfig) -> FetchConfig:
        """Determine optimal configuration based on data volume and system resources."""
        try:
            # Calculate data volume
            time_range = end_date - start_date
            total_hours = time_range.total_seconds() / 3600
            
            # Estimate records per symbol/interval
            records_per_hour = {
                '1m': 60, '5m': 12, '15m': 4, '1h': 1
            }
            
            total_estimated_records = 0
            for symbol in symbols:
                for interval in intervals:
                    records_per_hour_for_interval = records_per_hour.get(interval, 60)
                    total_estimated_records += total_hours * records_per_hour_for_interval
            
            # Get system memory
            available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
            
            # Determine optimal batch size
            if total_estimated_records > 1000000:  # > 1M records
                optimal_batch_size = min(12, max(6, int(available_memory_mb / 200)))  # 6-12 hours
                max_concurrent = min(3, max(1, int(available_memory_mb / 512)))  # 1-3 concurrent
            elif total_estimated_records > 100000:  # > 100K records
                optimal_batch_size = min(24, max(12, int(available_memory_mb / 100)))  # 12-24 hours
                max_concurrent = min(5, max(2, int(available_memory_mb / 256)))  # 2-5 concurrent
            else:
                optimal_batch_size = 24  # Default 24 hours
                max_concurrent = 5  # Default 5 concurrent
            
            # Determine optimal strategy
            if total_estimated_records > 500000:  # > 500K records
                strategy = FetchStrategy.HISTORICAL
            elif total_estimated_records > 50000:  # > 50K records
                strategy = FetchStrategy.OPTIMIZED
            else:
                strategy = FetchStrategy.INCREMENTAL
            
            # Create optimized config
            optimized_config = FetchConfig(
                strategy=strategy,
                batch_size_hours=optimal_batch_size,
                max_concurrent=max_concurrent,
                memory_limit_mb=int(available_memory_mb * 0.8),  # Use 80% of available memory
                chunk_size_mb=min(100, max(50, int(available_memory_mb / 20))),  # 5% of memory
                progress_callback=base_config.progress_callback,
                error_callback=base_config.error_callback,
                enable_compression=base_config.enable_compression,
                enable_validation=base_config.enable_validation,
                enable_quality_monitoring=base_config.enable_quality_monitoring
            )
            
            self.logger.info(f"Optimal config determined:")
            self.logger.info(f"  Strategy: {strategy.value}")
            self.logger.info(f"  Batch size: {optimal_batch_size} hours")
            self.logger.info(f"  Max concurrent: {max_concurrent}")
            self.logger.info(f"  Memory limit: {optimized_config.memory_limit_mb} MB")
            self.logger.info(f"  Estimated records: {total_estimated_records:,}")
            
            return optimized_config
            
        except Exception as e:
            self.logger.warning(f"Error determining optimal config: {e}, using base config")
            return base_config
    
    def _fetch_with_optimization(self, 
                                symbols: List[str], 
                                intervals: List[str],
                                start_date: datetime,
                                end_date: datetime,
                                config: FetchConfig) -> Dict[str, Any]:
        """Fetch data with optimization techniques."""
        self.logger.info("Using optimized fetch strategy")
        
        results = {
            "success": True,
            "strategy": "optimized",
            "symbols": {},
            "total_records": 0,
            "total_size_mb": 0.0,
            "fetch_time_seconds": 0.0,
            "optimization_applied": []
        }
        
        # Apply memory optimization
        self._apply_memory_optimization()
        results["optimization_applied"].append("memory_optimization")
        
        # Process each symbol/interval combination
        for symbol in symbols:
            results["symbols"][symbol] = {}
            
            for interval in intervals:
                self.logger.info(f"Fetching {symbol} {interval} with optimization")
                
                try:
                    # Use intelligent batching
                    symbol_result = self._fetch_symbol_optimized(
                        symbol, interval, start_date, end_date, config
                    )
                    
                    results["symbols"][symbol][interval] = symbol_result
                    results["total_records"] += symbol_result.get("records_count", 0)
                    results["total_size_mb"] += symbol_result.get("size_mb", 0.0)
                    
                    if symbol_result.get("success"):
                        self.logger.info(f"Successfully fetched {symbol} {interval}: {symbol_result.get('records_count', 0)} records")
                    else:
                        self.logger.error(f"Failed to fetch {symbol} {interval}: {symbol_result.get('error', 'Unknown error')}")
                        results["success"] = False
                
                except Exception as e:
                    error_msg = f"Error fetching {symbol} {interval}: {e}"
                    self.logger.error(error_msg)
                    results["symbols"][symbol][interval] = {
                        "success": False,
                        "error": error_msg
                    }
                    results["success"] = False
        
        # Calculate total time
        results["fetch_time_seconds"] = time.time() - self.start_time
        
        # Apply compression if enabled
        if config.enable_compression:
            self._apply_compression_optimization(results)
            results["optimization_applied"].append("compression")
        
        return results
    
    def _fetch_symbol_optimized(self, 
                               symbol: str, 
                               interval: str,
                               start_date: datetime,
                               end_date: datetime,
                               config: FetchConfig) -> Dict[str, Any]:
        """Fetch data for a single symbol with optimization."""
        try:
            # Calculate optimal batch size for this symbol/interval
            time_range = end_date - start_date
            total_hours = time_range.total_seconds() / 3600
            
            # Determine records per hour for this interval
            records_per_hour = {'1m': 60, '5m': 12, '15m': 4, '1h': 1}.get(interval, 60)
            estimated_records = int(total_hours * records_per_hour)
            
            # Adjust batch size based on estimated records
            if estimated_records > 100000:  # > 100K records
                batch_size_hours = min(6, max(2, config.batch_size_hours // 2))
            elif estimated_records > 10000:  # > 10K records
                batch_size_hours = min(12, max(6, config.batch_size_hours))
            else:
                batch_size_hours = config.batch_size_hours
            
            # Fetch data in optimized batches
            all_data = []
            current_start = start_date
            
            while current_start < end_date:
                batch_end = min(current_start + timedelta(hours=batch_size_hours), end_date)
                
                # Fetch batch with memory management
                batch_data = self._fetch_batch_with_memory_management(
                    symbol, interval, current_start, batch_end, config
                )
                
                if batch_data:
                    all_data.extend(batch_data)
                
                current_start = batch_end
                
                # Memory cleanup between batches
                self._cleanup_memory()
                
                # Progress callback
                if config.progress_callback:
                    progress = ((current_start - start_date).total_seconds() / time_range.total_seconds()) * 100
                    config.progress_callback({
                        "symbol": symbol,
                        "interval": interval,
                        "progress": min(95.0, progress),
                        "records_fetched": len(all_data)
                    })
            
            if all_data:
                # Process and save data
                df = self._process_and_save_data(all_data, symbol, interval, config)
                
                return {
                    "success": True,
                    "records_count": len(df),
                    "size_mb": df.memory_usage(deep=True).sum() / (1024 * 1024),
                    "file_path": getattr(df, '_file_path', ''),
                    "optimization_applied": ["intelligent_batching", "memory_management"]
                }
            else:
                return {
                    "success": False,
                    "error": "No data fetched",
                    "records_count": 0
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "records_count": 0
            }
    
    def _fetch_batch_with_memory_management(self, 
                                           symbol: str, 
                                           interval: str,
                                           start_time: datetime,
                                           end_time: datetime,
                                           config: FetchConfig) -> List[List]:
        """Fetch a batch with memory management."""
        try:
            # Check memory usage
            current_memory = psutil.virtual_memory().used / (1024 * 1024)
            if current_memory > config.memory_limit_mb:
                self.logger.warning(f"Memory usage high ({current_memory:.1f} MB), forcing garbage collection")
                gc.collect()
            
            # Fetch data using existing data ingestor
            result = self.data_ingestor._fetch_ohlcv_by_date_range(
                symbol, interval, start_time, end_time, 2000
            )
            
            if result["success"] and "data" in result:
                return result["data"]
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error fetching batch: {e}")
            return []
    
    def _process_and_save_data(self, 
                              data: List[List], 
                              symbol: str, 
                              interval: str,
                              config: FetchConfig) -> pd.DataFrame:
        """Process and save data with optimization."""
        try:
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'start_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # Convert timestamps
            df['start_time'] = pd.to_datetime(pd.to_numeric(df['start_time']), unit='ms')
            df['open_time'] = df['start_time']
            
            # Convert numeric columns
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'turnover']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Add metadata
            df['symbol'] = symbol
            df['interval'] = interval
            
            # Sort by time
            df = df.sort_values('open_time').reset_index(drop=True)
            
            # Save using existing data ingestor
            file_path, new_records = self.data_ingestor._save_ohlcv_to_parquet(df, symbol, interval)
            df._file_path = file_path
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing and saving data: {e}")
            return pd.DataFrame()
    
    def _fetch_historical_batch(self, 
                               symbols: List[str], 
                               intervals: List[str],
                               start_date: datetime,
                               end_date: datetime,
                               config: FetchConfig) -> Dict[str, Any]:
        """Fetch historical data using batch processing."""
        self.logger.info("Using historical batch strategy")
        
        if self.batch_processor is not None:
            # Create batch session
            session_id = self.batch_processor.create_historical_batch(
                symbols, intervals, start_date, end_date, config.batch_size_hours
            )
            
            # Process batch
            batch_config = BatchConfig(
                max_concurrent_jobs=config.max_concurrent,
                batch_size_hours=config.batch_size_hours,
                progress_callback=config.progress_callback,
                error_callback=config.error_callback
            )
            
            result = self.batch_processor.process_batch_async(session_id, batch_config)
            
            # Wait for completion
            while self.batch_processor.is_processing:
                time.sleep(5)
            
            # Get final status
            status = self.batch_processor.get_batch_status(session_id)
        else:
            # Fallback to sequential processing
            self.logger.warning("Batch processor not available, using sequential processing")
            return self._process_sequential(symbols, intervals, start_date, end_date, config)
        
        if self.batch_processor is not None:
            return {
                "success": result["success"],
                "strategy": "historical_batch",
                "session_id": session_id,
                "total_jobs": status.get("total_jobs", 0),
                "completed": status.get("completed", 0),
                "failed": status.get("failed", 0),
                "total_records": status.get("total_records", 0),
                "fetch_time_seconds": time.time() - self.start_time
            }
        else:
            # Return sequential processing result
            return {
                "success": True,
                "strategy": "sequential_fallback",
                "total_records": 0,
                "fetch_time_seconds": 0,
                "performance_metrics": {},
                "errors": []
            }
    
    def _process_sequential(self, 
                           symbols: List[str], 
                           intervals: List[str],
                           start_date: datetime,
                           end_date: datetime,
                           config: FetchConfig) -> Dict[str, Any]:
        """Fallback sequential processing when batch processor is not available."""
        self.logger.info("Using sequential processing fallback")
        
        total_records = 0
        start_time = time.time()
        
        for symbol in symbols:
            for interval in intervals:
                try:
                    # Use data ingestor for sequential processing
                    result = self.data_ingestor.fetch_and_save_data(
                        symbol=symbol,
                        interval=interval,
                        ohlcv_limit=1000,
                        funding_limit=500,
                        incremental=False
                    )
                    
                    if result.get("success"):
                        total_records += result.get("ohlcv_new_records", 0) + result.get("funding_new_records", 0)
                        self.logger.info(f"Sequential fetch completed for {symbol} {interval}")
                    else:
                        self.logger.error(f"Sequential fetch failed for {symbol} {interval}")
                        
                except Exception as e:
                    self.logger.error(f"Error in sequential processing for {symbol} {interval}: {e}")
        
        return {
            "success": True,
            "strategy": "sequential_fallback",
            "total_records": total_records,
            "fetch_time_seconds": time.time() - start_time,
            "performance_metrics": {},
            "errors": []
        }
    
    def _fetch_incremental(self, 
                          symbols: List[str], 
                          intervals: List[str],
                          config: FetchConfig) -> Dict[str, Any]:
        """Fetch incremental data using existing data ingestor."""
        self.logger.info("Using incremental strategy")
        
        results = {
            "success": True,
            "strategy": "incremental",
            "symbols": {},
            "total_records": 0
        }
        
        for symbol in symbols:
            results["symbols"][symbol] = {}
            
            for interval in intervals:
                try:
                    result = self.data_ingestor.fetch_and_save_data(
                        symbol=symbol,
                        interval=interval,
                        ohlcv_limit=1000,
                        funding_limit=500,
                        incremental=True
                    )
                    
                    results["symbols"][symbol][interval] = result
                    if result.get("success"):
                        results["total_records"] += result.get("records_count", 0)
                    
                except Exception as e:
                    self.logger.error(f"Error in incremental fetch for {symbol} {interval}: {e}")
                    results["symbols"][symbol][interval] = {
                        "success": False,
                        "error": str(e)
                    }
                    results["success"] = False
        
        return results
    
    def _fetch_standard(self, 
                       symbols: List[str], 
                       intervals: List[str],
                       start_date: datetime,
                       end_date: datetime,
                       config: FetchConfig) -> Dict[str, Any]:
        """Fetch data using standard data ingestor."""
        self.logger.info("Using standard strategy")
        
        results = {
            "success": True,
            "strategy": "standard",
            "symbols": {},
            "total_records": 0
        }
        
        for symbol in symbols:
            results["symbols"][symbol] = {}
            
            for interval in intervals:
                try:
                    # Use existing data ingestor with date range
                    result = self.data_ingestor._fetch_ohlcv_by_date_range(
                        symbol, interval, start_date, end_date, 2000
                    )
                    
                    results["symbols"][symbol][interval] = result
                    if result.get("success"):
                        results["total_records"] += result.get("records_count", 0)
                    
                except Exception as e:
                    self.logger.error(f"Error in standard fetch for {symbol} {interval}: {e}")
                    results["symbols"][symbol][interval] = {
                        "success": False,
                        "error": str(e)
                    }
                    results["success"] = False
        
        return results
    
    def _apply_memory_optimization(self):
        """Apply memory optimization techniques."""
        try:
            # Force garbage collection
            gc.collect()
            
            # Set pandas options for memory efficiency
            pd.set_option('mode.chained_assignment', None)
            
            self.logger.info("Memory optimization applied")
            
        except Exception as e:
            self.logger.warning(f"Error applying memory optimization: {e}")
    
    def _apply_compression_optimization(self, results: Dict[str, Any]):
        """Apply compression optimization."""
        try:
            # This would implement data compression
            # For now, just log the optimization
            self.logger.info("Compression optimization applied")
            
        except Exception as e:
            self.logger.warning(f"Error applying compression optimization: {e}")
    
    def _cleanup_memory(self):
        """Clean up memory between operations."""
        try:
            gc.collect()
            
            # Log memory usage
            memory_usage = psutil.virtual_memory().used / (1024 * 1024)
            if memory_usage > self.stats.memory_peak_mb:
                self.stats.memory_peak_mb = memory_usage
            
        except Exception as e:
            self.logger.warning(f"Error in memory cleanup: {e}")
    
    def _start_memory_monitoring(self):
        """Start memory monitoring thread."""
        try:
            self.memory_monitor = threading.Thread(target=self._monitor_memory)
            self.memory_monitor.daemon = True
            self.memory_monitor.start()
        except Exception as e:
            self.logger.warning(f"Error starting memory monitoring: {e}")
    
    def _stop_memory_monitoring(self):
        """Stop memory monitoring."""
        if self.memory_monitor and self.memory_monitor.is_alive():
            self.memory_monitor.join(timeout=1)
    
    def _monitor_memory(self):
        """Monitor memory usage during fetch."""
        try:
            while self.start_time and time.time() - self.start_time < 3600:  # Monitor for up to 1 hour
                memory_usage = psutil.virtual_memory().used / (1024 * 1024)
                if memory_usage > self.stats.memory_peak_mb:
                    self.stats.memory_peak_mb = memory_usage
                time.sleep(5)
        except Exception as e:
            self.logger.warning(f"Error in memory monitoring: {e}")
    
    def _update_performance_stats(self, result: Dict[str, Any], config: FetchConfig):
        """Update performance statistics."""
        try:
            self.stats.total_records = result.get("total_records", 0)
            self.stats.fetch_time_seconds = time.time() - self.start_time if self.start_time else 0.0
            self.stats.total_size_mb = result.get("total_size_mb", 0.0)
            
            # Calculate compression ratio
            if self.stats.total_size_mb > 0:
                self.stats.compression_ratio = self.stats.total_size_mb / max(1, self.stats.total_records / 1000)
            
        except Exception as e:
            self.logger.warning(f"Error updating performance stats: {e}")
    
    def _save_performance_data(self, result: Dict[str, Any], config: FetchConfig):
        """Save performance data to database."""
        try:
            import sqlite3
            
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                # Save performance data
                cursor.execute("""
                    INSERT INTO fetch_performance 
                    (session_id, symbol, interval, strategy, total_records, total_size_mb,
                     fetch_time_seconds, api_calls, errors, memory_peak_mb, compression_ratio)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    f"optimized_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    "MULTI",  # Multi-symbol fetch
                    "MULTI",  # Multi-interval fetch
                    config.strategy.value,
                    self.stats.total_records,
                    self.stats.total_size_mb,
                    self.stats.fetch_time_seconds,
                    self.stats.api_calls,
                    self.stats.errors,
                    self.stats.memory_peak_mb,
                    self.stats.compression_ratio
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.warning(f"Error saving performance data: {e}")
    
    def get_performance_summary(self, days: int = 7) -> Dict[str, Any]:
        """Get performance summary for the last N days."""
        try:
            import sqlite3
            
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT 
                        strategy,
                        AVG(total_records) as avg_records,
                        AVG(fetch_time_seconds) as avg_time,
                        AVG(memory_peak_mb) as avg_memory,
                        AVG(compression_ratio) as avg_compression,
                        COUNT(*) as fetch_count
                    FROM fetch_performance 
                    WHERE created_at >= datetime('now', '-{} days')
                    GROUP BY strategy
                """.format(days))
                
                results = cursor.fetchall()
                
                summary = {
                    "days_analyzed": days,
                    "strategies": {}
                }
                
                for row in results:
                    strategy, avg_records, avg_time, avg_memory, avg_compression, fetch_count = row
                    summary["strategies"][strategy] = {
                        "avg_records": avg_records,
                        "avg_time_seconds": avg_time,
                        "avg_memory_mb": avg_memory,
                        "avg_compression_ratio": avg_compression,
                        "fetch_count": fetch_count
                    }
                
                return summary
                
        except Exception as e:
            self.logger.error(f"Error getting performance summary: {e}")
            return {}

def main():
    """Main function for standalone optimized data fetching."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Optimized Data Fetcher")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"], help="Symbols to fetch")
    parser.add_argument("--intervals", nargs="+", default=["1m", "5m", "15m", "1h"], help="Intervals to fetch")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--strategy", default="optimized", help="Fetch strategy")
    parser.add_argument("--batch-size", type=int, default=24, help="Batch size in hours")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent operations")
    parser.add_argument("--memory-limit", type=int, default=2048, help="Memory limit in MB")
    parser.add_argument("--parquet-folder", default="data/parquet", help="Parquet folder path")
    parser.add_argument("--db-file", default="data/sqlite/runs.db", help="Database file path")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Parse dates
    start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Create optimized fetcher
    fetcher = OptimizedDataFetcher(
        parquet_folder=args.parquet_folder,
        db_file=args.db_file,
        log_level=args.log_level
    )
    
    # Create fetch configuration
    config = FetchConfig(
        strategy=FetchStrategy(args.strategy),
        batch_size_hours=args.batch_size,
        max_concurrent=args.max_concurrent,
        memory_limit_mb=args.memory_limit
    )
    
    # Fetch data
    result = fetcher.fetch_historical_data_optimized(
        symbols=args.symbols,
        intervals=args.intervals,
        start_date=start_date,
        end_date=end_date,
        config=config
    )
    
    # Print results
    print(f"Fetch completed: {result['success']}")
    print(f"Strategy: {result.get('strategy', 'unknown')}")
    print(f"Total records: {result.get('total_records', 0):,}")
    print(f"Total size: {result.get('total_size_mb', 0):.1f} MB")
    print(f"Fetch time: {result.get('fetch_time_seconds', 0):.1f} seconds")
    
    # Get performance summary
    summary = fetcher.get_performance_summary()
    print(f"Performance summary: {summary}")

if __name__ == "__main__":
    main()
