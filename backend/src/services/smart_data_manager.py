#!/usr/bin/env python3
"""
Smart Data Manager for Crypto Trading Bot

Intelligently chooses between Data Ingestor and Optimized Data Fetcher
based on data volume, system resources, and use case requirements.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import psutil

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from .data_ingestor import DataIngestor
try:
    from .optimized_data_fetcher import OptimizedDataFetcher, FetchConfig, FetchStrategy
except ImportError:
    # Fallback for when optimized fetcher is not available
    OptimizedDataFetcher = None
    FetchConfig = None
    FetchStrategy = None

class SmartDataManager:
    """
    Intelligent data management system that automatically chooses
    the optimal data fetching strategy based on requirements.
    
    Features:
    - Automatic strategy selection
    - Resource-aware optimization
    - Performance monitoring
    - Seamless fallback mechanisms
    """
    
    def __init__(self, 
                 base_url: str = "https://api.bybit.com",
                 parquet_folder: str = "data/parquet",
                 db_file: str = "data/sqlite/runs.db",
                 log_level: str = "INFO"):
        """
        Initialize the smart data manager.
        
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
        
        # Initialize both systems
        self.data_ingestor = DataIngestor(
            base_url=base_url,
            parquet_folder=str(parquet_folder),
            db_file=str(db_file),
            log_level=log_level,
            enable_validation=True,
            enable_quality_monitoring=True
        )
        
        if OptimizedDataFetcher is not None:
            self.optimized_fetcher = OptimizedDataFetcher(
                base_url=base_url,
                parquet_folder=str(parquet_folder),
                db_file=str(db_file),
                log_level=log_level
            )
        else:
            self.optimized_fetcher = None
        
        # System resource thresholds
        self.thresholds = {
            'small_dataset': 10000,      # < 10K records
            'medium_dataset': 100000,     # 10K - 100K records
            'large_dataset': 1000000,     # 100K - 1M records
            'huge_dataset': 10000000,     # > 1M records
            
            'low_memory': 1024,          # < 1GB available
            'medium_memory': 4096,       # 1-4GB available
            'high_memory': 8192,         # 4-8GB available
            'very_high_memory': 16384,   # > 8GB available
        }
    
    def fetch_data_smart(self, 
                        symbols: List[str], 
                        intervals: List[str],
                        start_date: Optional[datetime] = None,
                        end_date: Optional[datetime] = None,
                        incremental: bool = True,
                        force_strategy: Optional[str] = None) -> Dict[str, Any]:
        """
        Intelligently fetch data using the optimal strategy.
        
        Args:
            symbols: List of symbols to fetch
            intervals: List of intervals to fetch
            start_date: Start date (None for incremental)
            end_date: End date (None for incremental)
            incremental: Whether to use incremental fetching
            force_strategy: Force a specific strategy ('ingestor' or 'optimized')
            
        Returns:
            Fetch results with strategy used
        """
        try:
            # Determine optimal strategy
            if force_strategy:
                strategy = force_strategy
            else:
                strategy = self._determine_optimal_strategy(
                    symbols, intervals, start_date, end_date, incremental
                )
            
            self.logger.info(f"Using strategy: {strategy}")
            
            # Execute fetch with chosen strategy
            if strategy == "ingestor":
                return self._fetch_with_ingestor(symbols, intervals, start_date, end_date, incremental)
            elif strategy == "optimized":
                if self.optimized_fetcher is not None:
                    return self._fetch_with_optimized(symbols, intervals, start_date, end_date)
                else:
                    self.logger.warning("Optimized fetcher not available, falling back to Data Ingestor")
                    return self._fetch_with_ingestor(symbols, intervals, start_date, end_date, incremental)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
                
        except Exception as e:
            self.logger.error(f"Error in smart data fetch: {e}")
            return {
                "success": False,
                "error": str(e),
                "strategy": "failed"
            }
    
    def _determine_optimal_strategy(self, 
                                   symbols: List[str], 
                                   intervals: List[str],
                                   start_date: Optional[datetime],
                                   end_date: Optional[datetime],
                                   incremental: bool) -> str:
        """Determine the optimal fetching strategy."""
        
        # For incremental fetching, prefer data ingestor
        if incremental and start_date is None:
            self.logger.info("Incremental fetch detected - using Data Ingestor")
            return "ingestor"
        
        # Calculate estimated data volume
        estimated_records = self._estimate_data_volume(symbols, intervals, start_date, end_date)
        
        # Get system resources
        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
        cpu_count = psutil.cpu_count()
        
        self.logger.info(f"Estimated records: {estimated_records:,}")
        self.logger.info(f"Available memory: {available_memory:.0f} MB")
        self.logger.info(f"CPU cores: {cpu_count}")
        
        # Decision logic
        if estimated_records < self.thresholds['small_dataset']:
            # Small dataset - use data ingestor for simplicity
            self.logger.info("Small dataset - using Data Ingestor")
            return "ingestor"
        
        elif estimated_records < self.thresholds['medium_dataset']:
            # Medium dataset - use data ingestor unless low memory
            if available_memory < self.thresholds['low_memory']:
                self.logger.info("Medium dataset with low memory - using Optimized Fetcher")
                return "optimized"
            else:
                self.logger.info("Medium dataset - using Data Ingestor")
                return "ingestor"
        
        elif estimated_records < self.thresholds['large_dataset']:
            # Large dataset - use optimized fetcher unless very low memory or not available
            if available_memory < self.thresholds['low_memory'] or self.optimized_fetcher is None:
                self.logger.info("Large dataset with low memory or optimized fetcher not available - using Data Ingestor")
                return "ingestor"
            else:
                self.logger.info("Large dataset - using Optimized Fetcher")
                return "optimized"
        
        else:
            # Huge dataset - use optimized fetcher if available, otherwise use ingestor
            if self.optimized_fetcher is not None:
                self.logger.info("Huge dataset - using Optimized Fetcher")
                return "optimized"
            else:
                self.logger.info("Huge dataset but optimized fetcher not available - using Data Ingestor")
                return "ingestor"
    
    def _estimate_data_volume(self, 
                             symbols: List[str], 
                             intervals: List[str],
                             start_date: Optional[datetime],
                             end_date: Optional[datetime]) -> int:
        """Estimate the number of records that will be fetched."""
        
        if start_date is None or end_date is None:
            # For incremental, estimate based on recent data
            time_range_hours = 24  # Assume 24 hours for incremental
        else:
            time_range_hours = (end_date - start_date).total_seconds() / 3600
        
        # Records per hour for each interval
        records_per_hour = {
            '1m': 60, '5m': 12, '15m': 4, '1h': 1
        }
        
        total_records = 0
        for symbol in symbols:
            for interval in intervals:
                records_per_hour_for_interval = records_per_hour.get(interval, 60)
                total_records += time_range_hours * records_per_hour_for_interval
        
        return int(total_records)
    
    def _fetch_with_ingestor(self, 
                            symbols: List[str], 
                            intervals: List[str],
                            start_date: Optional[datetime],
                            end_date: Optional[datetime],
                            incremental: bool) -> Dict[str, Any]:
        """Fetch data using the standard data ingestor."""
        
        results = {
            "success": True,
            "strategy": "ingestor",
            "symbols": {},
            "total_records": 0,
            "fetch_time_seconds": 0.0
        }
        
        start_time = datetime.now()
        
        for symbol in symbols:
            results["symbols"][symbol] = {}
            
            for interval in intervals:
                try:
                    if incremental:
                        # Use incremental fetching
                        result = self.data_ingestor.fetch_and_save_data(
                            symbol=symbol,
                            interval=interval,
                            ohlcv_limit=1000,
                            funding_limit=500,
                            incremental=True
                        )
                    else:
                        # Use date range fetching
                        result = self.data_ingestor._fetch_ohlcv_by_date_range(
                            symbol, interval, start_date, end_date, 2000
                        )
                    
                    results["symbols"][symbol][interval] = result
                    
                    if result.get("success"):
                        results["total_records"] += result.get("records_count", 0)
                        self.logger.info(f"SUCCESS {symbol} {interval}: {result.get('records_count', 0)} records")
                    else:
                        self.logger.error(f"ERROR {symbol} {interval}: {result.get('error', 'Unknown error')}")
                        results["success"] = False
                
                except Exception as e:
                    error_msg = f"Error fetching {symbol} {interval}: {e}"
                    self.logger.error(error_msg)
                    results["symbols"][symbol][interval] = {
                        "success": False,
                        "error": error_msg
                    }
                    results["success"] = False
        
        results["fetch_time_seconds"] = (datetime.now() - start_time).total_seconds()
        return results
    
    def _fetch_with_optimized(self, 
                             symbols: List[str], 
                             intervals: List[str],
                             start_date: Optional[datetime],
                             end_date: Optional[datetime]) -> Dict[str, Any]:
        """Fetch data using the optimized data fetcher."""
        
        # Create optimized configuration
        config = FetchConfig(
            strategy=FetchStrategy.OPTIMIZED,
            batch_size_hours=24,
            max_concurrent=5,
            memory_limit_mb=2048,
            enable_compression=True,
            enable_validation=True,
            enable_quality_monitoring=True
        )
        
        # Use optimized fetcher
        result = self.optimized_fetcher.fetch_historical_data_optimized(
            symbols=symbols,
            intervals=intervals,
            start_date=start_date,
            end_date=end_date,
            config=config
        )
        
        result["strategy"] = "optimized"
        return result
    
    def get_system_recommendations(self) -> Dict[str, Any]:
        """Get system recommendations for optimal performance."""
        
        # Get system resources
        memory = psutil.virtual_memory()
        cpu_count = psutil.cpu_count()
        
        recommendations = {
            "system_resources": {
                "total_memory_gb": memory.total / (1024**3),
                "available_memory_gb": memory.available / (1024**3),
                "memory_percent": memory.percent,
                "cpu_cores": cpu_count
            },
            "recommendations": []
        }
        
        # Memory recommendations
        if memory.available < 1024:  # < 1GB
            recommendations["recommendations"].append({
                "category": "memory",
                "priority": "high",
                "message": "Low available memory. Consider using Data Ingestor for small datasets.",
                "action": "Use smart_data_manager.fetch_data_smart() with automatic strategy selection"
            })
        elif memory.available > 8192:  # > 8GB
            recommendations["recommendations"].append({
                "category": "performance",
                "priority": "medium",
                "message": "High available memory. Optimized Fetcher will provide best performance.",
                "action": "Use optimized fetcher for large datasets"
            })
        
        # CPU recommendations
        if cpu_count < 4:
            recommendations["recommendations"].append({
                "category": "performance",
                "priority": "medium",
                "message": "Limited CPU cores. Consider reducing max_concurrent for optimized fetcher.",
                "action": "Set max_concurrent=2 in FetchConfig"
            })
        elif cpu_count >= 8:
            recommendations["recommendations"].append({
                "category": "performance",
                "priority": "low",
                "message": "High CPU core count. Can handle high concurrency for optimized fetcher.",
                "action": "Set max_concurrent=8 in FetchConfig"
            })
        
        return recommendations

def main():
    """Main function for standalone smart data management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Smart Data Manager")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"], help="Symbols to fetch")
    parser.add_argument("--intervals", nargs="+", default=["1m", "5m", "15m", "1h"], help="Intervals to fetch")
    parser.add_argument("--start-date", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", help="End date (YYYY-MM-DD)")
    parser.add_argument("--incremental", action="store_true", help="Use incremental fetching")
    parser.add_argument("--force-strategy", choices=["ingestor", "optimized"], help="Force specific strategy")
    parser.add_argument("--recommendations", action="store_true", help="Show system recommendations")
    parser.add_argument("--parquet-folder", default="data/parquet", help="Parquet folder path")
    parser.add_argument("--db-file", default="data/sqlite/runs.db", help="Database file path")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create smart data manager
    manager = SmartDataManager(
        parquet_folder=args.parquet_folder,
        db_file=args.db_file,
        log_level=args.log_level
    )
    
    if args.recommendations:
        # Show system recommendations
        recommendations = manager.get_system_recommendations()
        print("System Recommendations:")
        print(f"Resources: {recommendations['system_resources']}")
        for rec in recommendations['recommendations']:
            print(f"  [{rec['priority'].upper()}] {rec['message']}")
        return
    
    # Parse dates if provided
    start_date = None
    end_date = None
    if args.start_date:
        start_date = datetime.strptime(args.start_date, '%Y-%m-%d')
    if args.end_date:
        end_date = datetime.strptime(args.end_date, '%Y-%m-%d')
    
    # Fetch data
    result = manager.fetch_data_smart(
        symbols=args.symbols,
        intervals=args.intervals,
        start_date=start_date,
        end_date=end_date,
        incremental=args.incremental,
        force_strategy=args.force_strategy
    )
    
    # Print results
    print(f"Fetch completed: {result['success']}")
    print(f"Strategy used: {result.get('strategy', 'unknown')}")
    print(f"Total records: {result.get('total_records', 0):,}")
    print(f"Fetch time: {result.get('fetch_time_seconds', 0):.1f} seconds")

if __name__ == "__main__":
    main()
