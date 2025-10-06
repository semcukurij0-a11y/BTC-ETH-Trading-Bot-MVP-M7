#!/usr/bin/env python3
"""
Crypto Trading Bot - Main Workflow Orchestrator

This script orchestrates the complete workflow:
Exchange (WS/REST) -> data_ingestor.py -> feature_engineering.py

Workflow:
1. Data Ingestion: Fetch OHLCV and funding data from Binance Futures API
2. Feature Engineering: Create ML-ready features from raw data
3. Output: Generate features.parquet files for machine learning
"""

import os
import sys
import logging
import argparse
import time
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import json

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.data_ingestor import DataIngestor
from services.smart_data_manager import SmartDataManager
from services.feature_engineering import FeatureEngineer
from services.health_service import HealthService
from services.scheduler_service import SchedulerService

class TradingBotWorkflow:
    """
    Main workflow orchestrator for the crypto trading bot.
    Coordinates data ingestion and feature engineering processes.
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the trading bot workflow.
        
        Args:
            config_file: Path to configuration file (optional)
        """
        self.config = self._load_config(config_file)
        self.logger = self._setup_logging()
        
        # Initialize services
        self.data_ingestor = DataIngestor(
            base_url=self.config.get("bybit_base_url", "https://api.bybit.com"),
            ws_url=self.config.get("bybit_ws_url", "wss://stream.bybit.com"),
            parquet_folder=self.config.get("parquet_folder", "data/parquet"),
            db_file=self.config.get("db_file", "data/sqlite/runs.db"),
            log_level=self.config.get("log_level", "INFO"),
            revalidation_bars=self.config.get("revalidation_bars", 5)  # Default 5 bars for revalidation
        )
        
        # Initialize Smart Data Manager for optimized data fetching
        self.smart_data_manager = SmartDataManager(
            base_url=self.config.get("bybit_base_url", "https://api.bybit.com"),
            parquet_folder=self.config.get("parquet_folder", "data/parquet"),
            db_file=self.config.get("db_file", "data/sqlite/runs.db"),
            log_level=self.config.get("log_level", "INFO")
        )
        
        self.feature_engineer = FeatureEngineer(
            data_folder=self.config.get("parquet_folder", "data/parquet"),
            output_folder=self.config.get("features_folder", "data/features"),
            log_level=self.config.get("log_level", "INFO"),
            enable_extra_features=self.config.get("enable_extra_features", False)  # MVP features by default
        )
        
        self.logger.info("Trading Bot Workflow initialized successfully")
    
    def _load_config(self, config_file: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults"""
        default_config = {
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "intervals": ["1m", "5m", "15m"],
            "ohlcv_limit": 1000,
            "funding_limit": 500,
            "days_back": 60,
            "fetch_by_days": True,  # Fetch by date range instead of count
            "target_periods": [30, 60, 90, 120],
            "run_mode": "single",  # single, continuous, schedule
            "schedule_interval": 60,  # seconds for continuous mode
            "incremental_fetch": True,  # Use incremental fetching for efficiency
            "parquet_folder": "data/parquet",
            "features_folder": "data/features",
            "db_file": "data/sqlite/runs.db",
            "log_level": "INFO",
            "bybit_base_url": "https://api.bybit.com",
            "bybit_ws_url": "wss://stream.bybit.com",
            "enable_extra_features": False,  # MVP features by default, 50+ extra indicators behind flag
            "revalidation_bars": 5  # Number of trailing bars to revalidate for late edits
        }
        
        if config_file and os.path.exists(config_file):
            try:
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
                print(f"Loaded configuration from {config_file}")
            except Exception as e:
                print(f"Error loading config file {config_file}: {e}")
                print("Using default configuration")
        
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.get("log_level", "INFO").upper())
        
        # Create logs directory
        os.makedirs("logs", exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f"logs/trading_bot_{datetime.now().strftime('%Y%m%d')}.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        return logging.getLogger(__name__)
    
    def run_data_ingestion(self, symbols: List[str], intervals: List[str]) -> Dict[str, Any]:
        """
        Run data ingestion for specified symbols and intervals using Smart Data Manager.
        
        Args:
            symbols: List of trading symbols
            intervals: List of time intervals
            
        Returns:
            Dictionary with ingestion results
        """
        self.logger.info(f"Starting data ingestion for symbols: {symbols}, intervals: {intervals}")
        
        # Determine if we should use incremental or historical fetching
        incremental = self.config.get("incremental_fetch", True)
        start_date = None
        end_date = None
        
        if not incremental and self.config.get("fetch_by_days", False):
            days_back = self.config.get("days_back", 30)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            self.logger.info(f"Historical fetch: {days_back} days back from {end_date}")
        elif not incremental:
            # Use a default historical range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=30)
            self.logger.info(f"Historical fetch: 30 days back from {end_date}")
        else:
            self.logger.info("Incremental fetch: fetching latest data")
        
        # Use Smart Data Manager for intelligent strategy selection
        try:
            result = self.smart_data_manager.fetch_data_smart(
                symbols=symbols,
                intervals=intervals,
                start_date=start_date,
                end_date=end_date,
                incremental=incremental
            )
            
            # Log strategy used
            strategy = result.get("strategy", "unknown")
            self.logger.info(f"Smart Data Manager used strategy: {strategy}")
            
            # Convert result format to match expected format
            results = {
                "success": result.get("success", False),
                "strategy_used": strategy,
                "total_records": result.get("total_records", 0),
                "fetch_time_seconds": result.get("fetch_time_seconds", 0),
                "ingested_data": result.get("symbols", {}),
                "errors": []
            }
            
            # Extract errors from result
            if not result.get("success", False):
                results["errors"].append(result.get("error", "Unknown error"))
            
            # Log performance metrics
            if result.get("success", False):
                self.logger.info(f"Data ingestion completed successfully")
                self.logger.info(f"Strategy: {strategy}")
                self.logger.info(f"Total records: {result.get('total_records', 0):,}")
                self.logger.info(f"Fetch time: {result.get('fetch_time_seconds', 0):.1f} seconds")
                
                # Calculate records per second
                if result.get('fetch_time_seconds', 0) > 0:
                    records_per_second = result.get('total_records', 0) / result.get('fetch_time_seconds', 1)
                    self.logger.info(f"Performance: {records_per_second:.0f} records/second")
            else:
                self.logger.error(f"Data ingestion failed: {result.get('error', 'Unknown error')}")
                results["success"] = False
            
            return results
            
        except Exception as e:
            error_msg = f"Error in Smart Data Manager: {e}"
            self.logger.error(error_msg)
            return {
                "success": False,
                "strategy_used": "failed",
                "total_records": 0,
                "fetch_time_seconds": 0,
                "ingested_data": {},
                "errors": [error_msg]
            }
    
    def run_feature_engineering(self, symbols: List[str], intervals: List[str]) -> Dict[str, Any]:
        """
        Run feature engineering for specified symbols and intervals.
        
        Args:
            symbols: List of trading symbols
            intervals: List of time intervals
            
        Returns:
            Dictionary with feature engineering results
        """
        self.logger.info(f"Starting feature engineering for symbols: {symbols}, intervals: {intervals}")
        
        results = {
            "success": True,
            "feature_files": {},
            "errors": []
        }
        
        for symbol in symbols:
            results["feature_files"][symbol] = {}
            
            for interval in intervals:
                try:
                    self.logger.info(f"Creating features for {symbol} {interval}")
                    
                    # Map interval to timeframe for feature engineering
                    timeframe_map = {
                        "1m": "1m",
                        "5m": "5m", 
                        "15m": "15m",
                        "1h": "1h",
                        "4h": "4h",
                        "1d": "1d"
                    }
                    timeframe = timeframe_map.get(interval, interval)
                    
                    filepath = self.feature_engineer.process_symbol(
                        symbol=symbol,
                        timeframe=timeframe,
                        days_back=self.config["days_back"],
                        target_periods=self.config["target_periods"]
                    )
                    
                    if filepath:
                        results["feature_files"][symbol][interval] = filepath
                        self.logger.info(f"Successfully created features for {symbol} {interval}: {filepath}")
                    else:
                        error_msg = f"Failed to create features for {symbol} {interval}"
                        self.logger.error(error_msg)
                        results["errors"].append(error_msg)
                        results["success"] = False
                        
                except Exception as e:
                    error_msg = f"Error creating features for {symbol} {interval}: {e}"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)
                    results["success"] = False
        
        return results
    
    def run_single_workflow(self) -> Dict[str, Any]:
        """
        Run a single complete workflow cycle with Smart Data Manager.
        
        Returns:
            Dictionary with workflow results
        """
        self.logger.info("Starting single workflow cycle with Smart Data Manager")
        
        # Get system recommendations
        self.get_system_recommendations()
        
        symbols = self.config["symbols"]
        intervals = self.config["intervals"]
        
        # Step 1: Data Ingestion with Smart Data Manager
        ingestion_results = self.run_data_ingestion(symbols, intervals)
        
        if not ingestion_results["success"]:
            self.logger.error("Data ingestion failed, skipping feature engineering")
            return {
                "success": False,
                "stage": "data_ingestion",
                "errors": ingestion_results["errors"]
            }
        
        # Step 2: Feature Engineering
        feature_results = self.run_feature_engineering(symbols, intervals)
        
        if not feature_results["success"]:
            self.logger.error("Feature engineering failed")
            return {
                "success": False,
                "stage": "feature_engineering",
                "errors": feature_results["errors"]
            }
        
        # Success
        self.logger.info("Workflow completed successfully")
        return {
            "success": True,
            "stage": "completed",
            "ingestion_results": ingestion_results,
            "feature_results": feature_results
        }
    
    def run_continuous_workflow(self):
        """
        Run continuous workflow with scheduled intervals using the scheduler service.
        """
        self.logger.info(f"Starting continuous workflow with {self.config['schedule_interval']}s intervals")
        
        # Create scheduler service
        scheduler_service = SchedulerService(
            config_file=None,  # Use current config
            log_level=self.config.get("log_level", "INFO")
        )
        
        # Add interval job
        scheduler_service.add_interval_job(
            interval_seconds=self.config["schedule_interval"],
            job_id="trading_workflow"
        )
        
        try:
            # Run the scheduler
            asyncio.run(scheduler_service.run_forever())
        except KeyboardInterrupt:
            self.logger.info("Continuous workflow stopped by user")
        except Exception as e:
            self.logger.error(f"Continuous workflow error: {e}")
        finally:
            scheduler_service.stop()
            self.cleanup()
    
    def run_with_health_service(self, 
                               host: str = "0.0.0.0", 
                               port: int = 8080,
                               mode: str = "single"):
        """
        Run the trading bot with health service.
        
        Args:
            host: Host to bind health service to
            port: Port to bind health service to
            mode: Run mode (single, continuous, or scheduled)
        """
        self.logger.info(f"Starting trading bot with health service on {host}:{port}")
        
        # Create health service
        health_service = HealthService(
            data_folder=self.config.get("parquet_folder", "data/parquet"),
            db_file=self.config.get("db_file", "data/sqlite/runs.db"),
            log_level=self.config.get("log_level", "INFO")
        )
        
        # Create scheduler service
        scheduler_service = SchedulerService(
            config_file=None,
            log_level=self.config.get("log_level", "INFO")
        )
        
        # Add job based on mode
        if mode == "continuous":
            scheduler_service.add_interval_job(
                interval_seconds=self.config["schedule_interval"],
                job_id="trading_workflow"
            )
        elif mode == "scheduled":
            # Add cron job (9:00 AM every day)
            scheduler_service.add_cron_job(
                cron_expression="0 9 * * 0-6",  # 9:00 AM every day (Sunday-Saturday)
                job_id="trading_workflow"
            )
        
        async def run_services():
            """Run both health service and scheduler concurrently."""
            tasks = []
            
            # Start health service
            health_task = asyncio.create_task(
                asyncio.to_thread(health_service.run_server, host, port)
            )
            tasks.append(health_task)
            
            # Start scheduler if not single mode
            if mode != "single":
                scheduler_service.start()
                scheduler_task = asyncio.create_task(scheduler_service.run_forever())
                tasks.append(scheduler_task)
            
            # Run single workflow if in single mode
            if mode == "single":
                workflow_task = asyncio.create_task(
                    asyncio.to_thread(self.run_single_workflow)
                )
                tasks.append(workflow_task)
            
            try:
                await asyncio.gather(*tasks)
            except KeyboardInterrupt:
                self.logger.info("Services stopped by user")
            finally:
                if mode != "single":
                    scheduler_service.stop()
                self.cleanup()
        
        try:
            asyncio.run(run_services())
        except KeyboardInterrupt:
            self.logger.info("Trading bot with health service stopped by user")
        except Exception as e:
            self.logger.error(f"Error running trading bot with health service: {e}")
        finally:
            self.cleanup()
    
    def get_system_recommendations(self) -> Dict[str, Any]:
        """
        Get system recommendations for optimal performance.
        
        Returns:
            Dictionary with system recommendations
        """
        try:
            recommendations = self.smart_data_manager.get_system_recommendations()
            
            self.logger.info("System Recommendations:")
            self.logger.info(f"   Memory: {recommendations['system_resources']['available_memory_gb']:.1f} GB available")
            self.logger.info(f"   CPU Cores: {recommendations['system_resources']['cpu_cores']}")
            self.logger.info(f"   Memory Usage: {recommendations['system_resources']['memory_percent']:.1f}%")
            
            for rec in recommendations['recommendations']:
                self.logger.info(f"   [{rec['priority'].upper()}] {rec['message']}")
            
            return recommendations
            
        except Exception as e:
            self.logger.error(f"Error getting system recommendations: {e}")
            return {"error": str(e)}
    
    def cleanup(self):
        """Clean up resources"""
        self.logger.info("Cleaning up resources")
        if hasattr(self, 'data_ingestor'):
            self.data_ingestor.close()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Crypto Trading Bot Workflow")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--mode", "-m", choices=["single", "continuous", "scheduled", "health"], 
                       default="single", help="Run mode")
    parser.add_argument("--symbols", nargs="+", help="Trading symbols (overrides config)")
    parser.add_argument("--intervals", nargs="+", help="Time intervals (overrides config)")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                       help="Log level (overrides config)")
    parser.add_argument("--host", default="0.0.0.0", help="Host for health service")
    parser.add_argument("--port", type=int, default=8080, help="Port for health service")
    parser.add_argument("--with-health", action="store_true", 
                       help="Run with health service (for single/continuous modes)")
    
    args = parser.parse_args()
    
    try:
        # Initialize workflow
        workflow = TradingBotWorkflow(config_file=args.config)
        
        # Override config with command line arguments
        if args.symbols:
            workflow.config["symbols"] = args.symbols
        if args.intervals:
            workflow.config["intervals"] = args.intervals
        if args.log_level:
            workflow.config["log_level"] = args.log_level
        
        # Run workflow based on mode
        if args.mode == "single":
            if args.with_health:
                workflow.run_with_health_service(
                    host=args.host, 
                    port=args.port, 
                    mode="single"
                )
            else:
                result = workflow.run_single_workflow()
                
                if result["success"]:
                    print("Workflow completed successfully!")
                    print(f"Processed symbols: {workflow.config['symbols']}")
                    print(f"Processed intervals: {workflow.config['intervals']}")
                else:
                    print(f"Workflow failed at stage: {result['stage']}")
                    if "errors" in result:
                        for error in result["errors"]:
                            print(f"   Error: {error}")
                    sys.exit(1)
        
        elif args.mode == "continuous":
            if args.with_health:
                workflow.run_with_health_service(
                    host=args.host, 
                    port=args.port, 
                    mode="continuous"
                )
            else:
                workflow.run_continuous_workflow()
        
        elif args.mode == "scheduled":
            workflow.run_with_health_service(
                host=args.host, 
                port=args.port, 
                mode="scheduled"
            )
        
        elif args.mode == "health":
            # Run only health service
            health_service = HealthService(
                data_folder=workflow.config.get("parquet_folder", "data/parquet"),
                db_file=workflow.config.get("db_file", "data/sqlite/runs.db"),
                log_level=workflow.config.get("log_level", "INFO")
            )
            health_service.run_server(host=args.host, port=args.port)
    
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
    
    finally:
        if 'workflow' in locals():
            workflow.cleanup()

if __name__ == "__main__":
    main()
