#!/usr/bin/env python3
"""
Batch Processor for Crypto Trading Bot

Provides optimized batch processing for large historical data fetches
with parallel processing, memory management, and progress tracking.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Callable
from pathlib import Path
import asyncio
import aiohttp
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
import threading
from queue import Queue
import json

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class BatchStatus(Enum):
    """Batch processing status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class BatchJob:
    """Individual batch job definition"""
    job_id: str
    symbol: str
    interval: str
    start_time: datetime
    end_time: datetime
    status: BatchStatus = BatchStatus.PENDING
    progress: float = 0.0
    records_fetched: int = 0
    error: Optional[str] = None
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()

@dataclass
class BatchConfig:
    """Batch processing configuration"""
    max_concurrent_jobs: int = 5
    batch_size_hours: int = 24
    max_retries: int = 3
    retry_delay: float = 1.0
    memory_limit_mb: int = 1024
    progress_callback: Optional[Callable] = None
    error_callback: Optional[Callable] = None

class BatchProcessor:
    """
    Optimized batch processor for large historical data fetches.
    
    Features:
    - Parallel processing with configurable concurrency
    - Memory management for large datasets
    - Progress tracking and reporting
    - Error handling and retry logic
    - Rate limiting and API optimization
    - Real-time monitoring
    """
    
    def __init__(self, 
                 base_url: str = "https://api.bybit.com",
                 db_file: str = "data/sqlite/runs.db",
                 log_level: str = "INFO"):
        """
        Initialize the batch processor.
        
        Args:
            base_url: API base URL
            db_file: SQLite database file path
            log_level: Logging level
        """
        self.base_url = base_url
        self.db_file = Path(db_file)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Batch processing state
        self.jobs: Dict[str, BatchJob] = {}
        self.active_jobs: Dict[str, BatchJob] = {}
        self.completed_jobs: Dict[str, BatchJob] = {}
        self.failed_jobs: Dict[str, BatchJob] = {}
        
        # Processing control
        self.is_processing = False
        self.stop_processing = False
        self.processing_thread: Optional[threading.Thread] = None
        
        # Statistics
        self.total_jobs = 0
        self.completed_jobs_count = 0
        self.failed_jobs_count = 0
        self.total_records_fetched = 0
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for batch processing tracking."""
        try:
            import sqlite3
            
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                # Create batch processing tables
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS batch_jobs (
                        id TEXT PRIMARY KEY,
                        symbol TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        start_time TIMESTAMP NOT NULL,
                        end_time TIMESTAMP NOT NULL,
                        status TEXT NOT NULL,
                        progress REAL DEFAULT 0.0,
                        records_fetched INTEGER DEFAULT 0,
                        error TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        started_at TIMESTAMP,
                        completed_at TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS batch_sessions (
                        id TEXT PRIMARY KEY,
                        session_name TEXT,
                        total_jobs INTEGER,
                        completed_jobs INTEGER,
                        failed_jobs INTEGER,
                        total_records INTEGER,
                        start_time TIMESTAMP,
                        end_time TIMESTAMP,
                        status TEXT,
                        config_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                self.logger.info("Batch processing database initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize batch processing database: {e}")
            raise
    
    def create_historical_batch(self, 
                               symbols: List[str], 
                               intervals: List[str],
                               start_date: datetime,
                               end_date: datetime,
                               batch_size_hours: int = 24,
                               session_name: str = None) -> str:
        """
        Create a batch of historical data fetch jobs.
        
        Args:
            symbols: List of symbols to fetch
            intervals: List of intervals to fetch
            start_date: Start date for historical data
            end_date: End date for historical data
            batch_size_hours: Hours per batch job
            session_name: Optional session name
            
        Returns:
            Session ID for tracking
        """
        session_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger.info(f"Creating historical batch session: {session_id}")
        self.logger.info(f"Symbols: {symbols}, Intervals: {intervals}")
        self.logger.info(f"Date range: {start_date} to {end_date}")
        self.logger.info(f"Batch size: {batch_size_hours} hours")
        
        # Create batch jobs
        jobs_created = 0
        current_time = start_date
        
        while current_time < end_date:
            batch_end = min(current_time + timedelta(hours=batch_size_hours), end_date)
            
            for symbol in symbols:
                for interval in intervals:
                    job_id = f"{session_id}_{symbol}_{interval}_{current_time.strftime('%Y%m%d_%H%M')}"
                    
                    job = BatchJob(
                        job_id=job_id,
                        symbol=symbol,
                        interval=interval,
                        start_time=current_time,
                        end_time=batch_end
                    )
                    
                    self.jobs[job_id] = job
                    jobs_created += 1
            
            current_time = batch_end
        
        # Save session to database
        self._save_batch_session(session_id, session_name, jobs_created, start_date, end_date)
        
        self.logger.info(f"Created {jobs_created} batch jobs for session {session_id}")
        return session_id
    
    def process_batch_async(self, 
                           session_id: str,
                           config: BatchConfig = None) -> Dict[str, Any]:
        """
        Process batch jobs asynchronously with parallel execution.
        
        Args:
            session_id: Session ID to process
            config: Batch processing configuration
            
        Returns:
            Processing results
        """
        if config is None:
            config = BatchConfig()
        
        self.logger.info(f"Starting async batch processing for session: {session_id}")
        
        # Get jobs for this session
        session_jobs = [job for job in self.jobs.values() if job.job_id.startswith(session_id)]
        
        if not session_jobs:
            return {
                "success": False,
                "error": f"No jobs found for session {session_id}",
                "total_jobs": 0,
                "completed_jobs": 0,
                "failed_jobs": 0
            }
        
        # Start processing thread
        self.is_processing = True
        self.stop_processing = False
        
        self.processing_thread = threading.Thread(
            target=self._process_jobs_parallel,
            args=(session_jobs, config)
        )
        self.processing_thread.start()
        
        return {
            "success": True,
            "session_id": session_id,
            "total_jobs": len(session_jobs),
            "status": "processing",
            "message": "Batch processing started"
        }
    
    def _process_jobs_parallel(self, jobs: List[BatchJob], config: BatchConfig):
        """Process jobs in parallel with thread pool."""
        try:
            with ThreadPoolExecutor(max_workers=config.max_concurrent_jobs) as executor:
                # Submit all jobs
                future_to_job = {
                    executor.submit(self._process_single_job, job, config): job 
                    for job in jobs
                }
                
                # Process completed jobs
                for future in as_completed(future_to_job):
                    job = future_to_job[future]
                    
                    try:
                        result = future.result()
                        if result["success"]:
                            job.status = BatchStatus.COMPLETED
                            job.completed_at = datetime.now()
                            job.records_fetched = result.get("records_count", 0)
                            self.completed_jobs[job.job_id] = job
                            self.completed_jobs_count += 1
                            self.total_records_fetched += job.records_fetched
                            
                            if config.progress_callback:
                                config.progress_callback(job, result)
                        else:
                            job.status = BatchStatus.FAILED
                            job.error = result.get("error", "Unknown error")
                            self.failed_jobs[job.job_id] = job
                            self.failed_jobs_count += 1
                            
                            if config.error_callback:
                                config.error_callback(job, result)
                    
                    except Exception as e:
                        job.status = BatchStatus.FAILED
                        job.error = str(e)
                        self.failed_jobs[job.job_id] = job
                        self.failed_jobs_count += 1
                        
                        self.logger.error(f"Job {job.job_id} failed: {e}")
                        
                        if config.error_callback:
                            config.error_callback(job, {"error": str(e)})
                    
                    # Update job in database
                    self._update_job_status(job)
                    
                    if self.stop_processing:
                        break
        
        except Exception as e:
            self.logger.error(f"Error in parallel job processing: {e}")
        finally:
            self.is_processing = False
            self.logger.info(f"Batch processing completed. Success: {self.completed_jobs_count}, Failed: {self.failed_jobs_count}")
    
    def _process_single_job(self, job: BatchJob, config: BatchConfig) -> Dict[str, Any]:
        """Process a single batch job."""
        job.status = BatchStatus.RUNNING
        job.started_at = datetime.now()
        
        self.logger.info(f"Processing job {job.job_id}: {job.symbol} {job.interval} from {job.start_time} to {job.end_time}")
        
        # Update job status
        self._update_job_status(job)
        
        try:
            # Fetch data for this job
            result = self._fetch_job_data(job, config)
            
            if result["success"]:
                job.progress = 100.0
                self.logger.info(f"Job {job.job_id} completed successfully: {result['records_count']} records")
            else:
                self.logger.error(f"Job {job.job_id} failed: {result['error']}")
            
            return result
            
        except Exception as e:
            error_msg = f"Error processing job {job.job_id}: {e}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def _fetch_job_data(self, job: BatchJob, config: BatchConfig) -> Dict[str, Any]:
        """Fetch data for a specific job."""
        try:
            # Calculate time range
            time_range = job.end_time - job.start_time
            total_hours = time_range.total_seconds() / 3600
            
            # Determine batch size based on interval
            if job.interval == '1m':
                records_per_hour = 60
            elif job.interval == '5m':
                records_per_hour = 12
            elif job.interval == '15m':
                records_per_hour = 4
            elif job.interval == '1h':
                records_per_hour = 1
            else:
                records_per_hour = 60  # Default to 1-minute equivalent
            
            # Calculate optimal batch size
            estimated_records = int(total_hours * records_per_hour)
            batch_limit = min(2000, max(100, estimated_records))
            
            # Fetch data in chunks if needed
            all_data = []
            current_start = job.start_time
            
            while current_start < job.end_time:
                # Calculate chunk end time
                chunk_hours = min(config.batch_size_hours, (job.end_time - current_start).total_seconds() / 3600)
                chunk_end = current_start + timedelta(hours=chunk_hours)
                
                # Fetch chunk data
                chunk_data = self._fetch_ohlcv_chunk(
                    job.symbol, 
                    job.interval, 
                    current_start, 
                    chunk_end, 
                    batch_limit
                )
                
                if chunk_data:
                    all_data.extend(chunk_data)
                    job.progress = min(95.0, ((chunk_end - job.start_time).total_seconds() / time_range.total_seconds()) * 100)
                
                current_start = chunk_end
                
                # Small delay to avoid rate limiting
                time.sleep(0.1)
            
            if all_data:
                # Convert to DataFrame and save
                df = self._process_fetched_data(all_data, job.symbol, job.interval)
                
                if not df.empty:
                    # Save to parquet (this would integrate with existing data ingestor)
                    file_path = self._save_batch_data(df, job.symbol, job.interval)
                    
                    return {
                        "success": True,
                        "records_count": len(df),
                        "file_path": file_path,
                        "last_record_time": df['open_time'].max() if 'open_time' in df.columns else None
                    }
                else:
                    return {"success": False, "error": "No valid data processed"}
            else:
                return {"success": False, "error": "No data fetched from API"}
                
        except Exception as e:
            return {"success": False, "error": f"Error fetching job data: {e}"}
    
    def _fetch_ohlcv_chunk(self, symbol: str, interval: str, start_time: datetime, end_time: datetime, limit: int) -> List[List]:
        """Fetch a chunk of OHLCV data."""
        try:
            import requests
            
            # Convert interval to Bybit format
            interval_map = {
                '1m': '1', '3m': '3', '5m': '5', '15m': '15', 
                '30m': '30', '1h': '60', '2h': '120', '4h': '240',
                '6h': '360', '12h': '720', '1d': 'D', '1w': 'W'
            }
            bybit_interval = interval_map.get(interval, '1')
            
            # Convert to milliseconds
            start_ms = int(start_time.timestamp() * 1000)
            end_ms = int(end_time.timestamp() * 1000)
            
            url = f"{self.base_url}/v5/market/kline"
            params = {
                'category': 'linear',
                'symbol': symbol,
                'interval': bybit_interval,
                'start': start_ms,
                'end': end_ms,
                'limit': limit
            }
            
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get("retCode") != 0:
                raise Exception(f"Bybit API Error: {data.get('retMsg', 'Unknown error')}")
            
            if "result" in data and "list" in data["result"]:
                return data["result"]["list"]
            else:
                return []
                
        except Exception as e:
            self.logger.error(f"Error fetching OHLCV chunk for {symbol} {interval}: {e}")
            return []
    
    def _process_fetched_data(self, data: List[List], symbol: str, interval: str) -> pd.DataFrame:
        """Process fetched data into DataFrame."""
        try:
            if not data:
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'start_time', 'open', 'high', 'low', 'close', 'volume', 'turnover'
            ])
            
            # Convert timestamps
            df['start_time'] = pd.to_datetime(pd.to_numeric(df['start_time']), unit='ms')
            df['open_time'] = df['start_time']  # Standardize column name
            
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
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error processing fetched data: {e}")
            return pd.DataFrame()
    
    def _save_batch_data(self, df: pd.DataFrame, symbol: str, interval: str) -> str:
        """Save batch data to parquet file."""
        try:
            # This would integrate with the existing data ingestor's save methods
            # For now, just return a placeholder path
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"batch_{symbol}_{interval}_{timestamp}.parquet"
            filepath = f"data/parquet/batch/{filename}"
            
            # Create directory if it doesn't exist
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            
            # Save to parquet
            df.to_parquet(filepath, index=False)
            
            self.logger.info(f"Saved batch data to {filepath}: {len(df)} records")
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving batch data: {e}")
            return ""
    
    def get_batch_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of a batch processing session."""
        session_jobs = [job for job in self.jobs.values() if job.job_id.startswith(session_id)]
        
        if not session_jobs:
            return {"error": f"No jobs found for session {session_id}"}
        
        # Calculate statistics
        total_jobs = len(session_jobs)
        completed = len([job for job in session_jobs if job.status == BatchStatus.COMPLETED])
        failed = len([job for job in session_jobs if job.status == BatchStatus.FAILED])
        running = len([job for job in session_jobs if job.status == BatchStatus.RUNNING])
        pending = len([job for job in session_jobs if job.status == BatchStatus.PENDING])
        
        # Calculate overall progress
        total_progress = sum(job.progress for job in session_jobs) / total_jobs if total_jobs > 0 else 0
        
        # Calculate total records
        total_records = sum(job.records_fetched for job in session_jobs)
        
        return {
            "session_id": session_id,
            "total_jobs": total_jobs,
            "completed": completed,
            "failed": failed,
            "running": running,
            "pending": pending,
            "progress": total_progress,
            "total_records": total_records,
            "is_processing": self.is_processing,
            "jobs": [
                {
                    "job_id": job.job_id,
                    "symbol": job.symbol,
                    "interval": job.interval,
                    "start_time": job.start_time.isoformat(),
                    "end_time": job.end_time.isoformat(),
                    "status": job.status.value,
                    "progress": job.progress,
                    "records_fetched": job.records_fetched,
                    "error": job.error
                }
                for job in session_jobs
            ]
        }
    
    def stop_batch_processing(self):
        """Stop batch processing gracefully."""
        self.logger.info("Stopping batch processing...")
        self.stop_processing = True
        
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=30)
        
        self.logger.info("Batch processing stopped")
    
    def _save_batch_session(self, session_id: str, session_name: str, total_jobs: int, start_time: datetime, end_time: datetime):
        """Save batch session to database."""
        try:
            import sqlite3
            
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO batch_sessions 
                    (id, session_name, total_jobs, start_time, end_time, status)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    session_id,
                    session_name or f"Batch_{session_id}",
                    total_jobs,
                    start_time,
                    end_time,
                    "created"
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error saving batch session: {e}")
    
    def _update_job_status(self, job: BatchJob):
        """Update job status in database."""
        try:
            import sqlite3
            
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO batch_jobs 
                    (id, symbol, interval, start_time, end_time, status, progress, 
                     records_fetched, error, started_at, completed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    job.job_id,
                    job.symbol,
                    job.interval,
                    job.start_time,
                    job.end_time,
                    job.status.value,
                    job.progress,
                    job.records_fetched,
                    job.error,
                    job.started_at,
                    job.completed_at
                ))
                
                conn.commit()
                
        except Exception as e:
            self.logger.error(f"Error updating job status: {e}")

def main():
    """Main function for standalone batch processing."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Batch Processor")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"], help="Symbols to process")
    parser.add_argument("--intervals", nargs="+", default=["1m", "5m", "15m", "1h"], help="Intervals to process")
    parser.add_argument("--start-date", required=True, help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end-date", required=True, help="End date (YYYY-MM-DD)")
    parser.add_argument("--batch-size", type=int, default=24, help="Batch size in hours")
    parser.add_argument("--max-concurrent", type=int, default=5, help="Max concurrent jobs")
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
    
    # Create batch processor
    processor = BatchProcessor(db_file=args.db_file, log_level=args.log_level)
    
    # Create batch configuration
    config = BatchConfig(
        max_concurrent_jobs=args.max_concurrent,
        batch_size_hours=args.batch_size
    )
    
    # Create batch session
    session_id = processor.create_historical_batch(
        symbols=args.symbols,
        intervals=args.intervals,
        start_date=start_date,
        end_date=end_date,
        batch_size_hours=args.batch_size
    )
    
    print(f"Created batch session: {session_id}")
    
    # Process batch
    result = processor.process_batch_async(session_id, config)
    print(f"Batch processing started: {result}")
    
    # Monitor progress
    while processor.is_processing:
        status = processor.get_batch_status(session_id)
        print(f"Progress: {status['progress']:.1f}% ({status['completed']}/{status['total_jobs']} jobs)")
        time.sleep(10)
    
    # Final status
    final_status = processor.get_batch_status(session_id)
    print(f"Final status: {final_status}")

if __name__ == "__main__":
    main()
