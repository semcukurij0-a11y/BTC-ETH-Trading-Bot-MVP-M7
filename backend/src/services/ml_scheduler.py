"""
ML Scheduler Service for Crypto Trading Bot

This module handles scheduled ML model training and retraining.
Supports weekly retrain functionality with optional scheduling.
"""

import os
import json
import logging
import schedule
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from .ml_trainer import MLTrainer
from .ml_inference import MLInference


class MLScheduler:
    """
    ML Scheduler for automated model training and retraining.
    
    Features:
    - Weekly retrain scheduling
    - Multi-symbol/interval training
    - Background training execution
    - Training status monitoring
    - Error handling and recovery
    """
    
    def __init__(self, 
                 config: Optional[Dict] = None,
                 trainer_config: Optional[Dict] = None,
                 inference_config: Optional[Dict] = None):
        """
        Initialize ML Scheduler.
        
        Args:
            config: Scheduler configuration
            trainer_config: ML Trainer configuration
            inference_config: ML Inference configuration
        """
        self.config = config or {}
        self.trainer_config = trainer_config or {}
        self.inference_config = inference_config or {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Scheduler configuration
        self.enabled = self.config.get('enabled', True)
        self.retrain_schedule = self.config.get('retrain_schedule', 'sunday 02:00')  # Weekly on Sunday at 2 AM
        self.max_workers = self.config.get('max_workers', 2)
        self.timeout_hours = self.config.get('timeout_hours', 6)
        
        # Training configuration
        self.symbols = self.config.get('symbols', ['BTCUSDT', 'ETHUSDT'])
        self.intervals = self.config.get('intervals', ['15m', '1h'])
        self.days_back = self.config.get('days_back', 30)
        self.target_hours = self.config.get('target_hours', 1)
        self.threshold = self.config.get('threshold', 0.001)
        
        # Initialize services
        self.trainer = MLTrainer(config=self.trainer_config)
        self.inference = MLInference("src/models", self.inference_config)
        
        # Training status
        self.training_status = {}
        self.last_training = {}
        self.training_errors = {}
        
        # Background thread
        self.scheduler_thread = None
        self.running = False
        
    def setup_schedule(self):
        """Setup the training schedule."""
        if not self.enabled:
            self.logger.info("ML Scheduler is disabled")
            return
        
        # Parse schedule
        schedule_parts = self.retrain_schedule.split()
        if len(schedule_parts) != 2:
            raise ValueError(f"Invalid schedule format: {self.retrain_schedule}. Use 'day HH:MM' format.")
        
        day, time_str = schedule_parts
        
        # Schedule weekly retrain
        if day.lower() == 'sunday':
            schedule.every().sunday.at(time_str).do(self.run_weekly_retrain)
        elif day.lower() == 'monday':
            schedule.every().monday.at(time_str).do(self.run_weekly_retrain)
        elif day.lower() == 'tuesday':
            schedule.every().tuesday.at(time_str).do(self.run_weekly_retrain)
        elif day.lower() == 'wednesday':
            schedule.every().wednesday.at(time_str).do(self.run_weekly_retrain)
        elif day.lower() == 'thursday':
            schedule.every().thursday.at(time_str).do(self.run_weekly_retrain)
        elif day.lower() == 'friday':
            schedule.every().friday.at(time_str).do(self.run_weekly_retrain)
        elif day.lower() == 'saturday':
            schedule.every().saturday.at(time_str).do(self.run_weekly_retrain)
        else:
            raise ValueError(f"Invalid day: {day}")
        
        self.logger.info(f"ML Scheduler configured: {self.retrain_schedule}")
    
    def run_weekly_retrain(self):
        """Run weekly retrain for all configured symbols/intervals."""
        self.logger.info("Starting weekly ML retrain")
        
        start_time = datetime.now()
        results = {}
        
        try:
            # Check which models need retraining
            models_to_train = []
            for symbol in self.symbols:
                for interval in self.intervals:
                    if self.trainer.should_retrain(symbol, interval):
                        models_to_train.append((symbol, interval))
                        self.logger.info(f"Model needs retraining: {symbol} {interval}")
                    else:
                        self.logger.info(f"Model is up to date: {symbol} {interval}")
            
            if not models_to_train:
                self.logger.info("No models need retraining")
                return
            
            # Train models in parallel
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit training tasks
                future_to_model = {}
                for symbol, interval in models_to_train:
                    future = executor.submit(
                        self._train_single_model,
                        symbol, interval
                    )
                    future_to_model[future] = (symbol, interval)
                
                # Collect results
                for future in as_completed(future_to_model, timeout=self.timeout_hours * 3600):
                    symbol, interval = future_to_model[future]
                    try:
                        result = future.result()
                        results[f"{symbol}_{interval}"] = result
                        self.training_status[f"{symbol}_{interval}"] = 'completed'
                        self.last_training[f"{symbol}_{interval}"] = datetime.now()
                        self.logger.info(f"Training completed: {symbol} {interval}")
                    except Exception as e:
                        error_msg = str(e)
                        results[f"{symbol}_{interval}"] = {'error': error_msg}
                        self.training_status[f"{symbol}_{interval}"] = 'failed'
                        self.training_errors[f"{symbol}_{interval}"] = error_msg
                        self.logger.error(f"Training failed: {symbol} {interval} - {error_msg}")
            
            # Summary
            duration = datetime.now() - start_time
            successful = sum(1 for r in results.values() if 'error' not in r)
            failed = len(results) - successful
            
            self.logger.info(f"Weekly retrain completed: {successful} successful, {failed} failed, {duration}")
            
        except Exception as e:
            self.logger.error(f"Weekly retrain failed: {e}")
    
    def _train_single_model(self, symbol: str, interval: str) -> Dict[str, Any]:
        """
        Train a single model.
        
        Args:
            symbol: Trading symbol
            interval: Timeframe
            
        Returns:
            Training result dictionary
        """
        try:
            self.training_status[f"{symbol}_{interval}"] = 'training'
            
            result = self.trainer.train_model(
                symbol=symbol,
                interval=interval,
                days_back=self.days_back,
                target_hours=self.target_hours,
                threshold=self.threshold
            )
            
            return result
            
        except Exception as e:
            raise Exception(f"Training failed for {symbol} {interval}: {e}")
    
    def train_model_now(self, symbol: str, interval: str) -> Dict[str, Any]:
        """
        Train a model immediately (manual trigger).
        
        Args:
            symbol: Trading symbol
            interval: Timeframe
            
        Returns:
            Training result dictionary
        """
        self.logger.info(f"Manual training triggered: {symbol} {interval}")
        
        try:
            result = self._train_single_model(symbol, interval)
            self.training_status[f"{symbol}_{interval}"] = 'completed'
            self.last_training[f"{symbol}_{interval}"] = datetime.now()
            
            return result
            
        except Exception as e:
            self.training_status[f"{symbol}_{interval}"] = 'failed'
            self.training_errors[f"{symbol}_{interval}"] = str(e)
            raise
    
    def get_training_status(self) -> Dict[str, Any]:
        """
        Get current training status for all models.
        
        Returns:
            Status dictionary
        """
        status = {
            'scheduler_enabled': self.enabled,
            'retrain_schedule': self.retrain_schedule,
            'models': {}
        }
        
        for symbol in self.symbols:
            for interval in self.intervals:
                key = f"{symbol}_{interval}"
                model_status = {
                    'symbol': symbol,
                    'interval': interval,
                    'status': self.training_status.get(key, 'unknown'),
                    'last_training': self.last_training.get(key),
                    'error': self.training_errors.get(key),
                    'needs_retrain': self.trainer.should_retrain(symbol, interval)
                }
                status['models'][key] = model_status
        
        return status
    
    def start_scheduler(self):
        """Start the scheduler in a background thread."""
        if not self.enabled:
            self.logger.info("ML Scheduler is disabled")
            return
        
        if self.running:
            self.logger.warning("Scheduler is already running")
            return
        
        self.setup_schedule()
        self.running = True
        
        def run_scheduler():
            self.logger.info("ML Scheduler started")
            while self.running:
                try:
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    self.logger.error(f"Scheduler error: {e}")
                    time.sleep(60)
        
        self.scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        self.logger.info("ML Scheduler thread started")
    
    def stop_scheduler(self):
        """Stop the scheduler."""
        if not self.running:
            return
        
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        
        self.logger.info("ML Scheduler stopped")
    
    def get_next_retrain_time(self) -> Optional[datetime]:
        """
        Get the next scheduled retrain time.
        
        Returns:
            Next retrain datetime or None
        """
        if not self.enabled:
            return None
        
        # Get next scheduled job
        jobs = schedule.get_jobs()
        if not jobs:
            return None
        
        # Find the retrain job
        for job in jobs:
            if job.job_func == self.run_weekly_retrain:
                return job.next_run
        
        return None
    
    def force_retrain_all(self) -> Dict[str, Any]:
        """
        Force retrain all models (ignoring age check).
        
        Returns:
            Training results dictionary
        """
        self.logger.info("Force retrain all models triggered")
        
        start_time = datetime.now()
        results = {}
        
        try:
            # Train all models
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                # Submit training tasks
                future_to_model = {}
                for symbol in self.symbols:
                    for interval in self.intervals:
                        future = executor.submit(
                            self._train_single_model,
                            symbol, interval
                        )
                        future_to_model[future] = (symbol, interval)
                
                # Collect results
                for future in as_completed(future_to_model, timeout=self.timeout_hours * 3600):
                    symbol, interval = future_to_model[future]
                    try:
                        result = future.result()
                        results[f"{symbol}_{interval}"] = result
                        self.training_status[f"{symbol}_{interval}"] = 'completed'
                        self.last_training[f"{symbol}_{interval}"] = datetime.now()
                        self.logger.info(f"Force training completed: {symbol} {interval}")
                    except Exception as e:
                        error_msg = str(e)
                        results[f"{symbol}_{interval}"] = {'error': error_msg}
                        self.training_status[f"{symbol}_{interval}"] = 'failed'
                        self.training_errors[f"{symbol}_{interval}"] = error_msg
                        self.logger.error(f"Force training failed: {symbol} {interval} - {error_msg}")
            
            # Summary
            duration = datetime.now() - start_time
            successful = sum(1 for r in results.values() if 'error' not in r)
            failed = len(results) - successful
            
            self.logger.info(f"Force retrain completed: {successful} successful, {failed} failed, {duration}")
            
            return {
                'successful': successful,
                'failed': failed,
                'duration': str(duration),
                'results': results
            }
            
        except Exception as e:
            self.logger.error(f"Force retrain failed: {e}")
            return {'error': str(e)}


def main():
    """Test the ML Scheduler."""
    import sys
    sys.path.append('src')
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = {
        'enabled': True,
        'retrain_schedule': 'sunday 02:00',
        'max_workers': 2,
        'timeout_hours': 6,
        'symbols': ['BTCUSDT'],
        'intervals': ['15m'],
        'days_back': 30,
        'target_hours': 1,
        'threshold': 0.001
    }
    
    trainer_config = {
        'model_type': 'lightgbm',
        'calibration_method': 'isotonic',
        'weekly_retrain': True
    }
    
    # Initialize scheduler
    scheduler = MLScheduler(config=config, trainer_config=trainer_config)
    
    # Test manual training
    try:
        result = scheduler.train_model_now('BTCUSDT', '15m')
        print("Manual training completed successfully!")
        print(f"Result: {result}")
        
    except Exception as e:
        print(f"Manual training failed: {e}")
    
    # Test status
    status = scheduler.get_training_status()
    print(f"Training status: {status}")


if __name__ == "__main__":
    main()
