#!/usr/bin/env python3
"""
Scheduler Service for Crypto Trading Bot

Provides scheduled execution of trading bot workflows using APScheduler.
"""

import os
import sys
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import json

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger
from apscheduler.triggers.cron import CronTrigger
from apscheduler.events import EVENT_JOB_EXECUTED, EVENT_JOB_ERROR
import signal

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import will be done dynamically to avoid circular imports
# from main import TradingBotWorkflow

class SchedulerService:
    """
    Scheduler service for the crypto trading bot.
    Manages scheduled execution of trading workflows.
    """
    
    def __init__(self, 
                 config_file: Optional[str] = None,
                 log_level: str = "INFO"):
        """
        Initialize the scheduler service.
        
        Args:
            config_file: Path to configuration file
            log_level: Logging level
        """
        self.config_file = config_file
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Initialize scheduler
        self.scheduler = AsyncIOScheduler()
        
        # Initialize workflow
        self.workflow = None
        
        # Track job statistics
        self.job_stats = {
            "total_executions": 0,
            "successful_executions": 0,
            "failed_executions": 0,
            "last_execution": None,
            "last_success": None,
            "last_failure": None
        }
        
        # Setup event listeners
        self._setup_event_listeners()
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_event_listeners(self):
        """Setup scheduler event listeners."""
        self.scheduler.add_listener(self._job_executed, EVENT_JOB_EXECUTED)
        self.scheduler.add_listener(self._job_error, EVENT_JOB_ERROR)
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown."""
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.stop()
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
    
    def _job_executed(self, event):
        """Handle successful job execution."""
        self.job_stats["total_executions"] += 1
        self.job_stats["successful_executions"] += 1
        self.job_stats["last_execution"] = datetime.now().isoformat()
        self.job_stats["last_success"] = datetime.now().isoformat()
        
        self.logger.info(f"Job {event.job_id} executed successfully")
    
    def _job_error(self, event):
        """Handle job execution error."""
        self.job_stats["total_executions"] += 1
        self.job_stats["failed_executions"] += 1
        self.job_stats["last_execution"] = datetime.now().isoformat()
        self.job_stats["last_failure"] = datetime.now().isoformat()
        
        self.logger.error(f"Job {event.job_id} failed: {event.exception}")
    
    async def _execute_workflow(self):
        """Execute the trading bot workflow."""
        try:
            self.logger.info("Starting scheduled workflow execution")
            
            if self.workflow is None:
                # Dynamic import to avoid circular dependency
                from main import TradingBotWorkflow
                self.workflow = TradingBotWorkflow(config_file=self.config_file)
            
            # Run the workflow
            result = self.workflow.run_single_workflow()
            
            if result["success"]:
                self.logger.info("Workflow completed successfully")
            else:
                self.logger.error(f"Workflow failed: {result.get('errors', 'Unknown error')}")
                
        except Exception as e:
            self.logger.error(f"Error executing workflow: {e}")
            raise
    
    def add_interval_job(self, 
                        interval_seconds: int = 60,
                        job_id: str = "trading_workflow"):
        """
        Add an interval-based job.
        
        Args:
            interval_seconds: Interval between executions in seconds
            job_id: Unique job identifier
        """
        self.scheduler.add_job(
            func=self._execute_workflow,
            trigger=IntervalTrigger(seconds=interval_seconds),
            id=job_id,
            name="Trading Bot Workflow",
            max_instances=1,  # Prevent overlapping executions
            replace_existing=True
        )
        
        self.logger.info(f"Added interval job '{job_id}' with {interval_seconds}s interval")
    
    def add_cron_job(self, 
                    cron_expression: str = "*/1 * * * *",  # Every minute
                    job_id: str = "trading_workflow"):
        """
        Add a cron-based job.
        
        Args:
            cron_expression: Cron expression for scheduling
            job_id: Unique job identifier
        """
        self.scheduler.add_job(
            func=self._execute_workflow,
            trigger=CronTrigger.from_crontab(cron_expression),
            id=job_id,
            name="Trading Bot Workflow",
            max_instances=1,  # Prevent overlapping executions
            replace_existing=True
        )
        
        self.logger.info(f"Added cron job '{job_id}' with expression '{cron_expression}'")
    
    def add_custom_schedule(self, 
                           schedule_config: Dict[str, Any],
                           job_id: str = "trading_workflow"):
        """
        Add a custom scheduled job based on configuration.
        
        Args:
            schedule_config: Dictionary containing schedule configuration
            job_id: Unique job identifier
        """
        schedule_type = schedule_config.get("type", "interval")
        
        if schedule_type == "interval":
            interval = schedule_config.get("interval", 60)
            self.add_interval_job(interval_seconds=interval, job_id=job_id)
        elif schedule_type == "cron":
            cron_expr = schedule_config.get("cron", "*/1 * * * *")
            self.add_cron_job(cron_expression=cron_expr, job_id=job_id)
        else:
            raise ValueError(f"Unknown schedule type: {schedule_type}")
    
    def start(self):
        """Start the scheduler."""
        try:
            self.scheduler.start()
            self.logger.info("Scheduler started successfully")
        except Exception as e:
            self.logger.error(f"Failed to start scheduler: {e}")
            raise
    
    def stop(self):
        """Stop the scheduler."""
        try:
            self.scheduler.shutdown(wait=True)
            self.logger.info("Scheduler stopped successfully")
        except Exception as e:
            self.logger.error(f"Error stopping scheduler: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get scheduler status information."""
        try:
            jobs = []
            for job in self.scheduler.get_jobs():
                jobs.append({
                    "id": job.id,
                    "name": job.name,
                    "next_run": job.next_run_time.isoformat() if job.next_run_time else None,
                    "trigger": str(job.trigger)
                })
            
            return {
                "running": self.scheduler.running,
                "jobs": jobs,
                "statistics": self.job_stats
            }
        except Exception as e:
            self.logger.error(f"Error getting scheduler status: {e}")
            return {"error": str(e)}
    
    async def run_forever(self):
        """Run the scheduler forever."""
        try:
            self.start()
            
            # Keep the service running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            self.logger.info("Received keyboard interrupt, shutting down...")
        except Exception as e:
            self.logger.error(f"Error in scheduler service: {e}")
        finally:
            self.stop()

def main():
    """Main function to run the scheduler service standalone."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Crypto Trading Bot Scheduler Service")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--interval", type=int, default=60, help="Interval in seconds")
    parser.add_argument("--cron", help="Cron expression for scheduling")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create scheduler service
    scheduler_service = SchedulerService(
        config_file=args.config,
        log_level=args.log_level
    )
    
    # Add job based on arguments
    if args.cron:
        scheduler_service.add_cron_job(cron_expression=args.cron)
    else:
        scheduler_service.add_interval_job(interval_seconds=args.interval)
    
    # Run the scheduler
    try:
        asyncio.run(scheduler_service.run_forever())
    except KeyboardInterrupt:
        print("Scheduler service stopped by user")

if __name__ == "__main__":
    main()
