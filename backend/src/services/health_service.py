#!/usr/bin/env python3
"""
Health Service for Crypto Trading Bot

Provides health check endpoints and system status monitoring.
"""

import os
import sys
import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
import sqlite3

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

class HealthService:
    """
    Health monitoring service for the crypto trading bot.
    Provides endpoints for health checks, system status, and monitoring.
    """
    
    def __init__(self, 
                 data_folder: str = "data/parquet",
                 db_file: str = "data/sqlite/runs.db",
                 log_level: str = "INFO"):
        """
        Initialize the health service.
        
        Args:
            data_folder: Directory containing parquet files
            db_file: SQLite database file path
            log_level: Logging level
        """
        self.data_folder = Path(data_folder)
        self.db_file = Path(db_file)
        self.app = FastAPI(title="Crypto Trading Bot Health Service", version="1.0.0")
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Setup routes
        self._setup_routes()
        
        # Track startup time
        self.startup_time = datetime.now()
        self.last_heartbeat = datetime.now()
    
    def _setup_routes(self):
        """Setup FastAPI routes."""
        
        @self.app.get("/health")
        async def health_check():
            """Basic health check endpoint."""
            try:
                # Update heartbeat
                self.last_heartbeat = datetime.now()
                
                # Check system status
                status = await self._get_system_status()
                
                return JSONResponse(content={
                    "status": "healthy",
                    "timestamp": datetime.now().isoformat(),
                    "uptime_seconds": (datetime.now() - self.startup_time).total_seconds(),
                    "last_heartbeat": self.last_heartbeat.isoformat(),
                    "system": status
                })
            except Exception as e:
                self.logger.error(f"Health check failed: {e}")
                raise HTTPException(status_code=503, detail=f"Health check failed: {str(e)}")
        
        @self.app.get("/health/detailed")
        async def detailed_health():
            """Detailed health check with system information."""
            try:
                status = await self._get_detailed_status()
                return JSONResponse(content=status)
            except Exception as e:
                self.logger.error(f"Detailed health check failed: {e}")
                raise HTTPException(status_code=503, detail=f"Detailed health check failed: {str(e)}")
        
        @self.app.get("/status")
        async def system_status():
            """System status endpoint."""
            try:
                status = await self._get_system_status()
                return JSONResponse(content=status)
            except Exception as e:
                self.logger.error(f"Status check failed: {e}")
                raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")
        
        @self.app.get("/data/status")
        async def data_status():
            """Data status endpoint showing parquet file information."""
            try:
                data_status = await self._get_data_status()
                return JSONResponse(content=data_status)
            except Exception as e:
                self.logger.error(f"Data status check failed: {e}")
                raise HTTPException(status_code=500, detail=f"Data status check failed: {str(e)}")
    
    async def _get_system_status(self) -> Dict[str, Any]:
        """Get basic system status."""
        try:
            # Check if data directories exist
            data_dir_exists = self.data_folder.exists()
            db_file_exists = self.db_file.exists()
            
            # Check disk space
            disk_usage = self._get_disk_usage()
            
            return {
                "data_directory_exists": data_dir_exists,
                "database_file_exists": db_file_exists,
                "disk_usage": disk_usage,
                "python_version": sys.version,
                "working_directory": str(Path.cwd())
            }
        except Exception as e:
            self.logger.error(f"Error getting system status: {e}")
            return {"error": str(e)}
    
    async def _get_detailed_status(self) -> Dict[str, Any]:
        """Get detailed system status."""
        try:
            system_status = await self._get_system_status()
            data_status = await self._get_data_status()
            
            return {
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - self.startup_time).total_seconds(),
                "last_heartbeat": self.last_heartbeat.isoformat(),
                "system": system_status,
                "data": data_status,
                "environment": {
                    "python_path": sys.executable,
                    "environment_variables": {
                        "PYTHONPATH": os.environ.get("PYTHONPATH", "Not set"),
                        "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO")
                    }
                }
            }
        except Exception as e:
            self.logger.error(f"Error getting detailed status: {e}")
            return {"error": str(e)}
    
    async def _get_data_status(self) -> Dict[str, Any]:
        """Get data status information."""
        try:
            data_status = {
                "parquet_files": [],
                "total_records": 0,
                "last_update": None
            }
            
            if self.data_folder.exists():
                # Get all parquet files
                parquet_files = list(self.data_folder.glob("*.parquet"))
                
                for file_path in parquet_files:
                    try:
                        df = pd.read_parquet(file_path)
                        file_info = {
                            "filename": file_path.name,
                            "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                            "records": len(df),
                            "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                        }
                        
                        # Get date range if it's OHLCV data
                        if "ohlcv" in file_path.name:
                            if 'open_time' in df.columns:
                                file_info["date_range"] = {
                                    "start": df['open_time'].min().isoformat(),
                                    "end": df['open_time'].max().isoformat()
                                }
                        elif "funding" in file_path.name:
                            if 'fundingTime' in df.columns:
                                file_info["date_range"] = {
                                    "start": df['fundingTime'].min().isoformat(),
                                    "end": df['fundingTime'].max().isoformat()
                                }
                        
                        data_status["parquet_files"].append(file_info)
                        data_status["total_records"] += len(df)
                        
                        # Track most recent update
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if data_status["last_update"] is None or file_time > data_status["last_update"]:
                            data_status["last_update"] = file_time.isoformat()
                            
                    except Exception as e:
                        self.logger.warning(f"Error reading parquet file {file_path}: {e}")
            
            return data_status
            
        except Exception as e:
            self.logger.error(f"Error getting data status: {e}")
            return {"error": str(e)}
    
    def _get_data_status_sync(self) -> Dict[str, Any]:
        """Get data status information (synchronous version)."""
        try:
            data_status = {
                "parquet_files": [],
                "total_records": 0,
                "last_update": None
            }
            
            if self.data_folder.exists():
                # Get all parquet files
                parquet_files = list(self.data_folder.glob("*.parquet"))
                
                for file_path in parquet_files:
                    try:
                        df = pd.read_parquet(file_path)
                        file_info = {
                            "filename": file_path.name,
                            "size_mb": round(file_path.stat().st_size / (1024 * 1024), 2),
                            "records": len(df),
                            "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
                        }
                        
                        # Get date range if it's OHLCV data
                        if "ohlcv" in file_path.name:
                            if 'open_time' in df.columns:
                                file_info["date_range"] = {
                                    "start": df['open_time'].min().isoformat(),
                                    "end": df['open_time'].max().isoformat()
                                }
                        elif "funding" in file_path.name:
                            if 'fundingTime' in df.columns:
                                file_info["date_range"] = {
                                    "start": df['fundingTime'].min().isoformat(),
                                    "end": df['fundingTime'].max().isoformat()
                                }
                        
                        data_status["parquet_files"].append(file_info)
                        data_status["total_records"] += len(df)
                        
                        # Track most recent update
                        file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
                        if data_status["last_update"] is None or file_time > data_status["last_update"]:
                            data_status["last_update"] = file_time.isoformat()
                            
                    except Exception as e:
                        self.logger.warning(f"Error reading parquet file {file_path}: {e}")
            
            return data_status
            
        except Exception as e:
            self.logger.error(f"Error getting data status: {e}")
            return {"error": str(e)}
    
    def _get_disk_usage(self) -> Dict[str, Any]:
        """Get disk usage information."""
        try:
            import shutil
            
            # Get disk usage for current directory
            total, used, free = shutil.disk_usage(".")
            
            return {
                "total_gb": round(total / (1024**3), 2),
                "used_gb": round(used / (1024**3), 2),
                "free_gb": round(free / (1024**3), 2),
                "usage_percent": round((used / total) * 100, 2)
            }
        except Exception as e:
            self.logger.error(f"Error getting disk usage: {e}")
            return {"error": str(e)}
    
    def update_heartbeat(self):
        """Update the heartbeat timestamp."""
        self.last_heartbeat = datetime.now()
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get basic health status (synchronous version for dashboard API)."""
        try:
            # Update heartbeat
            self.last_heartbeat = datetime.now()
            
            # Get system status synchronously
            data_dir_exists = self.data_folder.exists()
            db_file_exists = self.db_file.exists()
            disk_usage = self._get_disk_usage()
            
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - self.startup_time).total_seconds(),
                "last_heartbeat": self.last_heartbeat.isoformat(),
                "system": {
                    "data_directory_exists": data_dir_exists,
                    "database_file_exists": db_file_exists,
                    "disk_usage": disk_usage,
                    "python_version": sys.version,
                    "working_directory": str(Path.cwd())
                }
            }
        except Exception as e:
            self.logger.error(f"Health status check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def get_detailed_health(self) -> Dict[str, Any]:
        """Get detailed health status (synchronous version for dashboard API)."""
        try:
            system_status = {
                "data_directory_exists": self.data_folder.exists(),
                "database_file_exists": self.db_file.exists(),
                "disk_usage": self._get_disk_usage(),
                "python_version": sys.version,
                "working_directory": str(Path.cwd())
            }
            
            # Get data status synchronously
            data_status = self._get_data_status_sync()
            
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "uptime_seconds": (datetime.now() - self.startup_time).total_seconds(),
                "last_heartbeat": self.last_heartbeat.isoformat(),
                "system": system_status,
                "data": data_status,
                "environment": {
                    "python_path": sys.executable,
                    "environment_variables": {
                        "PYTHONPATH": os.environ.get("PYTHONPATH", "Not set"),
                        "LOG_LEVEL": os.environ.get("LOG_LEVEL", "INFO")
                    }
                }
            }
        except Exception as e:
            self.logger.error(f"Detailed health check failed: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def run_server(self, host: str = "0.0.0.0", port: int = 8080):
        """Run the health service server."""
        self.logger.info(f"Starting health service on {host}:{port}")
        uvicorn.run(self.app, host=host, port=port, log_level="info")

def main():
    """Main function to run the health service standalone."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Crypto Trading Bot Health Service")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--data-folder", default="data/parquet", help="Data folder path")
    parser.add_argument("--db-file", default="data/sqlite/runs.db", help="Database file path")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run health service
    health_service = HealthService(
        data_folder=args.data_folder,
        db_file=args.db_file,
        log_level=args.log_level
    )
    
    health_service.run_server(host=args.host, port=args.port)

if __name__ == "__main__":
    main()
