#!/usr/bin/env python3
"""
Performance Monitor for Crypto Trading Bot

Provides comprehensive performance monitoring and optimization recommendations
for data fetching operations.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import sqlite3
import json
import psutil
import time
from dataclasses import dataclass
from enum import Enum

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class PerformanceLevel(Enum):
    """Performance level classification"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    CRITICAL = "critical"

@dataclass
class PerformanceMetrics:
    """Performance metrics for data fetching"""
    fetch_time_seconds: float
    records_per_second: float
    memory_usage_mb: float
    api_calls: int
    error_rate: float
    compression_ratio: float
    throughput_mb_per_second: float

@dataclass
class OptimizationRecommendation:
    """Optimization recommendation"""
    category: str
    priority: str
    description: str
    impact: str
    implementation: str
    expected_improvement: str

class PerformanceMonitor:
    """
    Performance monitoring and optimization system for data fetching.
    
    Features:
    - Real-time performance monitoring
    - Historical performance analysis
    - Optimization recommendations
    - Performance benchmarking
    - Resource utilization tracking
    - Bottleneck identification
    """
    
    def __init__(self, 
                 db_file: str = "data/sqlite/runs.db",
                 log_level: str = "INFO"):
        """
        Initialize the performance monitor.
        
        Args:
            db_file: SQLite database file path
            log_level: Logging level
        """
        self.db_file = Path(db_file)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Performance thresholds
        self.thresholds = {
            'excellent_records_per_second': 1000,
            'good_records_per_second': 500,
            'fair_records_per_second': 200,
            'poor_records_per_second': 100,
            
            'excellent_memory_mb': 512,
            'good_memory_mb': 1024,
            'fair_memory_mb': 2048,
            'poor_memory_mb': 4096,
            
            'excellent_error_rate': 0.01,  # 1%
            'good_error_rate': 0.05,       # 5%
            'fair_error_rate': 0.10,       # 10%
            'poor_error_rate': 0.20,       # 20%
            
            'excellent_compression_ratio': 0.1,  # 10:1
            'good_compression_ratio': 0.2,       # 5:1
            'fair_compression_ratio': 0.5,       # 2:1
            'poor_compression_ratio': 1.0,       # 1:1
        }
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for performance monitoring."""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                # Create performance monitoring tables
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        start_time TIMESTAMP NOT NULL,
                        end_time TIMESTAMP,
                        total_records INTEGER,
                        total_size_mb REAL,
                        fetch_time_seconds REAL,
                        memory_peak_mb REAL,
                        api_calls INTEGER,
                        errors INTEGER,
                        strategy TEXT,
                        symbols TEXT,
                        intervals TEXT,
                        performance_score REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS performance_benchmarks (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        benchmark_name TEXT NOT NULL,
                        symbol TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        records_count INTEGER,
                        fetch_time_seconds REAL,
                        memory_usage_mb REAL,
                        records_per_second REAL,
                        throughput_mb_per_second REAL,
                        compression_ratio REAL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS optimization_recommendations (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        category TEXT NOT NULL,
                        priority TEXT NOT NULL,
                        description TEXT NOT NULL,
                        impact TEXT NOT NULL,
                        implementation TEXT NOT NULL,
                        expected_improvement TEXT NOT NULL,
                        applied BOOLEAN DEFAULT FALSE,
                        applied_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                self.logger.info("Performance monitoring database initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize performance monitoring database: {e}")
            raise
    
    def analyze_performance(self, session_id: str = None, days: int = 7) -> Dict[str, Any]:
        """Analyze performance data and generate insights."""
        try:
            # Get performance data
            if session_id:
                performance_data = self._get_session_performance(session_id)
            else:
                performance_data = self._get_recent_performance(days)
            
            if not performance_data:
                return {"error": "No performance data found"}
            
            # Calculate performance metrics
            metrics = self._calculate_performance_metrics(performance_data)
            
            # Generate optimization recommendations
            recommendations = self._generate_optimization_recommendations(metrics, performance_data)
            
            # Calculate performance score
            performance_score = self._calculate_performance_score(metrics)
            
            # Identify bottlenecks
            bottlenecks = self._identify_bottlenecks(metrics, performance_data)
            
            # Generate summary
            summary = self._generate_performance_summary(metrics, performance_score, bottlenecks)
            
            return {
                "session_id": session_id,
                "analysis_period_days": days,
                "performance_score": performance_score,
                "metrics": metrics.__dict__,
                "recommendations": [rec.__dict__ for rec in recommendations],
                "bottlenecks": bottlenecks,
                "summary": summary,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing performance: {e}")
            return {"error": str(e)}
    
    def _get_session_performance(self, session_id: str) -> List[Dict]:
        """Get performance data for a specific session."""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM fetch_performance 
                    WHERE session_id = ?
                    ORDER BY created_at DESC
                """, (session_id,))
                
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error getting session performance: {e}")
            return []
    
    def _get_recent_performance(self, days: int) -> List[Dict]:
        """Get recent performance data."""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM fetch_performance 
                    WHERE created_at >= datetime('now', '-{} days')
                    ORDER BY created_at DESC
                """.format(days))
                
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error getting recent performance: {e}")
            return []
    
    def _calculate_performance_metrics(self, performance_data: List[Dict]) -> PerformanceMetrics:
        """Calculate performance metrics from data."""
        if not performance_data:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0)
        
        # Calculate averages
        total_records = sum(row.get('total_records', 0) for row in performance_data)
        total_time = sum(row.get('fetch_time_seconds', 0) for row in performance_data)
        total_size = sum(row.get('total_size_mb', 0) for row in performance_data)
        total_api_calls = sum(row.get('api_calls', 0) for row in performance_data)
        total_errors = sum(row.get('errors', 0) for row in performance_data)
        avg_memory = np.mean([row.get('memory_peak_mb', 0) for row in performance_data])
        avg_compression = np.mean([row.get('compression_ratio', 1.0) for row in performance_data])
        
        # Calculate derived metrics
        records_per_second = total_records / max(1, total_time)
        error_rate = total_errors / max(1, total_api_calls)
        throughput_mb_per_second = total_size / max(1, total_time)
        
        return PerformanceMetrics(
            fetch_time_seconds=total_time,
            records_per_second=records_per_second,
            memory_usage_mb=avg_memory,
            api_calls=total_api_calls,
            error_rate=error_rate,
            compression_ratio=avg_compression,
            throughput_mb_per_second=throughput_mb_per_second
        )
    
    def _generate_optimization_recommendations(self, 
                                             metrics: PerformanceMetrics, 
                                             performance_data: List[Dict]) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on performance analysis."""
        recommendations = []
        
        # Records per second optimization
        if metrics.records_per_second < self.thresholds['good_records_per_second']:
            recommendations.append(OptimizationRecommendation(
                category="throughput",
                priority="high",
                description="Low records per second performance",
                impact="Significantly improves data fetching speed",
                implementation="Implement parallel processing and batch optimization",
                expected_improvement=f"Target: {self.thresholds['good_records_per_second']} records/sec"
            ))
        
        # Memory usage optimization
        if metrics.memory_usage_mb > self.thresholds['good_memory_mb']:
            recommendations.append(OptimizationRecommendation(
                category="memory",
                priority="medium",
                description="High memory usage detected",
                impact="Reduces memory footprint and enables larger datasets",
                implementation="Implement memory management and data streaming",
                expected_improvement=f"Target: <{self.thresholds['good_memory_mb']} MB"
            ))
        
        # Error rate optimization
        if metrics.error_rate > self.thresholds['good_error_rate']:
            recommendations.append(OptimizationRecommendation(
                category="reliability",
                priority="high",
                description="High error rate detected",
                impact="Improves data reliability and reduces retries",
                implementation="Implement retry logic and error handling",
                expected_improvement=f"Target: <{self.thresholds['good_error_rate']:.1%} error rate"
            ))
        
        # Compression optimization
        if metrics.compression_ratio > self.thresholds['good_compression_ratio']:
            recommendations.append(OptimizationRecommendation(
                category="storage",
                priority="low",
                description="Poor compression ratio",
                impact="Reduces storage requirements and improves I/O",
                implementation="Implement data compression and optimization",
                expected_improvement=f"Target: <{self.thresholds['good_compression_ratio']:.1f} compression ratio"
            ))
        
        # Strategy-specific recommendations
        strategies = [row.get('strategy', 'unknown') for row in performance_data]
        if 'standard' in strategies and len(strategies) > 1:
            recommendations.append(OptimizationRecommendation(
                category="strategy",
                priority="medium",
                description="Mixed strategy usage detected",
                impact="Optimizes fetch strategy selection",
                implementation="Use optimized or historical batch strategies",
                expected_improvement="20-50% performance improvement"
            ))
        
        return recommendations
    
    def _calculate_performance_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate overall performance score (0-100)."""
        score = 100.0
        
        # Deduct points for poor performance
        if metrics.records_per_second < self.thresholds['poor_records_per_second']:
            score -= 30
        elif metrics.records_per_second < self.thresholds['fair_records_per_second']:
            score -= 20
        elif metrics.records_per_second < self.thresholds['good_records_per_second']:
            score -= 10
        
        if metrics.memory_usage_mb > self.thresholds['poor_memory_mb']:
            score -= 20
        elif metrics.memory_usage_mb > self.thresholds['fair_memory_mb']:
            score -= 10
        
        if metrics.error_rate > self.thresholds['poor_error_rate']:
            score -= 25
        elif metrics.error_rate > self.thresholds['fair_error_rate']:
            score -= 15
        elif metrics.error_rate > self.thresholds['good_error_rate']:
            score -= 5
        
        if metrics.compression_ratio > self.thresholds['poor_compression_ratio']:
            score -= 10
        elif metrics.compression_ratio > self.thresholds['fair_compression_ratio']:
            score -= 5
        
        return max(0, min(100, score))
    
    def _identify_bottlenecks(self, metrics: PerformanceMetrics, performance_data: List[Dict]) -> List[str]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # API bottleneck
        if metrics.records_per_second < self.thresholds['fair_records_per_second']:
            bottlenecks.append("API rate limiting or network latency")
        
        # Memory bottleneck
        if metrics.memory_usage_mb > self.thresholds['fair_memory_mb']:
            bottlenecks.append("Memory usage limiting batch sizes")
        
        # Error bottleneck
        if metrics.error_rate > self.thresholds['fair_error_rate']:
            bottlenecks.append("High error rate causing retries")
        
        # Storage bottleneck
        if metrics.compression_ratio > self.thresholds['fair_compression_ratio']:
            bottlenecks.append("Poor compression increasing I/O overhead")
        
        # Strategy bottleneck
        strategies = [row.get('strategy', 'unknown') for row in performance_data]
        if 'standard' in strategies:
            bottlenecks.append("Suboptimal fetch strategy")
        
        return bottlenecks
    
    def _generate_performance_summary(self, 
                                    metrics: PerformanceMetrics, 
                                    performance_score: float,
                                    bottlenecks: List[str]) -> str:
        """Generate performance summary."""
        if performance_score >= 90:
            level = "EXCELLENT"
            emoji = "🟢"
        elif performance_score >= 75:
            level = "GOOD"
            emoji = "🟡"
        elif performance_score >= 60:
            level = "FAIR"
            emoji = "🟠"
        elif performance_score >= 40:
            level = "POOR"
            emoji = "🔴"
        else:
            level = "CRITICAL"
            emoji = "⚫"
        
        summary_parts = [
            f"{emoji} Performance Level: {level} ({performance_score:.1f}/100)",
            f"📊 Throughput: {metrics.records_per_second:.0f} records/sec",
            f"💾 Memory: {metrics.memory_usage_mb:.0f} MB",
            f"❌ Error Rate: {metrics.error_rate:.1%}",
            f"🗜️ Compression: {metrics.compression_ratio:.1f}x"
        ]
        
        if bottlenecks:
            summary_parts.append(f"⚠️ Bottlenecks: {', '.join(bottlenecks)}")
        
        return " | ".join(summary_parts)
    
    def benchmark_performance(self, 
                             symbol: str, 
                             interval: str,
                             records_count: int = 1000) -> Dict[str, Any]:
        """Benchmark performance for a specific symbol/interval combination."""
        try:
            self.logger.info(f"Benchmarking performance for {symbol} {interval}")
            
            # Record start time and memory
            start_time = time.time()
            start_memory = psutil.virtual_memory().used / (1024 * 1024)
            
            # Simulate data fetching (this would integrate with actual fetcher)
            # For now, just simulate the benchmark
            time.sleep(1)  # Simulate fetch time
            
            # Record end time and memory
            end_time = time.time()
            end_memory = psutil.virtual_memory().used / (1024 * 1024)
            
            # Calculate metrics
            fetch_time = end_time - start_time
            memory_usage = end_memory - start_memory
            records_per_second = records_count / fetch_time
            throughput_mb_per_second = (memory_usage / 1024) / fetch_time  # Rough estimate
            
            # Save benchmark
            benchmark_id = f"benchmark_{symbol}_{interval}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO performance_benchmarks 
                    (benchmark_name, symbol, interval, records_count, fetch_time_seconds,
                     memory_usage_mb, records_per_second, throughput_mb_per_second, compression_ratio)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    benchmark_id,
                    symbol,
                    interval,
                    records_count,
                    fetch_time,
                    memory_usage,
                    records_per_second,
                    throughput_mb_per_second,
                    1.0  # Default compression ratio
                ))
                
                conn.commit()
            
            return {
                "benchmark_id": benchmark_id,
                "symbol": symbol,
                "interval": interval,
                "records_count": records_count,
                "fetch_time_seconds": fetch_time,
                "memory_usage_mb": memory_usage,
                "records_per_second": records_per_second,
                "throughput_mb_per_second": throughput_mb_per_second,
                "performance_level": self._get_performance_level(records_per_second, memory_usage)
            }
            
        except Exception as e:
            self.logger.error(f"Error benchmarking performance: {e}")
            return {"error": str(e)}
    
    def _get_performance_level(self, records_per_second: float, memory_usage_mb: float) -> str:
        """Get performance level based on metrics."""
        if (records_per_second >= self.thresholds['excellent_records_per_second'] and 
            memory_usage_mb <= self.thresholds['excellent_memory_mb']):
            return "excellent"
        elif (records_per_second >= self.thresholds['good_records_per_second'] and 
              memory_usage_mb <= self.thresholds['good_memory_mb']):
            return "good"
        elif (records_per_second >= self.thresholds['fair_records_per_second'] and 
              memory_usage_mb <= self.thresholds['fair_memory_mb']):
            return "fair"
        elif (records_per_second >= self.thresholds['poor_records_per_second'] and 
              memory_usage_mb <= self.thresholds['poor_memory_mb']):
            return "poor"
        else:
            return "critical"
    
    def get_performance_dashboard(self, days: int = 7) -> Dict[str, Any]:
        """Get comprehensive performance dashboard."""
        try:
            # Get performance analysis
            analysis = self.analyze_performance(days=days)
            
            # Get system resources
            system_resources = {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent,
                "available_memory_gb": psutil.virtual_memory().available / (1024**3)
            }
            
            # Get recent benchmarks
            recent_benchmarks = self._get_recent_benchmarks(days)
            
            return {
                "timestamp": datetime.now().isoformat(),
                "analysis_period_days": days,
                "performance_analysis": analysis,
                "system_resources": system_resources,
                "recent_benchmarks": recent_benchmarks,
                "recommendations_summary": self._get_recommendations_summary(analysis.get('recommendations', []))
            }
            
        except Exception as e:
            self.logger.error(f"Error getting performance dashboard: {e}")
            return {"error": str(e)}
    
    def _get_recent_benchmarks(self, days: int) -> List[Dict]:
        """Get recent benchmark data."""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    SELECT * FROM performance_benchmarks 
                    WHERE created_at >= datetime('now', '-{} days')
                    ORDER BY created_at DESC
                    LIMIT 10
                """.format(days))
                
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                
                return [dict(zip(columns, row)) for row in rows]
                
        except Exception as e:
            self.logger.error(f"Error getting recent benchmarks: {e}")
            return []
    
    def _get_recommendations_summary(self, recommendations: List[Dict]) -> Dict[str, Any]:
        """Get summary of optimization recommendations."""
        if not recommendations:
            return {"total": 0, "by_priority": {}, "by_category": {}}
        
        by_priority = {}
        by_category = {}
        
        for rec in recommendations:
            priority = rec.get('priority', 'unknown')
            category = rec.get('category', 'unknown')
            
            by_priority[priority] = by_priority.get(priority, 0) + 1
            by_category[category] = by_category.get(category, 0) + 1
        
        return {
            "total": len(recommendations),
            "by_priority": by_priority,
            "by_category": by_category
        }

def main():
    """Main function for standalone performance monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance Monitor")
    parser.add_argument("--session-id", help="Session ID to analyze")
    parser.add_argument("--days", type=int, default=7, help="Days to analyze")
    parser.add_argument("--benchmark", help="Run benchmark for symbol:interval")
    parser.add_argument("--dashboard", action="store_true", help="Show performance dashboard")
    parser.add_argument("--db-file", default="data/sqlite/runs.db", help="Database file path")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create performance monitor
    monitor = PerformanceMonitor(db_file=args.db_file, log_level=args.log_level)
    
    if args.benchmark:
        # Run benchmark
        symbol, interval = args.benchmark.split(':')
        result = monitor.benchmark_performance(symbol, interval)
        print(f"Benchmark result: {result}")
    
    elif args.dashboard:
        # Show dashboard
        dashboard = monitor.get_performance_dashboard(args.days)
        print(f"Performance Dashboard:")
        print(f"  Analysis: {dashboard.get('performance_analysis', {}).get('summary', 'N/A')}")
        print(f"  System Resources: {dashboard.get('system_resources', {})}")
        print(f"  Recommendations: {dashboard.get('recommendations_summary', {})}")
    
    else:
        # Analyze performance
        analysis = monitor.analyze_performance(session_id=args.session_id, days=args.days)
        print(f"Performance Analysis:")
        print(f"  Score: {analysis.get('performance_score', 0):.1f}/100")
        print(f"  Summary: {analysis.get('summary', 'N/A')}")
        print(f"  Recommendations: {len(analysis.get('recommendations', []))}")

if __name__ == "__main__":
    main()
