#!/usr/bin/env python3
"""
Migration Script: Data Ingestor to Smart Data Manager

This script helps migrate from the old Data Ingestor to the new Smart Data Manager
with automatic strategy selection and performance optimization.
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

# Add src directory to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.services.smart_data_manager import SmartDataManager
from src.services.data_ingestor import DataIngestor

class DataManagerMigration:
    """
    Migration helper for transitioning from Data Ingestor to Smart Data Manager.
    """
    
    def __init__(self, 
                 parquet_folder: str = "data/parquet",
                 db_file: str = "data/sqlite/runs.db",
                 log_level: str = "INFO"):
        """
        Initialize the migration helper.
        
        Args:
            parquet_folder: Parquet storage folder
            db_file: SQLite database file path
            log_level: Logging level
        """
        self.parquet_folder = parquet_folder
        self.db_file = db_file
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Initialize both systems for comparison
        self.data_ingestor = DataIngestor(
            parquet_folder=parquet_folder,
            db_file=db_file,
            log_level=log_level
        )
        
        self.smart_data_manager = SmartDataManager(
            parquet_folder=parquet_folder,
            db_file=db_file,
            log_level=log_level
        )
    
    def compare_performance(self, 
                           symbols: List[str], 
                           intervals: List[str],
                           test_days: int = 7) -> Dict[str, Any]:
        """
        Compare performance between Data Ingestor and Smart Data Manager.
        
        Args:
            symbols: List of symbols to test
            intervals: List of intervals to test
            test_days: Number of days to test
            
        Returns:
            Performance comparison results
        """
        self.logger.info(f"Comparing performance for {symbols} {intervals}")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=test_days)
        
        results = {
            "data_ingestor": {},
            "smart_data_manager": {},
            "comparison": {}
        }
        
        # Test Data Ingestor
        self.logger.info("Testing Data Ingestor...")
        try:
            start_time = datetime.now()
            
            # Test each symbol/interval combination
            ingestor_results = {}
            for symbol in symbols:
                ingestor_results[symbol] = {}
                for interval in intervals:
                    result = self.data_ingestor.fetch_and_save_data(
                        symbol=symbol,
                        interval=interval,
                        ohlcv_limit=1000,
                        funding_limit=500,
                        incremental=True
                    )
                    ingestor_results[symbol][interval] = result
            
            ingestor_time = (datetime.now() - start_time).total_seconds()
            
            # Calculate total records
            total_records = 0
            for symbol in symbols:
                for interval in intervals:
                    if symbol in ingestor_results and interval in ingestor_results[symbol]:
                        result = ingestor_results[symbol][interval]
                        if result.get("success"):
                            total_records += result.get("ohlcv_new_records", 0) + result.get("funding_new_records", 0)
            
            results["data_ingestor"] = {
                "success": True,
                "total_records": total_records,
                "fetch_time_seconds": ingestor_time,
                "records_per_second": total_records / max(ingestor_time, 1),
                "results": ingestor_results
            }
            
        except Exception as e:
            self.logger.error(f"Data Ingestor test failed: {e}")
            results["data_ingestor"] = {
                "success": False,
                "error": str(e),
                "total_records": 0,
                "fetch_time_seconds": 0,
                "records_per_second": 0
            }
        
        # Test Smart Data Manager
        self.logger.info("Testing Smart Data Manager...")
        try:
            start_time = datetime.now()
            
            result = self.smart_data_manager.fetch_data_smart(
                symbols=symbols,
                intervals=intervals,
                start_date=start_date,
                end_date=end_date,
                incremental=True
            )
            
            smart_time = (datetime.now() - start_time).total_seconds()
            
            results["smart_data_manager"] = {
                "success": result.get("success", False),
                "strategy_used": result.get("strategy", "unknown"),
                "total_records": result.get("total_records", 0),
                "fetch_time_seconds": smart_time,
                "records_per_second": result.get("total_records", 0) / max(smart_time, 1),
                "result": result
            }
            
        except Exception as e:
            self.logger.error(f"Smart Data Manager test failed: {e}")
            results["smart_data_manager"] = {
                "success": False,
                "error": str(e),
                "strategy_used": "failed",
                "total_records": 0,
                "fetch_time_seconds": 0,
                "records_per_second": 0
            }
        
        # Calculate comparison
        if results["data_ingestor"]["success"] and results["smart_data_manager"]["success"]:
            ingestor_speed = results["data_ingestor"]["records_per_second"]
            smart_speed = results["smart_data_manager"]["records_per_second"]
            
            speed_improvement = (smart_speed - ingestor_speed) / max(ingestor_speed, 1) * 100
            
            results["comparison"] = {
                "speed_improvement_percent": speed_improvement,
                "ingestor_speed": ingestor_speed,
                "smart_speed": smart_speed,
                "recommendation": "Smart Data Manager" if speed_improvement > 0 else "Data Ingestor"
            }
        
        return results
    
    def generate_migration_report(self, 
                                 symbols: List[str], 
                                 intervals: List[str]) -> Dict[str, Any]:
        """
        Generate a comprehensive migration report.
        
        Args:
            symbols: List of symbols to analyze
            intervals: List of intervals to analyze
            
        Returns:
            Migration report
        """
        self.logger.info("Generating migration report...")
        
        # Get system recommendations
        recommendations = self.smart_data_manager.get_system_recommendations()
        
        # Compare performance
        performance_comparison = self.compare_performance(symbols, intervals)
        
        # Generate report
        report = {
            "timestamp": datetime.now().isoformat(),
            "system_recommendations": recommendations,
            "performance_comparison": performance_comparison,
            "migration_recommendations": self._generate_migration_recommendations(performance_comparison, recommendations)
        }
        
        return report
    
    def _generate_migration_recommendations(self, 
                                           performance_comparison: Dict[str, Any],
                                           system_recommendations: Dict[str, Any]) -> List[Dict[str, str]]:
        """Generate migration recommendations based on analysis."""
        recommendations = []
        
        # Performance-based recommendations
        if performance_comparison.get("comparison", {}).get("speed_improvement_percent", 0) > 20:
            recommendations.append({
                "category": "performance",
                "priority": "high",
                "message": "Smart Data Manager shows significant performance improvement",
                "action": "Migrate to Smart Data Manager for better performance"
            })
        elif performance_comparison.get("comparison", {}).get("speed_improvement_percent", 0) < -20:
            recommendations.append({
                "category": "performance",
                "priority": "medium",
                "message": "Data Ingestor performs better for your use case",
                "action": "Consider keeping Data Ingestor for small datasets"
            })
        
        # System resource recommendations
        for rec in system_recommendations.get("recommendations", []):
            if rec["category"] == "memory" and rec["priority"] == "high":
                recommendations.append({
                    "category": "resources",
                    "priority": "high",
                    "message": "Low memory detected - Smart Data Manager will optimize usage",
                    "action": "Migrate to Smart Data Manager for better memory management"
                })
            elif rec["category"] == "performance" and rec["priority"] == "medium":
                recommendations.append({
                    "category": "resources",
                    "priority": "medium",
                    "message": "High CPU core count - Smart Data Manager can utilize parallel processing",
                    "action": "Migrate to Smart Data Manager for parallel processing benefits"
                })
        
        # General recommendations
        recommendations.append({
            "category": "general",
            "priority": "low",
            "message": "Smart Data Manager provides automatic optimization",
            "action": "Use Smart Data Manager for automatic strategy selection"
        })
        
        return recommendations
    
    def print_migration_report(self, report: Dict[str, Any]):
        """Print a formatted migration report."""
        print("\n" + "="*80)
        print("📊 DATA MANAGER MIGRATION REPORT")
        print("="*80)
        print(f"📅 Generated: {report['timestamp']}")
        
        # System Resources
        system_resources = report['system_recommendations']['system_resources']
        print(f"\n💻 System Resources:")
        print(f"   Memory: {system_resources['available_memory_gb']:.1f} GB available")
        print(f"   CPU Cores: {system_resources['cpu_cores']}")
        print(f"   Memory Usage: {system_resources['memory_percent']:.1f}%")
        
        # Performance Comparison
        comparison = report['performance_comparison']
        print(f"\n📈 Performance Comparison:")
        
        if comparison['data_ingestor']['success']:
            print(f"   Data Ingestor:")
            print(f"     Records: {comparison['data_ingestor']['total_records']:,}")
            print(f"     Time: {comparison['data_ingestor']['fetch_time_seconds']:.1f}s")
            print(f"     Speed: {comparison['data_ingestor']['records_per_second']:.0f} records/sec")
        
        if comparison['smart_data_manager']['success']:
            print(f"   Smart Data Manager:")
            print(f"     Strategy: {comparison['smart_data_manager']['strategy_used']}")
            print(f"     Records: {comparison['smart_data_manager']['total_records']:,}")
            print(f"     Time: {comparison['smart_data_manager']['fetch_time_seconds']:.1f}s")
            print(f"     Speed: {comparison['smart_data_manager']['records_per_second']:.0f} records/sec")
        
        # Comparison Results
        if comparison.get('comparison'):
            comp = comparison['comparison']
            print(f"\n⚡ Performance Improvement:")
            print(f"   Speed Improvement: {comp['speed_improvement_percent']:+.1f}%")
            print(f"   Recommendation: {comp['recommendation']}")
        
        # Migration Recommendations
        print(f"\n💡 Migration Recommendations:")
        for i, rec in enumerate(report['migration_recommendations'], 1):
            print(f"   {i}. [{rec['priority'].upper()}] {rec['message']}")
            print(f"      Action: {rec['action']}")
        
        print("\n" + "="*80)
        print("📋 Migration complete. Check logs for detailed information.")
        print("="*80)

def main():
    """Main function for migration script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Manager Migration")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"], help="Symbols to test")
    parser.add_argument("--intervals", nargs="+", default=["1m", "5m", "15m", "1h"], help="Intervals to test")
    parser.add_argument("--test-days", type=int, default=7, help="Days to test")
    parser.add_argument("--parquet-folder", default="data/parquet", help="Parquet folder path")
    parser.add_argument("--db-file", default="data/sqlite/runs.db", help="Database file path")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create migration helper
    migration = DataManagerMigration(
        parquet_folder=args.parquet_folder,
        db_file=args.db_file,
        log_level=args.log_level
    )
    
    # Generate migration report
    report = migration.generate_migration_report(args.symbols, args.intervals)
    
    # Print report
    migration.print_migration_report(report)
    
    # Save report to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = f"migration_report_{timestamp}.json"
    
    import json
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Report saved to: {report_file}")

if __name__ == "__main__":
    main()
