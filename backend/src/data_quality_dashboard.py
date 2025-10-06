#!/usr/bin/env python3
"""
Data Quality Dashboard for Crypto Trading Bot

Provides a comprehensive dashboard for monitoring data quality across all symbols and intervals.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import sqlite3
import json

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.data_quality_monitor import DataQualityMonitor, QualityLevel
from services.data_validator import DataValidator

class DataQualityDashboard:
    """
    Comprehensive data quality dashboard for monitoring and reporting.
    
    Features:
    - Real-time quality monitoring
    - Historical quality trends
    - Alert management
    - Quality reports
    - Data gap analysis
    """
    
    def __init__(self, 
                 parquet_folder: str = "data/parquet",
                 db_file: str = "data/sqlite/runs.db",
                 log_level: str = "INFO"):
        """
        Initialize the data quality dashboard.
        
        Args:
            parquet_folder: Directory containing parquet files
            db_file: SQLite database file path
            log_level: Logging level
        """
        self.parquet_folder = Path(parquet_folder)
        self.db_file = Path(db_file)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Initialize services
        self.quality_monitor = DataQualityMonitor(
            parquet_folder=str(self.parquet_folder),
            db_file=str(self.db_file),
            log_level=log_level
        )
        
        self.validator = DataValidator(
            db_file=str(self.db_file),
            log_level=log_level
        )
    
    def generate_comprehensive_report(self, symbols: List[str], intervals: List[str]) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        print("🔍 Generating comprehensive data quality report...")
        
        # Monitor all data quality
        quality_results = self.quality_monitor.monitor_all_data_quality(symbols, intervals)
        
        # Get validation summaries
        validation_summaries = {}
        for symbol in symbols:
            for interval in intervals:
                for data_type in ["ohlcv", "funding"]:
                    key = f"{symbol}_{interval}_{data_type}"
                    validation_summaries[key] = self.validator.get_validation_summary(
                        symbol, interval, data_type, days=7
                    )
        
        # Analyze data gaps
        gap_analysis = self._analyze_data_gaps(symbols, intervals)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(quality_results, validation_summaries, gap_analysis)
        
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'overall_quality_score': quality_results['overall_score'],
            'quality_reports': quality_results['reports'],
            'alerts': quality_results['alerts'],
            'validation_summaries': validation_summaries,
            'gap_analysis': gap_analysis,
            'recommendations': recommendations,
            'summary': self._generate_summary(quality_results, validation_summaries, gap_analysis)
        }
        
        return report
    
    def _analyze_data_gaps(self, symbols: List[str], intervals: List[str]) -> Dict[str, Any]:
        """Analyze data gaps across symbols and intervals."""
        gap_analysis = {
            'total_gaps': 0,
            'critical_gaps': 0,
            'gap_details': [],
            'symbol_gaps': {},
            'interval_gaps': {}
        }
        
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                # Get gap information from database
                cursor.execute("""
                    SELECT symbol, interval, data_type, gap_start, gap_end, gap_duration_minutes, severity
                    FROM data_gaps 
                    WHERE resolved = FALSE
                    ORDER BY gap_start DESC
                """)
                
                gaps = cursor.fetchall()
                
                for gap in gaps:
                    symbol, interval, data_type, gap_start, gap_end, duration, severity = gap
                    
                    gap_info = {
                        'symbol': symbol,
                        'interval': interval,
                        'data_type': data_type,
                        'gap_start': gap_start,
                        'gap_end': gap_end,
                        'duration_minutes': duration,
                        'severity': severity
                    }
                    
                    gap_analysis['gap_details'].append(gap_info)
                    gap_analysis['total_gaps'] += 1
                    
                    if severity == 'critical':
                        gap_analysis['critical_gaps'] += 1
                    
                    # Track by symbol
                    if symbol not in gap_analysis['symbol_gaps']:
                        gap_analysis['symbol_gaps'][symbol] = 0
                    gap_analysis['symbol_gaps'][symbol] += 1
                    
                    # Track by interval
                    if interval not in gap_analysis['interval_gaps']:
                        gap_analysis['interval_gaps'][interval] = 0
                    gap_analysis['interval_gaps'][interval] += 1
                
        except Exception as e:
            self.logger.error(f"Error analyzing data gaps: {e}")
        
        return gap_analysis
    
    def _generate_recommendations(self, quality_results: Dict, validation_summaries: Dict, gap_analysis: Dict) -> List[str]:
        """Generate actionable recommendations based on analysis."""
        recommendations = []
        
        # Quality-based recommendations
        if quality_results['overall_score'] < 0.8:
            recommendations.append("🚨 CRITICAL: Overall data quality is below 80%. Immediate attention required.")
        
        if quality_results['alerts']:
            critical_alerts = [alert for alert in quality_results['alerts'] if alert['severity'] == 'critical']
            if critical_alerts:
                recommendations.append(f"🚨 {len(critical_alerts)} critical quality alerts detected. Review immediately.")
        
        # Validation-based recommendations
        low_validation_scores = [
            key for key, summary in validation_summaries.items()
            if summary.get('average_score', 1.0) < 0.9
        ]
        
        if low_validation_scores:
            recommendations.append(f"⚠️ {len(low_validation_scores)} data sources have validation scores below 90%.")
        
        # Gap-based recommendations
        if gap_analysis['critical_gaps'] > 0:
            recommendations.append(f"🚨 {gap_analysis['critical_gaps']} critical data gaps detected. Implement gap filling strategy.")
        
        if gap_analysis['total_gaps'] > 10:
            recommendations.append(f"⚠️ {gap_analysis['total_gaps']} total data gaps detected. Consider improving data source reliability.")
        
        # Symbol-specific recommendations
        for symbol, gap_count in gap_analysis['symbol_gaps'].items():
            if gap_count > 5:
                recommendations.append(f"📊 {symbol}: {gap_count} data gaps detected. Review data source for this symbol.")
        
        # Interval-specific recommendations
        for interval, gap_count in gap_analysis['interval_gaps'].items():
            if gap_count > 3:
                recommendations.append(f"⏰ {interval} interval: {gap_count} data gaps detected. Review interval-specific data source.")
        
        # General recommendations
        if not recommendations:
            recommendations.append("✅ Data quality is excellent. Continue current monitoring practices.")
        else:
            recommendations.append("📈 Consider implementing automated data quality monitoring and alerting.")
            recommendations.append("🔧 Review data source configurations and API rate limits.")
            recommendations.append("📊 Implement data quality dashboards for real-time monitoring.")
        
        return recommendations
    
    def _generate_summary(self, quality_results: Dict, validation_summaries: Dict, gap_analysis: Dict) -> str:
        """Generate executive summary of data quality status."""
        total_reports = len(quality_results['reports'])
        total_alerts = len(quality_results['alerts'])
        critical_alerts = len([a for a in quality_results['alerts'] if a['severity'] == 'critical'])
        
        summary_parts = []
        
        # Overall quality
        if quality_results['overall_score'] >= 0.9:
            summary_parts.append("🟢 Overall data quality is EXCELLENT")
        elif quality_results['overall_score'] >= 0.8:
            summary_parts.append("🟡 Overall data quality is GOOD")
        elif quality_results['overall_score'] >= 0.6:
            summary_parts.append("🟠 Overall data quality is FAIR")
        else:
            summary_parts.append("🔴 Overall data quality is POOR")
        
        # Alerts summary
        if critical_alerts > 0:
            summary_parts.append(f"🚨 {critical_alerts} critical alerts require immediate attention")
        elif total_alerts > 0:
            summary_parts.append(f"⚠️ {total_alerts} quality alerts detected")
        else:
            summary_parts.append("✅ No quality alerts")
        
        # Gap summary
        if gap_analysis['critical_gaps'] > 0:
            summary_parts.append(f"🚨 {gap_analysis['critical_gaps']} critical data gaps")
        elif gap_analysis['total_gaps'] > 0:
            summary_parts.append(f"⚠️ {gap_analysis['total_gaps']} data gaps detected")
        else:
            summary_parts.append("✅ No data gaps detected")
        
        # Validation summary
        low_scores = [s for s in validation_summaries.values() if s.get('average_score', 1.0) < 0.9]
        if low_scores:
            summary_parts.append(f"📊 {len(low_scores)} data sources need validation improvements")
        else:
            summary_parts.append("✅ All data sources pass validation")
        
        return " | ".join(summary_parts)
    
    def print_dashboard(self, symbols: List[str], intervals: List[str]):
        """Print a formatted data quality dashboard."""
        print("\n" + "="*80)
        print("📊 CRYPTO TRADING BOT - DATA QUALITY DASHBOARD")
        print("="*80)
        print(f"📅 Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"🔍 Monitoring: {len(symbols)} symbols, {len(intervals)} intervals")
        print("="*80)
        
        # Generate comprehensive report
        report = self.generate_comprehensive_report(symbols, intervals)
        
        # Print summary
        print(f"\n📈 OVERALL QUALITY SCORE: {report['overall_quality_score']:.1%}")
        print(f"📋 SUMMARY: {report['summary']}")
        
        # Print alerts
        if report['alerts']:
            print(f"\n🚨 ALERTS ({len(report['alerts'])}):")
            for alert in report['alerts'][:5]:  # Show first 5 alerts
                print(f"  • [{alert['severity'].upper()}] {alert['symbol']} {alert['interval']} {alert['data_type']}: {alert['message']}")
            if len(report['alerts']) > 5:
                print(f"  ... and {len(report['alerts']) - 5} more alerts")
        
        # Print gap analysis
        if report['gap_analysis']['total_gaps'] > 0:
            print(f"\n📊 DATA GAPS ({report['gap_analysis']['total_gaps']}):")
            print(f"  • Critical gaps: {report['gap_analysis']['critical_gaps']}")
            print(f"  • By symbol: {report['gap_analysis']['symbol_gaps']}")
            print(f"  • By interval: {report['gap_analysis']['interval_gaps']}")
        
        # Print validation summaries
        print(f"\n🔍 VALIDATION SUMMARY:")
        for key, summary in report['validation_summaries'].items():
            if summary.get('total_validations', 0) > 0:
                score = summary.get('average_score', 0)
                status = "🟢" if score >= 0.9 else "🟡" if score >= 0.8 else "🔴"
                print(f"  {status} {key}: {score:.1%} (last: {summary.get('last_validation', 'N/A')})")
        
        # Print recommendations
        print(f"\n💡 RECOMMENDATIONS ({len(report['recommendations'])}):")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*80)
        print("📊 Dashboard complete. Check logs for detailed information.")
        print("="*80)
    
    def save_report_to_file(self, symbols: List[str], intervals: List[str], output_file: str = None):
        """Save comprehensive report to JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_file = f"data_quality_report_{timestamp}.json"
        
        report = self.generate_comprehensive_report(symbols, intervals)
        
        # Convert datetime objects to strings for JSON serialization
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2, default=json_serializer)
        
        print(f"📄 Report saved to: {output_file}")
        return output_file

def main():
    """Main function for standalone data quality dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Quality Dashboard")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"], help="Symbols to monitor")
    parser.add_argument("--intervals", nargs="+", default=["1m", "5m", "15m", "1h"], help="Intervals to monitor")
    parser.add_argument("--parquet-folder", default="data/parquet", help="Parquet folder path")
    parser.add_argument("--db-file", default="data/sqlite/runs.db", help="Database file path")
    parser.add_argument("--output-file", help="Output file for JSON report")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create dashboard
    dashboard = DataQualityDashboard(
        parquet_folder=args.parquet_folder,
        db_file=args.db_file,
        log_level=args.log_level
    )
    
    # Print dashboard
    dashboard.print_dashboard(args.symbols, args.intervals)
    
    # Save report if requested
    if args.output_file:
        dashboard.save_report_to_file(args.symbols, args.intervals, args.output_file)

if __name__ == "__main__":
    main()
