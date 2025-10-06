#!/usr/bin/env python3
"""
Data Quality Monitor Service for Crypto Trading Bot

Provides comprehensive data quality monitoring, validation, and reporting
for OHLCV, funding, mark price, and index price data.
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
from dataclasses import dataclass
from enum import Enum

# Add src directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

class QualityLevel(Enum):
    """Data quality levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"

@dataclass
class QualityMetric:
    """Data quality metric result"""
    name: str
    value: float
    threshold: float
    level: QualityLevel
    message: str
    details: Optional[Dict[str, Any]] = None

@dataclass
class QualityReport:
    """Comprehensive data quality report"""
    symbol: str
    interval: str
    data_type: str
    timestamp: datetime
    overall_score: float
    overall_level: QualityLevel
    metrics: List[QualityMetric]
    recommendations: List[str]
    summary: str

class DataQualityMonitor:
    """
    Data quality monitoring service for crypto trading data.
    
    Monitors:
    - Data completeness and gaps
    - Price data validity (OHLC relationships)
    - Volume data consistency
    - Timestamp continuity
    - Data freshness
    - Statistical anomalies
    - Cross-symbol consistency
    """
    
    def __init__(self, 
                 parquet_folder: str = "data/parquet",
                 db_file: str = "data/sqlite/runs.db",
                 log_level: str = "INFO"):
        """
        Initialize the data quality monitor.
        
        Args:
            parquet_folder: Directory containing parquet files
            db_file: SQLite database file path
            log_level: Logging level
        """
        self.parquet_folder = Path(parquet_folder)
        self.db_file = Path(db_file)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Quality thresholds
        self.thresholds = {
            'completeness_min': 0.95,  # 95% data completeness
            'price_validity_min': 0.98,  # 98% valid OHLC relationships
            'volume_consistency_min': 0.90,  # 90% volume consistency
            'timestamp_continuity_min': 0.95,  # 95% timestamp continuity
            'freshness_hours': 2,  # Data should be less than 2 hours old
            'price_change_max': 0.5,  # Max 50% price change per interval
            'volume_spike_max': 10.0,  # Max 10x volume spike
            'gap_tolerance_minutes': 30  # Max 30-minute gaps
        }
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for quality tracking."""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                # Create quality monitoring tables
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS data_quality_reports (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        data_type TEXT NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        overall_score REAL,
                        overall_level TEXT,
                        report_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS data_quality_alerts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        data_type TEXT NOT NULL,
                        alert_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        message TEXT NOT NULL,
                        details TEXT,
                        timestamp TIMESTAMP NOT NULL,
                        resolved BOOLEAN DEFAULT FALSE,
                        resolved_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS data_gaps (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        data_type TEXT NOT NULL,
                        gap_start TIMESTAMP NOT NULL,
                        gap_end TIMESTAMP NOT NULL,
                        gap_duration_minutes INTEGER,
                        records_missing INTEGER,
                        severity TEXT NOT NULL,
                        detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        resolved_at TIMESTAMP,
                        resolved BOOLEAN DEFAULT FALSE
                    )
                """)
                
                conn.commit()
                self.logger.info("Data quality monitoring database initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize quality monitoring database: {e}")
            raise
    
    def validate_ohlcv_data(self, df: pd.DataFrame, symbol: str, interval: str) -> List[QualityMetric]:
        """Validate OHLCV data quality."""
        metrics = []
        
        if df.empty:
            metrics.append(QualityMetric(
                name="data_completeness",
                value=0.0,
                threshold=self.thresholds['completeness_min'],
                level=QualityLevel.FAILED,
                message="No data available"
            ))
            return metrics
        
        # 1. Data Completeness
        completeness = self._check_data_completeness(df, interval)
        metrics.append(completeness)
        
        # 2. Price Validity (OHLC relationships)
        price_validity = self._check_price_validity(df)
        metrics.append(price_validity)
        
        # 3. Volume Consistency
        volume_consistency = self._check_volume_consistency(df)
        metrics.append(volume_consistency)
        
        # 4. Timestamp Continuity
        timestamp_continuity = self._check_timestamp_continuity(df, interval)
        metrics.append(timestamp_continuity)
        
        # 5. Data Freshness
        data_freshness = self._check_data_freshness(df)
        metrics.append(data_freshness)
        
        # 6. Price Anomalies
        price_anomalies = self._check_price_anomalies(df)
        metrics.append(price_anomalies)
        
        # 7. Volume Anomalies
        volume_anomalies = self._check_volume_anomalies(df)
        metrics.append(volume_anomalies)
        
        return metrics
    
    def validate_funding_data(self, df: pd.DataFrame, symbol: str) -> List[QualityMetric]:
        """Validate funding rate data quality."""
        metrics = []
        
        if df.empty:
            metrics.append(QualityMetric(
                name="data_completeness",
                value=0.0,
                threshold=self.thresholds['completeness_min'],
                level=QualityLevel.FAILED,
                message="No funding data available"
            ))
            return metrics
        
        # 1. Data Completeness (8-hour intervals)
        expected_intervals = 3  # 3 funding periods per day
        actual_intervals = len(df)
        completeness = actual_intervals / expected_intervals if expected_intervals > 0 else 0
        
        metrics.append(QualityMetric(
            name="funding_completeness",
            value=completeness,
            threshold=0.8,  # 80% of expected funding periods
            level=self._get_quality_level(completeness, 0.8),
            message=f"Funding data completeness: {completeness:.2%}",
            details={"expected": expected_intervals, "actual": actual_intervals}
        ))
        
        # 2. Funding Rate Validity
        if 'fundingRate' in df.columns:
            valid_rates = df['fundingRate'].notna()
            rate_validity = valid_rates.sum() / len(df)
            
            metrics.append(QualityMetric(
                name="funding_rate_validity",
                value=rate_validity,
                threshold=0.95,
                level=self._get_quality_level(rate_validity, 0.95),
                message=f"Funding rate validity: {rate_validity:.2%}"
            ))
        
        # 3. Timestamp Validity
        if 'fundingTime' in df.columns:
            time_validity = df['fundingTime'].notna().sum() / len(df)
            
            metrics.append(QualityMetric(
                name="funding_timestamp_validity",
                value=time_validity,
                threshold=0.95,
                level=self._get_quality_level(time_validity, 0.95),
                message=f"Funding timestamp validity: {time_validity:.2%}"
            ))
        
        return metrics
    
    def _check_data_completeness(self, df: pd.DataFrame, interval: str) -> QualityMetric:
        """Check data completeness based on expected intervals."""
        if df.empty:
            return QualityMetric(
                name="data_completeness",
                value=0.0,
                threshold=self.thresholds['completeness_min'],
                level=QualityLevel.FAILED,
                message="No data available"
            )
        
        # Calculate expected records based on time range and interval
        time_range = df['open_time'].max() - df['open_time'].min()
        
        if interval == '1m':
            expected_records = int(time_range.total_seconds() / 60) + 1
        elif interval == '5m':
            expected_records = int(time_range.total_seconds() / 300) + 1
        elif interval == '15m':
            expected_records = int(time_range.total_seconds() / 900) + 1
        elif interval == '1h':
            expected_records = int(time_range.total_seconds() / 3600) + 1
        else:
            expected_records = len(df)  # Fallback
        
        actual_records = len(df)
        completeness = actual_records / expected_records if expected_records > 0 else 1.0
        
        return QualityMetric(
            name="data_completeness",
            value=completeness,
            threshold=self.thresholds['completeness_min'],
            level=self._get_quality_level(completeness, self.thresholds['completeness_min']),
            message=f"Data completeness: {completeness:.2%} ({actual_records}/{expected_records})",
            details={"expected": expected_records, "actual": actual_records}
        )
    
    def _check_price_validity(self, df: pd.DataFrame) -> QualityMetric:
        """Check OHLC price relationships validity."""
        if df.empty or not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return QualityMetric(
                name="price_validity",
                value=0.0,
                threshold=self.thresholds['price_validity_min'],
                level=QualityLevel.FAILED,
                message="Missing OHLC columns"
            )
        
        # Check OHLC relationships: high >= max(open, close), low <= min(open, close)
        valid_high = (df['high'] >= df[['open', 'close']].max(axis=1))
        valid_low = (df['low'] <= df[['open', 'close']].min(axis=1))
        valid_ohlc = valid_high & valid_low
        
        validity = valid_ohlc.sum() / len(df)
        
        return QualityMetric(
            name="price_validity",
            value=validity,
            threshold=self.thresholds['price_validity_min'],
            level=self._get_quality_level(validity, self.thresholds['price_validity_min']),
            message=f"Price validity: {validity:.2%}",
            details={"invalid_records": (~valid_ohlc).sum()}
        )
    
    def _check_volume_consistency(self, df: pd.DataFrame) -> QualityMetric:
        """Check volume data consistency."""
        if df.empty or 'volume' not in df.columns:
            return QualityMetric(
                name="volume_consistency",
                value=0.0,
                threshold=self.thresholds['volume_consistency_min'],
                level=QualityLevel.FAILED,
                message="Missing volume data"
            )
        
        # Check for negative volumes
        negative_volume = (df['volume'] < 0).sum()
        zero_volume = (df['volume'] == 0).sum()
        
        # Volume consistency score
        total_issues = negative_volume + zero_volume
        consistency = 1.0 - (total_issues / len(df))
        
        return QualityMetric(
            name="volume_consistency",
            value=consistency,
            threshold=self.thresholds['volume_consistency_min'],
            level=self._get_quality_level(consistency, self.thresholds['volume_consistency_min']),
            message=f"Volume consistency: {consistency:.2%}",
            details={"negative_volume": negative_volume, "zero_volume": zero_volume}
        )
    
    def _check_timestamp_continuity(self, df: pd.DataFrame, interval: str) -> QualityMetric:
        """Check timestamp continuity and gaps."""
        if df.empty or 'open_time' not in df.columns:
            return QualityMetric(
                name="timestamp_continuity",
                value=0.0,
                threshold=self.thresholds['timestamp_continuity_min'],
                level=QualityLevel.FAILED,
                message="Missing timestamp data"
            )
        
        # Sort by timestamp
        df_sorted = df.sort_values('open_time')
        
        # Calculate expected intervals
        if interval == '1m':
            expected_interval = pd.Timedelta(minutes=1)
        elif interval == '5m':
            expected_interval = pd.Timedelta(minutes=5)
        elif interval == '15m':
            expected_interval = pd.Timedelta(minutes=15)
        elif interval == '1h':
            expected_interval = pd.Timedelta(hours=1)
        else:
            expected_interval = pd.Timedelta(minutes=1)  # Default
        
        # Find gaps
        time_diffs = df_sorted['open_time'].diff().dropna()
        gaps = time_diffs > expected_interval * 1.5  # Allow 50% tolerance
        
        continuity = 1.0 - (gaps.sum() / len(time_diffs)) if len(time_diffs) > 0 else 1.0
        
        return QualityMetric(
            name="timestamp_continuity",
            value=continuity,
            threshold=self.thresholds['timestamp_continuity_min'],
            level=self._get_quality_level(continuity, self.thresholds['timestamp_continuity_min']),
            message=f"Timestamp continuity: {continuity:.2%}",
            details={"gaps": gaps.sum(), "total_intervals": len(time_diffs)}
        )
    
    def _check_data_freshness(self, df: pd.DataFrame) -> QualityMetric:
        """Check data freshness (how recent the data is)."""
        if df.empty or 'open_time' not in df.columns:
            return QualityMetric(
                name="data_freshness",
                value=0.0,
                threshold=1.0,
                level=QualityLevel.FAILED,
                message="Missing timestamp data"
            )
        
        # Get the most recent timestamp
        latest_time = df['open_time'].max()
        now = datetime.now()
        
        # Calculate hours since last update
        hours_old = (now - latest_time).total_seconds() / 3600
        
        # Freshness score (1.0 = fresh, 0.0 = very old)
        freshness = max(0.0, 1.0 - (hours_old / self.thresholds['freshness_hours']))
        
        level = QualityLevel.EXCELLENT if hours_old < 1 else \
                QualityLevel.GOOD if hours_old < 2 else \
                QualityLevel.WARNING if hours_old < 6 else \
                QualityLevel.CRITICAL
        
        return QualityMetric(
            name="data_freshness",
            value=freshness,
            threshold=0.5,  # At least 50% freshness
            level=level,
            message=f"Data freshness: {hours_old:.1f} hours old",
            details={"hours_old": hours_old, "latest_time": latest_time}
        )
    
    def _check_price_anomalies(self, df: pd.DataFrame) -> QualityMetric:
        """Check for price anomalies and extreme changes."""
        if df.empty or 'close' not in df.columns:
            return QualityMetric(
                name="price_anomalies",
                value=1.0,
                threshold=0.95,
                level=QualityLevel.EXCELLENT,
                message="No price data to analyze"
            )
        
        # Calculate price changes
        price_changes = df['close'].pct_change().abs()
        
        # Find extreme changes
        extreme_changes = price_changes > self.thresholds['price_change_max']
        anomaly_rate = extreme_changes.sum() / len(price_changes.dropna())
        
        # Anomaly score (1.0 = no anomalies, 0.0 = many anomalies)
        anomaly_score = 1.0 - anomaly_rate
        
        return QualityMetric(
            name="price_anomalies",
            value=anomaly_score,
            threshold=0.95,
            level=self._get_quality_level(anomaly_score, 0.95),
            message=f"Price anomaly rate: {anomaly_rate:.2%}",
            details={"extreme_changes": extreme_changes.sum(), "max_change": price_changes.max()}
        )
    
    def _check_volume_anomalies(self, df: pd.DataFrame) -> QualityMetric:
        """Check for volume anomalies and spikes."""
        if df.empty or 'volume' not in df.columns:
            return QualityMetric(
                name="volume_anomalies",
                value=1.0,
                threshold=0.90,
                level=QualityLevel.EXCELLENT,
                message="No volume data to analyze"
            )
        
        # Calculate volume spikes
        volume_rolling = df['volume'].rolling(window=20, min_periods=5).mean()
        volume_spikes = df['volume'] > (volume_rolling * self.thresholds['volume_spike_max'])
        
        spike_rate = volume_spikes.sum() / len(df)
        
        # Anomaly score (1.0 = no spikes, 0.0 = many spikes)
        anomaly_score = 1.0 - spike_rate
        
        return QualityMetric(
            name="volume_anomalies",
            value=anomaly_score,
            threshold=0.90,
            level=self._get_quality_level(anomaly_score, 0.90),
            message=f"Volume spike rate: {spike_rate:.2%}",
            details={"volume_spikes": volume_spikes.sum(), "max_volume": df['volume'].max()}
        )
    
    def _get_quality_level(self, value: float, threshold: float) -> QualityLevel:
        """Determine quality level based on value and threshold."""
        if value >= threshold:
            return QualityLevel.EXCELLENT
        elif value >= threshold * 0.9:
            return QualityLevel.GOOD
        elif value >= threshold * 0.8:
            return QualityLevel.WARNING
        elif value >= threshold * 0.5:
            return QualityLevel.CRITICAL
        else:
            return QualityLevel.FAILED
    
    def generate_quality_report(self, symbol: str, interval: str, data_type: str = "ohlcv") -> QualityReport:
        """Generate comprehensive quality report for a symbol/interval combination."""
        try:
            # Load data
            df = self._load_data(symbol, interval, data_type)
            
            # Validate data
            if data_type == "ohlcv":
                metrics = self.validate_ohlcv_data(df, symbol, interval)
            elif data_type == "funding":
                metrics = self.validate_funding_data(df, symbol)
            else:
                metrics = []
            
            # Calculate overall score
            if metrics:
                overall_score = np.mean([m.value for m in metrics])
                overall_level = self._get_quality_level(overall_score, 0.8)
            else:
                overall_score = 0.0
                overall_level = QualityLevel.FAILED
            
            # Generate recommendations
            recommendations = self._generate_recommendations(metrics)
            
            # Create summary
            summary = self._generate_summary(metrics, overall_level)
            
            # Create report
            report = QualityReport(
                symbol=symbol,
                interval=interval,
                data_type=data_type,
                timestamp=datetime.now(),
                overall_score=overall_score,
                overall_level=overall_level,
                metrics=metrics,
                recommendations=recommendations,
                summary=summary
            )
            
            # Save report to database
            self._save_quality_report(report)
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating quality report for {symbol} {interval}: {e}")
            raise
    
    def _load_data(self, symbol: str, interval: str, data_type: str) -> pd.DataFrame:
        """Load data from parquet files."""
        try:
            if data_type == "ohlcv":
                base_path = self.parquet_folder / "exchange=bybit" / "market=linear_perp" / f"symbol={symbol}" / f"timeframe={interval}"
            elif data_type == "funding":
                base_path = self.parquet_folder / "exchange=bybit" / "market=linear_perp" / f"symbol={symbol}" / "funding"
            else:
                raise ValueError(f"Unknown data type: {data_type}")
            
            if not base_path.exists():
                return pd.DataFrame()
            
            # Load all parquet files
            all_data = []
            for date_dir in base_path.iterdir():
                if date_dir.is_dir() and date_dir.name.startswith('date='):
                    if data_type == "ohlcv":
                        file_path = date_dir / "bars.parquet"
                    else:
                        file_path = date_dir / "funding.parquet"
                    
                    if file_path.exists():
                        df = pd.read_parquet(file_path)
                        all_data.append(df)
            
            if all_data:
                combined_df = pd.concat(all_data, ignore_index=True)
                if 'open_time' in combined_df.columns:
                    combined_df['open_time'] = pd.to_datetime(combined_df['open_time'])
                    combined_df = combined_df.sort_values('open_time')
                elif 'fundingTime' in combined_df.columns:
                    combined_df['fundingTime'] = pd.to_datetime(combined_df['fundingTime'])
                    combined_df = combined_df.sort_values('fundingTime')
                
                return combined_df
            else:
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error loading data for {symbol} {interval} {data_type}: {e}")
            return pd.DataFrame()
    
    def _generate_recommendations(self, metrics: List[QualityMetric]) -> List[str]:
        """Generate recommendations based on quality metrics."""
        recommendations = []
        
        for metric in metrics:
            if metric.level in [QualityLevel.CRITICAL, QualityLevel.FAILED]:
                if metric.name == "data_completeness":
                    recommendations.append("Consider backfilling missing data or checking data source connectivity")
                elif metric.name == "price_validity":
                    recommendations.append("Review data source for price data corruption or API issues")
                elif metric.name == "volume_consistency":
                    recommendations.append("Investigate volume data source and validate API responses")
                elif metric.name == "timestamp_continuity":
                    recommendations.append("Check for data gaps and implement gap detection alerts")
                elif metric.name == "data_freshness":
                    recommendations.append("Increase data fetch frequency or check for API connectivity issues")
                elif metric.name == "price_anomalies":
                    recommendations.append("Implement price anomaly detection and filtering")
                elif metric.name == "volume_anomalies":
                    recommendations.append("Add volume spike detection and filtering")
        
        if not recommendations:
            recommendations.append("Data quality is excellent - no immediate actions required")
        
        return recommendations
    
    def _generate_summary(self, metrics: List[QualityMetric], overall_level: QualityLevel) -> str:
        """Generate summary text for the quality report."""
        if not metrics:
            return "No data available for quality assessment"
        
        critical_issues = sum(1 for m in metrics if m.level in [QualityLevel.CRITICAL, QualityLevel.FAILED])
        warning_issues = sum(1 for m in metrics if m.level == QualityLevel.WARNING)
        
        if critical_issues > 0:
            return f"CRITICAL: {critical_issues} critical quality issues detected. Immediate attention required."
        elif warning_issues > 0:
            return f"WARNING: {warning_issues} quality warnings detected. Monitor closely."
        elif overall_level == QualityLevel.EXCELLENT:
            return "EXCELLENT: All quality metrics are within acceptable ranges."
        else:
            return f"GOOD: Data quality is acceptable with {overall_level.value} overall rating."
    
    def _save_quality_report(self, report: QualityReport):
        """Save quality report to database."""
        try:
            import json
            
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                # Convert metrics to JSON
                metrics_json = json.dumps([{
                    'name': m.name,
                    'value': m.value,
                    'threshold': m.threshold,
                    'level': m.level.value,
                    'message': m.message,
                    'details': m.details
                } for m in report.metrics])
                
                cursor.execute("""
                    INSERT INTO data_quality_reports 
                    (symbol, interval, data_type, timestamp, overall_score, overall_level, report_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    report.symbol,
                    report.interval,
                    report.data_type,
                    report.timestamp,
                    report.overall_score,
                    report.overall_level.value,
                    metrics_json
                ))
                
                conn.commit()
                self.logger.info(f"Quality report saved for {report.symbol} {report.interval}")
                
        except Exception as e:
            self.logger.error(f"Error saving quality report: {e}")
    
    def monitor_all_data_quality(self, symbols: List[str], intervals: List[str]) -> Dict[str, Any]:
        """Monitor data quality for all symbols and intervals."""
        results = {
            'overall_score': 0.0,
            'reports': [],
            'alerts': [],
            'recommendations': []
        }
        
        all_scores = []
        
        for symbol in symbols:
            for interval in intervals:
                try:
                    # Check OHLCV data
                    ohlcv_report = self.generate_quality_report(symbol, interval, "ohlcv")
                    results['reports'].append(ohlcv_report)
                    all_scores.append(ohlcv_report.overall_score)
                    
                    # Check funding data
                    funding_report = self.generate_quality_report(symbol, interval, "funding")
                    results['reports'].append(funding_report)
                    all_scores.append(funding_report.overall_score)
                    
                    # Generate alerts for critical issues
                    for report in [ohlcv_report, funding_report]:
                        if report.overall_level in [QualityLevel.CRITICAL, QualityLevel.FAILED]:
                            alert = {
                                'symbol': symbol,
                                'interval': interval,
                                'data_type': report.data_type,
                                'severity': report.overall_level.value,
                                'message': report.summary,
                                'timestamp': report.timestamp
                            }
                            results['alerts'].append(alert)
                    
                except Exception as e:
                    self.logger.error(f"Error monitoring {symbol} {interval}: {e}")
                    results['alerts'].append({
                        'symbol': symbol,
                        'interval': interval,
                        'severity': 'critical',
                        'message': f"Failed to monitor data quality: {e}",
                        'timestamp': datetime.now()
                    })
        
        # Calculate overall score
        if all_scores:
            results['overall_score'] = np.mean(all_scores)
        
        # Generate overall recommendations
        all_recommendations = []
        for report in results['reports']:
            all_recommendations.extend(report.recommendations)
        results['recommendations'] = list(set(all_recommendations))  # Remove duplicates
        
        return results

def main():
    """Main function for standalone data quality monitoring."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Quality Monitor")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT", "ETHUSDT"], help="Symbols to monitor")
    parser.add_argument("--intervals", nargs="+", default=["1m", "5m", "15m", "1h"], help="Intervals to monitor")
    parser.add_argument("--parquet-folder", default="data/parquet", help="Parquet folder path")
    parser.add_argument("--db-file", default="data/sqlite/runs.db", help="Database file path")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create monitor
    monitor = DataQualityMonitor(
        parquet_folder=args.parquet_folder,
        db_file=args.db_file,
        log_level=args.log_level
    )
    
    # Run monitoring
    results = monitor.monitor_all_data_quality(args.symbols, args.intervals)
    
    # Print results
    print(f"\n=== Data Quality Monitoring Results ===")
    print(f"Overall Score: {results['overall_score']:.2%}")
    print(f"Reports Generated: {len(results['reports'])}")
    print(f"Alerts: {len(results['alerts'])}")
    print(f"Recommendations: {len(results['recommendations'])}")
    
    if results['alerts']:
        print(f"\n=== ALERTS ===")
        for alert in results['alerts']:
            print(f"[{alert['severity'].upper()}] {alert['symbol']} {alert['interval']} {alert['data_type']}: {alert['message']}")
    
    if results['recommendations']:
        print(f"\n=== RECOMMENDATIONS ===")
        for i, rec in enumerate(results['recommendations'], 1):
            print(f"{i}. {rec}")

if __name__ == "__main__":
    main()
