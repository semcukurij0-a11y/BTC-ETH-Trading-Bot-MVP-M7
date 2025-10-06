#!/usr/bin/env python3
"""
Data Validator Service for Crypto Trading Bot

Provides real-time data validation during the ingestion process
to ensure data quality and integrity.
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

class ValidationResult(Enum):
    """Validation result types"""
    VALID = "valid"
    WARNING = "warning"
    INVALID = "invalid"
    CRITICAL = "critical"

@dataclass
class ValidationRule:
    """Data validation rule definition"""
    name: str
    description: str
    severity: ValidationResult
    enabled: bool = True
    threshold: Optional[float] = None

@dataclass
class ValidationIssue:
    """Data validation issue"""
    rule_name: str
    severity: ValidationResult
    message: str
    affected_records: int
    details: Optional[Dict[str, Any]] = None

class DataValidator:
    """
    Real-time data validator for crypto trading data.
    
    Validates data during ingestion to ensure:
    - Data format compliance
    - Value ranges and relationships
    - Timestamp consistency
    - Data completeness
    - Anomaly detection
    """
    
    def __init__(self, 
                 db_file: str = "data/sqlite/runs.db",
                 log_level: str = "INFO"):
        """
        Initialize the data validator.
        
        Args:
            db_file: SQLite database file path
            log_level: Logging level
        """
        self.db_file = Path(db_file)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Define validation rules
        self.rules = self._define_validation_rules()
        
        # Initialize database
        self._init_database()
    
    def _define_validation_rules(self) -> Dict[str, ValidationRule]:
        """Define validation rules for different data types."""
        return {
            # OHLCV Rules
            'ohlcv_required_columns': ValidationRule(
                name="ohlcv_required_columns",
                description="OHLCV data must have required columns",
                severity=ValidationResult.CRITICAL
            ),
            'ohlcv_price_relationships': ValidationRule(
                name="ohlcv_price_relationships",
                description="OHLC price relationships must be valid",
                severity=ValidationResult.CRITICAL
            ),
            'ohlcv_positive_values': ValidationRule(
                name="ohlcv_positive_values",
                description="OHLC and volume values must be positive",
                severity=ValidationResult.CRITICAL
            ),
            'ohlcv_timestamp_continuity': ValidationRule(
                name="ohlcv_timestamp_continuity",
                description="Timestamps must be continuous and properly ordered",
                severity=ValidationResult.WARNING
            ),
            'ohlcv_price_anomalies': ValidationRule(
                name="ohlcv_price_anomalies",
                description="Price changes should be within reasonable limits",
                severity=ValidationResult.WARNING,
                threshold=0.5  # 50% max price change
            ),
            'ohlcv_volume_anomalies': ValidationRule(
                name="ohlcv_volume_anomalies",
                description="Volume spikes should be within reasonable limits",
                severity=ValidationResult.WARNING,
                threshold=10.0  # 10x volume spike
            ),
            
            # Funding Rules
            'funding_required_columns': ValidationRule(
                name="funding_required_columns",
                description="Funding data must have required columns",
                severity=ValidationResult.CRITICAL
            ),
            'funding_rate_range': ValidationRule(
                name="funding_rate_range",
                description="Funding rates must be within reasonable range",
                severity=ValidationResult.WARNING,
                threshold=0.01  # 1% max funding rate
            ),
            'funding_timestamp_validity': ValidationRule(
                name="funding_timestamp_validity",
                description="Funding timestamps must be valid",
                severity=ValidationResult.CRITICAL
            ),
            
            # General Rules
            'data_completeness': ValidationRule(
                name="data_completeness",
                description="Data must have minimum completeness threshold",
                severity=ValidationResult.WARNING,
                threshold=0.95  # 95% completeness
            ),
            'timestamp_freshness': ValidationRule(
                name="timestamp_freshness",
                description="Data must be reasonably fresh",
                severity=ValidationResult.WARNING,
                threshold=24.0  # 24 hours max age
            )
        }
    
    def _init_database(self):
        """Initialize SQLite database for validation tracking."""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                # Create validation tracking tables
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS validation_results (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        data_type TEXT NOT NULL,
                        validation_timestamp TIMESTAMP NOT NULL,
                        total_records INTEGER,
                        valid_records INTEGER,
                        warning_records INTEGER,
                        invalid_records INTEGER,
                        critical_records INTEGER,
                        validation_score REAL,
                        issues_json TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS validation_issues (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        symbol TEXT NOT NULL,
                        interval TEXT NOT NULL,
                        data_type TEXT NOT NULL,
                        rule_name TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        message TEXT NOT NULL,
                        affected_records INTEGER,
                        details TEXT,
                        timestamp TIMESTAMP NOT NULL,
                        resolved BOOLEAN DEFAULT FALSE,
                        resolved_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                conn.commit()
                self.logger.info("Data validation database initialized")
                
        except Exception as e:
            self.logger.error(f"Failed to initialize validation database: {e}")
            raise
    
    def validate_ohlcv_data(self, df: pd.DataFrame, symbol: str, interval: str) -> Dict[str, Any]:
        """Validate OHLCV data and return validation results."""
        validation_result = {
            'valid': True,
            'issues': [],
            'score': 1.0,
            'total_records': len(df),
            'valid_records': 0,
            'warning_records': 0,
            'invalid_records': 0,
            'critical_records': 0
        }
        
        if df.empty:
            validation_result['valid'] = False
            validation_result['issues'].append(ValidationIssue(
                rule_name="data_completeness",
                severity=ValidationResult.CRITICAL,
                message="No data provided for validation",
                affected_records=0
            ))
            return validation_result
        
        # Apply validation rules
        issues = []
        
        # 1. Required columns validation
        required_columns = ['open_time', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(ValidationIssue(
                rule_name="ohlcv_required_columns",
                severity=ValidationResult.CRITICAL,
                message=f"Missing required columns: {missing_columns}",
                affected_records=len(df)
            ))
        
        # 2. Price relationships validation
        if all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            invalid_ohlc = self._validate_ohlc_relationships(df)
            if invalid_ohlc > 0:
                issues.append(ValidationIssue(
                    rule_name="ohlcv_price_relationships",
                    severity=ValidationResult.CRITICAL,
                    message=f"Invalid OHLC relationships in {invalid_ohlc} records",
                    affected_records=invalid_ohlc
                ))
        
        # 3. Positive values validation
        if all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            negative_values = self._validate_positive_values(df)
            if negative_values > 0:
                issues.append(ValidationIssue(
                    rule_name="ohlcv_positive_values",
                    severity=ValidationResult.CRITICAL,
                    message=f"Negative values found in {negative_values} records",
                    affected_records=negative_values
                ))
        
        # 4. Timestamp continuity validation
        if 'open_time' in df.columns:
            timestamp_issues = self._validate_timestamp_continuity(df, interval)
            if timestamp_issues > 0:
                issues.append(ValidationIssue(
                    rule_name="ohlcv_timestamp_continuity",
                    severity=ValidationResult.WARNING,
                    message=f"Timestamp continuity issues in {timestamp_issues} records",
                    affected_records=timestamp_issues
                ))
        
        # 5. Price anomaly validation
        if 'close' in df.columns:
            price_anomalies = self._validate_price_anomalies(df)
            if price_anomalies > 0:
                issues.append(ValidationIssue(
                    rule_name="ohlcv_price_anomalies",
                    severity=ValidationResult.WARNING,
                    message=f"Price anomalies detected in {price_anomalies} records",
                    affected_records=price_anomalies
                ))
        
        # 6. Volume anomaly validation
        if 'volume' in df.columns:
            volume_anomalies = self._validate_volume_anomalies(df)
            if volume_anomalies > 0:
                issues.append(ValidationIssue(
                    rule_name="ohlcv_volume_anomalies",
                    severity=ValidationResult.WARNING,
                    message=f"Volume anomalies detected in {volume_anomalies} records",
                    affected_records=volume_anomalies
                ))
        
        # 7. Data freshness validation
        if 'open_time' in df.columns:
            freshness_issues = self._validate_data_freshness(df)
            if freshness_issues > 0:
                issues.append(ValidationIssue(
                    rule_name="timestamp_freshness",
                    severity=ValidationResult.WARNING,
                    message=f"Data freshness issues in {freshness_issues} records",
                    affected_records=freshness_issues
                ))
        
        # Calculate validation score
        total_issues = sum(issue.affected_records for issue in issues)
        validation_result['issues'] = issues
        validation_result['valid'] = len([i for i in issues if i.severity == ValidationResult.CRITICAL]) == 0
        
        if len(df) > 0:
            validation_result['score'] = max(0.0, 1.0 - (total_issues / len(df)))
            validation_result['valid_records'] = len(df) - total_issues
            validation_result['critical_records'] = sum(
                issue.affected_records for issue in issues 
                if issue.severity == ValidationResult.CRITICAL
            )
            validation_result['warning_records'] = sum(
                issue.affected_records for issue in issues 
                if issue.severity == ValidationResult.WARNING
            )
            validation_result['invalid_records'] = total_issues
        
        # Save validation results
        self._save_validation_results(symbol, interval, "ohlcv", validation_result)
        
        return validation_result
    
    def validate_funding_data(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Validate funding rate data and return validation results."""
        validation_result = {
            'valid': True,
            'issues': [],
            'score': 1.0,
            'total_records': len(df),
            'valid_records': 0,
            'warning_records': 0,
            'invalid_records': 0,
            'critical_records': 0
        }
        
        if df.empty:
            validation_result['valid'] = False
            validation_result['issues'].append(ValidationIssue(
                rule_name="data_completeness",
                severity=ValidationResult.CRITICAL,
                message="No funding data provided for validation",
                affected_records=0
            ))
            return validation_result
        
        issues = []
        
        # 1. Required columns validation
        required_columns = ['symbol', 'fundingRate', 'fundingTime']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            issues.append(ValidationIssue(
                rule_name="funding_required_columns",
                severity=ValidationResult.CRITICAL,
                message=f"Missing required columns: {missing_columns}",
                affected_records=len(df)
            ))
        
        # 2. Funding rate range validation
        if 'fundingRate' in df.columns:
            extreme_rates = self._validate_funding_rate_range(df)
            if extreme_rates > 0:
                issues.append(ValidationIssue(
                    rule_name="funding_rate_range",
                    severity=ValidationResult.WARNING,
                    message=f"Extreme funding rates in {extreme_rates} records",
                    affected_records=extreme_rates
                ))
        
        # 3. Timestamp validation
        if 'fundingTime' in df.columns:
            timestamp_issues = self._validate_funding_timestamps(df)
            if timestamp_issues > 0:
                issues.append(ValidationIssue(
                    rule_name="funding_timestamp_validity",
                    severity=ValidationResult.CRITICAL,
                    message=f"Invalid funding timestamps in {timestamp_issues} records",
                    affected_records=timestamp_issues
                ))
        
        # Calculate validation score
        total_issues = sum(issue.affected_records for issue in issues)
        validation_result['issues'] = issues
        validation_result['valid'] = len([i for i in issues if i.severity == ValidationResult.CRITICAL]) == 0
        
        if len(df) > 0:
            validation_result['score'] = max(0.0, 1.0 - (total_issues / len(df)))
            validation_result['valid_records'] = len(df) - total_issues
            validation_result['critical_records'] = sum(
                issue.affected_records for issue in issues 
                if issue.severity == ValidationResult.CRITICAL
            )
            validation_result['warning_records'] = sum(
                issue.affected_records for issue in issues 
                if issue.severity == ValidationResult.WARNING
            )
            validation_result['invalid_records'] = total_issues
        
        # Save validation results
        self._save_validation_results(symbol, "1h", "funding", validation_result)
        
        return validation_result
    
    def _validate_ohlc_relationships(self, df: pd.DataFrame) -> int:
        """Validate OHLC price relationships."""
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return len(df)
        
        # Check: high >= max(open, close) and low <= min(open, close)
        invalid_high = (df['high'] < df[['open', 'close']].max(axis=1))
        invalid_low = (df['low'] > df[['open', 'close']].min(axis=1))
        
        return (invalid_high | invalid_low).sum()
    
    def _validate_positive_values(self, df: pd.DataFrame) -> int:
        """Validate that OHLC and volume values are positive."""
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            return len(df)
        
        negative_values = (
            (df['open'] <= 0) |
            (df['high'] <= 0) |
            (df['low'] <= 0) |
            (df['close'] <= 0) |
            (df['volume'] < 0)
        )
        
        return negative_values.sum()
    
    def _validate_timestamp_continuity(self, df: pd.DataFrame, interval: str) -> int:
        """Validate timestamp continuity."""
        if 'open_time' not in df.columns:
            return len(df)
        
        # Sort by timestamp
        df_sorted = df.sort_values('open_time')
        
        # Calculate expected interval
        if interval == '1m':
            expected_interval = pd.Timedelta(minutes=1)
        elif interval == '5m':
            expected_interval = pd.Timedelta(minutes=5)
        elif interval == '15m':
            expected_interval = pd.Timedelta(minutes=15)
        elif interval == '1h':
            expected_interval = pd.Timedelta(hours=1)
        else:
            expected_interval = pd.Timedelta(minutes=1)
        
        # Find gaps larger than expected interval
        time_diffs = df_sorted['open_time'].diff().dropna()
        gaps = time_diffs > expected_interval * 1.5  # Allow 50% tolerance
        
        return gaps.sum()
    
    def _validate_price_anomalies(self, df: pd.DataFrame) -> int:
        """Validate for price anomalies."""
        if 'close' not in df.columns:
            return 0
        
        # Calculate price changes
        price_changes = df['close'].pct_change().abs()
        
        # Find extreme changes
        threshold = self.rules['ohlcv_price_anomalies'].threshold or 0.5
        extreme_changes = price_changes > threshold
        
        return extreme_changes.sum()
    
    def _validate_volume_anomalies(self, df: pd.DataFrame) -> int:
        """Validate for volume anomalies."""
        if 'volume' not in df.columns:
            return 0
        
        # Calculate volume spikes
        volume_rolling = df['volume'].rolling(window=20, min_periods=5).mean()
        threshold = self.rules['ohlcv_volume_anomalies'].threshold or 10.0
        volume_spikes = df['volume'] > (volume_rolling * threshold)
        
        return volume_spikes.sum()
    
    def _validate_data_freshness(self, df: pd.DataFrame) -> int:
        """Validate data freshness."""
        if 'open_time' not in df.columns:
            return 0
        
        # Check if data is too old
        latest_time = df['open_time'].max()
        now = datetime.now()
        hours_old = (now - latest_time).total_seconds() / 3600
        
        threshold = self.rules['timestamp_freshness'].threshold or 24.0
        if hours_old > threshold:
            return len(df)
        
        return 0
    
    def _validate_funding_rate_range(self, df: pd.DataFrame) -> int:
        """Validate funding rate range."""
        if 'fundingRate' not in df.columns:
            return 0
        
        threshold = self.rules['funding_rate_range'].threshold or 0.01
        extreme_rates = df['fundingRate'].abs() > threshold
        
        return extreme_rates.sum()
    
    def _validate_funding_timestamps(self, df: pd.DataFrame) -> int:
        """Validate funding timestamps."""
        if 'fundingTime' not in df.columns:
            return len(df)
        
        # Check for invalid timestamps
        invalid_timestamps = df['fundingTime'].isna()
        
        return invalid_timestamps.sum()
    
    def _save_validation_results(self, symbol: str, interval: str, data_type: str, result: Dict[str, Any]):
        """Save validation results to database."""
        try:
            import json
            
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                # Convert issues to JSON
                issues_json = json.dumps([{
                    'rule_name': issue.rule_name,
                    'severity': issue.severity.value,
                    'message': issue.message,
                    'affected_records': issue.affected_records,
                    'details': issue.details
                } for issue in result['issues']])
                
                cursor.execute("""
                    INSERT INTO validation_results 
                    (symbol, interval, data_type, validation_timestamp, total_records, 
                     valid_records, warning_records, invalid_records, critical_records, 
                     validation_score, issues_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    symbol,
                    interval,
                    data_type,
                    datetime.now(),
                    result['total_records'],
                    result['valid_records'],
                    result['warning_records'],
                    result['invalid_records'],
                    result['critical_records'],
                    result['score'],
                    issues_json
                ))
                
                # Save individual issues
                for issue in result['issues']:
                    cursor.execute("""
                        INSERT INTO validation_issues 
                        (symbol, interval, data_type, rule_name, severity, message, 
                         affected_records, details, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        symbol,
                        interval,
                        data_type,
                        issue.rule_name,
                        issue.severity.value,
                        issue.message,
                        issue.affected_records,
                        json.dumps(issue.details) if issue.details else None,
                        datetime.now()
                    ))
                
                conn.commit()
                self.logger.info(f"Validation results saved for {symbol} {interval} {data_type}")
                
        except Exception as e:
            self.logger.error(f"Error saving validation results: {e}")
    
    def get_validation_summary(self, symbol: str, interval: str, data_type: str, days: int = 7) -> Dict[str, Any]:
        """Get validation summary for a symbol/interval combination."""
        try:
            with sqlite3.connect(self.db_file) as conn:
                cursor = conn.cursor()
                
                # Get recent validation results
                cursor.execute("""
                    SELECT * FROM validation_results 
                    WHERE symbol = ? AND interval = ? AND data_type = ?
                    AND validation_timestamp >= datetime('now', '-{} days')
                    ORDER BY validation_timestamp DESC
                """.format(days), (symbol, interval, data_type))
                
                results = cursor.fetchall()
                
                if not results:
                    return {
                        'symbol': symbol,
                        'interval': interval,
                        'data_type': data_type,
                        'total_validations': 0,
                        'average_score': 0.0,
                        'total_issues': 0,
                        'critical_issues': 0,
                        'warning_issues': 0,
                        'last_validation': None
                    }
                
                # Calculate summary statistics
                total_validations = len(results)
                average_score = np.mean([r[10] for r in results])  # validation_score column
                total_issues = sum(r[7] for r in results)  # invalid_records column
                critical_issues = sum(r[8] for r in results)  # critical_records column
                warning_issues = sum(r[6] for r in results)  # warning_records column
                last_validation = results[0][3]  # validation_timestamp column
                
                return {
                    'symbol': symbol,
                    'interval': interval,
                    'data_type': data_type,
                    'total_validations': total_validations,
                    'average_score': average_score,
                    'total_issues': total_issues,
                    'critical_issues': critical_issues,
                    'warning_issues': warning_issues,
                    'last_validation': last_validation
                }
                
        except Exception as e:
            self.logger.error(f"Error getting validation summary: {e}")
            return {}

def main():
    """Main function for standalone data validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Data Validator")
    parser.add_argument("--symbol", default="BTCUSDT", help="Symbol to validate")
    parser.add_argument("--interval", default="15m", help="Interval to validate")
    parser.add_argument("--data-type", default="ohlcv", help="Data type to validate")
    parser.add_argument("--db-file", default="data/sqlite/runs.db", help="Database file path")
    parser.add_argument("--log-level", default="INFO", help="Log level")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create validator
    validator = DataValidator(
        db_file=args.db_file,
        log_level=args.log_level
    )
    
    # This would typically load data from parquet files and validate
    # For now, just show the validation rules
    print(f"Data Validator initialized for {args.symbol} {args.interval} {args.data_type}")
    print(f"Validation rules: {len(validator.rules)}")
    
    for rule_name, rule in validator.rules.items():
        print(f"  - {rule_name}: {rule.description} ({rule.severity.value})")

if __name__ == "__main__":
    main()
