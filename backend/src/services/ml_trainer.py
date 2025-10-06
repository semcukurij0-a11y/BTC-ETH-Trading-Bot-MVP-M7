"""
ML Trainer Service for Crypto Trading Bot

This module handles training tree-based models (LightGBM/XGBoost) with probability calibration
for crypto trading predictions. Output is mapped to s_ml in [-1, +1] range.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import joblib
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, log_loss, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")


class MLTrainer:
    """
    Machine Learning Trainer for crypto trading predictions.
    
    Features:
    - Tree-based models (LightGBM/XGBoost)
    - Probability calibration
    - Weekly retrain capability
    - Output mapping to [-1, +1] range
    - Model versioning and persistence
    """
    
    def __init__(self, 
                 data_folder: str = "src/data",
                 models_folder: str = "src/models",
                 config: Optional[Dict] = None):
        """
        Initialize ML Trainer.
        
        Args:
            data_folder: Path to data directory
            models_folder: Path to models directory
            config: Configuration dictionary
        """
        self.data_folder = Path(data_folder)
        self.models_folder = Path(models_folder)
        self.config = config or {}
        
        # Create directories
        self.models_folder.mkdir(parents=True, exist_ok=True)
        (self.models_folder / "lightgbm").mkdir(exist_ok=True)
        (self.models_folder / "xgboost").mkdir(exist_ok=True)
        (self.models_folder / "calibrated").mkdir(exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.model_type = self.config.get('model_type', 'lightgbm')  # 'lightgbm' or 'xgboost'
        self.calibration_method = self.config.get('calibration_method', 'isotonic')  # 'isotonic' or 'sigmoid'
        self.weekly_retrain = self.config.get('weekly_retrain', True)
        self.test_size = self.config.get('test_size', 0.2)
        self.random_state = self.config.get('random_state', 42)
        
        # Feature configuration
        self.target_column = self.config.get('target_column', 'future_return_1h')
        self.feature_columns = self.config.get('feature_columns', [])
        self.exclude_columns = self.config.get('exclude_columns', ['open_time', 'symbol', 'interval'])
        
        # Model parameters
        self.lightgbm_params = self.config.get('lightgbm_params', {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': self.random_state
        })
        
        self.xgboost_params = self.config.get('xgboost_params', {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'random_state': self.random_state
        })
        
        # Initialize models
        self.model = None
        self.calibrated_model = None
        self.feature_importance = None
        self.training_metrics = {}
        
    def load_training_data(self, 
                          symbol: str = "BTCUSDT", 
                          interval: str = "15m",
                          days_back: int = 30) -> pd.DataFrame:
        """
        Load and prepare training data from features.
        
        Args:
            symbol: Trading symbol
            interval: Timeframe
            days_back: Number of days to look back
            
        Returns:
            Prepared DataFrame for training
        """
        self.logger.info(f"Loading training data for {symbol} {interval} ({days_back} days back)")
        
        # Load features data
        features_path = self.data_folder / "features"
        feature_files = list(features_path.glob(f"features_{symbol}_{interval}_*.parquet"))
        
        if not feature_files:
            raise FileNotFoundError(f"No feature files found for {symbol} {interval}")
        
        # Load most recent feature file
        latest_file = max(feature_files, key=lambda x: x.stat().st_mtime)
        self.logger.info(f"Loading features from: {latest_file}")
        
        df = pd.read_parquet(latest_file)
        
        # Filter by date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        df['open_time'] = pd.to_datetime(df['open_time'])
        df = df[(df['open_time'] >= start_date) & (df['open_time'] <= end_date)]
        
        self.logger.info(f"Loaded {len(df)} records for training")
        
        return df
    
    def create_target_variable(self, df: pd.DataFrame, 
                              target_hours: int = 1,
                              threshold: float = 0.001) -> pd.DataFrame:
        """
        Create target variable for classification.
        
        Args:
            df: Input DataFrame
            target_hours: Hours ahead to predict
            threshold: Minimum return threshold for positive class
            
        Returns:
            DataFrame with target variable
        """
        self.logger.info(f"Creating target variable: {target_hours}h ahead, threshold={threshold}")
        
        # Check if we have existing target columns from feature engineering
        target_columns = [col for col in df.columns if col.startswith('target_class_')]
        
        if target_columns:
            # Use existing target columns from feature engineering
            self.logger.info(f"Using existing target columns: {target_columns}")
            
            # Find the closest target period to our target_hours
            target_periods = []
            for col in target_columns:
                try:
                    period = int(col.split('_')[-1].replace('m', ''))
                    target_periods.append((period, col))
                except:
                    continue
            
            if target_periods:
                # Find closest period to target_hours * 60 minutes
                target_minutes = target_hours * 60
                closest_period, closest_col = min(target_periods, key=lambda x: abs(x[0] - target_minutes))
                
                self.logger.info(f"Using target column {closest_col} (closest to {target_hours}h)")
                df[self.target_column] = df[closest_col]
            else:
                # Fallback to neutral target
                df[self.target_column] = 0
                self.logger.warning("No valid target columns found, using neutral target")
        else:
            # Fallback: try to use mark_close or index_close if available
            price_column = None
            if 'mark_close' in df.columns:
                price_column = 'mark_close'
            elif 'index_close' in df.columns:
                price_column = 'index_close'
            elif 'close' in df.columns:
                price_column = 'close'
            
            if price_column:
                self.logger.info(f"Using {price_column} for target calculation")
                # Calculate future returns
                df = df.sort_values('open_time')
                periods_ahead = target_hours * 4  # 4 periods per hour for 15m
                df['future_close'] = df[price_column].shift(-periods_ahead)
                df['future_return'] = (df['future_close'] - df[price_column]) / df[price_column]
                
                # Create binary target: 1 if return > threshold, 0 otherwise
                df[self.target_column] = (df['future_return'] > threshold).astype(int)
            else:
                # No price data available, use neutral target
                self.logger.warning("No price data available for target calculation, using neutral target")
                df[self.target_column] = 0
        
        # Remove rows with NaN targets (last few rows)
        df = df.dropna(subset=[self.target_column])
        
        self.logger.info(f"Target distribution: {df[self.target_column].value_counts().to_dict()}")
        
        return df
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare features for training.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (features_df, feature_columns)
        """
        self.logger.info("Preparing features for training")
        
        # Get feature columns (exclude target and metadata)
        if not self.feature_columns:
            feature_columns = [col for col in df.columns 
                             if col not in self.exclude_columns + [self.target_column, 'future_return', 'future_close']]
        else:
            feature_columns = self.feature_columns
        
        # Select features
        features_df = df[feature_columns + [self.target_column, 'open_time']].copy()
        
        # Handle missing values
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        # Remove rows with infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan).dropna()
        
        self.logger.info(f"Selected {len(feature_columns)} features")
        self.logger.info(f"Features: {feature_columns[:10]}...")  # Show first 10 features
        
        return features_df, feature_columns
    
    def train_model(self, 
                   symbol: str = "BTCUSDT",
                   interval: str = "15m",
                   days_back: int = 30,
                   target_hours: int = 1,
                   threshold: float = 0.001) -> Dict[str, Any]:
        """
        Train the ML model.
        
        Args:
            symbol: Trading symbol
            interval: Timeframe
            days_back: Days of training data
            target_hours: Hours ahead to predict
            threshold: Return threshold for positive class
            
        Returns:
            Training results dictionary
        """
        self.logger.info(f"Starting model training for {symbol} {interval}")
        
        # Load and prepare data
        df = self.load_training_data(symbol, interval, days_back)
        df = self.create_target_variable(df, target_hours, threshold)
        features_df, feature_columns = self.prepare_features(df)
        
        # Prepare training data
        X = features_df[feature_columns]
        y = features_df[self.target_column]
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state, stratify=y
        )
        
        self.logger.info(f"Training set: {len(X_train)} samples")
        self.logger.info(f"Test set: {len(X_test)} samples")
        
        # Train model
        if self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMClassifier(**self.lightgbm_params)
        elif self.model_type == 'xgboost' and XGBOOST_AVAILABLE:
            self.model = xgb.XGBClassifier(**self.xgboost_params)
        else:
            raise ValueError(f"Model type {self.model_type} not available")
        
        # Fit model
        self.model.fit(X_train, y_train)
        
        # Calibrate probabilities
        self.calibrated_model = CalibratedClassifierCV(
            self.model, method=self.calibration_method, cv=3
        )
        self.calibrated_model.fit(X_train, y_train)
        
        # Evaluate model
        train_pred = self.calibrated_model.predict_proba(X_train)[:, 1]
        test_pred = self.calibrated_model.predict_proba(X_test)[:, 1]
        
        train_auc = roc_auc_score(y_train, train_pred)
        test_auc = roc_auc_score(y_test, test_pred)
        train_loss = log_loss(y_train, train_pred)
        test_loss = log_loss(y_test, test_pred)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': feature_columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Store metrics
        self.training_metrics = {
            'train_auc': train_auc,
            'test_auc': test_auc,
            'train_loss': train_loss,
            'test_loss': test_loss,
            'n_features': len(feature_columns),
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test),
            'target_distribution': y.value_counts().to_dict()
        }
        
        self.logger.info(f"Training completed - Test AUC: {test_auc:.4f}, Test Loss: {test_loss:.4f}")
        
        # Save model
        self.save_model(symbol, interval)
        
        return {
            'model': self.calibrated_model,
            'metrics': self.training_metrics,
            'feature_importance': self.feature_importance,
            'feature_columns': feature_columns
        }
    
    def save_model(self, symbol: str, interval: str) -> str:
        """
        Save trained model to disk.
        
        Args:
            symbol: Trading symbol
            interval: Timeframe
            
        Returns:
            Path to saved model
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{symbol}_{interval}_{self.model_type}_{timestamp}"
        
        # Save calibrated model
        model_path = self.models_folder / "calibrated" / f"{model_name}.joblib"
        joblib.dump(self.calibrated_model, model_path)
        
        # Save metadata
        metadata = {
            'symbol': symbol,
            'interval': interval,
            'model_type': self.model_type,
            'calibration_method': self.calibration_method,
            'training_metrics': self.training_metrics,
            'feature_columns': self.feature_columns,
            'timestamp': timestamp,
            'created_at': datetime.now().isoformat()
        }
        
        metadata_path = self.models_folder / "calibrated" / f"{model_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save feature importance
        if self.feature_importance is not None:
            importance_path = self.models_folder / "calibrated" / f"{model_name}_importance.csv"
            self.feature_importance.to_csv(importance_path, index=False)
        
        self.logger.info(f"Model saved to: {model_path}")
        
        return str(model_path)
    
    def load_latest_model(self, symbol: str, interval: str) -> Optional[Any]:
        """
        Load the latest trained model for a symbol/interval.
        
        Args:
            symbol: Trading symbol
            interval: Timeframe
            
        Returns:
            Loaded model or None
        """
        model_pattern = f"{symbol}_{interval}_{self.model_type}_*"
        model_files = list((self.models_folder / "calibrated").glob(f"{model_pattern}.joblib"))
        
        if not model_files:
            self.logger.warning(f"No trained model found for {symbol} {interval}")
            return None
        
        # Get latest model
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        
        self.logger.info(f"Loading model: {latest_model}")
        self.calibrated_model = joblib.load(latest_model)
        
        return self.calibrated_model
    
    def should_retrain(self, symbol: str, interval: str) -> bool:
        """
        Check if model should be retrained based on weekly schedule.
        
        Args:
            symbol: Trading symbol
            interval: Timeframe
            
        Returns:
            True if retraining is needed
        """
        if not self.weekly_retrain:
            return False
        
        # Check if model exists
        model_pattern = f"{symbol}_{interval}_{self.model_type}_*"
        model_files = list((self.models_folder / "calibrated").glob(f"{model_pattern}.joblib"))
        
        if not model_files:
            return True  # No model exists, need to train
        
        # Check age of latest model
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        model_age = datetime.now() - datetime.fromtimestamp(latest_model.stat().st_mtime)
        
        return model_age.days >= 7  # Retrain if older than 7 days
    
    def get_training_summary(self) -> Dict[str, Any]:
        """
        Get summary of training results.
        
        Returns:
            Training summary dictionary
        """
        return {
            'model_type': self.model_type,
            'calibration_method': self.calibration_method,
            'metrics': self.training_metrics,
            'feature_importance': self.feature_importance.to_dict() if self.feature_importance is not None else None,
            'model_available': self.calibrated_model is not None
        }


def main():
    """Test the ML Trainer."""
    import sys
    sys.path.append('src')
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = {
        'model_type': 'lightgbm',
        'calibration_method': 'isotonic',
        'weekly_retrain': True,
        'test_size': 0.2,
        'random_state': 42,
        'target_column': 'future_return_1h',
        'exclude_columns': ['open_time', 'symbol', 'interval', 'close', 'high', 'low', 'open', 'volume']
    }
    
    # Initialize trainer
    trainer = MLTrainer(config=config)
    
    # Train model
    try:
        results = trainer.train_model(
            symbol='BTCUSDT',
            interval='15m',
            days_back=30,
            target_hours=1,
            threshold=0.001
        )
        
        print("Training completed successfully!")
        print(f"Test AUC: {results['metrics']['test_auc']:.4f}")
        print(f"Test Loss: {results['metrics']['test_loss']:.4f}")
        
    except Exception as e:
        print(f"Training failed: {e}")


if __name__ == "__main__":
    main()
