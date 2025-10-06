"""
ML Inference Service for Crypto Trading Bot

This module handles real-time ML predictions for crypto trading.
Output is mapped to s_ml in [-1, +1] range for trading decisions.
"""

import os
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
import joblib
from sklearn.calibration import CalibratedClassifierCV
import warnings
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class MLInference:
    """
    Machine Learning Inference for real-time crypto trading predictions.
    
    Features:
    - Real-time predictions
    - Output mapping to [-1, +1] range (s_ml)
    - Model loading and caching
    - Feature preprocessing
    - Prediction confidence scoring
    """
    
    def __init__(self, 
                 models_folder: str = "src/models",
                 config: Optional[Dict] = None):
        """
        Initialize ML Inference.
        
        Args:
            models_folder: Path to models directory
            config: Configuration dictionary
        """
        self.models_folder = Path(models_folder)
        self.config = config or {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Model configuration
        self.model_type = self.config.get('model_type', 'lightgbm')
        self.calibration_method = self.config.get('calibration_method', 'isotonic')
        
        # Feature configuration
        self.feature_columns = self.config.get('feature_columns', [])
        self.exclude_columns = self.config.get('exclude_columns', ['open_time', 'symbol', 'interval'])
        
        # Prediction configuration
        self.confidence_threshold = self.config.get('confidence_threshold', 0.6)
        self.min_features_required = self.config.get('min_features_required', 0.8)
        
        # Model cache
        self.model_cache = {}
        self.metadata_cache = {}
        
    def load_model(self, symbol: str, interval: str, force_reload: bool = False) -> Tuple[Any, Dict]:
        """
        Load model and metadata for a symbol/interval.
        
        Args:
            symbol: Trading symbol
            interval: Timeframe
            force_reload: Force reload from disk
            
        Returns:
            Tuple of (model, metadata)
        """
        cache_key = f"{symbol}_{interval}"
        
        # Check cache first
        if not force_reload and cache_key in self.model_cache:
            return self.model_cache[cache_key], self.metadata_cache[cache_key]
        
        # Find model files
        model_pattern = f"{symbol}_{interval}_{self.model_type}_*"
        model_files = list((self.models_folder / "calibrated").glob(f"{model_pattern}.joblib"))
        
        if not model_files:
            raise FileNotFoundError(f"No trained model found for {symbol} {interval}")
        
        # Get latest model
        latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
        
        # Load model
        self.logger.info(f"Loading model: {latest_model}")
        model = joblib.load(latest_model)
        
        # Load metadata
        # Convert .joblib to _metadata.json
        metadata_file = latest_model.parent / f"{latest_model.stem}_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        else:
            self.logger.warning(f"Metadata file not found: {metadata_file}")
            metadata = {}
        
        # Cache model
        self.model_cache[cache_key] = model
        self.metadata_cache[cache_key] = metadata
        
        return model, metadata
    
    def preprocess_features(self, df: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
        """
        Preprocess features for prediction.
        
        Args:
            df: Input DataFrame
            feature_columns: List of feature column names
            
        Returns:
            Preprocessed DataFrame
        """
        # Select features
        available_features = [col for col in feature_columns if col in df.columns]
        missing_features = [col for col in feature_columns if col not in df.columns]
        
        if missing_features:
            self.logger.warning(f"Missing features: {missing_features}")
        
        # Check if we have enough features
        feature_coverage = len(available_features) / len(feature_columns)
        if feature_coverage < self.min_features_required:
            raise ValueError(f"Insufficient features: {feature_coverage:.2%} < {self.min_features_required:.2%}")
        
        # Select available features
        features_df = df[available_features].copy()
        
        # Handle missing values
        features_df = features_df.fillna(method='ffill').fillna(0)
        
        # Remove infinite values
        features_df = features_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Ensure correct order of features
        features_df = features_df[available_features]
        
        return features_df
    
    def predict_proba(self, 
                     df: pd.DataFrame, 
                     symbol: str, 
                     interval: str) -> np.ndarray:
        """
        Get probability predictions for the given data.
        
        Args:
            df: Input DataFrame with features
            symbol: Trading symbol
            interval: Timeframe
            
        Returns:
            Array of probabilities [prob_negative, prob_positive]
        """
        # Load model
        model, metadata = self.load_model(symbol, interval)
        
        # Get feature columns from metadata or config
        feature_columns = metadata.get('feature_columns', self.feature_columns)
        if not feature_columns:
            # Use default feature columns if none specified
            self.logger.warning("No feature columns in metadata, using default feature columns")
            feature_columns = [
                'atr', 'sma_20', 'sma_100', 'rsi', 'volume_zscore',
                'target_class_60m', 'target_reg_60m', 'target_reg_30m', 'target_reg_90m',
                'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
                'returns_1h', 'returns_4h', 'returns_24h',
                'volatility_1h', 'volatility_4h', 'volatility_24h',
                'price_position', 'volume_position'
            ]
        
        # Preprocess features
        features_df = self.preprocess_features(df, feature_columns)
        
        # Get predictions
        probabilities = model.predict_proba(features_df)
        
        return probabilities
    
    def predict_s_ml(self, 
                    df: pd.DataFrame, 
                    symbol: str, 
                    interval: str,
                    confidence_weight: bool = True) -> pd.Series:
        """
        Predict s_ml values in [-1, +1] range.
        
        Args:
            df: Input DataFrame with features
            symbol: Trading symbol
            interval: Timeframe
            confidence_weight: Whether to weight by prediction confidence
            
        Returns:
            Series of s_ml values in [-1, +1] range
        """
        # Get probability predictions
        probabilities = self.predict_proba(df, symbol, interval)
        
        # Extract positive class probabilities
        prob_positive = probabilities[:, 1]
        
        # Map to [-1, +1] range
        # 0.5 probability = 0 (neutral)
        # 1.0 probability = +1 (strong buy)
        # 0.0 probability = -1 (strong sell)
        s_ml = 2 * prob_positive - 1
        
        # Apply confidence weighting if requested
        if confidence_weight:
            # Calculate confidence as distance from 0.5
            confidence = np.abs(prob_positive - 0.5) * 2  # [0, 1]
            
            # Weight s_ml by confidence
            s_ml = s_ml * confidence
        
        return pd.Series(s_ml, index=df.index)
    
    def predict_with_confidence(self, 
                               df: pd.DataFrame, 
                               symbol: str, 
                               interval: str) -> pd.DataFrame:
        """
        Get predictions with confidence scores.
        
        Args:
            df: Input DataFrame with features
            symbol: Trading symbol
            interval: Timeframe
            
        Returns:
            DataFrame with predictions and confidence
        """
        # Get probability predictions
        probabilities = self.predict_proba(df, symbol, interval)
        
        # Extract probabilities
        prob_negative = probabilities[:, 0]
        prob_positive = probabilities[:, 1]
        
        # Calculate s_ml
        s_ml = 2 * prob_positive - 1
        
        # Calculate confidence
        confidence = np.maximum(prob_positive, prob_negative)
        
        # Calculate prediction strength
        prediction_strength = np.abs(s_ml)
        
        # Create result DataFrame
        result = pd.DataFrame({
            's_ml': s_ml,
            'prob_positive': prob_positive,
            'prob_negative': prob_negative,
            'confidence': confidence,
            'prediction_strength': prediction_strength,
            'signal': np.where(s_ml > 0.1, 'BUY', np.where(s_ml < -0.1, 'SELL', 'HOLD'))
        }, index=df.index)
        
        return result
    
    def get_latest_prediction(self, 
                             symbol: str, 
                             interval: str,
                             features_df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        Get the latest prediction for a symbol/interval.
        
        Args:
            symbol: Trading symbol
            interval: Timeframe
            features_df: Optional features DataFrame (if None, will load latest)
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Load latest features if not provided
            if features_df is None:
                from .feature_engineering import FeatureEngineer
                fe = FeatureEngineer()
                features_df = fe.load_latest_features(symbol, interval)
                
                if features_df is None or len(features_df) == 0:
                    raise ValueError(f"No features available for {symbol} {interval}")
            
            # Get latest row
            latest_row = features_df.iloc[-1:].copy()
            
            # Check if we have a trained model
            try:
                model, metadata = self.load_model(symbol, interval)
            except FileNotFoundError:
                # No trained model available, return neutral prediction
                self.logger.warning(f"No trained model found for {symbol} {interval}, returning neutral prediction")
                return {
                    'symbol': symbol,
                    'interval': interval,
                    's_ml': 0.0,
                    'signal': 'HOLD',
                    'confidence': 0.0,
                    'prediction_strength': 0.0,
                    'prob_positive': 0.5,
                    'prob_negative': 0.5,
                    'timestamp': datetime.now().isoformat(),
                    'model_type': self.model_type,
                    'confidence_threshold': self.confidence_threshold,
                    'note': 'No trained model available'
                }
            
            # Get prediction
            prediction_df = self.predict_with_confidence(latest_row, symbol, interval)
            prediction = prediction_df.iloc[0].to_dict()
            
            # Add metadata
            prediction.update({
                'symbol': symbol,
                'interval': interval,
                'timestamp': datetime.now().isoformat(),
                'model_type': self.model_type,
                'confidence_threshold': self.confidence_threshold
            })
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Prediction failed for {symbol} {interval}: {e}")
            return {
                'symbol': symbol,
                'interval': interval,
                's_ml': 0.0,
                'signal': 'HOLD',
                'confidence': 0.0,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def batch_predict(self, 
                     features_dict: Dict[str, pd.DataFrame],
                     symbols: List[str],
                     intervals: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Get batch predictions for multiple symbols/intervals.
        
        Args:
            features_dict: Dictionary of {symbol_interval: features_df}
            symbols: List of symbols
            intervals: List of intervals
            
        Returns:
            Dictionary of predictions by symbol and interval
        """
        results = {}
        
        for symbol in symbols:
            results[symbol] = {}
            for interval in intervals:
                key = f"{symbol}_{interval}"
                if key in features_dict:
                    try:
                        prediction = self.get_latest_prediction(symbol, interval, features_dict[key])
                        results[symbol][interval] = prediction
                    except Exception as e:
                        self.logger.error(f"Batch prediction failed for {symbol} {interval}: {e}")
                        results[symbol][interval] = {'error': str(e)}
                else:
                    results[symbol][interval] = {'error': 'No features available'}
        
        return results
    
    def get_model_info(self, symbol: str, interval: str) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Args:
            symbol: Trading symbol
            interval: Timeframe
            
        Returns:
            Model information dictionary
        """
        try:
            model, metadata = self.load_model(symbol, interval)
            
            return {
                'symbol': symbol,
                'interval': interval,
                'model_type': self.model_type,
                'calibration_method': self.calibration_method,
                'feature_count': len(metadata.get('feature_columns', [])),
                'training_metrics': metadata.get('training_metrics', {}),
                'created_at': metadata.get('created_at'),
                'model_available': True
            }
        except Exception as e:
            return {
                'symbol': symbol,
                'interval': interval,
                'model_available': False,
                'error': str(e)
            }
    
    def clear_cache(self):
        """Clear model cache."""
        self.model_cache.clear()
        self.metadata_cache.clear()
        self.logger.info("Model cache cleared")


def main():
    """Test the ML Inference."""
    import sys
    sys.path.append('src')
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = {
        'model_type': 'lightgbm',
        'calibration_method': 'isotonic',
        'confidence_threshold': 0.6,
        'min_features_required': 0.8,
        'exclude_columns': ['open_time', 'symbol', 'interval', 'close', 'high', 'low', 'open', 'volume']
    }
    
    # Initialize inference
    inference = MLInference(config=config)
    
    # Test prediction
    try:
        prediction = inference.get_latest_prediction('BTCUSDT', '15m')
        print("Prediction completed successfully!")
        print(f"Prediction: {prediction}")
        
    except Exception as e:
        print(f"Prediction failed: {e}")


if __name__ == "__main__":
    main()
