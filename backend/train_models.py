#!/usr/bin/env python3
"""
Machine Learning Model Trainer for BTCUSDT and ETHUSDT
Creates XGBoost models using shifted feature data
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
from pathlib import Path
import warnings
from datetime import datetime
import json

warnings.filterwarnings('ignore')

# Configuration
FEATURE_DIRS = {
    'btcusdt': 'feature/btcusdt/btcusdt_features.parquet',
    'ethusdt': 'feature/ethusdt/ethusdt_features.parquet'
}
MODEL_DIR = 'models'
SHIFTED_FEATURES = [
    'log_return_shifted', 'atr_14_shifted', 'sma_20_shifted', 
    'sma_100_shifted', 'rsi_14_shifted', 'volume_zscore_shifted'
]

def create_model_directory():
    """Create model directory structure."""
    Path(MODEL_DIR).mkdir(exist_ok=True)
    # Create subdirectories only for XGBoost models
    Path(f"{MODEL_DIR}/xgboost").mkdir(exist_ok=True)
    for symbol in ['btcusdt', 'ethusdt']:
        Path(f"{MODEL_DIR}/xgboost/{symbol}").mkdir(exist_ok=True)
    print(f"[OK] Created model directory structure in {MODEL_DIR}")

def load_and_prepare_data(symbol):
    """Load and prepare data for the given symbol."""
    print(f"\n=== Loading {symbol.upper()} Data ===")
    
    file_path = FEATURE_DIRS[symbol]
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Feature file not found: {file_path}")
    
    # Load data
    df = pd.read_parquet(file_path)
    print(f"[OK] Loaded {len(df)} rows from {file_path}")
    
    # Convert start_time to datetime
    df['start_time'] = pd.to_datetime(df['start_time'])
    
    # Select shifted features only
    feature_cols = [col for col in SHIFTED_FEATURES if col in df.columns]
    df_features = df[feature_cols].copy()
    
    # Remove rows with NaN values
    initial_rows = len(df_features)
    df_features = df_features.dropna()
    final_rows = len(df_features)
    
    print(f"[OK] Selected {len(feature_cols)} shifted features: {feature_cols}")
    print(f"[OK] Removed {initial_rows - final_rows} rows with NaN values")
    print(f"[OK] Final dataset: {final_rows} rows")
    
    # Get corresponding timestamps and prices
    timestamps = df['start_time'].iloc[df_features.index]
    prices = df[['open', 'high', 'low', 'close']].iloc[df_features.index]
    
    return df_features, timestamps, prices

def create_target_variables(df_features, prices):
    """Create target variables for supervised learning."""
    print("[INFO] Creating target variables...")
    
    targets = {}
    
    # Price movement direction (classification)
    price_change = (prices['close'] - prices['open']) / prices['open']
    targets['price_direction'] = (price_change > 0).astype(int)
    
    # Price volatility (regression)
    targets['price_volatility'] = (prices['high'] - prices['low']) / prices['open']
    
    # Log return (regression)
    targets['log_return'] = np.log(prices['close'] / prices['open'])
    
    # Future price movement (regression) - next hour
    targets['future_return'] = np.log(prices['close'].shift(-1) / prices['close'])
    
    # Remove NaN values
    for key, target in targets.items():
        targets[key] = target.dropna()
    
    print(f"[OK] Created {len(targets)} target variables")
    return targets

def train_xgboost_models(df_features, targets, symbol):
    """Train XGBoost models for classification and regression."""
    print(f"\n=== Training XGBoost Models for {symbol.upper()} ===")
    
    models = {}
    scaler = StandardScaler()
    
    # Scale features
    X_scaled = scaler.fit_transform(df_features)
    
    for target_name, target_values in targets.items():
        print(f"\n--- Training XGBoost for {target_name} ---")
        
        # Align features with target (remove NaN from target)
        valid_indices = target_values.index
        X_aligned = X_scaled[df_features.index.isin(valid_indices)]
        y_aligned = target_values.values
        
        if len(X_aligned) == 0:
            print(f"[SKIP] No valid data for {target_name}")
            continue
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_aligned, y_aligned, test_size=0.2, random_state=42, stratify=None
        )
        
        # Determine if classification or regression
        is_classification = target_name in ['price_direction']
        
        if is_classification:
            # Classification model
            model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            train_score = model.score(X_train, y_train)
            test_score = model.score(X_test, y_test)
            print(f"[OK] Classification accuracy - Train: {train_score:.4f}, Test: {test_score:.4f}")
            
        else:
            # Regression model
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Evaluate
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            train_mse = mean_squared_error(y_train, y_pred_train)
            test_mse = mean_squared_error(y_test, y_pred_test)
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            print(f"[OK] Regression MSE - Train: {train_mse:.6f}, Test: {test_mse:.6f}")
            print(f"[OK] Regression R2 - Train: {train_r2:.4f}, Test: {test_r2:.4f}")
        
        # Save model and metadata
        model_data = {
            'model': model,
            'scaler': scaler,
            'feature_names': SHIFTED_FEATURES,
            'target_name': target_name,
            'is_classification': is_classification,
            'train_size': len(X_train),
            'test_size': len(X_test),
            'timestamp': datetime.now().isoformat()
        }
        
        # Add evaluation metrics
        if is_classification:
            model_data['train_accuracy'] = train_score
            model_data['test_accuracy'] = test_score
        else:
            model_data['train_mse'] = train_mse
            model_data['test_mse'] = test_mse
            model_data['train_r2'] = train_r2
            model_data['test_r2'] = test_r2
        
        models[target_name] = model_data
        
        # Save individual model
        model_path = f"{MODEL_DIR}/xgboost/{symbol}/{target_name}_model.pkl"
        joblib.dump(model_data, model_path)
        print(f"[OK] Saved model to {model_path}")
    
    # Save combined model metadata
    metadata_path = f"{MODEL_DIR}/xgboost/{symbol}/model_metadata.json"
    metadata = {
        'symbol': symbol,
        'model_type': 'xgboost',
        'features_used': SHIFTED_FEATURES,
        'models_trained': list(models.keys()),
        'training_timestamp': datetime.now().isoformat(),
        'total_features': len(SHIFTED_FEATURES),
        'total_samples': len(df_features)
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"[OK] Saved metadata to {metadata_path}")
    return models

def main():
    """Main training function."""
    print("=" * 60)
    print("MACHINE LEARNING MODEL TRAINER")
    print("Training XGBoost models for BTCUSDT and ETHUSDT")
    print("=" * 60)
    
    # Create model directory structure
    create_model_directory()
    
    # Train models for each symbol
    for symbol in ['btcusdt', 'ethusdt']:
        print(f"\n{'='*20} {symbol.upper()} {'='*20}")
        
        try:
            # Load and prepare data
            df_features, timestamps, prices = load_and_prepare_data(symbol)
            
            # Create target variables
            targets = create_target_variables(df_features, prices)
            
            # Train XGBoost models
            xgb_models = train_xgboost_models(df_features, targets, symbol)
            
            print(f"\n[SUCCESS] Completed training for {symbol.upper()}")
            print(f"  - XGBoost models: {len(xgb_models)}")
            
        except Exception as e:
            print(f"[ERROR] Failed to train models for {symbol}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED!")
    print(f"Models saved in: {MODEL_DIR}/")
    print("=" * 60)

if __name__ == "__main__":
    main()
