#!/usr/bin/env python3
"""
Model Testing Script
Loads and tests the trained models for BTCUSDT and ETHUSDT
"""

import joblib
import json
import pandas as pd
import numpy as np
from pathlib import Path
import os

def load_model_metadata(symbol, model_type):
    """Load model metadata."""
    metadata_path = f"models/{model_type}/{symbol}/model_metadata.json"
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            return json.load(f)
    return None

def load_model(symbol, model_type, model_name):
    """Load a specific model."""
    model_path = f"models/{model_type}/{symbol}/{model_name}_model.pkl"
    if os.path.exists(model_path):
        return joblib.load(model_path)
    return None

def test_model_inference():
    """Test model inference with sample data."""
    print("=" * 60)
    print("MODEL TESTING AND INFERENCE DEMO")
    print("=" * 60)
    
    # Sample feature data (shifted features)
    sample_features = {
        'log_return_shifted': 0.001,
        'atr_14_shifted': 0.02,
        'sma_20_shifted': 0.98,
        'sma_100_shifted': 0.95,
        'rsi_14_shifted': 45.0,
        'volume_zscore_shifted': 0.5
    }
    
    print(f"Sample features: {sample_features}")
    print()
    
    for symbol in ['btcusdt', 'ethusdt']:
        print(f"--- Testing {symbol.upper()} Models ---")
        
        # Test XGBoost models
        print("\nXGBoost Models:")
        xgb_metadata = load_model_metadata(symbol, 'xgboost')
        if xgb_metadata:
            print(f"  Available models: {xgb_metadata['models_trained']}")
            
            for model_name in ['price_direction', 'price_volatility']:
                model_data = load_model(symbol, 'xgboost', model_name)
                if model_data:
                    # Prepare features
                    feature_array = np.array([[sample_features[feat] for feat in model_data['feature_names']]])
                    feature_scaled = model_data['scaler'].transform(feature_array)
                    
                    # Make prediction
                    prediction = model_data['model'].predict(feature_scaled)[0]
                    
                    if model_data['is_classification']:
                        print(f"  {model_name}: {prediction} (classification)")
                    else:
                        print(f"  {model_name}: {prediction:.6f} (regression)")
        
        # Test Unsupervised models
        print("\nUnsupervised Models:")
        unsup_metadata = load_model_metadata(symbol, 'unsupervised')
        if unsup_metadata:
            print(f"  Available models: {unsup_metadata['models_trained']}")
            
            # Test K-Means
            kmeans_model = load_model(symbol, 'unsupervised', 'kmeans_3')
            if kmeans_model:
                feature_array = np.array([[sample_features[feat] for feat in kmeans_model['feature_names']]])
                feature_scaled = kmeans_model['scaler'].transform(feature_array)
                cluster = kmeans_model['model'].predict(feature_scaled)[0]
                print(f"  K-Means (3 clusters): Cluster {cluster}")
            
            # Test Isolation Forest
            iso_model = load_model(symbol, 'unsupervised', 'isolation_forest')
            if iso_model:
                feature_array = np.array([[sample_features[feat] for feat in iso_model['feature_names']]])
                feature_scaled = iso_model['scaler'].transform(feature_array)
                anomaly_score = iso_model['model'].decision_function(feature_scaled)[0]
                is_anomaly = iso_model['model'].predict(feature_scaled)[0]
                print(f"  Isolation Forest: Score {anomaly_score:.4f}, Anomaly: {is_anomaly == -1}")
        
        print()

def display_model_summary():
    """Display a summary of all trained models."""
    print("=" * 60)
    print("TRAINED MODELS SUMMARY")
    print("=" * 60)
    
    for symbol in ['btcusdt', 'ethusdt']:
        print(f"\n{symbol.upper()}:")
        
        # XGBoost models
        xgb_metadata = load_model_metadata(symbol, 'xgboost')
        if xgb_metadata:
            print(f"  XGBoost Models ({len(xgb_metadata['models_trained'])}):")
            for model_name in xgb_metadata['models_trained']:
                model_data = load_model(symbol, 'xgboost', model_name)
                if model_data:
                    if model_data['is_classification']:
                        print(f"    - {model_name}: Accuracy - Train: {model_data['train_accuracy']:.4f}, Test: {model_data['test_accuracy']:.4f}")
                    else:
                        print(f"    - {model_name}: R2 - Train: {model_data['train_r2']:.4f}, Test: {model_data['test_r2']:.4f}")
        
        # Unsupervised models
        unsup_metadata = load_model_metadata(symbol, 'unsupervised')
        if unsup_metadata:
            print(f"  Unsupervised Models ({len(unsup_metadata['models_trained'])}):")
            for model_name in unsup_metadata['models_trained']:
                model_data = load_model(symbol, 'unsupervised', model_name)
                if model_data:
                    if model_data['model_type'] == 'kmeans':
                        print(f"    - {model_name}: {model_data['n_clusters']} clusters, Silhouette: {model_data['silhouette_score']:.4f}")
                    elif model_data['model_type'] == 'dbscan':
                        print(f"    - {model_name}: {model_data['n_clusters']} clusters, {model_data['n_noise']} noise points")
                    elif model_data['model_type'] == 'isolation_forest':
                        print(f"    - {model_name}: {model_data['n_anomalies']} anomalies ({model_data['anomaly_ratio']:.1%})")
                    elif model_data['model_type'] == 'pca':
                        print(f"    - {model_name}: {model_data['n_components']} components, {model_data['explained_variance_ratio']:.4f} variance")

def main():
    """Main function."""
    # Check if models directory exists
    if not os.path.exists('models'):
        print("[ERROR] Models directory not found. Please run train_models.py first.")
        return
    
    # Display model summary
    display_model_summary()
    
    # Test model inference
    test_model_inference()
    
    print("=" * 60)
    print("MODEL TESTING COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
