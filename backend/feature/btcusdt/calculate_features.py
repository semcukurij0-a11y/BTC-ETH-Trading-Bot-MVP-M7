#!/usr/bin/env python3
"""
BTCUSDT Feature Calculator
Calculates technical indicators for BTCUSDT 1h data:
- Log returns
- ATR(14) - Average True Range
- SMA(20) and SMA(100) - Simple Moving Averages
- RSI(14) - Relative Strength Index
- Volume Z-Score
- All features are shifted by 1 for model training
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuration
DATA_FILE = "../../data/btcusdt/1h_btc.parquet"
OUTPUT_FILE = "btcusdt_features.parquet"

def calculate_log_returns(prices):
    """Calculate log returns"""
    return np.log(prices / prices.shift(1))

def calculate_atr(high, low, close, period=14):
    """Calculate Average True Range (ATR)"""
    high_low = high - low
    high_close = np.abs(high - close.shift(1))
    low_close = np.abs(low - close.shift(1))
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_sma(prices, period):
    """Calculate Simple Moving Average"""
    return prices.rolling(window=period).mean()

def calculate_rsi(prices, period=14):
    """Calculate Relative Strength Index (RSI)"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_volume_zscore(volume, period=20):
    """Calculate Volume Z-Score"""
    volume_mean = volume.rolling(window=period).mean()
    volume_std = volume.rolling(window=period).std()
    z_score = (volume - volume_mean) / volume_std
    return z_score

def main():
    print("=== BTCUSDT Feature Calculator ===")
    
    # Load data
    data_path = Path(DATA_FILE)
    if not data_path.exists():
        print(f"Error: Data file not found at {data_path}")
        return
    
    print(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    
    print(f"Loaded {len(df)} rows of data")
    print(f"Date range: {df['start_time'].min()} to {df['start_time'].max()}")
    
    # Sort by start_time to ensure proper order
    df = df.sort_values('start_time').reset_index(drop=True)
    
    # Calculate features
    print("Calculating technical indicators...")
    
    # Log returns
    df['log_return'] = calculate_log_returns(df['close'])
    
    # ATR(14)
    df['atr_14'] = calculate_atr(df['high'], df['low'], df['close'], period=14)
    
    # SMA(20) and SMA(100)
    df['sma_20'] = calculate_sma(df['close'], period=20)
    df['sma_100'] = calculate_sma(df['close'], period=100)
    
    # RSI(14)
    df['rsi_14'] = calculate_rsi(df['close'], period=14)
    
    # Volume Z-Score
    df['volume_zscore'] = calculate_volume_zscore(df['volume'], period=20)
    
    # Create feature columns for model training (shifted by 1)
    print("Creating shifted features for model training...")
    
    features_df = pd.DataFrame()
    features_df['start_time'] = df['start_time']
    features_df['open'] = df['open']
    features_df['high'] = df['high']
    features_df['low'] = df['low']
    features_df['close'] = df['close']
    features_df['volume'] = df['volume']
    
    # Shift all features by 1 for model training (predict next period)
    features_df['log_return_shifted'] = df['log_return'].shift(1)
    features_df['atr_14_shifted'] = df['atr_14'].shift(1)
    features_df['sma_20_shifted'] = df['sma_20'].shift(1)
    features_df['sma_100_shifted'] = df['sma_100'].shift(1)
    features_df['rsi_14_shifted'] = df['rsi_14'].shift(1)
    features_df['volume_zscore_shifted'] = df['volume_zscore'].shift(1)
    
    # Add current period features (for reference)
    features_df['log_return_current'] = df['log_return']
    features_df['atr_14_current'] = df['atr_14']
    features_df['sma_20_current'] = df['sma_20']
    features_df['sma_100_current'] = df['sma_100']
    features_df['rsi_14_current'] = df['rsi_14']
    features_df['volume_zscore_current'] = df['volume_zscore']
    
    # Remove rows with NaN values (first 100 rows due to SMA(100))
    initial_rows = len(features_df)
    features_df = features_df.dropna()
    final_rows = len(features_df)
    
    print(f"Removed {initial_rows - final_rows} rows with NaN values")
    print(f"Final dataset: {final_rows} rows")
    
    # Save features
    output_path = Path(OUTPUT_FILE)
    features_df.to_parquet(output_path, index=False)
    
    print(f"Features saved to {output_path}")
    
    # Display sample of features
    print("\nSample of calculated features:")
    print(features_df[['start_time', 'close', 'log_return_shifted', 'atr_14_shifted', 
                       'sma_20_shifted', 'sma_100_shifted', 'rsi_14_shifted', 'volume_zscore_shifted']].head(10))
    
    # Display feature statistics
    print("\nFeature Statistics:")
    feature_cols = ['log_return_shifted', 'atr_14_shifted', 'sma_20_shifted', 
                    'sma_100_shifted', 'rsi_14_shifted', 'volume_zscore_shifted']
    print(features_df[feature_cols].describe())
    
    print("\n=== Feature calculation completed ===")

if __name__ == "__main__":
    main()
