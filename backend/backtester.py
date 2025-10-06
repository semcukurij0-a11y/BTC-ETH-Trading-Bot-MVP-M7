#!/usr/bin/env python3
"""
Leveraged Trading Signal Backtest System

A comprehensive backtesting system for leveraged cryptocurrency trading with:
- 5x leverage support
- Cross-asset correlation analysis
- Advanced risk management
- Technical indicator calculations
- Signal generation and position sizing
"""

import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import argparse
import requests
import time

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

warnings.filterwarnings('ignore')

# Available cryptocurrency assets for trading
AVAILABLE_ASSETS = {
    'BTC': 'BTCUSDT',
    'BNB': 'BNBUSDT', 
    'ETH': 'ETHUSDT',
    'SOL': 'SOLUSDT',
    'XRP': 'XRPUSDT',
    'AVAX': 'AVAXUSDT',
    'ADA': 'ADAUSDT',
    'DOT': 'DOTUSDT',
    'LTC': 'LTCUSDT',
    'LINK': 'LINKUSDT'
}

def load_data_from_parquet(symbol: str, data_dir: str = 'data') -> pd.DataFrame:
    """
    Load data from parquet files in the data directory.
    
    Args:
        symbol: Trading pair symbol (e.g., 'BTCUSDT', 'ETHUSDT')
        data_dir: Directory containing parquet files
        
    Returns:
        DataFrame with OHLCV data
    """
    # Map symbols to their respective directories and files
    symbol_mapping = {
        'BTCUSDT': ('btcusdt', '1h_btc.parquet'),
        'ETHUSDT': ('ethusdt', '1h_eth.parquet')
    }
    
    if symbol not in symbol_mapping:
        print(f"Symbol {symbol} not supported for parquet loading")
        return pd.DataFrame()
    
    folder, filename = symbol_mapping[symbol]
    
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parquet_path = os.path.join(script_dir, data_dir, folder, filename)
    
    if not os.path.exists(parquet_path):
        print(f"Parquet file not found: {parquet_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {script_dir}")
        return pd.DataFrame()
    
    try:
        print(f"Loading {symbol} data from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        
        # Rename columns to match expected format
        column_mapping = {
            'start_time': 'timestamp',
            'open': 'open',
            'high': 'high', 
            'low': 'low',
            'close': 'close',
            'volume': 'volume'
        }
        
        # Keep only the columns we need and rename them
        df = df[list(column_mapping.keys())].rename(columns=column_mapping)
        
        # Convert timestamp to datetime and make it timezone-naive
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        # Convert to timezone-naive if timezone-aware
        if df['timestamp'].dt.tz is not None:
            df['timestamp'] = df['timestamp'].dt.tz_localize(None)
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Successfully loaded {len(df)} rows of {symbol} data from parquet")
        return df
        
    except Exception as e:
        print(f"Error loading parquet data for {symbol}: {e}")
            return pd.DataFrame()
        
        

def backup_csv_file(csv_file_path: str) -> str:
    """
    Create a backup of the CSV file before making changes.
    
    Args:
        csv_file_path: Path to the CSV file to backup
        
    Returns:
        Path to the backup file
    """
    if not os.path.exists(csv_file_path):
        return ""
    
    try:
        # Create backup directory
        backup_dir = os.path.join(os.path.dirname(csv_file_path), 'backups')
        os.makedirs(backup_dir, exist_ok=True)
        
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.basename(csv_file_path)
        name, ext = os.path.splitext(filename)
        backup_filename = f"{name}_backup_{timestamp}{ext}"
        backup_path = os.path.join(backup_dir, backup_filename)
        
        # Copy file to backup location
        import shutil
        shutil.copy2(csv_file_path, backup_path)
        
        print(f"Created backup: {backup_path}")
        return backup_path
        
    except Exception as e:
        print(f"Error creating backup: {e}")
        return ""

def clean_existing_csv_file(csv_file_path: str, symbol: str) -> bool:
    """
    Clean an existing CSV file by removing corrupted data and fixing timestamps.
    
    Args:
        csv_file_path: Path to the CSV file to clean
        symbol: Trading symbol for context
        
    Returns:
        True if cleaning was successful, False otherwise
    """
    try:
        print(f"Cleaning existing CSV file for {symbol}...")
        
        # Create backup first
        backup_path = backup_csv_file(csv_file_path)
        if not backup_path:
            print(f"Failed to create backup for {symbol}")
            return False
        
        # Read the CSV file
        df = pd.read_csv(csv_file_path)
        original_count = len(df)
        print(f"Original data: {original_count} rows")
        
        # Clean the data
        df_clean = clean_ohlcv_data(df, symbol)
        cleaned_count = len(df_clean)
        
        if cleaned_count < original_count:
            removed_count = original_count - cleaned_count
            print(f"Removed {removed_count} corrupted rows ({removed_count/original_count*100:.1f}%)")
            
            # Save cleaned data
            df_clean.to_csv(csv_file_path, index=False)
            print(f"Saved cleaned data: {cleaned_count} rows")
            return True
        else:
            print(f"No corrupted data found for {symbol}")
            return True
            
    except Exception as e:
        print(f"Error cleaning CSV file for {symbol}: {e}")
        return False

def clean_ohlcv_data(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """
    Clean and fix OHLCV data to ensure data integrity.
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Trading symbol for context
        
    Returns:
        Cleaned DataFrame
    """
    if df.empty:
        return df
    
    df_clean = df.copy()
    
    # Clean timestamp column first
    if 'timestamp' in df_clean.columns:
        # Try to convert timestamp to proper datetime format
        try:
            # First try to parse as datetime
            df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], errors='coerce')
            
            # Remove rows with invalid timestamps
            valid_timestamp_mask = df_clean['timestamp'].notna()
            if not valid_timestamp_mask.all():
                invalid_timestamp_count = (~valid_timestamp_mask).sum()
                print(f"Removing {invalid_timestamp_count} rows with invalid timestamps for {symbol}")
                df_clean = df_clean[valid_timestamp_mask]
            
            # Convert back to Unix timestamp (milliseconds)
            df_clean['timestamp'] = df_clean['timestamp'].astype('int64') // 10**6
            
        except Exception as e:
            print(f"Error processing timestamps for {symbol}: {e}")
            # Fallback: try to convert to numeric
            df_clean['timestamp'] = pd.to_numeric(df_clean['timestamp'], errors='coerce')
            valid_timestamp_mask = (df_clean['timestamp'] > 0) & df_clean['timestamp'].notna()
            if not valid_timestamp_mask.all():
                invalid_timestamp_count = (~valid_timestamp_mask).sum()
                print(f"Removing {invalid_timestamp_count} rows with invalid timestamps for {symbol}")
                df_clean = df_clean[valid_timestamp_mask]
    
    # Ensure core OHLCV columns exist and are numeric
    ohlcv_columns = ['open', 'high', 'low', 'close', 'volume']
    for col in ohlcv_columns:
        if col in df_clean.columns:
            # Convert to numeric, coercing errors to NaN
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
            # Replace any non-positive values with the previous valid value
            if col in ['open', 'high', 'low', 'close']:
                # For price columns, ensure they're positive
                df_clean[col] = df_clean[col].replace([0, -0], np.nan)
                df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
                
                # Ensure high >= max(open, close) and low <= min(open, close)
                if col == 'high':
                    df_clean[col] = df_clean[col].clip(lower=df_clean[['open', 'close']].max(axis=1))
                elif col == 'low':
                    df_clean[col] = df_clean[col].clip(upper=df_clean[['open', 'close']].min(axis=1))
    
    # Remove any rows where core OHLCV data is still invalid
    valid_mask = (
        df_clean['open'].notna() & 
        df_clean['high'].notna() & 
        df_clean['low'].notna() & 
        df_clean['close'].notna() &
        (df_clean['open'] > 0) &
        (df_clean['high'] > 0) &
        (df_clean['low'] > 0) &
        (df_clean['close'] > 0)
    )
    
    if not valid_mask.all():
        invalid_count = (~valid_mask).sum()
        print(f"Cleaned {invalid_count} invalid OHLCV rows for {symbol}")
        df_clean = df_clean[valid_mask]
    
    return df_clean

def validate_data_integrity(df: pd.DataFrame, symbol: str) -> bool:
    """
    Validate the integrity of the data before saving.
    
    Args:
        df: DataFrame to validate
        symbol: Trading symbol for context
        
    Returns:
        True if data is valid, False otherwise
    """
    try:
        # Check basic structure
        if df.empty:
            print(f"Warning: {symbol} data is empty")
            return False
        
        # Check required columns
        required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            print(f"Warning: {symbol} missing required columns: {missing_columns}")
            return False
        
        # Check for duplicate timestamps
        if 'timestamp' in df.columns:
            duplicate_timestamps = df['timestamp'].duplicated().sum()
            if duplicate_timestamps > 0:
                print(f"Warning: {symbol} has {duplicate_timestamps} duplicate timestamps")
                # For incremental updates, we'll allow some duplicates but warn
                if duplicate_timestamps > len(df) * 0.01:  # More than 1% duplicates
                    print(f"Too many duplicates ({duplicate_timestamps} > {len(df) * 0.01:.0f}). Data integrity compromised.")
                    return False
                else:
                    print(f"Acceptable number of duplicates for incremental update.")
        
        # Check for reasonable price values
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if col in df.columns:
                if df[col].isna().sum() > len(df) * 0.1:  # More than 10% NaN
                    print(f"Warning: {symbol} has too many NaN values in {col}")
                    return False
                
                # Check for non-positive values but allow some tolerance for synthetic data
                non_positive_count = (df[col] <= 0).sum()
                if non_positive_count > 0:
                    non_positive_pct = non_positive_count / len(df) * 100
                    if non_positive_pct > 5:  # More than 5% non-positive values
                        print(f"Warning: {symbol} has {non_positive_count} non-positive values in {col} ({non_positive_pct:.1f}%)")
                        return False
                    else:
                        print(f"Info: {symbol} has {non_positive_count} non-positive values in {col} ({non_positive_pct:.1f}%) - acceptable for synthetic data")
        
        # Check volume
        if 'volume' in df.columns:
            if df['volume'].isna().sum() > len(df) * 0.1:
                print(f"Warning: {symbol} has too many NaN values in volume")
                return False
        
        print(f"Data integrity check passed for {symbol}")
        return True
        
    except Exception as e:
        print(f"Error validating data integrity for {symbol}: {e}")
        return False



def select_asset() -> str:
    """
    Interactive asset selection from available cryptocurrencies.
    
    Returns:
        Selected asset symbol (e.g., 'BTCUSDT')
    """
    print("\nAvailable Cryptocurrency Assets:")
    print("=" * 40)
    
    for i, (short_name, full_symbol) in enumerate(AVAILABLE_ASSETS.items(), 1):
        print(f"{i:2d}. {short_name:4s} ({full_symbol})")
    
    while True:
        try:
            choice = input(f"\nSelect asset (1-{len(AVAILABLE_ASSETS)}): ").strip()
            choice_num = int(choice)
            
            if 1 <= choice_num <= len(AVAILABLE_ASSETS):
                selected_asset = list(AVAILABLE_ASSETS.values())[choice_num - 1]
                short_name = list(AVAILABLE_ASSETS.keys())[choice_num - 1]
                print(f"Selected: {short_name} ({selected_asset})")
                return selected_asset
            else:
                print(f"Please enter a number between 1 and {len(AVAILABLE_ASSETS)}")
                
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit(0)

def get_data_period() -> Tuple[str, str]:
    """
    Get start and end dates for data fetching.
    
    Returns:
        Tuple of (start_date, end_date) in 'YYYY-MM-DD' format
    """
    print("\nData Period Selection:")
    print("=" * 30)
    
    # Default to last 12 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    
    print(f"Default period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} (12 months)")
    
    use_default = input("Use default period? (y/n): ").strip().lower()
    
    if use_default in ['y', 'yes', '']:
        return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    while True:
        try:
            start_input = input("Enter start date (YYYY-MM-DD): ").strip()
            if start_input:
                start_date = datetime.strptime(start_input, '%Y-%m-%d')
            
            end_input = input("Enter end date (YYYY-MM-DD): ").strip()
            if end_input:
                end_date = datetime.strptime(end_input, '%Y-%m-%d')
            
            if start_date >= end_date:
                print("Start date must be before end date")
                continue
                
            if (end_date - start_date).days > 365:
                print("Warning: Period is longer than 1 year. This may take a while...")
                confirm = input("Continue? (y/n): ").strip().lower()
                if confirm not in ['y', 'yes']:
                    continue
            
            return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
            
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD")
        except KeyboardInterrupt:
            print("\nExiting...")
            exit(0)

def get_backtest_period() -> Tuple[str, str]:
    """
    Get start and end dates for backtest analysis period.
    
    Returns:
        Tuple of (start_date, end_date) in YYYY-MM-DD format, or (None, None) for all data
    """
    print("\nBacktest Period Selection:")
    print("=" * 30)
    print("1. Last 1 month")
    print("2. Last 3 months")
    print("3. Last 6 months") 
    print("4. Last 12 months")
    print("5. Last 2 years")
    print("6. Custom period")
    print("7. Use all available data")
    
    while True:
        try:
            choice = input("Enter choice (1-7): ").strip()
            if choice == '1':
                end_date = datetime.now()
                start_date = end_date - timedelta(days=30)
                return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
            elif choice == '2':
                end_date = datetime.now()
                start_date = end_date - timedelta(days=90)
                return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
            elif choice == '3':
                end_date = datetime.now()
                start_date = end_date - timedelta(days=180)
                return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
            elif choice == '4':
                end_date = datetime.now()
                start_date = end_date - timedelta(days=365)
                return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
            elif choice == '5':
                end_date = datetime.now()
                start_date = end_date - timedelta(days=730)
                return start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
            elif choice == '6':
                start_date = input("Enter start date (YYYY-MM-DD): ").strip()
                end_date = input("Enter end date (YYYY-MM-DD): ").strip()
                
                # Validate date format
                datetime.strptime(start_date, '%Y-%m-%d')
                datetime.strptime(end_date, '%Y-%m-%d')
                
                # Validate date range
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                if start_dt >= end_dt:
                    print("Start date must be before end date.")
                    continue
                
                return start_date, end_date
            elif choice == '7':
                return None, None  # Use all available data
            else:
                print("Invalid choice. Please enter 1-7.")
        except ValueError:
            print("Invalid date format. Please use YYYY-MM-DD format.")
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return None, None

def filter_data_by_period(df: pd.DataFrame, start_date: str = None, end_date: str = None) -> pd.DataFrame:
    """
    Filter DataFrame by date period for backtest analysis.
    
    Args:
        df: DataFrame with timestamp column
        start_date: Start date in YYYY-MM-DD format (optional)
        end_date: End date in YYYY-MM-DD format (optional)
        
    Returns:
        Filtered DataFrame
    """
    if start_date is None and end_date is None:
        print("Using all available data for backtest")
        return df
    
    # Convert timestamp column to datetime if it's not already
    if 'timestamp' in df.columns:
        df_copy = df.copy()
        df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'])
        
        # Apply date filters
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df_copy = df_copy[df_copy['timestamp'] >= start_dt]
            print(f"Filtered data from {start_date}")
        
        if end_date:
            end_dt = pd.to_datetime(end_date)
            df_copy = df_copy[df_copy['timestamp'] <= end_dt]
            print(f"Filtered data until {end_date}")
        
        print(f"Backtest period: {df_copy['timestamp'].min().strftime('%Y-%m-%d')} to {df_copy['timestamp'].max().strftime('%Y-%m-%d')}")
        print(f"Data points for backtest: {len(df_copy)} rows")
        
        return df_copy
    else:
        print("Warning: No timestamp column found. Using all data.")
        return df

def save_trades_to_csv(trades: List[Dict], symbol: str, results: Dict) -> str:
    """
    Save backtest trades to a professional CSV file.
    
    Args:
        trades: List of trade dictionaries
        symbol: Trading symbol
        results: Backtest results dictionary
        
    Returns:
        Path to the saved CSV file
    """
    if not trades:
        print("No trades to save")
        return ""
    
    # Create output directory
    output_dir = 'backtest_trades'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{symbol}_trades_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)
    
    # Convert trades to DataFrame
    df_trades = pd.DataFrame(trades)
    
    # Add summary information as metadata
    summary_info = {
        'symbol': symbol,
        'total_return': results.get('total_return', 0),
        'sharpe_ratio': results.get('sharpe_ratio', 0),
        'max_drawdown': results.get('max_drawdown', 0),
        'total_trades': results.get('total_trades', 0),
        'win_rate': results.get('win_rate', 0),
        'avg_gain_per_trade_pct': results.get('avg_gain_per_trade_pct', 0),
        'margin_calls': results.get('margin_calls', 0),
        'leverage': results.get('leverage', 10),
        'profit_factor': results.get('profit_factor', 0),
        'gross_profit': results.get('gross_profit', 0),
        'gross_loss': results.get('gross_loss', 0),
        'largest_win': results.get('largest_win', 0),
        'largest_loss': results.get('largest_loss', 0),
        'win_loss_ratio': results.get('win_loss_ratio', 0),
        'backtest_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Create a comprehensive CSV with summary and trades
    with open(filepath, 'w', newline='', encoding='utf-8') as f:
        # Write summary header
        f.write("# LEVERAGED BACKTEST TRADES REPORT\n")
        f.write(f"# Symbol: {summary_info['symbol']}\n")
        f.write(f"# Backtest Date: {summary_info['backtest_date']}\n")
        f.write(f"# Total Return: {summary_info['total_return']:.2%}\n")
        f.write(f"# Sharpe Ratio: {summary_info['sharpe_ratio']:.4f}\n")
        f.write(f"# Max Drawdown: {summary_info['max_drawdown']:.2%}\n")
        f.write(f"# Total Trades: {summary_info['total_trades']}\n")
        f.write(f"# Win Rate: {summary_info['win_rate']:.2%}\n")
        f.write(f"# Avg Gain per Trade: {summary_info['avg_gain_per_trade_pct']:.4f}%\n")
        f.write(f"# Margin Calls: {summary_info['margin_calls']}\n")
        f.write(f"# Leverage: {summary_info['leverage']}x\n")
        
        # Profit Factor metrics
        profit_factor = summary_info['profit_factor']
        if profit_factor == float('inf'):
            f.write(f"# Profit Factor: ∞ (No losses)\n")
        else:
            f.write(f"# Profit Factor: {profit_factor:.2f}\n")
        
        f.write(f"# Gross Profit: ${summary_info['gross_profit']:.2f}\n")
        f.write(f"# Gross Loss: ${summary_info['gross_loss']:.2f}\n")
        f.write(f"# Largest Win: ${summary_info['largest_win']:.2f}\n")
        f.write(f"# Largest Loss: ${summary_info['largest_loss']:.2f}\n")
        f.write(f"# Win/Loss Ratio: {summary_info['win_loss_ratio']:.2f}\n")
        f.write("#\n")
        
        # Write trades data
        df_trades.to_csv(f, index=False)
    
    print(f"Trades saved to: {filepath}")
    print(f"Total trades exported: {len(trades)}")
    
    return filepath

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range (ATR) for volatility measurement.
    
    Args:
        high: High price series
        low: Low price series  
        close: Close price series
        period: Rolling window period for ATR calculation
        
    Returns:
        ATR series with same length as input data
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    atr = true_range.rolling(window=period).mean()
    return atr

def calculate_correlation_strength(corr_value: float) -> float:
    """
    Convert correlation value to normalized signal strength (0-1 scale).
    
    Args:
        corr_value: Correlation coefficient (-1 to 1)
        
    Returns:
        Signal strength value between 0.0 and 1.0
    """
    if pd.isna(corr_value):
        return 0.0
    
    abs_corr = abs(corr_value)
    
    if abs_corr >= 0.8:
        return 0.9
    elif abs_corr >= 0.6:
        return 0.7
    elif abs_corr >= 0.4:
        return 0.5
    elif abs_corr >= 0.2:
        return 0.3
    else:
        return 0.1

def calculate_cointegration_strength(residual: float, threshold: float = 2.0) -> float:
    """
    Calculate cointegration signal strength based on residual deviation from mean.
    
    Args:
        residual: Cointegration residual value
        threshold: Base threshold for signal strength calculation
        
    Returns:
        Signal strength value between 0.0 and 1.0
    """
    if pd.isna(residual):
        return 0.0
    
    abs_residual = abs(residual)
    
    if abs_residual >= threshold * 2:
        return 0.9
    elif abs_residual >= threshold:
        return 0.7
    elif abs_residual >= threshold * 0.5:
        return 0.5
    elif abs_residual >= threshold * 0.25:
        return 0.3
    else:
        return 0.1

def calculate_cross_asset_signal_strength(row: pd.Series, symbol: str) -> Dict[str, float]:
    """
    Calculate cross-asset signal strength using top 3-5 most correlated assets.
    
    Args:
        row: Data row containing correlation and residual columns
        symbol: Trading symbol for analysis
        
    Returns:
        Dictionary containing signal strength components
    """
    signal_components = {
        'correlation_strength': 0.0,
        'cointegration_strength': 0.0,
        'residual_strength': 0.0,
        'cross_asset_confidence': 0.0
    }
    
    # Get correlation columns for this symbol
    corr_columns = [col for col in row.index if col.startswith('corr_return_1_vs_') and col.endswith('USDT')]
    
    if not corr_columns:
        return signal_components
    
    # Focus on top 3-5 most correlated assets (simplified approach)
    corr_values = []
    for col in corr_columns:
        corr_value = row.get(col, 0)
        if not pd.isna(corr_value):
            corr_values.append((col, abs(corr_value)))  # Use absolute correlation
    
    # Sort by correlation strength and take top 5
    corr_values.sort(key=lambda x: x[1], reverse=True)
    top_correlations = corr_values[:5]  # Top 5 most correlated assets
    
    # Calculate correlation strength from top assets only
    if top_correlations:
        top_corr_strengths = [calculate_correlation_strength(corr_value) for _, corr_value in top_correlations]
        signal_components['correlation_strength'] = np.mean(top_corr_strengths)
    
    # Get corresponding residual columns for top correlated assets
    top_asset_names = [col.replace('corr_return_1_vs_', '') for col, _ in top_correlations]
    residual_columns = [f'residual_return_1_vs_{asset}' for asset in top_asset_names 
                       if f'residual_return_1_vs_{asset}' in row.index]
    
    # Calculate cointegration strength from top assets only
    if residual_columns:
        residual_strengths = []
        for col in residual_columns:
            residual_value = row.get(col, 0)
            if not pd.isna(residual_value):
                residual_strengths.append(calculate_cointegration_strength(residual_value))
        
        if residual_strengths:
            signal_components['cointegration_strength'] = np.mean(residual_strengths)
            signal_components['residual_strength'] = np.mean(residual_strengths)
    
    # Calculate cross-asset confidence from top assets only
    if residual_columns:
        cross_asset_signals = []
        for col in residual_columns:
            residual_value = row.get(col, 0)
            if not pd.isna(residual_value):
                # Strong divergence (potential reversal signal)
                if abs(residual_value) > 2.0:
                    cross_asset_signals.append(0.8)
                # Moderate divergence
                elif abs(residual_value) > 1.0:
                    cross_asset_signals.append(0.6)
                # Weak divergence
                elif abs(residual_value) > 0.5:
                    cross_asset_signals.append(0.4)
                else:
                    cross_asset_signals.append(0.2)
        
        if cross_asset_signals:
            signal_components['cross_asset_confidence'] = np.mean(cross_asset_signals)
    
    return signal_components

def calculate_kelly_fraction(returns: List[float], min_trades: int = 10) -> float:
    """
    Calculate Kelly fraction for optimal position sizing based on historical returns.
    
    Args:
        returns: List of historical trade returns
        min_trades: Minimum number of trades required for calculation
        
    Returns:
        Kelly fraction between 0.05 and 0.5 (5% to 50%)
    """
    if len(returns) < min_trades:
        return 0.25
    
    returns_array = np.array(returns)
    
    wins = returns_array[returns_array > 0]
    losses = returns_array[returns_array < 0]
    
    if len(wins) == 0 or len(losses) == 0:
        return 0.25
    
    win_rate = len(wins) / len(returns)
    avg_win = np.mean(wins)
    avg_loss = abs(np.mean(losses))
    
    if avg_loss == 0:
        return 0.25
    
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - win_rate
    
    kelly_fraction = (b * p - q) / b
    return max(0.05, min(0.5, kelly_fraction))

def calculate_profit_factor(trades: List[Dict[str, Any]]) -> Dict[str, float]:
    """
    Calculate profit factor and related trading metrics from trade history.
    
    Args:
        trades: List of trade dictionaries containing trade data
        
    Returns:
        Dictionary containing profit factor and related metrics
    """
    if not trades:
        return {
            'profit_factor': 0.0,
            'gross_profit': 0.0,
            'gross_loss': 0.0,
            'total_profit_trades': 0,
            'total_loss_trades': 0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'win_loss_ratio': 0.0
        }
    
    # Extract profit/loss data from trades
    profits = []
    losses = []
    profit_trades = 0
    loss_trades = 0
    
    for trade in trades:
        if trade.get('action') == 'SELL' and 'profit_loss' in trade:
            pnl = trade['profit_loss']
            if pnl > 0:
                profits.append(pnl)
                profit_trades += 1
            elif pnl < 0:
                losses.append(abs(pnl))  # Store as positive values
                loss_trades += 1
    
    # Calculate metrics
    gross_profit = sum(profits) if profits else 0.0
    gross_loss = sum(losses) if losses else 0.0
    
    # Calculate profit factor
    if gross_loss == 0:
        profit_factor = float('inf') if gross_profit > 0 else 0.0
    else:
        profit_factor = gross_profit / gross_loss
    
    # Additional metrics
    largest_win = max(profits) if profits else 0.0
    largest_loss = max(losses) if losses else 0.0
    avg_win = np.mean(profits) if profits else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    
    # Win/Loss ratio (average win / average loss)
    if avg_loss == 0:
        win_loss_ratio = float('inf') if avg_win > 0 else 0.0
    else:
        win_loss_ratio = avg_win / avg_loss
    
    return {
        'profit_factor': profit_factor,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'total_profit_trades': profit_trades,
        'total_loss_trades': loss_trades,
        'largest_win': largest_win,
        'largest_loss': largest_loss,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'win_loss_ratio': win_loss_ratio
    }

class LeveragedBacktestEngine:
    """
    Leveraged backtesting engine with 5x leverage and conservative risk management.
    
    This engine provides comprehensive backtesting capabilities for leveraged cryptocurrency
    trading with advanced features including:
    - Cross-asset correlation analysis
    - Dynamic position sizing using Kelly criterion
    - ATR-based stop losses and profit targets
    - Volatility regime filtering
    - Risk management with drawdown limits
    """
    
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 leverage: float = 5.0,
                 commission: float = 0.0001,
                 slippage: float = 0.00005,
                 max_position_size: float = 0.10,
                 base_position_size: float = 0.05,
                 stop_loss_pct: float = 0.001,
                 profit_target_pct: float = 0.008,
                 trailing_stop_pct: float = 0.15,
                 use_atr_stops: bool = True,
                 atr_multiplier: float = 0.6,
                 atr_profit_multiplier: float = 3.0,
                 atr_period: int = 14,
                 use_trend_following: bool = True,
                 use_fixed_take_profit: bool = False,
                 use_fixed_stop_loss: bool = True,
                 use_kelly_sizing: bool = True,
                 kelly_lookback: int = 50,
                 kelly_fraction_multiplier: float = 1.0,
                 use_volatility_filter: bool = False,
                 volatility_threshold: float = 0.03,
                 min_liquidity_hours: List[int] = None,
                 max_drawdown_limit: float = 0.015,
                 daily_stop_loss: float = 0.01,
                 min_cross_asset_confirmations: int = 1,
                 cointegration_threshold: float = 1.5,
                 use_volatility_regime_filter: bool = True,
                 max_atr_threshold: float = 0.67,
                 min_signal_strength: float = 0.4,
                 min_confidence: float = 0.3,
                 min_target_return: float = 0.002,
                 max_portfolio_risk: float = 0.02,
                 margin_call_threshold: float = 0.8,
                 cross_asset_lookback: int = 100):
        
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.commission = commission
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.base_position_size = base_position_size
        self.stop_loss_pct = stop_loss_pct
        self.profit_target_pct = profit_target_pct
        self.trailing_stop_pct = trailing_stop_pct
        self.use_atr_stops = use_atr_stops
        self.atr_multiplier = atr_multiplier
        self.atr_profit_multiplier = atr_profit_multiplier
        self.atr_period = atr_period
        self.use_trend_following = use_trend_following
        self.use_fixed_take_profit = use_fixed_take_profit
        self.use_fixed_stop_loss = use_fixed_stop_loss
        self.use_kelly_sizing = use_kelly_sizing
        self.kelly_lookback = kelly_lookback
        self.kelly_fraction_multiplier = kelly_fraction_multiplier
        self.use_volatility_filter = use_volatility_filter
        self.volatility_threshold = volatility_threshold
        self.min_liquidity_hours = min_liquidity_hours or [22, 23, 0, 1, 2, 3, 4, 5]
        self.max_drawdown_limit = max_drawdown_limit
        self.daily_stop_loss = daily_stop_loss
        self.min_cross_asset_confirmations = min_cross_asset_confirmations
        self.cointegration_threshold = cointegration_threshold
        self.use_volatility_regime_filter = use_volatility_regime_filter
        self.max_atr_threshold = max_atr_threshold
        self.min_signal_strength = min_signal_strength
        self.min_confidence = min_confidence
        self.min_target_return = min_target_return
        self.max_portfolio_risk = max_portfolio_risk
        self.margin_call_threshold = margin_call_threshold
        self.cross_asset_lookback = cross_asset_lookback
        
        self.cross_asset_data = {}
        self.cross_asset_returns = {}
        
        self.capital = initial_capital
        self.available_margin = initial_capital
        self.used_margin = 0.0
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.margin_calls = 0
        
        self.peak_capital = initial_capital
        self.current_drawdown = 0.0
        self.trading_halted = False
        
        self.daily_start_capital = initial_capital
        self.current_date = None
        self.daily_loss = 0.0
        
    def reset(self):
        """Reset all state."""
        self.capital = self.initial_capital
        self.available_margin = self.initial_capital
        self.used_margin = 0.0
        self.positions = {}
        self.trades = []
        self.portfolio_values = []
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.margin_calls = 0
        
        self.peak_capital = self.initial_capital
        self.current_drawdown = 0.0
        self.trading_halted = False
    
    def get_timestamp_column(self, df: pd.DataFrame) -> str:
        """Get timestamp column."""
        timestamp_cols = ['timestamps', 'timestamp', 'open_time', 'close_time']
        for col in timestamp_cols:
            if col in df.columns:
                return col
        raise ValueError("No timestamp column found. Expected one of: timestamps, timestamp, open_time, close_time")
    
    def calculate_margin_requirements(self, position_value: float) -> float:
        """Calculate margin requirement for leveraged position."""
        return position_value / self.leverage
    
    def check_margin_call(self) -> bool:
        """Check if margin call is triggered."""
        margin_ratio = self.used_margin / self.capital if self.capital > 0 else 0
        return margin_ratio >= self.margin_call_threshold
    
    def calculate_dynamic_profit_multiplier(self, df: pd.DataFrame, current_atr: float) -> float:
        """Calculate dynamic profit multiplier based on market conditions."""
        base_multiplier = self.atr_profit_multiplier
        
        if len(df) < 20:
            return base_multiplier
        
        recent_vol = df['close'].iloc[-20:].pct_change().std()
        recent_trend = (df['close'].iloc[-1] - df['close'].iloc[-20]) / df['close'].iloc[-20]
        
        if recent_vol < 0.01:
            volatility_adjustment = 0.4
        elif recent_vol > 0.03:
            volatility_adjustment = -0.3
        else:
            volatility_adjustment = 0.0
        
        if abs(recent_trend) > 0.05:
            trend_adjustment = 0.5
        elif abs(recent_trend) < 0.01:
            trend_adjustment = -0.2
        else:
            trend_adjustment = 0.0
        
        dynamic_multiplier = base_multiplier + volatility_adjustment + trend_adjustment
        return max(2.5, min(3.5, dynamic_multiplier))
    
    def calculate_simple_position_size(self, signal_strength: float, confidence: float) -> float:
        """
        Calculate position size without requiring trade history (for CSV processing).
        Uses the same logic as the original position sizing but without Kelly fraction.
        """
        base_fraction = self.base_position_size
        quality_multiplier = (abs(signal_strength) * confidence)
        
        if quality_multiplier > 0.8:
            return self.max_position_size
        elif quality_multiplier > 0.6:
            return base_fraction * 1.5
        elif quality_multiplier > 0.4:
            return base_fraction * 1.2
        else:
            return base_fraction * 0.8
    
    def calculate_atr_stop_levels(self, df: pd.DataFrame, entry_price: float, is_long: bool) -> Dict[str, float]:
        """Calculate ATR-based stop levels."""
        if not self.use_atr_stops:
            if is_long:
                stop_loss = entry_price * (1 - self.stop_loss_pct)
                profit_target = entry_price * (1 + self.profit_target_pct)
            else:
                stop_loss = entry_price * (1 + self.stop_loss_pct)
                profit_target = entry_price * (1 - self.profit_target_pct)
            
            return {
                'stop_loss': stop_loss,
                'profit_target': profit_target,
                'trailing_stop': self.trailing_stop_pct
            }
        
        atr = calculate_atr(df['high'], df['low'], df['close'], self.atr_period)
        current_atr = atr.iloc[-1] if not atr.empty else entry_price * 0.01
        
        atr_stop_distance = current_atr * self.atr_multiplier
        dynamic_profit_multiplier = self.calculate_dynamic_profit_multiplier(df, current_atr)
        
        if is_long:
            stop_loss = entry_price - atr_stop_distance
            profit_target = entry_price + (atr_stop_distance * dynamic_profit_multiplier)
        else:
            stop_loss = entry_price + atr_stop_distance
            profit_target = entry_price - (atr_stop_distance * dynamic_profit_multiplier)
        
        atr_trailing_pct = atr_stop_distance / entry_price
        
        return {
            'stop_loss': stop_loss,
            'profit_target': profit_target,
            'trailing_stop': atr_trailing_pct,
            'atr_value': current_atr,
            'dynamic_profit_multiplier': dynamic_profit_multiplier
        }
    
    def calculate_kelly_position_size(self, signal_strength: float, confidence: float) -> float:
        """Calculate position size using Kelly fraction based on past performance."""
        if not self.use_kelly_sizing:
            base_fraction = self.base_position_size
            quality_multiplier = (abs(signal_strength) * confidence)
            if quality_multiplier > 0.8:
                return self.max_position_size
            elif quality_multiplier > 0.6:
                return base_fraction * 1.5
            elif quality_multiplier > 0.4:
                return base_fraction * 1.2
            else:
                return base_fraction * 0.8
        
        recent_returns = []
        if len(self.trades) > 0:
            recent_trades = self.trades[-self.kelly_lookback:]
            for trade in recent_trades:
                if 'return_pct' in trade:
                    recent_returns.append(trade['return_pct'])
        
        kelly_fraction = calculate_kelly_fraction(recent_returns)
        
        kelly_range = 0.5 - 0.05
        position_range = self.max_position_size - self.base_position_size
        
        kelly_scaled = (kelly_fraction - 0.05) / kelly_range
        kelly_scaled = max(0.0, min(1.0, kelly_scaled))
        
        kelly_adjusted_base = self.base_position_size + (kelly_scaled * position_range)
        
        quality_multiplier = (abs(signal_strength) * confidence)
        if quality_multiplier > 0.8:
            position_fraction = min(self.max_position_size, kelly_adjusted_base * 1.2)
        elif quality_multiplier > 0.6:
            position_fraction = kelly_adjusted_base * 1.1
        elif quality_multiplier > 0.4:
            position_fraction = kelly_adjusted_base
        else:
            position_fraction = kelly_adjusted_base * 0.9
        
        position_fraction = max(0.01, min(self.max_position_size, position_fraction))
        
        return position_fraction
    
    def check_volatility_filter(self, df: pd.DataFrame, current_idx: int, timestamp: str) -> bool:
        """Optimized volatility filtering for faster backtesting while maintaining effectiveness."""
        if not self.use_volatility_filter:
            return True
        
        try:
            if ' ' in timestamp:
                time_part = timestamp.split(' ')[0]
                hour = int(time_part.split(':')[0])
            else:
                hour = 12
            
            if hour in self.min_liquidity_hours:
                return False
        except (ValueError, IndexError):
            pass
        
        if current_idx < 20:
            return True
        
        recent_prices = df['close'].iloc[current_idx-19:current_idx+1]
        price_changes = recent_prices.pct_change().dropna()
        current_volatility = price_changes.std()
        
        if current_volatility > self.volatility_threshold:
            return False
        
        recent_high = df['high'].iloc[current_idx-4:current_idx+1].max()
        recent_low = df['low'].iloc[current_idx-4:current_idx+1].min()
        price_range_pct = (recent_high - recent_low) / recent_low
        
        if price_range_pct > (self.volatility_threshold * 3.0):
            return False
        
        if 'atr_pct' in df.columns and current_idx > 0:
            current_atr_pct = df['atr_pct'].iloc[current_idx]
            if current_atr_pct > 0.05:
                return False
        
        if current_idx >= 50 and current_idx % 10 == 0:
            recent_20_vol = current_volatility
            previous_20_vol = df['close'].iloc[current_idx-39:current_idx-19].pct_change().std()
            
            if previous_20_vol > 0:
                volatility_trend = recent_20_vol / previous_20_vol
                if volatility_trend > 2.0:
                    return False
        
        return True
    
    def update_drawdown_tracking(self, current_portfolio_value: float):
        """Update drawdown tracking."""
        if current_portfolio_value > self.peak_capital:
            self.peak_capital = current_portfolio_value
            self.trading_halted = False  # Reset trading halt if we reach new peak
        
        self.current_drawdown = (current_portfolio_value - self.peak_capital) / self.peak_capital
    
    def check_drawdown_limit(self) -> bool:
        """Check if drawdown limit has been exceeded."""
        return abs(self.current_drawdown) >= self.max_drawdown_limit
    
    def check_daily_stop_loss(self, timestamp: str, current_portfolio_value: float) -> bool:
        """Check if daily stop-loss limit has been exceeded."""
        # Extract date from timestamp
        try:
            current_date = timestamp.split(' ')[0]  # Get date part
        except:
            return False
        
        # Reset daily tracking if new day
        if self.current_date != current_date:
            self.current_date = current_date
            self.daily_start_capital = current_portfolio_value
            self.daily_loss = 0.0
        
        # Calculate daily loss
        self.daily_loss = (self.daily_start_capital - current_portfolio_value) / self.daily_start_capital
        
        # Check if daily stop-loss exceeded
        return self.daily_loss >= self.daily_stop_loss
    
    def check_cross_asset_confirmation(self, row: pd.Series, signal_direction: int) -> bool:
        """Check if at least minimum number of cross-asset tokens confirm the signal."""
        confirmations = 0
        
        # Get correlation columns for cross-asset analysis
        corr_columns = [col for col in row.index if col.startswith('corr_return_1_vs_') and col.endswith('USDT')]
        residual_columns = [col for col in row.index if col.startswith('residual_return_1_vs_') and col.endswith('USDT')]
        
        # Track cross-asset analysis for debugging (optional)
        if hasattr(self, '_cross_asset_debug_count'):
            self._cross_asset_debug_count += 1
        else:
            self._cross_asset_debug_count = 0
            
        if self._cross_asset_debug_count < 3:  # Log first 3 checks for debugging
            print(f"    Cross-Asset Debug: signal_direction={signal_direction}, min_confirmations={self.min_cross_asset_confirmations}")
            print(f"    Found {len(corr_columns)} correlation columns, {len(residual_columns)} residual columns")
        
        # Check correlation confirmations
        corr_confirmations = 0
        for col in corr_columns:
            corr_value = row.get(col, 0)
            if not pd.isna(corr_value):
                # For long signals, look for positive correlation
                # For short signals, look for negative correlation
                if signal_direction == 1 and corr_value > 0.3:  # Positive correlation for longs
                    confirmations += 1
                    corr_confirmations += 1
                elif signal_direction == -1 and corr_value < -0.3:  # Negative correlation for shorts
                    confirmations += 1
                    corr_confirmations += 1
        
        # Check cointegration confirmations (residual > 1.5σ)
        residual_confirmations = 0
        for col in residual_columns:
            residual_value = row.get(col, 0)
            if not pd.isna(residual_value):
                if abs(residual_value) > self.cointegration_threshold:
                    if signal_direction == 1 and residual_value < -self.cointegration_threshold:
                        confirmations += 1
                        residual_confirmations += 1
                    elif signal_direction == -1 and residual_value > self.cointegration_threshold:
                        confirmations += 1
                        residual_confirmations += 1
        
        if self._cross_asset_debug_count < 3:
            print(f"    Confirmations: total={confirmations}, corr={corr_confirmations}, residual={residual_confirmations}")
            if confirmations < self.min_cross_asset_confirmations:
                print(f"    REJECTED: Insufficient confirmations ({confirmations} < {self.min_cross_asset_confirmations})")
            else:
                print(f"    PASSED: Sufficient confirmations ({confirmations} >= {self.min_cross_asset_confirmations})")
        
        return confirmations >= self.min_cross_asset_confirmations
    
    def check_volatility_regime_filter(self, row: pd.Series) -> bool:
        """Check if current volatility regime is suitable for trading."""
        if not self.use_volatility_regime_filter:
            return True
        
        atr_pct = row.get('atr_pct', 0)
        rolling_vol = row.get('rolling_vol_15m', 0)
        
        if pd.isna(atr_pct):
            atr_pct = 0
        if pd.isna(rolling_vol):
            rolling_vol = 0
        
        if hasattr(self, '_debug_count'):
            self._debug_count += 1
        else:
            self._debug_count = 0
            
        if self._debug_count < 5:
            print(f"    Volatility Filter Debug: atr_pct={atr_pct:.4f}, rolling_vol={rolling_vol:.4f}")
            print(f"    Thresholds: atr_max={self.max_atr_threshold:.4f}, vol_min=0.001, vol_max=0.05")
        
        if atr_pct > self.max_atr_threshold:
            if self._debug_count < 5:
                print(f"    REJECTED: ATR too high ({atr_pct:.4f} > {self.max_atr_threshold:.4f})")
            return False
        
        if rolling_vol <= 0.0000:
            if self._debug_count < 5:
                print(f"    REJECTED: Rolling vol too low ({rolling_vol:.4f} <= 0.001)")
            return False
        if rolling_vol >= 0.05:
            if self._debug_count < 5:
                print(f"    REJECTED: Rolling vol too high ({rolling_vol:.4f} >= 0.05)")
            return False
        
        if self._debug_count < 5:
            print(f"    PASSED: Volatility regime filter")
        return True
    
    def force_liquidate_positions(self, timestamp: str, reason: str = 'MARGIN_CALL') -> List[Dict[str, Any]]:
        """Force liquidate all positions due to margin call."""
        liquidated_trades = []
        
        for symbol in list(self.positions.keys()):
            position = self.positions[symbol]
            current_price = position['entry_price']
            
            exit_trade = self._execute_exit(symbol, current_price, timestamp, reason, 0, 0)
            if exit_trade:
                liquidated_trades.append(exit_trade)
                if reason == 'MARGIN_CALL':
                    self.margin_calls += 1
        
        return liquidated_trades
    
    def generate_leveraged_signal(self, row: pd.Series, symbol: str = '') -> Dict[str, Any]:
        """Generate high-quality signal optimized for leveraged trading with cross-asset analysis."""
        target_direction = row.get('target_direction', 0)
        target_return = row.get('target_return_1', 0)
        
        if pd.isna(target_direction):
            target_direction = 0
        if pd.isna(target_return):
            target_return = 0
        
        rsi_14 = row.get('rsi_14', 50)
        ma_slope_5 = row.get('ma_slope_5', 0)
        volume_ratio = row.get('volume_ratio', 1)
        rolling_vol_15m = row.get('rolling_vol_15m', 0)
        atr_pct = row.get('atr_pct', 0)
        
        if pd.isna(rsi_14):
            rsi_14 = 50
        if pd.isna(ma_slope_5):
            ma_slope_5 = 0
        if pd.isna(volume_ratio) or volume_ratio <= 0:
            volume_ratio = 1
        if pd.isna(rolling_vol_15m):
            rolling_vol_15m = 0
        if pd.isna(atr_pct):
            atr_pct = 0
        
        current_idx = row.name if hasattr(row, 'name') else 0
        
        correlations = self.calculate_cross_asset_correlations(symbol, current_idx)
        residuals = self.calculate_cross_asset_residuals(symbol, current_idx)
        
        row_with_cross_asset = row.copy()
        for key, value in correlations.items():
            row_with_cross_asset[key] = value
        for key, value in residuals.items():
            row_with_cross_asset[key] = value
        
        cross_asset_signals = calculate_cross_asset_signal_strength(row_with_cross_asset, symbol)
        correlation_strength = cross_asset_signals['correlation_strength']
        cointegration_strength = cross_asset_signals['cointegration_strength']
        cross_asset_confidence = cross_asset_signals['cross_asset_confidence']
        
        signal_strength = 0.0
        confidence = 0.0
        
        if target_direction == 1 and target_return > self.min_target_return:
            signal_strength = 0.5
            confidence = 0.4
            
            if rsi_14 < 25:
                signal_strength += 0.4
                confidence += 0.3
            elif rsi_14 < 30:
                signal_strength += 0.3
                confidence += 0.25
            elif rsi_14 < 35:
                signal_strength += 0.2
                confidence += 0.15
            elif rsi_14 > 70:
                signal_strength -= 0.3
                confidence -= 0.2
            
            if ma_slope_5 > 0.003:
                signal_strength += 0.3
                confidence += 0.25
            elif ma_slope_5 > 0.0015:
                signal_strength += 0.25
                confidence += 0.2
            elif ma_slope_5 > 0.0008:
                signal_strength += 0.15
                confidence += 0.1
            elif ma_slope_5 < -0.0008:
                signal_strength -= 0.25
                confidence -= 0.15
            
            if volume_ratio > 2.5:
                signal_strength += 0.25
                confidence += 0.2
            elif volume_ratio > 1.8:
                signal_strength += 0.2
                confidence += 0.15
            elif volume_ratio > 1.3:
                signal_strength += 0.1
                confidence += 0.1
            elif volume_ratio < 0.8:
                signal_strength -= 0.2
                confidence -= 0.15
            
            if 0.01 < rolling_vol_15m < 0.02:
                signal_strength += 0.2
                confidence += 0.15
            elif rolling_vol_15m > 0.03:
                signal_strength -= 0.4
                confidence -= 0.3
            
            if 0.015 < atr_pct < 0.025:
                signal_strength += 0.15
                confidence += 0.1
            elif atr_pct > 0.035:
                signal_strength -= 0.3
                confidence -= 0.2
            
            if correlation_strength > 0.7:
                signal_strength += 0.2
                confidence += 0.15
            elif correlation_strength > 0.5:
                signal_strength += 0.1
                confidence += 0.1
            elif correlation_strength < 0.2:
                signal_strength += 0.15
                confidence += 0.1
            
            if cointegration_strength > 0.7:
                signal_strength += 0.25
                confidence += 0.2
            elif cointegration_strength > 0.5:
                signal_strength += 0.15
                confidence += 0.1
            
            if cross_asset_confidence > 0.6:
                confidence += 0.15
                
        elif target_direction == 0 and target_return < -self.min_target_return:
            signal_strength = -0.5
            confidence = 0.4
            
            if rsi_14 > 75:
                signal_strength -= 0.4
                confidence += 0.3
            elif rsi_14 > 70:
                signal_strength -= 0.3
                confidence += 0.25
            elif rsi_14 > 65:
                signal_strength -= 0.2
                confidence += 0.15
            elif rsi_14 < 30:
                signal_strength += 0.3
                confidence -= 0.2
            
            if ma_slope_5 < -0.003:
                signal_strength -= 0.3
                confidence += 0.25
            elif ma_slope_5 < -0.0015:
                signal_strength -= 0.25
                confidence += 0.2
            elif ma_slope_5 < -0.0008:
                signal_strength -= 0.15
                confidence += 0.1
            elif ma_slope_5 > 0.0008:
                signal_strength += 0.25
                confidence -= 0.15
            
            if volume_ratio > 2.5:
                signal_strength -= 0.25
                confidence += 0.2
            elif volume_ratio > 1.8:
                signal_strength -= 0.2
                confidence += 0.15
            elif volume_ratio > 1.3:
                signal_strength -= 0.1
                confidence += 0.1
            elif volume_ratio < 0.8:
                signal_strength += 0.2
                confidence -= 0.15
            
            if 0.01 < rolling_vol_15m < 0.02:
                signal_strength -= 0.2
                confidence += 0.15
            elif rolling_vol_15m > 0.03:
                signal_strength += 0.4
                confidence -= 0.3
            
            if correlation_strength > 0.7:
                signal_strength -= 0.2
                confidence += 0.15
            elif correlation_strength > 0.5:
                signal_strength -= 0.1
                confidence += 0.1
            elif correlation_strength < 0.2:
                signal_strength -= 0.15
                confidence += 0.1
            
            if cointegration_strength > 0.7:
                signal_strength -= 0.25
                confidence += 0.2
            elif cointegration_strength > 0.5:
                signal_strength -= 0.15
                confidence += 0.1
            
            if cross_asset_confidence > 0.6:
                confidence += 0.15
        
        signal_strength = np.clip(signal_strength, -1.0, 1.0)
        confidence = np.clip(confidence, 0.0, 1.0)
        
        if abs(signal_strength) >= self.min_signal_strength and confidence >= self.min_confidence:
            signal_direction = 1 if signal_strength > 0 else -1
            if self.check_cross_asset_confirmation(row_with_cross_asset, signal_direction):
                signal = 'BUY' if signal_strength > 0 else 'SELL'
            else:
                signal = 'HOLD'
                signal_strength = 0.0
        elif abs(signal_strength) >= 0.2 and confidence >= 0.2:
            signal = 'WEAK_BUY' if signal_strength > 0 else 'WEAK_SELL'
        else:
            signal = 'HOLD'
            signal_strength = 0.0
        
        if signal in ['BUY', 'SELL']:
            position_size = self.base_position_size * (1 + signal_strength * confidence)
            position_size = min(position_size, self.max_position_size)
        else:
            position_size = 0.0
        
        return {
            'signal': signal,
            'signal_strength': signal_strength,
            'confidence': confidence,
            'position_size': position_size,
            'target_return': target_return,
            'rsi_14': rsi_14,
            'ma_slope_5': ma_slope_5,
            'volume_ratio': volume_ratio,
            'rolling_vol_15m': rolling_vol_15m,
            'atr_pct': atr_pct,
            'correlation_strength': correlation_strength,
            'cointegration_strength': cointegration_strength,
            'cross_asset_confidence': cross_asset_confidence
        }

    def generate_exit_signal(self, row: pd.Series, symbol: str, current_position: Dict) -> Dict[str, Any]:
        """
        Generate exit signal based on trend reversal detection.
        This is used for trend-following behavior instead of fixed take-profit.
        """
        if not self.use_trend_following:
            return {'exit_signal': 'HOLD', 'exit_strength': 0.0, 'exit_confidence': 0.0}
        
        # Get current position direction
        position_size = current_position.get('size', 0)
        is_long_position = position_size > 0
        
        # Generate current market signal using same logic as entry
        current_signal_data = self.generate_leveraged_signal(row, symbol)
        current_signal = current_signal_data.get('signal', 'HOLD')
        current_signal_strength = current_signal_data.get('signal_strength', 0.0)
        current_confidence = current_signal_data.get('confidence', 0.0)
        
        # Determine exit signal based on position direction and current signal
        exit_signal = 'HOLD'
        exit_strength = 0.0
        exit_confidence = 0.0
        
        if is_long_position:
            # For long positions, exit on SELL signals or trend reversal
            if current_signal in ['SELL', 'WEAK_SELL']:
                exit_signal = 'SELL'
                exit_strength = abs(current_signal_strength)
                exit_confidence = current_confidence
            elif current_signal == 'HOLD' and current_confidence < 0.2:
                # Exit if trend weakens significantly (low confidence)
                exit_signal = 'SELL'
                exit_strength = 0.3  # Moderate exit strength
                exit_confidence = 0.2
        else:
            # For short positions, exit on BUY signals or trend reversal
            if current_signal in ['BUY', 'WEAK_BUY']:
                exit_signal = 'BUY'
                exit_strength = abs(current_signal_strength)
                exit_confidence = current_confidence
            elif current_signal == 'HOLD' and current_confidence < 0.2:
                # Exit if trend weakens significantly (low confidence)
                exit_signal = 'BUY'
                exit_strength = 0.3  # Moderate exit strength
                exit_confidence = 0.2
        
        # Additional trend reversal detection using technical indicators
        rsi_14 = row.get('rsi', 50)
        ma_slope_5 = row.get('ma_slope', 0)
        
        if is_long_position:
            # For long positions, check for bearish reversal signals
            if rsi_14 > 70 and ma_slope_5 < -0.001:  # Overbought + declining MA
                if exit_signal == 'HOLD':
                    exit_signal = 'SELL'
                    exit_strength = 0.4
                    exit_confidence = 0.3
                else:
                    # Strengthen existing exit signal
                    exit_strength = max(exit_strength, 0.4)
                    exit_confidence = max(exit_confidence, 0.3)
        else:
            # For short positions, check for bullish reversal signals
            if rsi_14 < 30 and ma_slope_5 > 0.001:  # Oversold + rising MA
                if exit_signal == 'HOLD':
                    exit_signal = 'BUY'
                    exit_strength = 0.4
                    exit_confidence = 0.3
                else:
                    # Strengthen existing exit signal
                    exit_strength = max(exit_strength, 0.4)
                    exit_confidence = max(exit_confidence, 0.3)
        
        return {
            'exit_signal': exit_signal,
            'exit_strength': exit_strength,
            'exit_confidence': exit_confidence,
            'current_signal': current_signal,
            'current_signal_strength': current_signal_strength,
            'current_confidence': current_confidence,
            'rsi_14': rsi_14,
            'ma_slope_5': ma_slope_5
        }
    
    def manage_leveraged_position(self, symbol: str, current_price: float, 
                                 high: float, low: float, timestamp: str, row: pd.Series = None) -> Optional[Dict[str, Any]]:
        """Advanced position management for leveraged trading with trend-following exits."""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        entry_price = position['entry_price']
        position_size = position['size']
        
        # Update high/low tracking
        if 'highest_price' not in position:
            position['highest_price'] = current_price
            position['lowest_price'] = current_price
        
        position['highest_price'] = max(position['highest_price'], high)
        position['lowest_price'] = min(position['lowest_price'], low)
        
        # Calculate current metrics
        current_return = (current_price - entry_price) / entry_price
        mfe = (position['highest_price'] - entry_price) / entry_price
        mae = (entry_price - position['lowest_price']) / entry_price
        
        # Update position with current market value and unrealized P&L
        current_market_value = position_size * current_price
        entry_cost = position_size * entry_price
        unrealized_pnl = current_market_value - entry_cost
        
        position['value'] = current_market_value
        position['unrealized_pnl'] = unrealized_pnl
        
        # 1. ALWAYS check stop-loss for risk management (if enabled)
        if self.use_fixed_stop_loss and current_price <= position['stop_loss']:
            return self._execute_exit(symbol, current_price, timestamp, 'STOP_LOSS', mfe, mae)
        
        # 2. Check trend-following exit signals (if enabled)
        if self.use_trend_following and row is not None:
            exit_signal_data = self.generate_exit_signal(row, symbol, position)
            exit_signal = exit_signal_data.get('exit_signal', 'HOLD')
            exit_strength = exit_signal_data.get('exit_strength', 0.0)
            exit_confidence = exit_signal_data.get('exit_confidence', 0.0)
            
            # Exit if we have a strong exit signal
            if exit_signal != 'HOLD' and exit_strength >= 0.3 and exit_confidence >= 0.2:
                return self._execute_exit(symbol, current_price, timestamp, 'TREND_REVERSAL', mfe, mae)
        
        # 3. Check fixed take-profit (only if trend-following is disabled)
        if self.use_fixed_take_profit and not self.use_trend_following:
            if current_price >= position['profit_target']:
                return self._execute_exit(symbol, current_price, timestamp, 'PROFIT_TARGET', mfe, mae)
        
        # 4. Check trailing stop (only if trend-following is disabled)
        if not self.use_trend_following and mfe > 0.005:  # Only apply trailing stop after 0.5% profit
            peak_profit_price = position['highest_price']
            trailing_stop_price = peak_profit_price * (1 - position['trailing_stop_pct'])
            
            if current_price <= trailing_stop_price:
                return self._execute_exit(symbol, current_price, timestamp, 'TRAILING_STOP', mfe, mae)
        
        return None
    
    def _execute_exit(self, symbol: str, price: float, timestamp: str, 
                     exit_type: str, mfe: float, mae: float) -> Dict[str, Any]:
        """Execute leveraged position exit."""
        position = self.positions[symbol]
        position_size = position['size']
        entry_price = position['entry_price']
        margin_used = position['margin_used']
        
        # Calculate trade details
        execution_price = price * (1 - self.slippage)
        trade_value = position_size * execution_price
        commission_cost = trade_value * self.commission
        
        entry_value = position_size * entry_price
        profit_loss = trade_value - entry_value - commission_cost
        trade_return_pct = profit_loss / margin_used if margin_used > 0 else 0
        
        # Update capital and margin
        self.capital += profit_loss
        self.used_margin -= margin_used
        self.available_margin += margin_used
        del self.positions[symbol]
        
        
        # Update statistics
        self.total_trades += 1
        if profit_loss > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        return {
            'timestamp': timestamp,
            'symbol': symbol,
            'signal': exit_type,
            'action': 'SELL',
            'price': execution_price,
            'position_size': -position_size,
            'trade_value': trade_value,
            'commission': commission_cost,
            'profit_loss': profit_loss,
            'trade_return_pct': trade_return_pct * 100,
            'margin_used': margin_used,
            'capital_before': self.capital - profit_loss,
            'capital_after': self.capital,
            'mfe': mfe,
            'mae': mae,
            'exit_reason': exit_type
        }
    
    def execute_leveraged_trade(self, timestamp: str, symbol: str, signal_data: Dict, price: float, df: pd.DataFrame) -> Dict[str, Any]:
        """Execute leveraged trade with margin management."""
        signal = signal_data['signal']
        signal_strength = signal_data['signal_strength']
        confidence = signal_data['confidence']
        
        trade_info = {
            'timestamp': timestamp,
            'symbol': symbol,
            'signal': signal,
            'action': 'HOLD',
            'signal_strength': signal_strength,
            'confidence': confidence,
            'price': price
        }
        
        if signal in ['BUY', 'WEAK_BUY'] and symbol not in self.positions:
            
            # Calculate position size using Kelly fraction
            position_fraction = self.calculate_kelly_position_size(signal_strength, confidence)
            
            # Calculate leveraged position
            position_value = self.capital * position_fraction * self.leverage
            position_size = position_value / price
            margin_required = self.calculate_margin_requirements(position_value)
            
            # Check margin availability
            if margin_required <= self.available_margin:
                # Apply costs
                execution_price = price * (1 + self.slippage)
                trade_value = position_size * execution_price
                commission_cost = trade_value * self.commission
                
                # Calculate ATR-based stop levels
                atr_levels = self.calculate_atr_stop_levels(df, execution_price, True)
                
                # Create leveraged position
                self.positions[symbol] = {
                    'size': position_size,
                    'entry_price': execution_price,
                    'value': trade_value,
                    'margin_used': margin_required,
                    'entry_time': timestamp,
                    'highest_price': execution_price,
                    'lowest_price': execution_price,
                    'stop_loss': atr_levels['stop_loss'],
                    'profit_target': atr_levels['profit_target'],
                    'trailing_stop_pct': atr_levels['trailing_stop'],
                    'atr_value': atr_levels.get('atr_value', 0),
                    'unrealized_pnl': 0.0  # Initialize unrealized P&L
                }
                
                # Update margin
                self.used_margin += margin_required
                self.available_margin -= margin_required
                self.capital -= commission_cost
                
                trade_info.update({
                    'action': 'BUY',
                    'position_size': position_size,
                    'trade_value': trade_value,
                    'commission': commission_cost,
                    'margin_used': margin_required,
                    'leverage': self.leverage,
                    'capital_after': self.capital,
                    'available_margin': self.available_margin
                })
        
        return trade_info
    
    def add_signal_columns_to_csv(self, csv_file_path: str, symbol: str) -> str:
        """
        Add 'target_signal' and 'target_position_size' columns to CSV file.
        
        Args:
            csv_file_path: Path to the CSV file
            symbol: Trading symbol (e.g., 'BTCUSDT')
            
        Returns:
            Path to the updated CSV file
        """
        print(f"Adding signal columns to {symbol} CSV...")
        
        # Read the CSV file
        try:
            df = pd.read_csv(csv_file_path, low_memory=False)
            print(f"   Loaded {len(df)} rows from {csv_file_path}")
        except Exception as e:
            print(f"   Error reading CSV: {e}")
            return csv_file_path
        
        # Initialize new columns
        df['target_signal'] = 'HOLD'
        df['target_position_size'] = 0.0
        
        # Process each row to generate signals and position sizes
        signals_generated = 0
        positions_calculated = 0
        
        for idx, row in df.iterrows():
            if idx % 10000 == 0 and idx > 0:
                print(f"   Processing row {idx}/{len(df)}...")
            
            try:
                # Generate signal using existing logic
                signal_data = self.generate_leveraged_signal(row, symbol)
                signal = signal_data.get('signal', 'HOLD')
                signal_strength = signal_data.get('signal_strength', 0.0)
                confidence = signal_data.get('confidence', 0.0)
                
                # Calculate position size using existing logic
                if signal != 'HOLD':
                    # Use simplified position sizing for CSV processing (no trade history available)
                    position_size = self.calculate_simple_position_size(
                        signal_strength, confidence
                    )
                    positions_calculated += 1
                else:
                    position_size = 0.0
                
                # Update the dataframe
                df.at[idx, 'target_signal'] = signal
                df.at[idx, 'target_position_size'] = position_size
                
                if signal != 'HOLD':
                    signals_generated += 1
                    
            except Exception as e:
                # If there's an error, keep default values
                df.at[idx, 'target_signal'] = 'HOLD'
                df.at[idx, 'target_position_size'] = 0.0
                continue
        
        # Create output filename
        base_name = os.path.splitext(csv_file_path)[0]
        output_path = f"{base_name}_with_signals.csv"
        
        # Save the updated CSV
        try:
            df.to_csv(output_path, index=False)
            print(f"   Updated CSV saved to: {output_path}")
            print(f"   Generated {signals_generated} signals out of {len(df)} rows")
            print(f"   Calculated {positions_calculated} position sizes")
            
            # Print signal distribution
            signal_counts = df['target_signal'].value_counts()
            print(f"   Signal distribution:")
            for signal, count in signal_counts.items():
                percentage = (count / len(df)) * 100
                print(f"      {signal}: {count} ({percentage:.1f}%)")
            
            return output_path
            
        except Exception as e:
            print(f"   Error saving CSV: {e}")
            return csv_file_path

    def add_signal_columns_to_all_csvs(self, data_dir: str = 'features_with_residuals') -> Dict[str, str]:
        """
        Add signal columns to all CSV files in the specified directory.
        
        Args:
            data_dir: Directory containing CSV files
            
        Returns:
            Dictionary mapping original file paths to updated file paths
        """
        print(f"Adding signal columns to all CSV files in {data_dir}...")
        
        if not os.path.exists(data_dir):
            print(f"   Directory {data_dir} does not exist")
            return {}
        
        # Find all CSV files
        csv_files = [f for f in os.listdir(data_dir) if f.endswith('.csv') and 'features_with_residuals' in f]
        
        if not csv_files:
            print(f"   No CSV files found in {data_dir}")
            return {}
        
        print(f"   Found {len(csv_files)} CSV files to process")
        
        updated_files = {}
        
        for csv_file in csv_files:
            # Extract symbol from filename
            symbol = csv_file.replace('_features_with_residuals.csv', '')
            
            # Full path to the CSV file
            csv_path = os.path.join(data_dir, csv_file)
            
            print(f"\nProcessing {symbol}...")
            
            # Add signal columns
            updated_path = self.add_signal_columns_to_csv(csv_path, symbol)
            updated_files[csv_path] = updated_path
        
        print(f"\nSuccessfully processed {len(updated_files)} CSV files")
        return updated_files
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all technical indicators directly from OHLCV data."""
        print("  Calculating technical indicators from OHLCV data...")
        
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # 1. Calculate base returns
        df['return_1'] = df['close'].pct_change(1)
        df['return_5'] = df['close'].pct_change(5)
        df['return_15'] = df['close'].pct_change(15)
        df['log_return'] = np.where(
            df['close'].shift(1) > 0,
            np.log(df['close'] / df['close'].shift(1)),
            0
        )
        
        # 2. Rolling Volatility
        df['rolling_vol_20'] = df['return_1'].rolling(20).std()
        df['rolling_vol_60'] = df['return_1'].rolling(60).std()
        df['rolling_vol_15m'] = df['return_1'].rolling(15).std()  # Used in volatility filter
        
        # 3. ATR% (Average True Range normalized by price)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(14).mean()
        df['atr_pct'] = (df['atr'] / df['close']) * 100
        
        # 4. RSI (14)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi_14'] = 100 - (100 / (1 + rs))
        
        # 5. MA Slopes
        df['ma_5'] = df['close'].rolling(5).mean()
        df['ma_20'] = df['close'].rolling(20).mean()
        df['ma_slope_5'] = df['ma_5'].diff() / 5
        df['ma_slope_20'] = df['ma_20'].diff() / 20
        
        # 6. Candle Ratios
        body_size = np.abs(df['close'] - df['open'])
        upper_wick = df['high'] - np.maximum(df['open'], df['close'])
        lower_wick = np.minimum(df['open'], df['close']) - df['low']
        total_range = df['high'] - df['low']
        
        df['body_ratio'] = np.where(total_range > 0, body_size / total_range, 0)
        df['upper_wick_ratio'] = np.where(total_range > 0, upper_wick / total_range, 0)
        df['lower_wick_ratio'] = np.where(total_range > 0, lower_wick / total_range, 0)
        
        # 7. Volume Ratios
        df['volume_ma_20'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']
        
        # 8. Rollups: 3 / 9 / 36 Bars (Rolling Summaries)
        windows = [3, 9, 36]
        
        for window in windows:
            # Rolling returns over each window
            df[f'return_{window}'] = df['close'].pct_change(window)
            
            # Rolling volatility over each window
            df[f'vol_{window}'] = df['return_1'].rolling(window).std()
            
            # MA trends over each window
            df[f'ma_{window}'] = df['close'].rolling(window).mean()
            df[f'ma_trend_{window}'] = df[f'ma_{window}'].diff() / window
            
            # Volume anomalies over each window
            df[f'volume_ma_{window}'] = df['volume'].rolling(window).mean()
            df[f'volume_anomaly_{window}'] = df['volume'] / df[f'volume_ma_{window}']
            
            # RSI over different horizons
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
            rs = gain / loss
            df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
        
        print("  Technical indicators calculated: RSI, ATR, MA slopes, volume ratios, rollups")
        return df

    def calculate_target_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate target features directly from OHLCV data to avoid dependency on pre-calculated target columns."""
        print("  Calculating target features directly from OHLCV data...")
        
        # Create target features (shifted forward by 1 bar to prevent lookahead bias)
        target_features = ['return_1', 'return_5', 'return_15', 'log_return']
        
        for feature in target_features:
            if feature in df.columns:
                df[f'target_{feature}'] = df[feature].shift(-1)
        
        # Create target direction (binary: 1 if next return > 0, 0 otherwise)
        # Handle NaN values at the end of the dataset
        next_return = df['return_1'].shift(-1)
        df['target_direction'] = np.where(
            pd.isna(next_return), 
            0,  # Default to 0 for NaN values (end of dataset)
            np.where(next_return > 0, 1, 0)
        )
        
        # Fill NaN values in target columns with 0 (for the last row)
        for feature in target_features:
            target_col = f'target_{feature}'
            if target_col in df.columns:
                df[target_col] = df[target_col].fillna(0)
        
        print(f"  Target features calculated: target_direction, target_return_1, target_return_5, target_return_15, target_log_return")
        print(f"  Target direction distribution: {df['target_direction'].value_counts().to_dict()}")
        return df

    def fill_technical_indicator_nans(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fill NaN values in technical indicators with appropriate defaults."""
        print("  Filling NaN values in technical indicators...")
        
        # Fill NaN values with appropriate defaults
        technical_indicators = {
            'rsi_14': 50,  # Neutral RSI
            'ma_slope_5': 0,
            'ma_slope_20': 0,
            'volume_ratio': 1,  # Neutral volume ratio
            'rolling_vol_15m': 0,
            'rolling_vol_20': 0,
            'rolling_vol_60': 0,
            'atr_pct': 0,
            'body_ratio': 0,
            'upper_wick_ratio': 0,
            'lower_wick_ratio': 0
        }
        
        for indicator, default_value in technical_indicators.items():
            if indicator in df.columns:
                df[indicator] = df[indicator].fillna(default_value)
        
        # Fill rollup indicators
        for window in [3, 9, 36]:
            for suffix in ['', '_trend']:
                col = f'ma_{window}{suffix}'
                if col in df.columns:
                    df[col] = df[col].fillna(0)
            
            for suffix in ['', '_anomaly']:
                col = f'volume_ma_{window}{suffix}'
                if col in df.columns:
                    df[col] = df[col].fillna(1 if 'anomaly' in col else 0)
            
            for col in [f'vol_{window}', f'rsi_{window}']:
                if col in df.columns:
                    df[col] = df[col].fillna(0)
        
        print("  Technical indicator NaN values filled")
        return df

    def load_cross_asset_data(self, data_dir: str = 'historical_data') -> None:
        """Load OHLCV data for all assets to enable cross-asset correlation calculation."""
        print("  Loading cross-asset data for correlation calculation...")
        
        files = [f for f in os.listdir(data_dir) if f.endswith('_historical_5m_12months.csv')]
        symbols = [f.split('_')[0] for f in files]
        
        for symbol in symbols:
            file_path = os.path.join(data_dir, f"{symbol}_historical_5m_12months.csv")
            try:
                df = pd.read_csv(file_path)
                
                # Get timestamp column
                timestamp_col = self.get_timestamp_column(df)
                df['timestamp'] = pd.to_datetime(df[timestamp_col])
                df = df.sort_values('timestamp').reset_index(drop=True)
                
                # Calculate returns
                df['return_1'] = df['close'].pct_change(1)
                
                # Store data
                self.cross_asset_data[symbol] = df
                self.cross_asset_returns[symbol] = df['return_1'].values
                
            except Exception as e:
                print(f"    Warning: Could not load {symbol}: {e}")
        
        print(f"  Loaded cross-asset data for {len(self.cross_asset_data)} symbols")

    def calculate_cross_asset_correlations(self, current_symbol: str, current_idx: int) -> Dict[str, float]:
        """Calculate cross-asset correlations for the current symbol at the given index."""
        correlations = {}
        
        if current_symbol not in self.cross_asset_returns:
            return correlations
        
        # Get current symbol's returns for the lookback period
        start_idx = max(0, current_idx - self.cross_asset_lookback)
        end_idx = current_idx + 1
        
        current_returns = self.cross_asset_returns[current_symbol][start_idx:end_idx]
        
        # Calculate correlation with other assets
        for symbol, returns in self.cross_asset_returns.items():
            if symbol != current_symbol:
                try:
                    # Get corresponding returns for the other asset
                    other_returns = returns[start_idx:end_idx]
                    
                    # Ensure both arrays have the same length and no NaN values
                    min_len = min(len(current_returns), len(other_returns))
                    if min_len < 10:  # Need at least 10 data points
                        continue
                    
                    current_clean = current_returns[-min_len:]
                    other_clean = other_returns[-min_len:]
                    
                    # Remove NaN values
                    mask = ~(np.isnan(current_clean) | np.isnan(other_clean))
                    if np.sum(mask) < 10:  # Need at least 10 valid data points
                        continue
                    
                    current_clean = current_clean[mask]
                    other_clean = other_clean[mask]
                    
                    # Calculate correlation
                    if len(current_clean) > 1 and np.std(current_clean) > 0 and np.std(other_clean) > 0:
                        correlation = np.corrcoef(current_clean, other_clean)[0, 1]
                        if not np.isnan(correlation):
                            correlations[f'corr_return_1_vs_{symbol}'] = correlation
                            
                except Exception as e:
                    continue  # Skip this asset if calculation fails
        
        return correlations

    def calculate_cross_asset_residuals(self, current_symbol: str, current_idx: int) -> Dict[str, float]:
        """Calculate cross-asset cointegration residuals for the current symbol at the given index."""
        residuals = {}
        
        if current_symbol not in self.cross_asset_returns:
            return residuals
        
        # Get current symbol's returns for the lookback period
        start_idx = max(0, current_idx - self.cross_asset_lookback)
        end_idx = current_idx + 1
        
        current_returns = self.cross_asset_returns[current_symbol][start_idx:end_idx]
        
        # Calculate residuals with other assets
        for symbol, returns in self.cross_asset_returns.items():
            if symbol != current_symbol:
                try:
                    # Get corresponding returns for the other asset
                    other_returns = returns[start_idx:end_idx]
                    
                    # Ensure both arrays have the same length
                    min_len = min(len(current_returns), len(other_returns))
                    if min_len < 20:  # Need at least 20 data points for cointegration
                        continue
                    
                    current_clean = current_returns[-min_len:]
                    other_clean = other_returns[-min_len:]
                    
                    # Remove NaN values
                    mask = ~(np.isnan(current_clean) | np.isnan(other_clean))
                    if np.sum(mask) < 20:  # Need at least 20 valid data points
                        continue
                    
                    current_clean = current_clean[mask]
                    other_clean = other_clean[mask]
                    
                    # Simple cointegration residual calculation
                    # Use linear regression to find the relationship
                    if len(current_clean) > 1 and np.std(other_clean) > 0:
                        # Calculate beta (slope) using least squares
                        beta = np.cov(current_clean, other_clean)[0, 1] / np.var(other_clean)
                        
                        # Calculate residual (current - beta * other)
                        residual = current_clean[-1] - beta * other_clean[-1]
                        
                        # Normalize by standard deviation of residuals
                        if len(current_clean) > 10:
                            all_residuals = current_clean - beta * other_clean
                            residual_std = np.std(all_residuals)
                            if residual_std > 0:
                                normalized_residual = residual / residual_std
                                residuals[f'residual_return_1_vs_{symbol}'] = normalized_residual
                                
                except Exception as e:
                    continue  # Skip this asset if calculation fails
        
        return residuals

    def backtest_symbol(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Run leveraged backtest."""
        print(f"Leveraged backtesting {symbol} (5x leverage)...")
        
        self.reset()
        
        # Prepare data
        try:
            timestamp_col = self.get_timestamp_column(df)
        except ValueError as e:
            return {'error': str(e)}
        
        # Only require basic OHLCV columns - we'll calculate everything ourselves
        required_cols = [timestamp_col, 'open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            return {'error': f'Missing required columns: {missing_cols}'}
        
        df['timestamp'] = pd.to_datetime(df[timestamp_col])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Handle different timestamp formats
        try:
            df['formatted_timestamp'] = df['timestamp'].dt.strftime('%H:%M %Y/%m/%d')
        except:
            # Fallback for different date formats
            df['formatted_timestamp'] = df['timestamp'].astype(str)
        
        # Calculate all technical indicators directly from OHLCV data
        df = self.calculate_technical_indicators(df)
        
        # Fill NaN values in technical indicators
        df = self.fill_technical_indicator_nans(df)
        
        # Calculate target features directly from OHLCV data
        df = self.calculate_target_features(df)
        
        # Track portfolio values
        initial_portfolio_value = self.capital
        peak_portfolio_value = self.capital
        
        # Run backtest
        for idx, row in df.iterrows():
            timestamp = row['formatted_timestamp']
            price = row['close']
            high = row['high']
            low = row['low']
            
            # Check for margin call
            if self.check_margin_call():
                liquidated_trades = self.force_liquidate_positions(timestamp)
                self.trades.extend(liquidated_trades)
                continue
            
            # Update all positions with current market values and unrealized P&L
            for pos_symbol, position in self.positions.items():
                if pos_symbol == symbol:  # Only update the current symbol's position
                    current_market_value = position['size'] * price
                    entry_cost = position['size'] * position['entry_price']
                    unrealized_pnl = current_market_value - entry_cost
                    
                    position['value'] = current_market_value
                    position['unrealized_pnl'] = unrealized_pnl
            
            # Manage leveraged positions (pass row data for trend-following exits)
            exit_trade = self.manage_leveraged_position(symbol, price, high, low, timestamp, row)
            if exit_trade:
                self.trades.append(exit_trade)
                continue
            
            # Track filter rejections for analysis
            debug_info = {
                'timestamp': timestamp,
                'symbol': symbol,
                'idx': idx,
                'rejected_by': []
            }
            
            # Check volatility filter before generating signal
            if not self.check_volatility_filter(df, idx, timestamp):
                debug_info['rejected_by'].append('volatility_filter')
                if idx % 10000 == 0:  # Log every 10k iterations
                    print(f"  DEBUG {symbol} idx={idx}: Rejected by volatility_filter")
                continue  # Skip trading during extreme noise or low liquidity
            
            # Check volatility regime filter before generating signal
            if not self.check_volatility_regime_filter(row):
                debug_info['rejected_by'].append('volatility_regime_filter')
                if idx % 10000 == 0:  # Log every 10k iterations
                    print(f"  DEBUG {symbol} idx={idx}: Rejected by volatility_regime_filter")
                continue  # Skip trading during high volatility regimes
            
            # Check drawdown limit - halt trading if exceeded
            if self.check_drawdown_limit():
                debug_info['rejected_by'].append('drawdown_limit')
                if idx % 10000 == 0:  # Log every 10k iterations
                    print(f"  DEBUG {symbol} idx={idx}: Rejected by drawdown_limit")
                self.trading_halted = True
                continue  # Skip trading if drawdown limit exceeded
            
            # Check daily stop-loss - halt trading if exceeded
            current_portfolio_value = self.capital + sum(pos.get('unrealized_pnl', 0) for pos in self.positions.values())
            if self.check_daily_stop_loss(timestamp, current_portfolio_value):
                debug_info['rejected_by'].append('daily_stop_loss')
                if idx % 10000 == 0:  # Log every 10k iterations
                    print(f"  DEBUG {symbol} idx={idx}: Rejected by daily_stop_loss")
                self.trading_halted = True
                continue  # Skip trading if daily stop-loss exceeded
            
            # Generate leveraged signal
            signal_data = self.generate_leveraged_signal(row, symbol)
            
            # Log signal generation progress
            if idx % 10000 == 0:  # Log every 10k iterations
                print(f"  DEBUG {symbol} idx={idx}: Signal={signal_data['signal']}, Strength={signal_data['signal_strength']:.3f}, Confidence={signal_data['confidence']:.3f}")
            
            # Track signal rejections
            if signal_data['signal'] == 'HOLD':
                debug_info['rejected_by'].append('signal_generation')
                if idx % 10000 == 0:  # Log every 10k iterations
                    print(f"  DEBUG {symbol} idx={idx}: Rejected by signal_generation (HOLD)")
            
            # Execute leveraged trade
            trade_info = self.execute_leveraged_trade(timestamp, symbol, signal_data, price, df)
            if trade_info['action'] != 'HOLD':
                self.trades.append(trade_info)
            
            # Update portfolio tracking - FIXED: Use unrealized P&L instead of incorrect margin calculation
            current_portfolio_value = self.capital + sum(pos.get('unrealized_pnl', 0) for pos in self.positions.values())
            if current_portfolio_value > peak_portfolio_value:
                peak_portfolio_value = current_portfolio_value
            self.portfolio_values.append(current_portfolio_value)
        
            # Update drawdown tracking
            self.update_drawdown_tracking(current_portfolio_value)
            
        
        # Calculate final metrics - FIXED: Use unrealized P&L instead of incorrect margin calculation
        final_portfolio_value = self.capital + sum(pos.get('unrealized_pnl', 0) for pos in self.positions.values())
        total_return = (final_portfolio_value - initial_portfolio_value) / initial_portfolio_value
        
        # Calculate performance metrics
        sharpe_ratio = 0.0
        if len(self.portfolio_values) > 1:
            returns = pd.Series(self.portfolio_values).pct_change().dropna()
            if len(returns) > 0 and returns.std() > 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        
        max_drawdown = 0.0
        if len(self.portfolio_values) > 0:
            peak = np.maximum.accumulate(self.portfolio_values)
            drawdown = (self.portfolio_values - peak) / peak
            max_drawdown = np.min(drawdown)
        
        win_rate = self.winning_trades / self.total_trades if self.total_trades > 0 else 0
        
        # Calculate average gain per trade (key metric)
        avg_gain_per_trade_pct = 0.0
        trade_returns = []
        for trade in self.trades:
            if trade.get('action') == 'SELL' and 'trade_return_pct' in trade:
                trade_returns.append(trade['trade_return_pct'])
        
        if trade_returns:
            avg_gain_per_trade_pct = np.mean(trade_returns)
        
        # MFE/MAE analysis
        mfe_values = [trade.get('mfe', 0) for trade in self.trades if trade.get('action') == 'SELL']
        mae_values = [trade.get('mae', 0) for trade in self.trades if trade.get('action') == 'SELL']
        avg_mfe = np.mean(mfe_values) if mfe_values else 0
        avg_mae = np.mean(mae_values) if mae_values else 0
        
        # Calculate profit factor and related metrics
        profit_factor_metrics = calculate_profit_factor(self.trades)
        
        results = {
            'symbol': symbol,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'avg_gain_per_trade_pct': avg_gain_per_trade_pct,
            'avg_mfe': avg_mfe,
            'avg_mae': avg_mae,
            'margin_calls': self.margin_calls,
            'leverage': self.leverage,
            'trades': self.trades.copy(),
            # Profit factor metrics
            'profit_factor': profit_factor_metrics['profit_factor'],
            'gross_profit': profit_factor_metrics['gross_profit'],
            'gross_loss': profit_factor_metrics['gross_loss'],
            'total_profit_trades': profit_factor_metrics['total_profit_trades'],
            'total_loss_trades': profit_factor_metrics['total_loss_trades'],
            'largest_win': profit_factor_metrics['largest_win'],
            'largest_loss': profit_factor_metrics['largest_loss'],
            'avg_win': profit_factor_metrics['avg_win'],
            'avg_loss': profit_factor_metrics['avg_loss'],
            'win_loss_ratio': profit_factor_metrics['win_loss_ratio']
        }
        
        # Success criteria removed - no longer evaluating performance against thresholds
        
        # Print comprehensive results summary
        self._print_backtest_results(symbol, total_return, sharpe_ratio, max_drawdown, 
                                   win_rate, avg_gain_per_trade_pct, avg_mfe, avg_mae, 
                                   profit_factor_metrics)
        return results

    def _print_backtest_results(self, symbol: str, total_return: float, sharpe_ratio: float, 
                              max_drawdown: float, win_rate: float, avg_gain_per_trade_pct: float,
                              avg_mfe: float, avg_mae: float, 
                              profit_factor_metrics: Dict[str, float]) -> None:
        """Print formatted backtest results summary."""
        print(f"  {symbol} leveraged backtest completed:")
        print(f"    Total Return: {total_return:.2%}")
        print(f"    Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"    Max Drawdown: {max_drawdown:.2%}")
        print(f"    Total Trades: {self.total_trades}")
        print(f"    Win Rate: {win_rate:.2%}")
        print(f"    Avg Gain per Trade: {avg_gain_per_trade_pct:.4f}%")
        print(f"    Avg MFE: {avg_mfe:.4f}")
        print(f"    Avg MAE: {avg_mae:.4f}")
        print(f"    Margin Calls: {self.margin_calls}")
        
        # Profit Factor metrics
        profit_factor = profit_factor_metrics['profit_factor']
        if profit_factor == float('inf'):
            print(f"    Profit Factor: ∞ (No losses)")
        else:
            print(f"    Profit Factor: {profit_factor:.2f}")
        
        print(f"    Gross Profit: ${profit_factor_metrics['gross_profit']:.2f}")
        print(f"    Gross Loss: ${profit_factor_metrics['gross_loss']:.2f}")
        print(f"    Largest Win: ${profit_factor_metrics['largest_win']:.2f}")
        print(f"    Largest Loss: ${profit_factor_metrics['largest_loss']:.2f}")
        print(f"    Win/Loss Ratio: {profit_factor_metrics['win_loss_ratio']:.2f}")


def main():
    """
    Main function to run leveraged backtesting using existing data files.
    
    Executes comprehensive backtesting with the following strategy:
    - 5x leverage for maximum returns
    - Balanced allocation: 5.0-10.0% per trade (50-100% exposure)
    - Drawdown limit: Stop trading if portfolio drawdown > 1.5%
    - Daily stop-loss: Stop trading if daily loss > 1%
    - Cross-asset confirmation: At least 1 token must confirm signal
    - Cointegration filter: Trade only if residual > 1.5σ
    - Volatility regime filter: Skip trades if ATR > 67.0% or rolling_vol outside 0.0-5.0% range
    - ATR-based stops: 0.6× ATR(14) stop loss for risk management
    - Kelly fraction position sizing: Optimizes between 5.0-10.0% positions based on performance
    - Entry refinement: Signal strength ≥ 40%, Confidence ≥ 30% with enhanced cross-asset analysis
    - Exit strategy: TREND-FOLLOWING - Hold positions until trend reversal signals
    """
    print("Leveraged Trading Signal Backtest System (5x Leverage)")
    print("=" * 60)
    print("DATA LOADING FROM EXISTING FILES:")
    print("  • 5x leverage for maximum returns")
    print("  • Balanced allocation: 5.0-10.0% per trade (50-100% exposure)")
    print("  • Drawdown limit: Stop trading if portfolio drawdown > 1.5% (strict risk control)")
    print("  • Daily stop-loss: Stop trading if daily loss > 1%")
    print("  • Cross-asset confirmation: At least 1 token must confirm signal")
    print("  • Cointegration filter: Trade only if residual > 1.5σ")
    print("  • Volatility regime filter: Skip trades if ATR > 67.0% or rolling_vol outside 0.0-5.0% range")
    print("  • Volatility filter: Disabled for maximum trade frequency")
    print("  • ATR-based stops: 0.6× ATR(14) stop loss for risk management")
    print("  • Kelly fraction position sizing: Optimizes between 5.0-10.0% positions based on performance")
    print("  • Entry refinement: Signal strength ≥ 40%, Confidence ≥ 30% with enhanced cross-asset analysis")
    print("  • Exit strategy: TREND-FOLLOWING - Hold positions until trend reversal signals")
    print("  • Data source: Parquet files for BTC/ETH, CSV files for other symbols")
    print("=" * 60)
    
    # Interactive asset selection
    selected_symbol = select_asset()
    
    # Interactive backtest period selection
    backtest_start, backtest_end = get_backtest_period()
    
    # Load data from parquet files for BTC and ETH, fallback to CSV for others
    if selected_symbol in ['BTCUSDT', 'ETHUSDT']:
        print(f"\nLoading {selected_symbol} data from parquet files...")
        df = load_data_from_parquet(selected_symbol, 'data')
        if df.empty:
            print(f"Failed to load {selected_symbol} from parquet. Please check the data directory.")
            return
    else:
        # Load data from CSV for other symbols
    csv_file_path = os.path.join('features', f"{selected_symbol}.csv")
    if not os.path.exists(csv_file_path):
        print(f"CSV file not found: {csv_file_path}")
        print("Please ensure the features directory contains the required CSV files.")
        return
    
    try:
        print(f"Loading complete dataset from {csv_file_path}...")
        df = pd.read_csv(csv_file_path, low_memory=False)
        print(f"Successfully loaded {len(df)} rows of {selected_symbol} data from CSV")
    except Exception as e:
        print(f"Error loading CSV data: {e}")
        return
    
    # Filter data by selected backtest period
    df = filter_data_by_period(df, backtest_start, backtest_end)
    
    # Initialize backtest engine
    engine = LeveragedBacktestEngine()
    
    # Load cross-asset data for correlation analysis
    print("\nLoading cross-asset data for correlation analysis...")
    cross_asset_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
    
    for cross_symbol in cross_asset_symbols:
        if cross_symbol != selected_symbol:
            # Try parquet files first for BTC and ETH
            if cross_symbol in ['BTCUSDT', 'ETHUSDT']:
                try:
                    print(f"  Loading {cross_symbol} from parquet...")
                    cross_df = load_data_from_parquet(cross_symbol, 'data')
                    if not cross_df.empty:
                        # Store cross-asset data for correlation calculation
                        engine.cross_asset_data[cross_symbol] = cross_df
                        engine.cross_asset_returns[cross_symbol] = cross_df['close'].pct_change(1).values
                except Exception as e:
                    print(f"  Error loading {cross_symbol} from parquet: {e}")
            else:
                # Fallback to CSV for other symbols
            cross_csv_path = os.path.join('features', f"{cross_symbol}.csv")
            if os.path.exists(cross_csv_path):
                try:
                    print(f"  Loading {cross_symbol} from CSV...")
                    cross_df = pd.read_csv(cross_csv_path, low_memory=False)
                    if not cross_df.empty:
                        # Store cross-asset data for correlation calculation
                        engine.cross_asset_data[cross_symbol] = cross_df
                        engine.cross_asset_returns[cross_symbol] = cross_df['close'].pct_change(1).values
                except Exception as e:
                    print(f"  Error loading {cross_symbol}: {e}")
            else:
                    print(f"  {cross_symbol} data not found, skipping...")
    
    print(f"Loaded cross-asset data for {len(engine.cross_asset_data)} symbols")
    
    # Run backtest using loaded data
    data_source = "parquet" if selected_symbol in ['BTCUSDT', 'ETHUSDT'] else "CSV"
    print(f"\nStarting leveraged backtest for {selected_symbol} using {data_source} data...")
    results = engine.backtest_symbol(df, selected_symbol)
    
    if 'error' in results:
        print(f"Backtest failed: {results['error']}")
        return
    
    # Save trades to CSV
    if results.get('trades'):
        csv_path = save_trades_to_csv(results['trades'], selected_symbol, results)
        print(f"\nTrades exported to: {csv_path}")
    
    # Print results summary
    print("\n" + "=" * 60)
    print("LEVERAGED BACKTEST RESULTS")
    print("=" * 60)
    
    print(f"Symbol: {selected_symbol}")
    print(f"Total Data Points: {len(df)}")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Avg Gain per Trade: {results['avg_gain_per_trade_pct']:.4f}%")
    print(f"Avg MFE: {results['avg_mfe']:.4f}")
    print(f"Avg MAE: {results['avg_mae']:.4f}")
    print(f"Margin Calls: {results['margin_calls']}")
    print(f"Leverage: {results['leverage']}x")
    
    # Profit Factor metrics
    profit_factor = results.get('profit_factor', 0)
    if profit_factor == float('inf'):
        print(f"Profit Factor: ∞ (No losses)")
    else:
        print(f"Profit Factor: {profit_factor:.2f}")
    
    print(f"Gross Profit: ${results.get('gross_profit', 0):.2f}")
    print(f"Gross Loss: ${results.get('gross_loss', 0):.2f}")
    print(f"Largest Win: ${results.get('largest_win', 0):.2f}")
    print(f"Largest Loss: ${results.get('largest_loss', 0):.2f}")
    print(f"Win/Loss Ratio: {results.get('win_loss_ratio', 0):.2f}")
    
    print("=" * 60)
    
    print(f"\nLeveraged backtest completed!")
    if results.get('trades'):
        print(f"Trades saved to: backtest_trades/")
    print(f"Data source: CSV files with incremental updates")
    print(f"Timeframe: 5-minute candles")
    print(f"Features calculated: Technical indicators, cross-asset correlations, target signals")


def clean_all_csv_files():
    """
    Clean all existing CSV files in the features directory.
    """
    import os
    import glob
    
    features_dir = 'features'
    if not os.path.exists(features_dir):
        print(f"Features directory '{features_dir}' not found.")
        return
    
    # Find all CSV files in features directory
    csv_files = glob.glob(os.path.join(features_dir, '*.csv'))
    
    if not csv_files:
        print("No CSV files found in features directory.")
        return
    
    print(f"Found {len(csv_files)} CSV files to clean:")
    for csv_file in csv_files:
        print(f"  - {os.path.basename(csv_file)}")
    
    print("\nCleaning CSV files...")
    
    cleaned_count = 0
    for csv_file in csv_files:
        symbol = os.path.basename(csv_file).replace('.csv', '')
        if clean_existing_csv_file(csv_file, symbol):
            cleaned_count += 1
    
    print(f"\nCleaning complete: {cleaned_count}/{len(csv_files)} files cleaned successfully.")

def add_signals_to_csvs():
    """
    Add signal columns to all CSV files in the features directory.
    
    This function processes all CSV files in the 'features_with_residuals' directory
    and adds two new columns:
    - target_signal: Trading signal (BUY, SELL, HOLD, WEAK_BUY, WEAK_SELL)
    - target_position_size: Recommended position size as fraction of capital
    
    Returns:
        None
    """
    print("Adding Signal Columns to CSV Files")
    print("=" * 50)
    
    # Initialize the backtest engine
    engine = LeveragedBacktestEngine()
    
    # Add signal columns to all CSV files
    updated_files = engine.add_signal_columns_to_all_csvs('features_with_residuals')
    
    if updated_files:
        print(f"\nSummary of Updated Files:")
        for original_path, updated_path in updated_files.items():
            symbol = os.path.basename(original_path).replace('_features_with_residuals.csv', '')
            print(f"   {symbol}: {updated_path}")
        
        print(f"\nSuccessfully added signal columns to {len(updated_files)} files")
        print("New files have '_with_signals' suffix")
        print("New columns added: 'target_signal', 'target_position_size'")
    else:
        print("No files were processed")


def run_with_command_line():
    """
    Run the backtest with command-line arguments for automation.
    """
    parser = argparse.ArgumentParser(description='Leveraged Cryptocurrency Backtest System')
    parser.add_argument('--asset', type=str, choices=list(AVAILABLE_ASSETS.keys()), 
                       help='Asset to backtest (BTC, BNB, ETH, SOL, XRP, AVAX, ADA, DOT, LTC, LINK)')
    parser.add_argument('--start-date', type=str, 
                       help='Start date in YYYY-MM-DD format')
    parser.add_argument('--end-date', type=str, 
                       help='End date in YYYY-MM-DD format')
    parser.add_argument('--backtest-start', type=str,
                       help='Backtest start date in YYYY-MM-DD format')
    parser.add_argument('--backtest-end', type=str,
                       help='Backtest end date in YYYY-MM-DD format')
    parser.add_argument('--add-signals', action='store_true',
                       help='Add signal columns to CSV files instead of running backtest')
    parser.add_argument('--clean-csv', action='store_true',
                       help='Clean existing CSV files by removing corrupted data')
    
    args = parser.parse_args()
    
    if args.add_signals:
        add_signals_to_csvs()
        return
    
    if args.clean_csv:
        clean_all_csv_files()
        return
    
    # Use command line arguments if provided, otherwise use interactive mode
    if args.asset:
        selected_symbol = AVAILABLE_ASSETS[args.asset]
        
        print(f"Running backtest with command-line arguments:")
        print(f"  Asset: {args.asset} ({selected_symbol})")
        data_source = "parquet files" if selected_symbol in ['BTCUSDT', 'ETHUSDT'] else "CSV files"
        print(f"  Mode: Loading from {data_source}")
        
        # Load data from parquet files for BTC and ETH, fallback to CSV for others
        if selected_symbol in ['BTCUSDT', 'ETHUSDT']:
            print(f"\nLoading {selected_symbol} data from parquet files...")
            df = load_data_from_parquet(selected_symbol, 'data')
            if df.empty:
                print(f"Failed to load {selected_symbol} from parquet. Please check the data directory.")
                return
        else:
            # Load data from CSV for other symbols
        csv_file_path = os.path.join('features', f"{selected_symbol}.csv")
        if not os.path.exists(csv_file_path):
            print(f"CSV file not found: {csv_file_path}")
            print("Please ensure the features directory contains the required CSV files.")
            return
        
        try:
            print(f"Loading complete dataset from {csv_file_path}...")
            df = pd.read_csv(csv_file_path, low_memory=False)
            print(f"Successfully loaded {len(df)} rows of {selected_symbol} data from CSV")
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            return
        
        # Filter data by backtest period if specified
        backtest_start = args.backtest_start or args.start_date
        backtest_end = args.backtest_end or args.end_date
        df = filter_data_by_period(df, backtest_start, backtest_end)
        
        # Initialize backtest engine
        engine = LeveragedBacktestEngine()
        
        # Load cross-asset data for correlation analysis
        print("\nLoading cross-asset data for correlation analysis...")
        cross_asset_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
        
        for cross_symbol in cross_asset_symbols:
            if cross_symbol != selected_symbol:
                # Try parquet files first for BTC and ETH
                if cross_symbol in ['BTCUSDT', 'ETHUSDT']:
                    try:
                        print(f"  Loading {cross_symbol} from parquet...")
                        cross_df = load_data_from_parquet(cross_symbol, 'data')
                        if not cross_df.empty:
                            # Store cross-asset data for correlation calculation
                            engine.cross_asset_data[cross_symbol] = cross_df
                            engine.cross_asset_returns[cross_symbol] = cross_df['close'].pct_change(1).values
                    except Exception as e:
                        print(f"  Error loading {cross_symbol} from parquet: {e}")
                else:
                    # Fallback to CSV for other symbols
                cross_csv_path = os.path.join('features', f"{cross_symbol}.csv")
                if os.path.exists(cross_csv_path):
                    try:
                        print(f"  Loading {cross_symbol} from CSV...")
                        cross_df = pd.read_csv(cross_csv_path, low_memory=False)
                        if not cross_df.empty:
                            # Store cross-asset data for correlation calculation
                            engine.cross_asset_data[cross_symbol] = cross_df
                            engine.cross_asset_returns[cross_symbol] = cross_df['close'].pct_change(1).values
                    except Exception as e:
                        print(f"  Error loading {cross_symbol}: {e}")
                else:
                        print(f"  {cross_symbol} data not found, skipping...")
        
        print(f"Loaded cross-asset data for {len(engine.cross_asset_data)} symbols")
        
        # Run backtest using loaded data
        data_source = "parquet" if selected_symbol in ['BTCUSDT', 'ETHUSDT'] else "CSV"
        print(f"\nStarting leveraged backtest for {selected_symbol} using {data_source} data...")
        results = engine.backtest_symbol(df, selected_symbol)
        
        if 'error' in results:
            print(f"Backtest failed: {results['error']}")
            return
        
        # Save trades to CSV
        if results.get('trades'):
            csv_path = save_trades_to_csv(results['trades'], selected_symbol, results)
            print(f"\nTrades exported to: {csv_path}")
        
        # Print results summary
        print("\n" + "=" * 60)
        print("LEVERAGED BACKTEST RESULTS")
        print("=" * 60)
        
        print(f"Symbol: {selected_symbol}")
        print(f"Total Data Points: {len(df)}")
        print(f"Total Return: {results['total_return']:.2%}")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
        print(f"Max Drawdown: {results['max_drawdown']:.2%}")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.2%}")
        print(f"Avg Gain per Trade: {results['avg_gain_per_trade_pct']:.4f}%")
        print(f"Avg MFE: {results['avg_mfe']:.4f}")
        print(f"Avg MAE: {results['avg_mae']:.4f}")
        print(f"Margin Calls: {results['margin_calls']}")
        print(f"Leverage: {results['leverage']}x")
        
        # Profit Factor metrics
        profit_factor = results.get('profit_factor', 0)
        if profit_factor == float('inf'):
            print(f"Profit Factor: ∞ (No losses)")
        else:
            print(f"Profit Factor: {profit_factor:.2f}")
        
        print(f"Gross Profit: ${results.get('gross_profit', 0):.2f}")
        print(f"Gross Loss: ${results.get('gross_loss', 0):.2f}")
        print(f"Largest Win: ${results.get('largest_win', 0):.2f}")
        print(f"Largest Loss: ${results.get('largest_loss', 0):.2f}")
        print(f"Win/Loss Ratio: {results.get('win_loss_ratio', 0):.2f}")
        
        print("=" * 60)
        
        print(f"\nLeveraged backtest completed!")
        if results.get('trades'):
            print(f"Trades saved to: backtest_trades/")
        print(f"Data source: CSV files with incremental updates")
        print(f"Timeframe: 5-minute candles")
        print(f"Features calculated: Technical indicators, cross-asset correlations, target signals")
    else:
        # Run in interactive mode
        main()

if __name__ == "__main__":
    import sys
    
    # Check if user wants to add signal columns
    if len(sys.argv) > 1 and sys.argv[1] == "--add-signals":
        add_signals_to_csvs()
    else:
        run_with_command_line()
