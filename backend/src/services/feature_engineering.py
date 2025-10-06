import pandas as pd
import numpy as np
import os
import glob
from datetime import datetime, timedelta
import requests
import logging
import warnings
from typing import Optional, List, Dict, Any
from pathlib import Path
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """
    Feature engineering for crypto trading data with comprehensive technical indicators,
    regime detection, and target generation with proper data leakage prevention.
    """
    
    def __init__(self, data_folder="src/data/parquet", output_folder="src/data/features", log_level="INFO", 
                 enable_extra_features=False):
        self.data_folder = data_folder
        self.output_folder = output_folder
        self.fear_greed_cache = {}
        self.enable_extra_features = enable_extra_features  # Feature flag for 50+ extra indicators
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create output directory
        os.makedirs(output_folder, exist_ok=True)
        
        # MVP Feature parameters (enabled by default)
        self.atr_period = 14
        self.sma_periods = [20, 100]  # MVP: SMA(20/100)
        self.rsi_period = 14  # MVP: RSI(14)
        self.volume_zscore_period = 20  # MVP: volume z-score
        self.log_return_periods = [1]  # MVP: log_return_1 (always enabled)
        
        # Extra feature parameters (behind feature flag)
        if self.enable_extra_features:
            self.log_return_periods.extend([3, 6, 12])  # Add extra return periods
        self.rolling_std_period = 20
        
    def load_ohlcv_data(self, symbol="BTCUSDT", days_back=30, interval="1m"):
        """Load OHLCV data from partitioned parquet files"""
        # Try partitioned files first
        base_path = Path(self.data_folder) / "exchange=bybit" / "market=linear_perp" / f"symbol={symbol}" / f"timeframe={interval}"
        
        if base_path.exists():
            try:
                # Find all date directories
                date_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('date=')]
                if date_dirs:
                    # Load data from all partitions
                    all_data = []
                    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days_back) if days_back else None
                    
                    for date_dir in sorted(date_dirs):
                        file_path = date_dir / "bars.parquet"
                        if file_path.exists():
                            df = pd.read_parquet(file_path)
                            
                            # Filter data for the specified number of days back
                            if cutoff_date:
                                df = df[df['open_time'] >= cutoff_date]
                            
                            if not df.empty:
                                all_data.append(df)
                    
                    if all_data:
                        combined_df = pd.concat(all_data, ignore_index=True)
                        combined_df = combined_df.sort_values('open_time')
                        
                        # Ensure numeric types
                        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume',
                                       'mark_open', 'mark_high', 'mark_low', 'mark_close',
                                       'index_open', 'index_high', 'index_low', 'index_close']
                        for col in numeric_cols:
                            if col in combined_df.columns:
                                combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
                        
                        # Remove any rows with NaN values in critical columns
                        combined_df = combined_df.dropna(subset=['open', 'high', 'low', 'close', 'volume'])
                        
                        print(f"Loaded {len(combined_df)} OHLCV records for {symbol} from partitioned files")
                        return combined_df
            except Exception as e:
                print(f"Error loading partitioned files: {e}")
        
        # Fallback to old file pattern
        new_pattern_file = os.path.join(self.data_folder, f"{symbol}_{interval}_ohlcv.parquet")
        
        if os.path.exists(new_pattern_file):
            try:
                df = pd.read_parquet(new_pattern_file)
                if not df.empty:
                    # Filter to recent data based on days_back
                    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days_back)
                    df = df[df['open_time'] >= cutoff_date]
                    
                    # Ensure numeric types
                    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
                    for col in numeric_cols:
                        if col in df.columns:
                            df[col] = pd.to_numeric(df[col], errors='coerce')
                    
                    # Remove any rows with NaN values in critical columns
                    critical_cols = [col for col in ['open', 'high', 'low', 'close', 'volume'] if col in df.columns]
                    df = df.dropna(subset=critical_cols)
                    
                    print(f"Loaded {len(df)} OHLCV records for {symbol} from {new_pattern_file}")
                    return df
            except Exception as e:
                print(f"Error loading file {new_pattern_file}: {e}")
        
        # Fallback to old pattern for backward compatibility
        pattern = os.path.join(self.data_folder, f"ohlcv_{symbol}_{interval}_*.parquet")
        files = glob.glob(pattern)
        
        if not files:
            raise FileNotFoundError(f"No OHLCV data found for {symbol} in {self.data_folder}")
        
        # Sort files by timestamp (filename contains timestamp)
        files.sort()
        
        # Load recent files
        recent_files = files[-days_back:] if len(files) > days_back else files
        
        dfs = []
        for file in recent_files:
            try:
                df = pd.read_parquet(file)
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                print(f"Error loading {file}: {e}")
        
        if not dfs:
            raise ValueError("No valid OHLCV data loaded")
        
        # Combine and clean data
        df = pd.concat(dfs, ignore_index=True)
        df = df.drop_duplicates(subset=['open_time']).sort_values('open_time')
        df = df.reset_index(drop=True)
        
        # Ensure numeric types
        numeric_cols = ['open', 'high', 'low', 'close', 'volume', 'quote_volume']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values in critical columns
        critical_cols = [col for col in ['open', 'high', 'low', 'close', 'volume'] if col in df.columns]
        df = df.dropna(subset=critical_cols)
        
        print(f"Loaded {len(df)} OHLCV records for {symbol} from legacy files")
        return df
    
    def load_funding_data(self, symbol="BTCUSDT", days_back=30):
        """Load funding rate data from partitioned parquet files"""
        # Try partitioned files first
        base_path = Path(self.data_folder) / "exchange=bybit" / "market=linear_perp" / f"symbol={symbol}" / "funding"
        
        if base_path.exists():
            try:
                # Find all date directories
                date_dirs = [d for d in base_path.iterdir() if d.is_dir() and d.name.startswith('date=')]
                if date_dirs:
                    # Load data from all partitions
                    all_data = []
                    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days_back) if days_back else None
                    
                    for date_dir in sorted(date_dirs):
                        file_path = date_dir / "funding.parquet"
                        if file_path.exists():
                            df = pd.read_parquet(file_path)
                            
                            # Filter data for the specified number of days back
                            if cutoff_date:
                                df = df[df['fundingTime'] >= cutoff_date]
                            
                            if not df.empty:
                                all_data.append(df)
                    
                    if all_data:
                        combined_df = pd.concat(all_data, ignore_index=True)
                        combined_df = combined_df.sort_values('fundingTime')
                        
                        # Clean and process data
                        combined_df = combined_df.drop_duplicates(subset=['fundingTime'])
                        combined_df['fundingRate'] = pd.to_numeric(combined_df['fundingRate'], errors='coerce')
                        combined_df = combined_df.dropna(subset=['fundingRate'])
                        
                        print(f"Loaded {len(combined_df)} funding rate records for {symbol} from partitioned files")
                        return combined_df
            except Exception as e:
                print(f"Error loading partitioned funding files: {e}")
        
        # Fallback to old file pattern
        new_pattern_file = os.path.join(self.data_folder, f"{symbol}_funding.parquet")
        
        if os.path.exists(new_pattern_file):
            try:
                df = pd.read_parquet(new_pattern_file)
                if not df.empty:
                    # Filter to recent data based on days_back
                    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days_back)
                    df = df[df['fundingTime'] >= cutoff_date]
                    
                    # Clean and process data
                    df = df.drop_duplicates(subset=['fundingTime']).sort_values('fundingTime')
                    df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
                    df = df.dropna(subset=['fundingRate'])
                    
                    print(f"Loaded {len(df)} funding rate records for {symbol} from {new_pattern_file}")
                    return df
            except Exception as e:
                print(f"Error loading funding file {new_pattern_file}: {e}")
        
        # Try consolidated file pattern (old format)
        consolidated_file = os.path.join(self.data_folder, f"funding_rate_{symbol}.parquet")
        
        if os.path.exists(consolidated_file):
            try:
                df = pd.read_parquet(consolidated_file)
                if not df.empty:
                    # Filter to recent data based on days_back
                    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days_back)
                    df = df[df['fundingTime'] >= cutoff_date]
                    
                    # Clean and process data
                    df = df.drop_duplicates(subset=['fundingTime']).sort_values('fundingTime')
                    df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
                    df = df.dropna(subset=['fundingRate'])
                    
                    print(f"Loaded {len(df)} funding rate records for {symbol} from consolidated file")
                    return df
            except Exception as e:
                print(f"Error loading consolidated funding file {consolidated_file}: {e}")
        
        # Fallback to old pattern for backward compatibility
        pattern = os.path.join(self.data_folder, f"funding_rate_{symbol}_*.parquet")
        files = glob.glob(pattern)
        
        if not files:
            print(f"No funding rate data found for {symbol}")
            return None
        
        files.sort()
        recent_files = files[-days_back:] if len(files) > days_back else files
        
        dfs = []
        for file in recent_files:
            try:
                df = pd.read_parquet(file)
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                print(f"Error loading funding data {file}: {e}")
        
        if dfs:
            df = pd.concat(dfs, ignore_index=True)
            df = df.drop_duplicates(subset=['fundingTime']).sort_values('fundingTime')
            df['fundingRate'] = pd.to_numeric(df['fundingRate'], errors='coerce')
            df = df.dropna(subset=['fundingRate'])
            print(f"Loaded {len(df)} funding rate records for {symbol} from legacy files")
            return df
        return None
    
    def load_mark_price_data(self, symbol="BTCUSDT", days_back=30):
        """Load mark price data from consolidated parquet file"""
        # Try consolidated file first
        consolidated_file = os.path.join(self.data_folder, f"mark_price_{symbol}.parquet")
        
        if os.path.exists(consolidated_file):
            try:
                df = pd.read_parquet(consolidated_file)
                if not df.empty:
                    # Filter to recent data based on days_back
                    cutoff_date = pd.Timestamp.now() - pd.Timedelta(days=days_back)
                    df = df[df['timestamp'] >= cutoff_date]
                    
                    # Clean and process data
                    df = df.drop_duplicates(subset=['timestamp']).sort_values('timestamp')
                    df['markPrice'] = pd.to_numeric(df['markPrice'], errors='coerce')
                    df['lastFundingRate'] = pd.to_numeric(df['lastFundingRate'], errors='coerce')
                    df = df.dropna(subset=['markPrice'])
                    
                    print(f"Loaded {len(df)} mark price records for {symbol} from consolidated file")
                    return df
            except Exception as e:
                print(f"Error loading consolidated mark price file {consolidated_file}: {e}")
        
        print(f"No mark price data found for {symbol}")
        return None
    
    def get_fear_greed_index(self, date):
        """Get Fear & Greed index for a given date (cached)"""
        date_str = date.strftime('%Y-%m-%d')
        
        if date_str in self.fear_greed_cache:
            return self.fear_greed_cache[date_str]
        
        try:
            # Alternative.me Fear & Greed API
            url = "https://api.alternative.me/fng/"
            response = requests.get(url, timeout=5)
            data = response.json()
            
            if 'data' in data and len(data['data']) > 0:
                fng_value = int(data['data'][0]['value'])
                # Normalize to [0, 1] range
                normalized = fng_value / 100.0
                self.fear_greed_cache[date_str] = normalized
                return normalized
        except Exception as e:
            print(f"Error fetching Fear & Greed index: {e}")
        
        # Return neutral value if API fails
        return 0.5
    
    def calculate_log_returns(self, df):
        """Calculate log returns for multiple periods"""
        for period in self.log_return_periods:
            df[f'log_return_{period}'] = np.log(df['close'] / df['close'].shift(period))
        return df
    
    def calculate_atr(self, df, period=14):
        """Calculate Average True Range with shift(1) for anti-leak protection"""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift(1))
        low_close = np.abs(df['low'] - df['close'].shift(1))
        
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        df['atr'] = true_range.rolling(window=period).mean().shift(1)  # shift(1) to prevent look-ahead
        return df
    
    def calculate_sma(self, df):
        """Calculate Simple Moving Averages with shift(1) for anti-leak protection"""
        for period in self.sma_periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean().shift(1)  # shift(1) to prevent look-ahead
        return df
    
    def calculate_sma_slope(self, df, period=20):
        """Calculate SMA slope (rate of change) - extra feature only"""
        if self.enable_extra_features:
            sma = df['close'].rolling(window=period).mean()
            df[f'sma_{period}_slope'] = sma.diff().shift(1)  # shift(1) to prevent look-ahead
        return df
    
    def calculate_rsi(self, df, period=14):
        """Calculate Relative Strength Index with shift(1) for anti-leak protection"""
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain / loss
        df['rsi'] = (100 - (100 / (1 + rs))).shift(1)  # shift(1) to prevent look-ahead
        return df
    
    def calculate_volume_zscore(self, df, period=20):
        """Calculate volume z-score with shift(1) for anti-leak protection"""
        volume_mean = df['volume'].rolling(window=period).mean()
        volume_std = df['volume'].rolling(window=period).std()
        df['volume_zscore'] = ((df['volume'] - volume_mean) / volume_std).shift(1)  # shift(1) to prevent look-ahead
        return df
    
    def calculate_rolling_std(self, df, period=20):
        """Calculate rolling standard deviation of returns (extra feature only)"""
        if self.enable_extra_features:
            returns = df['close'].pct_change()
            df['rolling_std'] = returns.rolling(window=period).std().shift(1)  # shift(1) to prevent look-ahead
        return df
    
    def add_regime_flags(self, df):
        """Add market regime flags based on volatility and trend (extra features only)"""
        if self.enable_extra_features:
            # Volatility regime
            df['volatility_regime'] = pd.cut(
                df['rolling_std'], 
                bins=[0, df['rolling_std'].quantile(0.33), df['rolling_std'].quantile(0.67), float('inf')],
                labels=['low', 'medium', 'high']
            )
            
            # Trend regime based on SMA relationship
            df['trend_regime'] = 'sideways'
            df.loc[df['sma_20'] > df['sma_100'], 'trend_regime'] = 'uptrend'
            df.loc[df['sma_20'] < df['sma_100'], 'trend_regime'] = 'downtrend'
            
            # Convert to numeric for modeling
            df['volatility_regime_encoded'] = df['volatility_regime'].astype('category').cat.codes
            df['trend_regime_encoded'] = df['trend_regime'].astype('category').cat.codes
        
        return df
    
    def add_session_features(self, df):
        """Add session and time-based features"""
        df['hour'] = df['open_time'].dt.hour
        df['day_of_week'] = df['open_time'].dt.dayofweek
        df['day_of_month'] = df['open_time'].dt.day
        
        # Market sessions (UTC)
        df['session'] = 'other'
        df.loc[(df['hour'] >= 0) & (df['hour'] < 8), 'session'] = 'asian'
        df.loc[(df['hour'] >= 8) & (df['hour'] < 16), 'session'] = 'european'
        df.loc[(df['hour'] >= 16) & (df['hour'] < 24), 'session'] = 'american'
        
        # Cyclical encoding for time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        return df
    
    def add_fear_greed_features(self, df):
        """Add Fear & Greed index features with EMA smoothing"""
        # Get raw Fear & Greed values
        df['fear_greed_raw'] = df['open_time'].dt.date.apply(self.get_fear_greed_index)
        
        # Apply EMA smoothing (period 3 as specified)
        ema_period = 3
        alpha = 2 / (ema_period + 1)
        df['fear_greed'] = df['fear_greed_raw'].ewm(alpha=alpha, adjust=False).mean()
        
        # Ensure values stay in [0, 1] range
        df['fear_greed'] = df['fear_greed'].clip(0, 1)
        
        return df
    
    def add_funding_context(self, df, funding_df):
        """Add funding rate context with last_funding_rate and minutes_to_next_funding"""
        if funding_df is not None and not funding_df.empty:
            # Sort funding data by time
            funding_df = funding_df.sort_values('fundingTime')
            
            # Initialize new columns
            df['funding_rate'] = np.nan
            df['last_funding_rate'] = 0.0
            df['minutes_to_next_funding'] = 480  # Default to 8 hours (480 minutes)
            
            for idx, row in df.iterrows():
                # Find current active funding rate (most recent funding before or at this time)
                past_funding = funding_df[funding_df['fundingTime'] <= row['open_time']]
                if not past_funding.empty:
                    df.loc[idx, 'funding_rate'] = past_funding.iloc[-1]['fundingRate']
                    df.loc[idx, 'last_funding_rate'] = past_funding.iloc[-1]['fundingRate']
                else:
                    # No past funding data, use default
                    df.loc[idx, 'funding_rate'] = 0.0
                    df.loc[idx, 'last_funding_rate'] = 0.0
                
                # Find minutes to next funding
                future_funding = funding_df[funding_df['fundingTime'] > row['open_time']]
                if not future_funding.empty:
                    next_funding_time = future_funding.iloc[0]['fundingTime']
                    minutes_diff = (next_funding_time - row['open_time']).total_seconds() / 60
                    df.loc[idx, 'minutes_to_next_funding'] = int(minutes_diff)
                else:
                    # No future funding data, use default
                    df.loc[idx, 'minutes_to_next_funding'] = 480
        else:
            df['funding_rate'] = 0.0
            df['last_funding_rate'] = 0.0
            df['minutes_to_next_funding'] = 480  # Default to 8 hours
        
        return df
    
    def _ensure_mark_index_columns(self, df):
        """Ensure mark and index price columns exist for backward compatibility"""
        # Check if mark price columns exist, if not create them from regular OHLCV
        mark_cols = ['mark_open', 'mark_high', 'mark_low', 'mark_close']
        for col in mark_cols:
            if col not in df.columns:
                if col == 'mark_open':
                    df[col] = df['open']
                elif col == 'mark_high':
                    df[col] = df['high']
                elif col == 'mark_low':
                    df[col] = df['low']
                elif col == 'mark_close':
                    df[col] = df['close']
        
        # Check if index price columns exist, if not create them from regular OHLCV
        index_cols = ['index_open', 'index_high', 'index_low', 'index_close']
        for col in index_cols:
            if col not in df.columns:
                if col == 'index_open':
                    df[col] = df['open']
                elif col == 'index_high':
                    df[col] = df['high']
                elif col == 'index_low':
                    df[col] = df['low']
                elif col == 'index_close':
                    df[col] = df['close']
        
        # Check if quote_volume exists, if not create it from volume * close
        if 'quote_volume' not in df.columns:
            df['quote_volume'] = df['volume'] * df['close']
        
        return df
    
    def generate_targets(self, df, target_periods=[30, 60, 90, 120]):
        """Generate classification and regression targets with embargo and shift(1)"""
        for period in target_periods:
            # Calculate future returns
            future_close = df['close'].shift(-period)
            future_return = (future_close - df['close']) / df['close']
            future_return_bps = future_return * 10000  # Convert to basis points
            
            # Classification target: sign of return
            df[f'target_class_{period}m'] = np.sign(future_return)
            
            # Regression target: return in basis points
            df[f'target_reg_{period}m'] = future_return_bps
            
            # Apply shift(1) to prevent data leakage (use previous timestep's target)
            df[f'target_class_{period}m'] = df[f'target_class_{period}m'].shift(1)
            df[f'target_reg_{period}m'] = df[f'target_reg_{period}m'].shift(1)
            
            # Add embargo period to prevent data leakage
            embargo_start = len(df) - period - 1
            df.loc[embargo_start:, f'target_class_{period}m'] = np.nan
            df.loc[embargo_start:, f'target_reg_{period}m'] = np.nan
        
        return df
    
    def _load_existing_features(self, symbol, interval):
        """Load existing features file if it exists"""
        try:
            # Look for existing features files
            features_dir = Path(self.output_folder)
            if not features_dir.exists():
                return None
            
            # Find the most recent features file for this symbol and interval
            pattern = f"features_{symbol}_{interval}_*.parquet"
            feature_files = list(features_dir.glob(pattern))
            
            if not feature_files:
                return None
            
            # Get the most recent file
            latest_file = max(feature_files, key=lambda x: x.stat().st_mtime)
            print(f"Found existing features file: {latest_file.name}")
            
            # Load the existing features
            existing_df = pd.read_parquet(latest_file)
            return existing_df
            
        except Exception as e:
            print(f"Error loading existing features: {e}")
            return None
    
    def load_latest_features(self, symbol, interval):
        """Load the latest features for a symbol and interval"""
        try:
            # Look for existing features files
            features_dir = Path(self.output_folder)
            if not features_dir.exists():
                self.logger.warning(f"Features directory {features_dir} does not exist")
                return None
            
            # Find the most recent features file for this symbol and interval
            pattern = f"features_{symbol}_{interval}_*.parquet"
            feature_files = list(features_dir.glob(pattern))
            
            if not feature_files:
                self.logger.warning(f"No features files found for {symbol} {interval}")
                return None
            
            # Get the most recent file
            latest_file = max(feature_files, key=lambda x: x.stat().st_mtime)
            self.logger.info(f"Loading latest features from: {latest_file.name}")
            
            # Load the existing features
            features_df = pd.read_parquet(latest_file)
            
            # Ensure open_time is datetime
            if 'open_time' in features_df.columns:
                features_df['open_time'] = pd.to_datetime(features_df['open_time'])
                features_df = features_df.sort_values('open_time')
            
            self.logger.info(f"Loaded {len(features_df)} feature records for {symbol} {interval}")
            return features_df
            
        except Exception as e:
            self.logger.error(f"Error loading latest features for {symbol} {interval}: {e}")
            return None
    
    def load_features_for_training(self, symbol, interval, days_back=30):
        """Load features for training (alias for load_latest_features)"""
        return self.load_latest_features(symbol, interval)
    
    def create_feature_matrix(self, symbol="BTCUSDT", days_back=30, target_periods=[30, 60, 90, 120], interval="1m"):
        """Create feature matrix with incremental calculation"""
        print(f"Creating feature matrix for {symbol} {interval}...")
        
        # Check if features file already exists
        existing_features = self._load_existing_features(symbol, interval)
        
        if existing_features is not None:
            print(f"Found existing features with {len(existing_features)} records")
            last_timestamp = existing_features['open_time'].max()
            print(f"Last feature timestamp: {last_timestamp}")
            
            # Load only new data since last timestamp
            df = self.load_ohlcv_data(symbol, days_back, interval)
            funding_df = self.load_funding_data(symbol, days_back)
            
            # Filter to only new data (with minimal overlap for feature calculation)
            overlap_minutes = 30  # Keep 30 minutes of overlap for proper feature calculation
            cutoff_time = last_timestamp - pd.Timedelta(minutes=overlap_minutes)
            new_data_mask = df['open_time'] > cutoff_time
            df_new = df[new_data_mask].copy()
            
            if len(df_new) == 0:
                print("No new data to process")
                return existing_features
            
            print(f"Processing {len(df_new)} new records (with {overlap_minutes}m overlap)")
            
            # For incremental calculation, we need some historical data for proper feature calculation
            # So we'll use a minimal window for feature calculation but only keep new results
            df_for_calc = df[df['open_time'] >= cutoff_time - pd.Timedelta(hours=2)].copy()
            
        else:
            print("No existing features found, creating from scratch")
            # Load all data
            df = self.load_ohlcv_data(symbol, days_back, interval)
            funding_df = self.load_funding_data(symbol, days_back)
            df_for_calc = df.copy()
            existing_features = None
        
        print(f"Loaded {len(df_for_calc)} OHLCV records for feature calculation")
        
        # Calculate MVP features (always enabled)
        df_calc = self.calculate_atr(df_for_calc, self.atr_period)  # MVP: ATR(14)
        df_calc = self.calculate_sma(df_calc)  # MVP: SMA(20/100)
        df_calc = self.calculate_rsi(df_calc, self.rsi_period)  # MVP: RSI(14)
        df_calc = self.calculate_volume_zscore(df_calc, self.volume_zscore_period)  # MVP: volume z-score
        df_calc = self.calculate_log_returns(df_calc)  # MVP: log_return_1 (always enabled)
        df_calc = self.add_session_features(df_calc)  # MVP: session/time
        df_calc = self.add_fear_greed_features(df_calc)  # MVP: Fear & Greed (0..1)
        df_calc = self.add_funding_context(df_calc, funding_df)  # MVP: funding context
        df_calc = self.generate_targets(df_calc, target_periods)
        
        # Handle missing mark/index price columns for backward compatibility
        df_calc = self._ensure_mark_index_columns(df_calc)
        
        # Calculate extra features (only if enabled)
        if self.enable_extra_features:
            df_calc = self.calculate_log_returns(df_calc)
            df_calc = self.calculate_sma_slope(df_calc, 20)
            df_calc = self.calculate_rolling_std(df_calc, self.rolling_std_period)
            df_calc = self.add_regime_flags(df_calc)
        
        # Select MVP feature columns (ML features only - no raw OHLCV data)
        feature_cols = [
            'open_time',  # Time reference
            'atr', 'sma_20', 'sma_100', 'rsi', 'volume_zscore', 'log_return_1',  # Technical indicators
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',  # Session features
            'fear_greed', 'funding_rate', 'last_funding_rate', 'minutes_to_next_funding'  # Market context
        ]
        
        # Add extra feature columns if enabled
        if self.enable_extra_features:
            extra_cols = [
                'log_return_3', 'log_return_6', 'log_return_12',  # Additional return periods
                'rolling_std', 'sma_20_slope', 'volatility_regime_encoded', 'trend_regime_encoded'
            ]
            feature_cols.extend(extra_cols)
        
        # Add target columns
        target_cols = []
        for period in target_periods:
            target_cols.extend([f'target_class_{period}m', f'target_reg_{period}m'])
        
        all_cols = feature_cols + target_cols
        
        # Filter to only include columns that exist in the DataFrame
        available_cols = [col for col in all_cols if col in df_calc.columns]
        missing_cols = [col for col in all_cols if col not in df_calc.columns]
        
        if missing_cols:
            print(f"Warning: Missing columns in data: {missing_cols}")
        
        df_features = df_calc[available_cols].copy()
        
        # Remove rows with NaN values (due to rolling calculations and embargo)
        initial_len = len(df_features)
        df_features = df_features.dropna()
        final_len = len(df_features)
        
        if existing_features is not None:
            # Filter to only new features (after the last timestamp)
            df_new_features = df_features[df_features['open_time'] > last_timestamp].copy()
            
            if len(df_new_features) > 0:
                print(f"Generated {len(df_new_features)} new feature records")
                # Merge with existing features
                df_features = pd.concat([existing_features, df_new_features], ignore_index=True)
                df_features = df_features.sort_values('open_time').reset_index(drop=True)
                print(f"Combined features: {len(df_features)} total records")
            else:
                print("No new features to add")
                df_features = existing_features
        else:
            print(f"Feature matrix created: {final_len} rows (removed {initial_len - final_len} NaN rows)")
        
        return df_features
    
    def save_feature_matrix(self, df, symbol, timeframe="1m"):
        """Save feature matrix to parquet with proper keying"""
        if df.empty:
            print("No data to save")
            return
        
        # Create filename with symbol, timeframe, and date range
        start_date = df['open_time'].min().strftime('%Y%m%d')
        end_date = df['open_time'].max().strftime('%Y%m%d')
        
        filename = f"features_{symbol}_{timeframe}_{start_date}_{end_date}.parquet"
        filepath = os.path.join(self.output_folder, filename)
        
        # Save with compression
        df.to_parquet(filepath, compression='snappy', index=False)
        print(f"Saved feature matrix: {filepath}")
        
        return filepath
    
    def process_symbol(self, symbol="BTCUSDT", timeframe="1m", days_back=30, target_periods=[30, 60, 90, 120]):
        """Complete processing pipeline for a symbol"""
        try:
            # Create feature matrix
            df_features = self.create_feature_matrix(symbol, days_back, target_periods, timeframe)
            
            if df_features.empty:
                print(f"No features generated for {symbol}")
                return None
            
            # Save to parquet
            filepath = self.save_feature_matrix(df_features, symbol, timeframe)
            
            # Print summary statistics
            print(f"\nFeature Matrix Summary for {symbol}:")
            print(f"Shape: {df_features.shape}")
            print(f"Date range: {df_features['open_time'].min()} to {df_features['open_time'].max()}")
            print(f"Features: {len([col for col in df_features.columns if not col.startswith('target_')])}")
            print(f"Targets: {len([col for col in df_features.columns if col.startswith('target_')])}")
            
            return filepath
            
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
            return None

def main():
    """Main execution function"""
    # Initialize feature engineer with MVP features only (extra features disabled by default)
    fe = FeatureEngineer(enable_extra_features=False)
    
    # Process BTCUSDT with MVP features
    fe.process_symbol(
        symbol="BTCUSDT",
        timeframe="1m",
        days_back=30,
        target_periods=[30, 60, 90, 120]
    )

if __name__ == "__main__":
    main()

