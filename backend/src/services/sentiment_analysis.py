"""
Sentiment Analysis Module for Crypto Trading Bot

This module implements sentiment analysis for crypto markets and generates s_sent signals.
Features:
- Social media sentiment analysis (Twitter, Reddit)
- News sentiment analysis
- Market sentiment indicators
- Output: s_sent in [-1, +1] range
"""

import pandas as pd
import numpy as np
import logging
import requests
import json
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class SentimentAnalysisModule:
    """
    Sentiment Analysis Module for crypto trading signals.
    
    Implements:
    - Social media sentiment analysis
    - News sentiment analysis
    - Market sentiment indicators
    - s_sent signal generation in [-1, +1] range
    """
    
    def __init__(self, 
                 config: Optional[Dict] = None,
                 enable_social_media: bool = True,
                 enable_news_analysis: bool = True):
        """
        Initialize Sentiment Analysis Module.
        
        Args:
            config: Configuration dictionary
            enable_social_media: Enable social media sentiment analysis
            enable_news_analysis: Enable news sentiment analysis
        """
        self.config = config or {}
        self.enable_social_media = enable_social_media
        self.enable_news_analysis = enable_news_analysis
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Sentiment analysis parameters
        self.sentiment_window = self.config.get('sentiment_window', 24)  # hours
        self.sentiment_smoothing = self.config.get('sentiment_smoothing', 3)  # EMA period
        self.min_sentiment_samples = self.config.get('min_sentiment_samples', 10)
        
        # API configurations (placeholder - would need actual API keys)
        self.twitter_api_key = self.config.get('twitter_api_key', None)
        self.reddit_api_key = self.config.get('reddit_api_key', None)
        self.news_api_key = self.config.get('news_api_key', None)
        
        # Sentiment weights
        self.social_weight = self.config.get('social_weight', 0.4)
        self.news_weight = self.config.get('news_weight', 0.3)
        self.market_weight = self.config.get('market_weight', 0.3)
        
        # Sentiment cache
        self.sentiment_cache = {}
        
        # Sentiment keywords
        self.bullish_keywords = [
            'bullish', 'moon', 'pump', 'buy', 'long', 'hodl', 'diamond hands',
            'breakout', 'rally', 'surge', 'gains', 'profit', 'green', 'up'
        ]
        
        self.bearish_keywords = [
            'bearish', 'dump', 'crash', 'sell', 'short', 'paper hands',
            'correction', 'decline', 'loss', 'red', 'down', 'fear', 'panic'
        ]
    
    def analyze_social_media_sentiment(self, symbol: str = "BTCUSDT") -> float:
        """
        Analyze social media sentiment for a given symbol.
        
        Args:
            symbol: Trading symbol to analyze
            
        Returns:
            Sentiment score in [-1, +1] range
        """
        if not self.enable_social_media:
            return 0.0
        
        try:
            # Placeholder implementation - would integrate with actual APIs
            # Twitter API, Reddit API, etc.
            
            # Simulate sentiment analysis with random data for demo
            np.random.seed(hash(symbol) % 2**32)
            
            # Generate realistic sentiment based on market conditions
            base_sentiment = np.random.normal(0, 0.3)
            
            # Add some trending behavior
            trend_factor = np.sin(datetime.now().hour / 24 * 2 * np.pi) * 0.2
            
            sentiment = np.clip(base_sentiment + trend_factor, -1, 1)
            
            self.logger.info(f"Social media sentiment for {symbol}: {sentiment:.4f}")
            return sentiment
            
        except Exception as e:
            self.logger.error(f"Error analyzing social media sentiment: {e}")
            return 0.0
    
    def analyze_news_sentiment(self, symbol: str = "BTCUSDT") -> float:
        """
        Analyze news sentiment for a given symbol.
        
        Args:
            symbol: Trading symbol to analyze
            
        Returns:
            Sentiment score in [-1, +1] range
        """
        if not self.enable_news_analysis:
            return 0.0
        
        try:
            # Placeholder implementation - would integrate with news APIs
            # NewsAPI, Google News, etc.
            
            # Simulate news sentiment analysis
            np.random.seed(hash(symbol + "news") % 2**32)
            
            # Generate realistic news sentiment
            base_sentiment = np.random.normal(0, 0.2)
            
            # Add some market correlation
            market_factor = np.sin(datetime.now().day / 30 * 2 * np.pi) * 0.15
            
            sentiment = np.clip(base_sentiment + market_factor, -1, 1)
            
            self.logger.info(f"News sentiment for {symbol}: {sentiment:.4f}")
            return sentiment
            
        except Exception as e:
            self.logger.error(f"Error analyzing news sentiment: {e}")
            return 0.0
    
    def analyze_market_sentiment(self, df: pd.DataFrame) -> pd.Series:
        """
        Analyze market-based sentiment indicators.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Series with market sentiment scores
        """
        sentiment_scores = pd.Series(0.0, index=df.index)
        
        if df.empty:
            return sentiment_scores
        
        # Determine which price column to use
        price_column = None
        if 'close' in df.columns:
            price_column = 'close'
        elif 'mark_close' in df.columns:
            price_column = 'mark_close'
        elif 'index_close' in df.columns:
            price_column = 'index_close'
        
        # Price momentum sentiment
        if price_column:
            price_change = df[price_column].pct_change()
            momentum_sentiment = np.tanh(price_change * 10)  # Scale and normalize
        else:
            momentum_sentiment = pd.Series(0, index=df.index)
            self.logger.warning("No price column found for momentum sentiment, using neutral values")
        
        # Volume sentiment
        if 'volume' in df.columns:
            volume_change = df['volume'].pct_change()
            volume_sentiment = np.tanh(volume_change * 5)
        else:
            volume_sentiment = pd.Series(0, index=df.index)
            self.logger.warning("No volume column found for volume sentiment, using neutral values")
        
        # Volatility sentiment (inverse relationship)
        if price_column:
            volatility = df[price_column].rolling(20).std() / df[price_column].rolling(20).mean()
            volatility_sentiment = -np.tanh(volatility * 20)  # High volatility = negative sentiment
        else:
            volatility_sentiment = pd.Series(0, index=df.index)
            self.logger.warning("No price column found for volatility sentiment, using neutral values")
        
        # Combine market indicators
        market_sentiment = (
            momentum_sentiment * 0.5 +
            volume_sentiment * 0.3 +
            volatility_sentiment * 0.2
        )
        
        return market_sentiment.fillna(0)
    
    def calculate_ema_smoothing(self, series: pd.Series, period: int = 3) -> pd.Series:
        """
        Apply EMA smoothing to sentiment data.
        
        Args:
            series: Input sentiment series
            period: EMA period
            
        Returns:
            EMA smoothed series
        """
        alpha = 2 / (period + 1)
        ema = series.ewm(alpha=alpha, adjust=False).mean()
        return ema
    
    def generate_s_sent_signal(self, df: pd.DataFrame, symbol: str = "BTCUSDT") -> pd.DataFrame:
        """
        Generate s_sent signal in [-1, +1] range combining all sentiment sources.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Trading symbol
            
        Returns:
            DataFrame with s_sent signal
        """
        # Initialize sentiment columns
        df['social_sentiment'] = 0.0
        df['news_sentiment'] = 0.0
        df['market_sentiment'] = 0.0
        df['s_sent'] = 0.0
        df['s_sent_confidence'] = 0.0
        
        if df.empty:
            return df
        
        # Get social media sentiment (cached per day)
        date_key = df['open_time'].dt.date.iloc[0].strftime('%Y-%m-%d')
        if date_key not in self.sentiment_cache:
            social_sentiment = self.analyze_social_media_sentiment(symbol)
            news_sentiment = self.analyze_news_sentiment(symbol)
            self.sentiment_cache[date_key] = {
                'social': social_sentiment,
                'news': news_sentiment
            }
        
        # Apply cached sentiment to all rows for the day
        df['social_sentiment'] = self.sentiment_cache[date_key]['social']
        df['news_sentiment'] = self.sentiment_cache[date_key]['news']
        
        # Calculate market sentiment
        df['market_sentiment'] = self.analyze_market_sentiment(df)
        
        # Combine sentiment sources with weights
        df['s_sent_raw'] = (
            df['social_sentiment'] * self.social_weight +
            df['news_sentiment'] * self.news_weight +
            df['market_sentiment'] * self.market_weight
        )
        
        # Apply EMA smoothing
        df['s_sent'] = self.calculate_ema_smoothing(df['s_sent_raw'], self.sentiment_smoothing)
        
        # Calculate confidence based on agreement between sources
        sentiment_sources = []
        if self.enable_social_media:
            sentiment_sources.append(df['social_sentiment'])
        if self.enable_news_analysis:
            sentiment_sources.append(df['news_sentiment'])
        sentiment_sources.append(df['market_sentiment'])
        
        if len(sentiment_sources) > 1:
            # Calculate standard deviation as inverse confidence
            sentiment_std = pd.concat(sentiment_sources, axis=1).std(axis=1)
            df['s_sent_confidence'] = 1 - np.clip(sentiment_std, 0, 1)
        else:
            df['s_sent_confidence'] = 0.5
        
        # Add signal strength
        df['s_sent_strength'] = abs(df['s_sent'])
        
        return df
    
    def get_latest_s_sent_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get the latest sentiment signal.
        
        Args:
            df: DataFrame with sentiment analysis data
            
        Returns:
            Dictionary with latest signal information
        """
        if df.empty:
            return {
                's_sent': 0.0,
                's_sent_strength': 0.0,
                's_sent_confidence': 0.0,
                'signal': 'HOLD',
                'social_sentiment': 0.0,
                'news_sentiment': 0.0,
                'market_sentiment': 0.0,
                'timestamp': datetime.now().isoformat()
            }
        
        latest = df.iloc[-1]
        
        # Determine signal direction
        if latest['s_sent'] > 0.1:
            signal = 'BUY'
        elif latest['s_sent'] < -0.1:
            signal = 'SELL'
        else:
            signal = 'HOLD'
        
        return {
            's_sent': float(latest['s_sent']),
            's_sent_strength': float(latest['s_sent_strength']),
            's_sent_confidence': float(latest['s_sent_confidence']),
            'signal': signal,
            'social_sentiment': float(latest.get('social_sentiment', 0)),
            'news_sentiment': float(latest.get('news_sentiment', 0)),
            'market_sentiment': float(latest.get('market_sentiment', 0)),
            'timestamp': datetime.now().isoformat()
        }
    
    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Get sentiment analysis summary statistics.
        
        Args:
            df: DataFrame with sentiment analysis data
            
        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {}
        
        return {
            'total_signals': len(df),
            'buy_signals': len(df[df['s_sent'] > 0.1]),
            'sell_signals': len(df[df['s_sent'] < -0.1]),
            'hold_signals': len(df[(df['s_sent'] >= -0.1) & (df['s_sent'] <= 0.1)]),
            'avg_s_sent': float(df['s_sent'].mean()),
            'avg_confidence': float(df['s_sent_confidence'].mean()),
            'avg_social_sentiment': float(df['social_sentiment'].mean()),
            'avg_news_sentiment': float(df['news_sentiment'].mean()),
            'avg_market_sentiment': float(df['market_sentiment'].mean()),
            'social_media_enabled': self.enable_social_media,
            'news_analysis_enabled': self.enable_news_analysis
        }
    
    def clear_sentiment_cache(self):
        """Clear sentiment cache."""
        self.sentiment_cache.clear()
        self.logger.info("Sentiment cache cleared")


def main():
    """Test the Sentiment Analysis Module."""
    import sys
    sys.path.append('src')
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = {
        'sentiment_window': 24,
        'sentiment_smoothing': 3,
        'min_sentiment_samples': 10,
        'social_weight': 0.4,
        'news_weight': 0.3,
        'market_weight': 0.3
    }
    
    # Initialize sentiment analysis module
    sentiment_module = SentimentAnalysisModule(
        config=config, 
        enable_social_media=True, 
        enable_news_analysis=True
    )
    
    # Create sample data for testing
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'open_time': dates,
        'open': 50000 + np.cumsum(np.random.randn(len(dates)) * 100),
        'high': 0,
        'low': 0,
        'close': 0,
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    
    # Generate realistic OHLCV data
    for i in range(len(sample_data)):
        base_price = sample_data.iloc[i]['open']
        volatility = np.random.uniform(0.01, 0.03)
        
        sample_data.iloc[i, sample_data.columns.get_loc('high')] = base_price * (1 + volatility)
        sample_data.iloc[i, sample_data.columns.get_loc('low')] = base_price * (1 - volatility)
        sample_data.iloc[i, sample_data.columns.get_loc('close')] = base_price * (1 + np.random.uniform(-volatility, volatility))
    
    # Run sentiment analysis
    result_df = sentiment_module.generate_s_sent_signal(sample_data, "BTCUSDT")
    
    # Get latest signal
    latest_signal = sentiment_module.get_latest_s_sent_signal(result_df)
    
    print("Sentiment Analysis Test Results:")
    print(f"Latest s_sent signal: {latest_signal['s_sent']:.4f}")
    print(f"Signal: {latest_signal['signal']}")
    print(f"Confidence: {latest_signal['s_sent_confidence']:.4f}")
    print(f"Social Sentiment: {latest_signal['social_sentiment']:.4f}")
    print(f"News Sentiment: {latest_signal['news_sentiment']:.4f}")
    print(f"Market Sentiment: {latest_signal['market_sentiment']:.4f}")
    
    # Get summary
    summary = sentiment_module.get_sentiment_summary(result_df)
    print(f"\nSummary: {summary}")


if __name__ == "__main__":
    main()
