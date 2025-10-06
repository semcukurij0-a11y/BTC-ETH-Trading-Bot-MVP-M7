"""
Crypto Trading Bot Services

This package contains the core services for the crypto trading bot:
- data_ingestor: Handles data fetching from exchanges
- feature_engineering: Creates ML-ready features from raw data
"""

from .data_ingestor import DataIngestor
from .feature_engineering import FeatureEngineer

__all__ = ['DataIngestor', 'FeatureEngineer']
