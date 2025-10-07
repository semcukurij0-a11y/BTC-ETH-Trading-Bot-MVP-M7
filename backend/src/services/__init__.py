"""
Crypto Trading Bot Services

This package contains the core services for the crypto trading bot:
- data_ingestor: Handles data fetching from exchanges
- live_feature_engineer: Creates ML-ready features from raw data
"""

from .data_ingestor import DataIngestor

__all__ = ['DataIngestor']
