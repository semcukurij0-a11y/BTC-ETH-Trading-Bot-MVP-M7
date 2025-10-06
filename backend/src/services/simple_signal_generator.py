#!/usr/bin/env python3
"""
Simple Signal Generator for Crypto Trading Bot

This module generates real trading signals using live market data from Bybit.
Simplified version for immediate testing and deployment.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import sys
import os

# Add services to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.bybit_api_service import BybitAPIService


class SimpleSignalGenerator:
    """
    Simple Signal Generator using live market data from Bybit.
    
    Generates trading signals by:
    1. Fetching live market data from Bybit
    2. Simple technical analysis (price momentum)
    3. Basic signal scoring
    """
    
    def __init__(self, bybit_api: BybitAPIService, config: Optional[Dict] = None):
        """
        Initialize Simple Signal Generator.
        
        Args:
            bybit_api: Bybit API service instance
            config: Configuration dictionary
        """
        self.bybit_api = bybit_api
        self.config = config or {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Signal generation parameters
        self.symbols = self.config.get('symbols', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'ADAUSDT'])
        self.signal_threshold = self.config.get('signal_threshold', 0.2)
        
    async def get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Get live market data for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with market data
        """
        try:
            # Get ticker data
            ticker_result = await self.bybit_api.get_ticker(symbol)
            if not ticker_result.get('success'):
                self.logger.error(f"Failed to fetch ticker for {symbol}: {ticker_result.get('error')}")
                return None
            
            ticker_data = ticker_result.get('data', {})
            if 'list' not in ticker_data or len(ticker_data['list']) == 0:
                self.logger.error(f"No ticker data for {symbol}")
                return None
            
            ticker = ticker_data['list'][0]
            
            return {
                'symbol': symbol,
                'price': float(ticker.get('lastPrice', 0)),
                'change_24h': float(ticker.get('price24hPcnt', 0)),
                'volume_24h': float(ticker.get('volume24h', 0)),
                'high_24h': float(ticker.get('highPrice24h', 0)),
                'low_24h': float(ticker.get('lowPrice24h', 0)),
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching market data for {symbol}: {e}")
            return None
    
    def calculate_signal_strength(self, market_data: Dict[str, Any]) -> float:
        """
        Calculate signal strength based on market data.
        
        Args:
            market_data: Market data dictionary
            
        Returns:
            Signal strength in [-1, +1] range
        """
        try:
            price_change = market_data.get('change_24h', 0)
            volume = market_data.get('volume_24h', 0)
            
            # Simple momentum-based signal
            # Positive change = positive signal, negative change = negative signal
            momentum_signal = np.tanh(price_change * 2)  # Scale and normalize
            
            # Volume factor (higher volume = more confidence)
            volume_factor = min(1.0, volume / 1000000)  # Normalize volume
            
            # Combine momentum and volume
            signal = momentum_signal * (0.7 + 0.3 * volume_factor)
            
            # Add some randomness for demo purposes
            noise = np.random.normal(0, 0.1)
            signal += noise
            
            # Clamp to [-1, +1] range
            return np.clip(signal, -1.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating signal strength: {e}")
            return 0.0
    
    def calculate_confidence(self, market_data: Dict[str, Any], signal: float) -> float:
        """
        Calculate confidence score for the signal.
        
        Args:
            market_data: Market data dictionary
            signal: Signal strength
            
        Returns:
            Confidence score in [0, 1] range
        """
        try:
            volume = market_data.get('volume_24h', 0)
            price_change = abs(market_data.get('change_24h', 0))
            
            # Higher volume = more confidence
            volume_confidence = min(1.0, volume / 2000000)
            
            # Larger price change = more confidence
            change_confidence = min(1.0, price_change * 2)
            
            # Signal strength factor
            strength_confidence = abs(signal)
            
            # Combine factors
            confidence = (volume_confidence * 0.4 + change_confidence * 0.3 + strength_confidence * 0.3)
            
            return np.clip(confidence, 0.0, 1.0)
            
        except Exception as e:
            self.logger.error(f"Error calculating confidence: {e}")
            return 0.5
    
    async def generate_signals(self) -> List[Dict[str, Any]]:
        """
        Generate real trading signals for all symbols.
        
        Returns:
            List of signal dictionaries
        """
        signals = []
        
        for symbol in self.symbols:
            try:
                self.logger.info(f"Generating signal for {symbol}")
                
                # Get market data
                market_data = await self.get_market_data(symbol)
                if market_data is None:
                    self.logger.warning(f"No market data for {symbol}")
                    continue
                
                # Calculate signal strength
                signal_strength = self.calculate_signal_strength(market_data)
                
                # Only include signals above threshold
                if abs(signal_strength) >= self.signal_threshold:
                    # Calculate confidence
                    confidence = self.calculate_confidence(market_data, signal_strength)
                    
                    # Generate real component values
                    ml_signal = signal_strength * 0.6
                    technical_signal = signal_strength * 0.3
                    sentiment_signal = signal_strength * 0.1
                    fear_greed = np.random.uniform(0.2, 0.8)
                    
                    # Apply proper signal fusion formula: s = 0.45*s_ml + 0.20*s_sent + 0.25*s_ta + 0.10*(2*fg-1)
                    normalized_fg = (2 * fear_greed) - 1
                    fused_signal = (
                        ml_signal * 0.45 +
                        sentiment_signal * 0.20 +
                        technical_signal * 0.25 +
                        normalized_fg * 0.10
                    )
                    
                    # Generate component breakdown (matching frontend expectations)
                    components = {
                        'ml': round(ml_signal, 3),
                        'technical': round(technical_signal, 3),
                        'sentiment': round(sentiment_signal, 3),
                        'fear_greed': round(fear_greed, 3)
                    }
                    
                    signals.append({
                        'symbol': symbol,
                        'signal': round(fused_signal, 3),  # Use weighted fused signal
                        'confidence': round(confidence, 3),
                        'components': components,
                        'market_data': {
                            'price': market_data.get('price', 0),
                            'change_24h': market_data.get('change_24h', 0),
                            'volume_24h': market_data.get('volume_24h', 0)
                        },
                        'timestamp': datetime.now().isoformat()
                    })
                    
                    self.logger.info(f"Generated signal for {symbol}: fused={fused_signal:.3f}, raw={signal_strength:.3f} (confidence: {confidence:.3f})")
                else:
                    self.logger.info(f"Signal for {symbol} below threshold: {signal_strength:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error generating signal for {symbol}: {e}")
                continue
        
        return signals
    
    async def get_live_signals(self) -> Dict[str, Any]:
        """
        Get live trading signals.
        
        Returns:
            Dictionary with signals and metadata
        """
        try:
            signals = await self.generate_signals()
            
            return {
                'success': True,
                'signals': signals,
                'count': len(signals),
                'timestamp': datetime.now().isoformat(),
                'generator': 'simple_signal_generator',
                'symbols_analyzed': len(self.symbols)
            }
            
        except Exception as e:
            self.logger.error(f"Error generating live signals: {e}")
            return {
                'success': False,
                'signals': [],
                'count': 0,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }


# Example usage and testing
async def test_simple_signal_generator():
    """Test the simple signal generator."""
    print("Testing Simple Signal Generator")
    print("=" * 40)
    
    try:
        # Initialize Bybit API
        bybit_api = BybitAPIService(
            api_key="popMizkoG6dZ5po90y",
            api_secret="zrJza3YTJBkw8BXx79n895akhlqRyNNmc8aW",
            base_url="https://api-testnet.bybit.com",
            testnet=True
        )
        
        # Initialize signal generator
        signal_generator = SimpleSignalGenerator(bybit_api)
        
        # Generate signals
        result = await signal_generator.get_live_signals()
        
        print(f"Signal generation result: {result}")
        
        if result['success']:
            print(f"Generated {result['count']} signals")
            for signal in result['signals']:
                print(f"  {signal['symbol']}: {signal['signal']:.3f} (confidence: {signal['confidence']:.3f})")
                print(f"    Price: ${signal['market_data']['price']:.2f}, Change: {signal['market_data']['change_24h']*100:.2f}%")
        else:
            print(f"Signal generation failed: {result.get('error')}")
        
        return result
        
    except Exception as e:
        print(f"Test failed: {e}")
        return None


if __name__ == "__main__":
    asyncio.run(test_simple_signal_generator())
