#!/usr/bin/env python3
"""
Automatic Trading Service for Crypto Trading Bot

This service provides automatic trading execution based on signal generation:
- Continuous signal monitoring
- Automatic trade execution
- Risk management integration
- Performance tracking
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
import json

from .futures_trading import FuturesTradingModule, ExecutionMode
from .real_signal_generator import RealSignalGenerator
from .bybit_api_service import BybitAPIService
from .risk_management import RiskManagementModule


class AutomaticTradingService:
    """
    Automatic Trading Service for continuous signal monitoring and trade execution.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Automatic Trading Service.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Trading parameters
        self.symbols = self.config.get('symbols', ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
        self.signal_interval = self.config.get('signal_interval', 60)  # seconds
        self.trading_enabled = self.config.get('trading_enabled', True)
        self.execution_mode = ExecutionMode.SIMULATION  # Start with simulation
        
        # Initialize services
        api_key = self.config.get('api_key', 'popMizkoG6dZ5po90y')
        api_secret = self.config.get('api_secret', 'zrJza3YTJBkw8BXx79n895akhlqRyNNmc8aW')
        testnet = self.config.get('testnet', True)
        
        self.bybit_api = BybitAPIService(api_key, api_secret, testnet=testnet)
        self.signal_generator = RealSignalGenerator(self.bybit_api, self.config)
        self.futures_module = FuturesTradingModule(
            config=self.config,
            execution_mode=self.execution_mode
        )
        self.risk_module = RiskManagementModule(self.config)
        
        # Trading state
        self.is_running = False
        self.last_signal_check = {}
        self.active_signals = {}
        self.trading_stats = {
            'total_signals': 0,
            'signals_above_threshold': 0,
            'trades_executed': 0,
            'trades_successful': 0,
            'trades_failed': 0,
            'last_signal_time': None,
            'last_trade_time': None
        }
        
        # Signal thresholds
        self.min_signal_strength = self.config.get('min_signal_strength', 0.3)
        self.min_confidence = self.config.get('min_confidence', 0.6)
        
    async def start_automatic_trading(self):
        """Start automatic trading service."""
        if self.is_running:
            self.logger.warning("Automatic trading is already running")
            return
            
        self.is_running = True
        self.logger.info("Starting automatic trading service...")
        
        try:
            # Test Bybit connection
            connection_test = await self.bybit_api.test_connection()
            if not connection_test.get('success', False):
                self.logger.error("Bybit connection failed - cannot start automatic trading")
                return False
                
            self.logger.info("Bybit connection successful")
            
            # Start the main trading loop
            await self._trading_loop()
            
        except Exception as e:
            self.logger.error(f"Error in automatic trading service: {e}")
            return False
        finally:
            self.is_running = False
            
        return True
    
    async def stop_automatic_trading(self):
        """Stop automatic trading service."""
        self.logger.info("Stopping automatic trading service...")
        self.is_running = False
        
        # Emergency stop all trading
        self.futures_module.emergency_stop = True
        self.trading_enabled = False
        
        self.logger.info("Automatic trading service stopped")
    
    async def _trading_loop(self):
        """Main trading loop."""
        self.logger.info(f"Starting trading loop with {self.signal_interval}s intervals")
        
        while self.is_running and self.trading_enabled:
            try:
                # Check each symbol for trading signals
                for symbol in self.symbols:
                    await self._process_symbol_signals(symbol)
                
                # Wait for next iteration
                await asyncio.sleep(self.signal_interval)
                
            except Exception as e:
                self.logger.error(f"Error in trading loop: {e}")
                await asyncio.sleep(5)  # Short delay before retry
    
    async def _process_symbol_signals(self, symbol: str):
        """Process trading signals for a specific symbol."""
        try:
            self.logger.debug(f"Processing signals for {symbol}")
            
            # Generate signals for the symbol
            signals = await self.signal_generator.generate_signals()
            symbol_signals = [s for s in signals if s.get('symbol') == symbol]
            
            if not symbol_signals:
                self.logger.debug(f"No signals generated for {symbol}")
                return
            
            # Process each signal
            for signal in symbol_signals:
                await self._evaluate_and_execute_signal(symbol, signal)
                
        except Exception as e:
            self.logger.error(f"Error processing signals for {symbol}: {e}")
    
    async def _evaluate_and_execute_signal(self, symbol: str, signal: Dict[str, Any]):
        """Evaluate signal and execute trade if conditions are met."""
        try:
            # Extract signal data
            s_fused = signal.get('signal', 0.0)
            confidence = signal.get('confidence', 0.0)
            signal_strength = abs(s_fused)
            
            # Update stats
            self.trading_stats['total_signals'] += 1
            self.trading_stats['last_signal_time'] = datetime.now().isoformat()
            
            # Check signal thresholds
            if signal_strength < self.min_signal_strength or confidence < self.min_confidence:
                self.logger.debug(f"Signal below thresholds for {symbol}: strength={signal_strength:.3f}, confidence={confidence:.3f}")
                return
            
            self.trading_stats['signals_above_threshold'] += 1
            
            # Get current market data
            market_data = await self._get_market_data(symbol)
            if not market_data:
                self.logger.warning(f"Could not get market data for {symbol}")
                return
            
            # Prepare signal for execution
            execution_signal = {
                'symbol': symbol,
                's': s_fused,
                'confidence': confidence,
                'signal_strength': signal_strength,
                'components': signal.get('components', {}),
                'timestamp': signal.get('timestamp', datetime.now().isoformat())
            }
            
            # Execute trade signal
            self.logger.info(f"Executing trade signal for {symbol}: s_fused={s_fused:.3f}, confidence={confidence:.3f}")
            
            result = await self.futures_module.execute_trade_signal(execution_signal, market_data)
            
            # Update stats
            self.trading_stats['trades_executed'] += 1
            self.trading_stats['last_trade_time'] = datetime.now().isoformat()
            
            if result.get('success', False):
                self.trading_stats['trades_successful'] += 1
                self.logger.info(f"Trade executed successfully for {symbol}: {result.get('trade_id')}")
            else:
                self.trading_stats['trades_failed'] += 1
                self.logger.warning(f"Trade execution failed for {symbol}: {result.get('error')}")
                
        except Exception as e:
            self.logger.error(f"Error evaluating signal for {symbol}: {e}")
    
    async def _get_market_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current market data for a symbol."""
        try:
            # Get current price
            ticker_data = await self.bybit_api.get_ticker(symbol)
            if not ticker_data.get('success'):
                return None
                
            price = float(ticker_data['data'].get('lastPrice', 0))
            
            # Get ATR for volatility
            kline_data = await self.bybit_api.get_kline(symbol, '1h', 20)
            if not kline_data.get('success'):
                atr = 0.02 * price  # Default 2% of price
            else:
                # Calculate simple ATR
                closes = [float(k['close']) for k in kline_data['data']]
                highs = [float(k['high']) for k in kline_data['data']]
                lows = [float(k['low']) for k in kline_data['data']]
                
                tr_values = []
                for i in range(1, len(closes)):
                    tr = max(highs[i] - lows[i], abs(highs[i] - closes[i-1]), abs(lows[i] - closes[i-1]))
                    tr_values.append(tr)
                
                atr = sum(tr_values) / len(tr_values) if tr_values else 0.02 * price
            
            return {
                'price': price,
                'atr': atr,
                'symbol': symbol,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting market data for {symbol}: {e}")
            return None
    
    def get_trading_status(self) -> Dict[str, Any]:
        """Get current trading status."""
        return {
            'is_running': self.is_running,
            'trading_enabled': self.trading_enabled,
            'execution_mode': self.execution_mode.value,
            'symbols': self.symbols,
            'signal_interval': self.signal_interval,
            'stats': self.trading_stats,
            'futures_module_status': {
                'trading_enabled': self.futures_module.trading_enabled,
                'emergency_stop': self.futures_module.emergency_stop,
                'active_trades': len(self.futures_module.active_trades),
                'positions': len(self.futures_module.positions)
            }
        }
    
    def update_config(self, new_config: Dict[str, Any]):
        """Update trading configuration."""
        self.config.update(new_config)
        
        # Update parameters
        if 'signal_interval' in new_config:
            self.signal_interval = new_config['signal_interval']
        if 'min_signal_strength' in new_config:
            self.min_signal_strength = new_config['min_signal_strength']
        if 'min_confidence' in new_config:
            self.min_confidence = new_config['min_confidence']
        if 'trading_enabled' in new_config:
            self.trading_enabled = new_config['trading_enabled']
        
        self.logger.info(f"Configuration updated: {new_config}")
    
    def set_execution_mode(self, mode: ExecutionMode):
        """Set execution mode."""
        self.execution_mode = mode
        self.futures_module.execution_mode = mode
        self.logger.info(f"Execution mode set to: {mode.value}")


# Global instance for API integration
automatic_trading_service = None

def get_automatic_trading_service(config: Optional[Dict] = None) -> AutomaticTradingService:
    """Get or create the global automatic trading service instance."""
    global automatic_trading_service
    if automatic_trading_service is None:
        automatic_trading_service = AutomaticTradingService(config)
    return automatic_trading_service
