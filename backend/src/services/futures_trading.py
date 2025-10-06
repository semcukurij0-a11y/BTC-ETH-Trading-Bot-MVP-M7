"""
Futures Trading Module for Crypto Trading Bot

This module integrates risk management and order execution for futures trading:
- Position management and tracking
- Trade execution with risk controls
- Portfolio management
- Performance tracking
- Integration with signal generation
"""

import pandas as pd
import numpy as np
import logging
import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from .risk_management import RiskManagementModule
from .order_execution import OrderExecutionModule, ExecutionMode


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


class TradeStatus(Enum):
    PENDING = "pending"
    OPEN = "open"
    CLOSED = "closed"
    CANCELLED = "cancelled"


class FuturesTradingModule:
    """
    Futures Trading Module integrating risk management and order execution.
    """
    
    def __init__(self, 
                 config: Optional[Dict] = None,
                 execution_mode: ExecutionMode = ExecutionMode.SIMULATION):
        """
        Initialize Futures Trading Module.
        
        Args:
            config: Configuration dictionary
            execution_mode: Execution mode (simulation, paper, live)
        """
        self.config = config or {}
        self.execution_mode = execution_mode
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Initialize modules
        self.risk_module = RiskManagementModule(config)
        self.execution_module = OrderExecutionModule(config, self.risk_module, execution_mode)
        
        # Portfolio state
        self.portfolio = {
            'balance': self.config.get('initial_balance', 10000.0),
            'equity': self.config.get('initial_balance', 10000.0),
            'margin_used': 0.0,
            'free_margin': self.config.get('initial_balance', 10000.0),
            'unrealized_pnl': 0.0,
            'realized_pnl': 0.0
        }
        
        # Position tracking
        self.positions = {}
        self.position_history = []
        
        # Trade tracking
        self.active_trades = {}
        self.trade_history = []
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'average_win': 0.0,
            'average_loss': 0.0,
            'profit_factor': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'total_return': 0.0
        }
        
        # Market data cache
        self.market_data_cache = {}
        self.last_market_update = {}
        
        # Trading state
        self.trading_enabled = True
        self.emergency_stop = False
        
    async def execute_trade_signal(self, 
                                 signal: Dict[str, Any], 
                                 market_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute trade signal with comprehensive risk management.
        
        Args:
            signal: Trading signal from signal fusion
            market_data: Current market data
            
        Returns:
            Dictionary with trade execution result
        """
        try:
            # Check if trading is enabled
            if not self.trading_enabled or self.emergency_stop:
                return {
                    'success': False,
                    'error': 'Trading disabled or emergency stop active',
                    'trade_id': None
                }
            
            # Extract signal information
            symbol = signal.get('symbol', 'BTCUSDT')
            s_fused = signal.get('s', 0.0)
            confidence = signal.get('confidence', 0.0)
            signal_strength = signal.get('signal_strength', abs(s_fused))  # Use abs(s_fused) as fallback
            
            # Check signal strength and confidence
            min_confidence = self.config.get('min_confidence', 0.6)
            min_signal_strength = self.config.get('min_signal_strength', 0.3)
            
            if confidence < min_confidence or signal_strength < min_signal_strength:
                return {
                    'success': False,
                    'error': f'Signal below thresholds: confidence={confidence:.3f}, strength={signal_strength:.3f}',
                    'trade_id': None
                }
            
            # Determine trade direction
            if s_fused > 0.3:
                side = "Buy"
                position_side = PositionSide.LONG
            elif s_fused < -0.3:
                side = "Sell"
                position_side = PositionSide.SHORT
            else:
                return {
                    'success': False,
                    'error': f'Signal too weak: s_fused={s_fused:.3f}',
                    'trade_id': None
                }
            
            # Get current market data
            current_price = market_data.get('price', 0.0)
            current_atr = market_data.get('atr', 0.0)
            
            if current_price <= 0 or current_atr <= 0:
                return {
                    'success': False,
                    'error': 'Invalid market data',
                    'trade_id': None
                }
            
            # Check existing position
            existing_position = self.positions.get(symbol)
            if existing_position:
                # Handle position management
                return await self._manage_existing_position(
                    symbol, existing_position, signal, market_data
                )
            
            # Calculate position size
            position_size_result = self.risk_module.calculate_position_size(
                symbol=symbol,
                entry_price=current_price,
                stop_loss_price=0.0,  # Will be calculated below
                account_balance=self.portfolio['balance'],
                current_atr=current_atr
            )
            
            if not position_size_result.get('quantity', 0) > 0:
                return {
                    'success': False,
                    'error': 'Position size calculation failed',
                    'trade_id': None
                }
            
            # Calculate stop loss and take profit
            sl_tp_result = self.risk_module.calculate_stop_loss_take_profit(
                symbol=symbol,
                entry_price=current_price,
                side=side,
                current_atr=current_atr
            )
            
            # Update position size with actual stop loss
            position_size_result = self.risk_module.calculate_position_size(
                symbol=symbol,
                entry_price=current_price,
                stop_loss_price=sl_tp_result['stop_loss'],
                account_balance=self.portfolio['balance'],
                current_atr=current_atr
            )
            
            # Update leverage based on volatility
            self.risk_module.update_leverage(symbol, current_atr, current_price)
            
            # Place entry order
            order_result = await self.execution_module.place_order(
                symbol=symbol,
                side=side,
                order_type="Market",
                quantity=position_size_result['quantity'],
                reduce_only=False
            )
            
            if not order_result['success']:
                return {
                    'success': False,
                    'error': f'Order placement failed: {order_result["error"]}',
                    'trade_id': None
                }
            
            # Create trade record
            trade_id = self._generate_trade_id(symbol, side)
            trade = {
                'trade_id': trade_id,
                'symbol': symbol,
                'side': side,
                'position_side': position_side.value,
                'entry_price': current_price,
                'quantity': position_size_result['quantity'],
                'stop_loss': sl_tp_result['stop_loss'],
                'take_profit': sl_tp_result['take_profit'],
                'entry_order_id': order_result['order_id'],
                'status': TradeStatus.PENDING.value,
                'signal': signal,
                'market_data': market_data,
                'position_size': position_size_result,
                'sl_tp': sl_tp_result,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            # Track trade
            self.active_trades[trade_id] = trade
            
            # Update portfolio
            self._update_portfolio_on_trade(trade, 'entry')
            
            self.logger.info(f"Trade signal executed: {trade}")
            
            return {
                'success': True,
                'trade_id': trade_id,
                'trade': trade,
                'order_result': order_result
            }
            
        except Exception as e:
            self.logger.error(f"Error executing trade signal: {e}")
            return {
                'success': False,
                'error': str(e),
                'trade_id': None
            }
    
    async def _manage_existing_position(self, 
                                      symbol: str, 
                                      position: Dict, 
                                      signal: Dict, 
                                      market_data: Dict) -> Dict[str, Any]:
        """
        Manage existing position based on new signal.
        
        Args:
            symbol: Trading symbol
            position: Existing position
            signal: New trading signal
            market_data: Current market data
            
        Returns:
            Dictionary with position management result
        """
        try:
            current_price = market_data.get('price', 0.0)
            s_fused = signal.get('s', 0.0)
            confidence = signal.get('confidence', 0.0)
            
            # Check exit conditions
            exit_conditions = self._check_exit_conditions(position, current_price, s_fused, confidence)
            
            if exit_conditions['should_exit']:
                # Close position
                return await self._close_position(symbol, position, exit_conditions['reason'])
            
            # Check for position adjustment
            adjustment_conditions = self._check_position_adjustment(position, signal, market_data)
            
            if adjustment_conditions['should_adjust']:
                # Adjust position
                return await self._adjust_position(symbol, position, adjustment_conditions)
            
            return {
                'success': True,
                'action': 'hold',
                'reason': 'No exit or adjustment conditions met',
                'position': position
            }
            
        except Exception as e:
            self.logger.error(f"Error managing existing position for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _check_exit_conditions(self, 
                             position: Dict, 
                             current_price: float, 
                             s_fused: float, 
                             confidence: float) -> Dict[str, Any]:
        """
        Check if position should be closed.
        
        Args:
            position: Position information
            current_price: Current market price
            s_fused: Fused signal value
            confidence: Signal confidence
            
        Returns:
            Dictionary with exit conditions
        """
        try:
            entry_price = position['entry_price']
            stop_loss = position['stop_loss']
            take_profit = position['take_profit']
            side = position['side']
            
            # Check stop loss
            if side == "Buy" and current_price <= stop_loss:
                return {
                    'should_exit': True,
                    'reason': 'Stop loss hit',
                    'exit_price': stop_loss
                }
            elif side == "Sell" and current_price >= stop_loss:
                return {
                    'should_exit': True,
                    'reason': 'Stop loss hit',
                    'exit_price': stop_loss
                }
            
            # Check take profit
            if side == "Buy" and current_price >= take_profit:
                return {
                    'should_exit': True,
                    'reason': 'Take profit hit',
                    'exit_price': take_profit
                }
            elif side == "Sell" and current_price <= take_profit:
                return {
                    'should_exit': True,
                    'reason': 'Take profit hit',
                    'exit_price': take_profit
                }
            
            # Check opposite signal
            exit_threshold = self.config.get('exit_threshold', 0.2)
            if side == "Buy" and s_fused < -exit_threshold and confidence > 0.5:
                return {
                    'should_exit': True,
                    'reason': 'Opposite signal',
                    'exit_price': current_price
                }
            elif side == "Sell" and s_fused > exit_threshold and confidence > 0.5:
                return {
                    'should_exit': True,
                    'reason': 'Opposite signal',
                    'exit_price': current_price
                }
            
            return {
                'should_exit': False,
                'reason': 'No exit conditions met'
            }
            
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
            return {
                'should_exit': False,
                'reason': f'Error: {str(e)}'
            }
    
    def _check_position_adjustment(self, 
                                 position: Dict, 
                                 signal: Dict, 
                                 market_data: Dict) -> Dict[str, Any]:
        """
        Check if position should be adjusted.
        
        Args:
            position: Position information
            signal: New trading signal
            market_data: Current market data
            
        Returns:
            Dictionary with adjustment conditions
        """
        try:
            # For now, no position adjustments
            # This could be extended for partial closes, trailing stops, etc.
            
            return {
                'should_adjust': False,
                'reason': 'No adjustment conditions implemented'
            }
            
        except Exception as e:
            self.logger.error(f"Error checking position adjustment: {e}")
            return {
                'should_adjust': False,
                'reason': f'Error: {str(e)}'
            }
    
    async def _close_position(self, 
                            symbol: str, 
                            position: Dict, 
                            reason: str) -> Dict[str, Any]:
        """
        Close existing position.
        
        Args:
            symbol: Trading symbol
            position: Position information
            reason: Reason for closing
            
        Returns:
            Dictionary with position close result
        """
        try:
            # Determine exit side (opposite of entry)
            exit_side = "Sell" if position['side'] == "Buy" else "Buy"
            
            # Place exit order
            order_result = await self.execution_module.place_order(
                symbol=symbol,
                side=exit_side,
                order_type="Market",
                quantity=position['quantity'],
                reduce_only=True
            )
            
            if not order_result['success']:
                return {
                    'success': False,
                    'error': f'Exit order placement failed: {order_result["error"]}',
                    'trade_id': position.get('trade_id')
                }
            
            # Update position
            position['exit_order_id'] = order_result['order_id']
            position['exit_reason'] = reason
            position['status'] = TradeStatus.CLOSED.value
            position['closed_at'] = datetime.now().isoformat()
            
            # Calculate P&L
            pnl = self._calculate_trade_pnl(position)
            position['pnl'] = pnl
            position['pnl_percent'] = (pnl / (position['entry_price'] * position['quantity'])) * 100
            
            # Update trade history
            self.trade_history.append(position)
            
            # Remove from active trades
            trade_id = position.get('trade_id')
            if trade_id and trade_id in self.active_trades:
                del self.active_trades[trade_id]
            
            # Update portfolio
            self._update_portfolio_on_trade(position, 'exit')
            
            # Update performance metrics
            self._update_performance_metrics(position)
            
            # Update risk module
            self.risk_module.update_trade_result(symbol, pnl, pnl > 0)
            
            self.logger.info(f"Position closed: {position}")
            
            return {
                'success': True,
                'trade_id': trade_id,
                'position': position,
                'order_result': order_result,
                'pnl': pnl
            }
            
        except Exception as e:
            self.logger.error(f"Error closing position for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e),
                'trade_id': position.get('trade_id')
            }
    
    async def _adjust_position(self, 
                             symbol: str, 
                             position: Dict, 
                             adjustment_conditions: Dict) -> Dict[str, Any]:
        """
        Adjust existing position.
        
        Args:
            symbol: Trading symbol
            position: Position information
            adjustment_conditions: Adjustment conditions
            
        Returns:
            Dictionary with position adjustment result
        """
        try:
            # For now, no position adjustments implemented
            # This could be extended for partial closes, trailing stops, etc.
            
            return {
                'success': True,
                'action': 'no_adjustment',
                'reason': 'Position adjustment not implemented'
            }
            
        except Exception as e:
            self.logger.error(f"Error adjusting position for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_trade_pnl(self, position: Dict) -> float:
        """
        Calculate trade P&L.
        
        Args:
            position: Position information
            
        Returns:
            Trade P&L
        """
        try:
            entry_price = position['entry_price']
            exit_price = position.get('exit_price', 0.0)
            quantity = position['quantity']
            side = position['side']
            
            if exit_price <= 0:
                return 0.0
            
            if side == "Buy":
                # Long position
                pnl = (exit_price - entry_price) * quantity
            else:
                # Short position
                pnl = (entry_price - exit_price) * quantity
            
            return pnl
            
        except Exception as e:
            self.logger.error(f"Error calculating trade P&L: {e}")
            return 0.0
    
    def _update_portfolio_on_trade(self, trade: Dict, action: str):
        """
        Update portfolio based on trade action.
        
        Args:
            trade: Trade information
            action: 'entry' or 'exit'
        """
        try:
            if action == 'entry':
                # Update margin used
                margin_required = trade['position_size']['margin_required']
                self.portfolio['margin_used'] += margin_required
                self.portfolio['free_margin'] -= margin_required
                
            elif action == 'exit':
                # Update realized P&L
                pnl = trade.get('pnl', 0.0)
                self.portfolio['realized_pnl'] += pnl
                self.portfolio['balance'] += pnl
                
                # Free up margin
                margin_required = trade['position_size']['margin_required']
                self.portfolio['margin_used'] -= margin_required
                self.portfolio['free_margin'] += margin_required
            
            # Update equity
            self.portfolio['equity'] = (
                self.portfolio['balance'] + 
                self.portfolio['unrealized_pnl']
            )
            
        except Exception as e:
            self.logger.error(f"Error updating portfolio: {e}")
    
    def _update_performance_metrics(self, trade: Dict):
        """
        Update performance metrics based on trade result.
        
        Args:
            trade: Trade information
        """
        try:
            self.performance_metrics['total_trades'] += 1
            
            pnl = trade.get('pnl', 0.0)
            if pnl > 0:
                self.performance_metrics['winning_trades'] += 1
                self.performance_metrics['average_win'] = (
                    (self.performance_metrics['average_win'] * (self.performance_metrics['winning_trades'] - 1) + pnl) /
                    self.performance_metrics['winning_trades']
                )
            else:
                self.performance_metrics['losing_trades'] += 1
                self.performance_metrics['average_loss'] = (
                    (self.performance_metrics['average_loss'] * (self.performance_metrics['losing_trades'] - 1) + abs(pnl)) /
                    self.performance_metrics['losing_trades']
                )
            
            # Calculate win rate
            if self.performance_metrics['total_trades'] > 0:
                self.performance_metrics['win_rate'] = (
                    self.performance_metrics['winning_trades'] / 
                    self.performance_metrics['total_trades']
                )
            
            # Calculate profit factor
            if self.performance_metrics['average_loss'] > 0:
                self.performance_metrics['profit_factor'] = (
                    self.performance_metrics['average_win'] / 
                    self.performance_metrics['average_loss']
                )
            
            # Calculate total return
            initial_balance = self.config.get('initial_balance', 10000.0)
            self.performance_metrics['total_return'] = (
                (self.portfolio['balance'] - initial_balance) / initial_balance
            )
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def _generate_trade_id(self, symbol: str, side: str) -> str:
        """
        Generate unique trade ID.
        
        Args:
            symbol: Trading symbol
            side: Trade side
            
        Returns:
            Unique trade ID
        """
        timestamp = int(datetime.now().timestamp() * 1000)
        return f"{symbol}_{side}_{timestamp}"
    
    async def update_market_data(self, symbol: str, market_data: Dict):
        """
        Update market data cache.
        
        Args:
            symbol: Trading symbol
            market_data: Market data
        """
        try:
            self.market_data_cache[symbol] = market_data
            self.last_market_update[symbol] = datetime.now()
            
            # Update unrealized P&L for active positions
            if symbol in self.positions:
                position = self.positions[symbol]
                current_price = market_data.get('price', 0.0)
                
                if current_price > 0:
                    # Calculate unrealized P&L
                    entry_price = position['entry_price']
                    quantity = position['quantity']
                    side = position['side']
                    
                    if side == "Buy":
                        unrealized_pnl = (current_price - entry_price) * quantity
                    else:
                        unrealized_pnl = (entry_price - current_price) * quantity
                    
                    position['unrealized_pnl'] = unrealized_pnl
                    position['current_price'] = current_price
                    position['updated_at'] = datetime.now().isoformat()
                    
                    # Update portfolio unrealized P&L
                    self.portfolio['unrealized_pnl'] = sum(
                        pos.get('unrealized_pnl', 0.0) for pos in self.positions.values()
                    )
                    self.portfolio['equity'] = (
                        self.portfolio['balance'] + 
                        self.portfolio['unrealized_pnl']
                    )
            
        except Exception as e:
            self.logger.error(f"Error updating market data for {symbol}: {e}")
    
    async def emergency_stop(self, reason: str = "Manual emergency stop"):
        """
        Trigger emergency stop.
        
        Args:
            reason: Reason for emergency stop
        """
        try:
            self.emergency_stop = True
            self.trading_enabled = False
            
            self.logger.critical(f"Emergency stop triggered: {reason}")
            
            # Close all positions
            for symbol, position in self.positions.items():
                try:
                    await self._close_position(symbol, position, f"Emergency stop: {reason}")
                except Exception as e:
                    self.logger.error(f"Error closing position during emergency stop: {e}")
            
            # Cancel all pending orders
            for trade_id, trade in self.active_trades.items():
                try:
                    if 'entry_order_id' in trade:
                        await self.execution_module.cancel_order(
                            trade['entry_order_id'], 
                            trade['symbol']
                        )
                except Exception as e:
                    self.logger.error(f"Error cancelling order during emergency stop: {e}")
            
        except Exception as e:
            self.logger.error(f"Error during emergency stop: {e}")
    
    def get_portfolio_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive portfolio summary.
        
        Returns:
            Dictionary with portfolio summary
        """
        try:
            # Get risk summary
            risk_summary = self.risk_module.get_risk_summary()
            
            # Get execution summary
            execution_summary = self.execution_module.get_execution_summary()
            
            # Calculate additional metrics
            total_positions = len(self.positions)
            active_trades = len(self.active_trades)
            
            # Calculate portfolio metrics
            initial_balance = self.config.get('initial_balance', 10000.0)
            total_return = (self.portfolio['balance'] - initial_balance) / initial_balance
            
            # Calculate Sharpe ratio (simplified)
            sharpe_ratio = 0.0
            if self.performance_metrics['total_trades'] > 0:
                avg_return = total_return / self.performance_metrics['total_trades']
                # Simplified calculation - would need actual returns data
                sharpe_ratio = avg_return / 0.1  # Assuming 10% volatility
            
            summary = {
                'portfolio': self.portfolio,
                'positions': {
                    'total': total_positions,
                    'active_trades': active_trades,
                    'details': list(self.positions.values())
                },
                'performance': {
                    **self.performance_metrics,
                    'total_return': total_return,
                    'sharpe_ratio': sharpe_ratio
                },
                'trading_state': {
                    'trading_enabled': self.trading_enabled,
                    'emergency_stop': self.emergency_stop,
                    'execution_mode': self.execution_mode.value
                },
                'risk_summary': risk_summary,
                'execution_summary': execution_summary,
                'market_data': {
                    'cached_symbols': list(self.market_data_cache.keys()),
                    'last_updates': self.last_market_update
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting portfolio summary: {e}")
            return {
                'error': str(e)
            }
    
    async def close(self):
        """Close the futures trading module and cleanup resources."""
        try:
            await self.execution_module.close()
            self.logger.info("Futures trading module closed")
        except Exception as e:
            self.logger.error(f"Error closing futures trading module: {e}")


def main():
    """Main execution function for testing."""
    import asyncio
    
    async def test_futures_trading():
        # Configuration
        config = {
            'initial_balance': 10000.0,
            'min_confidence': 0.6,
            'min_signal_strength': 0.3,
            'exit_threshold': 0.2,
            'risk_per_trade': 0.0075,
            'max_leverage': 5.0,
            'atr_sl_multiplier': 2.5,
            'atr_tp_multiplier': 3.0
        }
        
        # Initialize futures trading module
        futures_module = FuturesTradingModule(
            config=config,
            execution_mode=ExecutionMode.SIMULATION
        )
        
        try:
            # Test trade signal execution
            signal = {
                'symbol': 'BTCUSDT',
                's': 0.5,
                'confidence': 0.7,
                'signal_strength': 0.4
            }
            
            market_data = {
                'price': 50000.0,
                'atr': 500.0
            }
            
            # Update market data
            await futures_module.update_market_data('BTCUSDT', market_data)
            
            # Execute trade signal
            trade_result = await futures_module.execute_trade_signal(signal, market_data)
            print("Trade Execution Result:", trade_result)
            
            # Get portfolio summary
            portfolio_summary = futures_module.get_portfolio_summary()
            print("Portfolio Summary:", portfolio_summary)
            
        finally:
            await futures_module.close()
    
    # Run test
    asyncio.run(test_futures_trading())


if __name__ == "__main__":
    main()



