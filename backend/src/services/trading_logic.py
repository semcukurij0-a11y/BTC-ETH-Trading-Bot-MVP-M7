"""
Trading Logic Module for Crypto Trading Bot

This module implements trading execution logic with entry/exit rules.
Features:
- Entry signals based on fused signals and confidence thresholds
- Exit signals on opposite signal or SL/TP
- Position management and risk control
- Trade execution simulation
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class PositionType(Enum):
    """Position type enumeration."""
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"


class TradeStatus(Enum):
    """Trade status enumeration."""
    OPEN = "OPEN"
    CLOSED = "CLOSED"
    STOPPED = "STOPPED"


class TradingLogicModule:
    """
    Trading Logic Module for executing trades based on fused signals.
    
    Implements:
    - Entry/exit signal generation
    - Position management
    - Stop loss and take profit
    - Risk management
    - Trade execution simulation
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Trading Logic Module.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Trading parameters
        self.min_confidence_threshold = self.config.get('min_confidence_threshold', 0.6)
        self.min_signal_strength = self.config.get('min_signal_strength', 0.3)
        
        # Risk management
        self.stop_loss_percent = self.config.get('stop_loss_percent', 0.02)  # 2%
        self.take_profit_percent = self.config.get('take_profit_percent', 0.04)  # 4%
        self.max_position_size = self.config.get('max_position_size', 1.0)  # 100% of capital
        self.max_daily_trades = self.config.get('max_daily_trades', 10)
        
        # Position management
        self.current_position = PositionType.HOLD
        self.current_trade = None
        self.daily_trade_count = 0
        self.last_trade_date = None
        
        # Trade history
        self.trade_history = []
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.0
        }
    
    def should_enter_long(self, signal_data: Dict[str, Any], current_price: float) -> bool:
        """
        Determine if should enter long position.
        
        Args:
            signal_data: Dictionary with signal information
            current_price: Current market price
            
        Returns:
            True if should enter long position
        """
        # Check confidence threshold
        if signal_data.get('s_fused_confidence', 0) < self.min_confidence_threshold:
            return False
        
        # Check signal strength
        if signal_data.get('s_fused_strength', 0) < self.min_signal_strength:
            return False
        
        # Check signal direction
        if signal_data.get('s_fused', 0) <= 0:
            return False
        
        # Check if already in position
        if self.current_position != PositionType.HOLD:
            return False
        
        # Check daily trade limit
        if self._is_daily_trade_limit_reached():
            return False
        
        return True
    
    def should_enter_short(self, signal_data: Dict[str, Any], current_price: float) -> bool:
        """
        Determine if should enter short position.
        
        Args:
            signal_data: Dictionary with signal information
            current_price: Current market price
            
        Returns:
            True if should enter short position
        """
        # Check confidence threshold
        if signal_data.get('s_fused_confidence', 0) < self.min_confidence_threshold:
            return False
        
        # Check signal strength
        if signal_data.get('s_fused_strength', 0) < self.min_signal_strength:
            return False
        
        # Check signal direction
        if signal_data.get('s_fused', 0) >= 0:
            return False
        
        # Check if already in position
        if self.current_position != PositionType.HOLD:
            return False
        
        # Check daily trade limit
        if self._is_daily_trade_limit_reached():
            return False
        
        return True
    
    def should_exit_position(self, signal_data: Dict[str, Any], current_price: float) -> Tuple[bool, str]:
        """
        Determine if should exit current position.
        
        Args:
            signal_data: Dictionary with signal information
            current_price: Current market price
            
        Returns:
            Tuple of (should_exit, reason)
        """
        if self.current_position == PositionType.HOLD or self.current_trade is None:
            return False, "No position to exit"
        
        # Check stop loss
        if self._check_stop_loss(current_price):
            return True, "Stop loss triggered"
        
        # Check take profit
        if self._check_take_profit(current_price):
            return True, "Take profit triggered"
        
        # Check opposite signal
        if self._check_opposite_signal(signal_data):
            return True, "Opposite signal"
        
        return False, "Hold position"
    
    def _check_stop_loss(self, current_price: float) -> bool:
        """Check if stop loss is triggered."""
        if self.current_trade is None:
            return False
        
        entry_price = self.current_trade['entry_price']
        
        if self.current_position == PositionType.LONG:
            stop_price = entry_price * (1 - self.stop_loss_percent)
            return current_price <= stop_price
        elif self.current_position == PositionType.SHORT:
            stop_price = entry_price * (1 + self.stop_loss_percent)
            return current_price >= stop_price
        
        return False
    
    def _check_take_profit(self, current_price: float) -> bool:
        """Check if take profit is triggered."""
        if self.current_trade is None:
            return False
        
        entry_price = self.current_trade['entry_price']
        
        if self.current_position == PositionType.LONG:
            tp_price = entry_price * (1 + self.take_profit_percent)
            return current_price >= tp_price
        elif self.current_position == PositionType.SHORT:
            tp_price = entry_price * (1 - self.take_profit_percent)
            return current_price <= tp_price
        
        return False
    
    def _check_opposite_signal(self, signal_data: Dict[str, Any]) -> bool:
        """Check if opposite signal is received."""
        signal_value = signal_data.get('s_fused', 0)
        
        if self.current_position == PositionType.LONG:
            return signal_value < -0.2  # Strong negative signal
        elif self.current_position == PositionType.SHORT:
            return signal_value > 0.2  # Strong positive signal
        
        return False
    
    def _is_daily_trade_limit_reached(self) -> bool:
        """Check if daily trade limit is reached."""
        today = datetime.now().date()
        
        if self.last_trade_date != today:
            self.daily_trade_count = 0
            self.last_trade_date = today
        
        return self.daily_trade_count >= self.max_daily_trades
    
    def enter_long_position(self, signal_data: Dict[str, Any], current_price: float, 
                          timestamp: datetime = None) -> Dict[str, Any]:
        """
        Enter long position.
        
        Args:
            signal_data: Dictionary with signal information
            current_price: Current market price
            timestamp: Trade timestamp
            
        Returns:
            Trade information dictionary
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate position size
        position_size = self.max_position_size
        
        # Create trade record
        trade = {
            'trade_id': len(self.trade_history) + 1,
            'entry_time': timestamp,
            'entry_price': current_price,
            'position_type': PositionType.LONG.value,
            'position_size': position_size,
            'stop_loss': current_price * (1 - self.stop_loss_percent),
            'take_profit': current_price * (1 + self.take_profit_percent),
            'status': TradeStatus.OPEN.value,
            'exit_time': None,
            'exit_price': None,
            'exit_reason': None,
            'pnl': 0.0,
            'pnl_percent': 0.0,
            'signal_data': signal_data.copy()
        }
        
        # Update state
        self.current_position = PositionType.LONG
        self.current_trade = trade
        self.daily_trade_count += 1
        
        self.logger.info(f"Entered LONG position at {current_price:.2f}")
        
        return trade
    
    def enter_short_position(self, signal_data: Dict[str, Any], current_price: float,
                           timestamp: datetime = None) -> Dict[str, Any]:
        """
        Enter short position.
        
        Args:
            signal_data: Dictionary with signal information
            current_price: Current market price
            timestamp: Trade timestamp
            
        Returns:
            Trade information dictionary
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate position size
        position_size = self.max_position_size
        
        # Create trade record
        trade = {
            'trade_id': len(self.trade_history) + 1,
            'entry_time': timestamp,
            'entry_price': current_price,
            'position_type': PositionType.SHORT.value,
            'position_size': position_size,
            'stop_loss': current_price * (1 + self.stop_loss_percent),
            'take_profit': current_price * (1 - self.take_profit_percent),
            'status': TradeStatus.OPEN.value,
            'exit_time': None,
            'exit_price': None,
            'exit_reason': None,
            'pnl': 0.0,
            'pnl_percent': 0.0,
            'signal_data': signal_data.copy()
        }
        
        # Update state
        self.current_position = PositionType.SHORT
        self.current_trade = trade
        self.daily_trade_count += 1
        
        self.logger.info(f"Entered SHORT position at {current_price:.2f}")
        
        return trade
    
    def exit_position(self, current_price: float, exit_reason: str,
                     timestamp: datetime = None) -> Dict[str, Any]:
        """
        Exit current position.
        
        Args:
            current_price: Current market price
            exit_reason: Reason for exit
            timestamp: Exit timestamp
            
        Returns:
            Updated trade information
        """
        if self.current_trade is None:
            return {}
        
        if timestamp is None:
            timestamp = datetime.now()
        
        # Calculate PnL
        entry_price = self.current_trade['entry_price']
        
        if self.current_position == PositionType.LONG:
            pnl_percent = (current_price - entry_price) / entry_price
        elif self.current_position == PositionType.SHORT:
            pnl_percent = (entry_price - current_price) / entry_price
        else:
            pnl_percent = 0.0
        
        pnl = pnl_percent * self.current_trade['position_size']
        
        # Update trade record
        self.current_trade.update({
            'exit_time': timestamp,
            'exit_price': current_price,
            'exit_reason': exit_reason,
            'status': TradeStatus.CLOSED.value,
            'pnl': pnl,
            'pnl_percent': pnl_percent
        })
        
        # Add to trade history
        self.trade_history.append(self.current_trade.copy())
        
        # Update performance metrics
        self._update_performance_metrics(self.current_trade)
        
        self.logger.info(f"Exited {self.current_position.value} position at {current_price:.2f}, "
                        f"PnL: {pnl_percent:.2%}, Reason: {exit_reason}")
        
        # Reset state
        self.current_position = PositionType.HOLD
        self.current_trade = None
        
        return self.trade_history[-1]
    
    def _update_performance_metrics(self, trade: Dict[str, Any]):
        """Update performance metrics."""
        self.performance_metrics['total_trades'] += 1
        
        if trade['pnl'] > 0:
            self.performance_metrics['winning_trades'] += 1
        else:
            self.performance_metrics['losing_trades'] += 1
        
        self.performance_metrics['total_pnl'] += trade['pnl']
        
        # Calculate win rate
        if self.performance_metrics['total_trades'] > 0:
            self.performance_metrics['win_rate'] = (
                self.performance_metrics['winning_trades'] / 
                self.performance_metrics['total_trades']
            )
        
        # Update max drawdown (simplified)
        if trade['pnl'] < 0:
            self.performance_metrics['max_drawdown'] = min(
                self.performance_metrics['max_drawdown'], 
                trade['pnl']
            )
    
    def process_trading_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process trading signals and generate trade decisions.
        
        Args:
            df: DataFrame with signal data and prices
            
        Returns:
            DataFrame with trading decisions
        """
        # Initialize trading columns
        df['trade_action'] = 'HOLD'
        df['trade_reason'] = ''
        df['position'] = 'HOLD'
        df['trade_pnl'] = 0.0
        
        if df.empty:
            return df
        
        for idx, row in df.iterrows():
            current_price = row.get('close', 0)
            signal_data = {
                's_fused': row.get('s_fused', 0),
                's_fused_confidence': row.get('s_fused_confidence', 0),
                's_fused_strength': row.get('s_fused_strength', 0)
            }
            
            # Check for exit first
            if self.current_position != PositionType.HOLD:
                should_exit, exit_reason = self.should_exit_position(signal_data, current_price)
                if should_exit:
                    self.exit_position(current_price, exit_reason, row.get('open_time'))
                    df.at[idx, 'trade_action'] = 'EXIT'
                    df.at[idx, 'trade_reason'] = exit_reason
                    df.at[idx, 'position'] = 'HOLD'
                    continue
            
            # Check for entry
            if self.current_position == PositionType.HOLD:
                if self.should_enter_long(signal_data, current_price):
                    self.enter_long_position(signal_data, current_price, row.get('open_time'))
                    df.at[idx, 'trade_action'] = 'ENTER_LONG'
                    df.at[idx, 'trade_reason'] = 'Long signal'
                    df.at[idx, 'position'] = 'LONG'
                elif self.should_enter_short(signal_data, current_price):
                    self.enter_short_position(signal_data, current_price, row.get('open_time'))
                    df.at[idx, 'trade_action'] = 'ENTER_SHORT'
                    df.at[idx, 'trade_reason'] = 'Short signal'
                    df.at[idx, 'position'] = 'SHORT'
                else:
                    df.at[idx, 'trade_action'] = 'HOLD'
                    df.at[idx, 'trade_reason'] = 'No signal'
                    df.at[idx, 'position'] = 'HOLD'
            else:
                df.at[idx, 'trade_action'] = 'HOLD'
                df.at[idx, 'trade_reason'] = 'In position'
                df.at[idx, 'position'] = self.current_position.value
        
        return df
    
    def get_current_position_info(self) -> Dict[str, Any]:
        """Get current position information."""
        if self.current_trade is None:
            return {
                'position': 'HOLD',
                'trade_id': None,
                'entry_price': None,
                'current_pnl': 0.0,
                'current_pnl_percent': 0.0
            }
        
        return {
            'position': self.current_position.value,
            'trade_id': self.current_trade['trade_id'],
            'entry_price': self.current_trade['entry_price'],
            'entry_time': self.current_trade['entry_time'],
            'stop_loss': self.current_trade['stop_loss'],
            'take_profit': self.current_trade['take_profit'],
            'position_size': self.current_trade['position_size']
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get trading performance summary."""
        return self.performance_metrics.copy()
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get complete trade history."""
        return self.trade_history.copy()
    
    def reset_trading_state(self):
        """Reset trading state."""
        self.current_position = PositionType.HOLD
        self.current_trade = None
        self.daily_trade_count = 0
        self.last_trade_date = None
        self.logger.info("Trading state reset")


def main():
    """Test the Trading Logic Module."""
    import sys
    sys.path.append('src')
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Configuration
    config = {
        'min_confidence_threshold': 0.6,
        'min_signal_strength': 0.3,
        'stop_loss_percent': 0.02,
        'take_profit_percent': 0.04,
        'max_position_size': 1.0,
        'max_daily_trades': 10
    }
    
    # Initialize trading logic module
    trading_module = TradingLogicModule(config=config)
    
    # Create sample data for testing
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='H')
    np.random.seed(42)
    
    sample_data = pd.DataFrame({
        'open_time': dates,
        'close': 50000 + np.cumsum(np.random.randn(len(dates)) * 100),
        's_fused': np.random.uniform(-1, 1, len(dates)),
        's_fused_confidence': np.random.uniform(0.5, 1.0, len(dates)),
        's_fused_strength': np.random.uniform(0.3, 1.0, len(dates))
    })
    
    # Process trading signals
    result_df = trading_module.process_trading_signals(sample_data)
    
    # Get current position
    position_info = trading_module.get_current_position_info()
    print(f"Current Position: {position_info}")
    
    # Get performance summary
    performance = trading_module.get_performance_summary()
    print(f"Performance: {performance}")
    
    # Count trades
    trades = result_df[result_df['trade_action'] != 'HOLD']
    print(f"Total trades executed: {len(trades)}")
    print(f"Trade actions: {trades['trade_action'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
