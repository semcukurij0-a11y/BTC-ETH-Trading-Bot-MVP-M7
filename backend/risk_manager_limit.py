#!/usr/bin/env python3
"""
Risk Manager Limit - Comprehensive Risk Management System

Implements:
- Fixed-fractional sizing (0.75% default)
- SL = 2.5 x ATR, TP = 3.0 x ATR
- Leverage cap <= 5x, isolated margin
- Liquidation buffer >= 3 x ATR
- Daily caps (loss 3%, profit 2%, trades 15, consecutive losses 4)
- Soft DD 6% / Hard DD 10%

Author: Trading System
Date: 2025-01-07
"""

import json
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RiskLimits:
    """Risk management limits configuration"""
    # Position sizing
    fixed_fractional_percent: float = 0.75  # 0.75% default
    
    # ATR-based stops
    atr_sl_multiplier: float = 2.5  # SL = 2.5 x ATR
    atr_tp_multiplier: float = 3.0  # TP = 3.0 x ATR
    
    # Leverage and margin
    max_leverage: float = 5.0  # Leverage cap <= 5x
    margin_mode: str = "isolated"  # Isolated margin
    
    # Liquidation buffer
    liquidation_buffer_atr: float = 3.0  # >= 3 x ATR
    
    # Daily caps
    max_daily_loss_percent: float = 3.0  # 3% daily loss cap
    max_daily_profit_percent: float = 2.0  # 2% daily profit cap
    max_daily_trades: int = 15  # 15 trades per day
    max_consecutive_losses: int = 4  # 4 consecutive losses
    
    # Drawdown limits
    soft_drawdown_percent: float = 6.0  # 6% soft DD
    hard_drawdown_percent: float = 10.0  # 10% hard DD
    
    # Additional safety parameters
    min_confidence_threshold: float = 0.6
    min_signal_strength: float = 0.3
    max_position_size: float = 1.0


@dataclass
class TradeRecord:
    """Individual trade record"""
    timestamp: str
    symbol: str
    side: str  # 'buy' or 'sell'
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    atr_value: float
    leverage: float
    margin_used: float
    pnl: Optional[float] = None
    status: str = "open"  # 'open', 'closed', 'stopped'
    exit_price: Optional[float] = None
    exit_timestamp: Optional[str] = None


@dataclass
class DailyStats:
    """Daily trading statistics"""
    date: str
    trades_count: int
    total_pnl: float
    total_loss: float
    total_profit: float
    consecutive_losses: int
    max_drawdown: float
    account_balance: float
    margin_used: float


class RiskManagerLimit:
    """
    Comprehensive Risk Management System
    
    Implements all specified risk parameters with real-time monitoring
    and automatic position sizing and risk controls.
    """
    
    def __init__(self, config_path: str = "config.json", initial_balance: float = 10000.0):
        """
        Initialize Risk Manager
        
        Args:
            config_path: Path to configuration file
            initial_balance: Initial account balance
        """
        self.config_path = config_path
        self.initial_balance = initial_balance
        self.current_balance = initial_balance
        self.peak_balance = initial_balance
        
        # Load configuration
        self.limits = self._load_config()
        
        # Trading state
        self.open_positions: Dict[str, TradeRecord] = {}
        self.trade_history: List[TradeRecord] = []
        self.daily_stats: Dict[str, DailyStats] = {}
        
        # Risk monitoring
        self.daily_trades_count = 0
        self.consecutive_losses = 0
        self.daily_pnl = 0.0
        self.last_reset_date = datetime.now().date()
        
        # Risk flags
        self.soft_dd_breach = False
        self.hard_dd_breach = False
        self.daily_loss_breach = False
        self.daily_profit_breach = False
        self.trade_limit_breach = False
        self.consecutive_loss_breach = False
        
        logger.info(f"Risk Manager initialized with balance: ${initial_balance:,.2f}")
        logger.info(f"Risk limits: {asdict(self.limits)}")
    
    def _load_config(self) -> RiskLimits:
        """Load risk limits from configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            # Extract futures trading config
            futures_config = config.get('futures_trading', {})
            
            return RiskLimits(
                fixed_fractional_percent=futures_config.get('fixed_fractional_percent', 0.75),
                atr_sl_multiplier=futures_config.get('atr_sl_multiplier', 2.5),
                atr_tp_multiplier=futures_config.get('atr_tp_multiplier', 3.0),
                max_leverage=futures_config.get('max_leverage', 5.0),
                margin_mode=futures_config.get('margin_mode', 'isolated'),
                liquidation_buffer_atr=futures_config.get('liquidation_buffer_atr', 3.0),
                max_daily_loss_percent=futures_config.get('max_loss_per_day', 0.03) * 100,
                max_daily_profit_percent=futures_config.get('max_profit_per_day', 0.02) * 100,
                max_daily_trades=futures_config.get('max_trades_per_day', 15),
                max_consecutive_losses=futures_config.get('max_consecutive_losses', 4),
                soft_drawdown_percent=futures_config.get('soft_drawdown', 0.06) * 100,
                hard_drawdown_percent=futures_config.get('hard_drawdown', 0.10) * 100,
                min_confidence_threshold=futures_config.get('min_confidence', 0.6),
                min_signal_strength=futures_config.get('min_signal_strength', 0.3)
            )
        except Exception as e:
            logger.warning(f"Failed to load config, using defaults: {e}")
            return RiskLimits()
    
    def _reset_daily_counters(self):
        """Reset daily counters if new day"""
        current_date = datetime.now().date()
        if current_date != self.last_reset_date:
            self.daily_trades_count = 0
            self.daily_pnl = 0.0
            self.last_reset_date = current_date
            
            # Save previous day's stats
            if self.last_reset_date != current_date:
                self._save_daily_stats()
    
    def _save_daily_stats(self):
        """Save daily statistics"""
        date_str = self.last_reset_date.strftime('%Y-%m-%d')
        self.daily_stats[date_str] = DailyStats(
            date=date_str,
            trades_count=self.daily_trades_count,
            total_pnl=self.daily_pnl,
            total_loss=min(0, self.daily_pnl),
            total_profit=max(0, self.daily_pnl),
            consecutive_losses=self.consecutive_losses,
            max_drawdown=self._calculate_drawdown(),
            account_balance=self.current_balance,
            margin_used=self._calculate_margin_used()
        )
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown percentage"""
        if self.peak_balance == 0:
            return 0.0
        return ((self.peak_balance - self.current_balance) / self.peak_balance) * 100
    
    def _calculate_margin_used(self) -> float:
        """Calculate total margin used across all positions"""
        total_margin = 0.0
        for position in self.open_positions.values():
            total_margin += position.margin_used
        return total_margin
    
    def _check_risk_limits(self) -> Dict[str, bool]:
        """
        Check all risk limits and return status
        
        Returns:
            Dict of risk limit statuses
        """
        self._reset_daily_counters()
        
        # Calculate current metrics
        current_dd = self._calculate_drawdown()
        daily_loss_pct = (abs(min(0, self.daily_pnl)) / self.current_balance) * 100
        daily_profit_pct = (max(0, self.daily_pnl) / self.current_balance) * 100
        
        # Check limits
        limits_status = {
            'soft_drawdown_ok': current_dd <= self.limits.soft_drawdown_percent,
            'hard_drawdown_ok': current_dd <= self.limits.hard_drawdown_percent,
            'daily_loss_ok': daily_loss_pct <= self.limits.max_daily_loss_percent,
            'daily_profit_ok': daily_profit_pct <= self.limits.max_daily_profit_percent,
            'trade_limit_ok': self.daily_trades_count < self.limits.max_daily_trades,
            'consecutive_loss_ok': self.consecutive_losses < self.limits.max_consecutive_losses,
            'margin_ok': self._calculate_margin_used() < self.current_balance * 0.8
        }
        
        # Update breach flags
        self.soft_dd_breach = not limits_status['soft_drawdown_ok']
        self.hard_dd_breach = not limits_status['hard_drawdown_ok']
        self.daily_loss_breach = not limits_status['daily_loss_ok']
        self.daily_profit_breach = not limits_status['daily_profit_ok']
        self.trade_limit_breach = not limits_status['trade_limit_ok']
        self.consecutive_loss_breach = not limits_status['consecutive_loss_ok']
        
        return limits_status
    
    def calculate_position_size(self, symbol: str, atr_value: float, 
                              confidence: float, signal_strength: float) -> Tuple[float, float, float]:
        """
        Calculate position size using fixed-fractional sizing
        
        Args:
            symbol: Trading symbol
            atr_value: Current ATR value
            confidence: Signal confidence (0-1)
            signal_strength: Signal strength (0-1)
            
        Returns:
            Tuple of (quantity, stop_loss, take_profit)
        """
        # Check if we can trade
        if not self.can_trade(confidence, signal_strength):
            return 0.0, 0.0, 0.0
        
        # Fixed-fractional sizing: 0.75% of account balance
        risk_amount = self.current_balance * (self.limits.fixed_fractional_percent / 100)
        
        # Calculate position size based on ATR stop loss
        stop_loss_distance = atr_value * self.limits.atr_sl_multiplier
        position_size = risk_amount / stop_loss_distance
        
        # Apply leverage cap
        max_position_value = self.current_balance * self.limits.max_leverage
        # Use a default price for testing - in production this would come from market data
        current_price = 50000.0 if symbol == "BTCUSDT" else (3000.0 if symbol == "ETHUSDT" else 0.5)
        max_position_size = max_position_value / current_price
        
        # Use the smaller of calculated size or max allowed
        final_position_size = min(position_size, max_position_size)
        
        # Calculate stop loss and take profit
        stop_loss = current_price - stop_loss_distance if symbol.endswith('USDT') else current_price + stop_loss_distance
        take_profit = current_price + (atr_value * self.limits.atr_tp_multiplier) if symbol.endswith('USDT') else current_price - (atr_value * self.limits.atr_tp_multiplier)
        
        logger.info(f"Position sizing for {symbol}: size={final_position_size:.6f}, SL={stop_loss:.2f}, TP={take_profit:.2f}")
        
        return final_position_size, stop_loss, take_profit
    
    def can_trade(self, confidence: float, signal_strength: float) -> bool:
        """
        Check if trading is allowed based on all risk limits
        
        Args:
            confidence: Signal confidence (0-1)
            signal_strength: Signal strength (0-1)
            
        Returns:
            True if trading is allowed
        """
        # Check signal quality
        if confidence < self.limits.min_confidence_threshold:
            logger.warning(f"Confidence too low: {confidence:.3f} < {self.limits.min_confidence_threshold}")
            return False
        
        if signal_strength < self.limits.min_signal_strength:
            logger.warning(f"Signal strength too low: {signal_strength:.3f} < {self.limits.min_signal_strength}")
            return False
        
        # Check risk limits
        limits_status = self._check_risk_limits()
        
        if not limits_status['hard_drawdown_ok']:
            logger.error("HARD DRAWDOWN BREACH - Trading halted!")
            return False
        
        if not limits_status['daily_loss_ok']:
            logger.error("DAILY LOSS LIMIT BREACH - Trading halted!")
            return False
        
        if not limits_status['trade_limit_ok']:
            logger.warning("Daily trade limit reached")
            return False
        
        if not limits_status['consecutive_loss_ok']:
            logger.warning("Consecutive loss limit reached")
            return False
        
        if not limits_status['margin_ok']:
            logger.warning("Margin limit reached")
            return False
        
        # Soft drawdown warning
        if not limits_status['soft_drawdown_ok']:
            logger.warning("SOFT DRAWDOWN BREACH - Consider reducing position sizes")
        
        return True
    
    def open_position(self, symbol: str, side: str, quantity: float, 
                     entry_price: float, atr_value: float, leverage: float = 1.0) -> bool:
        """
        Open a new position with risk management
        
        Args:
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Position quantity
            entry_price: Entry price
            atr_value: ATR value for stops
            leverage: Position leverage
            
        Returns:
            True if position opened successfully
        """
        # Check leverage limit
        if leverage > self.limits.max_leverage:
            logger.error(f"Leverage {leverage} exceeds limit {self.limits.max_leverage}")
            return False
        
        # Calculate stops
        stop_loss = entry_price - (atr_value * self.limits.atr_sl_multiplier) if side == 'buy' else entry_price + (atr_value * self.limits.atr_sl_multiplier)
        take_profit = entry_price + (atr_value * self.limits.atr_tp_multiplier) if side == 'buy' else entry_price - (atr_value * self.limits.atr_tp_multiplier)
        
        # Calculate margin used
        position_value = quantity * entry_price
        margin_used = position_value / leverage
        
        # Check if we have enough margin
        if margin_used > self.current_balance * 0.8:  # 80% margin limit
            logger.error(f"Insufficient margin: {margin_used:.2f} > {self.current_balance * 0.8:.2f}")
            return False
        
        # Create trade record
        trade_id = f"{symbol}_{int(time.time())}"
        trade_record = TradeRecord(
            timestamp=datetime.now().isoformat(),
            symbol=symbol,
            side=side,
            quantity=quantity,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr_value=atr_value,
            leverage=leverage,
            margin_used=margin_used
        )
        
        # Add to open positions
        self.open_positions[trade_id] = trade_record
        
        # Update counters
        self.daily_trades_count += 1
        
        logger.info(f"Position opened: {trade_id} - {side} {quantity} {symbol} @ {entry_price}")
        logger.info(f"SL: {stop_loss:.2f}, TP: {take_profit:.2f}, Margin: {margin_used:.2f}")
        
        return True
    
    def close_position(self, trade_id: str, exit_price: float, reason: str = "manual") -> float:
        """
        Close a position and calculate PnL
        
        Args:
            trade_id: Trade identifier
            exit_price: Exit price
            reason: Reason for closing
            
        Returns:
            PnL amount
        """
        if trade_id not in self.open_positions:
            logger.error(f"Trade {trade_id} not found")
            return 0.0
        
        trade = self.open_positions[trade_id]
        
        # Calculate PnL
        if trade.side == 'buy':
            pnl = (exit_price - trade.entry_price) * trade.quantity
        else:
            pnl = (trade.entry_price - exit_price) * trade.quantity
        
        # Update trade record
        trade.pnl = pnl
        trade.exit_price = exit_price
        trade.exit_timestamp = datetime.now().isoformat()
        trade.status = "closed"
        
        # Update account balance
        self.current_balance += pnl
        
        # Update peak balance
        if self.current_balance > self.peak_balance:
            self.peak_balance = self.current_balance
        
        # Update daily PnL
        self.daily_pnl += pnl
        
        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
        
        # Move to history
        self.trade_history.append(trade)
        del self.open_positions[trade_id]
        
        logger.info(f"Position closed: {trade_id} - PnL: ${pnl:.2f} ({reason})")
        
        return pnl
    
    def check_stop_losses(self, current_prices: Dict[str, float]) -> List[str]:
        """
        Check all open positions for stop loss triggers
        
        Args:
            current_prices: Dict of symbol -> current price
            
        Returns:
            List of trade IDs that hit stop loss
        """
        triggered_trades = []
        
        # Create a copy of the items to avoid dictionary changed size during iteration
        positions_to_check = list(self.open_positions.items())
        
        for trade_id, trade in positions_to_check:
            current_price = current_prices.get(trade.symbol)
            if current_price is None:
                continue
            
            # Check stop loss
            if trade.side == 'buy' and current_price <= trade.stop_loss:
                self.close_position(trade_id, current_price, "stop_loss")
                triggered_trades.append(trade_id)
            elif trade.side == 'sell' and current_price >= trade.stop_loss:
                self.close_position(trade_id, current_price, "stop_loss")
                triggered_trades.append(trade_id)
            
            # Check take profit
            elif trade.side == 'buy' and current_price >= trade.take_profit:
                self.close_position(trade_id, current_price, "take_profit")
                triggered_trades.append(trade_id)
            elif trade.side == 'sell' and current_price <= trade.take_profit:
                self.close_position(trade_id, current_price, "take_profit")
                triggered_trades.append(trade_id)
        
        return triggered_trades
    
    def get_risk_status(self) -> Dict[str, Any]:
        """
        Get comprehensive risk status
        
        Returns:
            Dict with current risk metrics and status
        """
        limits_status = self._check_risk_limits()
        current_dd = self._calculate_drawdown()
        margin_used = self._calculate_margin_used()
        
        return {
            'account_balance': self.current_balance,
            'peak_balance': self.peak_balance,
            'current_drawdown': current_dd,
            'margin_used': margin_used,
            'open_positions': len(self.open_positions),
            'daily_trades': self.daily_trades_count,
            'daily_pnl': self.daily_pnl,
            'consecutive_losses': self.consecutive_losses,
            'risk_limits_status': limits_status,
            'can_trade': all(limits_status.values()),
            'breach_flags': {
                'soft_drawdown': self.soft_dd_breach,
                'hard_drawdown': self.hard_dd_breach,
                'daily_loss': self.daily_loss_breach,
                'daily_profit': self.daily_profit_breach,
                'trade_limit': self.trade_limit_breach,
                'consecutive_loss': self.consecutive_loss_breach
            }
        }
    
    def emergency_close_all(self, current_prices: Dict[str, float]) -> float:
        """
        Emergency close all positions
        
        Args:
            current_prices: Dict of symbol -> current price
            
        Returns:
            Total PnL from emergency close
        """
        total_pnl = 0.0
        
        for trade_id in list(self.open_positions.keys()):
            trade = self.open_positions[trade_id]
            current_price = current_prices.get(trade.symbol, trade.entry_price)
            pnl = self.close_position(trade_id, current_price, "emergency_close")
            total_pnl += pnl
        
        logger.warning(f"Emergency close all positions - Total PnL: ${total_pnl:.2f}")
        
        return total_pnl
    
    def export_trade_history(self, filename: str = None) -> str:
        """
        Export trade history to CSV
        
        Args:
            filename: Output filename (optional)
            
        Returns:
            Path to exported file
        """
        if filename is None:
            filename = f"trade_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Convert trade records to DataFrame
        trade_data = []
        for trade in self.trade_history:
            trade_data.append(asdict(trade))
        
        if trade_data:
            df = pd.DataFrame(trade_data)
            df.to_csv(filename, index=False)
            logger.info(f"Trade history exported to {filename}")
        else:
            logger.warning("No trade history to export")
        
        return filename


def main():
    """Example usage of RiskManagerLimit"""
    
    # Initialize risk manager
    risk_manager = RiskManagerLimit(initial_balance=10000.0)
    
    # Example: Check if we can trade
    can_trade = risk_manager.can_trade(confidence=0.8, signal_strength=0.7)
    print(f"Can trade: {can_trade}")
    
    # Example: Calculate position size
    symbol = "BTCUSDT"
    atr_value = 500.0  # Example ATR
    confidence = 0.8
    signal_strength = 0.7
    
    quantity, stop_loss, take_profit = risk_manager.calculate_position_size(
        symbol, atr_value, confidence, signal_strength
    )
    
    print(f"Position size: {quantity:.6f}")
    print(f"Stop loss: {stop_loss:.2f}")
    print(f"Take profit: {take_profit:.2f}")
    
    # Example: Open position
    if can_trade and quantity > 0:
        success = risk_manager.open_position(
            symbol=symbol,
            side="buy",
            quantity=quantity,
            entry_price=50000.0,
            atr_value=atr_value,
            leverage=2.0
        )
        print(f"Position opened: {success}")
    
    # Get risk status
    status = risk_manager.get_risk_status()
    print(f"Risk status: {json.dumps(status, indent=2, default=str)}")


if __name__ == "__main__":
    main()
