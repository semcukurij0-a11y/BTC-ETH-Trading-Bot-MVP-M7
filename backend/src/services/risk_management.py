"""
Risk Management Module for Crypto Trading Bot (Futures)

This module implements comprehensive risk management for futures trading including:
- Position sizing with fixed-fractional risk
- Leverage management with dynamic adjustment
- ATR-based stop loss and take profit
- Liquidation buffer protection
- Daily limits and consecutive loss protection
- Kill-switch mechanisms
- Order handling and reconciliation
"""

import pandas as pd
import numpy as np
import logging
import json
import time
import hashlib
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    OCO = "oco"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


class RiskManagementModule:
    """
    Risk Management Module for futures trading with comprehensive risk controls.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize Risk Management Module.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Risk parameters
        self.risk_per_trade = self.config.get('risk_per_trade', 0.0075)  # 0.75% default
        self.max_leverage = self.config.get('max_leverage', 5.0)  # 5x default
        self.atr_sl_multiplier = self.config.get('atr_sl_multiplier', 2.5)
        self.atr_tp_multiplier = self.config.get('atr_tp_multiplier', 3.0)
        self.liquidation_buffer_atr = self.config.get('liquidation_buffer_atr', 3.0)
        self.liquidation_buffer_percent = self.config.get('liquidation_buffer_percent', 0.01)
        
        # Daily/Session limits (configurable)
        self.max_loss_per_day = self.config.get('max_loss_per_day', 0.03)  # 3.0% of equity (hard stop)
        self.max_profit_per_day = self.config.get('max_profit_per_day', 0.02)  # 2.0% of equity (lock profits)
        self.max_trades_per_day = self.config.get('max_trades_per_day', 15)  # 15 total entry orders
        self.max_consecutive_losses = self.config.get('max_consecutive_losses', 4)  # 4 consecutive losses
        self.max_loss_per_trade = self.config.get('max_loss_per_trade', 0.008)  # 0.8% of equity per trade
        
        # Kill-switch thresholds
        self.max_api_errors = self.config.get('max_api_errors', 10)
        self.margin_ratio_threshold = self.config.get('margin_ratio_threshold', 0.8)
        self.max_drawdown = self.config.get('max_drawdown', 0.15)  # 15% max drawdown
        
        # State tracking
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.api_errors = 0
        self.peak_balance = 0.0
        self.current_balance = 0.0
        self.session_start = datetime.now()
        self.trade_lock = False
        self.kill_switch_triggered = False
        
        # Order tracking
        self.pending_orders = {}
        self.filled_orders = {}
        self.order_counter = 0
        
        # Position tracking
        self.current_positions = {}
        self.position_history = []
        
        # Volatility regime tracking
        self.volatility_regime = "normal"  # normal, high, extreme
        self.current_leverage = self.max_leverage
        
    def calculate_position_size(self, 
                              symbol: str, 
                              entry_price: float, 
                              stop_loss_price: float, 
                              account_balance: float,
                              current_atr: float) -> Dict[str, Any]:
        """
        Calculate position size using fixed-fractional risk per trade.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_price: Stop loss price
            account_balance: Current account balance
            current_atr: Current ATR value
            
        Returns:
            Dictionary with position sizing information
        """
        try:
            # Calculate risk amount (use the smaller of risk_per_trade and max_loss_per_trade)
            risk_amount = account_balance * min(self.risk_per_trade, self.max_loss_per_trade)
            
            # Calculate stop loss distance
            if entry_price > stop_loss_price:  # Long position
                sl_distance = entry_price - stop_loss_price
            else:  # Short position
                sl_distance = stop_loss_price - entry_price
            
            # Calculate base position size
            base_quantity = risk_amount / sl_distance
            
            # Apply leverage cap
            max_quantity_by_leverage = (account_balance * self.current_leverage) / entry_price
            quantity = min(base_quantity, max_quantity_by_leverage)
            
            # Calculate notional value
            notional_value = quantity * entry_price
            
            # Check liquidation buffer
            liquidation_check = self._check_liquidation_buffer(
                symbol, entry_price, stop_loss_price, quantity, current_atr
            )
            
            if not liquidation_check['safe']:
                # Reduce position size to meet liquidation buffer
                quantity = liquidation_check['max_safe_quantity']
                notional_value = quantity * entry_price
                self.logger.warning(f"Position size reduced due to liquidation buffer: {symbol}")
            
            # Calculate margin required
            margin_required = notional_value / self.current_leverage
            
            # Calculate effective leverage
            effective_leverage = notional_value / account_balance
            
            result = {
                'quantity': quantity,
                'notional_value': notional_value,
                'margin_required': margin_required,
                'effective_leverage': effective_leverage,
                'risk_amount': risk_amount,
                'sl_distance': sl_distance,
                'liquidation_safe': liquidation_check['safe'],
                'leverage_capped': base_quantity > max_quantity_by_leverage
            }
            
            self.logger.info(f"Position sizing for {symbol}: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating position size for {symbol}: {e}")
            return {
                'quantity': 0,
                'notional_value': 0,
                'margin_required': 0,
                'effective_leverage': 0,
                'risk_amount': 0,
                'sl_distance': 0,
                'liquidation_safe': False,
                'leverage_capped': False,
                'error': str(e)
            }
    
    def _check_liquidation_buffer(self, 
                                symbol: str, 
                                entry_price: float, 
                                stop_loss_price: float, 
                                quantity: float, 
                                current_atr: float) -> Dict[str, Any]:
        """
        Check if position meets liquidation buffer requirements.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            stop_loss_price: Stop loss price
            quantity: Position quantity
            current_atr: Current ATR value
            
        Returns:
            Dictionary with liquidation safety information
        """
        try:
            # Calculate liquidation price (simplified - would need actual exchange data)
            # For long positions, liquidation price is lower
            # For short positions, liquidation price is higher
            
            if entry_price > stop_loss_price:  # Long position
                liquidation_price = entry_price * 0.1  # Simplified - 90% below entry
                buffer_distance = stop_loss_price - liquidation_price
            else:  # Short position
                liquidation_price = entry_price * 1.9  # Simplified - 90% above entry
                buffer_distance = liquidation_price - stop_loss_price
            
            # Check ATR-based buffer
            atr_buffer = current_atr * self.liquidation_buffer_atr
            atr_safe = buffer_distance >= atr_buffer
            
            # Check percentage-based buffer
            percent_buffer = entry_price * self.liquidation_buffer_percent
            percent_safe = buffer_distance >= percent_buffer
            
            # Calculate maximum safe quantity
            max_safe_quantity = quantity
            if not (atr_safe and percent_safe):
                # Reduce quantity to meet buffer requirements
                required_buffer = max(atr_buffer, percent_buffer)
                if entry_price > stop_loss_price:  # Long
                    max_safe_sl = liquidation_price + required_buffer
                else:  # Short
                    max_safe_sl = liquidation_price - required_buffer
                
                # Recalculate max safe quantity
                sl_distance = abs(entry_price - max_safe_sl)
                risk_amount = self.current_balance * self.risk_per_trade
                max_safe_quantity = risk_amount / sl_distance
            
            return {
                'safe': atr_safe and percent_safe,
                'atr_safe': atr_safe,
                'percent_safe': percent_safe,
                'buffer_distance': buffer_distance,
                'atr_buffer': atr_buffer,
                'percent_buffer': percent_buffer,
                'max_safe_quantity': max_safe_quantity,
                'liquidation_price': liquidation_price
            }
            
        except Exception as e:
            self.logger.error(f"Error checking liquidation buffer for {symbol}: {e}")
            return {
                'safe': False,
                'atr_safe': False,
                'percent_safe': False,
                'buffer_distance': 0,
                'atr_buffer': 0,
                'percent_buffer': 0,
                'max_safe_quantity': 0,
                'liquidation_price': 0,
                'error': str(e)
            }
    
    def calculate_stop_loss_take_profit(self, 
                                      symbol: str, 
                                      entry_price: float, 
                                      side: str, 
                                      current_atr: float) -> Dict[str, Any]:
        """
        Calculate ATR-based stop loss and take profit levels.
        
        Args:
            symbol: Trading symbol
            entry_price: Entry price
            side: 'buy' or 'sell'
            current_atr: Current ATR value
            
        Returns:
            Dictionary with SL/TP levels
        """
        try:
            # Calculate ATR-based distances
            sl_distance = current_atr * self.atr_sl_multiplier
            tp_distance = current_atr * self.atr_tp_multiplier
            
            if side.lower() == 'buy':
                # Long position
                stop_loss = entry_price - sl_distance
                take_profit = entry_price + tp_distance
            else:
                # Short position
                stop_loss = entry_price + sl_distance
                take_profit = entry_price - tp_distance
            
            # Calculate partial TP levels (optional laddering)
            tp_levels = self._calculate_partial_tp_levels(entry_price, side, tp_distance)
            
            result = {
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'sl_distance': sl_distance,
                'tp_distance': tp_distance,
                'atr_multiplier_sl': self.atr_sl_multiplier,
                'atr_multiplier_tp': self.atr_tp_multiplier,
                'partial_tp_levels': tp_levels
            }
            
            self.logger.info(f"SL/TP calculation for {symbol}: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating SL/TP for {symbol}: {e}")
            return {
                'stop_loss': 0,
                'take_profit': 0,
                'sl_distance': 0,
                'tp_distance': 0,
                'atr_multiplier_sl': 0,
                'atr_multiplier_tp': 0,
                'partial_tp_levels': [],
                'error': str(e)
            }
    
    def _calculate_partial_tp_levels(self, 
                                   entry_price: float, 
                                   side: str, 
                                   tp_distance: float) -> List[Dict[str, Any]]:
        """
        Calculate partial take profit levels for laddering.
        
        Args:
            entry_price: Entry price
            side: 'buy' or 'sell'
            tp_distance: Full TP distance
            
        Returns:
            List of partial TP levels
        """
        try:
            # Define partial TP levels (25%, 50%, 75% of full TP)
            tp_percentages = [0.25, 0.5, 0.75]
            tp_levels = []
            
            for i, percentage in enumerate(tp_percentages):
                partial_distance = tp_distance * percentage
                
                if side.lower() == 'buy':
                    tp_price = entry_price + partial_distance
                else:
                    tp_price = entry_price - partial_distance
                
                tp_levels.append({
                    'level': i + 1,
                    'percentage': percentage,
                    'price': tp_price,
                    'distance': partial_distance,
                    'quantity_percentage': 0.33  # Equal distribution
                })
            
            return tp_levels
            
        except Exception as e:
            self.logger.error(f"Error calculating partial TP levels: {e}")
            return []
    
    def update_leverage(self, symbol: str, current_atr: float, price: float) -> float:
        """
        Update leverage based on volatility regime.
        
        Args:
            symbol: Trading symbol
            current_atr: Current ATR value
            price: Current price
            
        Returns:
            Updated leverage value
        """
        try:
            # Calculate volatility as ATR/Price ratio
            volatility_ratio = current_atr / price
            
            # Define volatility thresholds
            high_volatility_threshold = 0.02  # 2%
            extreme_volatility_threshold = 0.05  # 5%
            
            # Determine volatility regime
            if volatility_ratio >= extreme_volatility_threshold:
                self.volatility_regime = "extreme"
                new_leverage = self.max_leverage * 0.3  # 30% of max leverage
            elif volatility_ratio >= high_volatility_threshold:
                self.volatility_regime = "high"
                new_leverage = self.max_leverage * 0.6  # 60% of max leverage
            else:
                self.volatility_regime = "normal"
                new_leverage = self.max_leverage
            
            # Update current leverage
            self.current_leverage = new_leverage
            
            self.logger.info(f"Leverage updated for {symbol}: {new_leverage}x (regime: {self.volatility_regime})")
            return new_leverage
            
        except Exception as e:
            self.logger.error(f"Error updating leverage for {symbol}: {e}")
            return self.current_leverage
    
    def check_daily_limits(self) -> Dict[str, Any]:
        """
        Check daily trading limits.
        
        Returns:
            Dictionary with limit status
        """
        try:
            # Check if new session (reset daily counters)
            if datetime.now().date() > self.session_start.date():
                self._reset_daily_counters()
            
            # Check daily loss limit (hard stop - flatten and halt)
            daily_loss_threshold = self.max_loss_per_day * self.current_balance
            loss_limit_breached = self.daily_pnl < -daily_loss_threshold
            
            # Check daily profit target (lock profits - no new entries for the day)
            daily_profit_threshold = self.max_profit_per_day * self.current_balance
            profit_target_reached = self.daily_pnl > daily_profit_threshold
            
            # Check daily trade limit
            trade_limit_reached = self.daily_trades >= self.max_trades_per_day
            
            # Check consecutive losses
            consecutive_loss_lock = self.consecutive_losses >= self.max_consecutive_losses
            
            # Overall status - can trade if no hard limits breached
            can_trade = not (loss_limit_breached or trade_limit_reached or consecutive_loss_lock)
            
            # Can enter new positions (profit target reached means no new entries)
            can_enter = can_trade and not profit_target_reached
            
            result = {
                'can_trade': can_trade,
                'can_enter': can_enter,
                'loss_limit_breached': loss_limit_breached,
                'profit_target_reached': profit_target_reached,
                'trade_limit_reached': trade_limit_reached,
                'consecutive_loss_lock': consecutive_loss_lock,
                'daily_pnl': self.daily_pnl,
                'daily_trades': self.daily_trades,
                'consecutive_losses': self.consecutive_losses,
                'daily_loss_threshold': daily_loss_threshold,
                'daily_profit_threshold': daily_profit_threshold,
                'max_loss_per_day': self.max_loss_per_day,
                'max_profit_per_day': self.max_profit_per_day,
                'max_trades_per_day': self.max_trades_per_day,
                'max_consecutive_losses': self.max_consecutive_losses,
                'max_loss_per_trade': self.max_loss_per_trade
            }
            
            if not can_trade:
                self.trade_lock = True
                self.logger.warning(f"Daily limits breached: {result}")
                
                # If daily loss limit breached, trigger emergency stop
                if loss_limit_breached:
                    self.logger.critical(f"Daily loss limit breached! PnL: {self.daily_pnl:.2f}, Threshold: {daily_loss_threshold:.2f}")
                    self.kill_switch_triggered = True
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error checking daily limits: {e}")
            return {
                'can_trade': False,
                'loss_limit_breached': True,
                'profit_target_reached': False,
                'trade_limit_reached': True,
                'consecutive_loss_lock': True,
                'error': str(e)
            }
    
    def _reset_daily_counters(self):
        """Reset daily counters for new session."""
        self.daily_pnl = 0.0
        self.daily_trades = 0
        self.consecutive_losses = 0
        self.trade_lock = False
        self.kill_switch_triggered = False  # Reset kill switch on new session
        self.session_start = datetime.now()
        self.logger.info("Daily counters reset for new session")
    
    def can_enter_new_position(self) -> bool:
        """
        Check if we can enter new positions.
        Different from can_trade - this specifically checks entry permissions.
        
        Returns:
            True if new positions can be opened
        """
        try:
            daily_limits = self.check_daily_limits()
            return daily_limits.get('can_enter', False)
        except Exception as e:
            self.logger.error(f"Error checking entry permission: {e}")
            return False
    
    def update_trade_result(self, symbol: str, pnl: float, is_win: bool):
        """
        Update trade result and counters.
        
        Args:
            symbol: Trading symbol
            pnl: Trade P&L
            is_win: Whether trade was profitable
        """
        try:
            # Update daily P&L
            self.daily_pnl += pnl
            
            # Update daily trade count
            self.daily_trades += 1
            
            # Update consecutive losses
            if is_win:
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
            
            # Update peak balance
            if self.current_balance > self.peak_balance:
                self.peak_balance = self.current_balance
            
            self.logger.info(f"Trade result updated for {symbol}: PnL={pnl:.4f}, Win={is_win}, "
                           f"Daily PnL={self.daily_pnl:.4f}, Consecutive Losses={self.consecutive_losses}")
            
        except Exception as e:
            self.logger.error(f"Error updating trade result for {symbol}: {e}")
    
    def check_kill_switch(self, 
                         margin_ratio: float, 
                         current_drawdown: float, 
                         api_error_count: int) -> Dict[str, Any]:
        """
        Check kill-switch conditions.
        
        Args:
            margin_ratio: Current margin ratio
            current_drawdown: Current drawdown percentage
            api_error_count: Number of API errors
            
        Returns:
            Dictionary with kill-switch status
        """
        try:
            # Check API error threshold
            api_errors_breached = api_error_count >= self.max_api_errors
            
            # Check margin ratio threshold
            margin_ratio_breached = margin_ratio >= self.margin_ratio_threshold
            
            # Check drawdown threshold
            drawdown_breached = current_drawdown >= self.max_drawdown
            
            # Overall kill-switch status
            kill_switch_triggered = api_errors_breached or margin_ratio_breached or drawdown_breached
            
            if kill_switch_triggered:
                self.kill_switch_triggered = True
                self.logger.critical(f"Kill-switch triggered: API errors={api_errors_breached}, "
                                   f"Margin ratio={margin_ratio_breached}, Drawdown={drawdown_breached}")
            
            result = {
                'kill_switch_triggered': kill_switch_triggered,
                'api_errors_breached': api_errors_breached,
                'margin_ratio_breached': margin_ratio_breached,
                'drawdown_breached': drawdown_breached,
                'api_error_count': api_error_count,
                'margin_ratio': margin_ratio,
                'current_drawdown': current_drawdown,
                'max_api_errors': self.max_api_errors,
                'margin_ratio_threshold': self.margin_ratio_threshold,
                'max_drawdown': self.max_drawdown
            }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error checking kill-switch: {e}")
            return {
                'kill_switch_triggered': True,
                'api_errors_breached': True,
                'margin_ratio_breached': True,
                'drawdown_breached': True,
                'error': str(e)
            }
    
    def generate_order_id(self, symbol: str, side: str, order_type: str) -> str:
        """
        Generate idempotent order ID.
        
        Args:
            symbol: Trading symbol
            side: Order side
            order_type: Order type
            
        Returns:
            Unique order ID
        """
        try:
            # Generate unique ID based on timestamp, symbol, and counter
            timestamp = int(time.time() * 1000)  # Milliseconds
            self.order_counter += 1
            
            # Create hash for uniqueness
            hash_input = f"{timestamp}_{symbol}_{side}_{order_type}_{self.order_counter}"
            order_hash = hashlib.md5(hash_input.encode()).hexdigest()[:8]
            
            order_id = f"{symbol}_{side}_{order_type}_{timestamp}_{order_hash}"
            
            return order_id
            
        except Exception as e:
            self.logger.error(f"Error generating order ID: {e}")
            return f"error_{int(time.time())}"
    
    def create_order(self, 
                    symbol: str, 
                    side: str, 
                    order_type: str, 
                    quantity: float, 
                    price: Optional[float] = None,
                    stop_price: Optional[float] = None,
                    reduce_only: bool = False) -> Dict[str, Any]:
        """
        Create order with risk management checks.
        
        Args:
            symbol: Trading symbol
            side: Order side ('buy' or 'sell')
            order_type: Order type
            quantity: Order quantity
            price: Order price (for limit orders)
            stop_price: Stop price (for stop orders)
            reduce_only: Whether order is reduce-only
            
        Returns:
            Dictionary with order information
        """
        try:
            # Check if new position entry is allowed
            daily_limits = self.check_daily_limits()
            if not daily_limits['can_enter']:
                return {
                    'success': False,
                    'error': 'New position entry not allowed due to daily limits',
                    'daily_limits': daily_limits
                }
            
            # Check kill-switch
            if self.kill_switch_triggered:
                return {
                    'success': False,
                    'error': 'Kill-switch triggered - trading halted'
                }
            
            # Generate order ID
            order_id = self.generate_order_id(symbol, side, order_type)
            
            # Create order object
            order = {
                'order_id': order_id,
                'symbol': symbol,
                'side': side,
                'order_type': order_type,
                'quantity': quantity,
                'price': price,
                'stop_price': stop_price,
                'reduce_only': reduce_only,
                'status': OrderStatus.PENDING.value,
                'created_at': datetime.now().isoformat(),
                'updated_at': datetime.now().isoformat()
            }
            
            # Add to pending orders
            self.pending_orders[order_id] = order
            
            self.logger.info(f"Order created: {order}")
            return {
                'success': True,
                'order': order
            }
            
        except Exception as e:
            self.logger.error(f"Error creating order for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def update_order_status(self, order_id: str, status: str, fill_price: Optional[float] = None, 
                          fill_quantity: Optional[float] = None) -> Dict[str, Any]:
        """
        Update order status and handle fills.
        
        Args:
            order_id: Order ID
            status: New order status
            fill_price: Fill price (if filled)
            fill_quantity: Fill quantity (if filled)
            
        Returns:
            Dictionary with update result
        """
        try:
            if order_id not in self.pending_orders:
                return {
                    'success': False,
                    'error': f'Order {order_id} not found in pending orders'
                }
            
            order = self.pending_orders[order_id]
            order['status'] = status
            order['updated_at'] = datetime.now().isoformat()
            
            if status == OrderStatus.FILLED.value:
                order['fill_price'] = fill_price
                order['fill_quantity'] = fill_quantity
                order['filled_at'] = datetime.now().isoformat()
                
                # Move to filled orders
                self.filled_orders[order_id] = order
                del self.pending_orders[order_id]
                
                self.logger.info(f"Order filled: {order_id} at {fill_price} for {fill_quantity}")
            
            elif status == OrderStatus.CANCELLED.value:
                order['cancelled_at'] = datetime.now().isoformat()
                del self.pending_orders[order_id]
                
                self.logger.info(f"Order cancelled: {order_id}")
            
            return {
                'success': True,
                'order': order
            }
            
        except Exception as e:
            self.logger.error(f"Error updating order status for {order_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def reconcile_orders(self, exchange_orders: List[Dict]) -> Dict[str, Any]:
        """
        Reconcile local orders with exchange orders.
        
        Args:
            exchange_orders: List of orders from exchange
            
        Returns:
            Dictionary with reconciliation results
        """
        try:
            reconciliation_results = {
                'matched_orders': 0,
                'missing_orders': 0,
                'extra_orders': 0,
                'status_mismatches': 0,
                'details': []
            }
            
            # Create exchange order lookup
            exchange_order_ids = {order['order_id']: order for order in exchange_orders}
            
            # Check pending orders
            for order_id, local_order in self.pending_orders.items():
                if order_id in exchange_order_ids:
                    exchange_order = exchange_order_ids[order_id]
                    
                    # Check for status mismatches
                    if local_order['status'] != exchange_order['status']:
                        reconciliation_results['status_mismatches'] += 1
                        reconciliation_results['details'].append({
                            'order_id': order_id,
                            'issue': 'status_mismatch',
                            'local_status': local_order['status'],
                            'exchange_status': exchange_order['status']
                        })
                        
                        # Update local order status
                        self.update_order_status(
                            order_id, 
                            exchange_order['status'],
                            exchange_order.get('fill_price'),
                            exchange_order.get('fill_quantity')
                        )
                    else:
                        reconciliation_results['matched_orders'] += 1
                else:
                    reconciliation_results['missing_orders'] += 1
                    reconciliation_results['details'].append({
                        'order_id': order_id,
                        'issue': 'missing_on_exchange',
                        'local_status': local_order['status']
                    })
            
            # Check for extra orders on exchange
            local_order_ids = set(self.pending_orders.keys()) | set(self.filled_orders.keys())
            for order_id in exchange_order_ids:
                if order_id not in local_order_ids:
                    reconciliation_results['extra_orders'] += 1
                    reconciliation_results['details'].append({
                        'order_id': order_id,
                        'issue': 'extra_on_exchange',
                        'exchange_status': exchange_order_ids[order_id]['status']
                    })
            
            self.logger.info(f"Order reconciliation completed: {reconciliation_results}")
            return reconciliation_results
            
        except Exception as e:
            self.logger.error(f"Error reconciling orders: {e}")
            return {
                'matched_orders': 0,
                'missing_orders': 0,
                'extra_orders': 0,
                'status_mismatches': 0,
                'details': [],
                'error': str(e)
            }
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive risk management summary.
        
        Returns:
            Dictionary with risk summary
        """
        try:
            # Calculate current drawdown
            current_drawdown = 0.0
            if self.peak_balance > 0:
                current_drawdown = (self.peak_balance - self.current_balance) / self.peak_balance
            
            # Get daily limits status
            daily_limits = self.check_daily_limits()
            
            # Get kill-switch status
            kill_switch_status = self.check_kill_switch(0.0, current_drawdown, self.api_errors)
            
            summary = {
                'account_balance': self.current_balance,
                'peak_balance': self.peak_balance,
                'current_drawdown': current_drawdown,
                'daily_pnl': self.daily_pnl,
                'daily_trades': self.daily_trades,
                'consecutive_losses': self.consecutive_losses,
                'current_leverage': self.current_leverage,
                'max_leverage': self.max_leverage,
                'volatility_regime': self.volatility_regime,
                'trade_lock': self.trade_lock,
                'kill_switch_triggered': self.kill_switch_triggered,
                'pending_orders': len(self.pending_orders),
                'filled_orders': len(self.filled_orders),
                'daily_limits': daily_limits,
                'kill_switch_status': kill_switch_status,
                'risk_parameters': {
                    'risk_per_trade': self.risk_per_trade,
                    'atr_sl_multiplier': self.atr_sl_multiplier,
                    'atr_tp_multiplier': self.atr_tp_multiplier,
                    'liquidation_buffer_atr': self.liquidation_buffer_atr,
                    'liquidation_buffer_percent': self.liquidation_buffer_percent
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting risk summary: {e}")
            return {
                'error': str(e)
            }


def main():
    """Main execution function for testing."""
    # Configuration
    config = {
        'risk_per_trade': 0.0075,  # 0.75%
        'max_leverage': 5.0,
        'atr_sl_multiplier': 2.5,
        'atr_tp_multiplier': 3.0,
        'liquidation_buffer_atr': 3.0,
        'liquidation_buffer_percent': 0.01,
        'max_loss_per_day': 0.05,
        'max_profit_per_day': 0.10,
        'max_trades_per_day': 20,
        'max_consecutive_losses': 3,
        'max_api_errors': 10,
        'margin_ratio_threshold': 0.8,
        'max_drawdown': 0.15
    }
    
    # Initialize risk management module
    risk_module = RiskManagementModule(config)
    
    # Test position sizing
    position_size = risk_module.calculate_position_size(
        symbol="BTCUSDT",
        entry_price=50000,
        stop_loss_price=49000,
        account_balance=10000,
        current_atr=500
    )
    print("Position Sizing Result:", position_size)
    
    # Test SL/TP calculation
    sl_tp = risk_module.calculate_stop_loss_take_profit(
        symbol="BTCUSDT",
        entry_price=50000,
        side="buy",
        current_atr=500
    )
    print("SL/TP Calculation Result:", sl_tp)
    
    # Test daily limits
    daily_limits = risk_module.check_daily_limits()
    print("Daily Limits Status:", daily_limits)
    
    # Test risk summary
    risk_summary = risk_module.get_risk_summary()
    print("Risk Summary:", risk_summary)


if __name__ == "__main__":
    main()



