"""
Order Execution Module for Crypto Trading Bot (Futures)

This module handles order execution, management, and reconciliation for futures trading:
- Order placement and management
- Retry logic with exponential backoff
- Order reconciliation with exchange
- Post-only and reduce-only order handling
- OCO (One-Cancels-Other) order support
- Conditional order execution
"""

import pandas as pd
import numpy as np
import logging
import json
import time
import asyncio
import aiohttp
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

from .risk_management import RiskManagementModule, OrderType, OrderSide, OrderStatus


class ExecutionMode(Enum):
    SIMULATION = "simulation"
    PAPER_TRADING = "paper_trading"
    LIVE_TRADING = "live_trading"


class OrderExecutionModule:
    """
    Order Execution Module for futures trading with comprehensive order management.
    """
    
    def __init__(self, 
                 config: Optional[Dict] = None,
                 risk_module: Optional[RiskManagementModule] = None,
                 execution_mode: ExecutionMode = ExecutionMode.SIMULATION):
        """
        Initialize Order Execution Module.
        
        Args:
            config: Configuration dictionary
            risk_module: Risk management module instance
            execution_mode: Execution mode (simulation, paper, live)
        """
        self.config = config or {}
        self.risk_module = risk_module or RiskManagementModule(config)
        self.execution_mode = execution_mode
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # API configuration
        self.api_base_url = self.config.get('api_base_url', 'https://api.bybit.com')
        self.api_key = self.config.get('api_key')
        self.api_secret = self.config.get('api_secret')
        self.testnet = self.config.get('testnet', True)
        
        # Order execution parameters
        self.max_retries = self.config.get('max_retries', 3)
        self.retry_delay = self.config.get('retry_delay', 1.0)  # seconds
        self.max_retry_delay = self.config.get('max_retry_delay', 30.0)  # seconds
        self.timeout = self.config.get('timeout', 10.0)  # seconds
        
        # Order management
        self.active_orders = {}
        self.order_history = []
        self.failed_orders = []
        
        # Performance tracking
        self.execution_stats = {
            'total_orders': 0,
            'successful_orders': 0,
            'failed_orders': 0,
            'cancelled_orders': 0,
            'average_execution_time': 0.0,
            'total_execution_time': 0.0
        }
        
        # Rate limiting
        self.rate_limit_remaining = 120  # requests per minute
        self.rate_limit_reset = time.time() + 60
        self.last_request_time = 0
        
        # Session management
        self.session = None
        self.session_lock = asyncio.Lock()
        
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def _rate_limit_check(self):
        """Check and enforce rate limits."""
        current_time = time.time()
        
        # Reset rate limit if time window has passed
        if current_time >= self.rate_limit_reset:
            self.rate_limit_remaining = 120
            self.rate_limit_reset = current_time + 60
        
        # Wait if rate limit exceeded
        if self.rate_limit_remaining <= 0:
            wait_time = self.rate_limit_reset - current_time
            if wait_time > 0:
                self.logger.warning(f"Rate limit exceeded, waiting {wait_time:.2f} seconds")
                await asyncio.sleep(wait_time)
                self.rate_limit_remaining = 120
                self.rate_limit_reset = time.time() + 60
        
        # Enforce minimum time between requests
        time_since_last = current_time - self.last_request_time
        min_interval = 0.1  # 100ms minimum between requests
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
        
        self.last_request_time = time.time()
        self.rate_limit_remaining -= 1
    
    async def _make_api_request(self, 
                               method: str, 
                               endpoint: str, 
                               params: Optional[Dict] = None,
                               data: Optional[Dict] = None,
                               headers: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Make API request with retry logic and error handling.
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            params: Query parameters
            data: Request data
            headers: Request headers
            
        Returns:
            API response
        """
        for attempt in range(self.max_retries + 1):
            try:
                # Rate limiting
                await self._rate_limit_check()
                
                # Get session
                session = await self._get_session()
                
                # Prepare request
                url = f"{self.api_base_url}{endpoint}"
                request_headers = headers or {}
                
                # Add authentication if needed
                if self.api_key and self.api_secret:
                    # Add signature and timestamp
                    timestamp = str(int(time.time() * 1000))
                    request_headers.update({
                        'X-BAPI-API-KEY': self.api_key,
                        'X-BAPI-TIMESTAMP': timestamp,
                        'X-BAPI-RECV-WINDOW': '5000'
                    })
                
                # Make request
                if method.upper() == 'GET':
                    async with session.get(url, params=params, headers=request_headers) as response:
                        result = await response.json()
                elif method.upper() == 'POST':
                    async with session.post(url, json=data, headers=request_headers) as response:
                        result = await response.json()
                elif method.upper() == 'DELETE':
                    async with session.delete(url, params=params, headers=request_headers) as response:
                        result = await response.json()
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                
                # Check response
                if response.status == 200:
                    if result.get('retCode') == 0:
                        return result
                    else:
                        error_msg = result.get('retMsg', 'Unknown API error')
                        raise Exception(f"API error: {error_msg}")
                else:
                    raise Exception(f"HTTP error: {response.status}")
                
            except Exception as e:
                self.logger.warning(f"API request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}")
                
                if attempt < self.max_retries:
                    # Exponential backoff
                    delay = min(self.retry_delay * (2 ** attempt), self.max_retry_delay)
                    await asyncio.sleep(delay)
                else:
                    self.logger.error(f"API request failed after {self.max_retries + 1} attempts: {e}")
                    raise
        
        return {}
    
    async def place_order(self, 
                         symbol: str, 
                         side: str, 
                         order_type: str, 
                         quantity: float,
                         price: Optional[float] = None,
                         stop_price: Optional[float] = None,
                         time_in_force: str = "GTC",
                         reduce_only: bool = False,
                         post_only: bool = False,
                         close_on_trigger: bool = False) -> Dict[str, Any]:
        """
        Place order with comprehensive error handling and retry logic.
        
        Args:
            symbol: Trading symbol
            side: Order side ('Buy' or 'Sell')
            order_type: Order type ('Market', 'Limit', 'Stop', 'StopLimit')
            quantity: Order quantity
            price: Order price (for limit orders)
            stop_price: Stop price (for stop orders)
            time_in_force: Time in force ('GTC', 'IOC', 'FOK')
            reduce_only: Whether order is reduce-only
            post_only: Whether order is post-only
            close_on_trigger: Whether to close position on trigger
            
        Returns:
            Dictionary with order placement result
        """
        try:
            start_time = time.time()
            
            # Validate order parameters
            validation_result = self._validate_order_parameters(
                symbol, side, order_type, quantity, price, stop_price
            )
            if not validation_result['valid']:
                return {
                    'success': False,
                    'error': validation_result['error'],
                    'order_id': None
                }
            
            # Check risk management
            if self.risk_module:
                risk_check = self.risk_module.check_daily_limits()
                if not risk_check['can_trade']:
                    return {
                        'success': False,
                        'error': 'Trading not allowed due to risk limits',
                        'order_id': None
                    }
            
            # Generate order ID
            order_id = self.risk_module.generate_order_id(symbol, side, order_type)
            
            # Prepare order data
            order_data = {
                'symbol': symbol,
                'side': side,
                'orderType': order_type,
                'qty': str(quantity),
                'timeInForce': time_in_force
            }
            
            # Add price for limit orders
            if order_type in ['Limit', 'StopLimit'] and price:
                order_data['price'] = str(price)
            
            # Add stop price for stop orders
            if order_type in ['Stop', 'StopLimit'] and stop_price:
                order_data['stopPrice'] = str(stop_price)
            
            # Add order flags
            if reduce_only:
                order_data['reduceOnly'] = True
            if post_only:
                order_data['postOnly'] = True
            if close_on_trigger:
                order_data['closeOnTrigger'] = True
            
            # Place order based on execution mode
            if self.execution_mode == ExecutionMode.SIMULATION:
                result = await self._simulate_order_placement(order_id, order_data)
            elif self.execution_mode == ExecutionMode.PAPER_TRADING:
                result = await self._paper_trade_order_placement(order_id, order_data)
            else:  # LIVE_TRADING
                result = await self._live_trade_order_placement(order_id, order_data)
            
            # Update execution stats
            execution_time = time.time() - start_time
            self._update_execution_stats(result['success'], execution_time)
            
            # Track order
            if result['success']:
                self.active_orders[order_id] = {
                    'order_id': order_id,
                    'symbol': symbol,
                    'side': side,
                    'order_type': order_type,
                    'quantity': quantity,
                    'price': price,
                    'stop_price': stop_price,
                    'status': OrderStatus.PENDING.value,
                    'created_at': datetime.now().isoformat(),
                    'execution_time': execution_time
                }
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error placing order for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e),
                'order_id': None
            }
    
    def _validate_order_parameters(self, 
                                 symbol: str, 
                                 side: str, 
                                 order_type: str, 
                                 quantity: float,
                                 price: Optional[float] = None,
                                 stop_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Validate order parameters.
        
        Args:
            symbol: Trading symbol
            side: Order side
            order_type: Order type
            quantity: Order quantity
            price: Order price
            stop_price: Stop price
            
        Returns:
            Validation result
        """
        try:
            # Validate symbol
            if not symbol or not isinstance(symbol, str):
                return {'valid': False, 'error': 'Invalid symbol'}
            
            # Validate side
            if side not in ['Buy', 'Sell']:
                return {'valid': False, 'error': 'Invalid side (must be Buy or Sell)'}
            
            # Validate order type
            valid_order_types = ['Market', 'Limit', 'Stop', 'StopLimit']
            if order_type not in valid_order_types:
                return {'valid': False, 'error': f'Invalid order type (must be one of {valid_order_types})'}
            
            # Validate quantity
            if not isinstance(quantity, (int, float)) or quantity <= 0:
                return {'valid': False, 'error': 'Invalid quantity (must be positive number)'}
            
            # Validate price for limit orders
            if order_type in ['Limit', 'StopLimit'] and (not price or price <= 0):
                return {'valid': False, 'error': 'Price required for limit orders'}
            
            # Validate stop price for stop orders
            if order_type in ['Stop', 'StopLimit'] and (not stop_price or stop_price <= 0):
                return {'valid': False, 'error': 'Stop price required for stop orders'}
            
            return {'valid': True, 'error': None}
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {str(e)}'}
    
    async def _simulate_order_placement(self, order_id: str, order_data: Dict) -> Dict[str, Any]:
        """Simulate order placement for testing."""
        try:
            # Simulate API delay
            await asyncio.sleep(0.1)
            
            # Simulate successful placement
            result = {
                'success': True,
                'order_id': order_id,
                'symbol': order_data['symbol'],
                'side': order_data['side'],
                'order_type': order_data['orderType'],
                'quantity': float(order_data['qty']),
                'price': float(order_data.get('price', 0)),
                'status': 'New',
                'execution_mode': 'simulation',
                'timestamp': datetime.now().isoformat()
            }
            
            self.logger.info(f"Simulated order placed: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error in simulated order placement: {e}")
            return {
                'success': False,
                'error': str(e),
                'order_id': order_id
            }
    
    async def _paper_trade_order_placement(self, order_id: str, order_data: Dict) -> Dict[str, Any]:
        """Place paper trading order (real API calls but no actual execution)."""
        try:
            # Use testnet for paper trading
            if not self.testnet:
                self.logger.warning("Paper trading should use testnet")
            
            # Make API request to testnet
            endpoint = "/v5/order/create"
            result = await self._make_api_request('POST', endpoint, data=order_data)
            
            if result.get('retCode') == 0:
                order_result = result.get('result', {})
                return {
                    'success': True,
                    'order_id': order_result.get('orderId', order_id),
                    'symbol': order_data['symbol'],
                    'side': order_data['side'],
                    'order_type': order_data['orderType'],
                    'quantity': float(order_data['qty']),
                    'price': float(order_data.get('price', 0)),
                    'status': order_result.get('orderStatus', 'New'),
                    'execution_mode': 'paper_trading',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': result.get('retMsg', 'Unknown error'),
                    'order_id': order_id
                }
                
        except Exception as e:
            self.logger.error(f"Error in paper trading order placement: {e}")
            return {
                'success': False,
                'error': str(e),
                'order_id': order_id
            }
    
    async def _live_trade_order_placement(self, order_id: str, order_data: Dict) -> Dict[str, Any]:
        """Place live trading order."""
        try:
            # Ensure not using testnet for live trading
            if self.testnet:
                return {
                    'success': False,
                    'error': 'Live trading cannot use testnet',
                    'order_id': order_id
                }
            
            # Make API request to live API
            endpoint = "/v5/order/create"
            result = await self._make_api_request('POST', endpoint, data=order_data)
            
            if result.get('retCode') == 0:
                order_result = result.get('result', {})
                return {
                    'success': True,
                    'order_id': order_result.get('orderId', order_id),
                    'symbol': order_data['symbol'],
                    'side': order_data['side'],
                    'order_type': order_data['orderType'],
                    'quantity': float(order_data['qty']),
                    'price': float(order_data.get('price', 0)),
                    'status': order_result.get('orderStatus', 'New'),
                    'execution_mode': 'live_trading',
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': result.get('retMsg', 'Unknown error'),
                    'order_id': order_id
                }
                
        except Exception as e:
            self.logger.error(f"Error in live trading order placement: {e}")
            return {
                'success': False,
                'error': str(e),
                'order_id': order_id
            }
    
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Cancel order.
        
        Args:
            order_id: Order ID to cancel
            symbol: Trading symbol
            
        Returns:
            Dictionary with cancellation result
        """
        try:
            # Check if order exists
            if order_id not in self.active_orders:
                return {
                    'success': False,
                    'error': f'Order {order_id} not found in active orders'
                }
            
            # Prepare cancellation data
            cancel_data = {
                'symbol': symbol,
                'orderId': order_id
            }
            
            # Cancel order based on execution mode
            if self.execution_mode == ExecutionMode.SIMULATION:
                result = await self._simulate_order_cancellation(order_id, cancel_data)
            else:
                endpoint = "/v5/order/cancel"
                result = await self._make_api_request('POST', endpoint, data=cancel_data)
                
                if result.get('retCode') == 0:
                    result = {
                        'success': True,
                        'order_id': order_id,
                        'status': 'Cancelled',
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    result = {
                        'success': False,
                        'error': result.get('retMsg', 'Unknown error'),
                        'order_id': order_id
                    }
            
            # Update order status
            if result['success']:
                if order_id in self.active_orders:
                    self.active_orders[order_id]['status'] = OrderStatus.CANCELLED.value
                    self.active_orders[order_id]['cancelled_at'] = datetime.now().isoformat()
                
                # Move to order history
                self.order_history.append(self.active_orders.get(order_id, {}))
                if order_id in self.active_orders:
                    del self.active_orders[order_id]
                
                self.execution_stats['cancelled_orders'] += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error cancelling order {order_id}: {e}")
            return {
                'success': False,
                'error': str(e),
                'order_id': order_id
            }
    
    async def _simulate_order_cancellation(self, order_id: str, cancel_data: Dict) -> Dict[str, Any]:
        """Simulate order cancellation."""
        try:
            await asyncio.sleep(0.05)  # Simulate API delay
            
            return {
                'success': True,
                'order_id': order_id,
                'status': 'Cancelled',
                'execution_mode': 'simulation',
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'order_id': order_id
            }
    
    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """
        Get order status from exchange.
        
        Args:
            order_id: Order ID
            symbol: Trading symbol
            
        Returns:
            Dictionary with order status
        """
        try:
            if self.execution_mode == ExecutionMode.SIMULATION:
                # Return simulated status
                if order_id in self.active_orders:
                    return {
                        'success': True,
                        'order': self.active_orders[order_id]
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Order not found'
                    }
            
            # Get order status from exchange
            endpoint = "/v5/order/realtime"
            params = {
                'symbol': symbol,
                'orderId': order_id
            }
            
            result = await self._make_api_request('GET', endpoint, params=params)
            
            if result.get('retCode') == 0:
                order_list = result.get('result', {}).get('list', [])
                if order_list:
                    order = order_list[0]
                    return {
                        'success': True,
                        'order': {
                            'order_id': order.get('orderId'),
                            'symbol': order.get('symbol'),
                            'side': order.get('side'),
                            'order_type': order.get('orderType'),
                            'quantity': float(order.get('qty', 0)),
                            'price': float(order.get('price', 0)),
                            'status': order.get('orderStatus'),
                            'created_at': order.get('createdTime'),
                            'updated_at': order.get('updatedTime')
                        }
                    }
                else:
                    return {
                        'success': False,
                        'error': 'Order not found on exchange'
                    }
            else:
                return {
                    'success': False,
                    'error': result.get('retMsg', 'Unknown error')
                }
                
        except Exception as e:
            self.logger.error(f"Error getting order status for {order_id}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def get_open_orders(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get open orders from exchange.
        
        Args:
            symbol: Optional symbol filter
            
        Returns:
            Dictionary with open orders
        """
        try:
            if self.execution_mode == ExecutionMode.SIMULATION:
                # Return simulated open orders
                open_orders = list(self.active_orders.values())
                if symbol:
                    open_orders = [order for order in open_orders if order['symbol'] == symbol]
                
                return {
                    'success': True,
                    'orders': open_orders,
                    'count': len(open_orders)
                }
            
            # Get open orders from exchange
            endpoint = "/v5/order/realtime"
            params = {}
            if symbol:
                params['symbol'] = symbol
            
            result = await self._make_api_request('GET', endpoint, params=params)
            
            if result.get('retCode') == 0:
                orders = result.get('result', {}).get('list', [])
                return {
                    'success': True,
                    'orders': orders,
                    'count': len(orders)
                }
            else:
                return {
                    'success': False,
                    'error': result.get('retMsg', 'Unknown error'),
                    'orders': [],
                    'count': 0
                }
                
        except Exception as e:
            self.logger.error(f"Error getting open orders: {e}")
            return {
                'success': False,
                'error': str(e),
                'orders': [],
                'count': 0
            }
    
    async def place_oco_order(self, 
                             symbol: str, 
                             side: str, 
                             quantity: float,
                             limit_price: float,
                             stop_price: float,
                             stop_limit_price: float) -> Dict[str, Any]:
        """
        Place OCO (One-Cancels-Other) order.
        
        Args:
            symbol: Trading symbol
            side: Order side ('Buy' or 'Sell')
            quantity: Order quantity
            limit_price: Limit order price
            stop_price: Stop order price
            stop_limit_price: Stop limit order price
            
        Returns:
            Dictionary with OCO order result
        """
        try:
            # Generate OCO order ID
            oco_order_id = self.risk_module.generate_order_id(symbol, side, "OCO")
            
            # Place limit order
            limit_result = await self.place_order(
                symbol=symbol,
                side=side,
                order_type="Limit",
                quantity=quantity,
                price=limit_price,
                time_in_force="GTC"
            )
            
            if not limit_result['success']:
                return {
                    'success': False,
                    'error': f"Failed to place limit order: {limit_result['error']}",
                    'oco_order_id': oco_order_id
                }
            
            # Place stop limit order
            stop_result = await self.place_order(
                symbol=symbol,
                side=side,
                order_type="StopLimit",
                quantity=quantity,
                price=stop_limit_price,
                stop_price=stop_price,
                time_in_force="GTC"
            )
            
            if not stop_result['success']:
                # Cancel limit order if stop order failed
                await self.cancel_order(limit_result['order_id'], symbol)
                return {
                    'success': False,
                    'error': f"Failed to place stop order: {stop_result['error']}",
                    'oco_order_id': oco_order_id
                }
            
            # Track OCO order
            oco_order = {
                'oco_order_id': oco_order_id,
                'symbol': symbol,
                'side': side,
                'quantity': quantity,
                'limit_order_id': limit_result['order_id'],
                'stop_order_id': stop_result['order_id'],
                'limit_price': limit_price,
                'stop_price': stop_price,
                'stop_limit_price': stop_limit_price,
                'status': 'Active',
                'created_at': datetime.now().isoformat()
            }
            
            self.active_orders[oco_order_id] = oco_order
            
            return {
                'success': True,
                'oco_order_id': oco_order_id,
                'limit_order_id': limit_result['order_id'],
                'stop_order_id': stop_result['order_id'],
                'oco_order': oco_order
            }
            
        except Exception as e:
            self.logger.error(f"Error placing OCO order for {symbol}: {e}")
            return {
                'success': False,
                'error': str(e),
                'oco_order_id': None
            }
    
    async def reconcile_orders(self) -> Dict[str, Any]:
        """
        Reconcile local orders with exchange orders.
        
        Returns:
            Dictionary with reconciliation results
        """
        try:
            # Get open orders from exchange
            exchange_orders_result = await self.get_open_orders()
            if not exchange_orders_result['success']:
                return {
                    'success': False,
                    'error': 'Failed to get exchange orders',
                    'reconciliation': {}
                }
            
            exchange_orders = exchange_orders_result['orders']
            
            # Use risk module reconciliation
            if self.risk_module:
                reconciliation = self.risk_module.reconcile_orders(exchange_orders)
            else:
                reconciliation = {
                    'matched_orders': 0,
                    'missing_orders': 0,
                    'extra_orders': 0,
                    'status_mismatches': 0,
                    'details': []
                }
            
            return {
                'success': True,
                'reconciliation': reconciliation,
                'exchange_orders_count': len(exchange_orders),
                'local_orders_count': len(self.active_orders)
            }
            
        except Exception as e:
            self.logger.error(f"Error reconciling orders: {e}")
            return {
                'success': False,
                'error': str(e),
                'reconciliation': {}
            }
    
    def _update_execution_stats(self, success: bool, execution_time: float):
        """Update execution statistics."""
        self.execution_stats['total_orders'] += 1
        self.execution_stats['total_execution_time'] += execution_time
        
        if success:
            self.execution_stats['successful_orders'] += 1
        else:
            self.execution_stats['failed_orders'] += 1
        
        # Update average execution time
        self.execution_stats['average_execution_time'] = (
            self.execution_stats['total_execution_time'] / 
            self.execution_stats['total_orders']
        )
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """
        Get execution module summary.
        
        Returns:
            Dictionary with execution summary
        """
        try:
            # Calculate success rate
            success_rate = 0.0
            if self.execution_stats['total_orders'] > 0:
                success_rate = (
                    self.execution_stats['successful_orders'] / 
                    self.execution_stats['total_orders']
                )
            
            summary = {
                'execution_mode': self.execution_mode.value,
                'active_orders': len(self.active_orders),
                'order_history': len(self.order_history),
                'failed_orders': len(self.failed_orders),
                'execution_stats': self.execution_stats,
                'success_rate': success_rate,
                'rate_limit_remaining': self.rate_limit_remaining,
                'rate_limit_reset': self.rate_limit_reset,
                'api_base_url': self.api_base_url,
                'testnet': self.testnet
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error getting execution summary: {e}")
            return {
                'error': str(e)
            }
    
    async def close(self):
        """Close the execution module and cleanup resources."""
        try:
            if self.session and not self.session.closed:
                await self.session.close()
            self.logger.info("Order execution module closed")
        except Exception as e:
            self.logger.error(f"Error closing execution module: {e}")


def main():
    """Main execution function for testing."""
    import asyncio
    
    async def test_execution():
        # Configuration
        config = {
            'api_base_url': 'https://api-testnet.bybit.com',
            'testnet': True,
            'max_retries': 3,
            'retry_delay': 1.0,
            'timeout': 10.0
        }
        
        # Initialize execution module
        execution_module = OrderExecutionModule(
            config=config,
            execution_mode=ExecutionMode.SIMULATION
        )
        
        try:
            # Test order placement
            order_result = await execution_module.place_order(
                symbol="BTCUSDT",
                side="Buy",
                order_type="Limit",
                quantity=0.001,
                price=50000
            )
            print("Order Placement Result:", order_result)
            
            # Test order status
            if order_result['success']:
                status_result = await execution_module.get_order_status(
                    order_result['order_id'], 
                    "BTCUSDT"
                )
                print("Order Status Result:", status_result)
            
            # Test execution summary
            summary = execution_module.get_execution_summary()
            print("Execution Summary:", summary)
            
        finally:
            await execution_module.close()
    
    # Run test
    asyncio.run(test_execution())


if __name__ == "__main__":
    main()



