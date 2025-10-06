"""
CCXT Futures Adapter for Crypto Trading Bot

This module provides a comprehensive CCXT-based futures trading adapter with:
- Idempotent order management with clientOrderId
- Tick size, step size, and minimum notional enforcement
- Reduce-only order support
- OCO (One-Cancels-Other) orders with mark-price stops
- Retry logic with exponential backoff (1s, 3s, 9s)
- Reconciliation loop for orphaned orders
- Circuit breaker for error burst protection
"""

import ccxt
import asyncio
import logging
import time
import json
import uuid
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
from enum import Enum
import pandas as pd
import numpy as np

class OrderStatus(Enum):
    NEW = "new"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"

class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    OCO = "oco"

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CCXTFuturesAdapter:
    """
    CCXT Futures Adapter with comprehensive order management and safety features.
    """
    
    def __init__(self, 
                 exchange_id: str = "bybit",
                 api_key: Optional[str] = None,
                 secret: Optional[str] = None,
                 sandbox: bool = True,
                 log_level: str = "INFO"):
        """
        Initialize CCXT Futures Adapter.
        
        Args:
            exchange_id: CCXT exchange identifier
            api_key: API key for authentication
            secret: API secret for authentication
            sandbox: Use sandbox/testnet mode
            log_level: Logging level
        """
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.secret = secret
        self.sandbox = sandbox
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Initialize exchange
        self.exchange = self._initialize_exchange()
        
        # Order management
        self.active_orders = {}
        self.order_history = []
        self.client_order_ids = set()
        
        # Circuit breaker
        self.circuit_breaker_state = CircuitBreakerState.CLOSED
        self.error_count = 0
        self.error_window_start = None
        self.max_errors = 10
        self.error_window_seconds = 60
        self.circuit_breaker_timeout = 300  # 5 minutes
        
        # Reconciliation
        self.reconciliation_interval = 30  # seconds
        self.last_reconciliation = None
        
        # Retry configuration
        self.retry_intervals = [1, 3, 9]  # seconds
        self.max_retries = 3
        
        self.logger.info(f"CCXT Futures Adapter initialized for {exchange_id}")
    
    def _initialize_exchange(self):
        """Initialize CCXT exchange with proper configuration."""
        try:
            exchange_class = getattr(ccxt, self.exchange_id)
            config = {
                'apiKey': self.api_key,
                'secret': self.secret,
                'sandbox': self.sandbox,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future',  # Use futures market
                    'adjustForTimeDifference': True,
                }
            }
            
            exchange = exchange_class(config)
            self.logger.info(f"Exchange {self.exchange_id} initialized successfully")
            return exchange
            
        except Exception as e:
            self.logger.error(f"Failed to initialize exchange {self.exchange_id}: {e}")
            raise
    
    def _generate_client_order_id(self, symbol: str, side: str, order_type: str) -> str:
        """Generate unique client order ID for idempotency."""
        timestamp = int(time.time() * 1000)
        unique_id = str(uuid.uuid4())[:8]
        return f"{symbol}_{side}_{order_type}_{timestamp}_{unique_id}"
    
    def _check_circuit_breaker(self) -> bool:
        """Check if circuit breaker should allow trading."""
        now = time.time()
        
        if self.circuit_breaker_state == CircuitBreakerState.OPEN:
            if now - self.error_window_start > self.circuit_breaker_timeout:
                self.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
                self.logger.info("Circuit breaker moved to HALF_OPEN state")
                return True
            return False
        
        elif self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
            return True
        
        else:  # CLOSED
            if self.error_window_start and now - self.error_window_start > self.error_window_seconds:
                self.error_count = 0
                self.error_window_start = None
            
            return True
    
    def _update_circuit_breaker(self, success: bool):
        """Update circuit breaker state based on operation result."""
        now = time.time()
        
        if success:
            if self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
                self.circuit_breaker_state = CircuitBreakerState.CLOSED
                self.error_count = 0
                self.logger.info("Circuit breaker moved to CLOSED state")
        else:
            if not self.error_window_start:
                self.error_window_start = now
            
            self.error_count += 1
            
            if self.error_count >= self.max_errors:
                self.circuit_breaker_state = CircuitBreakerState.OPEN
                self.logger.warning(f"Circuit breaker OPENED due to {self.error_count} errors")
    
    async def _retry_with_backoff(self, func, *args, **kwargs):
        """Execute function with exponential backoff retry logic."""
        for attempt in range(self.max_retries + 1):
            try:
                if not self._check_circuit_breaker():
                    raise Exception("Circuit breaker is OPEN - trading halted")
                
                result = await func(*args, **kwargs)
                self._update_circuit_breaker(True)
                return result
                
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed: {e}")
                
                if attempt < self.max_retries:
                    wait_time = self.retry_intervals[attempt]
                    self.logger.info(f"Retrying in {wait_time} seconds...")
                    await asyncio.sleep(wait_time)
                else:
                    self._update_circuit_breaker(False)
                    raise
    
    def _validate_order_parameters(self, symbol: str, side: str, amount: float, 
                                 price: Optional[float] = None, order_type: str = "limit") -> Dict[str, Any]:
        """Validate order parameters against exchange rules."""
        try:
            # Get market info
            market = self.exchange.market(symbol)
            
            # Validate symbol
            if not market:
                raise ValueError(f"Invalid symbol: {symbol}")
            
            # Validate amount against minimum
            min_amount = market.get('limits', {}).get('amount', {}).get('min', 0)
            if amount < min_amount:
                raise ValueError(f"Amount {amount} below minimum {min_amount}")
            
            # Validate step size
            step_size = market.get('precision', {}).get('amount', 0)
            if step_size > 0:
                amount = round(amount / step_size) * step_size
            
            # Validate price if provided
            if price is not None:
                min_price = market.get('limits', {}).get('price', {}).get('min', 0)
                if price < min_price:
                    raise ValueError(f"Price {price} below minimum {min_price}")
                
                # Validate tick size
                tick_size = market.get('precision', {}).get('price', 0)
                if tick_size > 0:
                    price = round(price / tick_size) * tick_size
            
            # Validate notional value
            if price is not None:
                notional = amount * price
                min_notional = market.get('limits', {}).get('cost', {}).get('min', 0)
                if notional < min_notional:
                    raise ValueError(f"Notional {notional} below minimum {min_notional}")
            
            return {
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'type': order_type,
                'market': market
            }
            
        except Exception as e:
            self.logger.error(f"Order validation failed: {e}")
            raise
    
    async def create_order(self, symbol: str, side: str, amount: float,
                          order_type: str = "limit", price: Optional[float] = None,
                          client_order_id: Optional[str] = None, 
                          reduce_only: bool = False,
                          time_in_force: str = "GTC") -> Dict[str, Any]:
        """
        Create an order with idempotent clientOrderId functionality.
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            amount: Order amount
            order_type: Order type (market/limit/stop)
            price: Order price (for limit orders)
            client_order_id: Client order ID for idempotency
            reduce_only: Whether this is a reduce-only order
            time_in_force: Time in force (GTC, IOC, FOK)
        
        Returns:
            Order information
        """
        try:
            # Generate client order ID if not provided
            if not client_order_id:
                client_order_id = self._generate_client_order_id(symbol, side, order_type)
            
            # Check for duplicate client order ID
            if client_order_id in self.client_order_ids:
                self.logger.warning(f"Duplicate client order ID: {client_order_id}")
                # Return existing order if found
                existing_order = self._find_order_by_client_id(client_order_id)
                if existing_order:
                    return existing_order
            
            # Validate order parameters
            validated_params = self._validate_order_parameters(
                symbol, side, amount, price, order_type
            )
            
            # Prepare order parameters
            order_params = {
                'symbol': validated_params['symbol'],
                'side': validated_params['side'],
                'amount': validated_params['amount'],
                'type': validated_params['type'],
                'clientOrderId': client_order_id,
                'reduceOnly': reduce_only,
                'timeInForce': time_in_force
            }
            
            if price is not None:
                order_params['price'] = validated_params['price']
            
            # Create order with retry logic
            async def _create_order():
                return await self.exchange.create_order(**order_params)
            
            result = await self._retry_with_backoff(_create_order)
            
            # Store order information
            self.client_order_ids.add(client_order_id)
            self.active_orders[result['id']] = {
                'order': result,
                'client_order_id': client_order_id,
                'created_at': datetime.now(),
                'status': OrderStatus.NEW
            }
            
            self.logger.info(f"Order created: {result['id']} (client: {client_order_id})")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to create order: {e}")
            raise
    
    async def create_oco_order(self, symbol: str, side: str, amount: float,
                              stop_price: float, limit_price: float,
                              client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create an OCO (One-Cancels-Other) order with mark-price stops.
        
        Args:
            symbol: Trading symbol
            side: Order side (buy/sell)
            amount: Order amount
            stop_price: Stop price for stop order
            limit_price: Limit price for limit order
            client_order_id: Client order ID for idempotency
        
        Returns:
            OCO order information
        """
        try:
            if not client_order_id:
                client_order_id = self._generate_client_order_id(symbol, side, "oco")
            
            # Create OCO order parameters
            oco_params = {
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'stopPrice': stop_price,
                'price': limit_price,
                'clientOrderId': client_order_id,
                'type': 'oco'
            }
            
            async def _create_oco_order():
                return await self.exchange.create_order(**oco_params)
            
            result = await self._retry_with_backoff(_create_oco_order)
            
            self.logger.info(f"OCO order created: {result['id']} (client: {client_order_id})")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to create OCO order: {e}")
            raise
    
    async def cancel_order(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Cancel an order."""
        try:
            async def _cancel_order():
                return await self.exchange.cancel_order(order_id, symbol)
            
            result = await self._retry_with_backoff(_cancel_order)
            
            # Update local order status
            if order_id in self.active_orders:
                self.active_orders[order_id]['status'] = OrderStatus.CANCELED
            
            self.logger.info(f"Order canceled: {order_id}")
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to cancel order {order_id}: {e}")
            raise
    
    async def get_order_status(self, order_id: str, symbol: str) -> Dict[str, Any]:
        """Get order status from exchange."""
        try:
            async def _get_order():
                return await self.exchange.fetch_order(order_id, symbol)
            
            result = await self._retry_with_backoff(_get_order)
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to get order status {order_id}: {e}")
            raise
    
    async def reconcile_orders(self) -> Dict[str, Any]:
        """
        Reconcile local orders with exchange orders to detect orphaned orders.
        """
        try:
            reconciliation_results = {
                'total_orders': len(self.active_orders),
                'reconciled_orders': 0,
                'orphaned_orders': 0,
                'updated_orders': 0,
                'errors': []
            }
            
            # Get all open orders from exchange
            exchange_orders = await self.exchange.fetch_open_orders()
            exchange_order_ids = {order['id'] for order in exchange_orders}
            
            # Check local orders against exchange
            for order_id, local_order in list(self.active_orders.items()):
                try:
                    if order_id in exchange_order_ids:
                        # Order exists on exchange, update status
                        exchange_order = next(
                            (o for o in exchange_orders if o['id'] == order_id), None
                        )
                        if exchange_order:
                            self.active_orders[order_id]['order'] = exchange_order
                            self.active_orders[order_id]['status'] = OrderStatus(exchange_order['status'])
                            reconciliation_results['updated_orders'] += 1
                    else:
                        # Order not found on exchange - might be filled or canceled
                        try:
                            order_status = await self.get_order_status(order_id, local_order['order']['symbol'])
                            self.active_orders[order_id]['order'] = order_status
                            self.active_orders[order_id]['status'] = OrderStatus(order_status['status'])
                            reconciliation_results['reconciled_orders'] += 1
                        except:
                            # Order truly orphaned
                            reconciliation_results['orphaned_orders'] += 1
                            self.logger.warning(f"Orphaned order detected: {order_id}")
                            
                            # Remove from active orders
                            del self.active_orders[order_id]
                    
                except Exception as e:
                    reconciliation_results['errors'].append(f"Error reconciling order {order_id}: {e}")
            
            self.last_reconciliation = datetime.now()
            self.logger.info(f"Reconciliation completed: {reconciliation_results}")
            return reconciliation_results
            
        except Exception as e:
            self.logger.error(f"Reconciliation failed: {e}")
            return {'error': str(e)}
    
    def _find_order_by_client_id(self, client_order_id: str) -> Optional[Dict[str, Any]]:
        """Find order by client order ID."""
        for order_data in self.active_orders.values():
            if order_data['client_order_id'] == client_order_id:
                return order_data['order']
        return None
    
    async def start_reconciliation_loop(self):
        """Start the periodic reconciliation loop."""
        while True:
            try:
                await self.reconcile_orders()
                await asyncio.sleep(self.reconciliation_interval)
            except Exception as e:
                self.logger.error(f"Reconciliation loop error: {e}")
                await asyncio.sleep(self.reconciliation_interval)
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get circuit breaker status."""
        return {
            'state': self.circuit_breaker_state.value,
            'error_count': self.error_count,
            'error_window_start': self.error_window_start,
            'trading_allowed': self._check_circuit_breaker()
        }
    
    def get_active_orders(self) -> Dict[str, Any]:
        """Get all active orders."""
        return {
            'active_orders': len(self.active_orders),
            'orders': list(self.active_orders.values())
        }
    
    async def close(self):
        """Close the adapter and cleanup resources."""
        try:
            await self.exchange.close()
            self.logger.info("CCXT Futures Adapter closed")
        except Exception as e:
            self.logger.error(f"Error closing adapter: {e}")
