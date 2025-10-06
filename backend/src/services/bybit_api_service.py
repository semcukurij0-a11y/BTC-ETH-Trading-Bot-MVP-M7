#!/usr/bin/env python3
"""
Bybit API Service for Crypto Trading Bot

This module provides authenticated API access to Bybit Testnet:
- REST API authentication and request signing
- WebSocket connection management
- Wallet balance fetching
- Order placement and management
- Position tracking
- Real-time market data subscriptions
"""

import hashlib
import hmac
import time
import json
import logging
import asyncio
import websockets
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from urllib.parse import urlencode
import pandas as pd


class BybitAPIService:
    """
    Bybit API Service with proper authentication for testnet trading.
    """
    
    def __init__(self, 
                 api_key: str,
                 api_secret: str,
                 base_url: str = "https://api-testnet.bybit.com",
                 ws_url: str = "wss://stream-testnet.bybit.com/realtime",
                 testnet: bool = True):
        """
        Initialize Bybit API Service.
        
        Args:
            api_key: Bybit API key
            api_secret: Bybit API secret
            base_url: REST API base URL
            ws_url: WebSocket URL
            testnet: Whether using testnet
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.base_url = base_url
        self.ws_url = ws_url
        self.testnet = testnet
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Session for connection pooling with optimized settings
        self.session = requests.Session()
        
        # Configure connection pooling for better performance
        adapter = HTTPAdapter(
            pool_connections=10,  # Number of connection pools
            pool_maxsize=20,      # Maximum connections in pool
            max_retries=Retry(
                total=2,          # Total retries
                backoff_factor=0.3,  # Backoff factor for retries
                status_forcelist=[500, 502, 503, 504]  # HTTP status codes to retry
            )
        )
        
        # Mount adapter for both HTTP and HTTPS
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)
        
        self.session.headers.update({
            'Content-Type': 'application/json',
            'User-Agent': 'Crypto-Trading-Bot/1.0',
            'Connection': 'keep-alive'  # Enable keep-alive connections
        })
        
        # WebSocket connection
        self.ws_connection = None
        self.ws_subscriptions = set()
        
        self.logger.info(f"Bybit API Service initialized for {'testnet' if testnet else 'mainnet'}")
    
    def _generate_signature(self, params: Dict[str, Any], timestamp: str, is_post: bool = False) -> str:
        """
        Generate HMAC SHA256 signature for Bybit API authentication.
        
        Args:
            params: Request parameters
            timestamp: Request timestamp
            is_post: Whether this is a POST request (JSON body) or GET request (query params)
            
        Returns:
            HMAC SHA256 signature
        """
        if params:
            if is_post:
                # For POST requests with JSON body, use exact JSON string format
                # Ensure the JSON string matches exactly what will be sent in the request
                param_string = json.dumps(params, separators=(',', ':'), ensure_ascii=False)
            else:
                # For GET requests, use query string format
                sorted_params = sorted(params.items())
                param_string = urlencode(sorted_params)
        else:
            param_string = ""
        
        # Create signature payload (correct format for Bybit v5)
        # Format: timestamp + api_key + recv_window + param_string
        signature_payload = f"{timestamp}{self.api_key}10000{param_string}"
        
        # Debug logging
        self.logger.debug(f"Signature payload: {signature_payload}")
        self.logger.debug(f"Params: {params}")
        self.logger.debug(f"Is POST: {is_post}")
        
        # Generate HMAC SHA256 signature
        signature = hmac.new(
            self.api_secret.encode('utf-8'),
            signature_payload.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        self.logger.debug(f"Generated signature: {signature}")
        return signature
    
    def _get_headers(self, params: Dict[str, Any] = None, is_post: bool = False) -> Dict[str, str]:
        """
        Get authenticated headers for API requests.
        
        Args:
            params: Request parameters
            is_post: Whether this is a POST request (JSON body) or GET request (query params)
            
        Returns:
            Headers dictionary with authentication
        """
        # Use current timestamp in milliseconds
        timestamp = str(int(time.time() * 1000))
        
        if params is None:
            params = {}
        
        # Generate signature using the original params (without adding auth params)
        signature = self._generate_signature(params, timestamp, is_post)
        
        return {
            'X-BAPI-API-KEY': self.api_key,
            'X-BAPI-SIGN': signature,
            'X-BAPI-SIGN-TYPE': '2',
            'X-BAPI-TIMESTAMP': timestamp,
            'X-BAPI-RECV-WINDOW': '10000'  # Increased recv_window to 10 seconds
        }
    
    async def get_wallet_balance(self, max_retries: int = 2) -> Dict[str, Any]:
        """
        Get wallet balance from Bybit v5 API with retry logic.
        
        Args:
            max_retries: Maximum number of retry attempts
            
        Returns:
            Wallet balance information
        """
        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}/v5/account/wallet-balance"
                
                params = {
                    'accountType': 'UNIFIED'
                }
                
                headers = self._get_headers(params)
                
                # Add timeout and retry logic
                response = self.session.get(
                    url, 
                    headers=headers, 
                    params=params,
                    timeout=30  # Increased from 10s to 30s for better reliability
                )
                response.raise_for_status()
                
                data = response.json()
                
                if data.get('retCode') == 0:
                    self.logger.info("Successfully fetched wallet balance")
                    return {
                        'success': True,
                        'data': data.get('result', {}),
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    error_msg = f"API error: {data.get('retMsg', 'Unknown error')}"
                    self.logger.error(error_msg)
                    if attempt < max_retries - 1:
                        self.logger.info(f"Retrying in 2 seconds... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(2)
                        continue
                    return {
                        'success': False,
                        'error': error_msg,
                        'timestamp': datetime.now().isoformat()
                    }
                    
            except requests.exceptions.ConnectionError as e:
                error_msg = f"Connection error: {str(e)}"
                self.logger.error(error_msg)
                if attempt < max_retries - 1:
                    # Faster retry for connection errors
                    retry_delay = 0.5 + (attempt * 0.5)  # 0.5s, 1.0s
                    self.logger.info(f"Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    continue
                return {
                    'success': False,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                }
            except requests.exceptions.Timeout as e:
                error_msg = f"Timeout error: {str(e)}"
                self.logger.error(error_msg)
                if attempt < max_retries - 1:
                    # Faster retry for close position
                    retry_delay = 0.3 + (attempt * 0.2)  # 0.3s, 0.5s, 0.7s
                    self.logger.info(f"Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    continue
                return {
                    'success': False,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                self.logger.error(error_msg)
                if attempt < max_retries - 1:
                    # Faster retry for close position
                    retry_delay = 0.3 + (attempt * 0.2)  # 0.3s, 0.5s, 0.7s
                    self.logger.info(f"Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    continue
                return {
                    'success': False,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                }
        
        # If we get here, all retries failed
        return {
            'success': False,
            'error': 'All retry attempts failed',
            'timestamp': datetime.now().isoformat()
        }
    
    async def get_positions(self, symbol: str = None, max_retries: int = 2) -> Dict[str, Any]:
        """
        Get current positions from Bybit v5 API with retry logic.
        
        Args:
            symbol: Trading symbol (optional)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Positions information
        """
        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}/v5/position/list"
                
                params = {
                    'category': 'linear'
                }
                if symbol:
                    params['symbol'] = symbol
                else:
                    # If no symbol specified, get all positions by using settleCoin
                    params['settleCoin'] = 'USDT'
                
                headers = self._get_headers(params)
                
                response = self.session.get(
                    url, 
                    headers=headers, 
                    params=params,
                    timeout=30  # Increased from 10s to 30s for better reliability
                )
                response.raise_for_status()
                
                data = response.json()
                
                if data.get('retCode') == 0:
                    self.logger.info(f"Successfully fetched positions for {symbol or 'all symbols'}")
                    return {
                        'success': True,
                        'data': data.get('result', {}),
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    error_msg = f"API error: {data.get('retMsg', 'Unknown error')}"
                    self.logger.error(error_msg)
                    if attempt < max_retries - 1:
                        self.logger.info(f"Retrying in 2 seconds... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(2)
                        continue
                    return {
                        'success': False,
                        'error': error_msg,
                        'timestamp': datetime.now().isoformat()
                    }
                    
            except requests.exceptions.ConnectionError as e:
                error_msg = f"Connection error: {str(e)}"
                self.logger.error(error_msg)
                if attempt < max_retries - 1:
                    # Faster retry for connection errors
                    retry_delay = 0.5 + (attempt * 0.5)  # 0.5s, 1.0s
                    self.logger.info(f"Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    continue
                return {
                    'success': False,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                }
            except requests.exceptions.Timeout as e:
                error_msg = f"Timeout error: {str(e)}"
                self.logger.error(error_msg)
                if attempt < max_retries - 1:
                    # Faster retry for close position
                    retry_delay = 0.3 + (attempt * 0.2)  # 0.3s, 0.5s, 0.7s
                    self.logger.info(f"Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    continue
                return {
                    'success': False,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                self.logger.error(error_msg)
                if attempt < max_retries - 1:
                    # Faster retry for close position
                    retry_delay = 0.3 + (attempt * 0.2)  # 0.3s, 0.5s, 0.7s
                    self.logger.info(f"Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    continue
                return {
                    'success': False,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                }
        
        # If we get here, all retries failed
        return {
            'success': False,
            'error': 'All retry attempts failed',
            'timestamp': datetime.now().isoformat()
        }
    
    async def place_order(self, 
                         symbol: str,
                         side: str,
                         order_type: str,
                         qty: float,
                         price: float = None,
                         time_in_force: str = "GTC") -> Dict[str, Any]:
        """
        Place an order on Bybit v5 API.
        
        Args:
            symbol: Trading symbol
            side: Order side (Buy/Sell)
            order_type: Order type (Market/Limit)
            qty: Order quantity
            price: Order price (for limit orders)
            time_in_force: Time in force (GTC, IOC, FOK)
            
        Returns:
            Order placement result
        """
        try:
            url = f"{self.base_url}/v5/order/create"
            
            params = {
                'category': 'linear',
                'symbol': symbol,
                'side': side,
                'orderType': order_type,
                'qty': str(qty),
                'timeInForce': time_in_force
            }
            
            if price and order_type == "Limit":
                params['price'] = str(price)
            
            headers = self._get_headers(params, is_post=True)
            
            response = self.session.post(url, headers=headers, json=params)
            response.raise_for_status()
            
            data = response.json()
            
            if data.get('retCode') == 0:
                self.logger.info(f"Successfully placed {side} {order_type} order for {symbol}")
                return {
                    'success': True,
                    'data': data.get('result', {}),
                    'timestamp': datetime.now().isoformat()
                }
            else:
                error_msg = f"API error: {data.get('retMsg', 'Unknown error')}"
                self.logger.error(error_msg)
                return {
                    'success': False,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            error_msg = f"Error placing order: {e}"
            self.logger.error(error_msg)
            return {
                'success': False,
                'error': error_msg,
                'timestamp': datetime.now().isoformat()
            }
    
    async def connect_websocket(self) -> bool:
        """
        Connect to Bybit WebSocket for real-time data.
        
        Returns:
            True if connection successful
        """
        try:
            self.ws_connection = await websockets.connect(self.ws_url)
            self.logger.info("WebSocket connected to Bybit")
            return True
        except Exception as e:
            self.logger.error(f"WebSocket connection failed: {e}")
            return False
    
    async def subscribe_to_trades(self, symbol: str) -> bool:
        """
        Subscribe to trade updates for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if subscription successful
        """
        if not self.ws_connection:
            if not await self.connect_websocket():
                return False
        
        try:
            subscription = {
                "op": "subscribe",
                "args": [f"trade.{symbol}"]
            }
            
            await self.ws_connection.send(json.dumps(subscription))
            self.ws_subscriptions.add(f"trade.{symbol}")
            self.logger.info(f"Subscribed to trades for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to trades for {symbol}: {e}")
            return False
    
    async def subscribe_to_orderbook(self, symbol: str) -> bool:
        """
        Subscribe to order book updates for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            True if subscription successful
        """
        if not self.ws_connection:
            if not await self.connect_websocket():
                return False
        
        try:
            subscription = {
                "op": "subscribe",
                "args": [f"orderBookL2_25.{symbol}"]
            }
            
            await self.ws_connection.send(json.dumps(subscription))
            self.ws_subscriptions.add(f"orderBookL2_25.{symbol}")
            self.logger.info(f"Subscribed to order book for {symbol}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to order book for {symbol}: {e}")
            return False
    
    async def subscribe_to_positions(self) -> bool:
        """
        Subscribe to position updates.
        
        Returns:
            True if subscription successful
        """
        if not self.ws_connection:
            if not await self.connect_websocket():
                return False
        
        try:
            subscription = {
                "op": "subscribe",
                "args": ["position"]
            }
            
            await self.ws_connection.send(json.dumps(subscription))
            self.ws_subscriptions.add("position")
            self.logger.info("Subscribed to position updates")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to subscribe to positions: {e}")
            return False
    
    async def listen_for_updates(self, callback=None):
        """
        Listen for WebSocket updates.
        
        Args:
            callback: Callback function for handling updates
        """
        if not self.ws_connection:
            self.logger.error("WebSocket not connected")
            return
        
        try:
            async for message in self.ws_connection:
                try:
                    data = json.loads(message)
                    
                    if callback:
                        await callback(data)
                    else:
                        self.logger.info(f"Received update: {data}")
                        
                except json.JSONDecodeError:
                    self.logger.error(f"Failed to parse message: {message}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.logger.warning("WebSocket connection closed")
        except Exception as e:
            self.logger.error(f"Error listening for updates: {e}")
    
    async def close_websocket(self):
        """Close WebSocket connection."""
        if self.ws_connection:
            await self.ws_connection.close()
            self.logger.info("WebSocket connection closed")
    
    async def test_connection(self) -> Dict[str, Any]:
        """
        Test API connection and authentication.
        
        Returns:
            Connection test result
        """
        try:
            # Test wallet balance endpoint
            balance_result = await self.get_wallet_balance()
            
            if balance_result['success']:
                return {
                    'success': True,
                    'message': 'Bybit API connection successful',
                    'testnet': self.testnet,
                    'timestamp': datetime.now().isoformat()
                }
            else:
                return {
                    'success': False,
                    'error': balance_result.get('error', 'Unknown error'),
                    'timestamp': datetime.now().isoformat()
                }
                
        except Exception as e:
            return {
                'success': False,
                'error': f"Connection test failed: {e}",
                'timestamp': datetime.now().isoformat()
            }
    
    async def get_order_history(self, symbol: str = None, max_retries: int = 2) -> Dict[str, Any]:
        """
        Get order history from Bybit v5 API.
        
        Args:
            symbol: Trading symbol (optional)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Order history information
        """
        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}/v5/order/history"
                
                params = {
                    'category': 'linear',
                    'limit': 50
                }
                if symbol:
                    params['symbol'] = symbol
                else:
                    params['settleCoin'] = 'USDT'
                
                headers = self._get_headers(params)
                
                response = self.session.get(
                    url, 
                    headers=headers, 
                    params=params,
                    timeout=30  # Increased from 10s to 30s for better reliability
                )
                response.raise_for_status()
                
                data = response.json()
                
                if data.get('retCode') == 0:
                    return {
                        'success': True,
                        'data': data.get('result', {}),
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    error_msg = f"API error: {data.get('retMsg', 'Unknown error')}"
                    self.logger.error(error_msg)
                    if attempt < max_retries - 1:
                        self.logger.info(f"Retrying in 2 seconds... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(2)
                        continue
                    return {
                        'success': False,
                        'error': error_msg,
                        'timestamp': datetime.now().isoformat()
                    }
                    
            except requests.exceptions.RequestException as e:
                error_msg = f"Request error: {str(e)}"
                self.logger.error(error_msg)
                if attempt < max_retries - 1:
                    # Faster retry for close position
                    retry_delay = 0.3 + (attempt * 0.2)  # 0.3s, 0.5s, 0.7s
                    self.logger.info(f"Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    continue
                return {
                    'success': False,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                self.logger.error(error_msg)
                if attempt < max_retries - 1:
                    # Faster retry for close position
                    retry_delay = 0.3 + (attempt * 0.2)  # 0.3s, 0.5s, 0.7s
                    self.logger.info(f"Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    continue
                return {
                    'success': False,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                }
        
        return {
            'success': False,
            'error': 'All retry attempts failed',
            'timestamp': datetime.now().isoformat()
        }
    
    async def get_ticker(self, symbol: str, max_retries: int = 2) -> Dict[str, Any]:
        """
        Get ticker data for a symbol from Bybit v5 API.
        
        Args:
            symbol: Trading symbol
            max_retries: Maximum number of retry attempts
            
        Returns:
            Ticker information
        """
        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}/v5/market/tickers"
                
                params = {
                    'category': 'linear',
                    'symbol': symbol
                }
                
                headers = self._get_headers(params)
                
                response = self.session.get(
                    url, 
                    headers=headers, 
                    params=params,
                    timeout=30  # Increased from 10s to 30s for better reliability
                )
                response.raise_for_status()
                
                data = response.json()
                
                if data.get('retCode') == 0:
                    return {
                        'success': True,
                        'data': data.get('result', {}),
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    error_msg = f"API error: {data.get('retMsg', 'Unknown error')}"
                    self.logger.error(error_msg)
                    if attempt < max_retries - 1:
                        self.logger.info(f"Retrying in 2 seconds... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(2)
                        continue
                    return {
                        'success': False,
                        'error': error_msg,
                        'timestamp': datetime.now().isoformat()
                    }
                    
            except requests.exceptions.RequestException as e:
                error_msg = f"Request error: {str(e)}"
                self.logger.error(error_msg)
                if attempt < max_retries - 1:
                    # Faster retry for close position
                    retry_delay = 0.3 + (attempt * 0.2)  # 0.3s, 0.5s, 0.7s
                    self.logger.info(f"Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    continue
                return {
                    'success': False,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                self.logger.error(error_msg)
                if attempt < max_retries - 1:
                    # Faster retry for close position
                    retry_delay = 0.3 + (attempt * 0.2)  # 0.3s, 0.5s, 0.7s
                    self.logger.info(f"Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    continue
                return {
                    'success': False,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                }
        
        return {
            'success': False,
            'error': 'All retry attempts failed',
            'timestamp': datetime.now().isoformat()
        }
    
    async def get_kline(self, symbol: str, interval: str, limit: int = 200, max_retries: int = 2) -> Dict[str, Any]:
        """
        Get kline (candlestick) data for a symbol from Bybit v5 API.
        
        Args:
            symbol: Trading symbol
            interval: Time interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles to fetch
            max_retries: Maximum number of retry attempts
            
        Returns:
            Kline data
        """
        for attempt in range(max_retries):
            try:
                url = f"{self.base_url}/v5/market/kline"
                
                params = {
                    'category': 'linear',
                    'symbol': symbol,
                    'interval': interval,
                    'limit': limit
                }
                
                headers = self._get_headers(params)
                
                response = self.session.get(
                    url, 
                    headers=headers, 
                    params=params,
                    timeout=30
                )
                
                response.raise_for_status()
                response_data = response.json()
                
                if response_data.get('retCode') == 0:
                    return {
                        'success': True,
                        'data': response_data.get('result', {}),
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    return {
                        'success': False,
                        'error': f"API error: {response_data.get('retMsg', 'Unknown error')}",
                        'timestamp': datetime.now().isoformat()
                    }
                    
            except Exception as e:
                self.logger.warning(f"Attempt {attempt + 1} failed for get_kline: {e}")
                if attempt == max_retries - 1:
                    return {
                        'success': False,
                        'error': str(e),
                        'timestamp': datetime.now().isoformat()
                    }
                await asyncio.sleep(1)
        
        return {
            'success': False,
            'error': 'All retry attempts failed',
            'timestamp': datetime.now().isoformat()
        }
    
    async def close_position(self, symbol: str, side: str = None, size: float = None, max_retries: int = 2) -> Dict[str, Any]:
        """
        Close a position on Bybit v5 API.
        
        Args:
            symbol: Trading symbol
            side: Position side ('Buy' or 'Sell') - if None, will close all positions for symbol
            size: Position size - if provided, skips position lookup (faster)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Close position result
        """
        for attempt in range(max_retries):
            try:
                # Use provided size if available (faster - no API call needed)
                if size is not None and side is not None:
                    position_size = float(size)
                    print(f"Using provided position details: {side} {position_size} {symbol}")
                else:
                    # Fallback: Get the current position to determine the size
                    print(f"Looking up position details for {symbol}")
                    positions_result = await self.get_positions(symbol)
                    if not positions_result.get('success'):
                        return {
                            'success': False,
                            'error': f"Failed to get position info: {positions_result.get('error', 'Unknown error')}",
                            'timestamp': datetime.now().isoformat()
                        }
                    
                    # Find the position for this symbol
                    position_size = None
                    position_side = None
                    
                    if positions_result.get('data') and 'list' in positions_result.get('data'):
                        for pos in positions_result['data']['list']:
                            if pos.get('symbol') == symbol and float(pos.get('size', 0)) > 0:
                                position_size = float(pos.get('size', 0))
                                position_side = pos.get('side', '')
                                break
                    
                    if not position_size or position_size <= 0:
                        return {
                            'success': False,
                            'error': f"No active position found for {symbol}",
                            'timestamp': datetime.now().isoformat()
                        }
                    
                    # Determine the opposite side for closing
                    if not side:
                        side = 'Sell' if position_side == 'Buy' else 'Buy'
                
                url = f"{self.base_url}/v5/order/create"
                
                # Prepare order data with actual position size
                order_data = {
                    'category': 'linear',
                    'symbol': symbol,
                    'side': side,
                    'orderType': 'Market',
                    'qty': str(position_size),  # Use actual position size
                    'reduceOnly': True,  # This ensures we're closing a position
                    'timeInForce': 'IOC'  # Immediate or Cancel
                }
                
                # Create the exact JSON string that will be sent
                json_string = json.dumps(order_data, separators=(',', ':'), ensure_ascii=False)
                
                # Generate headers with the exact JSON string
                timestamp = str(int(time.time() * 1000))
                signature_payload = f"{timestamp}{self.api_key}10000{json_string}"
                signature = hmac.new(
                    self.api_secret.encode('utf-8'),
                    signature_payload.encode('utf-8'),
                    hashlib.sha256
                ).hexdigest()
                
                headers = {
                    'X-BAPI-API-KEY': self.api_key,
                    'X-BAPI-SIGN': signature,
                    'X-BAPI-SIGN-TYPE': '2',
                    'X-BAPI-TIMESTAMP': timestamp,
                    'X-BAPI-RECV-WINDOW': '10000',
                    'Content-Type': 'application/json'
                }
                
                self.logger.info(f"Attempting to close position: {symbol} with side {side}, size {position_size}")
                self.logger.debug(f"Request URL: {url}")
                self.logger.debug(f"Request headers: {headers}")
                self.logger.debug(f"Request body: {json_string}")
                
                response = self.session.post(
                    url, 
                    headers=headers, 
                    data=json_string,  # Send the exact JSON string
                    timeout=20  # Increased timeout for better reliability
                )
                
                self.logger.info(f"Response status: {response.status_code}")
                self.logger.debug(f"Response headers: {dict(response.headers)}")
                
                response.raise_for_status()
                
                data = response.json()
                self.logger.debug(f"Response data: {data}")
                
                if data.get('retCode') == 0:
                    self.logger.info(f"Position closed successfully: {symbol}")
                    return {
                        'success': True,
                        'data': data.get('result', {}),
                        'timestamp': datetime.now().isoformat()
                    }
                else:
                    error_msg = f"API error: {data.get('retMsg', 'Unknown error')}"
                    self.logger.error(f"Bybit API error: {error_msg}")
                    if attempt < max_retries - 1:
                        # Faster retry for API errors in close position
                        retry_delay = 0.5 + (attempt * 0.3)  # 0.5s, 0.8s, 1.1s
                        self.logger.info(f"Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                        await asyncio.sleep(retry_delay)
                        continue
                    return {
                        'success': False,
                        'error': error_msg,
                        'timestamp': datetime.now().isoformat()
                    }
                    
            except requests.exceptions.RequestException as e:
                error_msg = f"Request error: {str(e)}"
                self.logger.error(error_msg)
                if attempt < max_retries - 1:
                    # Faster retry for close position
                    retry_delay = 0.3 + (attempt * 0.2)  # 0.3s, 0.5s, 0.7s
                    self.logger.info(f"Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    continue
                return {
                    'success': False,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                }
            except Exception as e:
                error_msg = f"Unexpected error: {str(e)}"
                self.logger.error(error_msg)
                if attempt < max_retries - 1:
                    # Faster retry for close position
                    retry_delay = 0.3 + (attempt * 0.2)  # 0.3s, 0.5s, 0.7s
                    self.logger.info(f"Retrying in {retry_delay}s... (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_delay)
                    continue
                return {
                    'success': False,
                    'error': error_msg,
                    'timestamp': datetime.now().isoformat()
                }
        
        return {
            'success': False,
            'error': 'All retry attempts failed',
            'timestamp': datetime.now().isoformat()
        }


# Example usage and testing functions
async def example_wallet_balance():
    """Example: Fetch wallet balance"""
    api = BybitAPIService(
        api_key="popMizkoG6dZ5po90y",
        api_secret="zrJza3YTJBkw8BXx79n895akhlqRyNNmc8aW",
        base_url="https://api-testnet.bybit.com",
        ws_url="wss://stream-testnet.bybit.com/realtime",
        testnet=True
    )
    
    result = await api.get_wallet_balance()
    print(f"Wallet Balance Result: {json.dumps(result, indent=2)}")
    return result


async def example_trade_subscription():
    """Example: Subscribe to trade updates"""
    api = BybitAPIService(
        api_key="popMizkoG6dZ5po90y",
        api_secret="zrJza3YTJBkw8BXx79n895akhlqRyNNmc8aW",
        base_url="https://api-testnet.bybit.com",
        ws_url="wss://stream-testnet.bybit.com/realtime",
        testnet=True
    )
    
    # Connect and subscribe
    await api.connect_websocket()
    await api.subscribe_to_trades("BTCUSDT")
    
    # Listen for updates
    async def handle_update(data):
        print(f"Trade Update: {json.dumps(data, indent=2)}")
    
    await api.listen_for_updates(handle_update)


if __name__ == "__main__":
    # Test the API service
    import asyncio
    
    async def main():
        print("Testing Bybit API Service...")
        
        # Test connection
        api = BybitAPIService(
            api_key="popMizkoG6dZ5po90y",
            api_secret="zrJza3YTJBkw8BXx79n895akhlqRyNNmc8aW",
            testnet=True
        )
        
        # Test connection
        result = await api.test_connection()
        print(f"Connection Test: {json.dumps(result, indent=2)}")
        
        # Test wallet balance
        balance = await api.get_wallet_balance()
        print(f"Wallet Balance: {json.dumps(balance, indent=2)}")
        
        # Test positions
        positions = await api.get_positions()
        print(f"Positions: {json.dumps(positions, indent=2)}")
    
    asyncio.run(main())
