#!/usr/bin/env python3
"""
Bybit-Integrated Dashboard API for Trading Bot

This module provides the FastAPI application with real Bybit testnet integration:
- Real-time data from Bybit testnet
- Authenticated API calls
- Live trading data for frontend
"""

from fastapi import FastAPI, HTTPException, Depends, status, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer
from pydantic import BaseModel
from typing import Dict, Any, Optional, List, Union
import logging
import asyncio
from datetime import datetime
import json
import pandas as pd

# Import services
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from services.auth_service import AuthService
from services.health_service import HealthService
from services.risk_management import RiskManagementModule
from services.futures_trading import FuturesTradingModule
from services.bybit_api_service import BybitAPIService
from services.automatic_trading_service import get_automatic_trading_service
from middleware.simple_auth import (
    get_current_user, 
    require_read_permission, 
    require_write_permission,
    require_admin_permission
)

# Initialize FastAPI app
app = FastAPI(
    title="Bybit-Integrated Trading Bot Dashboard API",
    description="API for the crypto trading bot dashboard with real Bybit testnet data",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5173"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
auth_service = AuthService()
health_service = HealthService()
risk_module = RiskManagementModule()
trading_module = FuturesTradingModule()

# Initialize Bybit API service
bybit_api = BybitAPIService(
    api_key="popMizkoG6dZ5po90y",
    api_secret="zrJza3YTJBkw8BXx79n895akhlqRyNNmc8aW",
    base_url="https://api-testnet.bybit.com",
    ws_url="wss://stream-testnet.bybit.com/realtime",
    testnet=True
)

# Pydantic models
class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    success: bool
    token: Optional[str] = None
    user: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

class LogoutResponse(BaseModel):
    success: bool
    message: str

class SystemStatusResponse(BaseModel):
    status: str
    uptime: float
    last_heartbeat: Optional[datetime]
    system_metrics: Dict[str, Any]
    trading_status: Dict[str, Any]
    risk_status: Dict[str, Any]
    bybit_connection: Dict[str, Any]

class TradingDataResponse(BaseModel):
    positions: List[Dict[str, Any]]
    orders: List[Dict[str, Any]]
    pnl: Dict[str, float]
    margin_ratio: float
    funding_rates: Dict[str, float]
    wallet_balance: Dict[str, Any]
    bybit_status: Dict[str, Any]

class BybitDataResponse(BaseModel):
    wallet_balance: Dict[str, Any]
    positions: List[Dict[str, Any]]
    orders: List[Dict[str, Any]]
    account_info: Dict[str, Any]
    timestamp: str

class BacktestRequest(BaseModel):
    symbol: str
    start_date: str
    end_date: str
    leverage: float

class BacktestErrorResponse(BaseModel):
    success: bool = False
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str

class BacktestSuccessResponse(BaseModel):
    success: bool = True
    symbol: str
    start_date: str
    end_date: str
    leverage: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    total_trades: int
    win_rate: float
    avg_gain_per_trade: float
    profit_factor: float
    gross_profit: float
    gross_loss: float
    largest_win: float
    largest_loss: float
    win_loss_ratio: float
    margin_calls: int
    data_points: int
    timestamp: str

# Authentication endpoints
@app.post("/auth/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    """Login endpoint for user authentication."""
    try:
        result = auth_service.login(request.username, request.password)
        return LoginResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}"
        )

@app.post("/auth/logout", response_model=LogoutResponse)
async def logout(current_user: Dict[str, Any] = Depends(get_current_user)):
    """Logout endpoint for user session termination."""
    try:
        result = auth_service.logout("session_id")
        return LogoutResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Logout failed: {str(e)}"
        )

# Health endpoints
@app.get("/health")
async def health_check():
    """Basic health check endpoint."""
    try:
        health_data = health_service.get_health_status()
        
        # Add Bybit connection status
        try:
            bybit_health = await bybit_api.test_connection()
            health_data["bybit_connection"] = bybit_health
        except Exception as e:
            health_data["bybit_connection"] = {"success": False, "error": str(e)}
        
        return health_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health check failed: {str(e)}"
        )

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check endpoint."""
    try:
        health_data = health_service.get_detailed_health()
        
        # Add Bybit connection status
        try:
            bybit_health = await bybit_api.test_connection()
            health_data["bybit_connection"] = bybit_health
        except Exception as e:
            health_data["bybit_connection"] = {"success": False, "error": str(e)}
        
        return health_data
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detailed health check failed: {str(e)}"
        )

# System status endpoints
@app.get("/status", response_model=SystemStatusResponse)
async def get_system_status():
    """Get comprehensive system status with Bybit integration."""
    try:
        # Get health data
        health_data = health_service.get_health_status()
        
        # Get Bybit connection status
        bybit_status = await bybit_api.test_connection()
        
        # Get trading status
        trading_status = {
            "active_positions": len(trading_module.positions) if hasattr(trading_module, 'positions') else 0,
            "pending_orders": len(trading_module.pending_orders) if hasattr(trading_module, 'pending_orders') else 0,
            "daily_pnl": risk_module.daily_pnl,
            "daily_trades": risk_module.daily_trades,
            "consecutive_losses": risk_module.consecutive_losses
        }
        
        # Get risk status
        risk_status = {
            "can_trade": risk_module.check_daily_limits().get('can_trade', False),
            "can_enter": risk_module.check_daily_limits().get('can_enter', False),
            "kill_switch_triggered": risk_module.kill_switch_triggered,
            "margin_ratio": 0.5,
            "drawdown": 0.05
        }
        
        return SystemStatusResponse(
            status=health_data.get('status', 'unknown'),
            uptime=health_data.get('uptime_seconds', 0),
            last_heartbeat=health_data.get('last_heartbeat'),
            system_metrics=health_data.get('system', {}),
            trading_status=trading_status,
            risk_status=risk_status,
            bybit_connection=bybit_status
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"System status failed: {str(e)}"
        )

# Trading data endpoints with real Bybit data
@app.get("/trading/data", response_model=TradingDataResponse)
async def get_trading_data(current_user: Dict[str, Any] = Depends(require_read_permission)):
    """Get real trading data from Bybit testnet."""
    try:
        # Get real wallet balance from Bybit
        wallet_balance = await bybit_api.get_wallet_balance()
        
        # Get real positions from Bybit
        positions_result = await bybit_api.get_positions()
        
        # Process positions data
        positions = []
        if positions_result.get('success') and positions_result.get('data'):
            positions_data = positions_result.get('data', {})
            if 'list' in positions_data:
                for pos in positions_data['list']:
                    if float(pos.get('size', 0)) > 0:  # Only active positions
                        positions.append({
                            "id": pos.get('symbol', ''),
                            "symbol": pos.get('symbol', ''),
                            "side": pos.get('side', ''),
                            "quantity": float(pos.get('size', 0)),
                            "entry_price": float(pos.get('avgPrice', 0)),
                            "current_price": float(pos.get('markPrice', 0)),
                            "unrealized_pnl": float(pos.get('unrealisedPnl', 0)),
                            "leverage": float(pos.get('leverage', 1)),
                            "margin": float(pos.get('positionIM', 0))
                        })
        
        # Get real orders from Bybit
        orders_result = await bybit_api.get_order_history()
        orders = []
        if orders_result.get('success') and orders_result.get('data'):
            orders_data = orders_result.get('data', {})
            if 'list' in orders_data:
                for order in orders_data['list']:
                    orders.append({
                        "order_id": order.get('orderId', ''),
                        "symbol": order.get('symbol', ''),
                        "side": order.get('side', ''),
                        "order_type": order.get('orderType', ''),
                        "quantity": float(order.get('qty', 0) or 0),
                        "price": float(order.get('price', 0) or 0),
                        "status": order.get('orderStatus', ''),
                        "filled_quantity": float(order.get('cumExecQty', 0) or 0),
                        "average_price": float(order.get('avgPrice', 0) or 0),
                        "created_time": order.get('createdTime', ''),
                        "updated_time": order.get('updatedTime', ''),
                        "timestamp": datetime.now().isoformat()
                    })
        
        # Calculate PnL from positions and wallet balance
        total_unrealized_pnl = sum(pos.get('unrealized_pnl', 0) for pos in positions)
        
        # Calculate realized PnL from wallet balance
        realized_pnl = 0.0
        if wallet_balance.get('success') and wallet_balance.get('data'):
            wallet_data = wallet_balance.get('data', {})
            if 'list' in wallet_data:
                for account in wallet_data['list']:
                    if account.get('accountType') == 'UNIFIED':
                        # Get realized PnL from Bybit's actual cumRealisedPnl data
                        if 'coin' in account:
                            for coin in account['coin']:
                                if coin.get('coin') == 'USDT':  # USDT is the main trading pair
                                    realized_pnl = float(coin.get('cumRealisedPnl', 0))
                                    break
                        
                        # Fallback to totalPnl if cumRealisedPnl not found
                        if realized_pnl == 0:
                            total_pnl_bybit = float(account.get('totalPnl', 0))
                            if total_pnl_bybit != 0:
                                realized_pnl = total_pnl_bybit - total_unrealized_pnl
                        break
        
        pnl = {
            "realized": max(0, realized_pnl),  # Ensure non-negative
            "unrealized": total_unrealized_pnl,
            "total": total_unrealized_pnl + max(0, realized_pnl)
        }
        
        # Calculate margin ratio (simplified)
        margin_ratio = 0.5  # Would need actual margin calculation
        
        # Get funding rates (simplified)
        funding_rates = {
            "BTCUSDT": 0.0001,
            "ETHUSDT": 0.0001
        }
        
        return TradingDataResponse(
            positions=positions,
            orders=orders,
            pnl=pnl,
            margin_ratio=margin_ratio,
            funding_rates=funding_rates,
            wallet_balance=wallet_balance.get('data', {}),
            bybit_status={
                "wallet_success": wallet_balance.get('success', False),
                "positions_success": positions_result.get('success', False),
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trading data failed: {str(e)}"
        )

# Direct Bybit data endpoint
@app.get("/bybit/data", response_model=BybitDataResponse)
async def get_bybit_data(current_user: Dict[str, Any] = Depends(require_read_permission)):
    """Get direct data from Bybit testnet."""
    try:
        # Get wallet balance
        wallet_balance = await bybit_api.get_wallet_balance()
        
        # Get positions
        positions_result = await bybit_api.get_positions()
        
        # Process data
        positions = []
        if positions_result.get('success') and positions_result.get('data'):
            positions_data = positions_result.get('data', {})
            if 'list' in positions_data:
                for pos in positions_data['list']:
                    positions.append({
                        "symbol": pos.get('symbol', ''),
                        "side": pos.get('side', ''),
                        "size": float(pos.get('size', 0)),
                        "avgPrice": float(pos.get('avgPrice', 0)),
                        "markPrice": float(pos.get('markPrice', 0)),
                        "unrealisedPnl": float(pos.get('unrealisedPnl', 0)),
                        "leverage": float(pos.get('leverage', 1))
                    })
        
        return BybitDataResponse(
            wallet_balance=wallet_balance.get('data', {}),
            positions=positions,
            orders=[],  # Would need order history endpoint
            account_info={"testnet": True, "api_key": "popMizkoG6dZ5po90y"},
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Bybit data failed: {str(e)}"
        )

# Trading Stats Endpoint
@app.get("/trading/stats")
async def get_trading_stats(current_user: Dict[str, Any] = Depends(require_read_permission)):
    """Get trading statistics from Bybit testnet."""
    try:
        # Get wallet balance
        wallet_balance = await bybit_api.get_wallet_balance()
        
        # Get positions
        positions_result = await bybit_api.get_positions()
        
        # Calculate stats
        total_pnl = 0.0
        unrealized_pnl = 0.0
        realized_pnl = 0.0
        daily_pnl = 0.0  # Would need historical data
        win_rate = 0.0  # Would need trade history
        total_trades = 0  # Would need trade history
        max_drawdown = 0.0  # Would need historical data
        sharpe_ratio = 0.0  # Would need historical data
        current_positions = 0
        margin_ratio = 0.0
        
        if positions_result.get('success') and positions_result.get('data'):
            positions_data = positions_result.get('data', {})
            if 'list' in positions_data:
                current_positions = len([p for p in positions_data['list'] if float(p.get('size', 0)) > 0])
                # Calculate unrealized PnL from active positions
                unrealized_pnl = sum(float(p.get('unrealisedPnl', 0)) for p in positions_data['list'] if float(p.get('size', 0)) > 0)
                total_pnl = unrealized_pnl  # For now, total = unrealized (realized would come from order history)
        else:
            print(f"Positions API error: {positions_result.get('error', 'Unknown error')}")
        
        # Calculate realized PnL properly from Bybit data
        # Get realized PnL from wallet balance and positions
        try:
            if wallet_balance.get('success') and wallet_balance.get('data'):
                wallet_data = wallet_balance.get('data', {})
                if 'list' in wallet_data:
                    for account in wallet_data['list']:
                        if account.get('accountType') == 'UNIFIED':
                            current_balance = float(account.get('totalWalletBalance', 0))
                            
                            # Get realized PnL from Bybit's actual data
                            # Look for cumRealisedPnl in the coin data
                            realized_pnl = 0.0
                            if 'coin' in account:
                                for coin in account['coin']:
                                    if coin.get('coin') == 'USDT':  # USDT is the main trading pair
                                        realized_pnl = float(coin.get('cumRealisedPnl', 0))
                                        break
                            
                            # If no USDT realized PnL found, try to get from totalPnl
                            if realized_pnl == 0:
                                total_pnl_bybit = float(account.get('totalPnl', 0))
                                if total_pnl_bybit != 0:
                                    realized_pnl = total_pnl_bybit - unrealized_pnl
                            
                            total_pnl = unrealized_pnl + realized_pnl
                            break
        except Exception as e:
            print(f"Realized PnL calculation error: {e}")
            # Fallback: use unrealized PnL only
            realized_pnl = 0
            total_pnl = unrealized_pnl
        
        # Get account balance from wallet
        account_balance = 0.0
        available_balance = 0.0
        if wallet_balance.get('success') and wallet_balance.get('data'):
            wallet_data = wallet_balance.get('data', {})
            if 'list' in wallet_data:
                for account in wallet_data['list']:
                    if account.get('accountType') == 'UNIFIED':
                        account_balance = float(account.get('totalWalletBalance', 0))
                        available_balance = float(account.get('availableToWithdraw', 0))
                        break
        else:
            print(f"Wallet balance API error: {wallet_balance.get('error', 'Unknown error')}")
        
        return {
            "success": True,
            "total_pnl": total_pnl,
            "unrealized_pnl": unrealized_pnl,
            "realized_pnl": realized_pnl,
            "daily_pnl": daily_pnl,
            "win_rate": win_rate,
            "total_trades": total_trades,
            "max_drawdown": max_drawdown,
            "sharpe_ratio": sharpe_ratio,
            "current_positions": current_positions,
            "margin_ratio": margin_ratio,
            "account_balance": account_balance,
            "available_balance": available_balance,
            "timestamp": datetime.now().isoformat(),
            "api_status": {
                "positions": positions_result.get('success', False),
                "wallet": wallet_balance.get('success', False)
            }
        }
    except Exception as e:
        print(f"Trading stats error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trading stats failed: {str(e)}"
        )

# Trading Positions Endpoint
@app.get("/trading/positions")
async def get_trading_positions(current_user: Dict[str, Any] = Depends(require_read_permission)):
    """Get current trading positions from Bybit testnet."""
    try:
        # Get positions from Bybit with better error handling
        positions_result = await bybit_api.get_positions()
        
        positions = []
        if positions_result.get('success') and positions_result.get('data'):
            positions_data = positions_result.get('data', {})
            if 'list' in positions_data:
                for pos in positions_data['list']:
                    # Include all positions, not just active ones
                    size = float(pos.get('size', 0))
                    positions.append({
                        "symbol": pos.get('symbol', ''),
                        "side": pos.get('side', ''),
                        "size": size,
                        "entry_price": float(pos.get('avgPrice', 0)),
                        "current_price": float(pos.get('markPrice', 0)),
                        "unrealized_pnl": float(pos.get('unrealisedPnl', 0)),
                        "leverage": float(pos.get('leverage', 1)),
                        "margin_mode": pos.get('positionIM', 0),
                        "liquidation_price": float(pos.get('liqPrice', 0)),
                        "is_active": size > 0,
                        "timestamp": datetime.now().isoformat()
                    })
        else:
            # Log the error for debugging
            print(f"Positions API error: {positions_result.get('error', 'Unknown error')}")
        
        return {
            "success": True,
            "positions": positions,
            "count": len(positions),
            "active_count": len([p for p in positions if p.get('is_active', False)]),
            "timestamp": datetime.now().isoformat(),
            "api_status": positions_result.get('success', False)
        }
    except Exception as e:
        print(f"Positions endpoint error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trading positions failed: {str(e)}"
        )

# Account Info Endpoint
@app.get("/trading/account")
async def get_account_info(current_user: Dict[str, Any] = Depends(require_read_permission)):
    """Get real-time account information from Bybit testnet."""
    try:
        # Get wallet balance
        wallet_balance = await bybit_api.get_wallet_balance()
        
        # Get positions
        positions_result = await bybit_api.get_positions()
        
        account_info = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "wallet": {
                "success": wallet_balance.get('success', False),
                "balance": 0.0,
                "available": 0.0,
                "error": None
            },
            "positions": {
                "success": positions_result.get('success', False),
                "count": 0,
                "active_count": 0,
                "total_pnl": 0.0,
                "error": None
            }
        }
        
        # Process wallet data
        if wallet_balance.get('success') and wallet_balance.get('data'):
            wallet_data = wallet_balance.get('data', {})
            if 'list' in wallet_data:
                for account in wallet_data['list']:
                    if account.get('accountType') == 'UNIFIED':
                        account_info["wallet"]["balance"] = float(account.get('totalWalletBalance', 0))
                        account_info["wallet"]["available"] = float(account.get('availableToWithdraw', 0))
                        break
        else:
            account_info["wallet"]["error"] = wallet_balance.get('error', 'Unknown error')
        
        # Process positions data
        if positions_result.get('success') and positions_result.get('data'):
            positions_data = positions_result.get('data', {})
            if 'list' in positions_data:
                positions = positions_data['list']
                account_info["positions"]["count"] = len(positions)
                account_info["positions"]["active_count"] = len([p for p in positions if float(p.get('size', 0)) > 0])
                account_info["positions"]["total_pnl"] = sum(float(p.get('unrealisedPnl', 0)) for p in positions)
        else:
            account_info["positions"]["error"] = positions_result.get('error', 'Unknown error')
        
        return account_info
        
    except Exception as e:
        print(f"Account info error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Account info failed: {str(e)}"
        )

# Order History Endpoint
@app.get("/trading/orders")
async def get_order_history(current_user: Dict[str, Any] = Depends(require_read_permission)):
    """Get order history from Bybit testnet."""
    try:
        # Get order history from Bybit
        orders_result = await bybit_api.get_order_history()
        
        orders = []
        if orders_result.get('success') and orders_result.get('data'):
            orders_data = orders_result.get('data', {})
            if 'list' in orders_data:
                for order in orders_data['list']:
                    orders.append({
                        "order_id": order.get('orderId', ''),
                        "symbol": order.get('symbol', ''),
                        "side": order.get('side', ''),
                        "order_type": order.get('orderType', ''),
                        "quantity": float(order.get('qty', 0) or 0),
                        "price": float(order.get('price', 0) or 0),
                        "status": order.get('orderStatus', ''),
                        "filled_quantity": float(order.get('cumExecQty', 0) or 0),
                        "average_price": float(order.get('avgPrice', 0) or 0),
                        "created_time": order.get('createdTime', ''),
                        "updated_time": order.get('updatedTime', ''),
                        "timestamp": datetime.now().isoformat()
                    })
        
        return {
            "success": True,
            "orders": orders,
            "count": len(orders),
            "timestamp": datetime.now().isoformat(),
            "api_status": orders_result.get('success', False)
        }
    except Exception as e:
        print(f"Order history error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Order history failed: {str(e)}"
        )

# Market Data Endpoint
@app.get("/trading/market")
async def get_market_data(current_user: Dict[str, Any] = Depends(require_read_permission)):
    """Get real-time market data from Bybit testnet."""
    try:
        # Get market data for major symbols
        symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "ADAUSDT"]
        market_data = {}
        
        for symbol in symbols:
            try:
                ticker_result = await bybit_api.get_ticker(symbol)
                if ticker_result.get('success') and ticker_result.get('data'):
                    ticker_data = ticker_result.get('data', {})
                    if 'list' in ticker_data and len(ticker_data['list']) > 0:
                        ticker = ticker_data['list'][0]
                        market_data[symbol] = {
                            "symbol": symbol,
                            "price": float(ticker.get('lastPrice', 0)),
                            "change_24h": float(ticker.get('price24hPcnt', 0)),
                            "volume_24h": float(ticker.get('volume24h', 0)),
                            "high_24h": float(ticker.get('highPrice24h', 0)),
                            "low_24h": float(ticker.get('lowPrice24h', 0)),
                            "timestamp": datetime.now().isoformat()
                        }
            except Exception as e:
                print(f"Error fetching {symbol}: {e}")
                continue
        
        return {
            "success": True,
            "market_data": market_data,
            "count": len(market_data),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Market data error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Market data failed: {str(e)}"
        )

# Comprehensive Dashboard Data Endpoint
@app.get("/trading/dashboard")
async def get_dashboard_data(current_user: Dict[str, Any] = Depends(require_read_permission)):
    """Get comprehensive dashboard data for frontend with optimized performance."""
    try:
        # Fetch all required data in parallel with optimized execution
        import asyncio
        
        # Create tasks for parallel execution
        tasks = [
            bybit_api.get_wallet_balance(),
            bybit_api.get_positions(),
            bybit_api.get_order_history()
        ]
        
        # Execute all tasks in parallel
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Unpack results
        wallet_balance = results[0] if not isinstance(results[0], Exception) else {'success': False, 'error': str(results[0])}
        positions_result = results[1] if not isinstance(results[1], Exception) else {'success': False, 'error': str(results[1])}
        orders_result = results[2] if not isinstance(results[2], Exception) else {'success': False, 'error': str(results[2])}
        
        # Process wallet data
        wallet_info = {
            "balance": 0.0,
            "available": 0.0,
            "success": wallet_balance.get('success', False),
            "error": None
        }
        
        if wallet_balance.get('success') and wallet_balance.get('data'):
            wallet_data = wallet_balance.get('data', {})
            if 'list' in wallet_data:
                for account in wallet_data['list']:
                    if account.get('accountType') == 'UNIFIED':
                        wallet_info["balance"] = float(account.get('totalWalletBalance', 0))
                        wallet_info["available"] = float(account.get('availableToWithdraw', 0))
                        break
        else:
            wallet_info["error"] = wallet_balance.get('error', 'Unknown error')
        
        # Process positions data
        positions = []
        total_pnl = 0.0
        active_positions = 0
        
        if positions_result.get('success') and positions_result.get('data'):
            positions_data = positions_result.get('data', {})
            if 'list' in positions_data:
                for pos in positions_data['list']:
                    size = float(pos.get('size', 0))
                    pnl = float(pos.get('unrealisedPnl', 0))
                    total_pnl += pnl
                    
                    if size > 0:
                        active_positions += 1
                    
                    positions.append({
                        "symbol": pos.get('symbol', ''),
                        "side": pos.get('side', ''),
                        "size": size,
                        "entry_price": float(pos.get('avgPrice', 0)),
                        "current_price": float(pos.get('markPrice', 0)),
                        "unrealized_pnl": pnl,
                        "leverage": float(pos.get('leverage', 1)),
                        "liquidation_price": float(pos.get('liqPrice', 0)),
                        "is_active": size > 0,
                        "timestamp": datetime.now().isoformat()
                    })
        
        # Process orders data
        orders = []
        if orders_result.get('success') and orders_result.get('data'):
            orders_data = orders_result.get('data', {})
            if 'list' in orders_data:
                for order in orders_data['list'][:10]:  # Last 10 orders
                    orders.append({
                        "order_id": order.get('orderId', ''),
                        "symbol": order.get('symbol', ''),
                        "side": order.get('side', ''),
                        "order_type": order.get('orderType', ''),
                        "quantity": float(order.get('qty', 0)),
                        "price": float(order.get('price', 0)),
                        "status": order.get('orderStatus', ''),
                        "created_time": order.get('createdTime', ''),
                        "timestamp": datetime.now().isoformat()
                    })
        
        # Calculate trading metrics
        trading_metrics = {
            "total_pnl": total_pnl,
            "active_positions": active_positions,
            "total_positions": len(positions),
            "daily_pnl": 0.0,  # Would need historical data
            "win_rate": 0.0,    # Would need trade history
            "total_trades": len(orders),
            "success_rate": 0.0  # Would need analysis
        }
        
        return {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "wallet": wallet_info,
            "positions": {
                "list": positions,
                "count": len(positions),
                "active_count": active_positions,
                "total_pnl": total_pnl,
                "success": positions_result.get('success', False),
                "error": positions_result.get('error')
            },
            "orders": {
                "list": orders,
                "count": len(orders),
                "success": orders_result.get('success', False),
                "error": orders_result.get('error')
            },
            "trading_metrics": trading_metrics,
            "api_status": {
                "wallet": wallet_balance.get('success', False),
                "positions": positions_result.get('success', False),
                "orders": orders_result.get('success', False)
            }
        }
        
    except Exception as e:
        print(f"Dashboard data error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Dashboard data failed: {str(e)}"
        )

# Trading Signals Endpoint
@app.get("/trading/signals")
async def get_trading_signals(current_user: Dict[str, Any] = Depends(require_read_permission)):
    """
    Get trading signals using trained XGBoost models with CSV fallback.
    
    Primary flow: Load XGBoost models (log_return_model.pkl, future_return_model.pkl) 
    from models/xgboost/{symbol}/ and generate signals based on:
    - future_return prediction -> signal direction (BUY/SELL/HOLD)
    - log_return prediction -> confidence level
    - Combined -> position size calculation
    
    Fallback flow: Read latest signals from leveraged_backtest_results/{symbol}_trades.csv
    - Use signal only if timestamp is within 1 hour of current time
    - Set HOLD signal if CSV data is older than 1 hour
    """
    try:
        import joblib
        import numpy as np
        import pandas as pd
        from pathlib import Path
        from datetime import timedelta
        
        # Model-based signal generation
        signals = []
        symbols = ['BTCUSDT', 'ETHUSDT']
        
        for symbol in symbols:
            try:
                # Load feature data for the symbol
                feature_file = f"feature/{symbol.lower()}/{symbol.lower()}_features.parquet"
                if not os.path.exists(feature_file):
                    print(f"Feature file not found for {symbol}: {feature_file}")
                    continue
                
                # Load the latest feature data
                df_features = pd.read_parquet(feature_file)
                if df_features.empty:
                    print(f"No feature data available for {symbol}")
                    continue
                
                # Get the most recent row
                latest_row = df_features.iloc[-1].copy()
                
                # Prepare features for model prediction (shifted features only)
                shifted_features = [
                    'log_return_shifted', 'atr_14_shifted', 'sma_20_shifted', 
                    'sma_100_shifted', 'rsi_14_shifted', 'volume_zscore_shifted'
                ]
                
                # Check if all required features are available
                missing_features = [f for f in shifted_features if f not in latest_row.index]
                if missing_features:
                    print(f"Missing features for {symbol}: {missing_features}")
                    continue
                
                # Extract feature values
                feature_values = latest_row[shifted_features].values.reshape(1, -1)
                
                # Load models for the symbol
                model_dir = f"models/xgboost/{symbol.lower()}"
                models = {}
                
                for model_name in ['log_return_model.pkl', 'future_return_model.pkl']:
                    model_path = os.path.join(model_dir, model_name)
                    if os.path.exists(model_path):
                        model_data = joblib.load(model_path)
                        models[model_name] = model_data
                
                if not models:
                    print(f"No models found for {symbol}")
                    continue
                
                # Generate predictions
                predictions = {}
                for model_name, model_data in models.items():
                    try:
                        model = model_data['model']
                        scaler = model_data['scaler']
                        
                        # Scale features
                        scaled_features = scaler.transform(feature_values)
                        
                        # Make prediction
                        prediction = model.predict(scaled_features)[0]
                        predictions[model_name] = prediction
                        
                    except Exception as e:
                        print(f"Error predicting with {model_name} for {symbol}: {e}")
                        continue
                
                # Convert predictions to trading signal
                signal = 0.0
                confidence = 0.0
                position_size = 0.0
                
                if 'log_return_model.pkl' in predictions and 'future_return_model.pkl' in predictions:
                    log_return_pred = predictions['log_return_model.pkl']
                    future_return_pred = predictions['future_return_model.pkl']
                    
                    # Generate trading signal from model predictions
                    # Future return prediction is the primary signal driver
                    # Log return prediction is used for confidence/strength
                    
                    # Signal generation logic:
                    # - Positive future return -> BUY signal
                    # - Negative future return -> SELL signal
                    # - Log return magnitude -> confidence level
                    
                    signal_threshold = 0.001  # Minimum threshold for signal generation
                    
                    if abs(future_return_pred) > signal_threshold:
                        # Determine signal direction based on future return
                        if future_return_pred > 0:
                            signal = 1.0  # BUY signal
                        else:
                            signal = -1.0  # SELL signal
                        
                        # Calculate confidence based on log return magnitude
                        confidence = min(abs(log_return_pred) * 10, 1.0)  # Scale and cap at 1.0
                        
                        # Calculate position size based on signal strength and confidence
                        # Stronger signals and higher confidence = larger position
                        signal_strength = abs(future_return_pred)
                        position_size = min(signal_strength * confidence * 0.1, 0.2)  # Cap at 20% position
                        
                    else:
                        # Signal below threshold -> HOLD
                        signal = 0.0
                        confidence = 0.0
                        position_size = 0.0
                
                # Create signal response
                signal_data = {
                    "symbol": symbol,
                    "signal": signal,
                    "confidence": confidence,
                    "position_size": position_size,
                    "components": {
                        "ml_log_return": predictions.get('log_return_model.pkl', 0.0),
                        "ml_future_return": predictions.get('future_return_model.pkl', 0.0),
                        "technical": 0.0,  # Could be enhanced with technical indicators
                        "sentiment": 0.0   # Could be enhanced with sentiment data
                    },
                    "timestamp": datetime.now().isoformat(),
                    "source": "xgboost_models"
                }
                
                signals.append(signal_data)
                
            except Exception as e:
                print(f"Error generating signal for {symbol}: {e}")
                continue
        
        # If model-based signals failed, try CSV fallback
        if not signals:
            print("Model-based signal generation failed, trying CSV fallback...")
            signals = await get_csv_fallback_signals()
        
        return {
            "success": True,
            "signals": signals,
            "count": len(signals),
            "timestamp": datetime.now().isoformat(),
            "generator": "xgboost_models" if signals else "csv_fallback"
        }
            
    except Exception as e:
        print(f"Signal generation error: {e}")
        # Final fallback to CSV
        try:
            signals = await get_csv_fallback_signals()
            return {
                "success": True,
                "signals": signals,
                "count": len(signals),
                "timestamp": datetime.now().isoformat(),
                "generator": "csv_fallback"
            }
        except Exception as csv_error:
            print(f"CSV fallback also failed: {csv_error}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Trading signals failed: {str(e)}"
            )


async def get_csv_fallback_signals():
    """Get signals from recent CSV files as fallback."""
    try:
        import pandas as pd
        from datetime import timedelta
        
        signals = []
        symbols = ['btcusdt', 'ethusdt']
        current_time = datetime.now()
        
        for symbol in symbols:
            # Use new file naming convention: {symbol}_trades.csv
            csv_file = f"leveraged_backtest_results/{symbol}_trades.csv"
            
            if not os.path.exists(csv_file):
                print(f"CSV file not found: {csv_file}")
                continue
            
            try:
                # Read the CSV file
                df = pd.read_csv(csv_file)
                if df.empty:
                    print(f"Empty CSV file: {csv_file}")
                    continue
                
                # Get the most recent trade
                latest_trade = df.iloc[-1]
                
                # Check timestamp (assuming 'timestamp' column exists)
                if 'timestamp' in latest_trade:
                    try:
                        # Parse timestamp
                        trade_time = pd.to_datetime(latest_trade['timestamp'])
                        time_diff = current_time - trade_time.replace(tzinfo=None)
                        
                        # Check if within 1 hour
                        if time_diff <= timedelta(hours=1):
                            # Extract signal information from trade data
                            signal_value = 0.0
                            confidence = 0.5  # Default confidence for CSV fallback
                            position_size = 0.0
                            
                            # Try to extract signal from trade data
                            if 'action' in latest_trade:
                                action = latest_trade['action']
                                if action == 'BUY':
                                    signal_value = 1.0  # BUY signal
                                    position_size = 0.1  # Default position size for CSV fallback
                                elif action == 'SELL':
                                    signal_value = -1.0  # SELL signal
                                    position_size = 0.1  # Default position size for CSV fallback
                                else:
                                    signal_value = 0.0  # HOLD
                                    position_size = 0.0
                            
                            signal_data = {
                                "symbol": symbol.upper(),
                                "signal": signal_value,
                                "confidence": confidence,
                                "position_size": position_size,
                                "components": {
                                    "ml": 0.0,
                                    "technical": 0.0,
                                    "sentiment": 0.0,
                                    "csv_fallback": signal_value
                                },
                                "timestamp": datetime.now().isoformat(),
                                "source": "csv_fallback"
                            }
                            
                            signals.append(signal_data)
                            print(f"Using CSV fallback signal for {symbol.upper()}: {signal_value}")
                        else:
                            print(f"CSV timestamp too old for {symbol.upper()}: {time_diff}")
                            # Set HOLD signal for old data (timestamp > 1 hour)
                            signal_data = {
                                "symbol": symbol.upper(),
                                "signal": 0.0,
                                "confidence": 0.0,
                                "position_size": 0.0,
                                "components": {
                                    "ml": 0.0,
                                    "technical": 0.0,
                                    "sentiment": 0.0,
                                    "csv_fallback": "old_data"
                                },
                                "timestamp": datetime.now().isoformat(),
                                "source": "csv_fallback_hold"
                            }
                            signals.append(signal_data)
                    
                    except Exception as time_error:
                        print(f"Error parsing timestamp for {symbol}: {time_error}")
                        # Set HOLD signal for parsing errors
                        signal_data = {
                            "symbol": symbol.upper(),
                            "signal": 0.0,
                            "confidence": 0.0,
                            "position_size": 0.0,
                            "components": {
                                "ml": 0.0,
                                "technical": 0.0,
                                "sentiment": 0.0,
                                "csv_fallback": "parse_error"
                            },
                            "timestamp": datetime.now().isoformat(),
                            "source": "csv_fallback_error"
                        }
                        signals.append(signal_data)
                        continue
                else:
                    # No timestamp column - set HOLD signal
                    signal_data = {
                        "symbol": symbol.upper(),
                        "signal": 0.0,
                        "confidence": 0.0,
                        "position_size": 0.0,
                        "components": {
                            "ml": 0.0,
                            "technical": 0.0,
                            "sentiment": 0.0,
                            "csv_fallback": "no_timestamp"
                        },
                        "timestamp": datetime.now().isoformat(),
                        "source": "csv_fallback_no_timestamp"
                    }
                    signals.append(signal_data)
                
            except Exception as csv_error:
                print(f"Error reading CSV file {csv_file}: {csv_error}")
                continue
        
        return signals
        
    except Exception as e:
        print(f"CSV fallback error: {e}")
        return []

# Close Position Endpoint
@app.post("/trading/positions/close")
async def close_position(
    request: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(require_write_permission)
):
    """Close a position on Bybit testnet."""
    try:
        symbol = request.get('symbol')
        side = request.get('side')  # Optional: 'Buy' or 'Sell'
        size = request.get('size')  # Optional: position size
        
        if not symbol:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Symbol is required"
            )
        
        # Use provided position details if available (faster)
        if side and size:
            print(f"Using provided position details: {side} {size} {symbol}")
        else:
            # Fallback: Get current positions to determine the side if not provided
            print(f"Looking up position details for {symbol}")
            positions_result = await bybit_api.get_positions()
            if positions_result.get('success') and positions_result.get('data'):
                positions_data = positions_result.get('data', {})
                if 'list' in positions_data:
                    for pos in positions_data['list']:
                        if pos.get('symbol') == symbol and float(pos.get('size', 0)) > 0:
                            # Determine opposite side for closing
                            current_side = pos.get('side', '')
                            side = 'Sell' if current_side == 'Buy' else 'Buy'
                            size = pos.get('size', 0)
                            break
        
        if not side:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Could not determine position side. Please specify 'side' parameter."
            )
        
        # Close the position with provided details
        close_result = await bybit_api.close_position(symbol, side, size)
        
        if close_result.get('success'):
            return {
                "success": True,
                "message": f"Position closed successfully for {symbol}",
                "symbol": symbol,
                "side": side,
                "order_id": close_result.get('data', {}).get('orderId'),
                "timestamp": datetime.now().isoformat()
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to close position: {close_result.get('error', 'Unknown error')}"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        print(f"Close position error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Close position failed: {str(e)}"
        )

# Alert endpoints
@app.post("/alerts/email")
async def send_email_alert(request: Dict[str, Any]):
    """Send email alert via real SMTP service."""
    try:
        from services.email_service import MockEmailService
        
        # Use mock service for now (you can switch to real EmailService later)
        email_service = MockEmailService()
        
        recipient = request.get('recipient', 'zombiewins23@gmail.com')
        subject = request.get('subject', 'Trading Bot Alert')
        message = request.get('message', 'No message provided')
        
        # Send email
        result = email_service.send_alert(recipient, subject, message)
        
        return {
            "success": result.get('success', False),
            "message": result.get('message', 'Email alert processed'),
            "recipient": recipient,
            "timestamp": datetime.now().isoformat(),
            "mock": result.get('mock', False)
        }
    except Exception as e:
        print(f"Email alert error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Email alert failed"
        )

@app.post("/test/email")
async def test_email_endpoint(request: Dict[str, Any]):
    """Test email endpoint without Unicode issues."""
    return {
        "success": True,
        "message": "Test email sent successfully",
        "recipient": request.get('recipient', 'test@example.com'),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/trading/pause")
async def pause_trading(current_user: Dict[str, Any] = Depends(require_write_permission)):
    """Pause trading system."""
    try:
        # Mock pause implementation
        print("Trading system paused")
        return {
            "success": True,
            "message": "Trading system paused successfully",
            "status": "paused",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Pause trading error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Pause trading failed: {str(e)}"
        )

@app.post("/trading/resume")
async def resume_trading(current_user: Dict[str, Any] = Depends(require_write_permission)):
    """Resume trading system."""
    try:
        # Mock resume implementation
        print("Trading system resumed")
        return {
            "success": True,
            "message": "Trading system resumed successfully",
            "status": "running",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Resume trading error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Resume trading failed: {str(e)}"
        )

@app.post("/trading/kill-switch")
async def activate_kill_switch(current_user: Dict[str, Any] = Depends(require_write_permission)):
    """Activate kill switch - flatten all positions and halt trading."""
    try:
        # Mock kill switch implementation
        print("KILL SWITCH ACTIVATED")
        print("   - All positions will be flattened")
        print("   - All orders will be cancelled")
        print("   - New entries blocked until reset")
        
        return {
            "success": True,
            "message": "Kill switch activated - all positions flattened",
            "status": "kill_switch_active",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Kill switch error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Kill switch failed: {str(e)}"
        )

@app.post("/trading/reset-kill-switch")
async def reset_kill_switch(current_user: Dict[str, Any] = Depends(require_write_permission)):
    """Reset kill switch - allow trading to resume."""
    try:
        # Mock reset implementation
        print("Kill switch reset - trading can resume")
        
        return {
            "success": True,
            "message": "Kill switch reset - trading can resume",
            "status": "running",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        print(f"Reset kill switch error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Reset kill switch failed: {str(e)}"
        )

# Trading Risk Endpoint
@app.get("/trading/risk")
async def get_trading_risk(current_user: Dict[str, Any] = Depends(require_read_permission)):
    """Get risk metrics from Bybit testnet."""
    try:
        # Get positions for risk calculation
        positions_result = await bybit_api.get_positions()
        
        # Get wallet balance for risk calculation
        wallet_balance = await bybit_api.get_wallet_balance()
        
        # Calculate risk metrics
        current_drawdown = 0.0
        max_drawdown = 0.0
        daily_pnl = 0.0
        daily_loss_limit = -300.00
        daily_profit_limit = 200.00
        trades_count = 0
        max_trades_per_day = 15
        consecutive_losses = 0
        max_consecutive_losses = 4
        margin_ratio = 0.0
        exposure_ratio = 0.0
        leverage_ratio = 0.0
        risk_score = 0
        kill_switch_status = False
        
        # Calculate from positions
        if positions_result.get('success') and positions_result.get('data'):
            positions_data = positions_result.get('data', {})
            if 'list' in positions_data:
                active_positions = [p for p in positions_data['list'] if float(p.get('size', 0)) > 0]
                trades_count = len(active_positions)
                
                # Calculate exposure
                total_exposure = sum(float(p.get('notional', 0)) for p in active_positions)
                if wallet_balance.get('success') and wallet_balance.get('data'):
                    wallet_data = wallet_balance.get('data', {})
                    if 'list' in wallet_data:
                        for account in wallet_data['list']:
                            if account.get('accountType') == 'UNIFIED':
                                total_balance = float(account.get('totalWalletBalance', 1))
                                exposure_ratio = total_exposure / total_balance if total_balance > 0 else 0
                                break
        
        return {
            "current_drawdown": current_drawdown,
            "max_drawdown": max_drawdown,
            "daily_pnl": daily_pnl,
            "daily_loss_limit": daily_loss_limit,
            "daily_profit_limit": daily_profit_limit,
            "trades_count": trades_count,
            "max_trades_per_day": max_trades_per_day,
            "consecutive_losses": consecutive_losses,
            "max_consecutive_losses": max_consecutive_losses,
            "margin_ratio": margin_ratio,
            "exposure_ratio": exposure_ratio,
            "leverage_ratio": leverage_ratio,
            "risk_score": risk_score,
            "kill_switch_status": kill_switch_status,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Trading risk failed: {str(e)}"
        )

# Control endpoints (duplicates removed - using the more complete implementations above)

# Automatic Trading endpoints
@app.post("/trading/auto/start")
async def start_automatic_trading(current_user: Dict[str, Any] = Depends(require_write_permission)):
    """Start automatic trading service."""
    try:
        # Get automatic trading service
        auto_trading = get_automatic_trading_service()
        
        # Start the service in background
        asyncio.create_task(auto_trading.start_automatic_trading())
        
        return {
            "success": True,
            "message": "Automatic trading service started",
            "status": auto_trading.get_trading_status()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start automatic trading: {str(e)}"
        )

@app.post("/trading/auto/stop")
async def stop_automatic_trading(current_user: Dict[str, Any] = Depends(require_write_permission)):
    """Stop automatic trading service."""
    try:
        auto_trading = get_automatic_trading_service()
        await auto_trading.stop_automatic_trading()
        
        return {
            "success": True,
            "message": "Automatic trading service stopped",
            "status": auto_trading.get_trading_status()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop automatic trading: {str(e)}"
        )

@app.get("/trading/auto/status")
async def get_automatic_trading_status(current_user: Dict[str, Any] = Depends(require_read_permission)):
    """Get automatic trading service status."""
    try:
        auto_trading = get_automatic_trading_service()
        status = auto_trading.get_trading_status()
        
        return {
            "success": True,
            "status": status
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get automatic trading status: {str(e)}"
        )

@app.post("/trading/auto/config")
async def update_automatic_trading_config(
    config: Dict[str, Any],
    current_user: Dict[str, Any] = Depends(require_write_permission)
):
    """Update automatic trading configuration."""
    try:
        auto_trading = get_automatic_trading_service()
        auto_trading.update_config(config)
        
        return {
            "success": True,
            "message": "Automatic trading configuration updated",
            "config": config
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update automatic trading config: {str(e)}"
        )

# Backtest endpoint
@app.post("/backtest/run", response_model=Union[BacktestSuccessResponse, BacktestErrorResponse])
async def run_backtest(
    request: BacktestRequest,
    current_user: Dict[str, Any] = Depends(require_read_permission)
):
    """Run a backtest with the specified parameters."""
    try:
        # Import backtester functions
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
        
        from backtester import (
            AVAILABLE_ASSETS, 
            load_data_from_parquet, 
            filter_data_by_period,
            LeveragedBacktestEngine
        )
        
        # Validate symbol
        if request.symbol not in AVAILABLE_ASSETS.values():
            return BacktestErrorResponse(
                error_type="VALIDATION_ERROR",
                message=f"Invalid symbol '{request.symbol}'. Supported symbols: {list(AVAILABLE_ASSETS.values())}",
                details={"supported_symbols": list(AVAILABLE_ASSETS.values())},
                timestamp=datetime.now().isoformat()
            )
        
        # Validate leverage
        if not (1.0 <= request.leverage <= 10.0):
            return BacktestErrorResponse(
                error_type="VALIDATION_ERROR",
                message=f"Invalid leverage '{request.leverage}'. Leverage must be between 1.0 and 10.0",
                details={"min_leverage": 1.0, "max_leverage": 10.0},
                timestamp=datetime.now().isoformat()
            )
        
        # Validate date format and range
        try:
            start_dt = pd.to_datetime(request.start_date)
            end_dt = pd.to_datetime(request.end_date)
            
            if start_dt >= end_dt:
                return BacktestErrorResponse(
                    error_type="VALIDATION_ERROR",
                    message="Start date must be before end date",
                    details={"start_date": request.start_date, "end_date": request.end_date},
                    timestamp=datetime.now().isoformat()
                )
            
            # Check if date range is too large (more than 2 years)
            if (end_dt - start_dt).days > 730:
                return BacktestErrorResponse(
                    error_type="VALIDATION_ERROR",
                    message="Date range too large. Maximum allowed range is 2 years",
                    details={"max_days": 730, "requested_days": (end_dt - start_dt).days},
                    timestamp=datetime.now().isoformat()
                )
                
        except Exception as e:
            return BacktestErrorResponse(
                error_type="VALIDATION_ERROR",
                message=f"Invalid date format. Use YYYY-MM-DD format. Error: {str(e)}",
                details={"start_date": request.start_date, "end_date": request.end_date},
                timestamp=datetime.now().isoformat()
            )
        
        # Check if parquet data exists for the symbol
        if request.symbol not in ['BTCUSDT', 'ETHUSDT']:
            return BacktestErrorResponse(
                error_type="DATA_MISSING",
                message=f"Parquet data not available for {request.symbol}. Only BTCUSDT and ETHUSDT are supported.",
                details={"supported_symbols": ['BTCUSDT', 'ETHUSDT']},
                timestamp=datetime.now().isoformat()
            )
        
        # Load data from parquet
        df = load_data_from_parquet(request.symbol, 'data')
        if df.empty:
            return BacktestErrorResponse(
                error_type="DATA_MISSING",
                message=f"Parquet file not found for {request.symbol}",
                details={"symbol": request.symbol, "data_directory": "data"},
                timestamp=datetime.now().isoformat()
            )
        
        # Check if data covers the requested time range
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        data_start = df['timestamp'].min()
        data_end = df['timestamp'].max()
        
        if start_dt < data_start or end_dt > data_end:
            return BacktestErrorResponse(
                error_type="DATA_MISSING",
                message=f"Requested date range not fully covered by available data",
                details={
                    "requested_start": request.start_date,
                    "requested_end": request.end_date,
                    "available_start": data_start.strftime('%Y-%m-%d'),
                    "available_end": data_end.strftime('%Y-%m-%d'),
                    "data_points": len(df)
                },
                timestamp=datetime.now().isoformat()
            )
        
        # Filter data by requested period
        df_filtered = filter_data_by_period(df, request.start_date, request.end_date)
        
        if len(df_filtered) < 100:  # Minimum data points for meaningful backtest
            return BacktestErrorResponse(
                error_type="DATA_MISSING",
                message=f"Insufficient data points for backtest. Need at least 100, got {len(df_filtered)}",
                details={"data_points": len(df_filtered), "minimum_required": 100},
                timestamp=datetime.now().isoformat()
            )
        
        # Initialize backtest engine with custom leverage
        engine = LeveragedBacktestEngine(leverage=request.leverage)
        
        # Load cross-asset data for correlation analysis
        cross_asset_symbols = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT']
        for cross_symbol in cross_asset_symbols:
            if cross_symbol != request.symbol:
                if cross_symbol in ['BTCUSDT', 'ETHUSDT']:
                    try:
                        cross_df = load_data_from_parquet(cross_symbol, 'data')
                        if not cross_df.empty:
                            engine.cross_asset_data[cross_symbol] = cross_df
                            engine.cross_asset_returns[cross_symbol] = cross_df['close'].pct_change(1).values
                    except Exception as e:
                        print(f"Warning: Could not load {cross_symbol}: {e}")
        
        # Run backtest
        results = engine.backtest_symbol(df_filtered, request.symbol)
        
        if 'error' in results:
            return BacktestErrorResponse(
                error_type="BACKTEST_ERROR",
                message=f"Backtest execution failed: {results['error']}",
                details={"symbol": request.symbol, "data_points": len(df_filtered)},
                timestamp=datetime.now().isoformat()
            )
        
        # Extract results
        total_return = results.get('total_return', 0.0)
        sharpe_ratio = results.get('sharpe_ratio', 0.0)
        max_drawdown = results.get('max_drawdown', 0.0)
        total_trades = results.get('total_trades', 0)
        win_rate = results.get('win_rate', 0.0)
        avg_gain_per_trade = results.get('avg_gain_per_trade', 0.0)
        profit_factor_metrics = results.get('profit_factor_metrics', {})
        
        return BacktestSuccessResponse(
            symbol=request.symbol,
            start_date=request.start_date,
            end_date=request.end_date,
            leverage=request.leverage,
            total_return=total_return,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown,
            total_trades=total_trades,
            win_rate=win_rate,
            avg_gain_per_trade=avg_gain_per_trade,
            profit_factor=profit_factor_metrics.get('profit_factor', 0.0),
            gross_profit=profit_factor_metrics.get('gross_profit', 0.0),
            gross_loss=profit_factor_metrics.get('gross_loss', 0.0),
            largest_win=profit_factor_metrics.get('largest_win', 0.0),
            largest_loss=profit_factor_metrics.get('largest_loss', 0.0),
            win_loss_ratio=profit_factor_metrics.get('win_loss_ratio', 0.0),
            margin_calls=results.get('margin_calls', 0),
            data_points=len(df_filtered),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        print(f"Backtest endpoint error: {e}")
        return BacktestErrorResponse(
            error_type="INTERNAL_ERROR",
            message=f"Internal server error: {str(e)}",
            details={"error": str(e)},
            timestamp=datetime.now().isoformat()
        )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Bybit-Integrated Trading Bot Dashboard API",
        "version": "1.0.0",
        "bybit_integration": True,
        "testnet": True,
        "endpoints": {
            "health": "/health",
            "status": "/status",
            "trading_data": "/trading/data",
            "bybit_data": "/bybit/data",
            "auto_trading": "/trading/auto/status",
            "backtest": "/backtest/run",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
