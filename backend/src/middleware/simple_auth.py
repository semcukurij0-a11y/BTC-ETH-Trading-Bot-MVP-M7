"""
Simple Authentication Middleware for FastAPI

This module provides simplified authentication for the trading bot API:
- Basic JWT token validation
- Simple permission checking
- No complex dependency chains
"""

from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import logging

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.auth_service import AuthService

# Initialize auth service
auth_service = AuthService()

# HTTP Bearer token scheme
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    """
    Get current authenticated user from JWT token.
    
    Args:
        credentials: HTTP Bearer credentials
        
    Returns:
        User data dictionary
        
    Raises:
        HTTPException: If authentication fails
    """
    try:
        token = credentials.credentials
        user = auth_service.verify_access_token(token)
        
        logging.getLogger(__name__).debug(f"User {user['username']} authenticated successfully")
        return user
        
    except HTTPException:
        raise
    except Exception as e:
        logging.getLogger(__name__).error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed"
        )

def require_read_permission(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Require read permission.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User data if authorized
        
    Raises:
        HTTPException: If user lacks permission
    """
    if not auth_service.has_permission(current_user, "read"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Read permission required"
        )
    return current_user

def require_write_permission(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Require write permission.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User data if authorized
        
    Raises:
        HTTPException: If user lacks permission
    """
    if not auth_service.has_permission(current_user, "write"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Write permission required"
        )
    return current_user

def require_admin_permission(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    """
    Require admin permission.
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User data if authorized
        
    Raises:
        HTTPException: If user lacks permission
    """
    if not auth_service.has_permission(current_user, "admin"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin permission required"
        )
    return current_user
