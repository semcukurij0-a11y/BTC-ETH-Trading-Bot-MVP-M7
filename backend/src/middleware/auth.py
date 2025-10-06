"""
Authentication Middleware for FastAPI

This module provides authentication middleware for the trading bot API:
- JWT token validation
- User authentication
- Permission checking
- Request logging
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

class AuthMiddleware:
    """
    Authentication middleware for FastAPI endpoints.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    async def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
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
            
            self.logger.debug(f"User {user['username']} authenticated successfully")
            return user
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )
    
    async def get_current_active_user(self, current_user: Dict[str, Any] = Depends(self.get_current_user)) -> Dict[str, Any]:
        """
        Get current active user (additional validation can be added here).
        
        Args:
            current_user: Current user from get_current_user
            
        Returns:
            Active user data dictionary
        """
        # Additional validation can be added here (e.g., check if user is active)
        return current_user
    
    def require_permission(self, permission: str):
        """
        Create a dependency that requires a specific permission.
        
        Args:
            permission: Required permission
            
        Returns:
            Dependency function
        """
        async def permission_checker(current_user: Dict[str, Any] = Depends(self.get_current_active_user)) -> Dict[str, Any]:
            if not auth_service.has_permission(current_user, permission):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Permission '{permission}' required"
                )
            return current_user
        
        return permission_checker
    
    def require_role(self, role: str):
        """
        Create a dependency that requires a specific role.
        
        Args:
            role: Required role
            
        Returns:
            Dependency function
        """
        async def role_checker(current_user: Dict[str, Any] = Depends(self.get_current_active_user)) -> Dict[str, Any]:
            if current_user.get("role") != role and current_user.get("role") != "admin":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Role '{role}' required"
                )
            return current_user
        
        return role_checker

# Initialize middleware
auth_middleware = AuthMiddleware()

# Common dependencies
get_current_user = auth_middleware.get_current_user
get_current_active_user = auth_middleware.get_current_active_user

# Permission-based dependencies
require_read_permission = auth_middleware.require_permission("read")
require_write_permission = auth_middleware.require_permission("write")
require_execute_permission = auth_middleware.require_permission("execute")
require_admin_permission = auth_middleware.require_permission("admin")

# Role-based dependencies
require_trader_role = auth_middleware.require_role("trader")
require_admin_role = auth_middleware.require_role("admin")
