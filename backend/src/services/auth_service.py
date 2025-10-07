"""
Authentication Service for Trading Bot Dashboard

This module provides authentication and authorization functionality:
- User authentication with JWT tokens
- Password hashing and validation
- Session management
- Role-based access control
"""

import jwt
from jwt.exceptions import InvalidTokenError, ExpiredSignatureError
import hashlib
import secrets
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import HTTPException, status
import logging

class AuthService:
    """
    Authentication service for the trading bot dashboard.
    """
    
    def __init__(self, secret_key: str = None):
        """
        Initialize authentication service.
        
        Args:
            secret_key: JWT secret key (defaults to generated key)
        """
        self.logger = logging.getLogger(__name__)
        
        # JWT configuration - use a fixed secret key for consistency
        self.secret_key = secret_key or "trading-bot-secret-key-2024"
        self.algorithm = "HS256"
        self.access_token_expire_minutes = 30
        self.refresh_token_expire_days = 7
        
        # Password hashing - using simple hash for demo purposes
        
        # Default users (in production, this would be in a database)
        self.users = {
            "admin": {
                "id": "1",
                "username": "admin",
                "password_hash": self._hash_password("admin123"),
                "role": "admin",
                "permissions": ["read", "write", "execute", "admin"]
            },
            "trader": {
                "id": "2", 
                "username": "trader",
                "password_hash": self._hash_password("trader123"),
                "role": "trader",
                "permissions": ["read", "write"]
            },
            "viewer": {
                "id": "3",
                "username": "viewer", 
                "password_hash": self._hash_password("viewer123"),
                "role": "viewer",
                "permissions": ["read"]
            }
        }
        
        # Active sessions
        self.active_sessions = {}
        
        self.logger.info("Authentication service initialized")
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key for JWT."""
        return secrets.token_urlsafe(32)
    
    def _hash_password(self, password: str) -> str:
        """Hash a password using SHA-256 (for demo purposes)."""
        return hashlib.sha256(password.encode()).hexdigest()
    
    def _verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify a password against its hash."""
        return self._hash_password(plain_password) == hashed_password
    
    def _create_access_token(self, data: Dict[str, Any]) -> str:
        """Create a JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.access_token_expire_minutes)
        to_encode.update({"exp": expire, "type": "access"})
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def _create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create a JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        encoded_jwt = jwt.encode(to_encode, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def _verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Verify and decode a JWT token."""
        try:
            self.logger.debug(f"Verifying token: {token[:20]}...")
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            self.logger.debug(f"Token payload: {payload}")
            
            if payload.get("type") != token_type:
                self.logger.warning(f"Invalid token type. Expected: {token_type}, Got: {payload.get('type')}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token type"
                )
            
            return payload
        except ExpiredSignatureError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        except InvalidTokenError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )
    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """
        Authenticate a user with username and password.
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            User data if authentication successful, None otherwise
        """
        try:
            if username not in self.users:
                return None
            
            user = self.users[username]
            if not self._verify_password(password, user["password_hash"]):
                return None
            
            # Remove password hash from response
            user_data = {
                "id": user["id"],
                "username": user["username"],
                "role": user["role"],
                "permissions": user["permissions"]
            }
            
            self.logger.info(f"User {username} authenticated successfully")
            return user_data
            
        except Exception as e:
            self.logger.error(f"Authentication error for user {username}: {e}")
            return None
    
    def login(self, username: str, password: str) -> Dict[str, Any]:
        """
        Login a user and return tokens.
        
        Args:
            username: Username
            password: Plain text password
            
        Returns:
            Dictionary with login result
        """
        try:
            user = self.authenticate_user(username, password)
            if not user:
                return {
                    "success": False,
                    "message": "Invalid username or password"
                }
            
            # Create tokens
            access_token = self._create_access_token({"sub": user["id"], "username": user["username"]})
            refresh_token = self._create_refresh_token({"sub": user["id"], "username": user["username"]})
            
            # Store session
            session_id = secrets.token_urlsafe(16)
            self.active_sessions[session_id] = {
                "user_id": user["id"],
                "username": user["username"],
                "created_at": datetime.utcnow(),
                "last_activity": datetime.utcnow()
            }
            
            return {
                "success": True,
                "token": access_token,
                "refresh_token": refresh_token,
                "user": user,
                "session_id": session_id
            }
            
        except Exception as e:
            self.logger.error(f"Login error for user {username}: {e}")
            return {
                "success": False,
                "message": "Login failed"
            }
    
    def logout(self, session_id: str) -> Dict[str, Any]:
        """
        Logout a user and invalidate session.
        
        Args:
            session_id: Session ID to invalidate
            
        Returns:
            Dictionary with logout result
        """
        try:
            if session_id in self.active_sessions:
                del self.active_sessions[session_id]
                self.logger.info(f"Session {session_id} invalidated")
            
            return {"success": True, "message": "Logged out successfully"}
            
        except Exception as e:
            self.logger.error(f"Logout error for session {session_id}: {e}")
            return {"success": False, "message": "Logout failed"}
    
    def refresh_access_token(self, refresh_token: str) -> Dict[str, Any]:
        """
        Refresh an access token using a refresh token.
        
        Args:
            refresh_token: Valid refresh token
            
        Returns:
            Dictionary with new access token
        """
        try:
            payload = self._verify_token(refresh_token, "refresh")
            user_id = payload.get("sub")
            username = payload.get("username")
            
            if not user_id or not username:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid refresh token"
                )
            
            # Create new access token
            new_access_token = self._create_access_token({"sub": user_id, "username": username})
            
            return {
                "success": True,
                "token": new_access_token
            }
            
        except Exception as e:
            self.logger.error(f"Token refresh error: {e}")
            return {
                "success": False,
                "message": "Token refresh failed"
            }
    
    def verify_access_token(self, token: str) -> Dict[str, Any]:
        """
        Verify an access token and return user data.
        
        Args:
            token: Access token to verify
            
        Returns:
            Dictionary with user data
        """
        try:
            self.logger.debug(f"Verifying access token: {token[:20]}...")
            payload = self._verify_token(token, "access")
            user_id = payload.get("sub")
            username = payload.get("username")
            
            self.logger.debug(f"Token payload - user_id: {user_id}, username: {username}")
            
            if not user_id or not username:
                self.logger.warning("Missing user_id or username in token")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )
            
            # Get user data
            user = None
            for user_data in self.users.values():
                if user_data["id"] == user_id:
                    user = {
                        "id": user_data["id"],
                        "username": user_data["username"],
                        "role": user_data["role"],
                        "permissions": user_data["permissions"]
                    }
                    break
            
            if not user:
                self.logger.warning(f"User not found for user_id: {user_id}")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="User not found"
                )
            
            self.logger.debug(f"Token verification successful for user: {user['username']}")
            return user
            
        except Exception as e:
            self.logger.error(f"Token verification error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token verification failed"
            )
    
    def has_permission(self, user: Dict[str, Any], permission: str) -> bool:
        """
        Check if a user has a specific permission.
        
        Args:
            user: User data
            permission: Permission to check
            
        Returns:
            True if user has permission, False otherwise
        """
        if not user:
            return False
        
        return permission in user.get("permissions", []) or user.get("role") == "admin"
    
    def get_active_sessions(self) -> List[Dict[str, Any]]:
        """
        Get list of active sessions.
        
        Returns:
            List of active sessions
        """
        return [
            {
                "session_id": session_id,
                "username": session_data["username"],
                "created_at": session_data["created_at"].isoformat(),
                "last_activity": session_data["last_activity"].isoformat()
            }
            for session_id, session_data in self.active_sessions.items()
        ]
    
    def cleanup_expired_sessions(self):
        """Clean up expired sessions."""
        current_time = datetime.utcnow()
        expired_sessions = []
        
        for session_id, session_data in self.active_sessions.items():
            # Remove sessions older than 7 days
            if (current_time - session_data["last_activity"]).days > 7:
                expired_sessions.append(session_id)
        
        for session_id in expired_sessions:
            del self.active_sessions[session_id]
        
        if expired_sessions:
            self.logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")
