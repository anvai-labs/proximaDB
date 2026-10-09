"""
ProximaDB Python SDK - Enhanced Authentication Module

This module provides comprehensive authentication support for ProximaDB,
including JWT tokens, API keys, RBAC, and multi-provider authentication.

Copyright 2025 ProximaDB Contributors
Licensed under the Apache License, Version 2.0
"""

import asyncio
import logging
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


class AuthMethod(Enum):
    """Authentication methods supported by ProximaDB"""

    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    OAUTH2 = "oauth2"
    CLIENT_CERTIFICATE = "client_certificate"


class Permission(Enum):
    """ProximaDB permissions for RBAC"""

    # Collection permissions
    CREATE_COLLECTION = "CreateCollection"
    DELETE_COLLECTION = "DeleteCollection"
    LIST_COLLECTIONS = "ListCollections"
    READ_COLLECTION_METADATA = "ReadCollectionMetadata"
    UPDATE_COLLECTION_METADATA = "UpdateCollectionMetadata"

    # Vector permissions
    INSERT_VECTORS = "InsertVectors"
    DELETE_VECTORS = "DeleteVectors"
    SEARCH_VECTORS = "SearchVectors"
    UPDATE_VECTORS = "UpdateVectors"
    READ_VECTORS = "ReadVectors"

    # Graph permissions
    CREATE_GRAPH_RELATIONS = "CreateGraphRelations"
    DELETE_GRAPH_RELATIONS = "DeleteGraphRelations"
    TRAVERSE_GRAPH = "TraverseGraph"
    READ_GRAPH_RELATIONS = "ReadGraphRelations"

    # Query permissions
    EXECUTE_SQL_QUERIES = "ExecuteSqlQueries"
    EXECUTE_SKS_FUNCTIONS = "ExecuteSksFunctions"

    # System permissions
    VIEW_SYSTEM_METRICS = "ViewSystemMetrics"
    VIEW_SYSTEM_HEALTH = "ViewSystemHealth"
    CONFIGURE_SYSTEM = "ConfigureSystem"

    # Admin permissions
    MANAGE_USERS = "ManageUsers"
    MANAGE_ROLES = "ManageRoles"
    MANAGE_API_KEYS = "ManageApiKeys"
    VIEW_AUDIT_LOGS = "ViewAuditLogs"


@dataclass
class AuthResult:
    """Result of authentication containing user information and permissions"""

    user_id: str
    tenant_id: str | None = None
    roles: list[str] = None
    permissions: list[Permission] = None
    auth_method: AuthMethod = AuthMethod.API_KEY
    token_expires_at: datetime | None = None
    access_token: str | None = None
    refresh_token: str | None = None

    def __post_init__(self):
        if self.roles is None:
            self.roles = []
        if self.permissions is None:
            self.permissions = []

    def is_expired(self) -> bool:
        """Check if the authentication token is expired"""
        if self.token_expires_at is None:
            return False
        return datetime.now(UTC) >= self.token_expires_at

    def has_permission(self, permission: Permission) -> bool:
        """Check if the user has a specific permission"""
        return permission in self.permissions


@dataclass
class AuthConfig:
    """Configuration for ProximaDB authentication"""

    # Authentication method
    method: AuthMethod = AuthMethod.API_KEY

    # Basic auth settings
    enabled: bool = False
    api_key: str | None = None

    # JWT settings
    jwt_token: str | None = None
    jwt_refresh_token: str | None = None
    auto_refresh_jwt: bool = True
    refresh_threshold_minutes: int = 5

    # OAuth2 settings
    oauth2_token: str | None = None
    oauth2_provider: str | None = None
    oauth2_client_id: str | None = None
    oauth2_client_secret: str | None = None
    oauth2_redirect_uri: str | None = None

    # Certificate auth
    client_cert_path: str | None = None
    client_key_path: str | None = None
    client_cert_file: str | None = None  # Alias for client_cert_path
    client_key_file: str | None = None  # Alias for client_key_path
    ca_cert_path: str | None = None

    # Session management
    session_timeout_seconds: int = 3600
    max_concurrent_sessions: int = 5

    # Callbacks
    token_refresh_callback: Callable[["AuthResult"], None] | None = None
    auth_error_callback: Callable[[Exception], None] | None = None


class AuthenticationError(Exception):
    """Authentication failed"""

    pass


class AuthorizationError(Exception):
    """Authorization denied"""

    def __init__(self, message: str, required_permission: Permission | None = None):
        super().__init__(message)
        self.required_permission = required_permission


class TokenExpiredError(AuthenticationError):
    """JWT token has expired"""

    pass


class ProximaDBAuth:
    """
    ProximaDB Authentication Manager

    Handles authentication and authorization for ProximaDB clients,
    supporting multiple authentication methods including API keys,
    JWT tokens, OAuth2, and client certificates.
    """

    def __init__(
        self,
        config: AuthConfig,
        base_url: str,
        session: requests.Session | None = None,
    ):
        """Initialize authentication manager"""
        self.config = config
        self.base_url = base_url.rstrip("/")
        self.session = session or self._create_session()
        self.auth_result: AuthResult | None = None
        self._refresh_lock = asyncio.Lock() if hasattr(asyncio, "Lock") else None

        # Configure session for authentication endpoints
        if self.config.client_cert_path and self.config.client_key_path:
            self.session.cert = (
                self.config.client_cert_path,
                self.config.client_key_path,
            )

        if self.config.ca_cert_path:
            self.session.verify = self.config.ca_cert_path

    def _create_session(self) -> requests.Session:
        """Create HTTP session with retry strategy"""
        session = requests.Session()

        # Configure retry strategy
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)

        return session

    def authenticate(self) -> AuthResult:
        """
        Perform authentication using the configured method

        Returns:
            AuthResult: Authentication result with user info and permissions

        Raises:
            AuthenticationError: If authentication fails
        """
        if not self.config.enabled:
            # Return default auth result for disabled authentication
            return AuthResult(
                user_id="anonymous",
                auth_method=AuthMethod.API_KEY,
                permissions=list(
                    Permission
                ),  # Give all permissions when auth is disabled
            )

        # Determine authentication method
        if self.config.api_key:
            return self._authenticate_api_key()
        elif self.config.jwt_token:
            return self._authenticate_jwt()
        elif self.config.oauth2_token:
            return self._authenticate_oauth2()
        elif self.config.client_cert_path:
            return self._authenticate_client_cert()
        else:
            raise AuthenticationError("No authentication method configured")

    def _authenticate_api_key(self) -> AuthResult:
        """Authenticate using API key"""
        if not self.config.api_key:
            raise AuthenticationError("API key not configured")

        # For API key authentication, we don't need to call the server
        # The server will validate the key on each request
        return AuthResult(
            user_id="api_key_user",
            auth_method=AuthMethod.API_KEY,
            permissions=self._get_default_permissions(),
        )

    def _authenticate_jwt(self) -> AuthResult:
        """Authenticate using JWT token"""
        if not self.config.jwt_token:
            raise AuthenticationError("JWT token not configured")

        try:
            # Validate token with server
            response = self.session.post(
                f"{self.base_url}/auth/validate",
                headers={"Authorization": f"Bearer {self.config.jwt_token}"},
                timeout=10,
            )

            if response.status_code == 401:
                raise AuthenticationError("Invalid JWT token")
            elif response.status_code == 403:
                raise AuthorizationError("JWT token lacks required permissions")
            elif response.status_code != 200:
                raise AuthenticationError(
                    f"JWT validation failed: {response.status_code}"
                )

            auth_data = response.json()

            return AuthResult(
                user_id=auth_data.get("user_id", "jwt_user"),
                tenant_id=auth_data.get("tenant_id"),
                roles=auth_data.get("roles", []),
                permissions=[Permission(p) for p in auth_data.get("permissions", [])],
                auth_method=AuthMethod.JWT_TOKEN,
                token_expires_at=self._parse_expiration(auth_data.get("expires_at")),
                access_token=self.config.jwt_token,
                refresh_token=self.config.jwt_refresh_token,
            )

        except requests.exceptions.RequestException as e:
            logger.warning(
                f"JWT validation request failed, using offline validation: {e}"
            )
            return self._validate_jwt_offline()

    def _authenticate_oauth2(self) -> AuthResult:
        """Authenticate using OAuth2 token"""
        if not self.config.oauth2_token:
            raise AuthenticationError("OAuth2 token not configured")

        try:
            # Validate OAuth2 token with server
            response = self.session.post(
                f"{self.base_url}/auth/oauth2/validate",
                headers={"Authorization": f"Bearer {self.config.oauth2_token}"},
                json={
                    "provider": self.config.oauth2_provider,
                    "token": self.config.oauth2_token,
                },
                timeout=10,
            )

            if response.status_code != 200:
                raise AuthenticationError(
                    f"OAuth2 validation failed: {response.status_code}"
                )

            auth_data = response.json()

            return AuthResult(
                user_id=auth_data.get("user_id", "oauth2_user"),
                tenant_id=auth_data.get("tenant_id"),
                roles=auth_data.get("roles", []),
                permissions=[Permission(p) for p in auth_data.get("permissions", [])],
                auth_method=AuthMethod.OAUTH2,
                token_expires_at=self._parse_expiration(auth_data.get("expires_at")),
                access_token=self.config.oauth2_token,
            )

        except requests.exceptions.RequestException as e:
            raise AuthenticationError(f"OAuth2 authentication failed: {e}")

    def _authenticate_client_cert(self) -> AuthResult:
        """Authenticate using client certificate"""
        if not self.config.client_cert_path:
            raise AuthenticationError("Client certificate not configured")

        try:
            # Test certificate authentication
            response = self.session.get(
                f"{self.base_url}/auth/cert/validate", timeout=10
            )

            if response.status_code != 200:
                raise AuthenticationError(
                    f"Client certificate validation failed: {response.status_code}"
                )

            auth_data = response.json()

            return AuthResult(
                user_id=auth_data.get("user_id", "cert_user"),
                tenant_id=auth_data.get("tenant_id"),
                roles=auth_data.get("roles", []),
                permissions=[Permission(p) for p in auth_data.get("permissions", [])],
                auth_method=AuthMethod.CLIENT_CERTIFICATE,
            )

        except requests.exceptions.RequestException as e:
            raise AuthenticationError(f"Certificate authentication failed: {e}")

    def _validate_jwt_offline(self) -> AuthResult:
        """Offline JWT validation (basic checks only)"""
        # This would implement basic JWT parsing and validation
        # For now, return a basic result
        warnings.warn("Using offline JWT validation - permissions may not be accurate")

        return AuthResult(
            user_id="jwt_user_offline",
            auth_method=AuthMethod.JWT_TOKEN,
            permissions=self._get_default_permissions(),
            access_token=self.config.jwt_token,
            refresh_token=self.config.jwt_refresh_token,
        )

    def _get_default_permissions(self) -> list[Permission]:
        """Get default permissions for fallback scenarios"""
        return [
            Permission.LIST_COLLECTIONS,
            Permission.READ_COLLECTION_METADATA,
            Permission.INSERT_VECTORS,
            Permission.SEARCH_VECTORS,
            Permission.READ_VECTORS,
            Permission.READ_GRAPH_RELATIONS,
            Permission.EXECUTE_SQL_QUERIES,
            Permission.EXECUTE_SKS_FUNCTIONS,
            Permission.VIEW_SYSTEM_HEALTH,
        ]

    def _parse_expiration(self, expires_str: str | None) -> datetime | None:
        """Parse expiration timestamp from string"""
        if not expires_str:
            return None

        try:
            # Try parsing ISO format
            return datetime.fromisoformat(expires_str.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            try:
                # Try parsing Unix timestamp (could be seconds or milliseconds)
                timestamp = float(expires_str)
                # If timestamp is in milliseconds (> 1e12), convert to seconds
                if timestamp > 1e12:
                    timestamp = timestamp / 1000.0
                return datetime.fromtimestamp(timestamp, tz=UTC)
            except (ValueError, TypeError):
                logger.warning(f"Could not parse expiration time: {expires_str}")
                return None

    def get_auth_headers(self) -> dict[str, str]:
        """
        Get authentication headers for requests

        Returns:
            Dict containing appropriate authorization headers
        """
        if not self.config.enabled:
            return {}

        if not self.auth_result:
            self.auth_result = self.authenticate()

        # Check if token needs refresh
        if self._should_refresh_token():
            self._refresh_token()

        headers = {}

        if self.auth_result.auth_method == AuthMethod.API_KEY and self.config.api_key:
            headers["Authorization"] = f"API-Key {self.config.api_key}"
        elif self.auth_result.auth_method in [AuthMethod.JWT_TOKEN, AuthMethod.OAUTH2]:
            token = self.auth_result.access_token
            if token:
                headers["Authorization"] = f"Bearer {token}"

        return headers

    def _should_refresh_token(self) -> bool:
        """Check if token should be refreshed"""
        if not self.config.auto_refresh_jwt:
            return False

        if not self.auth_result or not self.auth_result.token_expires_at:
            return False

        # Refresh if token expires within threshold
        threshold = timedelta(minutes=self.config.refresh_threshold_minutes)
        return datetime.now(UTC) >= (self.auth_result.token_expires_at - threshold)

    def _refresh_token(self) -> None:
        """Refresh JWT token using refresh token"""
        if not self.config.jwt_refresh_token:
            logger.warning("Cannot refresh JWT token: no refresh token available")
            return

        try:
            response = self.session.post(
                f"{self.base_url}/auth/refresh",
                headers={"Authorization": f"Bearer {self.config.jwt_refresh_token}"},
                timeout=10,
            )

            if response.status_code != 200:
                raise AuthenticationError(
                    f"Token refresh failed: {response.status_code}"
                )

            token_data = response.json()

            # Update configuration and auth result
            self.config.jwt_token = token_data.get("access_token")
            self.config.jwt_refresh_token = token_data.get("refresh_token")

            if self.auth_result:
                self.auth_result.access_token = token_data.get("access_token")
                self.auth_result.refresh_token = token_data.get("refresh_token")
                self.auth_result.token_expires_at = self._parse_expiration(
                    token_data.get("expires_at")
                )

            # Call refresh callback if configured
            if self.config.token_refresh_callback and self.auth_result:
                self.config.token_refresh_callback(self.auth_result)

            logger.info("JWT token refreshed successfully")

        except requests.exceptions.RequestException as e:
            error_msg = f"Token refresh failed: {e}"
            logger.error(error_msg)

            if self.config.auth_error_callback:
                self.config.auth_error_callback(AuthenticationError(error_msg))

            raise AuthenticationError(error_msg)

    def check_permission(self, permission: Permission) -> bool:
        """
        Check if the authenticated user has a specific permission

        Args:
            permission: Permission to check

        Returns:
            bool: True if user has permission, False otherwise
        """
        if not self.config.enabled:
            return True  # All permissions granted when auth is disabled

        if not self.auth_result:
            self.auth_result = self.authenticate()

        return self.auth_result.has_permission(permission)

    def require_permission(self, permission: Permission) -> None:
        """
        Require a specific permission, raising AuthorizationError if not granted

        Args:
            permission: Required permission

        Raises:
            AuthorizationError: If permission is not granted
        """
        if not self.check_permission(permission):
            raise AuthorizationError(
                f"Permission required: {permission.value}",
                required_permission=permission,
            )

    def login(self, username: str, password: str) -> AuthResult:
        """
        Perform login with username/password and obtain JWT tokens

        Args:
            username: Username
            password: Password

        Returns:
            AuthResult with JWT tokens

        Raises:
            AuthenticationError: If login fails
        """
        try:
            response = self.session.post(
                f"{self.base_url}/auth/login",
                json={"username": username, "password": password},
                timeout=10,
            )

            if response.status_code != 200:
                raise AuthenticationError(f"Login failed: {response.status_code}")

            auth_data = response.json()

            # Update configuration with new tokens
            self.config.jwt_token = auth_data.get("access_token")
            self.config.jwt_refresh_token = auth_data.get("refresh_token")
            self.config.auth_method = AuthMethod.JWT_TOKEN

            # Create auth result
            self.auth_result = AuthResult(
                user_id=auth_data.get("user_id", username),
                tenant_id=auth_data.get("tenant_id"),
                roles=auth_data.get("roles", []),
                permissions=[Permission(p) for p in auth_data.get("permissions", [])],
                auth_method=AuthMethod.JWT_TOKEN,
                token_expires_at=self._parse_expiration(auth_data.get("expires_at")),
                access_token=auth_data.get("access_token"),
                refresh_token=auth_data.get("refresh_token"),
            )

            return self.auth_result

        except requests.exceptions.RequestException as e:
            raise AuthenticationError(f"Login request failed: {e}")

    def logout(self) -> None:
        """Logout and invalidate tokens"""
        if self.auth_result and self.auth_result.access_token:
            try:
                self.session.post(
                    f"{self.base_url}/auth/logout",
                    headers={
                        "Authorization": f"Bearer {self.auth_result.access_token}"
                    },
                    timeout=10,
                )
            except requests.exceptions.RequestException:
                pass  # Ignore logout errors

        # Clear tokens and auth result
        self.config.jwt_token = None
        self.config.jwt_refresh_token = None
        self.config.oauth2_token = None
        self.auth_result = None

    def get_user_info(self) -> dict[str, Any] | None:
        """Get information about the authenticated user"""
        if not self.auth_result:
            return None

        return {
            "user_id": self.auth_result.user_id,
            "tenant_id": self.auth_result.tenant_id,
            "roles": self.auth_result.roles,
            "permissions": [p.value for p in self.auth_result.permissions],
            "auth_method": self.auth_result.auth_method.value,
            "expires_at": (
                self.auth_result.token_expires_at.isoformat()
                if self.auth_result.token_expires_at
                else None
            ),
        }


# Convenience functions for creating auth configurations


def create_api_key_auth(api_key: str, **kwargs) -> AuthConfig:
    """Create API key authentication configuration"""
    return AuthConfig(enabled=True, api_key=api_key, **kwargs)


def create_jwt_auth(
    access_token: str,
    refresh_token: str | None = None,
    auto_refresh: bool = True,
    **kwargs,
) -> AuthConfig:
    """Create JWT authentication configuration"""
    return AuthConfig(
        enabled=True,
        jwt_token=access_token,
        jwt_refresh_token=refresh_token,
        auto_refresh_jwt=auto_refresh,
        **kwargs,
    )


def create_oauth2_auth(
    access_token: str,
    provider: str = "oauth2",
    client_id: str | None = None,
    **kwargs,
) -> AuthConfig:
    """Create OAuth2 authentication configuration"""
    return AuthConfig(
        enabled=True,
        oauth2_token=access_token,
        oauth2_provider=provider,
        oauth2_client_id=client_id,
        **kwargs,
    )


def create_cert_auth(
    cert_path: str, key_path: str, ca_path: str | None = None, **kwargs
) -> AuthConfig:
    """Create client certificate authentication configuration"""
    return AuthConfig(
        enabled=True,
        client_cert_path=cert_path,
        client_key_path=key_path,
        ca_cert_path=ca_path,
        **kwargs,
    )
