"""
ProximaDB Python Client - Exception Classes

Copyright 2025 ProximaDB

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from typing import Any


class ProximaDBError(Exception):
    """Base exception for all ProximaDB client errors"""

    def __init__(
        self,
        message: str,
        error_code: str | None = None,
        details: dict[str, Any] | None = None,
        request_id: str | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.request_id = request_id
        self.retryable = retryable

    def __str__(self) -> str:
        parts = [self.message]
        if self.error_code:
            parts.append(f"Error Code: {self.error_code}")
        if self.request_id:
            parts.append(f"Request ID: {self.request_id}")
        return " | ".join(parts)


class ProximaDBNotSupportedError(ProximaDBError):
    """The active transport/adapter lacks the requested operation.

    Raised by the v2 facade when an adapter declines an operation with
    ``NotImplementedError`` (an optional-namespace method the transport
    does not implement — e.g. graph-collection management on gRPC). This is
    a *capability* signal, not a request failure: compatibility wrappers
    (``graph.py`` batch helpers) treat it like the v1-era ``AttributeError``
    and fall back to per-item operations instead of failing the batch.
    """


class AuthenticationError(ProximaDBError):
    """Authentication failed - invalid API key or token"""

    def __init__(self, message: str = "Authentication failed", **kwargs) -> None:
        super().__init__(message, error_code="AUTH_FAILED", **kwargs)


class AuthorizationError(ProximaDBError):
    """Authorization failed - insufficient permissions"""

    def __init__(self, message: str = "Insufficient permissions", **kwargs) -> None:
        super().__init__(message, error_code="AUTH_INSUFFICIENT", **kwargs)


class CollectionNotFoundError(ProximaDBError):
    """Collection does not exist"""

    def __init__(self, collection_id: str, **kwargs) -> None:
        message = f"Collection not found: {collection_id}"
        super().__init__(message, error_code="COLLECTION_NOT_FOUND", **kwargs)
        self.collection_id = collection_id


class CollectionExistsError(ProximaDBError):
    """Collection already exists"""

    def __init__(self, collection_name: str, **kwargs) -> None:
        message = f"Collection already exists: {collection_name}"
        super().__init__(message, error_code="COLLECTION_EXISTS", **kwargs)
        self.collection_name = collection_name


class VectorNotFoundError(ProximaDBError):
    """Vector does not exist"""

    def __init__(self, vector_id: str, **kwargs) -> None:
        message = f"Vector not found: {vector_id}"
        super().__init__(message, error_code="VECTOR_NOT_FOUND", **kwargs)
        self.vector_id = vector_id


class VectorDimensionError(ProximaDBError):
    """Vector dimension mismatch"""

    def __init__(self, expected: int, actual: int, **kwargs) -> None:
        message = f"Vector dimension mismatch: expected {expected}, got {actual}"
        super().__init__(message, error_code="DIMENSION_MISMATCH", **kwargs)
        self.expected_dimension = expected
        self.actual_dimension = actual


class InvalidVectorError(ProximaDBError):
    """Invalid vector data"""

    def __init__(self, message: str = "Invalid vector data", **kwargs) -> None:
        super().__init__(message, error_code="INVALID_VECTOR", **kwargs)


class RateLimitError(ProximaDBError):
    """Rate limit exceeded"""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        retry_after: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(message, error_code="RATE_LIMIT_EXCEEDED", **kwargs)
        self.retry_after = retry_after  # seconds to wait before retry


class QuotaExceededError(ProximaDBError):
    """Usage quota exceeded"""

    def __init__(
        self,
        message: str = "Usage quota exceeded",
        quota_type: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(message, error_code="QUOTA_EXCEEDED", **kwargs)
        self.quota_type = quota_type


class ValidationError(ProximaDBError):
    """Request validation failed"""

    def __init__(self, message: str, field: str | None = None, **kwargs) -> None:
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.field = field


class ServerError(ProximaDBError):
    """Internal server error"""

    def __init__(
        self,
        message: str = "Internal server error",
        status_code: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(message, error_code="SERVER_ERROR", **kwargs)
        self.status_code = status_code


class NetworkError(ProximaDBError):
    """Network communication error"""

    def __init__(
        self,
        message: str = "Network error",
        original_error: Exception | None = None,
        **kwargs,
    ) -> None:
        super().__init__(message, error_code="NETWORK_ERROR", **kwargs)
        self.original_error = original_error


class TransportError(ProximaDBError):
    """Transport layer error (REST/gRPC)"""

    def __init__(
        self,
        message: str = "Transport error",
        transport_type: str | None = None,
        original_error: Exception | None = None,
        **kwargs,
    ) -> None:
        super().__init__(message, error_code="TRANSPORT_ERROR", **kwargs)
        self.transport_type = transport_type
        self.original_error = original_error


class TimeoutError(ProximaDBError):
    """Request timeout"""

    def __init__(
        self,
        message: str = "Request timeout",
        timeout_seconds: float | None = None,
        **kwargs,
    ) -> None:
        super().__init__(message, error_code="TIMEOUT", **kwargs)
        self.timeout_seconds = timeout_seconds


class ConfigurationError(ProximaDBError):
    """Client configuration error"""

    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)


class IndexError(ProximaDBError):
    """Vector index error"""

    def __init__(self, message: str, index_type: str | None = None, **kwargs) -> None:
        super().__init__(message, error_code="INDEX_ERROR", **kwargs)
        self.index_type = index_type


class BatchError(ProximaDBError):
    """Batch operation error"""

    def __init__(
        self,
        message: str,
        successful_count: int = 0,
        failed_count: int = 0,
        errors: list | None = None,
        **kwargs,
    ) -> None:
        super().__init__(message, error_code="BATCH_ERROR", **kwargs)
        self.successful_count = successful_count
        self.failed_count = failed_count
        self.errors = errors or []


class WALError(ProximaDBError):
    """Write-Ahead Log operation error"""

    def __init__(self, message: str = "WAL operation failed", **kwargs) -> None:
        super().__init__(message, error_code="WAL_ERROR", **kwargs)


class StreamingError(ProximaDBError):
    """Streaming operation error"""

    def __init__(self, message: str, **kwargs) -> None:
        super().__init__(message, error_code="STREAMING_ERROR", **kwargs)


def map_http_error(
    status_code: int, response_data: dict, headers: dict | None = None
) -> ProximaDBError:
    """Map an HTTP error response to a typed exception.

    Accepts the canonical ProximaDB error envelope
    ``{"error": {"type", "message", "code", "request_id", "details"?}}`` as well
    as the legacy flat shape ``{"error_code", "message", "request_id", "details"}``.
    The server's ``type`` is lowercase snake_case (e.g. ``collection_not_found``);
    it is upper-cased so the branches below (``COLLECTION_NOT_FOUND``) match.
    ``request_id`` falls back to the ``X-Request-ID`` response header.
    """
    err = response_data.get("error")
    if isinstance(err, dict):
        # Canonical nested envelope.
        error_code = str(err.get("type") or "unknown").upper()
        message = err.get("message", f"HTTP {status_code} error")
        request_id = err.get("request_id") or response_data.get("request_id")
        details = err.get("details") or {}
    else:
        # Legacy flat shape (or a bare string error).
        error_code = str(response_data.get("error_code") or err or "UNKNOWN").upper()
        message = response_data.get("message", f"HTTP {status_code} error")
        request_id = response_data.get("request_id")
        details = response_data.get("details", {})

    if not request_id and headers:
        # Header lookups are case-insensitive for httpx.Headers; guard plain dicts.
        request_id = (
            headers.get("x-request-id")
            or headers.get("X-Request-ID")
            or (headers.get("X-Request-Id") if hasattr(headers, "get") else None)
        )

    if status_code == 400:
        if error_code == "VALIDATION_ERROR":
            return ValidationError(message, request_id=request_id, details=details)
        elif error_code == "DIMENSION_MISMATCH":
            expected = details.get("expected_dimension")
            actual = details.get("actual_dimension")
            if expected and actual:
                return VectorDimensionError(expected, actual, request_id=request_id)
        return ProximaDBError(message, error_code, details, request_id, retryable=False)

    elif status_code == 401:
        return AuthenticationError(message, request_id=request_id, details=details)

    elif status_code == 403:
        return AuthorizationError(message, request_id=request_id, details=details)

    elif status_code == 404:
        if error_code == "COLLECTION_NOT_FOUND":
            collection_id = details.get("collection_id", "unknown")
            return CollectionNotFoundError(collection_id, request_id=request_id)
        elif error_code == "VECTOR_NOT_FOUND":
            vector_id = details.get("vector_id", "unknown")
            return VectorNotFoundError(vector_id, request_id=request_id)
        return ProximaDBError(message, error_code, details, request_id, retryable=False)

    elif status_code == 409:
        if error_code == "COLLECTION_EXISTS":
            collection_name = details.get("collection_name", "unknown")
            return CollectionExistsError(collection_name, request_id=request_id)
        return ProximaDBError(message, error_code, details, request_id, retryable=False)

    elif status_code == 429:
        retry_after = details.get("retry_after")
        return RateLimitError(
            message, retry_after=retry_after, request_id=request_id, details=details
        )

    elif status_code == 413:
        quota_type = details.get("quota_type")
        return QuotaExceededError(
            message, quota_type=quota_type, request_id=request_id, details=details
        )

    elif 500 <= status_code < 600:
        return ServerError(
            message, status_code=status_code, request_id=request_id, details=details
        )

    else:
        return ProximaDBError(message, error_code, details, request_id, retryable=False)


def map_grpc_error(grpc_error) -> ProximaDBError:
    """Map gRPC errors to appropriate exceptions"""
    try:
        import grpc

        status_code = grpc_error.code()
        message = grpc_error.details()

        if status_code == grpc.StatusCode.UNAUTHENTICATED:
            return AuthenticationError(message)
        elif status_code == grpc.StatusCode.PERMISSION_DENIED:
            return AuthorizationError(message)
        elif status_code == grpc.StatusCode.NOT_FOUND:
            return ProximaDBError(message, error_code="NOT_FOUND")
        elif status_code == grpc.StatusCode.ALREADY_EXISTS:
            return ProximaDBError(message, error_code="ALREADY_EXISTS")
        elif status_code == grpc.StatusCode.INVALID_ARGUMENT:
            return ValidationError(message)
        elif status_code == grpc.StatusCode.RESOURCE_EXHAUSTED:
            return RateLimitError(message)
        elif status_code == grpc.StatusCode.DEADLINE_EXCEEDED:
            return TimeoutError(message)
        elif status_code == grpc.StatusCode.UNAVAILABLE:
            return NetworkError(message, original_error=grpc_error)
        elif status_code == grpc.StatusCode.INTERNAL:
            return ServerError(message)
        else:
            return ProximaDBError(
                message,
                error_code=status_code.name,
                details={"grpc_error": str(grpc_error)},
            )

    except ImportError:
        # grpc not available
        return ProximaDBError(str(grpc_error), error_code="GRPC_ERROR")
