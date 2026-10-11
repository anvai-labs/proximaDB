"""
ProximaDB Python Client - Configuration

Copyright 2024 Vijaykumar Singh

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

import os
from enum import Enum
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Protocol(str, Enum):
    """Supported communication protocols"""

    AUTO = "auto"  # Auto-select best available
    GRPC = "grpc"  # gRPC (high performance, binary protocol)
    REST = "rest"  # REST (web compatibility)
    EMBEDDED = "embedded"  # In-process Python embedded engine
    ARROW_FLIGHT = "arrow_flight"  # Apache Arrow Flight (bulk data transfer)


class PortMode(str, Enum):
    """Server port configuration mode"""

    MULTI = "multi"  # Legacy multi-port mode (different ports for REST/gRPC)
    UNIFIED = "unified"  # New unified port mode (single port for all protocols)


class LogLevel(str, Enum):
    """Logging levels"""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


# Import retry configuration from resilience module
from .resilience import NetworkRetryPolicy as RetryConfig


class ConnectionConfig(BaseModel):
    """Connection pool configuration"""

    pool_size: int = Field(default=10, ge=1, le=100)
    pool_maxsize: int = Field(default=100, ge=1, le=1000)
    keepalive_timeout: float = Field(default=30.0, ge=1.0)
    read_timeout: float = Field(default=30.0, ge=1.0)
    connect_timeout: float = Field(default=10.0, ge=1.0)
    total_timeout: float = Field(default=60.0, ge=1.0)


class CompressionConfig(BaseModel):
    """Compression configuration for Release 1.0

    Clean configuration without legacy compatibility.
    """

    enabled: bool = Field(default=False, description="Enable compression")
    algorithm: str = Field(default="gzip", description="Compression algorithm")
    threshold_bytes: int = Field(
        default=1024, ge=0, description="Minimum size to compress"
    )
    level: int | None = Field(default=None, ge=1, le=9, description="Compression level")


class TLSConfig(BaseModel):
    """TLS/SSL configuration"""

    verify: bool = True
    ca_bundle: str | None = None
    cert_file: str | None = None
    key_file: str | None = None
    sni_hostname: str | None = None


class ClientConfig(BaseModel):
    """Complete client configuration"""

    # Connection settings
    url: str = Field(..., description="ProximaDB server URL")
    api_key: str | None = Field(default=None, description="API key for authentication")
    protocol: Protocol = Field(
        default=Protocol.AUTO, description="Communication protocol"
    )
    port_mode: PortMode = Field(
        default=PortMode.UNIFIED, description="Server port mode (unified or multi)"
    )
    uds_path: str | None = Field(
        default=None,
        description=(
            "Path to a Unix-domain socket to connect through instead of TCP. "
            "Set this to talk to an embedded server started with "
            "transport='uds' (the embedded default since #264), whose rest_url "
            "is the host-header sentinel 'http://localhost' and carries no port."
        ),
    )

    # Timeouts
    timeout: float = Field(
        default=30.0, ge=0.1, le=300.0, description="Default request timeout"
    )

    # Retry configuration
    retry: RetryConfig = Field(default_factory=RetryConfig)

    # Connection configuration
    connection: ConnectionConfig = Field(default_factory=ConnectionConfig)

    # Compression
    compression: CompressionConfig = Field(default_factory=CompressionConfig)

    # TLS/SSL
    tls: TLSConfig = Field(default_factory=TLSConfig)

    # Headers and metadata
    user_agent: str | None = Field(default=None, description="Custom User-Agent header")
    custom_headers: dict[str, str] = Field(default_factory=dict)

    # Logging
    log_level: LogLevel = Field(default=LogLevel.INFO)
    enable_debug_logging: bool = False

    # Performance tuning
    enable_keepalive: bool = True
    enable_http2: bool = True
    max_concurrent_requests: int = Field(default=100, ge=1, le=1000)

    # Client behavior
    validate_inputs: bool = True
    auto_convert_numpy: bool = True
    default_batch_size: int = Field(default=1000, ge=1, le=10000)

    model_config = ConfigDict(use_enum_values=True)

    @field_validator("url")
    def validate_url(cls, v: str) -> str:
        """Validate and normalize URL"""
        if not v:
            raise ValueError("URL cannot be empty")

        # Detect an explicit scheme by the "://" separator rather than
        # urlparse().scheme: urlparse mis-parses a bare host:port like
        # "127.0.0.1:5679" as scheme="127.0.0.1" on some Python/urllib
        # versions and as scheme="" on others, which made this validator
        # (and test_url_grpc_hostport_kept_asis) flap between 3.10/3.11/3.12.
        if "://" not in v:
            # No explicit scheme. A "host:port" form is the gRPC / Arrow
            # Flight endpoint shorthand — keep it as-is.
            if ":" in v:
                return v
            # Bare hostname -> default to https.
            return f"https://{v}"

        parsed = urlparse(v)
        if parsed.scheme not in ("http", "https", "grpc", "embedded"):
            raise ValueError("URL must use http, https, grpc, or embedded scheme")
        if not parsed.netloc:
            raise ValueError("URL must include hostname")

        return v

    @field_validator("api_key")
    def validate_api_key(cls, v: str | None) -> str | None:
        """Validate API key format"""
        if v and len(v) < 10:
            raise ValueError("API key appears to be too short")
        return v

    @classmethod
    def from_env(cls, **overrides) -> "ClientConfig":
        """Create configuration from environment variables"""
        config_dict = {}

        # Basic settings
        if url := os.getenv("PROXIMADB_URL"):
            config_dict["url"] = url
        if api_key := os.getenv("PROXIMADB_API_KEY"):
            config_dict["api_key"] = api_key
        if protocol := os.getenv("PROXIMADB_PROTOCOL"):
            config_dict["protocol"] = Protocol(protocol.lower())
        if port_mode := os.getenv("PROXIMADB_PORT_MODE"):
            config_dict["port_mode"] = PortMode(port_mode.lower())
        if timeout := os.getenv("PROXIMADB_TIMEOUT"):
            config_dict["timeout"] = float(timeout)

        # Retry settings
        retry_config = {}
        if max_retries := os.getenv("PROXIMADB_MAX_RETRIES"):
            retry_config["max_retries"] = int(max_retries)
        if backoff_factor := os.getenv("PROXIMADB_BACKOFF_FACTOR"):
            retry_config["backoff_factor"] = float(backoff_factor)
        if retry_config:
            config_dict["retry"] = RetryConfig(**retry_config)

        # Connection settings
        connection_config = {}
        if pool_size := os.getenv("PROXIMADB_POOL_SIZE"):
            connection_config["pool_size"] = int(pool_size)
        if read_timeout := os.getenv("PROXIMADB_READ_TIMEOUT"):
            connection_config["read_timeout"] = float(read_timeout)
        if connection_config:
            config_dict["connection"] = ConnectionConfig(**connection_config)

        # TLS settings
        tls_config = {}
        if verify := os.getenv("PROXIMADB_TLS_VERIFY"):
            tls_config["verify"] = verify.lower() in ("true", "1", "yes")
        if ca_bundle := os.getenv("PROXIMADB_CA_BUNDLE"):
            tls_config["ca_bundle"] = ca_bundle
        if cert_file := os.getenv("PROXIMADB_CERT_FILE"):
            tls_config["cert_file"] = cert_file
        if key_file := os.getenv("PROXIMADB_KEY_FILE"):
            tls_config["key_file"] = key_file
        if tls_config:
            config_dict["tls"] = TLSConfig(**tls_config)

        # Logging
        if log_level := os.getenv("PROXIMADB_LOG_LEVEL"):
            config_dict["log_level"] = log_level.upper()
        if debug := os.getenv("PROXIMADB_DEBUG"):
            config_dict["enable_debug_logging"] = debug.lower() in ("true", "1", "yes")

        # Performance
        if batch_size := os.getenv("PROXIMADB_BATCH_SIZE"):
            config_dict["default_batch_size"] = int(batch_size)
        if max_concurrent := os.getenv("PROXIMADB_MAX_CONCURRENT"):
            config_dict["max_concurrent_requests"] = int(max_concurrent)

        # Apply overrides
        config_dict.update(overrides)

        # URL is required
        if "url" not in config_dict:
            raise ValueError(
                "URL must be provided via PROXIMADB_URL environment variable or constructor"
            )

        # Convert protocol string to enum if needed
        if "protocol" in config_dict and isinstance(config_dict["protocol"], str):
            config_dict["protocol"] = Protocol(config_dict["protocol"].lower())

        return cls(**config_dict)

    def get_base_headers(self) -> dict[str, str]:
        """Get base headers for requests"""
        headers = {
            "User-Agent": self.user_agent or f"proximadb-python/{self._get_version()}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        # Add compression headers if enabled
        if self.compression.enabled:
            headers["Accept-Encoding"] = "gzip, deflate, br, zstd"

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        # Add custom headers
        headers.update(self.custom_headers)

        return headers

    def get_grpc_metadata(self) -> list:
        """Get gRPC metadata for requests"""
        metadata = []

        if self.api_key:
            metadata.append(("authorization", f"Bearer {self.api_key}"))

        metadata.append(
            ("user-agent", self.user_agent or f"proximadb-python/{self._get_version()}")
        )

        # Add custom headers as metadata
        for key, value in self.custom_headers.items():
            metadata.append((key.lower(), value))

        return metadata

    def _get_version(self) -> str:
        """Get client version"""
        try:
            from . import __version__

            return __version__
        except ImportError:
            return "unknown"

    def is_secure(self) -> bool:
        """Check if connection should use TLS"""
        parsed = urlparse(self.url)
        return parsed.scheme == "https"

    def get_host_port(self) -> tuple:
        """Get host and port from URL"""
        parsed = urlparse(self.url)
        host = parsed.hostname or "localhost"

        if parsed.port:
            port = parsed.port
        else:
            port = 443 if self.is_secure() else 80

        return host, port

    def should_use_grpc(self) -> bool:
        """Determine if gRPC should be used"""
        if self.protocol == Protocol.GRPC:
            return True
        elif self.protocol == Protocol.EMBEDDED:
            return False
        elif self.protocol == Protocol.REST:
            return False
        else:  # AUTO
            # Use gRPC by default for better performance
            return True

    def get_protocol_url(self, target_protocol: Protocol) -> str:
        """Get URL for specific protocol with correct port.

        In unified mode, returns the same URL for all protocols since they
        share a single port. In multi-port mode, returns protocol-specific URLs.
        """
        parsed = urlparse(self.url)
        host = parsed.hostname or "localhost"
        scheme = parsed.scheme or "http"

        if target_protocol == Protocol.EMBEDDED:
            return self.url

        # Unified mode: all protocols share the same port
        if self.port_mode == PortMode.UNIFIED:
            # Use the port from the URL, or default unified port (5678)
            port = parsed.port or 5678
            # For gRPC and Arrow Flight, return host:port format (without scheme)
            if target_protocol in (Protocol.GRPC, Protocol.ARROW_FLIGHT):
                return f"{host}:{port}"
            else:
                return f"{scheme}://{host}:{port}"

        # Multi-port mode (legacy): different ports for each protocol
        # Note: In unified mode (default), all protocols share port 5678
        if target_protocol == Protocol.REST:
            port = 5678  # REST API port
        elif target_protocol == Protocol.GRPC:
            port = 5679  # Legacy gRPC API port (use unified mode for single port)
        elif target_protocol == Protocol.ARROW_FLIGHT:
            port = 5680  # Legacy Arrow Flight port (use unified mode for single port)
        else:
            # Keep original port for AUTO or unknown protocols
            port = parsed.port or (443 if scheme == "https" else 80)

        # For gRPC and Arrow Flight, don't include scheme in the endpoint
        if target_protocol in (Protocol.GRPC, Protocol.ARROW_FLIGHT):
            return f"{host}:{port}"
        else:
            return f"{scheme}://{host}:{port}"

    def is_unified_mode(self) -> bool:
        """Check if client is configured for unified port mode."""
        return self.port_mode == PortMode.UNIFIED


# Default configuration instance
DEFAULT_CONFIG = ClientConfig(
    url="http://localhost:5678",
    protocol=Protocol.AUTO,
    timeout=30.0,
)


def load_config(
    url: str | None = None,
    api_key: str | None = None,
    config_file: str | None = None,
    **kwargs,
) -> ClientConfig:
    """Load configuration from multiple sources with precedence:
    1. Explicit parameters
    2. Configuration file
    3. Environment variables
    4. Defaults
    """
    config_dict = {}

    # Load from config file if provided
    if config_file:
        config_dict.update(load_config_file(config_file))

    # Override with explicit parameters
    if url:
        config_dict["url"] = url
    if api_key:
        config_dict["api_key"] = api_key

    config_dict.update(kwargs)

    # Create from environment with overrides
    return ClientConfig.from_env(**config_dict)


def load_config_file(file_path: str) -> dict:
    """Load configuration from file (JSON, YAML, or TOML)"""
    import json
    from pathlib import Path

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {file_path}")

    content = path.read_text()

    if file_path.endswith((".yml", ".yaml")):
        try:
            import yaml

            return yaml.safe_load(content)
        except ImportError:
            raise ImportError("PyYAML is required to load YAML configuration files")

    elif file_path.endswith(".toml"):
        # stdlib since 3.11, which is this package's floor — the `tomli` backport
        # it used to import unconditionally is no longer a dependency.
        import tomllib

        return tomllib.loads(content)

    else:  # JSON
        return json.loads(content)
