"""
ProximaDB Python Client - Synchronous Client

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

# Pre-existing typo in this file used lowercase `callable` (the builtin
# function) instead of `typing.Callable` in several annotations
# (e.g. `Optional[callable]`). That was silently accepted by
# `typing.Optional[…]` (lazy generic). Ruff's PEP 604 modernization
# rewrote them to `callable | None`, which Python evaluates eagerly at
# class-body parse time → `TypeError: unsupported operand types for |`
# because `callable` is a builtin function, not a type. Adding
# `from __future__ import annotations` defers all annotation
# evaluation to string-form, restoring import-time correctness without
# requiring a wholesale `callable` → `Callable` rename.
from __future__ import annotations

import base64
import gzip
import json
import logging
import time
import warnings
from typing import Any

import httpx
import numpy as np
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from ..batching_unified import (
    BatchConfig,
    ThreadedBatchProcessor,
)
from ..cache import ResponseCache
from ..config import ClientConfig, load_config
from ..exceptions import (
    NetworkError,
    ProximaDBError,
    RateLimitError,
    TimeoutError,
    map_http_error,
)
from ..models import (
    BatchResult,
    Collection,
    CollectionConfig,
    CollectionStats,
    DeleteResult,
    FilterDict,
    HealthStatus,
    MetadataDict,
    OperationMetrics,
    ProbeResponse,
    SchemaDefinition,
    SchemaResponse,
    SearchResult,
    ServerCapabilities,
    UpdateSchemaResponse,
    VectorArray,
)
from ..models_v2 import ProximaRecord
from ..proto_conversion import ProtoConverter
from . import _rest_codegen as _gen

logger = logging.getLogger(__name__)


def _unwrap_v2_list(data: Any) -> tuple[list[dict[str, Any]], bool]:
    """Unwrap a v2 list-endpoint ``data`` payload into ``(items, has_more)``.

    The xCatalog v2 graph query endpoints (``/query/nodes``, ``/query/edges``)
    return paginated results as ``{"items": [...], "has_more": bool}`` nested
    under ``data``. Older builds returned ``data`` as a bare list. This tolerates
    both so callers always get a plain list of records.
    """
    if isinstance(data, dict):
        items = data.get("items", [])
        return (list(items) if items else []), bool(data.get("has_more", False))
    if isinstance(data, list):
        return data, False
    return [], False


def _convert_quantization_config_to_proto(quant_config) -> dict[str, Any]:
    """Convert SDK's flat QuantizationConfig to proto's nested structure

    The SDK uses flat fields like bits_per_subvector, num_subvectors, bits_per_vector,
    but the proto expects these nested in a custom_levels array of QuantizationLevel messages.

    Args:
        quant_config: SDK QuantizationConfig object

    Returns:
        Dict with proto-compatible nested structure
    """

    # Start with dict conversion
    try:
        # Pydantic v2
        quant_dict = quant_config.model_dump(exclude_none=True)
    except Exception:
        try:
            # Pydantic v1
            quant_dict = quant_config.dict(exclude_none=True)
        except Exception:
            quant_dict = {}

    # Build proto structure
    proto_dict = {"enabled": quant_dict.get("enabled", False)}

    # Determine strategy based on type
    quant_type = quant_dict.get("type", "NONE")
    if isinstance(quant_type, str):
        quant_type_str = quant_type
    else:
        quant_type_str = getattr(quant_type, "value", "NONE")

    if quant_type_str == "NONE" or not quant_dict.get("enabled"):
        proto_dict["strategy"] = 0  # SMART_DEFAULTS
        proto_dict["custom_levels"] = []
        return proto_dict

    proto_dict["strategy"] = 1  # CUSTOM_LEVELS

    # Map quantization type to proto enum
    QUANT_TYPE_MAP = {"BINARY": 0, "SCALAR": 1, "PRODUCT": 2, "UNIFORM": 3, "NONE": 4}

    # Build custom_levels array from flat SDK fields
    # QuantizationLevel has many fields - provide sensible defaults for all
    level = {
        "level_id": "sdk_level_0",
        "type": QUANT_TYPE_MAP.get(quant_type_str.upper(), 4),  # Default to NONE
        "bits": quant_dict.get("bits_per_subvector")
        or quant_dict.get("bits_per_vector")
        or 8,
        "num_subvectors": quant_dict.get("num_subvectors", 8),
        "adaptive_subvectors": quant_dict.get("adaptive_subvectors", True),
        "scale": quant_dict.get("scale", 1.0),
        "offset": quant_dict.get("offset", 0.0),
        "clamp_values": quant_dict.get("clamp_values", True),
        "threshold": quant_dict.get("threshold", 0.0),
        "sign_based": quant_dict.get("sign_based", True),
        "enable_in_storage": quant_dict.get("enable_in_storage", True),
        "enable_in_index": quant_dict.get("enable_in_index", True),
        "search_priority": quant_dict.get("search_priority", 1),
        "min_recall": quant_dict.get("min_recall", 0.95),
        "enable_validation": quant_dict.get("enable_validation", True),
    }

    # Add other optional top-level fields if present
    if quant_dict.get("accuracy_threshold"):
        proto_dict["binary_filter_selectivity"] = quant_dict["accuracy_threshold"]

    if quant_dict.get("progressive_quantization"):
        proto_dict["enable_progressive_search"] = quant_dict["progressive_quantization"]

    # Set custom_levels as array (proto3 repeated field)
    # Always include the level since we're providing all required fields
    proto_dict["custom_levels"] = [level]

    logger.debug(
        f"Converted quantization config from SDK flat structure to proto nested structure: {proto_dict}"
    )

    return proto_dict


class ProximaDBClient:
    """Synchronous ProximaDB client"""

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        config: ClientConfig | None = None,
        enable_batching: bool = False,
        batch_config: BatchConfig | None = None,
        enable_caching: bool = False,
        cache_config: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Initialize ProximaDB client

        Args:
            url: ProximaDB server URL
            api_key: API key for authentication
            config: Client configuration object
            enable_batching: Enable request batching for improved throughput
            batch_config: Configuration for batching behavior
            enable_caching: Enable response caching for read operations
            cache_config: Configuration for caching behavior
            **kwargs: Additional configuration parameters
        """
        if config is None:
            config = load_config(url=url, api_key=api_key, **kwargs)

        self.config = config
        self._setup_logging()

        # Initialize HTTP client
        self._http_client = self._create_http_client()
        # Capability probes (cached after first attempt)
        self._sks_search_supported: bool | None = None
        self._sks_entities_supported: bool | None = None

        # Initialize request batching if enabled
        self.enable_batching = enable_batching
        self._batch_processor: ThreadedBatchProcessor | None = None

        if enable_batching:
            # Create batch processor for REST operations
            batch_config = batch_config or BatchConfig()
            self._batch_config = batch_config
            # Create the actual batch processor
            self._batch_processor = ThreadedBatchProcessor(
                config=batch_config, execute_batch_fn=self._execute_batch
            )
            self._batch_processor.start()
            logger.info("Batching enabled with config: %s", batch_config)

        # Initialize response caching if enabled
        self.enable_caching = enable_caching
        self._response_cache: ResponseCache | None = None

        if enable_caching:
            self._response_cache = ResponseCache(
                # Use default cache settings if not provided
                default_ttl=(
                    cache_config.get("default_ttl_seconds", 300)
                    if cache_config
                    else 300
                ),
                config=cache_config,  # Store config for test introspection
            )
            logger.info("Enabled response caching for read operations")

        # SKS capability cache and warmup tracking
        self._sks_search_supported: bool | None = None
        self._sks_entities_supported: bool | None = None
        self._warmed_collections: set[str] = set()
        logger.info(f"Initialized ProximaDB client for {self.config.url}")

    def _auto_warmup(self, collection_id: str) -> None:
        """Auto-warmup SKS support once per collection (best-effort)."""
        if not collection_id or collection_id in self._warmed_collections:
            return
        if self._sks_search_supported is None or self._sks_entities_supported is None:
            try:
                self.warmup_sks_capabilities(collection_id)
            except Exception:
                pass
        self._warmed_collections.add(collection_id)

    def get_capabilities(self) -> dict[str, Any]:
        """Return cached capability flags and warmed collections."""
        return {
            "sks_search_supported": self._sks_search_supported,
            "sks_entities_supported": self._sks_entities_supported,
            "warmed_collections": list(self._warmed_collections),
        }

    def server_capabilities(self, refresh: bool = False) -> dict[str, Any]:
        """Negotiate capabilities with the server (GET /api/v2/_meta/capabilities).

        The result (api_version, features, limits, error-envelope contract) is
        cached after the first call. Returns ``{}`` on older servers that don't
        expose the endpoint, so callers can degrade gracefully.
        """
        cached = getattr(self, "_server_capabilities", None)
        if cached is not None and not refresh:
            return cached
        try:
            call = _gen.get_capabilities()
            resp = self._make_request(call.method, call.endpoint)
            caps = resp.json() if hasattr(resp, "json") else {}
            if not isinstance(caps, dict):
                caps = {}
        except Exception:
            caps = {}
        self._server_capabilities = caps  # type: ignore[attr-defined]
        return caps

    def supports(self, feature: str) -> bool:
        """True if the server advertises ``feature`` (via server_capabilities)."""
        return feature in (self.server_capabilities().get("features") or [])

    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        if self.config.enable_debug_logging:
            level = logging.DEBUG
        else:
            # Handle both enum and string log levels
            if hasattr(self.config.log_level, "value"):
                level = getattr(logging, self.config.log_level.value)
            else:
                level = getattr(
                    logging, str(self.config.log_level).upper(), logging.INFO
                )

        logging.getLogger("proximadb").setLevel(level)

    def _create_http_client(self) -> httpx.Client:
        """Create configured HTTP client with compression support"""
        timeout = httpx.Timeout(
            connect=self.config.connection.connect_timeout,
            read=self.config.connection.read_timeout,
            write=self.config.timeout,
            pool=self.config.connection.total_timeout,
        )

        limits = httpx.Limits(
            max_keepalive_connections=self.config.connection.pool_size,
            max_connections=self.config.connection.pool_maxsize,
            keepalive_expiry=self.config.connection.keepalive_timeout,
        )

        uds_path = getattr(self.config, "uds_path", None)
        if uds_path:
            # Portless embedded mode: the server binds a Unix-domain socket and
            # no TCP port at all, so `config.url` is only a host header and the
            # connection has to be made through a UDS transport. TLS and HTTP/2
            # are deliberately not forwarded — neither applies to a local socket,
            # and httpx ignores those Client kwargs once `transport` is given.
            return httpx.Client(
                base_url=self.config.url,
                headers=self.config.get_base_headers(),
                timeout=timeout,
                transport=httpx.HTTPTransport(uds=uds_path, limits=limits),
            )

        return httpx.Client(
            base_url=self.config.url,
            headers=self.config.get_base_headers(),
            timeout=timeout,
            limits=limits,
            verify=self.config.tls.verify,
            cert=(
                (self.config.tls.cert_file, self.config.tls.key_file)
                if self.config.tls.cert_file
                else None
            ),
            http2=self.config.enable_http2,
        )

    def _compress_data(self, data: bytes) -> bytes:
        """Compress data using configured algorithm"""
        algorithm = self.config.compression.algorithm.lower()
        level = self.config.compression.level

        if algorithm == "gzip":
            return gzip.compress(data, compresslevel=level or 6)
        elif algorithm == "deflate":
            import zlib

            return zlib.compress(data, level=level or 6)
        elif algorithm == "zstd":
            try:
                import zstandard

                cctx = zstandard.ZstdCompressor(level=level or 3)
                return cctx.compress(data)
            except ImportError:
                logger.warning("zstd not available, falling back to gzip")
                return gzip.compress(data, compresslevel=6)
        elif algorithm == "br" or algorithm == "brotli":
            try:
                import brotli

                return brotli.compress(data, quality=level or 4)
            except ImportError:
                logger.warning("brotli not available, falling back to gzip")
                return gzip.compress(data, compresslevel=6)
        else:
            # Default to gzip if unknown algorithm
            return gzip.compress(data, compresslevel=6)

    def _make_request(self, method: str, endpoint: str, **kwargs) -> httpx.Response:
        """Make HTTP request with retry logic and optional compression"""

        # Debug log the endpoint
        logger.debug(f"Making {method} request to {endpoint}")

        # Handle request compression if enabled
        # Use rest_enabled for REST protocol
        if (
            hasattr(self.config, "compression")
            and self.config.compression.enabled
            and "json" in kwargs
        ):
            json_data = kwargs.pop("json")
            json_bytes = json.dumps(json_data).encode("utf-8")

            # Debug the payload size
            logger.debug(f"Request payload size: {len(json_bytes)} bytes")

            # Only compress if data is larger than threshold
            if len(json_bytes) > self.config.compression.threshold_bytes:
                compressed_data = self._compress_data(json_bytes)
                kwargs["content"] = compressed_data
                kwargs["headers"] = kwargs.get("headers", {})

                # Set correct Content-Encoding based on algorithm
                algorithm = self.config.compression.algorithm.lower()
                if algorithm == "br" or algorithm == "brotli":
                    kwargs["headers"]["Content-Encoding"] = "br"
                elif algorithm == "deflate":
                    kwargs["headers"]["Content-Encoding"] = "deflate"
                elif algorithm == "zstd":
                    kwargs["headers"]["Content-Encoding"] = "zstd"
                else:  # default to gzip
                    kwargs["headers"]["Content-Encoding"] = "gzip"

                kwargs["headers"]["Content-Type"] = "application/json"

                logger.debug(
                    f"Compressed request: {len(json_bytes)} -> {len(compressed_data)} bytes "
                    f"({100 * (1 - len(compressed_data) / len(json_bytes)):.1f}% reduction)"
                )
            else:
                # Data too small to benefit from compression
                kwargs["json"] = json_data

        @retry(
            stop=stop_after_attempt(self.config.retry.max_retries + 1),
            wait=wait_exponential(
                multiplier=self.config.retry.backoff_factor,
                max=self.config.retry.max_backoff,
            ),
            retry=retry_if_exception_type((NetworkError, TimeoutError, RateLimitError)),
            reraise=True,
        )
        def _request():
            try:
                response = self._http_client.request(method, endpoint, **kwargs)

                if response.status_code >= 400:
                    self._handle_error_response(response)

                return response

            except httpx.TimeoutException as e:
                raise TimeoutError(
                    f"Request timeout: {e}", timeout_seconds=self.config.timeout
                )
            except httpx.NetworkError as e:
                raise NetworkError(f"Network error: {e}", original_error=e)
            except httpx.HTTPStatusError as e:
                self._handle_error_response(e.response)

        return _request()

    def _handle_error_response(self, response: httpx.Response) -> None:
        """Handle error responses from server"""
        try:
            error_data = response.json()
        except Exception:
            error_data = {
                "message": response.text or f"HTTP {response.status_code} error"
            }

        # DEBUG: Log error details (incl. the server correlation id)
        request_id = response.headers.get("x-request-id")
        logger.error(
            f"❌ HTTP {response.status_code} ERROR - URL: {response.url} "
            f"request_id={request_id}"
        )
        logger.error(f"❌ ERROR DATA: {error_data}")

        raise map_http_error(response.status_code, error_data, headers=response.headers)

    def _http_post(self, endpoint: str, data: Any) -> dict[str, Any]:
        """Helper method for POST requests"""
        response = self._make_request("POST", endpoint, json=data)
        return response.json()

    def _normalize_vectors(self, vectors: VectorArray) -> list[list[float]]:
        """Normalize vectors to list of lists format"""
        if isinstance(vectors, np.ndarray):
            if vectors.dtype != np.float32:
                vectors = vectors.astype(np.float32)
            return vectors.tolist()
        return vectors

    def _validate_vector_dimensions(
        self, vectors: VectorArray, expected_dim: int | None = None
    ) -> None:
        """Validate vector dimensions"""
        if not self.config.validate_inputs:
            return

        if isinstance(vectors, np.ndarray):
            if vectors.ndim != 2:
                raise ValueError("Vector array must be 2-dimensional")
            actual_dim = vectors.shape[1]
        elif isinstance(vectors, list) and vectors:
            actual_dim = len(vectors[0]) if vectors[0] else 0
        else:
            return

        if expected_dim is not None and actual_dim != expected_dim:
            from ..exceptions import VectorDimensionError

            raise VectorDimensionError(expected_dim, actual_dim)

    def health(self) -> HealthStatus:
        """Check server health status"""
        call = _gen.get_health()
        response = self._make_request(call.method, call.endpoint)
        data = response.json()

        # Transform response to match HealthStatus model
        # Server returns: timestamp (seconds), components
        # Model expects: timestamp_ms (milliseconds), services

        # Handle nested response structure
        if "data" in data and isinstance(data["data"], dict):
            health_data = data["data"]
        else:
            health_data = data

        # Convert timestamp to milliseconds if present
        timestamp_ms = health_data.get("timestamp_ms")
        if timestamp_ms is None and "timestamp" in health_data:
            timestamp_ms = (
                health_data["timestamp"] * 1000
            )  # Convert seconds to milliseconds
        elif timestamp_ms is None:
            timestamp_ms = int(time.time() * 1000)  # Current timestamp in milliseconds

        # Map components to services
        services = health_data.get("services", health_data.get("components", {}))
        if services is None:
            services = {}

        return HealthStatus(
            status=health_data.get("status", "unknown"),
            version=health_data.get("version", "0.0.0"),
            uptime_seconds=health_data.get("uptime_seconds", 0),
            services=services,
            timestamp_ms=timestamp_ms,
        )

    def live(self) -> ProbeResponse:
        """Kubernetes liveness probe — GET /health/live."""
        call = _gen.get_liveness()
        response = self._make_request(call.method, call.endpoint)
        return ProbeResponse.model_validate(response.json())

    def ready(self) -> ProbeResponse:
        """Kubernetes readiness probe — GET /health/ready."""
        call = _gen.get_readiness()
        response = self._make_request(call.method, call.endpoint)
        return ProbeResponse.model_validate(response.json())

    def create_collection(
        self,
        name: str,
        config: CollectionConfig | None = None,
        *,
        schema: SchemaDefinition | dict[str, Any] | None = None,
        initial_capacity: int | None = None,
        **kwargs,
    ) -> Collection:
        """Create a new vector collection

        Args:
            name: Collection name
            config: Collection configuration
            schema: Optional SchemaDefinition (OpenAPI CreateCollectionRequest.schema)
            initial_capacity: Optional initial capacity hint
            **kwargs: Additional configuration parameters

        Returns:
            Created collection metadata

        Example:
            >>> client = ProximaDBClient()
            >>> config = CollectionConfig(
            ...     dimension=128,
            ...     distance_metric="cosine",
            ...     filterable_metadata_fields=["category", "price", "brand"]
            ... )
            >>> collection = client.create_collection("products", config)
        """
        if config is None:
            # Include name in kwargs for config creation
            config = CollectionConfig(name=name, **kwargs)

        # Validate filterable metadata fields limit in client
        if (
            config.filterable_metadata_fields
            and len(config.filterable_metadata_fields) > 16
        ):
            warnings.warn(
                f"Collection '{name}' specifies {len(config.filterable_metadata_fields)} filterable metadata fields. "
                f"Only the first 16 will be used for Parquet optimization. Additional metadata can still be "
                f"inserted via vector operations (stored in extra_meta).",
                UserWarning,
            )

        # Inform users about potential server fallbacks for better UX
        fallback_warnings = []

        if config.distance_metric:
            metric_str = (
                config.distance_metric
                if isinstance(config.distance_metric, str)
                else config.distance_metric.value
            )
            if not ServerCapabilities.is_supported("distance_metric", metric_str):
                fallback = ServerCapabilities.get_fallback_for(
                    "distance_metric", metric_str
                )
                if fallback:
                    fallback_warnings.append(
                        f"💡 Distance metric '{metric_str}' will fallback to '{fallback}' (server decision). "
                        f"For guaranteed support, use: {', '.join(ServerCapabilities().supported_distance_metrics)}"
                    )

        if config.storage_engine:
            engine_str = (
                config.storage_engine
                if isinstance(config.storage_engine, str)
                else config.storage_engine.value
            )
            if not ServerCapabilities.is_supported("storage_engine", engine_str):
                fallback = ServerCapabilities.get_fallback_for(
                    "storage_engine", engine_str
                )
                if fallback:
                    fallback_warnings.append(
                        f"💡 Storage engine '{engine_str}' will fallback to '{fallback}' (server decision). "
                        f"For guaranteed support, use: {', '.join(ServerCapabilities().supported_storage_engines)}"
                    )

        if config.index_configs:
            for ic in config.index_configs or []:
                algo_str = ProtoConverter.index_type_to_str(ic.algorithm)
                if not ServerCapabilities.is_supported("indexing_algorithm", algo_str):
                    fallback = ServerCapabilities.get_fallback_for(
                        "indexing_algorithm", algo_str
                    )
                    if fallback:
                        fallback_warnings.append(
                            f"💡 Indexing algorithm '{algo_str}' will fallback to '{fallback}' (server decision). "
                            f"For guaranteed support, use: {', '.join(ServerCapabilities().supported_indexing_algorithms)}"
                        )

        # Issue all fallback warnings at once for better UX
        if fallback_warnings:
            combined_message = (
                f"ProximaDB server will make intelligent fallback decisions for collection '{name}':\n"
                + "\n".join(f"  • {warning}" for warning in fallback_warnings)
                + "\n\n📚 Server uses smart defaults to ensure your collection works. "
                "Check the returned collection config to see final server decisions."
            )
            warnings.warn(combined_message, UserWarning)

        # Enum mappings for proto values
        DISTANCE_METRIC_MAP = {
            "cosine": 1,
            "euclidean": 2,
            "dot_product": 3,
            "hamming": 4,
            "manhattan": 5,
            "jaccard": 6,
            "angular": 7,
            "chebyshev": 8,
            "canberra": 9,
            "minkowski": 10,
            "bray_curtis": 11,
            "hellinger": 12,
            "custom": 13,
        }
        STORAGE_ENGINE_MAP = {
            "viper": 1,
            "sst": 2,
            "nova": 3,
            "helix": 4,
            "swift": 5,
            "raptor": 6,
            "mmap": 7,
            "hybrid": 8,
        }

        # Convert distance_metric to integer
        distance_metric_str = (
            config.distance_metric
            if isinstance(config.distance_metric, str)
            else getattr(config.distance_metric, "value", "cosine")
        )
        distance_metric_int = DISTANCE_METRIC_MAP.get(
            distance_metric_str.lower(), 1
        )  # Default to COSINE

        # Convert storage_engine to integer
        storage_engine_str = getattr(config, "storage_engine", "sst")  # Default to SST
        if not isinstance(storage_engine_str, str):
            storage_engine_str = getattr(storage_engine_str, "value", "sst")
        storage_engine_int = STORAGE_ENGINE_MAP.get(
            storage_engine_str.lower(), 2
        )  # Default to SST=2

        # Build config object as expected by server (all required proto fields)
        config_data: dict[str, Any] = {
            "name": name,
            "dimension": config.dimension,
            "distance_metric": distance_metric_int,
            "storage_engine": storage_engine_int,
            "tags": [],  # Required
            "filterable_columns": [],  # Required (will be populated below if needed)
            "index_configs": [],  # Required (will be populated below)
            "primary_index": "",  # Required (will be set below)
            "auto_index_selection": True,  # Required
            "embedding_models": [],  # Required
        }

        # Add optional description if provided
        if config.description:
            config_data["description"] = config.description

        # Enum mapping for indexing algorithms
        INDEXING_ALGORITHM_MAP = {
            "hnsw": 1,
            "ivf": 2,
            "pq": 3,
            "flat": 4,
            "annoy": 5,
            "lsh": 6,
        }

        # Build index_configs aligned with proto
        index_configs: list[dict[str, Any]] = []
        primary_index_name: str | None = None
        if getattr(config, "index_configs", None):
            for ic in config.index_configs or []:
                algo_str = ProtoConverter.index_type_to_str(ic.algorithm)
                algo_int = INDEXING_ALGORITHM_MAP.get(
                    algo_str.lower(), 1
                )  # Default to HNSW
                entry: dict[str, Any] = {
                    "index_name": ic.index_name,
                    "algorithm": algo_int,
                    "parameters": {},
                    "enabled": True,
                    "update_mode": 0,
                    "enable_background_optimization": True,
                    "build_concurrency": 4,
                    "memory_limit_mb": 512,
                    "checkpoint_interval_ms": 60000,
                    "is_primary": bool(getattr(ic, "is_primary", False)),
                    "use_cases": [],
                    "selectivity_threshold": 0.5,
                    "use_quantization": False,
                    "queue_representation": "vector",
                }
                if entry["is_primary"]:
                    primary_index_name = ic.index_name
                index_configs.append(entry)
        else:
            # Default to a primary HNSW index when none provided
            primary_index_name = f"{name}_primary"
            index_configs.append(
                {
                    "index_name": primary_index_name,
                    "algorithm": 1,  # HNSW enum value
                    "parameters": {},
                    "enabled": True,
                    "update_mode": 0,
                    "enable_background_optimization": True,
                    "build_concurrency": 4,
                    "memory_limit_mb": 512,
                    "checkpoint_interval_ms": 60000,
                    "is_primary": True,
                    "use_cases": [],
                    "selectivity_threshold": 0.5,
                    "use_quantization": False,
                    "queue_representation": "vector",
                }
            )

        config_data["index_configs"] = index_configs
        if primary_index_name:
            config_data["primary_index"] = primary_index_name

        # Quantization (check both quantization and quantization_config for compatibility)
        quant = getattr(config, "quantization_config", None) or getattr(
            config, "quantization", None
        )
        if quant:
            # Convert SDK's flat structure to proto's nested custom_levels structure
            config_data["quantization"] = _convert_quantization_config_to_proto(quant)

        request_data = {
            "operation": 1,  # COLLECTION_CREATE enum value
            "collection_config": config_data,
            "query_params": {},  # Required map field
            "options": {},  # Required map field
            "migration_config": {},  # Required map field
        }

        # Debug print
        logger.debug(f"Collection create request: {request_data}")

        # Add VIPER-specific optimization fields
        if config.filterable_metadata_fields:
            config_data["filterable_columns"] = [
                {"name": field, "data_type": "string", "indexed": True}
                for field in config.filterable_metadata_fields
            ]

        # Add filterable_columns if directly specified (higher priority)
        if config.filterable_columns:
            # Map FilterableDataType to proto integer values
            FILTERABLE_DATATYPE_MAP = {
                "string": 1,
                "integer": 2,
                "float": 3,
                "boolean": 4,
                "datetime": 5,
                "array_string": 6,
                "array_integer": 7,
                "array_float": 8,
            }

            # Serialize FilterableColumn objects to dicts with proto integer data_type
            config_data["filterable_columns"] = [
                {
                    "name": col.name,
                    "data_type": FILTERABLE_DATATYPE_MAP.get(
                        (
                            col.data_type
                            if isinstance(col.data_type, str)
                            else col.data_type.value
                        ).lower(),
                        1,  # Default to STRING
                    ),
                    "indexed": col.indexed,
                    "supports_range": (
                        col.supports_range if hasattr(col, "supports_range") else False
                    ),
                    "estimated_cardinality": (
                        col.estimated_cardinality
                        if hasattr(col, "estimated_cardinality")
                        and col.estimated_cardinality
                        else None
                    ),
                }
                for col in config.filterable_columns
            ]

        # Add WAL flush configuration
        if hasattr(config, "flush_config") and config.flush_config:
            if hasattr(config.flush_config, "max_wal_size_mb"):
                config_data["max_wal_size_mb"] = config.flush_config.max_wal_size_mb

        if config.description:
            config_data["description"] = config.description

        v2_request_data: dict[str, Any] = {
            "name": name,
            "dimension": config.dimension,
            "engine": getattr(config.storage_engine, "value", config.storage_engine),
            "distance_metric": getattr(
                config.distance_metric, "value", config.distance_metric
            ),
            "enable_proxima_record": True,
        }
        precision = getattr(config, "canonical_embedding_precision", None)
        if precision is not None:
            # EmbeddingPrecision is a str-Enum; send the canonical lower-case
            # string (server normalises aliases via the same dispatch the
            # pgwire / Arrow Flight surfaces use).
            v2_request_data["canonical_embedding_precision"] = getattr(
                precision, "value", precision
            )
        if schema is not None:
            if isinstance(schema, SchemaDefinition):
                v2_request_data["schema"] = schema.model_dump(
                    exclude_none=True, by_alias=True
                )
            else:
                v2_request_data["schema"] = schema
        if initial_capacity is not None:
            v2_request_data["initial_capacity"] = initial_capacity
        # Spec-driven request shape/path via the generated REST transport
        # (TD-126 Phase 4); the HTTP call still goes through _make_request.
        call = _gen.create_collection(v2_request_data)
        response = self._make_request(call.method, call.endpoint, json=call.json)
        response_data = response.json()

        # Handle unified API response format
        if "collection_id" in response_data:
            return Collection(
                id=response_data.get("collection_id", name),
                config=CollectionConfig(
                    name=response_data.get("name", name),
                    dimension=response_data.get("dimension", config.dimension),
                    distance_metric=v2_request_data.get("distance_metric") or "cosine",
                    storage_engine=response_data.get("engine") or "auto",
                ),
                created_at=response_data.get("created_at"),
            )
        elif "collection" in response_data:
            coll_data = response_data["collection"]
            if not coll_data:
                if "error" in response_data:
                    raise ProximaDBError(
                        f"Collection creation failed: {response_data['error']}"
                    )
                raise ProximaDBError(f"Unexpected response format: {response_data}")

            # Build config from response if present; otherwise fallback to request config
            cfg_src = coll_data.get("config", {})

            # Map Proto enum integers to string values for Pydantic
            # Server returns Proto enums as integers
            # NOTE: These MUST match the proto enum values from proximadb.proto
            # DistanceMetric: COSINE=1, EUCLIDEAN=2, DOT_PRODUCT=3, HAMMING=4, MANHATTAN=5, JACCARD=6, ...
            DISTANCE_METRIC_MAP = {
                1: "cosine",
                2: "euclidean",
                3: "dot_product",
                4: "hamming",
                5: "manhattan",
                6: "jaccard",
                7: "chebyshev",
                8: "canberra",
                9: "minkowski",
                10: "angular",
                11: "bray_curtis",
                12: "hellinger",
                13: "custom",
            }
            STORAGE_ENGINE_MAP = {
                # NOTE: These MUST match the proto enum values from proximadb.proto
                # StorageEngine: VIPER=1, SST=2, NOVA=3, HELIX=4, SWIFT=5, RAPTOR=6, MMAP=7, HYBRID=8
                1: "viper",
                2: "sst",
                3: "nova",
                4: "helix",
                5: "swift",
                6: "raptor",
                7: "mmap",
                8: "hybrid",
            }

            # Extract and convert distance_metric
            dm_val = cfg_src.get("distance_metric")
            if isinstance(dm_val, int):
                distance_metric = DISTANCE_METRIC_MAP.get(dm_val, "cosine")
            elif dm_val:
                distance_metric = dm_val
            else:
                dm = getattr(config, "distance_metric", "cosine")
                distance_metric = dm.value if hasattr(dm, "value") else dm

            # Extract and convert storage_engine
            se_val = cfg_src.get("storage_engine")
            if isinstance(se_val, int):
                storage_engine = STORAGE_ENGINE_MAP.get(se_val, "viper")
            elif se_val:
                storage_engine = se_val
            else:
                se = getattr(config, "storage_engine", "viper")
                storage_engine = se.value if hasattr(se, "value") else se

            # Deserialize filterable_columns from server response
            filterable_columns = None
            if "filterable_columns" in cfg_src and cfg_src["filterable_columns"]:
                from proximadb_sdk.models import FilterableColumn

                # Map proto integer values to FilterableDataType strings
                FILTERABLE_DATATYPE_REVERSE_MAP = {
                    0: "string",
                    1: "string",
                    2: "integer",
                    3: "float",
                    4: "boolean",
                    5: "datetime",
                    6: "array_string",
                    7: "array_integer",
                    8: "array_float",
                }

                filterable_columns = [
                    FilterableColumn(
                        name=col["name"],
                        data_type=FILTERABLE_DATATYPE_REVERSE_MAP.get(
                            (
                                col.get("data_type")
                                if isinstance(col.get("data_type"), int)
                                else 1
                            ),
                            "string",
                        ),
                        indexed=col.get("indexed", True),
                        supports_range=col.get("supports_range", False),
                        estimated_cardinality=col.get("estimated_cardinality"),
                    )
                    for col in cfg_src["filterable_columns"]
                ]

            cfg = CollectionConfig(
                name=cfg_src.get("name", name),
                dimension=cfg_src.get("dimension", config.dimension),
                distance_metric=distance_metric,
                storage_engine=storage_engine,
                filterable_columns=filterable_columns,
            )
            return Collection(
                id=coll_data.get("id", name),
                config=cfg,
                created_at=coll_data.get("created_at"),
                updated_at=coll_data.get("updated_at"),
            )
        else:
            # Fallback for other response formats
            return Collection(**response_data)

    def get_collection(self, collection_id: str) -> Collection:
        """Get collection metadata"""
        # Use the updated GET endpoint
        logger.debug(
            f"Collection get request to GET /api/v2/collections/{collection_id}"
        )
        call = _gen.get_collection(collection_id)
        response = self._make_request(call.method, call.endpoint)
        response_data = response.json()

        # Check for error responses
        if isinstance(response_data, dict):
            if response_data.get("error_message") or response_data.get("error"):
                error_msg = response_data.get("error_message") or response_data.get(
                    "error"
                )
                if isinstance(error_msg, str) and (
                    "not found" in error_msg.lower()
                    or "does not exist" in error_msg.lower()
                ):
                    raise CollectionNotFoundError(
                        f"Collection '{collection_id}' not found"
                    )
                raise ProximaDBError(f"Failed to get collection: {error_msg}")

            # If success is explicitly false, treat as error
            if "success" in response_data and response_data["success"] is False:
                error_msg = response_data.get("error_message", "Unknown error")
                if "not found" in str(error_msg).lower():
                    raise CollectionNotFoundError(
                        f"Collection '{collection_id}' not found"
                    )
                raise ProximaDBError(f"Failed to get collection: {error_msg}")

        # Server returns simplified format:
        # {
        #   "id": "uuid",
        #   "name": "collection_name",
        #   "dimension": 768,
        #   "metric": "cosine",
        #   "created_at": 1737567890123,
        #   "updated_at": 1737567890123,
        #   "vector_count": 1000,
        #   "indexed": true
        # }

        # Handle nested response structure - collection data may be in "collection" field
        if "collection" in response_data and isinstance(
            response_data["collection"], dict
        ):
            collection_data = response_data["collection"]
        else:
            collection_data = response_data

        # Debug: Check if required fields are present
        collection_id_from_response = collection_data.get("id", collection_id)

        # Handle nested config structure (server returns dimension in config.dimension)
        if "dimension" not in collection_data and "config" in collection_data:
            if isinstance(collection_data["config"], dict):
                # Server returns nested config structure - extract dimension from config
                logger.debug("Extracting dimension from nested config structure")
                collection_data = collection_data["config"]

        # Convert proto enum values to string names if needed
        distance_metric_value = collection_data.get(
            "metric", collection_data.get("distance_metric", "cosine")
        )
        if isinstance(distance_metric_value, int):
            # Map proto enum values to string names
            distance_metric_map = {
                1: "cosine",
                2: "euclidean",
                3: "dot_product",
                4: "manhattan",
                5: "hamming",
            }
            distance_metric_value = distance_metric_map.get(
                distance_metric_value, "cosine"
            )

        storage_engine_value = collection_data.get("storage_engine", "viper")
        if isinstance(storage_engine_value, int):
            # Map proto enum values to string names
            storage_engine_map = {
                1: "viper",
                2: "sst",
                3: "nova",
                4: "helix",
                5: "swift",
                6: "raptor",
            }
            storage_engine_value = storage_engine_map.get(storage_engine_value, "viper")

        # Create CollectionConfig from response
        # Note: name might not be in config if it's at the parent level
        collection_name = collection_data.get(
            "name", collection_data.get("collection_name")
        )
        if not collection_name or len(collection_name) < 8:
            # Fallback: use collection_id but pad it to meet minimum length
            collection_name = (
                collection_id
                if len(collection_id) >= 8
                else f"collection_{collection_id}"
            )

        config = CollectionConfig(
            name=collection_name,
            dimension=collection_data.get(
                "dimension", 128
            ),  # Use reasonable default if missing
            distance_metric=distance_metric_value,
            storage_engine=storage_engine_value,
            primary_indexing_algorithm=None,
        )

        # Create CollectionStats if available
        stats = CollectionStats(
            vector_count=collection_data.get("vector_count", 0),
            index_size_bytes=collection_data.get("index_size_bytes", 0),
            data_size_bytes=collection_data.get("data_size_bytes", 0),
        )

        return Collection(
            id=collection_id_from_response,
            config=config,
            stats=stats,
            created_at=collection_data.get("created_at"),
            updated_at=collection_data.get("updated_at"),
        )

    def list_collections(
        self,
        limit: int | None = None,
        offset: int | None = None,
        include_stats: bool | None = None,
    ) -> list[Collection]:
        """List collections.

        Args:
            limit: Page size (OpenAPI default 100).
            offset: Page offset (OpenAPI default 0).
            include_stats: Include CollectionStats in each entry.
        """
        logger.debug("Collection list request to GET /api/v2/collections")
        call = _gen.list_collections(
            limit=limit, offset=offset, include_stats=include_stats
        )
        request_kwargs: dict[str, Any] = {}
        if call.params:
            request_kwargs["params"] = call.params
        response = self._make_request(call.method, call.endpoint, **request_kwargs)
        response_data = response.json()

        # Server returns:
        # {
        #   "collections": [
        #     {
        #       "id": "uuid",
        #       "name": "collection_name",
        #       "dimension": 768,
        #       "metric": "cosine",
        #       "created_at": 1737567890123,
        #       "updated_at": 1737567890123,
        #       "vector_count": 1000,
        #       "indexed": true
        #     },
        #     ...
        #   ],
        #   "total_count": 2
        # }

        collections = []
        collections_data = response_data.get("collections", [])

        for coll_data in collections_data:
            # Extract config - it can be nested or flat
            cfg_src = coll_data.get("config", coll_data)

            # Map Proto enum integers to strings (same as create_collection)
            DISTANCE_METRIC_MAP = {
                0: "cosine",
                1: "cosine",
                2: "euclidean",
                3: "dot_product",
                4: "manhattan",
                5: "hamming",
                6: "jaccard",
                7: "chebyshev",
                8: "canberra",
                9: "minkowski",
                10: "angular",
                11: "bray_curtis",
                12: "hellinger",
                13: "custom",
            }
            STORAGE_ENGINE_MAP = {
                # NOTE: These MUST match the proto enum values from proximadb.proto
                # StorageEngine: VIPER=1, SST=2, NOVA=3, HELIX=4, SWIFT=5, RAPTOR=6, MMAP=7, HYBRID=8
                1: "viper",
                2: "sst",
                3: "nova",
                4: "helix",
                5: "swift",
                6: "raptor",
                7: "mmap",
                8: "hybrid",
            }

            dm_val = cfg_src.get("distance_metric", cfg_src.get("metric"))
            distance_metric = (
                DISTANCE_METRIC_MAP.get(dm_val, "cosine")
                if isinstance(dm_val, int)
                else (dm_val or "cosine")
            )

            se_val = cfg_src.get("storage_engine")
            storage_engine = (
                STORAGE_ENGINE_MAP.get(se_val, "viper")
                if isinstance(se_val, int)
                else (se_val or "viper")
            )

            # Create CollectionConfig from response
            config = CollectionConfig(
                name=cfg_src.get("name", coll_data.get("name", "")),
                dimension=cfg_src.get("dimension", coll_data.get("dimension", 0)),
                distance_metric=distance_metric,
                storage_engine=storage_engine,
                primary_indexing_algorithm=None,
            )

            # Create CollectionStats if available
            stats_src = coll_data.get("stats", coll_data)
            stats = CollectionStats(
                vector_count=stats_src.get("vector_count", 0),
                index_size_bytes=stats_src.get("index_size_bytes", 0),
                data_size_bytes=stats_src.get("data_size_bytes", 0),
            )

            collections.append(
                Collection(
                    id=coll_data["id"],
                    config=config,
                    stats=stats,
                    created_at=coll_data.get("created_at"),
                    updated_at=coll_data.get("updated_at"),
                )
            )

        return collections

    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection"""
        # Use standard REST DELETE endpoint
        logger.debug(
            f"Collection delete request to DELETE /api/v2/collections/{collection_id}"
        )
        call = _gen.delete_collection(collection_id)
        response = self._make_request(call.method, call.endpoint)
        return response.json().get("success", False)

    def get_schema(self, collection_id: str) -> SchemaResponse:
        """Get a collection's schema — GET /api/v2/collections/{id}/schema."""
        response = self._make_request(
            "GET", f"/api/v2/collections/{collection_id}/schema"
        )
        return SchemaResponse.model_validate(response.json())

    def update_schema(
        self,
        collection_id: str,
        schema: SchemaDefinition | dict[str, Any],
        *,
        force: bool = False,
    ) -> UpdateSchemaResponse:
        """Update a collection's schema — PUT /api/v2/collections/{id}/schema.

        The body conforms to OpenAPI UpdateSchemaRequest, which extends
        SchemaDefinition with an optional `force` flag.
        """
        if isinstance(schema, SchemaDefinition):
            body = schema.model_dump(exclude_none=True, by_alias=True)
        else:
            body = dict(schema)
        body["force"] = bool(force)
        response = self._make_request(
            "PUT", f"/api/v2/collections/{collection_id}/schema", json=body
        )
        return UpdateSchemaResponse.model_validate(response.json())

    def get_collection_stats(self, collection_id: str) -> CollectionStats:
        """Get collection statistics"""
        response = self._make_request("GET", f"/collections/{collection_id}/stats")
        return CollectionStats(**response.json())

    def insert_vector(
        self,
        collection_id: str,
        vector_id: str,
        vector: list[float] | np.ndarray,
        metadata: MetadataDict | None = None,
        upsert: bool = False,
    ) -> BatchResult:
        """Insert a single vector

        Args:
            collection_id: Target collection ID
            vector_id: Unique vector identifier
            vector: Vector data
            metadata: Optional metadata dictionary
            upsert: Update if vector already exists

        Returns:
            Insert operation result
        """
        return self.insert_records(
            collection_id,
            [
                {
                    "id": vector_id,
                    "vector": vector,
                    "props": metadata or {},
                }
            ],
            upsert=upsert,
        )

    def _record_write_result(self, response_data: dict[str, Any]) -> BatchResult:
        """Convert v2 record batch response JSON to the SDK batch result."""
        inserted_count = int(response_data.get("inserted_count", 0) or 0)
        failed_count = int(response_data.get("failed_count", 0) or 0)
        total_count = inserted_count + failed_count
        return BatchResult(
            total=total_count,
            success=inserted_count,
            failed=failed_count,
            errors=response_data.get("errors", []),
            duration_ms=response_data.get("duration_ms", 0.0),
            metrics=OperationMetrics(
                total_processed=total_count,
                successful_count=inserted_count,
                failed_count=failed_count,
                processing_time_us=int(response_data.get("duration_ms", 0) * 1000),
            ),
        )

    def _proxima_rest_value(self, value: Any) -> Any:
        """Convert Python values to v2 REST ProximaValue JSON."""
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, (bytes, bytearray, memoryview)):
            return {
                "type": "binary",
                "value": base64.b64encode(bytes(value)).decode("ascii"),
            }
        if isinstance(value, tuple):
            return {
                "type": "array",
                "value": [self._proxima_rest_value(v) for v in value],
            }
        if isinstance(value, dict):
            if set(value.keys()) == {"type", "value"}:
                return value
            return {
                "type": "jsonb",
                "value": {str(k): self._json_scalar(v) for k, v in value.items()},
            }
        return self._json_scalar(value)

    def _json_scalar(self, value: Any) -> Any:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    def _normalize_record_payload(self, record: Any, index: int = 0) -> dict[str, Any]:
        """Normalize SDK record-like inputs to the v2 REST ProximaRecord shape."""
        if hasattr(record, "model_dump"):
            record = record.model_dump(exclude_none=True)
        elif hasattr(record, "dict"):
            record = record.dict(exclude_none=True)

        if not isinstance(record, dict):
            if hasattr(record, "id") and hasattr(record, "vector"):
                record = {
                    "id": getattr(record, "id", None),
                    "vector": record.vector,
                    "props": getattr(record, "metadata", {}) or {},
                }
            else:
                raise TypeError(f"Unsupported record input: {type(record)!r}")

        vector = record.get("vector")
        if vector is None and "embeddings" in record and record["embeddings"]:
            first = record["embeddings"][0]
            vector = first.get("values") if isinstance(first, dict) else first
        if vector is None:
            raise ValueError("record is missing vector")
        if isinstance(vector, np.ndarray):
            vector = vector.astype(np.float32, copy=False).tolist()
        vector = [float(v) for v in vector]
        if not vector:
            raise ValueError("record vector cannot be empty")

        props = {}
        for prop_source in ("props", "metadata", "flexible_fields"):
            values = record.get(prop_source)
            if isinstance(values, dict):
                props.update(
                    {str(k): self._proxima_rest_value(v) for k, v in values.items()}
                )
        typed_fields = record.get("typed_fields")
        if isinstance(typed_fields, dict):
            for key, typed_value in typed_fields.items():
                if hasattr(typed_value, "model_dump"):
                    typed_value = typed_value.model_dump(exclude_none=True)
                if hasattr(typed_value, "dict"):
                    typed_value = typed_value.dict(exclude_none=True)
                if isinstance(typed_value, dict):
                    props[str(key)] = {
                        "type": typed_value.get("value_type")
                        or typed_value.get("type")
                        or typed_value.get("declared_type"),
                        "value": typed_value.get("value"),
                    }
                else:
                    props[str(key)] = self._proxima_rest_value(typed_value)

        text_fields = record.get("text_fields") or []
        normalized = {
            "id": record.get("id") or record.get("oid") or f"record_{index}",
            "vector": vector,
            "props": props,
        }
        if text_fields:
            normalized["text_fields"] = text_fields
        return normalized

    def insert_records(
        self,
        collection_id: str,
        records: list[ProximaRecord | dict[str, Any]],
        *,
        upsert: bool = False,
        batch_size: int | None = None,
        validate_schema: bool = True,
    ) -> BatchResult:
        """Insert record batches through the v2 ProximaRecord REST surface."""
        self._auto_warmup(collection_id)
        record_data = [
            self._normalize_record_payload(record, index)
            for index, record in enumerate(records)
        ]
        effective_batch_size = batch_size or self.config.default_batch_size

        total_successful = 0
        total_failed = 0
        all_errors = []
        for start in range(0, len(record_data), effective_batch_size):
            batch_data = record_data[start : start + effective_batch_size]
            call = _gen.insert_records(
                collection_id,
                {
                    "records": batch_data,
                    "validate_schema": validate_schema,
                    "upsert": upsert,
                },
            )
            response = self._make_request(call.method, call.endpoint, json=call.json)
            result = self._record_write_result(response.json())
            total_successful += result.success
            total_failed += result.failed
            all_errors.extend(result.errors or [])

        if total_successful > 0:
            self._invalidate_collection_cache(collection_id)

        return BatchResult(
            total=total_successful + total_failed,
            success=total_successful,
            failed=total_failed,
            errors=all_errors,
            duration_ms=0.0,
            metrics=OperationMetrics(
                total_processed=total_successful + total_failed,
                successful_count=total_successful,
                failed_count=total_failed,
            ),
        )

    def upsert_records(
        self,
        collection_id: str,
        records: list[ProximaRecord | dict[str, Any]],
        *,
        batch_size: int | None = None,
        validate_schema: bool = True,
    ) -> BatchResult:
        """Upsert records through the canonical record batch surface."""
        return self.insert_records(
            collection_id,
            records,
            upsert=True,
            batch_size=batch_size,
            validate_schema=validate_schema,
        )

    def _convert_metadata_to_rest_format(
        self, metadata_dict: dict[str, Any]
    ) -> dict[str, dict[str, Any]]:
        """Convert Python dict metadata to REST API SqlValue format

        The server expects metadata as a dict of SqlValues:
        {
            "key1": {"string_value": "value"},
            "key2": {"int64_value": 42},
            "key3": {"bool_value": true}
        }
        """
        if not metadata_dict:
            return {}

        sql_metadata = {
            str(key): self._convert_value_to_rest_sql_value(value)
            for key, value in metadata_dict.items()
        }

        return sql_metadata

    def _convert_value_to_rest_sql_value(self, value: Any) -> dict[str, Any]:
        """Convert Python values to the REST JSON shape for v1 SqlValue."""
        if value is None:
            return {"null_value": None}
        if isinstance(value, bool):
            return {"bool_value": value}
        if isinstance(value, int) and not isinstance(value, bool):
            return {"int64_value": value}
        if isinstance(value, float):
            return {"number_value": value}
        if isinstance(value, str):
            return {"string_value": value}
        if isinstance(value, (bytes, bytearray, memoryview)):
            return {"bytes_value": base64.b64encode(bytes(value)).decode("ascii")}
        if isinstance(value, (list, tuple)):
            return {
                "array_value": {
                    "values": [
                        self._convert_value_to_rest_sql_value(item) for item in value
                    ]
                }
            }
        if isinstance(value, dict):
            return {
                "object_value": {
                    "fields": {
                        str(key): self._convert_value_to_rest_sql_value(item)
                        for key, item in value.items()
                    }
                }
            }
        return {"string_value": str(value)}

    def insert_vectors(
        self,
        collection_id: str,
        vectors: VectorArray | list[dict[str, Any]],  # Accept VectorRecord dicts
        ids: list[str] | None = None,
        metadata: list[MetadataDict] | None = None,
        upsert: bool = False,
        batch_size: int | None = None,
        vector_records: (
            list[dict[str, Any]] | None
        ) = None,  # NEW: Full VectorRecord dicts
    ) -> BatchResult:
        """Insert multiple vectors

        Args:
            collection_id: Target collection ID
            vectors: Vector data array (or VectorRecord dicts if vector_records is None)
            ids: List of unique vector identifiers (optional if vector_records provided)
            metadata: Optional list of metadata dictionaries
            upsert: Update if vectors already exist
            batch_size: Override default batch size
            vector_records: Full VectorRecord dicts with all fields (version, source, etc.)

        Returns:
            Batch insert operation result
        """
        if vector_records is not None:
            records = vector_records
        elif isinstance(vectors, list) and vectors and isinstance(vectors[0], dict):
            records = vectors
        else:
            vectors_list = self._normalize_vectors(vectors)
            if ids is None:
                ids = [f"record_{i}" for i in range(len(vectors_list))]
            if len(vectors_list) != len(ids):
                raise ValueError("Number of vectors must match number of IDs")
            if metadata and len(metadata) != len(vectors_list):
                raise ValueError(
                    "Number of metadata items must match number of vectors"
                )
            records = [
                {
                    "id": vector_id,
                    "vector": vector,
                    "props": metadata[i] if metadata else {},
                }
                for i, (vector_id, vector) in enumerate(zip(ids, vectors_list))
            ]

        return self.insert_records(
            collection_id,
            records,
            upsert=upsert,
            batch_size=batch_size,
        )

    def search(
        self,
        collection_id: str,
        vector: list[float] | np.ndarray,
        top_k: int = 10,
        metadata_filter: FilterDict | None = None,
        include_vectors: bool = False,
        include_metadata: bool = True,
        optimization_level: str = "high",
        use_storage_aware: bool = True,
        quantization_level: str = "FP32",
        enable_simd: bool = True,
        timeout: float | None = None,
        search_hints: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Search for similar vectors with storage-aware optimizations

        Args:
            collection_id: Target collection ID
            vector: Query vector
            top_k: Number of results to return
            metadata_filter: Metadata filter conditions
            include_vectors: Include vector data in results
            include_metadata: Include metadata in results
            optimization_level: Search optimization level ('high', 'medium', 'low')
            use_storage_aware: Enable storage-aware polymorphic search
            quantization_level: Vector quantization level ('FP32', 'PQ8', 'PQ4', 'Binary')
            enable_simd: Enable SIMD vectorization optimizations
            timeout: Request timeout override

        Returns:
            List of search results ordered by similarity
        """
        # Normalize query vector
        if isinstance(vector, np.ndarray):
            if vector.dtype != np.float32:
                vector = vector.astype(np.float32)
            vector = vector.tolist()

        request_data = {
            "vector": vector,
            "top_k": top_k,
            "include_vector": include_vectors,
            "include_text": False,
        }
        if metadata_filter:
            request_data["filters"] = [
                {"field": key, "op": "eq", "value": value}
                for key, value in metadata_filter.items()
            ]

        # Add search optimization if hints provided
        if search_hints:
            from ..search_utils import build_search_optimization_rest

            optimization = build_search_optimization_rest(
                enable_two_stage=search_hints.get("enable_two_stage"),
                quantization_hint=search_hints.get(
                    "quantization_hint", quantization_level
                ),
                accuracy_threshold=search_hints.get("accuracy_threshold"),
                enable_clustering_hint=search_hints.get("enable_clustering_hint"),
                enable_metadata_filtering_hint=search_hints.get(
                    "enable_metadata_filtering_hint"
                ),
                custom_hints=search_hints.get("custom_hints"),
            )
            if optimization:
                request_data["search_optimization"] = optimization

        call = _gen.search_records(collection_id, request_data)
        response = self._make_request(
            call.method,
            call.endpoint,
            json=call.json,
            timeout=timeout or self.config.timeout,
        )
        response_data = response.json()

        # Check for errors in response
        if isinstance(response_data, dict) and response_data.get("error_message"):
            error_msg = response_data.get("error_message")
            if isinstance(error_msg, str) and "not found" in error_msg.lower():
                return []
            raise ProximaDBError(f"Search failed: {error_msg}")

        # Handle proto-aligned response format
        results: list[SearchResult] = []

        # Debug: Log response structure
        if not isinstance(response_data, dict):
            logger.warning(
                f"Expected dict response, got {type(response_data)}: {response_data}"
            )
            return []

        # Get results from response - handle both direct array and nested object
        results_data = response_data.get("results", [])

        # Handle case where results is nested (e.g., {"results": {"results": [...]}})
        if isinstance(results_data, dict):
            results_list = results_data.get("results", [])
        else:
            results_list = results_data

        # Handle case where results might be None
        if results_list is None:
            logger.warning("Response contains null results field")
            return []

        # Handle case where results is still not a list
        if not isinstance(results_list, list):
            logger.warning(
                f"Expected results to be list, got {type(results_list)}: {results_list}"
            )
            return []

        for result in results_list:
            # Handle case where result might not be a dict
            if isinstance(result, str):
                # Skip malformed results
                logger.warning(
                    f"Skipping malformed result (expected dict, got string): {result}"
                )
                continue
            elif not isinstance(result, dict):
                logger.warning(
                    f"Skipping malformed result (expected dict, got {type(result)}): {result}"
                )
                continue

            results.append(
                SearchResult(
                    id=result.get("id", ""),
                    score=result.get("score", 0.0),
                    rank=result.get("rank", 0),
                    vector=(result.get("vector", []) if include_vectors else None),
                    metadata=(
                        result.get("props", result.get("metadata", {}))
                        if include_metadata
                        else None
                    ),
                    # Add all SearchVectorRecord fields (proto field 5-13)
                    version=result.get("version"),
                    similarity=result.get("similarity"),
                    timestamp=result.get("timestamp"),
                    source=result.get("source"),
                    expanded_context=result.get("expanded_context"),
                    semantic_similarity=result.get("semantic_similarity"),
                    quantization_info=result.get("quantization_info"),
                    engine_stats=result.get("engine_stats"),
                    index_path=result.get("index_path"),
                )
            )
        return results

    def search_envelope(
        self,
        collection_id: str,
        vector: list[float] | np.ndarray,
        top_k: int = 10,
        include_vectors: bool = False,
        include_metadata: bool = True,
        timeout: float | None = None,
    ) -> SearchEnvelope:
        """Search returning SKS envelope (cursor/progress) when supported.

        Falls back to legacy path and returns envelope with no cursor/progress.
        """
        from ..models import SearchEnvelope

        self._auto_warmup(collection_id)
        # Normalize vector
        if isinstance(vector, np.ndarray):
            if vector.dtype != np.float32:
                vector = vector.astype(np.float32)
            vector = vector.tolist()

        # Keep SDK REST on the OpenAPI v2 record search contract. The legacy SKS
        # envelope route is intentionally outside the published SDK surface.
        results = self.search(
            collection_id,
            vector,
            top_k,
            include_vectors=include_vectors,
            include_metadata=include_metadata,
            timeout=timeout,
        )
        return SearchEnvelope(
            items=results, total=None, cursor=None, has_more=False, progress=None
        )

    # -----------------------------
    # Graph Operations (REST)
    # -----------------------------
    def graph_shortest_path(
        self,
        start_node_id: str,
        target_node_id: str,
        max_depth: int | None = None,
        edge_types: list[str] | None = None,
        algorithm: str = "DIJKSTRA",
        k: int | None = None,
        enable_prefetch: bool | None = None,
        prefetch_budget: int | None = None,
        timeout: float | None = None,
        graph_id: str = "default",
    ) -> dict[str, Any]:
        """Compute shortest path via REST with optional prefetch overrides.

        Per-call overrides can be sent as JSON fields or HTTP headers. This method
        sends overrides as headers to keep the body stable.
        """
        body = {
            "start_node_id": start_node_id,
            "target_node_id": target_node_id,
            "algorithm": algorithm,
        }
        if max_depth is not None:
            body["max_depth"] = max_depth
        if edge_types:
            body["edge_types"] = edge_types
        if k is not None:
            body["k"] = k

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if enable_prefetch is not None:
            headers["x-graph-prefetch-enabled"] = "true" if enable_prefetch else "false"
        if prefetch_budget is not None:
            headers["x-graph-prefetch-budget"] = str(prefetch_budget)

        # Also include overrides in body for endpoints that accept JSON fields
        if enable_prefetch is not None:
            body["enable_prefetch"] = bool(enable_prefetch)
        if prefetch_budget is not None:
            body["prefetch_budget"] = int(prefetch_budget)

        resp = self._make_request(
            "POST",
            f"/api/v2/graphs/{graph_id}/shortest-path",
            json=body,
            headers=headers,
            timeout=timeout or self.config.timeout,
        )
        return resp.json()

    def graph_traverse(
        self,
        start_node_id: str,
        max_depth: int = 3,
        edge_types: list[str] | None = None,
        algorithm: str = "BFS",
        limit: int | None = None,
        timeout_ms: int | None = None,
        max_frontier: int | None = None,
        enable_prefetch: bool | None = None,
        prefetch_budget: int | None = None,
        timeout: float | None = None,
        graph_id: str = "default",
    ) -> dict[str, Any]:
        """Perform graph traversal via REST with optional prefetch overrides.

        Overrides are sent via headers. Returns the traversal response JSON.
        """
        body: dict[str, Any] = {
            "start_node_id": start_node_id,
            "max_depth": max_depth,
            "algorithm": algorithm,
        }
        if edge_types:
            body["edge_types"] = edge_types
        if limit is not None:
            body["limit"] = limit
        if timeout_ms is not None:
            body["timeout_ms"] = timeout_ms
        if max_frontier is not None:
            body["max_frontier"] = max_frontier

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if enable_prefetch is not None:
            headers["x-graph-prefetch-enabled"] = "true" if enable_prefetch else "false"
        if prefetch_budget is not None:
            headers["x-graph-prefetch-budget"] = str(prefetch_budget)

        # Also include overrides in body for compatibility
        if enable_prefetch is not None:
            body["enable_prefetch"] = bool(enable_prefetch)
        if prefetch_budget is not None:
            body["prefetch_budget"] = int(prefetch_budget)

        resp = self._make_request(
            "POST",
            f"/api/v2/graphs/{graph_id}/traverse",
            json=body,
            headers=headers,
            timeout=timeout or self.config.timeout,
        )
        return resp.json()

    def search_next_page(
        self,
        collection_id: str,
        cursor: str,
        include_vectors: bool = False,
        include_metadata: bool = True,
        timeout: float | None = None,
    ) -> SearchEnvelope:
        """Fetch next SKS search page by cursor. Returns empty envelope if unsupported."""
        from ..models import SearchEnvelope, SearchProgress

        self._auto_warmup(collection_id)
        if not cursor:
            return SearchEnvelope(
                items=[], total=None, cursor=None, has_more=False, progress=None
            )
        if self._sks_search_supported is not True:
            # Unsupported or unknown
            return SearchEnvelope(
                items=[], total=None, cursor=None, has_more=False, progress=None
            )
        # Query param per SKS design
        try:
            resp = self._make_request(
                "POST",
                f"/api/v2/collections/{collection_id}/search?cursor={cursor}",
                json={
                    "include_vector": include_vectors,
                    "include_metadata": include_metadata,
                },
                timeout=timeout or self.config.timeout,
            )
            data = resp.json()
            if not isinstance(data, dict) or "items" not in data:
                return SearchEnvelope(
                    items=[], total=None, cursor=None, has_more=False, progress=None
                )
            items: list[SearchResult] = []
            for item in data.get("items", []) or []:
                items.append(
                    SearchResult(
                        id=item.get("entity_id") or item.get("id", ""),
                        score=item.get("score", 0.0),
                        vector=None if not include_vectors else item.get("vector"),
                        metadata=(
                            (item.get("typed_metadata") or item.get("metadata") or {})
                            if include_metadata
                            else None
                        ),
                    )
                )
            progress = None
            if data.get("progress"):
                pr = data["progress"]
                progress = SearchProgress(
                    stage=pr.get("stage", 0),
                    stages=pr.get("stages", 0),
                    complete=pr.get("complete", False),
                )
            next_cursor = None
            has_more = False
            if data.get("page"):
                page = data["page"]
                next_cursor = page.get("cursor")
                has_more = page.get("has_more", False)
            return SearchEnvelope(
                items=items,
                total=data.get("total"),
                cursor=next_cursor,
                has_more=has_more,
                progress=progress,
            )
        except Exception:
            return SearchEnvelope(
                items=[], total=None, cursor=None, has_more=False, progress=None
            )

    def search_batch(
        self,
        collection_id: str,
        queries: VectorArray,
        k: int = 10,
        filter: FilterDict | None = None,
        **kwargs,
    ) -> list[list[SearchResult]]:
        """Search multiple queries in batch

        Args:
            collection_id: Target collection ID
            queries: Array of query vectors
            k: Number of results per query
            filter: Metadata filter conditions
            **kwargs: Additional search parameters

        Returns:
            List of search results for each query
        """
        queries_list = self._normalize_vectors(queries)

        request_data = {
            "queries": queries_list,
            "k": k,
            "filter": filter,
            "params": {
                "include_metadata": kwargs.get("include_metadata", True),
                "include_vectors": kwargs.get("include_vectors", False),
                "exact_search": kwargs.get("exact", False),
            },
        }

        if ef := kwargs.get("ef"):
            request_data["params"]["ef"] = ef

        response = self._make_request(
            "POST",
            f"/collections/{collection_id}/search/batch",
            json=request_data,
        )

        batch_response = response.json()
        return [
            [SearchResult(**result) for result in query_results]
            for query_results in batch_response["results"]
        ]

    def delete_vector(self, collection_id: str, vector_id: str) -> DeleteResult:
        """Delete a single record by ID."""
        self._auto_warmup(collection_id)

        response = self._make_request(
            "DELETE", f"/api/v2/collections/{collection_id}/records/{vector_id}"
        )
        response_data = response.json()
        return DeleteResult(
            success=response_data.get("success", False),
            deleted_count=1 if response_data.get("success", False) else 0,
            errors=[],
        )

    def delete_vectors(self, collection_id: str, vector_ids: list[str]) -> DeleteResult:
        """Delete multiple records by ID through the record delete endpoint."""
        self._auto_warmup(collection_id)

        deleted_count = 0
        errors = []
        for vector_id in vector_ids:
            try:
                result = self.delete_vector(collection_id, vector_id)
                if result.success:
                    deleted_count += result.deleted_count
                else:
                    errors.extend(result.errors or [f"Delete failed for {vector_id}"])
            except Exception as e:
                errors.append(f"Delete failed for {vector_id}: {str(e)}")

        return DeleteResult(
            success=not errors,
            deleted_count=deleted_count,
            errors=errors,
        )

    def get_vector(
        self,
        collection_id: str,
        vector_id: str,
        include_vector: bool = True,
        include_metadata: bool = True,
    ) -> dict[str, Any] | None:
        """Get a single vector by ID"""
        self._auto_warmup(collection_id)

        call = _gen.get_record(
            collection_id,
            vector_id,
            include_vector=include_vector,
            include_text=False,
        )
        response = self._make_request(call.method, call.endpoint, params=call.params)
        data = response.json()

        # Handle VectorOperationResponse format
        if isinstance(data, dict):
            # Check if it's an error
            if data.get("error_code") or (
                data.get("success") == False and data.get("error_code") == "NOT_FOUND"
            ):
                raise ProximaDBError(f"Vector not found: {vector_id}")

            # Extract vector from results if present
            if "results" in data and data["results"]:
                results_data = data["results"]
                if isinstance(results_data, dict) and "results" in results_data:
                    vectors = results_data["results"]
                    if vectors and len(vectors) > 0:
                        return vectors[0]  # Return first result

        return data

    def scan_records(
        self,
        collection_id: str,
        *,
        filter: list[dict[str, Any]] | dict[str, Any] | None = None,
        limit: int | None = None,
        cursor: str | None = None,
        include_vector: bool = False,
        include_text: bool = False,
    ) -> tuple[list[SearchResult], str | None]:
        """Vector-free metadata scan of one page of records (TD-099/TD-126).

        Wires the generated ``scan_records`` operation (method/URL/body shape
        sourced from the OpenAPI spec) through ``_make_request``. Returns the
        page of records mapped to :class:`SearchResult` plus the server's
        ``next_cursor`` (``None`` at end-of-scan).

        ``filter`` accepts the scan endpoint's metadata-filter shapes: the typed
        list form ``[{"field", "op", "value"}]`` or a simple equality map
        ``{field: value}``; it is pushed into the scan predicate (applied before
        the limit). This is a true metadata scan, not an ANN ranking.
        """
        body: dict[str, Any] = {
            "include_vector": include_vector,
            "include_text": include_text,
        }
        if filter:
            body["filter"] = filter
        if limit is not None:
            body["limit"] = limit
        if cursor:
            body["cursor"] = cursor

        call = _gen.scan_records(collection_id, body)
        response = self._make_request(call.method, call.endpoint, json=call.json)
        response_data = response.json()

        if isinstance(response_data, dict) and response_data.get("error_message"):
            error_msg = response_data.get("error_message")
            if isinstance(error_msg, str) and "not found" in error_msg.lower():
                return [], None
            raise ProximaDBError(f"Scan failed: {error_msg}")
        if not isinstance(response_data, dict):
            logger.warning(
                f"Expected dict scan response, got {type(response_data)}: "
                f"{response_data}"
            )
            return [], None

        records_data = response_data.get("records") or []
        results: list[SearchResult] = []
        for rank, record in enumerate(records_data, start=1):
            if not isinstance(record, dict):
                continue
            metadata = record.get("props", record.get("metadata", {})) or {}
            results.append(
                SearchResult(
                    id=str(record.get("id", "")),
                    score=0.0,
                    rank=rank,
                    vector=(record.get("vector") if include_vector else None),
                    metadata=metadata,
                    version=record.get("version"),
                    timestamp=record.get("timestamp"),
                    source=record.get("source"),
                )
            )

        next_cursor = response_data.get("next_cursor")
        if not next_cursor:
            next_cursor = None
        return results, next_cursor

    def upsert_vectors(
        self,
        collection_id: str,
        records: list[Any],
    ) -> BatchResult:
        """Upsert multiple vectors (insert or update)

        Args:
            collection_id: Target collection ID
            records: List of VectorRecord objects

        Returns:
            Batch operation result
        """
        # Convert VectorRecord objects to the format expected by insert_vectors
        vectors = []
        ids = []
        metadatas = []

        for record in records:
            vectors.append(record.vector)
            ids.append(record.id)
            metadatas.append(record.metadata if record.metadata else {})

        return self.insert_vectors(
            collection_id=collection_id,
            vectors=vectors,
            ids=ids,
            metadata=metadatas,
            upsert=True,
        )

    def update_vector(
        self,
        collection_id: str,
        vector_id: str,
        vector: list[float] | np.ndarray | None = None,
        metadata: MetadataDict | None = None,
    ) -> BatchResult:
        """Update an existing vector"""
        if vector is None:
            raise ValueError("v2 record updates require a replacement vector")

        if vector is not None:
            if isinstance(vector, np.ndarray):
                if vector.dtype != np.float32:
                    vector = vector.astype(np.float32)
                vector = vector.tolist()

        return self.insert_records(
            collection_id,
            [
                {
                    "id": vector_id,
                    "vector": vector or [],
                    "props": metadata or {},
                }
            ],
            upsert=True,
        )

    def close(self) -> None:
        """Close the client and cleanup resources"""
        # Close batch processor first
        if self._batch_processor:
            self._batch_processor.stop()
            self._batch_processor = None

        # Close response cache
        if self._response_cache:
            self._response_cache.close()
            self._response_cache = None

        if hasattr(self, "_http_client"):
            self._http_client.close()

    # Batch processing
    def _execute_batch(self, operation, collection_id, batch_data) -> list[Any]:
        """Execute a batch of requests

        Args:
            operation: BatchOperationType (e.g., INSERT_VECTORS, UPSERT_VECTORS, DELETE_VECTORS)
            collection_id: Collection ID
            batch_data: List of request data. Each item is a list of vector dicts or a list of IDs.
                        For INSERT/UPSERT: [[{id, vector, metadata}, ...], [{id, vector, metadata}, ...], ...]
                        For DELETE: [[id1, id2, ...], [id3, ...], ...]

        Returns:
            List of results for each item (or single result for the entire batch)
        """
        from proximadb_sdk.batching_unified import BatchOperationType

        try:
            # Route to appropriate method based on operation
            if operation == BatchOperationType.INSERT_VECTORS:
                # Flatten batch_data: it's a list of lists of dicts
                # [[{dict1}, {dict2}], [{dict3}], ...] -> [{dict1}, {dict2}, {dict3}, ...]
                all_vectors = []
                for item_list in batch_data:
                    if isinstance(item_list, list):
                        all_vectors.extend(item_list)
                    else:
                        all_vectors.append(item_list)

                # Extract vectors, ids, and metadata from all collected items
                vectors = [item["vector"] for item in all_vectors]
                ids = [item["id"] for item in all_vectors]
                metadata = [item.get("metadata", {}) for item in all_vectors]

                # Call insert_vectors once for the entire batch
                result = self.insert_vectors(
                    collection_id, vectors=vectors, ids=ids, metadata=metadata
                )
                return [{"success": True, "result": result}]

            elif operation == BatchOperationType.UPSERT_VECTORS:
                # Flatten batch_data
                all_vectors = []
                for item_list in batch_data:
                    if isinstance(item_list, list):
                        all_vectors.extend(item_list)
                    else:
                        all_vectors.append(item_list)

                # Extract vectors, ids, and metadata
                vectors = [item["vector"] for item in all_vectors]
                ids = [item["id"] for item in all_vectors]
                metadata = [item.get("metadata", {}) for item in all_vectors]

                # Call upsert_vectors once for the entire batch
                result = self.upsert_vectors(
                    collection_id, vectors=vectors, ids=ids, metadata=metadata
                )
                return [{"success": True, "result": result}]

            elif operation == BatchOperationType.DELETE_VECTORS:
                # Flatten batch_data: it's a list of lists of IDs
                # [[id1, id2], [id3], ...] -> [id1, id2, id3, ...]
                all_ids = []
                for id_list in batch_data:
                    if isinstance(id_list, list):
                        all_ids.extend(id_list)
                    else:
                        all_ids.append(id_list)

                # Call delete_vectors once for the entire batch
                result = self.delete_vectors(collection_id, ids=all_ids)
                return [{"success": True, "result": result}]

            else:
                return [{"success": False, "error": f"Unknown operation: {operation}"}]

        except Exception as e:
            return [{"success": False, "error": str(e)}]

    # Helper methods for cache-aware operations
    def _cached_get(
        self,
        operation: str,
        collection_id: str,
        params: dict[str, Any],
        fetch_func: callable,
        ttl_seconds: float | None = None,
    ) -> Any:
        """Helper method for cache-aware GET operations"""
        if not self.enable_caching or not self._response_cache:
            return fetch_func()

        # Include collection_id in params for cache key
        cache_params = {"collection_id": collection_id, **params}

        # Try to get from cache first
        cached_result = self._response_cache.get(operation, cache_params)
        if cached_result is not None:
            return cached_result

        # Fetch from server
        result = fetch_func()

        # Cache the result
        if result is not None:
            self._response_cache.set(
                operation,
                cache_params,
                result,
                ttl=ttl_seconds,
                collection_id=collection_id,
            )

        return result

    def _invalidate_collection_cache(self, collection_id: str):
        """Invalidate cache entries for a collection after write operations"""
        if self.enable_caching and self._response_cache:
            self._response_cache.invalidate_collection(collection_id)

    # Cache-aware read operations
    def search_cached(
        self,
        collection_id: str,
        vector: list[float] | np.ndarray,
        top_k: int = 10,
        metadata_filter: FilterDict | None = None,
        include_vectors: bool = False,
        include_metadata: bool = True,
        ttl_seconds: float | None = None,
        **kwargs,
    ) -> list[SearchResult]:
        """Cache-aware vector search

        Args:
            collection_id: Target collection ID
            vector: Query vector
            top_k: Number of results to return
            metadata_filter: Metadata filter conditions
            include_vectors: Include vector data in results
            include_metadata: Include metadata in results
            ttl_seconds: Cache TTL override
            **kwargs: Additional search parameters

        Returns:
            List of search results
        """
        if not self.enable_caching:
            return self.search(
                collection_id,
                vector,
                top_k,
                metadata_filter,
                include_vectors,
                include_metadata,
                **kwargs,
            )

        # Create cache key parameters
        cache_params = {
            "vector": vector if isinstance(vector, list) else vector.tolist(),
            "top_k": top_k,
            "metadata_filter": metadata_filter,
            "include_vectors": include_vectors,
            "include_metadata": include_metadata,
            **kwargs,
        }

        def fetch_func():
            return self.search(
                collection_id,
                vector,
                top_k,
                metadata_filter,
                include_vectors,
                include_metadata,
                **kwargs,
            )

        return self._cached_get(
            "search_vectors", collection_id, cache_params, fetch_func, ttl_seconds
        )

    def get_vector_cached(
        self,
        collection_id: str,
        vector_id: str,
        include_metadata: bool = True,
        ttl_seconds: float | None = None,
    ) -> dict[str, Any] | None:
        """Cache-aware vector retrieval

        Args:
            collection_id: Target collection ID
            vector_id: Vector identifier
            include_metadata: Include metadata in response
            ttl_seconds: Cache TTL override

        Returns:
            Vector data or None if not found
        """
        if not self.enable_caching:
            return self.get_vector(collection_id, vector_id, include_metadata)

        cache_params = {"vector_id": vector_id, "include_metadata": include_metadata}

        def fetch_func():
            return self.get_vector(collection_id, vector_id, include_metadata)

        return self._cached_get(
            "get_vector", collection_id, cache_params, fetch_func, ttl_seconds
        )

    def list_collections_cached(
        self, ttl_seconds: float | None = None
    ) -> list[Collection]:
        """Cache-aware collection listing

        Args:
            ttl_seconds: Cache TTL override

        Returns:
            List of collections
        """
        if not self.enable_caching:
            return self.list_collections()

        cache_params = {}  # No parameters for list collections

        def fetch_func():
            return self.list_collections()

        return self._cached_get(
            "list_collections", "_global", cache_params, fetch_func, ttl_seconds
        )

    def get_collection_cached(
        self, collection_id: str, ttl_seconds: float | None = None
    ) -> Collection | None:
        """Cache-aware collection retrieval

        Args:
            collection_id: Collection identifier
            ttl_seconds: Cache TTL override

        Returns:
            Collection object or None if not found
        """
        if not self.enable_caching:
            return self.get_collection(collection_id)

        cache_params = {}  # No additional parameters

        def fetch_func():
            return self.get_collection(collection_id)

        return self._cached_get(
            "get_collection", collection_id, cache_params, fetch_func, ttl_seconds
        )

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics

        Returns:
            Dictionary of cache statistics

        Raises:
            RuntimeError: If caching is not enabled
        """
        if not self.enable_caching or not self._response_cache:
            return {"error": "Caching is not enabled"}

        metrics = self._response_cache.get_metrics()
        # Convert CacheMetrics to dict for backward compatibility
        if hasattr(metrics, "__dict__"):
            metrics_dict = vars(metrics)
            # Add computed fields that tests expect
            metrics_dict["hit_rate_percent"] = (
                metrics.hit_rate * 100 if hasattr(metrics, "hit_rate") else 0.0
            )
            # Use backend's actual cache size, not cumulative hits+misses
            metrics_dict["total_entries"] = (
                self._response_cache.backend.size()
                if hasattr(self._response_cache.backend, "size")
                else 0
            )
            return metrics_dict
        return metrics

    def clear_cache(self) -> int:
        """Clear all cached responses

        Returns:
            Number of entries cleared

        Raises:
            RuntimeError: If caching is not enabled
        """
        if not self.enable_caching or not self._response_cache:
            raise RuntimeError(
                "Caching is not enabled. Initialize client with enable_caching=True"
            )

        # ResponseCache doesn't have clear(), use backend.clear() instead
        return self._response_cache.backend.clear()

    def invalidate_collection_cache(self, collection_id: str) -> int:
        """Invalidate cached responses for a collection

        Args:
            collection_id: Collection to invalidate

        Returns:
            Number of entries invalidated

        Raises:
            RuntimeError: If caching is not enabled
        """
        if not self.enable_caching or not self._response_cache:
            raise RuntimeError(
                "Caching is not enabled. Initialize client with enable_caching=True"
            )

        return self._response_cache.invalidate_collection(collection_id)

    def warm_cache(
        self,
        warmup_operations: list[tuple[str, str, dict[str, Any], Any]],
        batch_size: int | None = None,
    ) -> int:
        """Warm cache with predefined operations

        Args:
            warmup_operations: List of (operation, collection_id, params, response) tuples
            batch_size: Batch size for warming

        Returns:
            Number of entries warmed

        Raises:
            RuntimeError: If caching is not enabled
        """
        if not self.enable_caching or not self._response_cache:
            raise RuntimeError(
                "Caching is not enabled. Initialize client with enable_caching=True"
            )

        return self._response_cache.warm_cache(warmup_operations, batch_size)

    # Batched operations for improved throughput
    def insert_vectors_batched(
        self,
        collection_id: str,
        vectors: VectorArray,
        ids: list[str],
        metadata: list[MetadataDict] | None = None,
        callback: callable | None = None,
        priority: int = 1,
    ) -> str:
        """Submit vectors for batched insertion

        Args:
            collection_id: Target collection ID
            vectors: Vector data array
            ids: List of unique vector identifiers
            metadata: Optional list of metadata dictionaries
            callback: Optional callback function for result
            priority: Request priority (higher = more urgent)

        Returns:
            Request ID for tracking

        Raises:
            RuntimeError: If batching is not enabled
        """
        if not self.enable_batching or not self._batch_processor:
            raise RuntimeError(
                "Batching is not enabled. Initialize client with enable_batching=True"
            )

        vectors_list = self._normalize_vectors(vectors)

        if len(vectors_list) != len(ids):
            raise ValueError("Number of vectors must match number of IDs")

        if metadata and len(metadata) != len(vectors_list):
            raise ValueError("Number of metadata items must match number of vectors")

        # Prepare vector data
        vector_data = []
        for i, (vector_id, vector) in enumerate(zip(ids, vectors_list)):
            metadata_items = self._convert_metadata_to_rest_format(
                metadata[i] if metadata else {}
            )
            item = {
                "id": vector_id,
                "vector": vector,
                "metadata": metadata_items,
                "timestamp": int(time.time()),
            }
            vector_data.append(item)

        from proximadb_sdk.batching_unified import BatchOperationType, BatchRequest

        request = BatchRequest(
            operation=BatchOperationType.INSERT_VECTORS,
            collection_id=collection_id,
            data=vector_data,
            callback=callback,
            priority=priority,
        )
        return self._batch_processor.submit_request(request)

    def upsert_vectors_batched(
        self,
        collection_id: str,
        vectors: VectorArray,
        ids: list[str],
        metadata: list[MetadataDict] | None = None,
        callback: callable | None = None,
        priority: int = 1,
    ) -> str:
        """Submit vectors for batched upsert

        Args:
            collection_id: Target collection ID
            vectors: Vector data array
            ids: List of unique vector identifiers
            metadata: Optional list of metadata dictionaries
            callback: Optional callback function for result
            priority: Request priority (higher = more urgent)

        Returns:
            Request ID for tracking
        """
        if not self.enable_batching or not self._batch_processor:
            raise RuntimeError(
                "Batching is not enabled. Initialize client with enable_batching=True"
            )

        vectors_list = self._normalize_vectors(vectors)

        if len(vectors_list) != len(ids):
            raise ValueError("Number of vectors must match number of IDs")

        if metadata and len(metadata) != len(vectors_list):
            raise ValueError("Number of metadata items must match number of vectors")

        # Prepare vector data
        vector_data = []
        for i, (vector_id, vector) in enumerate(zip(ids, vectors_list)):
            metadata_items = self._convert_metadata_to_rest_format(
                metadata[i] if metadata else {}
            )
            item = {
                "id": vector_id,
                "vector": vector,
                "metadata": metadata_items,
                "timestamp": int(time.time()),
            }
            vector_data.append(item)

        from proximadb_sdk.batching_unified import BatchOperationType, BatchRequest

        request = BatchRequest(
            operation=BatchOperationType.UPSERT_VECTORS,
            collection_id=collection_id,
            data=vector_data,
            callback=callback,
            priority=priority,
        )
        return self._batch_processor.submit_request(request)

    def delete_vectors_batched(
        self,
        collection_id: str,
        ids: list[str],
        callback: callable | None = None,
        priority: int = 1,
    ) -> str:
        """Submit vector IDs for batched deletion

        Args:
            collection_id: Target collection ID
            ids: List of vector IDs to delete
            callback: Optional callback function for result
            priority: Request priority (higher = more urgent)

        Returns:
            Request ID for tracking
        """
        if not self.enable_batching or not self._batch_processor:
            raise RuntimeError(
                "Batching is not enabled. Initialize client with enable_batching=True"
            )

        from proximadb_sdk.batching_unified import BatchOperationType, BatchRequest

        request = BatchRequest(
            operation=BatchOperationType.DELETE_VECTORS,
            collection_id=collection_id,
            data=ids,
            callback=callback,
            priority=priority,
        )
        return self._batch_processor.submit_request(request)

    def get_batch_metrics(self) -> dict[str, Any]:
        """Get batching performance metrics

        Returns:
            Dictionary of batch processing metrics

        Raises:
            RuntimeError: If batching is not enabled
        """
        if not self.enable_batching or not self._batch_processor:
            raise RuntimeError(
                "Batching is not enabled. Initialize client with enable_batching=True"
            )

        return self._batch_processor.get_metrics()

    def reset_batch_metrics(self) -> None:
        """Reset batching performance metrics

        Raises:
            RuntimeError: If batching is not enabled
        """
        if not self.enable_batching or not self._batch_processor:
            raise RuntimeError(
                "Batching is not enabled. Initialize client with enable_batching=True"
            )

        self._batch_processor.reset_metrics()

    # === GRAPH OPERATIONS ===

    def batch_create_nodes(
        self,
        nodes: list[dict[str, Any]],
        graph_id: str = "default",
    ) -> dict[str, Any]:
        """Create many graph nodes in ONE request via the batch endpoint.

        The per-item ``create_node`` loop cost 1,000 round trips per 1,000
        nodes; this posts the whole page to
        ``/api/v2/graphs/{graph_id}/nodes/batch``.
        """
        call = _gen.batch_create_nodes(graph_id, {"nodes": nodes})
        response = self._make_request(call.method, call.endpoint, json=call.json)
        return response.json()

    def batch_create_edges(
        self,
        edges: list[dict[str, Any]],
        graph_id: str = "default",
    ) -> dict[str, Any]:
        """Create many graph edges in ONE request via the batch endpoint.

        The server applies per-edge admission (rejections are reported in the
        response's ``errors[]``; the valid rest of the batch lands).
        """
        call = _gen.batch_create_edges(graph_id, {"edges": edges})
        response = self._make_request(call.method, call.endpoint, json=call.json)
        return response.json()

    def create_node(
        self,
        node_id: str,
        labels: list[str],
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        graph_id: str = "default",
    ) -> dict[str, Any]:
        """Create a graph node via REST

        Args:
            node_id: Unique identifier for the node
            labels: List of labels for the node
            properties: Optional dictionary of node properties
            embedding: Optional embedding vector for the node
            graph_id: Graph collection ID (defaults to "default")

        Returns:
            Dictionary representation of the created node
        """
        node_data = {
            "id": node_id,
            "labels": labels,
            "properties": properties or {},
        }
        if embedding:
            node_data["embedding"] = embedding

        payload = {"node": node_data}

        call = _gen.create_node(graph_id, payload)
        response = self._make_request(call.method, call.endpoint, json=call.json)
        return response.json()

    def create_edge(
        self,
        edge_id: str,
        from_node_id: str,
        to_node_id: str,
        edge_type: str,
        properties: dict[str, Any] | None = None,
        weight: float | None = None,
        graph_id: str = "default",
    ) -> dict[str, Any]:
        """Create a graph edge via REST

        Args:
            edge_id: Unique identifier for the edge
            from_node_id: Source node ID
            to_node_id: Target node ID
            edge_type: Type/label of the edge
            properties: Optional dictionary of edge properties
            weight: Optional edge weight
            graph_id: Graph collection ID (defaults to "default")

        Returns:
            Dictionary representation of the created edge
        """
        edge_data = {
            "id": edge_id,
            "from_node_id": from_node_id,
            "to_node_id": to_node_id,
            "edge_type": edge_type,
            "properties": properties or {},
        }
        if weight is not None:
            edge_data["weight"] = weight

        payload = {"edge": edge_data}

        call = _gen.create_edge(graph_id, payload)
        response = self._make_request(call.method, call.endpoint, json=call.json)
        return response.json()

    def traverse_graph(
        self,
        start_node_id: str,
        max_depth: int = 3,
        edge_types: list[str] | None = None,
        node_labels: list[str] | None = None,
        algorithm: str = "BFS",
        limit: int | None = None,
        graph_id: str = "default",
    ) -> dict[str, Any]:
        """Traverse graph from a starting node via REST

        Args:
            start_node_id: ID of the node to start traversal from
            max_depth: Maximum depth to traverse (default: 3)
            edge_types: Optional list of edge types to follow
            node_labels: Optional list of node labels to include
            algorithm: Traversal algorithm - "BFS", "DFS", or "PARALLEL_BFS" (default: "BFS")
            limit: Optional limit on number of results
            graph_id: Graph collection ID (defaults to "default")

        Returns:
            Dictionary with nodes, edges, paths, and traversal statistics
        """
        payload = {
            "start_node_id": start_node_id,
            "max_depth": max_depth,
            "edge_types": edge_types or [],
            "node_labels": node_labels or [],
            "return_path": True,  # Required by Rust REST handler
            "algorithm": algorithm.upper(),
        }
        if limit is not None:
            payload["limit"] = limit

        call = _gen.traverse_graph(graph_id, payload)
        response = self._make_request(call.method, call.endpoint, json=call.json)
        result = response.json()

        # Transform REST response to match gRPC format
        # REST may return: {"success": true, "data": {...}, ...}
        # gRPC returns: {"nodes": [...], "edges": [...], "paths": [...], "stats": {...}}
        # If data is nested, extract it; otherwise use result directly
        data = result.get("data", result)

        return {
            "nodes": data.get("nodes", []),
            "edges": data.get("edges", []),
            "paths": data.get("paths", []),
            "stats": data.get(
                "stats",
                {
                    "nodes_visited": 0,
                    "edges_traversed": 0,
                    "max_depth_reached": 0,
                    "execution_time_microseconds": 0,
                },
            ),
        }

    def query_nodes(
        self,
        labels: list[str] | None = None,
        properties: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        graph_id: str = "default",
    ) -> dict[str, Any]:
        """Query nodes by labels and properties via REST

        Args:
            labels: Optional list of labels to filter by
            properties: Optional dictionary of properties to filter by
            limit: Optional maximum number of results
            offset: Optional offset for pagination
            graph_id: Graph collection ID (defaults to "default")

        Returns:
            Dictionary with success status, nodes list, and total count
        """
        payload = {"labels": labels or [], "properties": properties or {}}
        if limit is not None:
            payload["limit"] = limit
        if offset is not None:
            payload["offset"] = offset

        response = self._http_client.post(
            f"/api/v2/graphs/{graph_id}/query/nodes", json=payload
        )
        response.raise_for_status()
        result = response.json()

        # Transform REST response to match gRPC format.
        # v2 list endpoints wrap results in a paginated envelope:
        #   {"success": true, "data": {"items": [...], "has_more": bool}}
        # (older builds returned data as a bare list; tolerate both).
        nodes, has_more = _unwrap_v2_list(result.get("data"))
        return {
            "success": result.get("success", True),
            "nodes": nodes,
            "total_count": len(nodes),
            "has_more": has_more,
            "next_token": result.get("next_token"),
        }

    def query_edges(
        self,
        edge_type: str = "",
        from_node_id: str | None = None,
        to_node_id: str | None = None,
        properties: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        graph_id: str = "default",
    ) -> dict[str, Any]:
        """Query edges by endpoints, type, and properties via REST."""
        payload: dict[str, Any] = {
            "edge_type": edge_type,
            "properties": properties or {},
        }
        if from_node_id is not None:
            payload["from_node_id"] = from_node_id
        if to_node_id is not None:
            payload["to_node_id"] = to_node_id
        if limit is not None:
            payload["limit"] = limit
        if offset is not None:
            payload["offset"] = offset

        response = self._http_client.post(
            f"/api/v2/graphs/{graph_id}/query/edges", json=payload
        )
        response.raise_for_status()
        result = response.json()
        # v2 wraps results in {"data": {"items": [...], "has_more": bool}}.
        edges, has_more = _unwrap_v2_list(result.get("data"))
        return {
            "success": result.get("success", True),
            "edges": edges,
            "total_count": len(edges),
            "has_more": has_more,
            "next_token": result.get("next_token"),
        }

    def get_node(
        self,
        node_id: str,
        graph_id: str = "default",
    ) -> dict[str, Any] | None:
        """Get a graph node by ID via REST."""
        call = _gen.get_node(graph_id, node_id)
        response = self._make_request(call.method, call.endpoint)
        result = response.json()
        return result.get("data", result)

    def get_outgoing_edges(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        graph_id: str = "default",
    ) -> list[dict[str, Any]]:
        """Get outgoing graph edges for a node via REST."""
        edge_types = edge_types or [""]
        edges: list[dict[str, Any]] = []
        for edge_type in edge_types:
            result = self.query_edges(
                edge_type=edge_type,
                from_node_id=node_id,
                graph_id=graph_id,
                limit=10000,
            )
            edges.extend(result.get("edges", []))
        return edges

    def get_incoming_edges(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        graph_id: str = "default",
    ) -> list[dict[str, Any]]:
        """Get incoming graph edges for a node via REST."""
        edge_types = edge_types or [""]
        edges: list[dict[str, Any]] = []
        for edge_type in edge_types:
            result = self.query_edges(
                edge_type=edge_type,
                to_node_id=node_id,
                graph_id=graph_id,
                limit=10000,
            )
            edges.extend(result.get("edges", []))
        return edges

    def delete_node(
        self,
        node_id: str,
        graph_id: str = "default",
    ) -> dict[str, Any]:
        """Delete a graph node by ID via REST."""
        call = _gen.delete_node(graph_id, node_id)
        response = self._make_request(call.method, call.endpoint)
        result = response.json()
        return result.get("data", result)

    # ==================== Graph Collection Management ====================
    # Methods for managing graph collections (create, delete, list, get)

    def create_graph(
        self,
        graph_id: str,
        name: str | None = None,
        description: str | None = None,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a new graph collection

        Args:
            graph_id: Unique identifier for the graph collection
            name: Optional human-readable name (defaults to graph_id)
            description: Optional description of the graph
            schema: Optional schema definition for the graph

        Returns:
            Dictionary containing the created graph collection metadata

        Example:
            >>> graph = client.create_graph(
            ...     graph_id="social_network",
            ...     name="Social Network Graph",
            ...     description="User relationships and interactions"
            ... )
        """
        payload = {"graph_id": graph_id, "name": name, "description": description}
        if schema is not None:
            payload["schema"] = schema

        call = _gen.create_graph(payload)
        response = self._make_request(call.method, call.endpoint, json=call.json)
        return response.json()

    def delete_graph(self, graph_id: str) -> dict[str, Any]:
        """Delete a graph collection

        Args:
            graph_id: ID of the graph collection to delete

        Returns:
            Dictionary confirming deletion

        Example:
            >>> result = client.delete_graph("social_network")
        """
        call = _gen.delete_graph(graph_id)
        response = self._make_request(call.method, call.endpoint)
        return response.json()

    def get_graph(self, graph_id: str) -> dict[str, Any]:
        """Get graph collection metadata

        Args:
            graph_id: ID of the graph collection

        Returns:
            Dictionary containing graph collection metadata

        Example:
            >>> graph = client.get_graph("social_network")
            >>> print(graph["name"])
        """
        call = _gen.get_graph(graph_id)
        response = self._make_request(call.method, call.endpoint)
        return response.json()

    def list_graphs(self) -> dict[str, Any]:
        """List all graph collections

        Returns:
            Dictionary containing list of all graph collections

        Example:
            >>> graphs = client.list_graphs()
            >>> for graph in graphs.get("data", []):
            ...     print(graph["graph_id"])
        """
        call = _gen.list_graphs()
        response = self._make_request(call.method, call.endpoint)
        return response.json()

    def get_graph_stats(self, graph_id: str) -> dict[str, Any]:
        """Get statistics for a graph collection

        Args:
            graph_id: ID of the graph collection

        Returns:
            Dictionary containing graph statistics (node count, edge count, etc.)

        Example:
            >>> stats = client.get_graph_stats("social_network")
            >>> print(f"Nodes: {stats['node_count']}, Edges: {stats['edge_count']}")
        """
        call = _gen.get_graph_stats(graph_id)
        response = self._make_request(call.method, call.endpoint)
        return response.json()

    # ==================== End Graph Collection Management ====================

    # ==================== AQL/UQL Query API ====================

    def execute_query(
        self,
        query: str,
        *,
        language: str = "uql",
        parameters: list[Any] | None = None,
        collection: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Execute an AQL, UQL, or federated SQL-extension query."""
        payload: dict[str, Any] = {"language": language, "query": query}
        if parameters is not None:
            payload["parameters"] = parameters
        if collection is not None:
            payload["collection"] = collection
        if limit is not None:
            payload["limit"] = limit

        call = _gen.execute_query(payload)
        response = self._make_request(call.method, call.endpoint, json=call.json)
        return response.json()

    def explain_query(
        self,
        query: str,
        *,
        language: str = "uql",
        collection: str | None = None,
    ) -> dict[str, Any]:
        """Explain an AQL or UQL query through the OpenAPI v2 query surface."""
        payload: dict[str, Any] = {"language": language, "query": query}
        if collection is not None:
            payload["collection"] = collection

        call = _gen.explain_query(payload)
        response = self._make_request(call.method, call.endpoint, json=call.json)
        return response.json()

    def execute_uql(
        self,
        query: str,
        *,
        parameters: list[Any] | None = None,
        collection: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Execute a UQL query through the OpenAPI v2 query surface."""
        return self.execute_query(
            query,
            language="uql",
            parameters=parameters,
            collection=collection,
            limit=limit,
        )

    def execute_aql(
        self,
        query: str,
        *,
        parameters: list[Any] | None = None,
        collection: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Execute an AQL query through the OpenAPI v2 query surface."""
        return self.execute_query(
            query,
            language="aql",
            parameters=parameters,
            collection=collection,
            limit=limit,
        )

    def execute_federated(
        self,
        query: str,
        *,
        parameters: list[Any] | None = None,
        collection: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Execute federated SQL extensions through the OpenAPI v2 query surface."""
        return self.execute_query(
            query,
            language="federated",
            parameters=parameters,
            collection=collection,
            limit=limit,
        )

    # ==================== End AQL/UQL Query API ====================

    # ==================== SQL Query API ====================

    def execute_sql(
        self,
        query: str,
        parameters: list[Any] | None = None,
        collection: str | None = None,
    ) -> dict[str, Any]:
        """Execute SQL query against the database.

        Args:
            query: SQL query string
            parameters: Optional list of parameter values for prepared statements
            collection: Optional collection name to use as default context

        Returns:
            Dictionary containing:
                - rows: List of row dictionaries
                - row_count: Number of rows returned
                - rows_scanned: Number of rows scanned
                - rows_returned: Number of rows returned
                - execution_time_ms: Query execution time in milliseconds
                - columns: List of column names
                - column_types: List of column types

        Example:
            >>> result = client.execute_sql(
            ...     "SELECT id, metadata FROM my_collection WHERE metadata.price > 100 LIMIT 10"
            ... )
            >>> for row in result['rows']:
            ...     print(row)
        """
        # Canonical v2 query facade. Raw SQL is submitted via the unified query
        # language ("uql"); the legacy /api/v1/sql/execute route was removed in
        # the API standardization hard-rename.
        payload: dict[str, Any] = {"language": "uql", "query": query}

        if parameters:
            payload["parameters"] = parameters

        if collection:
            payload["collection"] = collection

        try:
            response = self._http_client.post("/api/v2/query", json=payload)
            response.raise_for_status()
            result = response.json()

            # Ensure consistent response format
            if "data" in result:
                # Unwrap if response is wrapped
                result = result["data"]

            # Ensure row_count is present for consistency
            if "rows" in result and "row_count" not in result:
                result["row_count"] = len(result["rows"])

            return result

        except httpx.HTTPStatusError as e:
            raise map_http_error(e)
        except httpx.TimeoutException as e:
            raise TimeoutError(f"SQL query timed out: {e}")
        except httpx.RequestError as e:
            raise NetworkError(f"Network error executing SQL: {e}")

    # ==================== End SQL Query API ====================

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def __del__(self):
        """Destructor - cleanup resources"""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup


# Convenience functions
def connect(
    url: str | None = None, api_key: str | None = None, **kwargs
) -> ProximaDBClient:
    """Create a ProximaDB client with simplified parameters"""
    return ProximaDBClient(url=url, api_key=api_key, **kwargs)


def quick_search(
    collection_id: str,
    query: list[float] | np.ndarray,
    k: int = 10,
    url: str | None = None,
    api_key: str | None = None,
) -> list[SearchResult]:
    """Quick one-off search without creating persistent client"""
    with connect(url=url, api_key=api_key) as client:
        return client.search(collection_id, query, k)
