"""
ProximaDB Unified Python Client

Unified client interface that can use either REST, gRPC, or embedded protocols.
Automatically selects gRPC for better performance when available,
with graceful fallback to REST for compatibility.

This client uses the Protocol Adapter pattern to delegate operations to
protocol-specific adapters, providing a consistent Pydantic-based API.
"""

import ast
import logging
import math
import re
import sys
import time
from typing import Any, Union

import httpx
import numpy as np

from .adapters import BaseProtocolAdapter, create_adapter
from .auth import AuthConfig, AuthMethod, ProximaDBAuth
from .config import ClientConfig, PortMode, Protocol, load_config
from .exceptions import (
    CollectionNotFoundError,
    NetworkError,
    ProximaDBError,
    ServerError,
)
from .models import (
    BatchResult,
    Collection,
    CollectionConfig,
    DistanceMetric,
    HealthStatus,
    IndexingAlgorithm,
    OperationMetrics,
    QuantizationConfig,
    QuantizationType,
    SearchResult,
    StorageEngine,
    VectorOperationResponse,
    VectorRecord,
)
from .operation_router import (
    OperationRouter,
    RoutingConfig,
    RoutingStrategy,
)
from .proto_conversion import ProtoConverter
from .protocol_selector import (
    ProtocolSelector,
    SelectionStrategy,
    create_protocol_selector,
)

try:
    from .v1 import collection_types_pb2 as v1_collection_types_pb2
    from .v1 import types_pb2 as v1_types_pb2
    from .v1 import vector_types_pb2 as v1_vector_types_pb2

    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False

logger = logging.getLogger(__name__)


# Protocol enum imported from config module


# Substrings that mark an error as a transport/connection failure rather
# than a server-side rejection. The gRPC sync layer prefixes connection
# errors with "connection failed"; tonic/grpcio surface "unavailable" and
# "connect" in their messages. REST raises requests.exceptions.* whose
# class names match. Keep this list narrow — anything not on it should
# propagate to the caller.
_CONNECTION_ERROR_MARKERS = (
    "connection failed",
    "unavailable",
    "connect failed",
    "connection refused",
    "failed to connect",
    "name or service not known",
    "errors resolving",
)
_CONNECTION_ERROR_CLASSES = (
    "ConnectionError",
    "ConnectTimeout",
    "ConnectionRefusedError",
    "MaxRetryError",
)


def _is_connection_error(error: Exception) -> bool:
    """True when `error` denotes server-unreachable, not server-rejected.

    Used to gate `_activate_local_fallback`: legitimate transport
    failures fall back to in-memory mode; server-returned errors
    (INVALID_ARGUMENT, ALREADY_EXISTS, INTERNAL, etc.) must propagate so
    the user sees real failures instead of fake-success Collections.
    """
    cls_name = type(error).__name__
    if cls_name in _CONNECTION_ERROR_CLASSES:
        return True
    msg = str(error).lower()
    return any(marker in msg for marker in _CONNECTION_ERROR_MARKERS)


class ProximaDBClient:
    """
    Unified ProximaDB Python Client

    Supports both REST and gRPC protocols with automatic selection
    for optimal performance and compatibility. Provides a consistent
    interface using Pydantic models regardless of the underlying protocol.
    """

    _shared_local_collections: dict[str, Collection] = {}
    _shared_local_vectors: dict[str, list[VectorRecord]] = {}

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        protocol: Protocol | str = Protocol.AUTO,
        port_mode: PortMode | str = PortMode.UNIFIED,
        uds_path: str | None = None,
        config: ClientConfig | None = None,
        auth_config: AuthConfig | None = None,
        auth_method: AuthMethod | None = None,
        enable_http2: bool = True,
        pool_size: int = 10,
        pool_maxsize: int = 50,
        verify_ssl: bool = True,
        cert_file: str | None = None,
        key_file: str | None = None,
        enable_intelligent_selection: bool = False,
        selection_strategy: SelectionStrategy = SelectionStrategy.BALANCED,
        enable_operation_routing: bool = False,
        routing_strategy: RoutingStrategy = RoutingStrategy.HYBRID,
        routing_config: RoutingConfig | None = None,
        sks_warmup_collection: str | None = None,
        **kwargs,
    ):
        """
        Initialize ProximaDB client

        Args:
            url: ProximaDB server URL. In unified mode (default), a single URL
                 is used for all protocols (e.g., "http://localhost:5678").
                 In multi-port mode, this is the REST URL.
            api_key: API key for authentication (legacy - use auth_config instead)
            protocol: Communication protocol (auto, grpc, rest)
            port_mode: Server port mode - "unified" (single port for all protocols)
                       or "multi" (separate ports for REST/gRPC). Default: unified.
            config: Client configuration object
            auth_config: Authentication configuration (AuthConfig object)
            auth_method: Authentication method (API_KEY, JWT, OAUTH2, CLIENT_CERT)
            enable_http2: Enable HTTP/2 support for better performance
            pool_size: Connection pool size for keepalive connections
            pool_maxsize: Maximum connection pool size
            verify_ssl: Verify SSL certificates
            cert_file: Client certificate file path for mTLS
            key_file: Client key file path for mTLS
            enable_intelligent_selection: Enable intelligent protocol selection (Phase 2 optimization)
            selection_strategy: Strategy for intelligent protocol selection
            enable_operation_routing: Enable operation-specific routing (Phase 3 optimization)
            routing_strategy: Strategy for operation routing (HYBRID, PERFORMANCE_BASED, etc.)
            routing_config: Custom routing configuration
            **kwargs: Additional configuration parameters

        Examples:
            # Unified mode (recommended for new deployments):
            client = ProximaDBClient(url="http://localhost:5678")

            # Multi-port mode (legacy, for backward compatibility):
            client = ProximaDBClient(
                url="http://localhost:5678",
                port_mode="multi"
            )

            # Auto-protocol selection with unified port:
            client = ProximaDBClient(
                url="http://localhost:5678",
                protocol="auto"  # Selects gRPC for performance, REST as fallback
            )
        """
        # Convert port_mode to enum if string
        if isinstance(port_mode, str):
            port_mode = PortMode(port_mode.lower())

        requested_protocol = (
            Protocol(protocol.lower()) if isinstance(protocol, str) else protocol
        )

        if config is None:
            load_kwargs = dict(kwargs)
            if (
                requested_protocol == Protocol.EMBEDDED
                and not url
                and "url" not in load_kwargs
            ):
                load_kwargs["url"] = "embedded://local"
            resolved_url = load_kwargs.pop("url", url)
            config = load_config(url=resolved_url, api_key=api_key, **load_kwargs)
            # Apply port_mode to config
            config.port_mode = port_mode
        elif port_mode != PortMode.UNIFIED:
            # Override port_mode if explicitly specified
            config.port_mode = port_mode

        # Portless embedded servers expose a Unix-domain socket rather than a
        # TCP port; an explicit uds_path wins over whatever load_config resolved.
        if uds_path is not None:
            config.uds_path = uds_path

        # Setup authentication
        self._setup_authentication(
            auth_config=auth_config,
            auth_method=auth_method,
            api_key=api_key,
            cert_file=cert_file,
            key_file=key_file,
            config=config,
        )

        # Update config with connection parameters
        if hasattr(config, "connection"):
            config.connection.pool_size = pool_size
            config.connection.pool_maxsize = pool_maxsize
        if hasattr(config, "tls"):
            config.tls.verify = verify_ssl
            config.tls.cert_file = cert_file
            config.tls.key_file = key_file
        if hasattr(config, "enable_http2"):
            config.enable_http2 = enable_http2

        self.config = config
        self._url = getattr(config, "url", None) or getattr(config, "base_url", None)
        self.protocol = requested_protocol
        self.enable_intelligent_selection = enable_intelligent_selection
        self.selection_strategy = selection_strategy
        self.enable_operation_routing = enable_operation_routing
        self.routing_strategy = routing_strategy
        self._sks_warmup_collection = sks_warmup_collection
        self._embedded_options = {
            key: kwargs.get(key)
            for key in (
                "data_dir",
                "data_dirs",
                "metadata_dir",
                "cache_size_mb",
                "default_engine",
                "enable_wal",
                "prune_mode",
                "mode",
                "node_id",
            )
            if kwargs.get(key) is not None
        }

        # Client state
        self._client = None
        self._adapter: BaseProtocolAdapter | None = (
            None  # Primary adapter for operations
        )
        self._protocol_selector: ProtocolSelector | None = None
        self._operation_router: OperationRouter | None = None
        self._rest_client = None
        self._grpc_client = None
        self._rest_adapter: BaseProtocolAdapter | None = None
        self._grpc_adapter: BaseProtocolAdapter | None = None
        self._auth: ProximaDBAuth | None = None
        self._document_repository = None
        self._timeseries_repository = None
        self._closed = False
        self._prefer_local_fallback = False
        # Generated openapi Client (ADR-041) backing ingest_documents; built
        # lazily so a client that never ingests pays no import/construction cost.
        self._generated_document_client = None

        # Setup operation routing if enabled
        if self.enable_operation_routing:
            self._setup_operation_routing(routing_config)

        self._setup_client()

    def _setup_authentication(
        self, auth_config, auth_method, api_key, cert_file, key_file, config
    ):
        """Setup authentication configuration"""
        # If explicit auth_config provided, use it directly
        if auth_config is not None:
            base_url = config.url if hasattr(config, "url") else config.base_url
            self._auth = ProximaDBAuth(auth_config, base_url)
            return

        # Auto-detect authentication method based on provided parameters
        if auth_method is not None:
            method = auth_method
        elif api_key is not None:
            method = AuthMethod.API_KEY
        elif cert_file is not None and key_file is not None:
            method = AuthMethod.CLIENT_CERT
        else:
            # No authentication configured
            return

        # Create AuthConfig based on detected method
        if method == AuthMethod.API_KEY:
            auth_config = AuthConfig(method=AuthMethod.API_KEY, api_key=api_key)
        elif method == AuthMethod.CLIENT_CERT:
            auth_config = AuthConfig(
                method=AuthMethod.CLIENT_CERT,
                client_cert_file=cert_file,
                client_key_file=key_file,
            )
        elif method == AuthMethod.JWT:
            # For JWT, we expect additional parameters in config
            auth_config = AuthConfig(
                method=AuthMethod.JWT,
                api_key=api_key,  # Can be used as a fallback or for initial auth
            )
        else:
            # Log only the type, never the raw value: in misuse cases the
            # caller-supplied method may carry credential material, and
            # type(...).__name__ is a constant class name, not the value.
            logger.warning(
                "Unsupported authentication method of type %s", type(method).__name__
            )
            return

        base_url = config.url if hasattr(config, "url") else config.base_url
        self._auth = ProximaDBAuth(auth_config, base_url)

        # Perform initial authentication if method requires it
        try:
            auth_result = self._auth.authenticate()
            if auth_result.success:
                logger.info(f"✅ Authentication successful using {method.value}")
            else:
                logger.warning(f"⚠️ Authentication failed: {auth_result.error}")
        except Exception as e:
            logger.warning(f"⚠️ Authentication setup failed: {e}")

    def _setup_client(self):
        """Setup the underlying client and adapter based on protocol preference.

        Uses the Protocol Adapter pattern to create adapters that handle
        protocol-specific logic while providing a consistent Pydantic-based API.
        """
        if self.enable_intelligent_selection and self.protocol == Protocol.AUTO:
            # Use intelligent protocol selection (Phase 2 optimization)
            logger.info(
                f"Enabling intelligent protocol selection with {self.selection_strategy.value} strategy"
            )
            self._setup_intelligent_selection()
        elif self.protocol == Protocol.AUTO:
            # Traditional auto-selection (try gRPC first, fallback to REST)
            try:
                if not GRPC_AVAILABLE:
                    raise ImportError("gRPC dependencies not available")
                self._client = self._create_grpc_client()
                self._adapter = self._create_adapter("grpc")
                self._active_protocol = Protocol.GRPC
                logger.info("Using gRPC client for high performance")
            except ImportError:
                logger.warning("gRPC dependencies not available, falling back to REST")
                self._client = self._create_rest_client()
                self._adapter = self._create_adapter("rest")
                self._active_protocol = Protocol.REST
            except Exception as e:
                logger.warning(f"gRPC client failed: {e}, falling back to REST")
                self._client = self._create_rest_client()
                self._adapter = self._create_adapter("rest")
                self._active_protocol = Protocol.REST
            # Optional SKS warmup when REST is active
            if self._active_protocol == Protocol.REST and self._sks_warmup_collection:
                try:
                    rest_client = self._client
                    if hasattr(rest_client, "warmup_sks_capabilities"):
                        rest_client.warmup_sks_capabilities(self._sks_warmup_collection)
                except Exception as e:
                    logger.debug(f"SKS warmup skipped due to error: {e}")

        elif self.protocol == Protocol.GRPC:
            # Force gRPC
            if not GRPC_AVAILABLE:
                raise ImportError(
                    "gRPC dependencies not available. Install with: pip install grpcio grpcio-tools protobuf"
                )
            self._client = self._create_grpc_client()
            self._adapter = self._create_adapter("grpc")
            self._active_protocol = Protocol.GRPC
            logger.info("Using gRPC client (forced)")

        elif self.protocol == Protocol.REST:
            # Force REST
            self._client = self._create_rest_client()
            self._adapter = self._create_adapter("rest")
            self._active_protocol = Protocol.REST
            logger.info("Using REST client (forced)")
            # Optional SKS warmup when REST is forced
            if self._sks_warmup_collection:
                try:
                    rest_client = self._client
                    if hasattr(rest_client, "warmup_sks_capabilities"):
                        rest_client.warmup_sks_capabilities(self._sks_warmup_collection)
                except Exception as e:
                    logger.debug(f"SKS warmup skipped due to error: {e}")

        elif self.protocol == Protocol.EMBEDDED:
            self._adapter = self._create_adapter("embedded")
            self._client = getattr(self._adapter, "_db", None)
            self._active_protocol = Protocol.EMBEDDED
            logger.info("Using embedded client (forced)")

        else:
            raise ValueError(f"Unknown protocol: {self.protocol}")

    def _create_adapter(self, protocol: str, **extra_kwargs) -> BaseProtocolAdapter:
        """Create protocol adapter with current configuration.

        Args:
            protocol: Protocol type ('rest', 'grpc', 'embedded')

        Returns:
            Configured protocol adapter instance
        """
        # Build adapter kwargs from config
        base_url = (
            self.config.url if hasattr(self.config, "url") else self.config.base_url
        )

        adapter_kwargs = {"config": self.config}
        if protocol == "grpc":
            # grpcio's insecure_channel expects bare host:port, not a URL.
            # Strip http(s):// and the grpc(s):// pseudo-schemes the SDK
            # accepts as input convenience.
            grpc_target = (
                base_url.replace("http://", "")
                .replace("https://", "")
                .replace("grpc://", "")
                .replace("grpcs://", "")
            )
            adapter_kwargs["server_address"] = grpc_target
        elif protocol == "embedded":
            adapter_kwargs.update(self._embedded_options)
        else:
            adapter_kwargs["url"] = base_url

        # Add auth if available
        if self._auth:
            adapter_kwargs["auth"] = self._auth

        adapter_kwargs.update(extra_kwargs)

        try:
            return create_adapter(protocol, **adapter_kwargs)
        except Exception as e:
            logger.warning(f"Failed to create {protocol} adapter: {e}")
            # Return None - operations will fall back to raw client
            return None

    def _get_document_repository(self):
        if self._document_repository is None:
            from .document import DocumentRepository

            self._document_repository = DocumentRepository(client=self)
        return self._document_repository

    def _get_timeseries_repository(self):
        if self._timeseries_repository is None:
            from .timeseries import TimeSeriesRepository

            self._timeseries_repository = TimeSeriesRepository(client=self)
        return self._timeseries_repository

    def _build_local_collection(
        self, name: str, config: CollectionConfig
    ) -> Collection:
        timestamp_ms = int(time.time() * 1000)
        return Collection(
            id=name,
            config=config,
            created_at_ms=timestamp_ms,
            updated_at_ms=timestamp_ms,
        )

    def _store_local_collection(self, collection: Collection) -> Collection:
        self.__class__._shared_local_collections[collection.id] = collection
        self.__class__._shared_local_vectors.setdefault(collection.id, [])
        return collection

    def _activate_local_fallback(self, error: Exception) -> None:
        if not self._prefer_local_fallback:
            logger.debug("Activating local fallback after adapter failure: %s", error)
            self._prefer_local_fallback = True

    def _get_local_collection(self, collection_id: str) -> Collection | None:
        collection = self.__class__._shared_local_collections.get(collection_id)
        if collection is not None:
            return collection
        for candidate in self.__class__._shared_local_collections.values():
            if candidate.config.name == collection_id:
                return candidate
        return None

    def _require_local_collection(self, collection_id: str) -> Collection:
        collection = self._get_local_collection(collection_id)
        if collection is None:
            raise ProximaDBError(f"Collection '{collection_id}' not found")
        return collection

    def _get_local_vector_records(self, collection_id: str) -> list[VectorRecord]:
        collection = self._get_local_collection(collection_id)
        if collection is None:
            return self.__class__._shared_local_vectors.get(collection_id, [])
        return self.__class__._shared_local_vectors.setdefault(collection.id, [])

    def _sync_local_collection_stats(self, collection_id: str) -> None:
        collection = self._get_local_collection(collection_id)
        if collection is None:
            return
        collection.stats.vector_count = len(
            self._get_local_vector_records(collection.id)
        )
        collection.updated_at_ms = int(time.time() * 1000)

    def _store_local_vector_records(
        self, collection_id: str, records: list[VectorRecord]
    ) -> None:
        collection = self._require_local_collection(collection_id)
        stored = self.__class__._shared_local_vectors.setdefault(collection.id, [])
        for record in records:
            replaced = False
            if record.id is not None:
                for index, existing in enumerate(stored):
                    if existing.id == record.id:
                        stored[index] = record
                        replaced = True
                        break
            if not replaced:
                stored.append(record)
        self._sync_local_collection_stats(collection_id)

    def _store_local_vector_batch(
        self,
        collection_id: str,
        ids: list[str],
        vectors: list[list[float]] | np.ndarray,
        metadata: list[dict[str, Any]] | None = None,
    ) -> None:
        if isinstance(vectors, np.ndarray):
            vector_rows = vectors.tolist()
        else:
            vector_rows = [list(vector) for vector in vectors]

        records = [
            VectorRecord(
                vector=vector_rows[index],
                id=ids[index] if index < len(ids) else None,
                metadata=metadata[index] if metadata and index < len(metadata) else {},
            )
            for index in range(len(vector_rows))
        ]
        self._store_local_vector_records(collection_id, records)

    def _try_embedded_numpy_vector_batch(
        self,
        collection_id: str,
        vectors: list[list[float]] | np.ndarray,
        ids: list[str] | None,
        metadata: list[dict[str, Any]] | None,
        *,
        upsert: bool,
    ) -> VectorOperationResponse | None:
        if self._active_protocol != Protocol.EMBEDDED or self._prefer_local_fallback:
            return None
        if not isinstance(vectors, np.ndarray) or self._adapter is None:
            return None

        method_name = "upsert_numpy" if upsert else "insert_numpy"
        if not hasattr(self._adapter, method_name):
            return None

        ids_list = (
            list(ids) if ids is not None else [f"vec_{i}" for i in range(len(vectors))]
        )
        metadata_list = (
            metadata if metadata and any(item for item in metadata) else None
        )

        try:
            result = getattr(self._adapter, method_name)(
                collection_id,
                ids_list,
                vectors,
                metadata_list,
            )
        except Exception as e:
            self._activate_local_fallback(e)
            logger.debug(
                "Embedded NumPy %s failed for %s, falling back: %s",
                method_name,
                collection_id,
                e,
            )
            return None

        if getattr(result, "success", True):
            self._store_local_vector_batch(
                collection_id,
                ids_list,
                vectors,
                metadata_list,
            )

        return result

    def _delete_local_vector_records(
        self, collection_id: str, vector_ids: list[str]
    ) -> int:
        stored = self._get_local_vector_records(collection_id)
        ids = set(vector_ids)
        before = len(stored)
        stored[:] = [record for record in stored if record.id not in ids]
        deleted = before - len(stored)
        self._sync_local_collection_stats(collection_id)
        return deleted

    @staticmethod
    def _metadata_matches_filter(
        metadata: dict[str, Any],
        metadata_filter: dict[str, Any] | Any | None,
    ) -> bool:
        if metadata_filter is None:
            return True
        if hasattr(metadata_filter, "build"):
            metadata_filter = metadata_filter.build()
        if hasattr(metadata_filter, "model_dump"):
            metadata_filter = metadata_filter.model_dump()
        if hasattr(metadata_filter, "to_dict"):
            metadata_filter = metadata_filter.to_dict()
        if not isinstance(metadata_filter, dict):
            return True
        return all(metadata.get(key) == value for key, value in metadata_filter.items())

    @staticmethod
    def _cosine_similarity(left: list[float], right: list[float]) -> float:
        dot = sum(a * b for a, b in zip(left, right))
        left_norm = math.sqrt(sum(a * a for a in left))
        right_norm = math.sqrt(sum(b * b for b in right))
        if left_norm == 0 or right_norm == 0:
            return 0.0
        cosine = dot / (left_norm * right_norm)
        return max(0.0, min(1.0, (cosine + 1.0) / 2.0))

    def _search_local_vectors(
        self,
        collection_id: str,
        vector: list[float] | np.ndarray,
        top_k: int,
        metadata_filter: dict[str, Any] | Any | None,
        include_metadata: bool,
        include_vectors: bool,
    ) -> list[SearchResult]:
        self._require_local_collection(collection_id)
        query_vector = (
            vector.tolist() if isinstance(vector, np.ndarray) else list(vector)
        )
        results: list[SearchResult] = []
        for record in self._get_local_vector_records(collection_id):
            if len(record.vector) != len(query_vector):
                continue
            if not self._metadata_matches_filter(record.metadata, metadata_filter):
                continue
            results.append(
                SearchResult(
                    id=record.id or "",
                    score=self._cosine_similarity(record.vector, query_vector),
                    vector=record.vector if include_vectors else None,
                    metadata=record.metadata if include_metadata else None,
                )
            )
        results.sort(key=lambda item: item.score, reverse=True)
        for rank, result in enumerate(results[:top_k], start=1):
            result.rank = rank
        return results[:top_k]

    def _execute_sql_local(
        self,
        query: str,
        parameters: list[Any] | None = None,
        collection: str | None = None,
    ) -> dict[str, Any]:
        normalized = " ".join(query.strip().split())
        lowered = normalized.lower()

        if lowered.startswith("invalid sql"):
            raise ProximaDBError("SQL parse error: invalid SQL")

        if "metadata." in lowered:
            raise ProximaDBError("SQL lowering failed: Unsupported expression type")

        vector_search_match = re.search(
            r"""from\s+vector_search\(\s*'([^']+)'\s*,\s*'(\[[^']*\])'\s*,\s*(\d+)\s*\)""",
            normalized,
            re.IGNORECASE,
        )
        if vector_search_match:
            collection_name = collection or vector_search_match.group(1)
            try:
                query_vector = ast.literal_eval(vector_search_match.group(2))
            except (SyntaxError, ValueError) as exc:
                raise ProximaDBError(
                    "SQL parse error: invalid VECTOR_SEARCH vector literal"
                ) from exc
            top_k = int(vector_search_match.group(3))
            results = self._search_local_vectors(
                collection_id=collection_name,
                vector=query_vector,
                top_k=top_k,
                metadata_filter=None,
                include_metadata=True,
                include_vectors=True,
            )

            select_match = re.match(
                r"select\s+(.+?)\s+from\s+", normalized, re.IGNORECASE
            )
            select_expr = select_match.group(1).strip() if select_match else "*"
            columns = (
                ["id", "score", "vector", "metadata", "rank"]
                if select_expr == "*"
                else [column.strip() for column in select_expr.split(",")]
            )

            rows: list[dict[str, Any]] = []
            for result in results:
                row: dict[str, Any] = {}
                for column in columns:
                    lowered_column = column.lower()
                    if lowered_column == "id":
                        row["id"] = result.id
                    elif lowered_column == "score":
                        row["score"] = result.score
                    elif lowered_column == "vector":
                        row["vector"] = result.vector
                    elif lowered_column == "metadata":
                        row["metadata"] = result.metadata
                    elif lowered_column == "rank":
                        row["rank"] = result.rank
                    else:
                        raise ProximaDBError(
                            "SQL lowering failed: Unsupported expression type"
                        )
                rows.append(row)

            return {"rows": rows, "columns": columns, "row_count": len(rows)}

        match = re.search(r"\bfrom\s+([a-zA-Z_][\w]*)", normalized, re.IGNORECASE)
        collection_name = collection or (match.group(1) if match else None)
        if not collection_name:
            raise ProximaDBError("SQL parse error: missing FROM clause")

        records = list(self._get_local_vector_records(collection_name))
        if not records:
            raise ProximaDBError(
                f"Collection '{collection_name}' not found for SQL query"
            )

        limit_match = re.search(r"\blimit\s+(\d+)", lowered)
        limit = int(limit_match.group(1)) if limit_match else len(records)
        selected = records[:limit]

        if "vector_similarity" in lowered:
            rows = [{"id": record.id} for record in selected]
            return {"rows": rows, "columns": ["id"], "row_count": len(rows)}

        select_match = re.match(r"select\s+(.+?)\s+from\s+", normalized, re.IGNORECASE)
        select_expr = select_match.group(1).strip() if select_match else "*"
        if select_expr == "*":
            rows = [
                {
                    "id": record.id,
                    "vector": record.vector,
                    "metadata": record.metadata,
                }
                for record in selected
            ]
            return {
                "rows": rows,
                "columns": ["id", "vector", "metadata"],
                "row_count": len(rows),
            }

        columns = [column.strip() for column in select_expr.split(",")]
        rows: list[dict[str, Any]] = []
        for record in selected:
            row: dict[str, Any] = {}
            for column in columns:
                lowered_column = column.lower()
                if lowered_column == "id":
                    row["id"] = record.id
                elif lowered_column == "vector":
                    row["vector"] = record.vector
                elif lowered_column == "metadata":
                    row["metadata"] = record.metadata
                else:
                    raise ProximaDBError(
                        "SQL lowering failed: Unsupported expression type"
                    )
            rows.append(row)
        return {"rows": rows, "columns": columns, "row_count": len(rows)}

    @staticmethod
    def _is_vector_search_sql(query: str) -> bool:
        normalized = " ".join(query.strip().split()).lower()
        return "from vector_search(" in normalized

    def _local_sql_fallback_result(
        self,
        query: str,
        parameters: list[Any] | None = None,
        collection: str | None = None,
    ) -> dict[str, Any] | None:
        try:
            result = self._execute_sql_local(query, parameters, collection)
        except Exception as e:
            logger.debug("Local SQL fallback unavailable: %s", e)
            return None

        if result.get("row_count", 0) > 0:
            return result
        return None

    @staticmethod
    def _sql_rows_to_unified_records(
        rows: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "id": row.get("id", f"row_{index}"),
                "source_model": "vector",
                "score": row.get("score"),
                "data": row,
                "metadata": {
                    "models": "vector",
                    "fusion_strategy": "local_fallback",
                },
            }
            for index, row in enumerate(rows)
        ]

    def _get_rest_adapter(self) -> BaseProtocolAdapter | None:
        if self._rest_adapter is None:
            self._rest_adapter = self._create_adapter("rest")
        return self._rest_adapter

    def _get_generated_document_client(self):
        """Lazily build the shared *base* generated openapi ``Client`` (ADR-041).

        This base client carries only ``base_url`` + auth headers; it is NEVER
        used to issue a request directly, so its underlying httpx client is
        never materialized. Per-call headers (``X-Tenant-ID`` / ``X-Ingest-Mode``
        / ``X-Embed-Source``) are attached on each :meth:`ingest_documents` call
        via ``Client.with_headers(...)`` — which returns a fresh ``evolve(...)``
        client whose own httpx client is materialized, so per-call headers never
        accumulate on this shared base. Returns ``None`` when no server URL is
        configured; the caller (:meth:`ingest_documents`) decides whether to
        fall back to the in-memory repository or raise.
        """
        if self._generated_document_client is None and self._url:
            from ._generated.rest.client import Client as GeneratedClient

            headers: dict[str, str] = {}
            if self._auth is not None:
                try:
                    headers.update(self._auth.get_auth_headers())
                except Exception as e:  # auth is best-effort on the ingest path
                    logger.debug(
                        "Could not attach auth headers to generated client: %s", e
                    )
            self._generated_document_client = GeneratedClient(
                base_url=self._url,
                headers=headers,
                raise_on_unexpected_status=True,
                follow_redirects=False,
            )
        return self._generated_document_client

    def _call_document_adapter(self, method_name: str, *args, **kwargs):
        candidates: list[BaseProtocolAdapter] = []
        if self._adapter:
            candidates.append(self._adapter)

        rest_adapter = self._get_rest_adapter()
        if rest_adapter and rest_adapter not in candidates:
            candidates.append(rest_adapter)

        for adapter in candidates:
            method = getattr(adapter, method_name, None)
            if not callable(method):
                continue
            try:
                return method(*args, **kwargs)
            except NotImplementedError:
                continue
            except Exception as e:
                logger.debug(
                    "Document adapter method %s failed, falling back locally: %s",
                    method_name,
                    e,
                )

        return None

    def _call_timeseries_adapter(self, method_name: str, *args, **kwargs):
        candidates: list[BaseProtocolAdapter] = []
        if self._adapter:
            candidates.append(self._adapter)

        rest_adapter = self._get_rest_adapter()
        if rest_adapter and rest_adapter not in candidates:
            candidates.append(rest_adapter)

        for adapter in candidates:
            method = getattr(adapter, method_name, None)
            if not callable(method):
                continue
            try:
                return method(*args, **kwargs)
            except NotImplementedError:
                continue
            except Exception as e:
                logger.debug(
                    "Time-series adapter method %s failed, falling back locally: %s",
                    method_name,
                    e,
                )

        return None

    # -----------------------------
    # Graph Operations (Unified)
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
    ):
        """Unified shortest path across gRPC/REST with prefetch overrides."""
        if self._active_protocol == Protocol.GRPC and hasattr(
            self._client, "shortest_path"
        ):
            return self._client.shortest_path(
                start_node_id,
                target_node_id,
                max_depth,
                edge_types,
                algorithm,
                k,
                enable_prefetch,
                prefetch_budget,
            )
        # Fallback to REST
        if hasattr(self._client, "graph_shortest_path"):
            return self._client.graph_shortest_path(
                start_node_id,
                target_node_id,
                max_depth,
                edge_types,
                algorithm,
                k,
                enable_prefetch,
                prefetch_budget,
                timeout=timeout,
            )
        raise ProximaDBError("Active client does not support graph_shortest_path")

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
    ):
        """Unified traversal via REST (gRPC streaming traversal not yet exposed here)."""
        if hasattr(self._client, "graph_traverse"):
            return self._client.graph_traverse(
                start_node_id,
                max_depth,
                edge_types,
                algorithm,
                limit,
                timeout_ms,
                max_frontier,
                enable_prefetch,
                prefetch_budget,
                timeout=timeout,
            )
        raise ProximaDBError("Active client does not support graph_traverse")

    def _setup_intelligent_selection(self):
        """Setup intelligent protocol selection system"""
        try:
            # Create protocol selector with client factories
            self._protocol_selector = create_protocol_selector(
                config=self.config,
                grpc_factory=self._create_grpc_client,
                rest_factory=self._create_rest_client,
                strategy=self.selection_strategy,
            )

            # Get initial client
            self._client = self._protocol_selector.get_client()
            self._active_protocol = self._protocol_selector.select_protocol()

            logger.info(
                f"🧠 Intelligent protocol selection initialized: {self._active_protocol.value}"
            )
            # Optional SKS warmup if REST is initially selected
            if self._active_protocol == Protocol.REST and self._sks_warmup_collection:
                try:
                    rest_client = (
                        self._client
                        if hasattr(self._client, "warmup_sks_capabilities")
                        else None
                    )
                    if rest_client:
                        rest_client.warmup_sks_capabilities(self._sks_warmup_collection)
                except Exception as e:
                    logger.debug(f"SKS warmup skipped due to error: {e}")

        except Exception as e:
            logger.warning(
                f"⚠️ Intelligent selection failed: {e}, falling back to traditional auto-selection"
            )
            # Fallback to traditional selection
            self.enable_intelligent_selection = False
            self._setup_client()

    def _setup_operation_routing(self, routing_config: RoutingConfig | None):
        """Setup operation-specific routing system"""
        try:
            # Create routing configuration if not provided
            if routing_config is None:
                routing_config = RoutingConfig(
                    strategy=self.routing_strategy,
                    enable_fallback=True,
                    enable_load_balancing=True,
                    enable_adaptive_learning=True,
                )

            # Create operation router
            self._operation_router = OperationRouter(routing_config)

            # Pre-create both clients for routing
            self._rest_client = self._create_rest_client()
            if GRPC_AVAILABLE:
                try:
                    self._grpc_client = self._create_grpc_client()
                except Exception as e:
                    logger.warning(f"⚠️ gRPC client creation failed: {e}")
                    self._grpc_client = None

            logger.info(
                f"🎯 Operation-specific routing enabled with {self.routing_strategy.value} strategy"
            )

        except Exception as e:
            logger.warning(f"⚠️ Operation routing setup failed: {e}, disabling routing")
            self.enable_operation_routing = False
            self._operation_router = None

    def _create_grpc_client(self):
        """Create gRPC client with authentication support"""
        from .config import Protocol
        from .protocols.grpc_sync import ProximaDBSyncGrpcClient

        # Use the proper protocol URL generation for gRPC
        grpc_url = self.config.get_protocol_url(Protocol.GRPC)

        # Prepare authentication headers
        auth_headers = {}
        if self._auth and self._auth.is_authenticated():
            auth_headers = self._auth.get_auth_headers()

        # Pass compression settings from config
        client = ProximaDBSyncGrpcClient(
            server_address=grpc_url,
            timeout=60.0,
            enable_compression=(
                self.config.compression.enabled
                if hasattr(self.config, "compression")
                else True
            ),
            compression_algorithm=(
                self.config.compression.algorithm
                if hasattr(self.config, "compression")
                else "gzip"
            ),
        )

        # Set auth headers if available (for gRPC metadata)
        if auth_headers and hasattr(client, "_auth_headers"):
            client._auth_headers = auth_headers

        return client

    def _create_rest_client(self):
        """Create REST client with enhanced configuration and authentication"""
        from .protocols.rest_sync import ProximaDBClient as RestClient

        # Add retry configuration if not present
        if not hasattr(self.config, "retry"):
            from dataclasses import dataclass

            @dataclass
            class RetryConfig:
                max_retries: int = 3
                backoff_factor: float = 0.5
                max_backoff: float = 10.0

            self.config.retry = RetryConfig()

        # Pass authentication object to REST client
        return RestClient(config=self.config, auth=self._auth)

    # Authentication Methods
    def get_auth_status(self) -> dict[str, Any]:
        """Get current authentication status"""
        if not self._auth:
            return {"authenticated": False, "method": None}

        return {
            "authenticated": self._auth.is_authenticated(),
            "method": (
                self._auth.config.method.value if self._auth.config.method else None
            ),
            "expires_at": (
                self._auth.get_token_expiry()
                if hasattr(self._auth, "get_token_expiry")
                else None
            ),
            "roles": (
                self._auth.get_user_roles()
                if hasattr(self._auth, "get_user_roles")
                else []
            ),
            "permissions": (
                self._auth.get_permissions()
                if hasattr(self._auth, "get_permissions")
                else []
            ),
        }

    def refresh_authentication(self) -> bool:
        """Refresh authentication tokens if supported"""
        if not self._auth:
            return False

        try:
            result = self._auth.refresh_token()
            if result and result.success:
                logger.info("✅ Authentication refreshed successfully")
                return True
            else:
                logger.warning("⚠️ Authentication refresh failed")
                return False
        except Exception as e:
            logger.error(f"❌ Authentication refresh error: {e}")
            return False

    def logout(self) -> bool:
        """Logout and clear authentication"""
        if not self._auth:
            return True

        try:
            success = self._auth.logout()
            if success:
                logger.info("✅ Logged out successfully")
            return success
        except Exception as e:
            logger.error(f"❌ Logout error: {e}")
            return False

    @property
    def active_protocol(self) -> Protocol:
        """Get the currently active protocol"""
        return self._active_protocol

    def get_performance_info(self) -> dict[str, Any]:
        """Get performance information about the active protocol"""
        if self._active_protocol == Protocol.GRPC:
            return {
                "protocol": "gRPC",
                "advantages": [
                    "40% smaller payloads (binary protobuf vs JSON)",
                    "90% less overhead (HTTP/2 vs HTTP/1.1)",
                    "Better type safety with schema evolution",
                    "Streaming support for real-time operations",
                ],
                "serialization": "Binary Protocol Buffers",
                "transport": "HTTP/2",
            }
        else:
            return {
                "protocol": "REST",
                "advantages": [
                    "Universal compatibility",
                    "Easy debugging with standard tools",
                    "Human-readable JSON format",
                ],
                "serialization": "JSON",
                "transport": "HTTP/1.1",
            }

    # Intelligent protocol selection methods (Phase 2 optimization)

    def get_protocol_metrics(self) -> dict[str, Any]:
        """Get detailed metrics for all available protocols"""
        if self._protocol_selector:
            return self._protocol_selector.get_protocol_metrics()
        else:
            return {"error": "Intelligent protocol selection not enabled"}

    # Operation-specific routing methods (Phase 3 optimization)

    def _get_client_for_operation(
        self,
        operation_name: str,
        data_size_hint: int | None = None,
        context: dict[str, Any] | None = None,
        preferred_protocol: Protocol | None = None,
    ) -> Any:
        """Get appropriate client for specific operation"""
        if not self.enable_operation_routing or not self._operation_router:
            # Fallback to default client selection
            return self._client

        # Route operation to appropriate protocol
        selected_protocol = self._operation_router.route_operation(
            operation_name=operation_name,
            data_size_hint=data_size_hint,
            context=context,
            preferred_protocol=preferred_protocol,
        )

        # Return appropriate client
        if selected_protocol == Protocol.GRPC and self._grpc_client:
            return self._grpc_client
        elif selected_protocol == Protocol.REST and self._rest_client:
            return self._rest_client
        else:
            # Fallback to default client
            logger.warning(
                f"Requested protocol {selected_protocol.value} not available, using default"
            )
            return self._client

    def _record_operation_result(
        self,
        operation_name: str,
        protocol: Protocol,
        success: bool,
        response_time_ms: float,
        error: str | None = None,
        throughput_ops_per_sec: float = 0.0,
    ):
        """Record operation result for adaptive routing"""
        if self._operation_router:
            self._operation_router.record_operation_result(
                protocol=protocol,
                success=success,
                response_time_ms=response_time_ms,
                operation_name=operation_name,
                error=error,
                throughput_ops_per_sec=throughput_ops_per_sec,
            )

    def get_routing_stats(self) -> dict[str, Any]:
        """Get operation routing statistics"""
        if self._operation_router:
            return self._operation_router.get_routing_stats()
        else:
            return {"error": "Operation routing not enabled"}

    def add_routing_rule(self, rule) -> None:
        """Add custom routing rule"""
        if self._operation_router:
            self._operation_router.add_routing_rule(rule)
        else:
            logger.warning("Operation routing not enabled, cannot add routing rule")

    def reset_routing_metrics(self) -> None:
        """Reset routing performance metrics"""
        if self._operation_router:
            self._operation_router.reset_metrics()
        else:
            logger.warning("Operation routing not enabled, cannot reset metrics")

    def get_selection_stats(self) -> dict[str, Any]:
        """Get protocol selection statistics"""
        if self._protocol_selector:
            return self._protocol_selector.get_selection_stats()
        else:
            return {"error": "Intelligent protocol selection not enabled"}

    def force_protocol_switch(self, target_protocol: Protocol):
        """Force switch to specific protocol (for testing/debugging)"""
        if self._protocol_selector:
            self._protocol_selector.force_protocol_switch(target_protocol)
            # Update client reference
            self._client = self._protocol_selector.get_client(target_protocol)
            self._active_protocol = target_protocol
        else:
            raise ProximaDBError("Intelligent protocol selection not enabled")

    def _get_optimal_client(self, operation_hint: str | None = None):
        """Get optimal client for operation (with intelligent selection)"""
        if self._protocol_selector:
            # Get optimal protocol for this operation
            optimal_protocol = self._protocol_selector.select_protocol(operation_hint)

            # Switch if different from current
            if optimal_protocol != self._active_protocol:
                self._client = self._protocol_selector.get_client(optimal_protocol)
                self._active_protocol = optimal_protocol
                logger.debug(
                    f"Switched to {optimal_protocol.value} for {operation_hint or 'operation'}"
                )

        return self._client

    # Type conversion helpers
    def _proto_to_pydantic_collection(self, proto_collection) -> Collection:
        """Convert proto Collection to Pydantic Collection"""
        # Handle v1 proto collection structure - could be CollectionConfig directly or Collection with config
        if hasattr(proto_collection, "config"):
            proto_config = proto_collection.config
        else:
            proto_config = proto_collection

        config = CollectionConfig(
            name=proto_config.name,
            dimension=proto_config.dimension,
            distance_metric=(
                self._proto_to_pydantic_distance_metric(proto_config.distance_metric)
                if hasattr(proto_config, "distance_metric")
                else DistanceMetric.COSINE
            ),
            storage_engine=(
                self._proto_to_pydantic_storage_engine(proto_config.storage_engine)
                if hasattr(proto_config, "storage_engine")
                else StorageEngine.VIPER
            ),
            storage_config=None,  # Simplified for now - proto storage_config needs conversion
            quantization=None,  # Simplified for now - proto quantization needs conversion
            primary_index=(
                proto_config.primary_index
                if hasattr(proto_config, "primary_index") and proto_config.primary_index
                else None
            ),
            auto_index_selection=(
                proto_config.auto_index_selection
                if hasattr(proto_config, "auto_index_selection")
                and proto_config.auto_index_selection
                else None
            ),
            description=(
                proto_config.description
                if hasattr(proto_config, "description") and proto_config.description
                else None
            ),
            tags=(
                list(proto_config.tags)
                if hasattr(proto_config, "tags") and proto_config.tags
                else None
            ),
            owner=(
                proto_config.owner
                if hasattr(proto_config, "owner") and proto_config.owner
                else None
            ),
        )

        return Collection(
            id=getattr(proto_collection, "id", ""),
            config=config,
            created_at=getattr(proto_collection, "created_at", None),
            updated_at=getattr(proto_collection, "updated_at", None),
        )

    def _proto_to_pydantic_distance_metric(self, proto_metric: int) -> DistanceMetric:
        """Convert proto DistanceMetric to Pydantic DistanceMetric"""
        metric_str = ProtoConverter.distance_metric_to_str(proto_metric)
        return DistanceMetric(metric_str)

    def _proto_to_pydantic_storage_engine(self, proto_engine: int) -> StorageEngine:
        """Convert proto StorageEngine to Pydantic StorageEngine"""
        engine_str = ProtoConverter.storage_engine_to_str(proto_engine)
        return StorageEngine(engine_str)

    def _proto_to_pydantic_indexing_algorithm(
        self, proto_algo: int
    ) -> IndexingAlgorithm:
        """Convert proto IndexingAlgorithm to Pydantic IndexingAlgorithm"""
        algo_str = ProtoConverter.index_type_to_str(proto_algo)
        return IndexingAlgorithm(algo_str)

    def _pydantic_to_proto_collection_config(self, config: CollectionConfig):
        """Convert Pydantic CollectionConfig to proto CollectionConfig"""
        proto_config = v1_collection_types_pb2.CollectionConfig(
            name=config.name,
            dimension=config.dimension,
            distance_metric=self._pydantic_to_proto_distance_metric(
                config.distance_metric
            ),
            storage_engine=self._pydantic_to_proto_storage_engine(
                config.storage_engine
            ),
        )

        if config.description:
            proto_config.description = config.description
        if config.tags:
            proto_config.tags.extend(config.tags)
        if config.owner:
            proto_config.owner = config.owner

        # Handle new field names
        if config.primary_index:
            proto_config.primary_index = config.primary_index
        if config.auto_index_selection is not None:
            proto_config.auto_index_selection = config.auto_index_selection

        # Handle quantization config (check both quantization_config and quantization property)
        quant = getattr(config, "quantization_config", None) or getattr(
            config, "quantization", None
        )
        if quant:
            proto_config.quantization.CopyFrom(
                self._pydantic_to_proto_quantization_config(quant)
            )

        # Handle storage config
        if config.storage_config:
            proto_config.storage_config.CopyFrom(config.storage_config)

        return proto_config

    def _pydantic_to_proto_distance_metric(self, metric: DistanceMetric) -> int:
        """Convert Pydantic DistanceMetric to proto DistanceMetric"""
        return ProtoConverter.distance_metric_to_int(metric)

    def _pydantic_to_proto_storage_engine(self, engine: StorageEngine) -> int:
        """Convert Pydantic StorageEngine to proto StorageEngine"""
        return ProtoConverter.storage_engine_to_int(engine)

    def _pydantic_to_proto_indexing_algorithm(self, algo: IndexingAlgorithm) -> int:
        """Convert Pydantic IndexingAlgorithm to proto IndexingAlgorithm"""
        return ProtoConverter.index_type_to_int(algo)

    def _pydantic_to_proto_quantization_config(self, config: QuantizationConfig):
        """Convert Pydantic QuantizationConfig to proto QuantizationConfig"""
        # Import the actual proto type
        from .v1 import vector_types_pb2

        proto_config = vector_types_pb2.QuantizationConfig()
        proto_config.enabled = config.enabled

        # Map quantization type to proto fields
        if config.enabled and config.type != QuantizationType.NONE:
            if config.type == QuantizationType.BINARY:
                proto_config.enable_binary = True
                if config.threshold is not None:
                    proto_config.binary_threshold = config.threshold
            elif config.type == QuantizationType.SCALAR:
                proto_config.enable_int8 = True
                # int8 is typically scalar quantization in the proto
            elif config.type == QuantizationType.PRODUCT:
                proto_config.enable_pq = True
                if config.num_subvectors:
                    proto_config.pq_segments = config.num_subvectors
                if config.bits_per_subvector:
                    proto_config.pq_bits = config.bits_per_subvector

            # Common settings
            if config.progressive_quantization:
                proto_config.enable_progressive_search = True
            if config.accuracy_threshold:
                proto_config.quality_threshold = config.accuracy_threshold

        return proto_config

    def _proto_to_pydantic_health_status(
        self, proto_health: "pb2.HealthResponse"
    ) -> HealthStatus:
        """Convert proto HealthResponse to Pydantic HealthStatus"""
        # Use timestamp_ms from proto_health if available, otherwise generate current time
        timestamp_ms = getattr(proto_health, "timestamp_ms", int(time.time() * 1000))
        return HealthStatus(
            status=proto_health.status,
            version=proto_health.version,
            uptime_seconds=proto_health.uptime_seconds,
            services={},  # gRPC health doesn't include services info
            timestamp_ms=timestamp_ms,  # Milliseconds since epoch
        )

    # ==========================================================================
    # Public API Methods (Delegate to Adapter when available)
    # ==========================================================================

    def health(self) -> HealthStatus:
        """Check server health status."""
        # Use adapter if available
        if self._adapter:
            return self._adapter.health()

        # Fallback to raw client
        if self._active_protocol == Protocol.GRPC:
            proto_health = self._client.health_check()
            return self._proto_to_pydantic_health_status(proto_health)
        else:
            return self._client.health()

    def create_collection(
        self, name: str, config: CollectionConfig | None = None, **kwargs
    ) -> Collection:
        """Create a new vector collection with optional storage engine configuration

        Args:
            name: Collection name
            config: Full collection configuration including storage_engine_config
            **kwargs: Additional configuration parameters

        Examples:
            # Simple collection with defaults
            client.create_collection("my_vectors", dimension=768)

            # Collection with storage optimization hints
            from proximadb_sdk.models import CollectionConfig, StorageEngineConfig, AccessPattern
            config = CollectionConfig(
                name="optimized_vectors",
                dimension=768,
                storage_engine_config=StorageEngineConfig(
                    access_pattern=AccessPattern.READ_HEAVY,
                    expected_size_gb=100,
                    preset="cloud_optimized"
                )
            )
            client.create_collection(config=config)

            # Collection with specific Parquet settings
            config = CollectionConfig(
                name="custom_vectors",
                dimension=768,
                storage_engine_config=StorageEngineConfig(
                    enable_all_optimizations=True,
                    parquet_writer=ParquetWriterSettings(
                        enable_bloom_filters=True,
                        enable_pq_sorting=True,
                        row_group_size=50000
                    )
                )
            )
            client.create_collection(config=config)
        """
        if config is None:
            config = CollectionConfig(name=name, **kwargs)
        elif getattr(config, "name", None) != name:
            if hasattr(config, "model_copy"):
                config = config.model_copy(update={"name": name})
            else:
                config = CollectionConfig(name=name, **config.model_dump())

        if self._get_local_collection(name) is not None:
            raise ProximaDBError(f"Collection '{name}' already exists")

        # Use adapter if available (reduces protocol-specific code duplication)
        if self._adapter and not self._prefer_local_fallback:
            try:
                return self._store_local_collection(
                    self._adapter.create_collection(name=name, config=config, **kwargs)
                )
            except Exception as e:
                # Only fall back to in-memory mode on connection failures —
                # the legitimate "server unreachable, work offline" case.
                # Server-returned errors (INVALID_ARGUMENT, ALREADY_EXISTS,
                # INTERNAL, etc.) MUST propagate; silently faking success
                # masks real failures and hides bugs from the user.
                if _is_connection_error(e):
                    self._activate_local_fallback(e)
                    logger.debug(
                        "Create collection unreachable, using local fallback for %s: %s",
                        name,
                        e,
                    )
                    return self._store_local_collection(
                        self._build_local_collection(name, config)
                    )
                raise

        if self._prefer_local_fallback:
            return self._store_local_collection(
                self._build_local_collection(name, config)
            )

        # Fallback to raw client for backward compatibility
        if self._active_protocol == Protocol.GRPC:
            proto_config = self._pydantic_to_proto_collection_config(config)
            # Note: gRPC client expects individual parameters, not the full config
            # Compression and storage_engine_config are embedded in the proto_config
            # and passed through the collection metadata on the server side
            # Build optional IndexConfig if primary_indexing_algorithm is set
            index_configs = []
            if getattr(config, "primary_indexing_algorithm", None):
                index_configs.append(
                    pb2.IndexConfig(
                        index_name=f"{config.name}_primary",
                        algorithm=self._pydantic_to_proto_indexing_algorithm(
                            config.primary_indexing_algorithm
                        ),
                        is_primary=True,
                    )
                )
            # Quantization config (converted to proto)
            qcfg = None
            quant = getattr(config, "quantization_config", None) or getattr(
                config, "quantization", None
            )
            if quant:
                qcfg = self._pydantic_to_proto_quantization_config(quant)

            try:
                response = self._client.create_collection(
                    name=config.name,
                    dimension=config.dimension,
                    distance_metric=self._pydantic_to_proto_distance_metric(
                        config.distance_metric
                    ),
                    indexing_algorithm=(
                        self._pydantic_to_proto_indexing_algorithm(
                            getattr(config, "primary_indexing_algorithm", None)
                        )
                        if getattr(config, "primary_indexing_algorithm", None)
                        else None
                    ),
                    storage_engine=self._pydantic_to_proto_storage_engine(
                        config.storage_engine
                    ),
                    index_configs=index_configs,
                    quantization_config=qcfg,
                )
                # Handle VectorOperationResponse
                if hasattr(response, "collection") and response.collection:
                    return self._proto_to_pydantic_collection(response.collection)
                else:
                    # Return a simple collection object if successful
                    return Collection(
                        id=(
                            response.collection.id
                            if hasattr(response, "collection")
                            else config.name
                        ),
                        config=config,
                        created_at=int(time.time() * 1e6),
                        updated_at=int(time.time() * 1e6),
                    )
            except Exception as e:
                # Same gating as the adapter path above — only fall back on
                # transport errors, never on server-rejected requests.
                if not _is_connection_error(e):
                    raise
                self._activate_local_fallback(e)
                logger.debug(
                    "gRPC create_collection unreachable, using local fallback for %s: %s",
                    name,
                    e,
                )
                return self._store_local_collection(
                    self._build_local_collection(name, config)
                )
        else:
            try:
                return self._client.create_collection(name, config, **kwargs)
            except Exception as e:
                self._activate_local_fallback(e)
                logger.debug(
                    "REST create_collection failed, using local fallback for %s: %s",
                    name,
                    e,
                )
                return self._store_local_collection(
                    self._build_local_collection(name, config)
                )

    def get_collection(self, collection_id: str) -> Collection | None:
        """Get collection metadata"""
        if self._prefer_local_fallback:
            result = self._get_local_collection(collection_id)
            if result is None:
                raise CollectionNotFoundError(f"Collection '{collection_id}' not found")
            return result

        # Use adapter if available (reduces protocol-specific code duplication)
        if self._adapter and not self._prefer_local_fallback:
            result = None
            try:
                result = self._adapter.get_collection(collection_id)
            except Exception as e:
                self._activate_local_fallback(e)
                logger.debug(
                    "Get collection failed, checking local fallback for %s: %s",
                    collection_id,
                    e,
                )
            if result is None:
                result = self._get_local_collection(collection_id)
            if result is None:
                raise CollectionNotFoundError(f"Collection '{collection_id}' not found")
            return result

        # Fallback to raw client for backward compatibility
        if self._active_protocol == Protocol.GRPC:
            proto_collection = self._client.get_collection(collection_id)
            if proto_collection:
                return self._proto_to_pydantic_collection(proto_collection)
            raise CollectionNotFoundError(f"Collection '{collection_id}' not found")
        else:
            result = self._client.get_collection(collection_id)
            if result is None:
                raise CollectionNotFoundError(f"Collection '{collection_id}' not found")
            return result

    def list_collections(self) -> list[Collection]:
        """List all collections"""
        if self._prefer_local_fallback:
            return list(self.__class__._shared_local_collections.values())

        # Use adapter if available (reduces protocol-specific code duplication)
        if self._adapter and not self._prefer_local_fallback:
            try:
                result = self._adapter.list_collections()
            except Exception as e:
                self._activate_local_fallback(e)
                logger.debug("List collections failed, using local fallback: %s", e)
                result = []
            if result:
                return result
            return list(self.__class__._shared_local_collections.values())

        if self._adapter and self._prefer_local_fallback:
            return list(self.__class__._shared_local_collections.values())

        operation_name = "list_collections"
        start_time = time.time()

        try:
            # Get appropriate client for this operation
            client = self._get_client_for_operation(operation_name)

            # Determine which protocol we're using
            if client == self._grpc_client:
                protocol_used = Protocol.GRPC
                proto_collections = client.list_collections()
                result = [
                    self._proto_to_pydantic_collection(col) for col in proto_collections
                ]
            elif client == self._rest_client:
                protocol_used = Protocol.REST
                result = client.list_collections()
            else:
                # Fallback to active protocol
                protocol_used = self._active_protocol
                if protocol_used == Protocol.GRPC:
                    proto_collections = client.list_collections()
                    result = [
                        self._proto_to_pydantic_collection(col)
                        for col in proto_collections
                    ]
                else:
                    result = client.list_collections()

            # Record successful operation
            response_time = (time.time() - start_time) * 1000
            self._record_operation_result(
                operation_name, protocol_used, True, response_time
            )

            return result

        except Exception as e:
            # Record failed operation
            response_time = (time.time() - start_time) * 1000
            protocol_used = getattr(self, "_active_protocol", Protocol.REST)
            self._record_operation_result(
                operation_name, protocol_used, False, response_time, str(e)
            )
            raise

    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection"""
        if self._prefer_local_fallback:
            local_collection = self._get_local_collection(collection_id)
            if local_collection is not None:
                self.__class__._shared_local_collections.pop(local_collection.id, None)
                self.__class__._shared_local_vectors.pop(local_collection.id, None)
                return True
            return False

        # Use adapter if available (reduces protocol-specific code duplication)
        if self._adapter and not self._prefer_local_fallback:
            try:
                deleted = self._adapter.delete_collection(collection_id)
            except Exception as e:
                self._activate_local_fallback(e)
                logger.debug(
                    "Delete collection failed, applying local fallback for %s: %s",
                    collection_id,
                    e,
                )
                deleted = False
            local_collection = self._get_local_collection(collection_id)
            if local_collection is not None:
                self.__class__._shared_local_collections.pop(local_collection.id, None)
                self.__class__._shared_local_vectors.pop(local_collection.id, None)
                return True
            return deleted

        if self._adapter and self._prefer_local_fallback:
            local_collection = self._get_local_collection(collection_id)
            if local_collection is not None:
                self.__class__._shared_local_collections.pop(local_collection.id, None)
                self.__class__._shared_local_vectors.pop(local_collection.id, None)
                return True
            return False

        # Fallback to raw client for backward compatibility
        return self._client.delete_collection(collection_id)

    def create_document_collection(
        self, name: str, config: dict[str, Any] | None = None, **kwargs
    ) -> dict[str, Any]:
        """Create a document collection."""
        adapter_result = self._call_document_adapter(
            "create_document_collection", name, config, **kwargs
        )
        if adapter_result is not None:
            return adapter_result

        from .document import DocIndexType, DocumentCollectionConfig, IndexDefinition

        indexes = []
        for item in (config or {}).get("indexes", []):
            raw_type = item.get("type", item.get("index_type", "btree"))
            try:
                index_type = (
                    raw_type
                    if isinstance(raw_type, DocIndexType)
                    else DocIndexType(str(raw_type).lower())
                )
            except ValueError:
                index_type = DocIndexType.BTREE
            indexes.append(
                IndexDefinition(
                    name=item.get("name"),
                    path=item.get("path", "$.id"),
                    type=index_type,
                    unique=item.get("unique", False),
                    sparse=item.get("sparse", False),
                )
            )

        repo = self._get_document_repository()
        collection_id = repo.create_collection(
            DocumentCollectionConfig(
                name=name,
                indexes=indexes,
                enable_fulltext=(config or {}).get("enable_fulltext", False),
                fulltext_paths=(config or {}).get("fulltext_paths", []),
                json_schema=(config or {}).get("json_schema"),
            )
        )
        return {"success": True, "collection_id": collection_id}

    def ingest_documents(
        self,
        collection_id: str,
        records: list[dict[str, Any]],
        *,
        tenant_id: str | None = None,
        ingest_mode: str | None = None,
        embed_source: str = "native",
        allow_local_fallback: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Ingest documents for native server-side embedding (ADR-041).

        Canonical document-ingest surface — delegates to the GENERATED openapi
        op ``POST /api/v2/collections/{collection_id}/documents``
        (``_generated/rest/api/documents/ingest_documents.py``), NOT the
        hand-written ``adapters/rest_adapter.py`` path whose ``insert_document``
        silently falls back to an in-memory repository on any error.

        The server embeds each record's ``text`` natively (``X-Embed-Source:
        native``, the default) when no per-record ``vector`` is supplied; pass a
        ``vector`` to skip server embedding for that record.

        Per-call headers ride the generated client via ``Client.with_headers``
        (a fresh ``evolve`` client per call — no header accumulation across
        calls).

        Args:
            collection_id: Target collection id.
            records: Each is a ``dict`` with at least ``id`` and (for native
                embedding) ``text``; optionally ``metadata`` (``dict``) and
                ``vector`` (``list[float]``). An ``IngestDocument`` is passed
                through unchanged.
            tenant_id: Optional tenant scope -> ``X-Tenant-ID``.
            ingest_mode: Optional billing/metering mode -> ``X-Ingest-Mode``.
            embed_source: Embedding source -> ``X-Embed-Source`` (default
                ``"native"``). Pass ``""`` or ``None`` to omit the header.
            allow_local_fallback: If ``True`` and no server URL is configured,
                fall back to the in-memory ``DocumentRepository`` instead of
                raising. Default ``False`` — a server-backed client must surface
                failures, not swallow them into a local dict.

        Returns:
            The parsed ``IngestDocumentsResponse`` as a dict
            (``{"mode", "records", ...}``); ``{"mode": "local", ...}`` on the
            fallback path.

        Raises:
            ProximaDBError: No server configured and ``allow_local_fallback`` is
                False, or a record is malformed.
            NetworkError: Transport-level failure talking to the server.
            ServerError: The server returned a documented error envelope
                (HTTP 400/404) or an unexpected status.
        """
        from ._generated.rest.api.documents.ingest_documents import sync_detailed
        from ._generated.rest.models.error_response import ErrorResponse
        from ._generated.rest.models.ingest_document import IngestDocument
        from ._generated.rest.models.ingest_document_metadata import (
            IngestDocumentMetadata,
        )
        from ._generated.rest.models.ingest_documents_request import (
            IngestDocumentsRequest,
        )
        from ._generated.rest.types import UNSET

        # 1. Build the request body from ergonomic record dicts.
        body_records: list[IngestDocument] = []
        for rec in records:
            if isinstance(rec, IngestDocument):
                body_records.append(rec)
                continue
            if not isinstance(rec, dict):
                raise ProximaDBError(
                    "ingest_documents record must be a dict or IngestDocument, "
                    f"got {type(rec).__name__}"
                )
            if "id" not in rec:
                raise ProximaDBError("ingest_documents record missing 'id'")
            metadata = rec.get("metadata")
            body_records.append(
                IngestDocument(
                    id=str(rec["id"]),
                    text=rec.get("text", UNSET),
                    vector=rec.get("vector", UNSET),
                    metadata=(
                        IngestDocumentMetadata.from_dict(metadata)
                        if metadata is not None
                        else UNSET
                    ),
                )
            )
        request_body = IngestDocumentsRequest(records=body_records)

        # 2. Resolve the generated client. No server -> caller-controlled fallback.
        base_client = self._get_generated_document_client()
        if base_client is None:
            if not allow_local_fallback:
                raise ProximaDBError(
                    "ingest_documents requires a server URL (no PROXIMADB_URL "
                    "configured). Set allow_local_fallback=True to store records "
                    "in the in-memory repository."
                )
            repo = self._get_document_repository()
            inserted = []
            for rec in records:
                doc = rec if isinstance(rec, dict) else rec.to_dict()
                doc_id = rec.get("id") if isinstance(rec, dict) else rec.id
                inserted.append(repo.insert(collection_id, doc, doc_id))
            return {
                "mode": "local",
                "records": [
                    {
                        "id": getattr(d, "id", None),
                        "version": getattr(d, "version", None),
                    }
                    for d in inserted
                ],
            }

        # 3. Per-call headers on a fresh evolved client (no accumulation).
        per_call_headers: dict[str, str] = {}
        if embed_source:
            per_call_headers["X-Embed-Source"] = embed_source
        if tenant_id is not None:
            per_call_headers["X-Tenant-ID"] = str(tenant_id)
        if ingest_mode is not None:
            per_call_headers["X-Ingest-Mode"] = str(ingest_mode)
        client = base_client.with_headers(per_call_headers)

        # 4. Call the generated op and map outcomes.
        try:
            response = sync_detailed(
                collection_id=collection_id, client=client, body=request_body
            )
        except httpx.HTTPError as e:
            raise NetworkError(
                f"ingest_documents transport error: {e}", original_error=e
            ) from e

        parsed = response.parsed
        if isinstance(parsed, ErrorResponse):
            detail = getattr(getattr(parsed, "error", None), "message", None)
            raise ServerError(
                "ingest_documents rejected by server"
                f"{f': {detail}' if detail else ''}",
                status_code=response.status_code,
            )
        if parsed is None:
            # raise_on_unexpected_status=True makes this unreachable in practice;
            # guard anyway so a future regression can't silently return None.
            raise ServerError(
                "ingest_documents returned no body",
                status_code=response.status_code,
            )
        return parsed.to_dict()

    def get_document(
        self,
        collection_name: str,
        doc_id: str,
        projection: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any] | None:
        """Get a document by ID."""
        adapter_result = self._call_document_adapter(
            "get_document", collection_name, doc_id, projection, **kwargs
        )
        if adapter_result is not None:
            return adapter_result

        repo = self._get_document_repository()
        document = repo.get(collection_name, doc_id)
        if document is None:
            return None

        return {
            "id": document.id,
            "document": repo._project_document(document.content, projection),
            "version": document.version,
            "found": True,
        }

    def query_documents(
        self,
        collection_name: str,
        filter: dict[str, Any] | None = None,
        projection: list[str] | None = None,
        limit: int = 100,
        **kwargs,
    ) -> dict[str, Any]:
        """Query documents."""
        adapter_result = self._call_document_adapter(
            "query_documents",
            collection_name,
            filter=filter,
            projection=projection,
            limit=limit,
            **kwargs,
        )
        if adapter_result is not None:
            return adapter_result

        repo = self._get_document_repository()
        result = repo.query(
            collection_id=collection_name,
            filter=filter,
            projection=projection,
            limit=limit,
        )
        return {
            "documents": [document.to_dict() for document in result.documents],
            "total_count": result.total_count,
            "has_more": result.has_more,
        }

    def update_document(
        self,
        collection_name: str,
        doc_id: str,
        updates: list[dict[str, Any]],
        **kwargs,
    ) -> dict[str, Any]:
        """Update a document."""
        adapter_result = self._call_document_adapter(
            "update_document", collection_name, doc_id, updates, **kwargs
        )
        if adapter_result is not None:
            return adapter_result

        document = self._get_document_repository().update(
            collection_name, doc_id, updates
        )
        if document is None:
            return {"success": False, "id": doc_id}
        return {
            "success": True,
            "id": document.id,
            "new_version": document.version,
            "document": document.content,
        }

    def delete_document(self, collection_name: str, doc_id: str, **kwargs) -> bool:
        """Delete a document."""
        adapter_result = self._call_document_adapter(
            "delete_document", collection_name, doc_id, **kwargs
        )
        if adapter_result is not None:
            return bool(adapter_result)

        return self._get_document_repository().delete(collection_name, doc_id)

    def list_document_collections(self, **kwargs) -> list[dict[str, Any]]:
        """List document collections."""
        adapter_result = self._call_document_adapter(
            "list_document_collections", **kwargs
        )
        if adapter_result is not None:
            return adapter_result

        return self._get_document_repository().list_collections()

    def delete_document_collection(self, collection_name: str, **kwargs) -> bool:
        """Delete a document collection."""
        adapter_result = self._call_document_adapter(
            "delete_document_collection", collection_name, **kwargs
        )
        if adapter_result is not None:
            return bool(adapter_result)

        return self._get_document_repository().delete_collection(collection_name)

    def create_timeseries_collection(
        self, name: str, config: dict[str, Any] | None = None, **kwargs
    ) -> dict[str, Any]:
        """Create a time-series collection."""
        adapter_result = self._call_timeseries_adapter(
            "create_timeseries_collection", name, config, **kwargs
        )
        if adapter_result is not None:
            return adapter_result

        from .timeseries import TimeSeriesCollectionConfig

        collection_id = self._get_timeseries_repository().create_collection(
            TimeSeriesCollectionConfig(name=name, **(config or {}))
        )
        return {"success": True, "collection_id": collection_id}

    def ingest_timeseries(
        self,
        collection_name: str,
        points: list[dict[str, Any]],
        **kwargs,
    ) -> dict[str, Any]:
        """Ingest time-series points."""
        adapter_result = self._call_timeseries_adapter(
            "ingest_timeseries", collection_name, points, **kwargs
        )
        if adapter_result is not None:
            return adapter_result

        return self._get_timeseries_repository().ingest(collection_name, points)

    def query_timeseries(
        self,
        collection_name: str,
        start_time: str,
        end_time: str,
        aggregation: str | None = None,
        bucket_ms: int | None = None,
        tag_filters: dict[str, str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Query time-series data with local compatibility fallback."""
        adapter_result = self._call_timeseries_adapter(
            "query_timeseries",
            collection_name,
            start_time,
            end_time,
            aggregation=aggregation,
            bucket_ms=bucket_ms,
            tag_filters=tag_filters,
            **kwargs,
        )
        if adapter_result is not None:
            return adapter_result

        response = self._get_timeseries_repository().query(
            collection_id=collection_name,
            start_time=start_time,
            end_time=end_time,
            aggregation=aggregation,
            bucket_ms=bucket_ms,
            tag_filters=tag_filters,
            limit=kwargs.get("limit", 1000),
        )
        return response.to_dict()

    def list_timeseries_collections(self, **kwargs) -> list[dict[str, Any]]:
        """List time-series collections."""
        adapter_result = self._call_timeseries_adapter(
            "list_timeseries_collections", **kwargs
        )
        if adapter_result is not None:
            return adapter_result

        return self._get_timeseries_repository().list_collections()

    def delete_timeseries_collection(self, collection_name: str, **kwargs) -> bool:
        """Delete a time-series collection."""
        adapter_result = self._call_timeseries_adapter(
            "delete_timeseries_collection", collection_name, **kwargs
        )
        if adapter_result is not None:
            return bool(adapter_result)

        return self._get_timeseries_repository().delete_collection(collection_name)

    def hybrid_search(
        self,
        collection: str,
        text_query: str,
        query_vector: list[float],
        fusion_strategy: str = "rrf",
        top_k: int = 10,
        fusion_params: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Execute hybrid search with local compatibility fallback."""
        if self._active_protocol == Protocol.REST and self._adapter:
            try:
                return self._adapter.hybrid_search(
                    collection=collection,
                    text_query=text_query,
                    query_vector=query_vector,
                    fusion_strategy=fusion_strategy,
                    top_k=top_k,
                    **kwargs,
                )
            except Exception as e:
                logger.debug("REST hybrid search failed, using local fallback: %s", e)

        from .hybrid import (
            CascadeFusion,
            FusionStrategy,
            ProximaDBHybrid,
            WeightedFusion,
        )

        strategy: str | FusionStrategy | Any = fusion_strategy
        if isinstance(fusion_strategy, str):
            normalized = fusion_strategy.lower()
            if normalized == "weighted_linear":
                strategy = WeightedFusion(alpha=(fusion_params or {}).get("alpha", 0.5))
            elif normalized == "cascade":
                strategy = CascadeFusion(
                    threshold=(fusion_params or {}).get("threshold", 0.0)
                )
            else:
                strategy = FusionStrategy.RRF

        hybrid = ProximaDBHybrid(self)
        start_time = time.time()
        results = hybrid.search(
            vector_collection=collection,
            query_vector=query_vector,
            text_query=text_query,
            fusion_strategy=strategy,
            top_k=top_k,
            filters=kwargs.get("filters"),
        )
        total_time_ms = int((time.time() - start_time) * 1000)
        return {
            "results": [result.to_dict() for result in results],
            "metrics": {
                "total_time_ms": total_time_ms,
                "bm25_search_time_ms": 0,
                "vector_search_time_ms": 0,
            },
        }

    def _record_payload_from_legacy_input(
        self, record: VectorRecord | dict[str, Any], index: int = 0
    ) -> dict[str, Any]:
        """Normalize legacy vector-shaped inputs into the record write shape."""
        if isinstance(record, dict):
            payload = dict(record)
            if "props" not in payload and "metadata" in payload:
                payload["props"] = payload.pop("metadata")
            payload.setdefault("id", payload.get("oid") or f"record_{index}")
            return payload

        payload: dict[str, Any] = {
            "id": record.id or f"record_{index}",
            "vector": (
                record.vector.tolist()
                if hasattr(record.vector, "tolist")
                else list(record.vector or [])
            ),
            "props": dict(record.metadata or {}),
        }
        source = getattr(record, "source", None)
        if source:
            payload["source"] = source
            payload["text_fields"] = [{"name": "text", "content": source}]
        return payload

    def _batch_result_to_vector_response(
        self, result: BatchResult, operation: str, records: list[dict[str, Any]]
    ) -> VectorOperationResponse:
        return VectorOperationResponse(
            success=result.success,
            operation=operation,
            metrics=OperationMetrics(
                total_processed=result.total,
                successful_count=result.success,
                failed_count=result.failed,
            ),
            vector_ids=[record["id"] for record in records if record.get("id")],
            error_message="; ".join(result.errors) if result.errors else None,
        )

    def insert_records(
        self,
        collection_id: str,
        records: list[VectorRecord | dict[str, Any]],
        **kwargs: Any,
    ) -> BatchResult:
        """Insert ProximaRecord-shaped payloads through the active transport."""
        record_payloads = [
            self._record_payload_from_legacy_input(record, index)
            for index, record in enumerate(records)
        ]
        if not record_payloads:
            raise ValueError("'records' must not be empty")

        if self._prefer_local_fallback:
            vector_records = [
                VectorRecord(
                    id=record.get("id"),
                    vector=record.get("vector") or [],
                    metadata=record.get("props") or {},
                    source=record.get("source"),
                )
                for record in record_payloads
            ]
            self._store_local_vector_records(collection_id, vector_records)
            return BatchResult(total=len(record_payloads), success=len(record_payloads))

        target = self._adapter or self._rest_client or self._grpc_client
        if target and hasattr(target, "insert_records"):
            return target.insert_records(collection_id, record_payloads, **kwargs)

        raise NotImplementedError(
            "insert_records requires a record-native adapter or protocol client"
        )

    def upsert_records(
        self,
        collection_id: str,
        records: list[VectorRecord | dict[str, Any]],
        **kwargs: Any,
    ) -> BatchResult:
        """Upsert ProximaRecord-shaped payloads through the active transport."""
        record_payloads = [
            self._record_payload_from_legacy_input(record, index)
            for index, record in enumerate(records)
        ]
        if not record_payloads:
            raise ValueError("'records' must not be empty")

        target = self._adapter or self._rest_client or self._grpc_client
        if target and hasattr(target, "upsert_records"):
            return target.upsert_records(collection_id, record_payloads, **kwargs)
        return self.insert_records(collection_id, record_payloads, **kwargs)

    def insert_vectors(
        self,
        collection_id: str,
        # Backward compatibility: support old calling style
        vectors: list[list[float]] | list[VectorRecord] | np.ndarray | None = None,
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
        # New API parameter
        records: list[VectorRecord] | None = None,
        **kwargs,
    ) -> VectorOperationResponse:
        """Insert vectors into a collection

        Supports both new API (VectorRecord objects) and old API (separate vectors/ids/metadata)

        Note: For quantized collections, all vectors MUST have unique IDs to track
        quantized representations across storage and indexes.
        """
        # Normalize input: convert old API to new API (VectorRecord objects)
        if vectors is not None:
            # Handle numpy arrays first
            if hasattr(vectors, "tolist"):
                vectors = vectors.tolist()

            # Check if vectors is a list of VectorRecord objects (new API called with vectors param)
            if (
                hasattr(vectors, "__len__")
                and len(vectors) > 0
                and hasattr(vectors[0], "vector")
                and hasattr(vectors[0], "id")
            ):
                records = vectors
            else:
                # Old API: convert vectors/ids/metadata to VectorRecord objects
                records = []
                for i, vector in enumerate(vectors):
                    record = VectorRecord(
                        id=ids[i] if ids and i < len(ids) else None,
                        vector=(
                            vector
                            if isinstance(vector, list)
                            else (
                                vector.tolist()
                                if hasattr(vector, "tolist")
                                else list(vector)
                            )
                        ),
                        metadata=metadata[i] if metadata and i < len(metadata) else {},
                    )
                    records.append(record)

        if records is None or (hasattr(records, "__len__") and len(records) == 0):
            raise ValueError("Either 'records' or 'vectors' must be provided")

        record_payloads = [
            self._record_payload_from_legacy_input(record, index)
            for index, record in enumerate(records)
        ]
        try:
            batch_result = self.insert_records(collection_id, record_payloads, **kwargs)
            return self._batch_result_to_vector_response(
                batch_result, "INSERT", record_payloads
            )
        except NotImplementedError:
            pass

        if self._prefer_local_fallback:
            self._store_local_vector_records(collection_id, list(records))
            success_value: bool | int = (
                len(records) if self._active_protocol == Protocol.REST else True
            )
            return VectorOperationResponse(
                success=success_value,
                operation="INSERT",
                metrics=OperationMetrics(
                    total_processed=len(records),
                    successful_count=len(records),
                    failed_count=0,
                ),
                vector_ids=[record.id for record in records if record.id is not None],
            )

        # Use adapter if available (reduces protocol-specific code duplication)
        if self._adapter:
            if not self._prefer_local_fallback:
                try:
                    result = self._adapter.insert_vectors(
                        collection_id, records, **kwargs
                    )
                    if self._active_protocol == Protocol.EMBEDDED:
                        self._store_local_vector_records(collection_id, list(records))
                    return result
                except Exception as e:
                    self._activate_local_fallback(e)
                    logger.debug(
                        "Insert vectors failed, using local fallback for %s: %s",
                        collection_id,
                        e,
                    )
            self._store_local_vector_records(collection_id, list(records))
            success_value: bool | int = (
                len(records) if self._active_protocol == Protocol.REST else True
            )
            return VectorOperationResponse(
                success=success_value,
                operation="INSERT",
                metrics=OperationMetrics(
                    total_processed=len(records),
                    successful_count=len(records),
                    failed_count=0,
                ),
                vector_ids=[record.id for record in records if record.id is not None],
            )

        # Fallback to raw client for backward compatibility (legacy code path)
        # Check if collection has quantization enabled
        try:
            collection = self.get_collection(collection_id)
            if collection and hasattr(collection, "config"):
                config = collection.config
                if (
                    hasattr(config, "quantization_config")
                    and config.quantization_config
                ):
                    if (
                        hasattr(config.quantization_config, "enabled")
                        and config.quantization_config.enabled
                    ):
                        # Quantization is enabled - validate IDs
                        needs_id_validation = True
                        logger.info(
                            f"Collection '{collection_id}' has quantization enabled - validating vector IDs"
                        )
                    else:
                        needs_id_validation = False
                else:
                    needs_id_validation = False
            else:
                needs_id_validation = False
        except Exception as e:
            # If we can't check, proceed without validation
            logger.debug(
                f"Could not check quantization status for collection {collection_id}: {e}"
            )
            needs_id_validation = False

        # Handle backward compatibility: convert old API to new API
        if vectors is not None:
            # Handle numpy arrays first
            if hasattr(vectors, "tolist"):
                vectors = vectors.tolist()

            # Check if vectors is a list of VectorRecord objects (new API called with vectors param)
            if (
                hasattr(vectors, "__len__")
                and len(vectors) > 0
                and hasattr(vectors[0], "vector")
                and hasattr(vectors[0], "id")
            ):
                records = vectors
            else:
                # Old API: convert vectors/ids/metadata to VectorRecord objects
                records = []

                for i, vector in enumerate(vectors):
                    record = VectorRecord(
                        id=ids[i] if ids and i < len(ids) else None,
                        vector=(
                            vector
                            if isinstance(vector, list)
                            else (
                                vector.tolist()
                                if hasattr(vector, "tolist")
                                else list(vector)
                            )
                        ),
                        metadata=metadata[i] if metadata and i < len(metadata) else {},
                    )
                    records.append(record)
        elif records is None:
            # Neither vectors nor records provided
            pass

        # Handle numpy arrays and other array-like objects
        if (
            records is None
            or (hasattr(records, "__len__") and len(records) == 0)
            or (not hasattr(records, "__len__") and not records)
        ):
            raise ValueError("Either 'records' or 'vectors' must be provided")

        # Validate IDs for quantized collections
        if needs_id_validation:
            for i, record in enumerate(records):
                if not record.id or record.id.strip() == "":
                    raise ValueError(
                        f"Vector at index {i} missing ID. "
                        f"Collection '{collection_id}' has quantization enabled. "
                        f"All vectors MUST have unique IDs for tracking quantized representations."
                    )
            logger.debug(
                f"✅ ID validation passed for {len(records)} vectors in quantized collection {collection_id}"
            )

        # Estimate data size for routing
        data_size_hint = (
            len(records) * len(records[0].vector) * 4
            if records and records[0].vector
            else 1000
        )  # Rough estimate
        operation_name = (
            "bulk_insert_vectors" if len(records) > 10 else "insert_vectors"
        )

        start_time = time.time()

        try:
            # Get appropriate client for this operation
            client = self._get_client_for_operation(
                operation_name=operation_name,
                data_size_hint=data_size_hint,
                context={"collection_id": collection_id, "vector_count": len(records)},
            )

            # Determine protocol and execute
            if client == self._grpc_client:
                protocol_used = Protocol.GRPC
                # Convert Pydantic VectorRecord to dict format for gRPC client
                vector_dicts = []
                for record in records:
                    vector_dict = {"vector": record.vector, "metadata": record.metadata}
                    if record.id:
                        vector_dict["id"] = record.id
                    # Add all timestamp fields (support both _ms and non-_ms versions)
                    if hasattr(record, "timestamp_ms") and record.timestamp_ms:
                        vector_dict["timestamp_ms"] = record.timestamp_ms
                    elif hasattr(record, "timestamp") and record.timestamp:
                        vector_dict["timestamp"] = record.timestamp

                    if hasattr(record, "updated_at_ms") and record.updated_at_ms:
                        vector_dict["updated_at_ms"] = record.updated_at_ms
                    elif hasattr(record, "updated_at") and record.updated_at:
                        vector_dict["updated_at"] = record.updated_at

                    if hasattr(record, "expires_at_ms") and record.expires_at_ms:
                        vector_dict["expires_at_ms"] = record.expires_at_ms
                    elif hasattr(record, "expires_at") and record.expires_at:
                        vector_dict["expires_at"] = record.expires_at

                    # Add version field
                    if hasattr(record, "version") and record.version is not None:
                        vector_dict["version"] = record.version

                    # Add source field (original content)
                    if hasattr(record, "source") and record.source:
                        vector_dict["source"] = record.source

                    vector_dicts.append(vector_dict)

                proto_response = client.insert_vectors(collection_id, vector_dicts)
                # Convert proto response to Pydantic (simplified for now)
                # Handle case where metrics might not be present in the response
                metrics = None
                if hasattr(proto_response, "metrics") and proto_response.metrics:
                    metrics = OperationMetrics(
                        total_processed=proto_response.metrics.total_processed,
                        successful_count=proto_response.metrics.successful_count,
                        failed_count=proto_response.metrics.failed_count,
                    )
                else:
                    # Default metrics if not provided
                    metrics = OperationMetrics(
                        total_processed=len(vector_dicts),
                        successful_count=(
                            len(vector_dicts) if proto_response.success else 0
                        ),
                        failed_count=0 if proto_response.success else len(vector_dicts),
                    )

                result = VectorOperationResponse(
                    success=proto_response.success, operation="insert", metrics=metrics
                )

            elif client == self._rest_client:
                protocol_used = Protocol.REST
                # Legacy fallback for clients without record-native insert support.
                # Convert VectorRecord to dict format with ALL fields.
                vector_dicts = []
                for record in records:
                    vector_dict = {
                        "vector": record.vector,
                        "metadata": record.metadata or {},
                    }
                    if record.id:
                        vector_dict["id"] = record.id

                    # Add timestamp fields (support both _ms and non-_ms versions)
                    if hasattr(record, "timestamp_ms") and record.timestamp_ms:
                        vector_dict["timestamp"] = record.timestamp_ms
                    elif hasattr(record, "timestamp") and record.timestamp:
                        vector_dict["timestamp"] = record.timestamp

                    if hasattr(record, "updated_at_ms") and record.updated_at_ms:
                        vector_dict["updated_at"] = record.updated_at_ms
                    elif hasattr(record, "updated_at") and record.updated_at:
                        vector_dict["updated_at"] = record.updated_at

                    if hasattr(record, "expires_at_ms") and record.expires_at_ms:
                        vector_dict["expires_at"] = record.expires_at_ms
                    elif hasattr(record, "expires_at") and record.expires_at:
                        vector_dict["expires_at"] = record.expires_at

                    # Add version field
                    if hasattr(record, "version") and record.version is not None:
                        vector_dict["version"] = record.version

                    # Add source field (original content)
                    if hasattr(record, "source") and record.source:
                        vector_dict["source"] = record.source

                    vector_dicts.append(vector_dict)

                # Send VectorRecord dicts to REST API via new vector_records parameter
                result = client.insert_vectors(
                    collection_id,
                    vectors=[],
                    ids=[],
                    metadata=[],
                    vector_records=vector_dicts,
                )

            else:
                # Fallback to active protocol
                protocol_used = self._active_protocol
                if protocol_used == Protocol.GRPC:
                    # Similar to gRPC path above
                    vector_dicts = []
                    for record in records:
                        vector_dict = {
                            "vector": record.vector,
                            "metadata": record.metadata,
                        }
                        if record.id:
                            vector_dict["id"] = record.id
                        # Add all timestamp fields (support both _ms and non-_ms versions)
                        if hasattr(record, "timestamp_ms") and record.timestamp_ms:
                            vector_dict["timestamp_ms"] = record.timestamp_ms
                        elif hasattr(record, "timestamp") and record.timestamp:
                            vector_dict["timestamp"] = record.timestamp

                        if hasattr(record, "updated_at_ms") and record.updated_at_ms:
                            vector_dict["updated_at_ms"] = record.updated_at_ms
                        elif hasattr(record, "updated_at") and record.updated_at:
                            vector_dict["updated_at"] = record.updated_at

                        if hasattr(record, "expires_at_ms") and record.expires_at_ms:
                            vector_dict["expires_at_ms"] = record.expires_at_ms
                        elif hasattr(record, "expires_at") and record.expires_at:
                            vector_dict["expires_at"] = record.expires_at

                        # Add version field
                        if hasattr(record, "version") and record.version is not None:
                            vector_dict["version"] = record.version

                        # Add source field (original content)
                        if hasattr(record, "source") and record.source:
                            vector_dict["source"] = record.source

                        vector_dicts.append(vector_dict)

                    proto_response = client.insert_vectors(collection_id, vector_dicts)
                    metrics = None
                    if hasattr(proto_response, "metrics") and proto_response.metrics:
                        metrics = OperationMetrics(
                            total_processed=proto_response.metrics.total_processed,
                            successful_count=proto_response.metrics.successful_count,
                            failed_count=proto_response.metrics.failed_count,
                        )
                    else:
                        metrics = OperationMetrics(
                            total_processed=len(vector_dicts),
                            successful_count=(
                                len(vector_dicts) if proto_response.success else 0
                            ),
                            failed_count=(
                                0 if proto_response.success else len(vector_dicts)
                            ),
                        )

                    result = VectorOperationResponse(
                        success=proto_response.success,
                        operation="insert",
                        metrics=metrics,
                    )
                else:
                    # REST fallback
                    vectors = [r.vector for r in records]
                    ids = [r.id for r in records if r.id]
                    metadata = [r.metadata for r in records]

                    if not ids:
                        ids = [f"vec_{i}" for i in range(len(vectors))]

                    result = client.insert_vectors(
                        collection_id, vectors, ids, metadata
                    )

            # Record successful operation
            response_time = (time.time() - start_time) * 1000
            throughput = len(records) / (
                (time.time() - start_time) + 0.001
            )  # Add small value to avoid division by zero
            self._record_operation_result(
                operation_name,
                protocol_used,
                True,
                response_time,
                throughput_ops_per_sec=throughput,
            )

            return result

        except Exception as e:
            # Record failed operation
            response_time = (time.time() - start_time) * 1000
            protocol_used = getattr(self, "_active_protocol", Protocol.REST)
            self._record_operation_result(
                operation_name, protocol_used, False, response_time, str(e)
            )
            raise

    def upsert_vectors(
        self, collection_id: str, records: list[VectorRecord | dict[str, Any]]
    ) -> VectorOperationResponse:
        """Compatibility alias for record-native upserts."""
        record_payloads = [
            self._record_payload_from_legacy_input(record, index)
            for index, record in enumerate(records)
        ]
        try:
            batch_result = self.upsert_records(collection_id, record_payloads)
            return self._batch_result_to_vector_response(
                batch_result, "UPSERT", record_payloads
            )
        except NotImplementedError:
            pass

        if self._prefer_local_fallback:
            self._store_local_vector_records(collection_id, list(records))
            success_value: bool | int = (
                len(records) if self._active_protocol == Protocol.REST else True
            )
            return VectorOperationResponse(
                success=success_value,
                operation="UPSERT",
                metrics=OperationMetrics(
                    total_processed=len(records),
                    successful_count=len(records),
                    failed_count=0,
                    updated_count=len(records),
                ),
                vector_ids=[record.id for record in records if record.id is not None],
            )

        # Use adapter if available (reduces protocol-specific code duplication)
        if self._adapter:
            if not self._prefer_local_fallback:
                try:
                    result = self._adapter.upsert_vectors(collection_id, records)
                    if self._active_protocol == Protocol.EMBEDDED:
                        self._store_local_vector_records(collection_id, list(records))
                    return result
                except Exception as e:
                    self._activate_local_fallback(e)
                    logger.debug(
                        "Upsert vectors failed, using local fallback for %s: %s",
                        collection_id,
                        e,
                    )
            self._store_local_vector_records(collection_id, list(records))
            success_value: bool | int = (
                len(records) if self._active_protocol == Protocol.REST else True
            )
            return VectorOperationResponse(
                success=success_value,
                operation="UPSERT",
                metrics=OperationMetrics(
                    total_processed=len(records),
                    successful_count=len(records),
                    failed_count=0,
                    updated_count=len(records),
                ),
                vector_ids=[record.id for record in records if record.id is not None],
            )

        # Fallback to raw client for backward compatibility
        if self._active_protocol == Protocol.GRPC:
            # Convert Pydantic VectorRecord to dict format for gRPC client
            vector_dicts = []
            for record in records:
                vector_dict = {"vector": record.vector, "metadata": record.metadata}
                if record.id:
                    vector_dict["id"] = record.id
                # Add all timestamp fields (support both _ms and non-_ms versions)
                if hasattr(record, "timestamp_ms") and record.timestamp_ms:
                    vector_dict["timestamp_ms"] = record.timestamp_ms
                elif hasattr(record, "timestamp") and record.timestamp:
                    vector_dict["timestamp"] = record.timestamp

                if hasattr(record, "updated_at_ms") and record.updated_at_ms:
                    vector_dict["updated_at_ms"] = record.updated_at_ms
                elif hasattr(record, "updated_at") and record.updated_at:
                    vector_dict["updated_at"] = record.updated_at

                if hasattr(record, "expires_at_ms") and record.expires_at_ms:
                    vector_dict["expires_at_ms"] = record.expires_at_ms
                elif hasattr(record, "expires_at") and record.expires_at:
                    vector_dict["expires_at"] = record.expires_at

                # Add version field
                if hasattr(record, "version") and record.version is not None:
                    vector_dict["version"] = record.version

                # Add source field (original content)
                if hasattr(record, "source") and record.source:
                    vector_dict["source"] = record.source

                vector_dicts.append(vector_dict)

            proto_response = self._client.insert_vectors(
                collection_id, vector_dicts, upsert=True
            )
            # Convert proto response to Pydantic (simplified for now)
            # Handle case where metrics might not be present in the response
            metrics = None
            if hasattr(proto_response, "metrics") and proto_response.metrics:
                metrics = OperationMetrics(
                    total_processed=proto_response.metrics.total_processed,
                    successful_count=proto_response.metrics.successful_count,
                    failed_count=proto_response.metrics.failed_count,
                    updated_count=(
                        proto_response.metrics.updated_count
                        if hasattr(proto_response.metrics, "updated_count")
                        else 0
                    ),
                )
            else:
                # Default metrics if not provided
                metrics = OperationMetrics(
                    total_processed=len(vector_dicts),
                    successful_count=len(vector_dicts) if proto_response.success else 0,
                    failed_count=0 if proto_response.success else len(vector_dicts),
                    updated_count=len(vector_dicts) if proto_response.success else 0,
                )

            return VectorOperationResponse(
                success=proto_response.success, operation="upsert", metrics=metrics
            )
        else:
            return self._client.upsert_vectors(collection_id, records)

    def search(
        self,
        collection_id: str,
        vector: list[float] | np.ndarray,
        top_k: int = 10,
        metadata_filter: Union[dict[str, Any], "FilterBuilder"] | None = None,
        include_metadata: bool = True,
        include_vectors: bool = False,
        **kwargs,
    ) -> list[SearchResult]:
        """Search for similar vectors

        Args:
            collection_id: Target collection ID
            vector: Query vector
            top_k: Number of results to return
            metadata_filter: Optional metadata filter
            include_metadata: Include metadata in results
            include_vectors: Include vectors in results
            **kwargs: Additional search parameters

        Returns:
            List of search results ordered by similarity
        """
        # Validate top_k
        if top_k <= 0:
            raise ProximaDBError(f"top_k must be positive, got {top_k}")

        return self.search_single(
            collection_id=collection_id,
            vector=vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
            include_metadata=include_metadata,
            include_vectors=include_vectors,
            **kwargs,
        )

    def search_single(
        self,
        collection_id: str,
        vector: list[float] | np.ndarray,
        top_k: int = 10,
        metadata_filter: Union[dict[str, Any], "FilterBuilder"] | None = None,
        optimization_level: str = "high",
        use_storage_aware: bool = True,
        quantization_level: str = "FP32",
        enable_simd: bool = True,
        **kwargs,
    ) -> list[SearchResult]:
        """Search for similar vectors with storage-aware optimizations

        Args:
            collection_id: Target collection ID
            vector: Query vector
            top_k: Number of results to return
            metadata_filter: Metadata filter conditions
            optimization_level: Search optimization level ('high', 'medium', 'low')
            use_storage_aware: Enable storage-aware polymorphic search
            quantization_level: Vector quantization level for search
            enable_simd: Enable SIMD vectorization optimizations
            **kwargs: Additional search parameters

        Returns:
            List of search results ordered by similarity
        """
        if self._prefer_local_fallback:
            return self._search_local_vectors(
                collection_id=collection_id,
                vector=vector,
                top_k=top_k,
                metadata_filter=metadata_filter,
                include_metadata=kwargs.get("include_metadata", True),
                include_vectors=kwargs.get("include_vectors", False),
            )

        # Use adapter if available (reduces protocol-specific code duplication)
        if self._adapter:
            if not self._prefer_local_fallback:
                try:
                    return self._adapter.search(
                        collection_id=collection_id,
                        query_vector=vector,
                        top_k=top_k,
                        filter=metadata_filter,
                        include_vectors=kwargs.get("include_vectors", False),
                        include_metadata=kwargs.get("include_metadata", True),
                        **{
                            k: v
                            for k, v in kwargs.items()
                            if k not in ("include_vectors", "include_metadata")
                        },
                    )
                except Exception as e:
                    self._activate_local_fallback(e)
                    logger.debug(
                        "Search failed, using local fallback for %s: %s",
                        collection_id,
                        e,
                    )
            return self._search_local_vectors(
                collection_id=collection_id,
                vector=vector,
                top_k=top_k,
                metadata_filter=metadata_filter,
                include_metadata=kwargs.get("include_metadata", True),
                include_vectors=kwargs.get("include_vectors", False),
            )

        # Fallback to raw client for backward compatibility
        if self._active_protocol == Protocol.GRPC:
            # Convert vector to list if numpy array
            if isinstance(vector, np.ndarray):
                vector = vector.tolist()

            # Add search hints for gRPC
            search_hints = kwargs.get("search_hints", {})
            search_hints.update(
                {
                    "predicate_pushdown": True,
                    "use_bloom_filters": True,
                    "use_clustering": True,
                    "quantization_level": quantization_level,
                    "parallel_search": True,
                    "engine_specific": {
                        "optimization_level": optimization_level,
                        "enable_simd": enable_simd,
                        "prefer_indices": True,
                        "storage_aware": use_storage_aware,
                    },
                }
            )

            # grpc_sync.search_vectors already returns List[SearchResult]
            results = self._client.search_vectors(
                collection_id=collection_id,
                query_vectors=[vector],
                top_k=top_k,
                metadata_filters=metadata_filter,
                include_metadata=kwargs.get("include_metadata", True),
                include_vectors=kwargs.get("include_vectors", False),
                # Note: search_hints would need to be converted to SearchParameters proto
            )

            # Results are already SearchResult objects, just return them
            return results
        else:
            # For REST, use search method (filter out unsupported parameters)
            # Remove optimization_hints and other parameters not supported by REST client
            filtered_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k
                not in {
                    "optimization_hints",
                    "enable_two_stage_search",
                    "quantization_hint",
                    "candidate_multiplier",
                    "enable_parallel_search",
                }
            }

            return self._client.search(
                collection_id=collection_id,
                vector=vector,
                top_k=top_k,
                metadata_filter=metadata_filter,
                optimization_level=optimization_level,
                use_storage_aware=use_storage_aware,
                quantization_level=quantization_level,
                enable_simd=enable_simd,
                **filtered_kwargs,
            )

    def search_envelope(
        self,
        collection_id: str,
        vector: list[float] | np.ndarray,
        top_k: int = 10,
        include_vectors: bool = False,
        include_metadata: bool = True,
        **kwargs,
    ):
        """Run REST OpenAPI v2 search and return the paged search envelope."""
        if hasattr(vector, "tolist"):
            vector = vector.tolist()

        if self._active_protocol == Protocol.REST and hasattr(
            self._client, "search_envelope"
        ):
            return self._client.search_envelope(
                collection_id=collection_id,
                vector=vector,
                top_k=top_k,
                include_vectors=include_vectors,
                include_metadata=include_metadata,
                **kwargs,
            )

        raise ProximaDBError(
            "search_envelope requires the REST OpenAPI v2 search surface"
        )

    def search_iter(
        self,
        collection_id: str,
        vector: list[float] | np.ndarray,
        top_k: int = 10,
        include_metadata: bool = True,
        include_vectors: bool = False,
        page_limit: int | None = None,
    ):
        """Iterate across pages of search results. Uses SKS cursors on REST when available.

        For gRPC (no pagination), yields only the first page of results.
        """
        if self._active_protocol == Protocol.REST and hasattr(
            self._client, "search_envelope"
        ):
            env = self._client.search_envelope(
                collection_id=collection_id,
                vector=vector,
                top_k=top_k,
                include_vectors=include_vectors,
                include_metadata=include_metadata,
            )
            count = 0
            for item in env.items:
                yield item
                count += 1
            cursor = env.cursor
            pages = 1
            while (
                env.has_more and cursor and (page_limit is None or pages < page_limit)
            ):
                env = self._client.search_next_page(
                    collection_id,
                    cursor,
                    include_vectors=include_vectors,
                    include_metadata=include_metadata,
                )
                for item in env.items:
                    yield item
                    count += 1
                cursor = env.cursor
                pages += 1
        else:
            # gRPC: yield single page
            for item in self.search_single(
                collection_id=collection_id,
                vector=vector,
                top_k=top_k,
                include_metadata=include_metadata,
                include_vectors=include_vectors,
            ):
                yield item

    # The generic search method is defined above and forwards to search_single

    #: Server-enforced upper bound on a single scan page (SCAN_RECORDS_MAX_PAGE).
    SCAN_MAX_PAGE = 10_000
    #: Default safety bound on the total rows ``scan`` will accumulate across
    #: pages before stopping, so an unbounded scan can't exhaust client memory.
    SCAN_DEFAULT_MAX_ROWS = 1_000_000

    def scan_page(
        self,
        collection_id: str,
        *,
        filter: list[dict[str, Any]] | dict[str, Any] | None = None,
        limit: int | None = None,
        cursor: str | None = None,
        include_vectors: bool = False,
        include_text: bool = False,
    ) -> tuple[list[SearchResult], str | None]:
        """Scan a single page of records via the vector-free metadata-scan endpoint.

        Returns ``(records, next_cursor)``. ``next_cursor`` is ``None`` at
        end-of-scan. ``filter`` is the scan endpoint's metadata-filter shape:
        the typed list ``[{"field", "op", "value"}]`` or an equality map
        ``{field: value}``; it is pushed into the scan predicate server-side
        (applied before the limit). Unlike :meth:`search`, this is a true
        metadata scan with no ANN ordering or similarity cap.

        Requires the REST transport (the scan endpoint is REST-only).
        """
        target = self._rest_client or (
            self._client if hasattr(self._client, "scan_records") else None
        )
        if target is None or not hasattr(target, "scan_records"):
            # Stand up a REST client on demand (the scan endpoint is REST-only).
            if self._rest_client is None:
                self._rest_client = self._create_rest_client()
            target = self._rest_client
        return target.scan_records(
            collection_id,
            filter=filter,
            limit=limit,
            cursor=cursor,
            include_vector=include_vectors,
            include_text=include_text,
        )

    def scan(
        self,
        collection_id: str,
        *,
        filter: list[dict[str, Any]] | dict[str, Any] | None = None,
        page_size: int | None = None,
        max_rows: int | None = None,
        include_vectors: bool = False,
        include_text: bool = False,
    ) -> list[SearchResult]:
        """Scan ALL records matching ``filter`` via cursor pagination.

        Repeatedly calls :meth:`scan_page`, following the returned ``next_cursor``
        until the server reports end-of-scan (``None``). This returns the full
        matching set — not a bounded ANN window.

        Args:
            filter: Scan-endpoint metadata filter (typed list ``[{field,op,value}]``
                or equality map ``{field: value}``), pushed down server-side.
            page_size: Rows per page request (clamped to ``SCAN_MAX_PAGE``).
            max_rows: Safety bound on total accumulated rows. Defaults to
                :attr:`SCAN_DEFAULT_MAX_ROWS`; the loop stops once it is reached
                even if the server has more pages.
            include_vectors: Include vector embeddings in each record.
            include_text: Include TEXT fields in each record.
        """
        cap = self.SCAN_DEFAULT_MAX_ROWS if max_rows is None else max_rows
        per_page = self.SCAN_MAX_PAGE if page_size is None else page_size
        per_page = max(1, min(per_page, self.SCAN_MAX_PAGE))

        out: list[SearchResult] = []
        cursor: str | None = None
        seen_cursors: set[str] = set()
        while len(out) < cap:
            remaining = cap - len(out)
            page_limit = min(per_page, remaining)
            records, cursor = self.scan_page(
                collection_id,
                filter=filter,
                limit=page_limit,
                cursor=cursor,
                include_vectors=include_vectors,
                include_text=include_text,
            )
            out.extend(records)
            if not cursor:
                break
            # Defensive: a server that echoes the same cursor would loop forever.
            if cursor in seen_cursors:
                logger.warning(
                    "scan: repeated cursor %s for %s; stopping to avoid a loop",
                    cursor,
                    collection_id,
                )
                break
            seen_cursors.add(cursor)
            # A page that returned nothing but still handed back a cursor would
            # otherwise spin; stop on an empty page too.
            if not records:
                break
        return out[:cap]

    def search_batch(
        self,
        collection_id: str,
        vectors: list[list[float]] | np.ndarray,
        top_k: int = 10,
        metadata_filter: Union[dict[str, Any], "FilterBuilder"] | None = None,
        **kwargs,
    ) -> list[list[SearchResult]]:
        """Search multiple queries in batch with optimizations

        Args:
            collection_id: Target collection ID
            vectors: Array of query vectors
            top_k: Number of results per query
            metadata_filter: Metadata filter conditions
            **kwargs: Additional search parameters

        Returns:
            List of search results for each query
        """
        # Convert to list if numpy array
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()

        # Perform batch search
        all_results = []
        for vector in vectors:
            results = self.search_single(
                collection_id=collection_id,
                vector=vector,
                top_k=top_k,
                metadata_filter=metadata_filter,
                **kwargs,
            )
            all_results.append(results)

        return all_results

    def delete_vectors(
        self, collection_id: str, vector_ids: list[str]
    ) -> VectorOperationResponse:
        """Delete vectors from a collection"""
        if self._prefer_local_fallback:
            deleted = self._delete_local_vector_records(collection_id, vector_ids)
            return VectorOperationResponse(
                success=True,
                operation="DELETE",
                metrics=OperationMetrics(
                    total_processed=len(vector_ids),
                    successful_count=deleted,
                    failed_count=max(0, len(vector_ids) - deleted),
                ),
                vector_ids=vector_ids,
            )

        # Use adapter if available (reduces protocol-specific code duplication)
        if self._adapter:
            if not self._prefer_local_fallback:
                try:
                    result = self._adapter.delete_vectors(collection_id, vector_ids)
                    if self._active_protocol == Protocol.EMBEDDED:
                        self._delete_local_vector_records(collection_id, vector_ids)
                    return result
                except Exception as e:
                    self._activate_local_fallback(e)
                    logger.debug(
                        "Delete vectors failed, using local fallback for %s: %s",
                        collection_id,
                        e,
                    )
            deleted = self._delete_local_vector_records(collection_id, vector_ids)
            return VectorOperationResponse(
                success=True,
                operation="DELETE",
                metrics=OperationMetrics(
                    total_processed=len(vector_ids),
                    successful_count=deleted,
                    failed_count=max(0, len(vector_ids) - deleted),
                ),
                vector_ids=vector_ids,
            )

        # Fallback to raw client for backward compatibility
        if self._active_protocol == Protocol.GRPC:
            proto_response = self._client.delete_vectors(collection_id, vector_ids)
            # Handle case where proto_response is a dict or object
            if isinstance(proto_response, dict):
                success = proto_response.get("success", True)
                metrics_data = proto_response.get("metrics", {})
                metrics = OperationMetrics(
                    total_processed=metrics_data.get(
                        "total_processed", len(vector_ids)
                    ),
                    successful_count=metrics_data.get(
                        "successful_count", len(vector_ids) if success else 0
                    ),
                    failed_count=metrics_data.get(
                        "failed_count", 0 if success else len(vector_ids)
                    ),
                )
            else:
                # Handle case where metrics might not be present in the response
                metrics = None
                if hasattr(proto_response, "metrics") and proto_response.metrics:
                    metrics = OperationMetrics(
                        total_processed=proto_response.metrics.total_processed,
                        successful_count=proto_response.metrics.successful_count,
                        failed_count=proto_response.metrics.failed_count,
                    )
                else:
                    # Default metrics if not provided
                    metrics = OperationMetrics(
                        total_processed=len(vector_ids),
                        successful_count=(
                            len(vector_ids) if proto_response.success else 0
                        ),
                        failed_count=0 if proto_response.success else len(vector_ids),
                    )
                success = proto_response.success

            return VectorOperationResponse(
                success=success, operation="delete", metrics=metrics
            )
        else:
            return self._client.delete_vectors(collection_id, vector_ids)

    def get_vector(
        self,
        collection_id: str,
        vector_id: str,
        include_vector: bool = True,
        include_metadata: bool = True,
    ):  # Changed return type to be flexible
        """Get a single vector by ID"""
        if self._prefer_local_fallback:
            for record in self._get_local_vector_records(collection_id):
                if record.id == vector_id:
                    return VectorRecord(
                        id=record.id,
                        vector=(
                            record.vector
                            if include_vector
                            else [0.0] * len(record.vector)
                        ),
                        metadata=record.metadata if include_metadata else {},
                        timestamp_ms=record.timestamp_ms,
                        updated_at_ms=record.updated_at_ms,
                        expires_at_ms=record.expires_at_ms,
                        version=record.version,
                        source=record.source,
                    )
            self._require_local_collection(collection_id)
            raise ProximaDBError(f"Vector '{vector_id}' not found")

        if not self._prefer_local_fallback:
            if self._active_protocol == Protocol.GRPC:
                try:
                    result = self._client.get_vector(
                        collection_id, vector_id, include_vector, include_metadata
                    )
                    return result
                except Exception as e:
                    self._activate_local_fallback(e)
                    logger.debug(
                        "Get vector failed, using local fallback for %s/%s: %s",
                        collection_id,
                        vector_id,
                        e,
                    )
            else:
                try:
                    result = self._client.get_vector(
                        collection_id, vector_id, include_vector, include_metadata
                    )
                    return result
                except Exception as e:
                    self._activate_local_fallback(e)
                    logger.debug(
                        "Get vector failed, using local fallback for %s/%s: %s",
                        collection_id,
                        vector_id,
                        e,
                    )

        for record in self._get_local_vector_records(collection_id):
            if record.id == vector_id:
                return VectorRecord(
                    id=record.id,
                    vector=(
                        record.vector if include_vector else [0.0] * len(record.vector)
                    ),
                    metadata=record.metadata if include_metadata else {},
                    timestamp_ms=record.timestamp_ms,
                    updated_at_ms=record.updated_at_ms,
                    expires_at_ms=record.expires_at_ms,
                    version=record.version,
                    source=record.source,
                )
        self._require_local_collection(collection_id)
        raise ProximaDBError(f"Vector '{vector_id}' not found")

    def insert_vector(
        self,
        collection_id: str,
        vector_id: str,
        vector: list[float] | np.ndarray,
        metadata: dict[str, Any] | None = None,
        timestamp_ms: int | None = None,
        updated_at_ms: int | None = None,
        expires_at_ms: int | None = None,
        version: int | None = None,
        source: str | None = None,
        upsert: bool = False,
    ) -> VectorOperationResponse:
        """Insert a single vector with full VectorRecord support

        Args:
            collection_id: Collection ID or name
            vector_id: Vector identifier
            vector: Vector data (list or numpy array of floats)
            metadata: Optional metadata key-value pairs
            timestamp_ms: Optional timestamp in milliseconds (auto-generated if not provided)
            updated_at_ms: Optional last update timestamp in milliseconds
            expires_at_ms: Optional expiration timestamp in milliseconds
            version: Optional version number (default: 0)
            source: Optional original content that generated this vector
            upsert: If True, update existing vector

        Returns:
            VectorOperationResponse
        """
        # Create VectorRecord with all supported fields
        record = VectorRecord(id=vector_id, vector=vector, metadata=metadata or {})

        # Add optional timestamp fields if provided
        if timestamp_ms is not None:
            record.timestamp_ms = timestamp_ms
        if updated_at_ms is not None:
            record.updated_at_ms = updated_at_ms
        if expires_at_ms is not None:
            record.expires_at_ms = expires_at_ms

        # Add version if provided
        if version is not None:
            record.version = version

        # Add source field (original content)
        if source is not None:
            record.source = source

        if upsert:
            return self.upsert_vectors(collection_id, [record])
        else:
            return self.insert_vectors(collection_id, [record])

    def delete_vector(
        self, collection_id: str, vector_id: str
    ) -> VectorOperationResponse:
        """Delete a single vector - alias for batch delete with one vector

        Args:
            collection_id: Collection ID or name
            vector_id: Vector identifier to delete

        Returns:
            VectorOperationResponse
        """
        return self.delete_vectors(collection_id, [vector_id])

    def execute_sql(
        self,
        query: str,
        parameters: list[Any] | None = None,
        collection: str | None = None,
    ) -> dict[str, Any]:
        """Execute SQL query with vector similarity support

        Args:
            query: SQL query string (e.g., "SELECT * FROM my_collection ORDER BY VECTOR_SIMILARITY(vector, [0.1, 0.2, ...], 'cosine') LIMIT 10")
            parameters: Optional query parameters (not yet supported)
            collection: Optional collection hint (if not specified in FROM clause)

        Returns:
            Dict with 'rows', 'columns', and 'row_count' keys

        Example:
            >>> result = client.execute_sql(
            ...     "SELECT id, metadata FROM my_collection WHERE metadata.category = 'electronics' ORDER BY VECTOR_SIMILARITY(vector, [0.1, 0.2], 'cosine') LIMIT 5"
            ... )
            >>> for row in result['rows']:
            ...     print(row['id'], row['metadata'])
        """
        if self._active_protocol == Protocol.EMBEDDED and self._client is not None:
            try:
                native_result = self._client.execute_sql(query, parameters, collection)
                if self._is_vector_search_sql(query) and not native_result.get("rows"):
                    local_result = self._local_sql_fallback_result(
                        query, parameters, collection
                    )
                    if local_result is not None:
                        return local_result
                return native_result
            except Exception as e:
                logger.debug("Embedded SQL failed, using adapter/local fallback: %s", e)
                if self._adapter and hasattr(self._adapter, "execute_sql"):
                    try:
                        return self._adapter.execute_sql(
                            query, parameters=parameters, collection=collection
                        )
                    except Exception as adapter_error:
                        logger.debug(
                            "Embedded SQL adapter fallback failed, using local fallback: %s",
                            adapter_error,
                        )
                return self._execute_sql_local(query, parameters, collection)
        if self._active_protocol == Protocol.GRPC:
            # Use gRPC SQL service
            try:
                return self._client.execute_sql(query, parameters, collection)
            except Exception as e:
                logger.debug("gRPC SQL failed, using local fallback: %s", e)
                return self._execute_sql_local(query, parameters, collection)
        else:
            try:
                return self._execute_sql_rest(query, parameters, collection)
            except Exception as e:
                logger.debug("REST SQL failed, using local fallback: %s", e)
                return self._execute_sql_local(query, parameters, collection)

    def execute_unified_query(
        self,
        query: str,
        query_vector: list[float] | None = None,
        fusion_strategy: str | None = None,
    ) -> list[dict[str, Any]]:
        """Execute a federated multi-model query.

        Embedded mode uses the native binding. REST mode uses the OpenAPI v2
        UQL query surface so existing clients do not need a separate migration.
        """
        if self._active_protocol == Protocol.EMBEDDED and self._client is not None:
            try:
                native_result = list(
                    self._client.execute_unified_query(
                        query, query_vector, fusion_strategy
                    )
                    or []
                )
                if self._is_vector_search_sql(query) and not native_result:
                    local_result = self._local_sql_fallback_result(query)
                    if local_result is not None:
                        return self._sql_rows_to_unified_records(
                            local_result.get("rows", [])
                        )
                return native_result
            except Exception as e:
                logger.debug(
                    "Embedded unified query failed, using adapter fallback: %s", e
                )
                if self._adapter and hasattr(self._adapter, "execute_unified_query"):
                    try:
                        return self._adapter.execute_unified_query(
                            query,
                            query_vector=query_vector,
                            fusion_strategy=fusion_strategy,
                        )
                    except Exception as adapter_error:
                        logger.debug(
                            "Embedded unified adapter fallback failed, using local fallback: %s",
                            adapter_error,
                        )
                return list(
                    self._sql_rows_to_unified_records(
                        self._execute_sql_local(query, collection=None).get("rows", [])
                    )
                )

        if self._adapter and hasattr(self._adapter, "execute_unified_query"):
            return self._adapter.execute_unified_query(
                query,
                query_vector=query_vector,
                fusion_strategy=fusion_strategy,
            )

        if self._adapter and hasattr(self._adapter, "execute_query"):
            result = self._adapter.execute_query(query, language="uql")
            if isinstance(result, dict):
                rows = (
                    result.get("records")
                    or result.get("rows")
                    or result.get("data")
                    or []
                )
                return rows if isinstance(rows, list) else [rows]
            return result

        raise NotImplementedError(
            "execute_unified_query requires embedded mode or a REST adapter with /api/v2/query"
        )

    def execute_query(
        self,
        query: str,
        *,
        language: str = "uql",
        parameters: list[Any] | None = None,
        collection: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Execute AQL/UQL through the OpenAPI v2 REST query surface."""
        if self._adapter and hasattr(self._adapter, "execute_query"):
            return self._adapter.execute_query(
                query,
                language=language,
                parameters=parameters,
                collection=collection,
                limit=limit,
            )
        if self._client is not None and hasattr(self._client, "execute_query"):
            return self._client.execute_query(
                query,
                language=language,
                parameters=parameters,
                collection=collection,
                limit=limit,
            )
        raise NotImplementedError(
            "execute_query requires the REST OpenAPI v2 query surface"
        )

    def execute_uql(
        self,
        query: str,
        *,
        parameters: list[Any] | None = None,
        collection: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Execute UQL through the OpenAPI v2 REST query surface."""
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
        """Execute AQL through the OpenAPI v2 REST query surface."""
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
        """Execute federated SQL extensions through the OpenAPI v2 REST surface."""
        return self.execute_query(
            query,
            language="federated",
            parameters=parameters,
            collection=collection,
            limit=limit,
        )

    def create_observability_namespace(
        self,
        name: str,
        retention_days: int | None = None,
    ) -> dict[str, Any]:
        """Create a namespace for logs, metrics, and traces."""
        if self._active_protocol == Protocol.EMBEDDED and self._client is not None:
            try:
                result = self._client.create_observability_namespace(
                    name, retention_days
                )
                return (
                    result
                    if isinstance(result, dict)
                    else {"success": True, "namespace": name}
                )
            except Exception as e:
                logger.debug("Embedded create_observability_namespace failed: %s", e)

        if self._adapter and hasattr(self._adapter, "create_observability_namespace"):
            return self._adapter.create_observability_namespace(
                name, retention_days=retention_days
            )

        raise NotImplementedError(
            "Observability namespaces are currently supported in embedded mode only"
        )

    def ingest_logs(self, namespace: str, logs: list[dict[str, Any]]) -> int:
        """Ingest structured log events into an observability namespace."""
        if self._active_protocol == Protocol.EMBEDDED and self._client is not None:
            try:
                return int(self._client.ingest_logs(namespace, logs))
            except Exception as e:
                logger.debug("Embedded log ingest failed: %s", e)

        if self._adapter and hasattr(self._adapter, "ingest_logs"):
            return int(self._adapter.ingest_logs(namespace, logs))

        raise NotImplementedError(
            "Log ingest is currently supported in embedded mode only"
        )

    def query_logs(
        self,
        namespace: str,
        start_time_ns: int,
        end_time_ns: int,
        query: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query log events from an observability namespace."""
        if self._active_protocol == Protocol.EMBEDDED and self._client is not None:
            try:
                return list(
                    self._client.query_logs(
                        namespace, start_time_ns, end_time_ns, query, limit
                    )
                    or []
                )
            except Exception as e:
                logger.debug("Embedded log query failed: %s", e)

        if self._adapter and hasattr(self._adapter, "query_logs"):
            return list(
                self._adapter.query_logs(
                    namespace,
                    start_time_ns=start_time_ns,
                    end_time_ns=end_time_ns,
                    query=query,
                    limit=limit,
                )
                or []
            )

        raise NotImplementedError(
            "Log query is currently supported in embedded mode only"
        )

    def ingest_metrics(self, namespace: str, samples: list[dict[str, Any]]) -> int:
        """Ingest metric samples into an observability namespace."""
        if self._active_protocol == Protocol.EMBEDDED and self._client is not None:
            try:
                return int(self._client.ingest_metrics(namespace, samples))
            except Exception as e:
                logger.debug("Embedded metric ingest failed: %s", e)

        if self._adapter and hasattr(self._adapter, "ingest_metrics"):
            return int(self._adapter.ingest_metrics(namespace, samples))

        raise NotImplementedError(
            "Metric ingest is currently supported in embedded mode only"
        )

    def aggregate_metrics(
        self,
        namespace: str,
        metric_name: str,
        aggregation: str = "avg",
        start_time: str | None = None,
        end_time: str | None = None,
        step_seconds: int = 60,
    ) -> list[dict[str, Any]]:
        """Aggregate metrics from an observability namespace."""
        if self._active_protocol == Protocol.EMBEDDED and self._client is not None:
            try:
                return list(
                    self._client.aggregate_metrics(
                        namespace,
                        metric_name,
                        aggregation,
                        start_time,
                        end_time,
                        step_seconds,
                    )
                    or []
                )
            except Exception as e:
                logger.debug("Embedded metric aggregation failed: %s", e)

        if self._adapter and hasattr(self._adapter, "aggregate_metrics"):
            return list(
                self._adapter.aggregate_metrics(
                    namespace,
                    metric_name=metric_name,
                    aggregation=aggregation,
                    start_time=start_time,
                    end_time=end_time,
                    step_seconds=step_seconds,
                )
                or []
            )

        raise NotImplementedError(
            "Metric aggregation is currently supported in embedded mode only"
        )

    def ingest_traces(self, namespace: str, traces: list[dict[str, Any]]) -> int:
        """Ingest trace spans into an observability namespace."""
        if self._active_protocol == Protocol.EMBEDDED and self._client is not None:
            try:
                return int(self._client.ingest_traces(namespace, traces))
            except Exception as e:
                logger.debug("Embedded trace ingest failed: %s", e)

        if self._adapter and hasattr(self._adapter, "ingest_traces"):
            return int(self._adapter.ingest_traces(namespace, traces))

        raise NotImplementedError(
            "Trace ingest is currently supported in embedded mode only"
        )

    def query_traces(
        self,
        namespace: str,
        start_time_ns: int,
        end_time_ns: int,
        trace_id: str | None = None,
        service: str | None = None,
        operation: str | None = None,
        min_duration_ns: int | None = None,
        status: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Query trace spans in an observability namespace."""
        if self._active_protocol == Protocol.EMBEDDED and self._client is not None:
            try:
                return list(
                    self._client.query_traces(
                        namespace,
                        start_time_ns,
                        end_time_ns,
                        trace_id,
                        service,
                        operation,
                        min_duration_ns,
                        status,
                        limit,
                    )
                    or []
                )
            except Exception as e:
                logger.debug("Embedded trace query failed: %s", e)

        if self._adapter and hasattr(self._adapter, "query_traces"):
            return list(
                self._adapter.query_traces(
                    namespace,
                    start_time_ns=start_time_ns,
                    end_time_ns=end_time_ns,
                    trace_id=trace_id,
                    service=service,
                    operation=operation,
                    min_duration_ns=min_duration_ns,
                    status=status,
                    limit=limit,
                )
                or []
            )

        raise NotImplementedError(
            "Trace query is currently supported in embedded mode only"
        )

    def get_trace(self, namespace: str, trace_id: str) -> dict[str, Any]:
        """Get all spans for a specific trace ID."""
        if self._active_protocol == Protocol.EMBEDDED and self._client is not None:
            try:
                result = self._client.get_trace(namespace, trace_id)
                return (
                    result
                    if isinstance(result, dict)
                    else {"spans": list(result or []), "complete": True}
                )
            except Exception as e:
                logger.debug("Embedded get_trace failed: %s", e)

        if self._adapter and hasattr(self._adapter, "get_trace"):
            return self._adapter.get_trace(namespace, trace_id=trace_id)

        raise NotImplementedError(
            "get_trace is currently supported in embedded mode only"
        )

    def _execute_sql_rest(
        self,
        query: str,
        parameters: list[Any] | None = None,
        collection: str | None = None,
    ) -> dict[str, Any]:
        """Execute SQL query via the canonical v2 query facade.

        The legacy /api/v1/sql/execute route was removed in the API
        standardization hard-rename; raw SQL is submitted via the unified
        query language ("uql") on /api/v2/query.
        """
        # Build request payload
        payload = {"language": "uql", "query": query}
        if parameters is not None:
            payload["parameters"] = parameters
        if collection is not None:
            payload["collection"] = collection

        # Make REST request
        if hasattr(self._client, "_session"):
            # Using REST client directly
            response = self._client._session.post(
                f"{self._client._base_url}/api/v2/query", json=payload
            )
            response.raise_for_status()
            return response.json()
        else:
            # Need to use requests directly
            import requests

            headers = {}
            if hasattr(self._client, "_api_key") and self._client._api_key:
                headers["X-API-Key"] = self._client._api_key

            base_url = getattr(self._client, "_rest_url", None) or getattr(
                self._client, "_base_url", "http://localhost:5678"
            )
            response = requests.post(
                f"{base_url}/api/v2/query", json=payload, headers=headers
            )
            if not response.ok:
                # Try to get error details from response
                try:
                    error_data = response.json()
                    error_msg = error_data.get("message", response.text)
                except:
                    error_msg = response.text
                raise Exception(
                    f"SQL execution failed (HTTP {response.status_code}): {error_msg}"
                )
            return response.json()

    def close(self):
        """Close the client and cleanup resources"""
        if self._closed:
            return

        if self._client and hasattr(self._client, "close"):
            try:
                self._client.close()
            except Exception:
                pass

        # Stop operation router background thread if enabled
        if self._operation_router:
            try:
                self._operation_router.stop()
            except Exception:
                pass
            self._operation_router = None

        # Stop protocol selector background thread if enabled
        if self._protocol_selector:
            try:
                self._protocol_selector.stop()
            except Exception:
                pass
            self._protocol_selector = None

        self._closed = True

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()

    def __del__(self):
        """Destructor - cleanup resources"""
        if sys.is_finalizing():
            return
        try:
            self.close()
        except Exception:
            pass

    # Legacy compatibility methods
    def insert(
        self,
        collection_id: str,
        vectors: list[list[float]] | np.ndarray,
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> VectorOperationResponse:
        """Legacy insert method for backward compatibility"""
        fast_result = self._try_embedded_numpy_vector_batch(
            collection_id,
            vectors,
            ids,
            metadata,
            upsert=False,
        )
        if fast_result is not None:
            return fast_result

        records = []

        # Convert vectors to list format
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()

        # Build records
        for i, vector in enumerate(vectors):
            record = VectorRecord(
                vector=vector,
                id=ids[i] if ids and i < len(ids) else None,
                metadata=metadata[i] if metadata and i < len(metadata) else {},
            )
            records.append(record)

        return self.insert_vectors(collection_id, records)

    def upsert(
        self,
        collection_id: str,
        vectors: list[list[float]] | np.ndarray,
        ids: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> VectorOperationResponse:
        """Legacy upsert method for backward compatibility"""
        fast_result = self._try_embedded_numpy_vector_batch(
            collection_id,
            vectors,
            ids,
            metadata,
            upsert=True,
        )
        if fast_result is not None:
            return fast_result

        records = []

        # Convert vectors to list format
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()

        # Build records
        for i, (vector, vector_id) in enumerate(zip(vectors, ids)):
            record = VectorRecord(
                vector=vector,
                id=vector_id,
                metadata=metadata[i] if metadata and i < len(metadata) else {},
            )
            records.append(record)

        return self.upsert_vectors(collection_id, records)

    def delete(self, collection_id: str, ids: list[str]) -> VectorOperationResponse:
        """Legacy delete method for backward compatibility"""
        return self.delete_vectors(collection_id, ids)

    def _invoke_graph_method(
        self,
        method_name: str,
        graph_id: str | None = None,
        **kwargs,
    ) -> Any:
        method = getattr(self._client, method_name)
        if graph_id is None:
            return method(**kwargs)

        try:
            return method(graph_id=graph_id, **kwargs)
        except TypeError:
            return method(graph=graph_id, **kwargs)

    # ===========================
    # Graph API Methods
    # ===========================

    def batch_create_nodes(
        self,
        nodes: list[dict[str, Any]],
        graph_id: str | None = None,
    ) -> dict[str, Any]:
        """Create many nodes in one request (falls to the protocol's batch
        endpoint; raises AttributeError on transports without one — callers
        like ``ProximaDBGraph`` fall back to per-item creates)."""
        return self._invoke_graph_method(
            "batch_create_nodes", graph_id=graph_id, nodes=nodes
        )

    def batch_create_edges(
        self,
        edges: list[dict[str, Any]],
        graph_id: str | None = None,
    ) -> dict[str, Any]:
        """Create many edges in one request with per-edge admission — the
        server reports rejections in ``errors[]`` while the valid rest lands."""
        return self._invoke_graph_method(
            "batch_create_edges", graph_id=graph_id, edges=edges
        )

    def create_node(
        self,
        node_id: str,
        labels: list[str],
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        graph_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a graph node.

        Args:
            node_id: Unique identifier for the node
            labels: List of labels for the node (e.g., ["Person", "Employee"])
            properties: Dictionary of node properties
            embedding: Optional vector embedding for the node

        Returns:
            Dictionary with created node information

        Example:
            >>> client.create_node(
            ...     node_id="person_123",
            ...     labels=["Person"],
            ...     properties={"name": "Alice", "age": 30}
            ... )
        """
        # Input validation
        if not isinstance(node_id, str):
            raise TypeError(f"node_id must be str, got {type(node_id).__name__}")
        if not isinstance(labels, list):
            raise TypeError(f"labels must be list, got {type(labels).__name__}")
        if properties is not None and not isinstance(properties, dict):
            raise TypeError(
                f"properties must be dict or None, got {type(properties).__name__}"
            )
        if embedding is not None and not isinstance(embedding, (list, type(None))):
            raise TypeError(
                f"embedding must be list or None, got {type(embedding).__name__}"
            )

        kwargs = {
            "node_id": node_id,
            "labels": labels,
            "properties": properties,
            "embedding": embedding,
        }
        result = self._invoke_graph_method("create_node", graph_id=graph_id, **kwargs)
        if self._active_protocol == Protocol.EMBEDDED and not isinstance(result, dict):
            return {"success": True, "node_id": node_id, "result": result}
        return result

    def create_edge(
        self,
        edge_id: str,
        from_node_id: str,
        to_node_id: str,
        edge_type: str,
        properties: dict[str, Any] | None = None,
        weight: float | None = None,
        graph_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a graph edge between two nodes.

        Args:
            edge_id: Unique identifier for the edge
            from_node_id: ID of the source node
            to_node_id: ID of the target node
            edge_type: Type of relationship (e.g., "KNOWS", "WORKS_WITH")
            properties: Dictionary of edge properties
            weight: Optional numeric weight for the edge

        Returns:
            Dictionary with created edge information

        Example:
            >>> client.create_edge(
            ...     edge_id="edge_123",
            ...     from_node_id="person_123",
            ...     to_node_id="person_456",
            ...     edge_type="KNOWS",
            ...     properties={"since": 2020},
            ...     weight=1.0
            ... )
        """
        # Input validation
        if not isinstance(edge_id, str):
            raise TypeError(f"edge_id must be str, got {type(edge_id).__name__}")
        if not isinstance(from_node_id, str):
            raise TypeError(
                f"from_node_id must be str, got {type(from_node_id).__name__}"
            )
        if not isinstance(to_node_id, str):
            raise TypeError(f"to_node_id must be str, got {type(to_node_id).__name__}")
        if not isinstance(edge_type, str):
            raise TypeError(f"edge_type must be str, got {type(edge_type).__name__}")
        if properties is not None and not isinstance(properties, dict):
            raise TypeError(
                f"properties must be dict or None, got {type(properties).__name__}"
            )
        if weight is not None and not isinstance(weight, (int, float)):
            raise TypeError(
                f"weight must be number or None, got {type(weight).__name__}"
            )

        kwargs = {
            "edge_id": edge_id,
            "from_node_id": from_node_id,
            "to_node_id": to_node_id,
            "edge_type": edge_type,
            "properties": properties,
            "weight": weight,
        }
        result = self._invoke_graph_method("create_edge", graph_id=graph_id, **kwargs)
        if self._active_protocol == Protocol.EMBEDDED and not isinstance(result, dict):
            return {"success": True, "edge_id": edge_id, "result": result}
        return result

    def traverse_graph(
        self,
        start_node_id: str,
        max_depth: int = 3,
        edge_types: list[str] | None = None,
        node_labels: list[str] | None = None,
        algorithm: str = "BFS",
        limit: int | None = None,
        graph_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Traverse the graph starting from a node.

        Args:
            start_node_id: ID of the starting node
            max_depth: Maximum traversal depth (default: 3)
            edge_types: Filter by specific edge types (default: all types)
            node_labels: Filter by specific node labels (default: all labels)
            algorithm: Traversal algorithm - "BFS", "DFS", or "PARALLEL_BFS" (default: "BFS")
            limit: Maximum number of nodes to return (default: unlimited)

        Returns:
            Dictionary with:
                - nodes: List of visited nodes
                - edges: List of traversed edges
                - paths: List of paths found
                - stats: Traversal statistics

        Example:
            >>> result = client.traverse_graph(
            ...     start_node_id="person_123",
            ...     max_depth=2,
            ...     edge_types=["KNOWS"],
            ...     algorithm="BFS",
            ...     limit=50
            ... )
            >>> print(f"Found {len(result['nodes'])} nodes")
        """
        # Input validation
        if not isinstance(start_node_id, str):
            raise TypeError(
                f"start_node_id must be str, got {type(start_node_id).__name__}"
            )
        if not isinstance(max_depth, int):
            raise TypeError(f"max_depth must be int, got {type(max_depth).__name__}")
        if max_depth < 1:
            raise ValueError(f"max_depth must be >= 1, got {max_depth}")
        if edge_types is not None and not isinstance(edge_types, list):
            raise TypeError(
                f"edge_types must be list or None, got {type(edge_types).__name__}"
            )
        if node_labels is not None and not isinstance(node_labels, list):
            raise TypeError(
                f"node_labels must be list or None, got {type(node_labels).__name__}"
            )
        if not isinstance(algorithm, str):
            raise TypeError(f"algorithm must be str, got {type(algorithm).__name__}")
        if algorithm not in ["BFS", "DFS", "PARALLEL_BFS"]:
            raise ValueError(
                f"algorithm must be one of BFS/DFS/PARALLEL_BFS, got {algorithm}"
            )
        if limit is not None and not isinstance(limit, int):
            raise TypeError(f"limit must be int or None, got {type(limit).__name__}")

        kwargs = {
            "start_node_id": start_node_id,
            "max_depth": max_depth,
            "edge_types": edge_types,
            "node_labels": node_labels,
            "algorithm": algorithm,
            "limit": limit,
        }
        result = self._invoke_graph_method(
            "traverse_graph", graph_id=graph_id, **kwargs
        )
        if self._active_protocol == Protocol.EMBEDDED and not isinstance(result, dict):
            return {"nodes": list(result or []), "edges": [], "paths": [], "stats": {}}
        return result

    def query_nodes(
        self,
        labels: list[str] | None = None,
        properties: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        graph_id: str | None = None,
    ) -> dict[str, Any]:
        """
        Query graph nodes by labels and properties.

        Args:
            labels: Filter by node labels (default: all labels)
            properties: Filter by properties (exact match, default: no filter)
            limit: Maximum number of nodes to return (default: unlimited)
            offset: Number of nodes to skip for pagination (default: 0)

        Returns:
            Dictionary with:
                - nodes: List of matching nodes
                - total_count: Total number of matching nodes (if available)
                - has_more: Whether more results are available

        Example:
            >>> # Query all Person nodes
            >>> result = client.query_nodes(labels=["Person"], limit=10)
            >>>
            >>> # Query with property filter
            >>> result = client.query_nodes(
            ...     labels=["Person"],
            ...     properties={"age": 30},
            ...     limit=20,
            ...     offset=0
            ... )
        """
        # Input validation
        if labels is not None and not isinstance(labels, list):
            raise TypeError(f"labels must be list or None, got {type(labels).__name__}")
        if properties is not None and not isinstance(properties, dict):
            raise TypeError(
                f"properties must be dict or None, got {type(properties).__name__}"
            )
        if limit is not None and not isinstance(limit, int):
            raise TypeError(f"limit must be int or None, got {type(limit).__name__}")
        if offset is not None and not isinstance(offset, int):
            raise TypeError(f"offset must be int or None, got {type(offset).__name__}")

        kwargs = {
            "labels": labels,
            "properties": properties,
            "limit": limit,
            "offset": offset,
        }
        result = self._invoke_graph_method("query_nodes", graph_id=graph_id, **kwargs)
        if self._active_protocol == Protocol.EMBEDDED and not isinstance(result, dict):
            nodes = list(result or [])
            return {"nodes": nodes, "total_count": len(nodes), "has_more": False}
        return result

    def get_node(
        self,
        node_id: str,
        graph_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Get a graph node by ID."""
        result = self._invoke_graph_method(
            "get_node", graph_id=graph_id, node_id=node_id
        )
        if result is None:
            return None
        if self._active_protocol == Protocol.EMBEDDED and not isinstance(result, dict):
            return {
                "id": getattr(result, "id", node_id),
                "labels": list(getattr(result, "labels", []) or []),
                "properties": dict(getattr(result, "properties", {}) or {}),
            }
        return result

    def get_outgoing_edges(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        graph_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get outgoing edges for a graph node."""
        result = self._invoke_graph_method(
            "get_outgoing_edges",
            graph_id=graph_id,
            node_id=node_id,
            edge_types=edge_types,
        )
        return list(result or [])

    def get_incoming_edges(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        graph_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get incoming edges for a graph node."""
        result = self._invoke_graph_method(
            "get_incoming_edges",
            graph_id=graph_id,
            node_id=node_id,
            edge_types=edge_types,
        )
        return list(result or [])

    def delete_node(
        self,
        node_id: str,
        graph_id: str | None = None,
    ) -> bool:
        """Delete a graph node by ID."""
        return bool(
            self._invoke_graph_method("delete_node", graph_id=graph_id, node_id=node_id)
        )

    # ==================== Graph Collection Management ====================

    def create_graph(
        self,
        graph_id: str,
        name: str | None = None,
        description: str | None = None,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Create a new graph collection.

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
        try:
            return self._client.create_graph(
                graph_id=graph_id, name=name, description=description, schema=schema
            )
        except TypeError:
            result = self._client.create_graph(graph_id, None)
            return (
                result
                if isinstance(result, dict)
                else {"success": True, "graph_id": graph_id}
            )

    def delete_graph(self, graph_id: str) -> dict[str, Any]:
        """
        Delete a graph collection.

        Args:
            graph_id: ID of the graph collection to delete

        Returns:
            Dictionary confirming deletion

        Example:
            >>> result = client.delete_graph("social_network")
        """
        return self._client.delete_graph(graph_id)

    def get_graph(self, graph_id: str) -> dict[str, Any]:
        """
        Get graph collection metadata.

        Args:
            graph_id: ID of the graph collection

        Returns:
            Dictionary containing graph collection metadata

        Example:
            >>> graph = client.get_graph("social_network")
            >>> print(graph["name"])
        """
        return self._client.get_graph(graph_id)

    def list_graphs(self) -> dict[str, Any]:
        """
        List all graph collections.

        Returns:
            Dictionary containing list of all graph collections

        Example:
            >>> graphs = client.list_graphs()
            >>> for graph in graphs.get("data", []):
            ...     print(graph["graph_id"])
        """
        return self._client.list_graphs()

    def get_graph_stats(self, graph_id: str) -> dict[str, Any]:
        """
        Get statistics for a graph collection.

        Args:
            graph_id: ID of the graph collection

        Returns:
            Dictionary containing graph statistics (node count, edge count, etc.)

        Example:
            >>> stats = client.get_graph_stats("social_network")
            >>> print(f"Nodes: {stats['node_count']}, Edges: {stats['edge_count']}")
        """
        return self._invoke_graph_method("get_graph_stats", graph_id=graph_id)

    # ==================== End Graph Collection Management ====================

    def get_collection_stats(self, collection_id: str) -> dict[str, Any]:
        """Get collection statistics (legacy compatibility)"""
        collection = self.get_collection(collection_id)
        if collection:
            return {
                "id": collection.id,
                "name": collection.config.name,
                "dimension": collection.config.dimension,
                "created_at": collection.created_at,
                "updated_at": collection.updated_at,
                "vector_count": getattr(collection, "vector_count", 0),
                "index_count": getattr(collection, "index_count", 0),
                "status": getattr(collection, "status", "active"),
            }
        return {}


# Convenience functions for backward compatibility
def connect(
    url: str | None = None,
    api_key: str | None = None,
    protocol: Protocol | str = Protocol.AUTO,
    **kwargs,
) -> ProximaDBClient:
    """Create a ProximaDB client with simplified parameters"""
    return ProximaDBClient(url=url, api_key=api_key, protocol=protocol, **kwargs)


def connect_grpc(
    url: str | None = None, api_key: str | None = None, **kwargs
) -> ProximaDBClient:
    """Create a ProximaDB client using gRPC protocol (good performance, ecosystem compatibility)"""
    try:
        return ProximaDBClient(
            url=url, api_key=api_key, protocol=Protocol.GRPC, **kwargs
        )
    except ProximaDBError as e:
        # If gRPC fails due to import issues, fall back to AUTO (which will use REST)
        if "import" in str(e).lower() or "pb2" in str(e).lower():
            logger.warning(
                f"gRPC client failed due to import issues, falling back to AUTO mode: {e}"
            )
            return ProximaDBClient(
                url=url, api_key=api_key, protocol=Protocol.AUTO, **kwargs
            )
        else:
            raise


def connect_rest(
    url: str | None = None, api_key: str | None = None, **kwargs
) -> ProximaDBClient:
    """Create a ProximaDB client using REST protocol (web compatibility)"""
    return ProximaDBClient(url=url, api_key=api_key, protocol=Protocol.REST, **kwargs)


def connect_unified(
    url: str | None = None,
    api_key: str | None = None,
    protocol: Protocol | str = Protocol.AUTO,
    **kwargs,
) -> ProximaDBClient:
    """Create a ProximaDB client for unified port mode (single port for all protocols).

    This is the recommended connection method for ProximaDB servers running in
    unified port mode (default since v0.2.0). A single URL is used for all
    protocols, and the server automatically detects and routes requests.

    Args:
        url: ProximaDB server URL (e.g., "http://localhost:5678")
        api_key: Optional API key for authentication
        protocol: Protocol to use - "auto" (default), "grpc", or "rest".
                  With "auto", the client will use gRPC for performance
                  with automatic fallback to REST if needed.
        **kwargs: Additional client configuration parameters

    Returns:
        ProximaDBClient configured for unified port mode

    Example:
        # Simple unified connection (recommended)
        client = connect_unified("http://localhost:5678")

        # With explicit protocol selection
        client = connect_unified(
            url="http://localhost:5678",
            protocol="grpc"  # Force gRPC even on unified port
        )
    """
    return ProximaDBClient(
        url=url,
        api_key=api_key,
        protocol=protocol,
        port_mode=PortMode.UNIFIED,
        **kwargs,
    )


def connect_legacy(
    url: str | None = None,
    api_key: str | None = None,
    protocol: Protocol | str = Protocol.AUTO,
    **kwargs,
) -> ProximaDBClient:
    """Create a ProximaDB client for legacy multi-port mode.

    Use this for older ProximaDB deployments that use separate ports for
    REST (5678) and gRPC (5679). For new deployments, use connect_unified().

    Args:
        url: ProximaDB REST server URL (e.g., "http://localhost:5678")
        api_key: Optional API key for authentication
        protocol: Protocol to use - "auto" (default), "grpc", or "rest"
        **kwargs: Additional client configuration parameters

    Returns:
        ProximaDBClient configured for multi-port mode
    """
    return ProximaDBClient(
        url=url, api_key=api_key, protocol=protocol, port_mode=PortMode.MULTI, **kwargs
    )


def connect_arrow_flight(
    url: str | None = None,
    api_key: str | None = None,
    port_mode: PortMode | str = PortMode.UNIFIED,
    **kwargs,
) -> ProximaDBClient:
    """Create a ProximaDB client using Arrow Flight protocol for bulk data transfer.

    Arrow Flight is optimized for high-throughput bulk operations like:
    - Large batch vector inserts (millions of vectors)
    - Bulk data export/import
    - Streaming large result sets

    In unified mode (default), Arrow Flight uses the same port as REST/gRPC.
    In multi-port mode, Arrow Flight uses port 5680.

    Args:
        url: ProximaDB server URL (e.g., "http://localhost:5678")
        api_key: Optional API key for authentication
        port_mode: Server port mode - "unified" (default) or "multi"
        **kwargs: Additional client configuration parameters

    Returns:
        ProximaDBClient configured for Arrow Flight protocol

    Example:
        # Unified mode (recommended)
        client = connect_arrow_flight("http://localhost:5678")

        # Multi-port mode (legacy)
        client = connect_arrow_flight(
            url="http://localhost:5680",  # Arrow Flight port
            port_mode="multi"
        )

    Note:
        Arrow Flight requires pyarrow to be installed:
        pip install pyarrow
    """
    if isinstance(port_mode, str):
        port_mode = PortMode(port_mode.lower())

    return ProximaDBClient(
        url=url,
        api_key=api_key,
        protocol=Protocol.ARROW_FLIGHT,
        port_mode=port_mode,
        **kwargs,
    )
