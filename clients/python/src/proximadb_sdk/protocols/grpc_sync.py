"""
ProximaDB Synchronous gRPC Client Wrapper

Provides a synchronous interface with connection pooling for optimal performance.
Features:
- Load-balanced gRPC connection pool (15-25% throughput improvement)
- Automatic channel health monitoring
- Thread-safe concurrent operations
"""

import logging
from dataclasses import dataclass
from datetime import UTC
from typing import Any

from ..exceptions import ProximaDBError
from ..models import (
    BatchResult,
    SearchResult,
    VectorOperationResponse,
)
from ..models_v2 import ProximaRecord
from . import _grpc_v2_codec as _codec
from .connection_pools import GrpcChannelContext, GrpcConnectionPool


@dataclass
class HealthCheckResponse:
    """Health check response with server status"""

    healthy: bool
    latency_ms: float
    status: str
    server_address: str
    details: str | None = None
    version: str | None = None


@dataclass
class DeleteCollectionResponse:
    """Delete collection response"""

    success: bool
    collection_id: str
    status: str = "deleted"


class CollectionWrapper:
    """
    Wrapper for protobuf Collection objects to provide convenient attribute access.

    Protobuf Collection objects have the structure:
    - collection.id (collection ID)
    - collection.config.name (collection name)
    - collection.config.dimension (vector dimension)

    This wrapper provides backward-compatible access via:
    - .name (maps to .config.name)
    - .dimension (maps to .config.dimension)
    - .id (maps to .id)
    - All other attributes are passed through to the underlying protobuf object
    """

    def __init__(self, proto_collection):
        """Initialize wrapper with a protobuf Collection object"""
        self._proto = proto_collection

    @property
    def name(self):
        """Get collection name from config.name"""
        if hasattr(self._proto, "config") and hasattr(self._proto.config, "name"):
            return self._proto.config.name
        return None

    @property
    def dimension(self):
        """Get collection dimension from config.dimension"""
        if hasattr(self._proto, "config") and hasattr(self._proto.config, "dimension"):
            return self._proto.config.dimension
        return None

    @property
    def id(self):
        """Get collection ID"""
        return getattr(self._proto, "id", None)

    @property
    def config(self):
        """Get collection config"""
        return getattr(self._proto, "config", None)

    @property
    def stats(self):
        """Get collection stats if available"""
        return getattr(self._proto, "stats", None)

    def __getattr__(self, name):
        """Pass through any other attribute access to the underlying protobuf object"""
        return getattr(self._proto, name)

    def __repr__(self):
        return f"CollectionWrapper(name={self.name}, id={self.id}, dimension={self.dimension})"


class SearchResultsWrapper:
    """
    Wrapper for search results to provide backward-compatible .results attribute.

    Wraps a list of SearchResult objects to provide:
    - .results attribute (the list itself)
    - Direct list access via indexing, iteration, len()
    """

    def __init__(self, results_list: list[Any]):
        """Initialize wrapper with a list of SearchResult objects"""
        self.results = results_list

    def __len__(self):
        """Support len() operation"""
        return len(self.results)

    def __iter__(self):
        """Support iteration"""
        return iter(self.results)

    def __getitem__(self, index):
        """Support indexing"""
        return self.results[index]

    def __repr__(self):
        return f"SearchResultsWrapper(count={len(self.results)})"


class VectorWrapper:
    """
    Wrapper for vector dict to provide attribute access.

    Converts a dict like {'id': ..., 'vector': ..., 'metadata': ...}
    to an object with attribute access: obj.id, obj.vector, obj.metadata
    """

    def __init__(self, vector_dict: dict[str, Any]):
        """Initialize wrapper with a vector dictionary"""
        self._dict = vector_dict
        # Set attributes from dict
        for key, value in vector_dict.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        """Support dict-like access"""
        return self._dict[key]

    def get(self, key, default=None):
        """Support dict.get() method"""
        return self._dict.get(key, default)

    def __repr__(self):
        return f"VectorWrapper(id={getattr(self, 'id', None)})"


class DictWrapper:
    """
    Generic wrapper to convert any dict to an object with attribute access.

    Used for operation results like delete, update, etc.
    """

    def __init__(self, data_dict: dict[str, Any]):
        """Initialize wrapper with a dictionary"""
        self._dict = data_dict
        # Set attributes from dict
        for key, value in data_dict.items():
            setattr(self, key, value)

    def __getitem__(self, key):
        """Support dict-like access"""
        return self._dict[key]

    def get(self, key, default=None):
        """Support dict.get() method"""
        return self._dict.get(key, default)

    def __repr__(self):
        return f"DictWrapper({self._dict})"


class _StubWithDefaultRpcKwargs:
    """Inject default gRPC call kwargs such as auth metadata into generated stubs."""

    def __init__(self, stub: Any, default_kwargs: dict[str, Any]):
        self._stub = stub
        self._default_kwargs = default_kwargs

    def __getattr__(self, name: str):
        attr = getattr(self._stub, name)
        if not callable(attr):
            return attr

        def _call(*args, **kwargs):
            for key, value in self._default_kwargs.items():
                kwargs.setdefault(key, value)
            return attr(*args, **kwargs)

        return _call


try:
    import grpc

    from proximadb_sdk.v1 import (
        collection_pb2_grpc as v1_collection_pb2_grpc,  # type: ignore
    )
    from proximadb_sdk.v1 import (
        collection_types_pb2 as v1_collection_types_pb2,  # type: ignore
    )
    from proximadb_sdk.v1 import types_pb2 as v1_types_pb2  # type: ignore
    from proximadb_sdk.v1 import vector_pb2_grpc as v1_vector_pb2_grpc  # type: ignore
    from proximadb_sdk.v1 import vector_types_pb2 as v1_vector_types_pb2  # type: ignore

    # Canonical v2 graph service (generated via Makefile: gen-proto).
    # Replaces the deprecated proximadb.v1.GraphService.
    try:
        from proximadb.v2 import graph_pb2 as v2_graph_pb2  # type: ignore
        from proximadb.v2 import graph_pb2_grpc as v2_graph_pb2_grpc  # type: ignore
    except Exception:  # pragma: no cover - optional
        v2_graph_pb2_grpc = None
        v2_graph_pb2 = None
    try:
        from proximadb.v2 import record_pb2 as v2_record_pb2  # type: ignore
        from proximadb.v2 import record_pb2_grpc as v2_record_pb2_grpc  # type: ignore
    except Exception:  # pragma: no cover - broken generated-stub install
        v2_record_pb2 = None
        v2_record_pb2_grpc = None
    GRPC_AVAILABLE = True
except ImportError:
    GRPC_AVAILABLE = False
    v2_record_pb2 = None
    v2_record_pb2_grpc = None

logger = logging.getLogger(__name__)


def _endpoint_is_far(server_address: str) -> bool:
    """KOU locality heuristic: is this endpoint remote (-> compress) or local (-> skip)?

    LOCAL/free (no gzip): loopback, RFC1918 private, link-local, ``localhost``,
    ``*.local``. FAR/chargeable (gzip by default): any public host/IP -- the
    common internet / cross-region / cross-cloud case. Mirrors the gateway's
    far-client gzip (anvaiops #168) and the KOU egress model (proximaDB #110).
    """
    import ipaddress

    host = server_address.split("://", 1)[-1].rsplit(":", 1)[0].strip("[]")
    if not host or host == "localhost" or host.endswith(".local"):
        return False
    try:
        ip = ipaddress.ip_address(host)
        return not (
            ip.is_loopback or ip.is_private or ip.is_link_local or ip.is_unspecified
        )
    except ValueError:
        # A non-localhost DNS hostname -> assume remote (far).
        return True


def _v2_algo_name(a: Any) -> str:
    """Normalize an index algorithm (str or IndexingAlgorithm int) to the
    v2-native lowercase string the server expects."""
    if a is None:
        return ""
    if isinstance(a, str):
        return a.lower()
    return {1: "hnsw", 2: "ivf", 3: "pq", 4: "flat", 5: "annoy", 6: "lsh"}.get(
        int(a), ""
    )


def _build_v2_index_specs(index_configs: Any, indexing_algorithm: Any) -> list:
    """Build a list of v2 V2IndexSpec from loose index_configs (dicts or objects
    with ``__dict__``) and/or a default indexing_algorithm. Empty -> server
    auto-selects a sensible index."""
    specs = []
    for ic in index_configs or []:
        d = ic if isinstance(ic, dict) else getattr(ic, "__dict__", {})
        kwargs: dict[str, Any] = {
            "algorithm": _v2_algo_name(d.get("algorithm", indexing_algorithm))
        }
        hnsw = d.get("hnsw") or d.get("hnsw_config")
        if isinstance(hnsw, dict):
            kwargs["hnsw"] = v2_record_pb2.V2HnswConfig(
                **{
                    k: int(hnsw[k])
                    for k in ("m", "ef_construction", "ef_search")
                    if hnsw.get(k) is not None
                }
            )
        ivf = d.get("ivf") or d.get("ivf_config")
        if isinstance(ivf, dict):
            kwargs["ivf"] = v2_record_pb2.V2IvfConfig(
                **{
                    k: int(ivf[k])
                    for k in ("n_lists", "n_probe")
                    if ivf.get(k) is not None
                }
            )
        if d.get("is_primary"):
            kwargs["is_primary"] = True
        specs.append(v2_record_pb2.V2IndexSpec(**kwargs))
    if not specs and indexing_algorithm is not None:
        specs.append(
            v2_record_pb2.V2IndexSpec(algorithm=_v2_algo_name(indexing_algorithm))
        )
    return specs


def _build_v2_quantization(quantization_config: Any):
    """Build a v2 V2QuantizationConfig from a loose dict/object, or None."""
    if quantization_config is None:
        return None
    qc = (
        quantization_config
        if isinstance(quantization_config, dict)
        else getattr(quantization_config, "__dict__", {})
    )
    return v2_record_pb2.V2QuantizationConfig(
        enabled=bool(qc.get("enabled", True)),
        strategy=str(qc.get("strategy", "") or "").lower(),
    )


class ProximaDBSyncGrpcClient:
    """
    High-performance synchronous gRPC client with connection pooling

    Features:
    - Connection pool with 5 channels for load balancing
    - Automatic health monitoring and failover
    - 15-25% throughput improvement over single-channel approach
    """

    def __init__(
        self,
        server_address: str,
        timeout: float = 60.0,
        enable_compression: (
            bool | None
        ) = None,  # None = auto: gzip FAR clients, skip LOCAL
        compression_algorithm: str = "gzip",
        pool_size: int = 5,
        max_message_size: int = 64 * 1024 * 1024,
        metadata: list[tuple[str, str]] | None = None,
        use_tls: bool = False,
    ):
        """Initialize sync gRPC client with connection pool

        Args:
            server_address: gRPC server address. Use "localhost:5678" for unified port mode
                           (recommended) or "localhost:5679" for legacy multi-port mode.
            timeout: Request timeout in seconds
            enable_compression: gzip compression. None (default) auto-enables for far/remote
                           endpoints and skips local ones (the server supports gzip). True/False forces.
            compression_algorithm: Compression algorithm ('gzip', default: 'gzip')
            pool_size: Number of gRPC channels in pool (default: 5)
            max_message_size: Maximum message size in bytes (default: 64MB)
        """
        self.server_address = server_address
        self.timeout = timeout
        # KOU egress decision: compress by default for FAR clients (internet /
        # cross-region / cross-cloud); skip LOCAL/embedded to save CPU. None
        # auto-detects from the endpoint; True/False overrides.
        if enable_compression is None:
            enable_compression = _endpoint_is_far(server_address)
        self.enable_compression = enable_compression
        self.compression_algorithm = compression_algorithm.lower()
        self.pool_size = pool_size
        self.max_message_size = max_message_size
        self.metadata = metadata or []
        self.use_tls = use_tls

        # Initialize connection pool instead of single client
        self._connection_pool = None
        self._init_connection_pool()

        # Alias for backward compatibility with tests
        self._pool = self._connection_pool

    def _python_to_sql_value(self, value: Any):
        """Encode Python values into v1 SqlValue without losing nested shape."""
        from google.protobuf.struct_pb2 import NullValue

        sv = v1_types_pb2.SqlValue()
        if value is None:
            sv.null_value = NullValue.NULL_VALUE
        elif isinstance(value, bool):
            sv.bool_value = value
        elif isinstance(value, int) and not isinstance(value, bool):
            sv.int64_value = value
        elif isinstance(value, float):
            sv.number_value = value
        elif isinstance(value, (bytes, bytearray, memoryview)):
            sv.bytes_value = bytes(value)
        elif isinstance(value, (list, tuple)):
            sv.array_value.values.extend(
                self._python_to_sql_value(item) for item in value
            )
        elif isinstance(value, dict):
            for key, item in value.items():
                sv.object_value.fields[str(key)].CopyFrom(
                    self._python_to_sql_value(item)
                )
        else:
            sv.string_value = str(value)
        return sv

    def _sql_value_to_python(self, value) -> Any:
        """Decode v1 SqlValue rows into native Python values recursively."""
        kind = value.WhichOneof("value")
        if kind == "string_value":
            return value.string_value
        if kind == "number_value":
            return value.number_value
        if kind == "bool_value":
            return value.bool_value
        if kind == "int64_value":
            return value.int64_value
        if kind == "bytes_value":
            return bytes(value.bytes_value)
        if kind == "jsonb_value":
            # TD-PROTO-2: tag-9 JSONB bytes — surface the raw bytes like
            # bytes_value (mirrors client_v1._convert_from_sql_value); the
            # two SDK decoders must agree.
            return bytes(value.jsonb_value)
        if kind == "array_value":
            return [
                self._sql_value_to_python(item) for item in value.array_value.values
            ]
        if kind == "object_value":
            return {
                key: self._sql_value_to_python(item)
                for key, item in value.object_value.fields.items()
            }
        return None

    def _init_connection_pool(self):
        """Initialize gRPC connection pool for optimal performance"""
        try:
            import grpc

            # Map compression algorithm
            compression = None
            if self.enable_compression:
                if self.compression_algorithm == "gzip":
                    compression = grpc.Compression.Gzip
                elif self.compression_algorithm == "deflate":
                    compression = grpc.Compression.Deflate
                else:
                    logger.warning(
                        f"Unknown compression algorithm: {self.compression_algorithm}, using gzip"
                    )
                    compression = grpc.Compression.Gzip

            self._connection_pool = GrpcConnectionPool(
                endpoint=self.server_address,
                pool_size=self.pool_size,
                max_message_size=self.max_message_size,
                use_tls=self.use_tls,
                compression=compression,
            )

            # Update alias for backward compatibility
            self._pool = self._connection_pool

            logger.info(
                f"Initialized gRPC connection pool: {self.pool_size} channels to {self.server_address}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize gRPC connection pool: {e}")
            raise ProximaDBError(f"gRPC connection pool initialization failed: {e}")

    def _rpc_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"timeout": self.timeout}
        if self.metadata:
            kwargs["metadata"] = self.metadata
        return kwargs

    def _execute_with_pool(self, operation_name: str, operation_func):
        """Execute operation using connection pool with automatic error handling"""
        if not GRPC_AVAILABLE:
            raise ProximaDBError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )

        try:
            with GrpcChannelContext(self._connection_pool) as channel:
                # Create stub for this operation
                # Use versioned VectorService exclusively (v1)
                stub = _StubWithDefaultRpcKwargs(
                    v1_vector_pb2_grpc.VectorServiceStub(channel), self._rpc_kwargs()
                )

                # Execute the operation with timeout
                return operation_func(stub)

        except grpc.RpcError as e:
            logger.error(f"gRPC {operation_name} RPC error: {e.code()} - {e.details()}")
            details = e.details() or str(e)
            if e.code() == grpc.StatusCode.UNAVAILABLE or "connect" in details.lower():
                raise ProximaDBError(f"{operation_name} connection failed: {details}")
            raise ProximaDBError(f"{operation_name} RPC failed: {details}")
        except Exception as e:
            logger.error(f"gRPC {operation_name} failed: {e}")
            raise ProximaDBError(f"{operation_name} failed: {e}")

    def _execute_collection_with_pool(self, operation_name: str, operation_func):
        """Execute collection operation using connection pool with CollectionService"""
        if not GRPC_AVAILABLE:
            raise ProximaDBError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )

        try:
            with GrpcChannelContext(self._connection_pool) as channel:
                # v2: collection ops are served by ProximaRecordService (v1 gRPC
                # CollectionService is flag-gated off). RPC names match v1.
                stub = _StubWithDefaultRpcKwargs(
                    v2_record_pb2_grpc.ProximaRecordServiceStub(channel),
                    self._rpc_kwargs(),
                )

                # Execute the operation with timeout
                return operation_func(stub)

        except grpc.RpcError as e:
            logger.error(f"gRPC {operation_name} RPC error: {e.code()} - {e.details()}")
            details = e.details() or str(e)
            if e.code() == grpc.StatusCode.UNAVAILABLE or "connect" in details.lower():
                raise ProximaDBError(f"{operation_name} connection failed: {details}")
            raise ProximaDBError(f"{operation_name} RPC failed: {details}")
        except Exception as e:
            logger.error(f"gRPC {operation_name} failed: {e}")
            raise ProximaDBError(f"{operation_name} failed: {e}")

    def _execute_record_with_pool(self, operation_name: str, operation_func):
        """Execute record operation using the v2 ProximaRecordService."""
        if not GRPC_AVAILABLE or v2_record_pb2_grpc is None:
            raise ProximaDBError(
                "v2 record gRPC stubs not available. Regenerate Python protobuf stubs."
            )

        try:
            with GrpcChannelContext(self._connection_pool) as channel:
                stub = _StubWithDefaultRpcKwargs(
                    v2_record_pb2_grpc.ProximaRecordServiceStub(channel),
                    self._rpc_kwargs(),
                )
                return operation_func(stub)

        except grpc.RpcError as e:
            logger.error(f"gRPC {operation_name} RPC error: {e.code()} - {e.details()}")
            details = e.details() or str(e)
            if e.code() == grpc.StatusCode.UNAVAILABLE or "connect" in details.lower():
                raise ProximaDBError(f"{operation_name} connection failed: {details}")
            raise ProximaDBError(f"{operation_name} RPC failed: {details}")
        except Exception as e:
            logger.error(f"gRPC {operation_name} failed: {e}")
            raise ProximaDBError(f"{operation_name} failed: {e}")

    def _execute_fusion_with_pool(self, operation_name: str, operation_func):
        """Execute fusion operation using the v2 ProximaFusionService."""
        try:
            from proximadb.v2 import (
                fusion_pb2_grpc as v2_fusion_pb2_grpc,  # type: ignore
            )
        except ImportError:
            raise ProximaDBError(
                "v2 fusion gRPC stubs not available. Regenerate Python protobuf stubs."
            )

        if not GRPC_AVAILABLE:
            raise ProximaDBError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )

        try:
            with GrpcChannelContext(self._connection_pool) as channel:
                stub = _StubWithDefaultRpcKwargs(
                    v2_fusion_pb2_grpc.ProximaFusionServiceStub(channel),
                    self._rpc_kwargs(),
                )
                return operation_func(stub)

        except grpc.RpcError as e:
            logger.error(f"gRPC {operation_name} RPC error: {e.code()} - {e.details()}")
            details = e.details() or str(e)
            if e.code() == grpc.StatusCode.UNAVAILABLE or "connect" in details.lower():
                raise ProximaDBError(f"{operation_name} connection failed: {details}")
            raise ProximaDBError(f"{operation_name} RPC failed: {details}")
        except Exception as e:
            logger.error(f"gRPC {operation_name} failed: {e}")
            raise ProximaDBError(f"{operation_name} failed: {e}")

    def _execute_entity_with_pool(self, operation_name: str, operation_func):
        """Execute entity operation using the v2 ProximaEntityService."""
        try:
            from proximadb.v2 import (
                entity_pb2_grpc as v2_entity_pb2_grpc,  # type: ignore
            )
        except ImportError:
            raise ProximaDBError(
                "v2 entity gRPC stubs not available. Regenerate Python protobuf stubs."
            )

        if not GRPC_AVAILABLE:
            raise ProximaDBError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )

        try:
            with GrpcChannelContext(self._connection_pool) as channel:
                stub = _StubWithDefaultRpcKwargs(
                    v2_entity_pb2_grpc.ProximaEntityServiceStub(channel),
                    self._rpc_kwargs(),
                )
                return operation_func(stub)

        except grpc.RpcError as e:
            logger.error(f"gRPC {operation_name} RPC error: {e.code()} - {e.details()}")
            details = e.details() or str(e)
            if e.code() == grpc.StatusCode.UNAVAILABLE or "connect" in details.lower():
                raise ProximaDBError(f"{operation_name} connection failed: {details}")
            raise ProximaDBError(f"{operation_name} RPC failed: {details}")
        except Exception as e:
            logger.error(f"gRPC {operation_name} failed: {e}")
            raise ProximaDBError(f"{operation_name} failed: {e}")

    def _execute_document_with_pool(self, operation_name: str, operation_func):
        """Execute document operation using the v2 ProximaDocumentService."""
        try:
            from proximadb.v2 import (
                document_pb2_grpc as v2_document_pb2_grpc,  # type: ignore
            )
        except ImportError:
            raise ProximaDBError(
                "v2 document gRPC stubs not available. Regenerate Python protobuf stubs."
            )

        if not GRPC_AVAILABLE:
            raise ProximaDBError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )

        try:
            with GrpcChannelContext(self._connection_pool) as channel:
                stub = _StubWithDefaultRpcKwargs(
                    v2_document_pb2_grpc.ProximaDocumentServiceStub(channel),
                    self._rpc_kwargs(),
                )
                return operation_func(stub)

        except grpc.RpcError as e:
            logger.error(f"gRPC {operation_name} RPC error: {e.code()} - {e.details()}")
            details = e.details() or str(e)
            if e.code() == grpc.StatusCode.UNAVAILABLE or "connect" in details.lower():
                raise ProximaDBError(f"{operation_name} connection failed: {details}")
            raise ProximaDBError(f"{operation_name} RPC failed: {details}")
        except Exception as e:
            logger.error(f"gRPC {operation_name} failed: {e}")
            raise ProximaDBError(f"{operation_name} failed: {e}")

    def get_pool_metrics(self):
        """Get connection pool performance metrics"""
        if self._connection_pool:
            return self._connection_pool.get_metrics()
        return None

    def close(self):
        """Close the connection pool and cleanup"""
        if self._connection_pool:
            try:
                self._connection_pool.close()
                logger.info("gRPC connection pool closed")
            except Exception as e:
                logger.warning(f"Error closing gRPC connection pool: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def health_check(self) -> HealthCheckResponse:
        """
        Check server health via gRPC health check

        Returns:
            HealthCheckResponse with:
                - healthy: bool
                - latency_ms: float
                - status: str
                - server_address: str
                - details: Optional[str]
                - version: Optional[str]
        """
        import time

        if not GRPC_AVAILABLE:
            raise ProximaDBError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )

        try:
            start_time = time.time()

            # Use list_collections as a lightweight health check
            # This verifies the server is responding and the connection pool works
            with GrpcChannelContext(self._connection_pool) as channel:
                # v2: lightweight health probe via ProximaRecordService.ListCollections.
                stub = _StubWithDefaultRpcKwargs(
                    v2_record_pb2_grpc.ProximaRecordServiceStub(channel),
                    self._rpc_kwargs(),
                )
                req = v2_record_pb2.V2ListCollectionsRequest(limit=1)
                stub.ListCollections(req, timeout=self.timeout)

                latency_ms = (time.time() - start_time) * 1000

                return HealthCheckResponse(
                    healthy=True,
                    latency_ms=latency_ms,
                    status="connected",
                    server_address=self.server_address,
                )

        except grpc.RpcError as e:
            return HealthCheckResponse(
                healthy=False,
                latency_ms=-1,
                status=f"error: {e.code()}",
                server_address=self.server_address,
                details=e.details(),
            )
        except Exception as e:
            return HealthCheckResponse(
                healthy=False,
                latency_ms=-1,
                status=f"error: {type(e).__name__}",
                server_address=self.server_address,
                details=str(e),
            )

    # Health check via REST endpoint (gRPC doesn't have dedicated Health service in v1)

    # Graph (canonical v2 ProximaGraphService)
    def shortest_path(
        self,
        start_node_id: str,
        target_node_id: str,
        max_depth: int | None = None,
        edge_types: list[str] | None = None,
        algorithm: str = "DIJKSTRA",
        k: int | None = None,
        enable_prefetch: bool | None = None,
        prefetch_budget: int | None = None,
        graph_id: str = "default",
    ) -> dict[str, Any]:
        """Compute shortest path via ProximaGraphService.ShortestPath.

        Per-call prefetch overrides are v2-native request fields
        (``enable_prefetch`` / ``prefetch_budget``) rather than gRPC metadata.
        Returns the ``GraphShortestPathResponse`` proto (``node_ids``,
        ``total_weight``, ``found``).
        """
        if not GRPC_AVAILABLE:
            raise ProximaDBError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )
        if v2_graph_pb2_grpc is None or v2_graph_pb2 is None:
            raise ProximaDBError(
                "ProximaGraphService stubs not found. Run: make -C clients/python gen-proto"
            )

        def _op(channel):
            stub = _StubWithDefaultRpcKwargs(
                v2_graph_pb2_grpc.ProximaGraphServiceStub(channel), self._rpc_kwargs()
            )
            algo_enum = {
                "DIJKSTRA": v2_graph_pb2.GRAPH_SHORTEST_PATH_ALGORITHM_DIJKSTRA,
                "ASTAR": v2_graph_pb2.GRAPH_SHORTEST_PATH_ALGORITHM_ASTAR,
            }.get(
                algorithm.upper(),
                v2_graph_pb2.GRAPH_SHORTEST_PATH_ALGORITHM_DIJKSTRA,
            )

            req = v2_graph_pb2.GraphShortestPathRequest(
                graph_id=graph_id,
                start_node_id=start_node_id,
                target_node_id=target_node_id,
                max_depth=max_depth or 0,
                edge_types=edge_types or [],
                algorithm=algo_enum,
                k=k or 0,
            )
            if enable_prefetch is not None:
                req.enable_prefetch = enable_prefetch
            if prefetch_budget is not None:
                req.prefetch_budget = prefetch_budget

            return stub.ShortestPath(req, timeout=self.timeout)

        try:
            with GrpcChannelContext(self._connection_pool) as channel:
                return _op(channel)
        except grpc.RpcError as e:
            logger.error(f"gRPC shortest_path RPC error: {e.code()} - {e.details()}")
            raise ProximaDBError(f"shortest_path RPC failed: {e.details()}")
        except Exception as e:
            logger.error(f"gRPC shortest_path failed: {e}")
            raise ProximaDBError(f"shortest_path failed: {e}")

    # SQL (v1)
    def execute_sql(
        self,
        query: str,
        parameters: list | None = None,
        collection: str | None = None,
    ):
        """Execute SQL over gRPC (a supported SQL surface).

        gRPC SQL is a first-class, supported path — token/SSO-authenticated and
        high-throughput programmatic SQL for SaaS/SDK clients. It is coequal with
        pgwire (the PostgreSQL-ecosystem surface for psql/JDBC/BI/ORM); use
        whichever fits the client. For non-SQL queries (UQL/AQL/Federated) use the
        ``/api/v2/query`` surface.

        Args:
            query: SQL text
            parameters: Optional list of rich values (scalars, bytes, lists, dicts)
            collection: Optional default collection context
        Returns:
            ExecuteQueryResponse as dict-like (via proto object fields)
        """
        if not GRPC_AVAILABLE:
            raise ProximaDBError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )

        try:
            with GrpcChannelContext(self._connection_pool) as channel:
                # v2: SQL is served by ProximaRecordService.ExecuteQuery (the v1
                # QueryService is flag-gated off). Note: parameterized params,
                # rows_scanned and column_types are not carried by the v2 query
                # messages yet; values arrive as decoded TypedValue.
                stub = _StubWithDefaultRpcKwargs(
                    v2_record_pb2_grpc.ProximaRecordServiceStub(channel),
                    self._rpc_kwargs(),
                )
                req = v2_record_pb2.V2QueryRequest(
                    query=query, collection_id=collection or ""
                )
                resp = stub.ExecuteQuery(req, timeout=self.timeout)
                rows = [
                    {
                        k: self._v2_typed_value_to_python(v)
                        for k, v in row.values.items()
                    }
                    for row in resp.rows
                ]
                return {
                    "rows": rows,
                    "row_count": len(rows),
                    "rows_returned": resp.rows_returned,
                    "execution_time_ms": resp.execution_time_ms,
                    "columns": list(resp.columns),
                }
        except grpc.RpcError as e:
            logger.error(f"gRPC execute_sql RPC error: {e.code()} - {e.details()}")
            raise ProximaDBError(f"execute_sql RPC failed: {e.details()}")
        except Exception as e:
            logger.error(f"gRPC execute_sql failed: {e}")
            raise ProximaDBError(f"execute_sql failed: {e}")

    # Collections (v1)
    def create_collection_v1(
        self,
        name: str,
        dimension: int,
        distance_metric: int,
        storage_engine: int,
        tags: list | None = None,
        description: str | None = None,
    ):
        def _op(channel):
            stub = _StubWithDefaultRpcKwargs(
                v1_collection_pb2_grpc.CollectionServiceStub(channel),
                self._rpc_kwargs(),
            )
            cfg = v1_collection_types_pb2.CollectionConfig(
                name=name,
                dimension=dimension,
                distance_metric=distance_metric,
                storage_engine=storage_engine,
                tags=tags or [],
                description=description or "",
            )
            return stub.CreateCollection(cfg, timeout=self.timeout)

        return self._execute_with_pool("create_collection_v1", _op)

    def get_collection_v1(self, collection_id: str):
        def _op(channel):
            stub = _StubWithDefaultRpcKwargs(
                v1_collection_pb2_grpc.CollectionServiceStub(channel),
                self._rpc_kwargs(),
            )
            req = v1_collection_types_pb2.GetCollectionRequest(
                collection_id=collection_id
            )
            return stub.GetCollection(req, timeout=self.timeout)

        return self._execute_with_pool("get_collection_v1", _op)

    def list_collections_v1(
        self,
        limit: int | None = None,
        offset: int | None = None,
        include_stats: bool | None = None,
    ):
        def _op(channel):
            stub = _StubWithDefaultRpcKwargs(
                v1_collection_pb2_grpc.CollectionServiceStub(channel),
                self._rpc_kwargs(),
            )
            req = v1_collection_types_pb2.ListCollectionsRequest(
                limit=limit or 0,
                offset=offset or 0,
                include_stats=include_stats or False,
            )
            return stub.ListCollections(req, timeout=self.timeout)

        return self._execute_with_pool("list_collections_v1", _op)

    def delete_collection_v1(self, collection_id: str):
        def _op(channel):
            stub = _StubWithDefaultRpcKwargs(
                v1_collection_pb2_grpc.CollectionServiceStub(channel),
                self._rpc_kwargs(),
            )
            req = v1_collection_types_pb2.DeleteCollectionRequest(
                collection_id=collection_id
            )
            return stub.DeleteCollection(req, timeout=self.timeout)

        return self._execute_with_pool("delete_collection_v1", _op)

    # Collection Operations - Unified Interface
    def create_collection(
        self,
        name: str,
        dimension: int,
        distance_metric: int = None,
        indexing_algorithm: int = None,
        storage_engine: int = None,
        engine: int = None,  # Alias for storage_engine (backward compatibility)
        filterable_columns: list[Any] = None,
        index_configs: list[Any] = None,
        quantization_config: Any = None,
        canonical_embedding_precision: int | None = None,
    ) -> Any:
        """Create collection with unified interface

        Args:
            name: Collection identifier
            dimension: Vector dimension
            distance_metric: Distance metric enum value
            indexing_algorithm: Indexing algorithm enum value
            storage_engine: Storage engine enum value
            engine: Alias for storage_engine (for backward compatibility)
            filterable_columns: Fields that can be filtered
            index_configs: Index configuration parameters
            quantization_config: Quantization configuration

        Returns:
            Collection creation result
        """
        # Handle backward compatibility: engine is an alias for storage_engine
        if engine is not None and storage_engine is None:
            storage_engine = engine

        # Convert string storage engine names to enum integers if needed
        if storage_engine is not None and isinstance(storage_engine, str):
            from proximadb_sdk.models import StorageEngineType

            storage_engine_map = {
                "viper": StorageEngineType.VIPER,
                "sst": StorageEngineType.SST,
                "nova": StorageEngineType.NOVA,
                "helix": StorageEngineType.HELIX,
                "swift": StorageEngineType.SWIFT,
                "raptor": StorageEngineType.RAPTOR,
            }
            storage_engine_str = storage_engine.lower()
            if storage_engine_str in storage_engine_map:
                storage_engine = int(storage_engine_map[storage_engine_str])
            else:
                raise ValueError(
                    f"Unknown storage engine: {storage_engine}. Valid options: {list(storage_engine_map.keys())}"
                )

        def _create_collection_operation(stub):
            # v2 V2CollectionConfig is self-contained: distance_metric/storage_engine
            # are lowercase strings (mapped from the int enums). index_specs +
            # quantization are v2-native structured config; canonical_embedding_precision
            # remains server-default on the v2 path.
            dm_str = ""
            if distance_metric is not None:
                try:
                    from proximadb_sdk.models import DistanceMetricType

                    dm_str = DistanceMetricType(distance_metric).name.lower()
                except (ValueError, ImportError):
                    dm_str = ""
            se_str = ""
            if storage_engine is not None:
                if isinstance(storage_engine, str):
                    se_str = storage_engine.lower()
                else:
                    try:
                        from proximadb_sdk.models import StorageEngineType

                        se_str = StorageEngineType(storage_engine).name.lower()
                    except (ValueError, ImportError):
                        se_str = ""
            config = v2_record_pb2.V2CollectionConfig(
                name=name,
                dimension=dimension,
                distance_metric=dm_str,
                storage_engine=se_str,
                filterable_columns=[
                    c if isinstance(c, str) else getattr(c, "name", str(c))
                    for c in (filterable_columns or [])
                ],
                index_specs=_build_v2_index_specs(index_configs, indexing_algorithm),
                quantization=_build_v2_quantization(quantization_config),
            )
            response = stub.CreateCollection(config, timeout=self.timeout)
            return CollectionWrapper(response)

        return self._execute_collection_with_pool(
            "create_collection", _create_collection_operation
        )

    def get_collection(self, name: str) -> Any:
        """Get collection metadata"""

        def _get_collection_operation(stub):
            request = v2_record_pb2.V2GetCollectionRequest(collection_id=name)
            response = stub.GetCollection(request, timeout=self.timeout)
            return CollectionWrapper(response)

        return self._execute_collection_with_pool(
            "get_collection", _get_collection_operation
        )

    def list_collections(self) -> list[Any]:
        """List all collections"""

        def _list_collections_operation(stub):
            request = v2_record_pb2.V2ListCollectionsRequest()
            response = stub.ListCollections(request, timeout=self.timeout)
            return [CollectionWrapper(coll) for coll in response.collections]

        return self._execute_collection_with_pool(
            "list_collections", _list_collections_operation
        )

    def delete_collection(self, collection_id: str) -> DeleteCollectionResponse:
        """Delete collection"""

        def _delete_collection_operation(stub):
            request = v2_record_pb2.V2DeleteCollectionRequest(
                collection_id=collection_id
            )
            response = stub.DeleteCollection(request, timeout=self.timeout)
            return DeleteCollectionResponse(
                success=response.success, collection_id=collection_id, status="deleted"
            )

        return self._execute_collection_with_pool(
            "delete_collection", _delete_collection_operation
        )

    # Record Operations - Unified Interface
    def _python_to_v2_typed_value(self, value: Any):
        """Encode Python values into v2 ProximaValue/TypedValue protobufs.

        Thin delegate to the transport-agnostic codec shared with the
        native-async gRPC client (``protocols/_grpc_v2_codec``).
        """
        return _codec.python_to_v2_typed_value(v2_record_pb2, value)

    def _v2_typed_value_to_python(self, value: Any) -> Any:
        """Decode v2 TypedValue protobufs into Python values (shared codec)."""
        return _codec.v2_typed_value_to_python(value)

    def _normalize_vector_alias_records(
        self, vectors: list[dict[str, Any] | Any]
    ) -> list[dict[str, Any]]:
        """Normalize legacy vector alias inputs to v2 ProximaRecord payloads."""
        records: list[dict[str, Any]] = []
        for index, vector_data in enumerate(vectors):
            if hasattr(vector_data, "model_dump"):
                vector_dict = vector_data.model_dump(exclude_none=True)
            elif hasattr(vector_data, "dict"):
                vector_dict = vector_data.dict(exclude_none=True)
            elif hasattr(vector_data, "__dict__") and not isinstance(vector_data, dict):
                vector_dict = {
                    key: value
                    for key, value in vector_data.__dict__.items()
                    if not key.startswith("_")
                }
            else:
                vector_dict = vector_data

            if not isinstance(vector_dict, dict):
                vector_dict = {"id": f"record_{index}", "vector": vector_dict}

            props = (
                vector_dict.get("props")
                or vector_dict.get("metadata")
                or vector_dict.get("typed_fields")
                or {}
            )
            record = {
                "id": vector_dict.get("id")
                or vector_dict.get("oid")
                or f"record_{index}",
                "vector": vector_dict.get("vector"),
                "props": props,
            }
            for field in (
                "timestamp_ms",
                "timestamp",
                "updated_at_ms",
                "updated_at",
                "expires_at_ms",
                "expires_at",
                "version",
                "source",
            ):
                if vector_dict.get(field) is not None:
                    record[field] = vector_dict[field]
            records.append(record)
        return records

    def _record_proto_for_grpc(
        self, record: ProximaRecord | dict[str, Any], index: int = 0
    ):
        """Build a v2 ProximaRecord proto (shared transport-agnostic codec)."""
        return _codec.record_proto_for_grpc(v2_record_pb2, record, index)

    def _v2_record_batch_result(self, response) -> BatchResult:
        """Parse a v2 batch result into a BatchResult (shared codec)."""
        return _codec.v2_record_batch_result(response)

    def insert_records(
        self,
        collection_id: str,
        records: list[ProximaRecord | dict[str, Any]],
        **kwargs,
    ) -> BatchResult:
        upsert = bool(kwargs.pop("upsert", False))
        if upsert:
            return self.upsert_records(collection_id, records, **kwargs)

        if v2_record_pb2 is None or v2_record_pb2_grpc is None:
            raise ProximaDBError("v2 ProximaRecord gRPC stubs are required")

        request = v2_record_pb2.ProximaRecordBatch(
            collection_id=collection_id,
            write_mode=v2_record_pb2.INSERT,
            validate_schema=bool(kwargs.get("validate_schema", True)),
            return_ids=bool(kwargs.get("return_ids", True)),
            return_errors=bool(kwargs.get("return_errors", True)),
        )
        request.records.extend(
            self._record_proto_for_grpc(record, index)
            for index, record in enumerate(records)
        )
        if kwargs.get("schema_id"):
            request.schema_id = str(kwargs["schema_id"])

        def _insert_records_operation(stub):
            return stub.InsertRecords(request, timeout=self.timeout)

        return self._v2_record_batch_result(
            self._execute_record_with_pool("insert_records", _insert_records_operation)
        )

    def upsert_records(
        self,
        collection_id: str,
        records: list[ProximaRecord | dict[str, Any]],
        **kwargs,
    ) -> BatchResult:
        """Upsert ProximaRecord-shaped payloads."""
        kwargs.pop("upsert", None)
        if v2_record_pb2 is None or v2_record_pb2_grpc is None:
            raise ProximaDBError("v2 ProximaRecord gRPC stubs are required")

        request = v2_record_pb2.ProximaRecordBatch(
            collection_id=collection_id,
            write_mode=v2_record_pb2.UPSERT,
            validate_schema=bool(kwargs.get("validate_schema", True)),
            return_ids=bool(kwargs.get("return_ids", True)),
            return_errors=bool(kwargs.get("return_errors", True)),
        )
        request.records.extend(
            self._record_proto_for_grpc(record, index)
            for index, record in enumerate(records)
        )
        if kwargs.get("schema_id"):
            request.schema_id = str(kwargs["schema_id"])

        def _upsert_records_operation(stub):
            return stub.UpsertRecords(request, timeout=self.timeout)

        return self._v2_record_batch_result(
            self._execute_record_with_pool("upsert_records", _upsert_records_operation)
        )

    # Vector Compatibility Aliases
    def insert_vectors(
        self, collection_id: str, vectors: list[dict[str, Any]], upsert: bool = False
    ) -> VectorOperationResponse:
        """Insert vectors through the v2 ProximaRecord gRPC surface.

        Args:
            collection_id: Target collection ID
            vectors: List of vector objects with format:
                    [{"id": "vec1", "vector": [0.1, 0.2, ...], "metadata": {...}}, ...]
            upsert: Whether to update existing vectors

        Returns:
            VectorOperationResponse with operation details
        """
        records = self._normalize_vector_alias_records(vectors)
        batch_result = (
            self.upsert_records(collection_id, records)
            if upsert
            else self.insert_records(collection_id, records)
        )
        return VectorOperationResponse(
            success=batch_result.failed == 0,
            operation="UPSERT" if upsert else "INSERT",
            metrics=batch_result.metrics,
            vector_ids=[record["id"] for record in records],
            error_message=(
                "; ".join(batch_result.errors) if batch_result.errors else None
            ),
        )

    def search_vectors(
        self,
        collection_id: str,
        query_vectors: list[list[float]] = None,
        query_vector: list[float] = None,
        top_k: int = 10,
        metadata_filters: dict[str, Any] | None = None,
        include_vectors: bool = False,
        include_metadata: bool = True,
        search_hints: dict[str, Any] | None = None,
    ) -> SearchResult:
        """Search vectors through the v2 ProximaRecord gRPC surface.

        Args:
            collection_id: Target collection ID
            query_vector: Query vector
            top_k: Number of results to return
            metadata_filters: Metadata filter conditions
            include_vectors: Include vector data in results
            include_metadata: Include metadata in results

        Returns:
            SearchResult with found vectors
        """
        # Handle both query_vector and query_vectors params
        if query_vectors is None and query_vector is not None:
            query_vectors = [query_vector]
        elif query_vectors is None:
            raise ValueError("Either query_vector or query_vectors must be provided")

        def _search_vectors_operation(stub):
            search_results = []
            for qv in query_vectors:
                request = v2_record_pb2.TypedSearchRequest(
                    collection_id=collection_id,
                    top_k=top_k,
                    include_vector=include_vectors,
                    include_text_fields=False,
                )
                request.query_vector.extend(float(value) for value in qv)
                request.filter_logic = v2_record_pb2.AND
                if metadata_filters:
                    for key, value in metadata_filters.items():
                        filter_condition = request.filters.add()
                        filter_condition.field_name = str(key)
                        filter_condition.operator = v2_record_pb2.EQ
                        filter_condition.value.CopyFrom(
                            self._python_to_v2_typed_value(value)
                        )
                if search_hints:
                    request.search_hints.update(
                        {str(key): str(value) for key, value in search_hints.items()}
                    )

                response = stub.Search(request, timeout=self.timeout)
                for rank, result in enumerate(response.results):
                    metadata = None
                    if include_metadata:
                        metadata = {
                            key: self._v2_typed_value_to_python(value)
                            for key, value in result.props.items()
                        }
                    search_results.append(
                        SearchResult(
                            id=result.id,
                            score=result.score,
                            rank=rank,
                            vector=(
                                list(result.vector)
                                if include_vectors and result.vector
                                else None
                            ),
                            metadata=metadata,
                            timestamp=(
                                result.timestamp_ms
                                if result.HasField("timestamp_ms")
                                else None
                            ),
                            version=(
                                result.version if result.HasField("version") else None
                            ),
                            source=(
                                result.source if result.HasField("source") else None
                            ),
                        )
                    )

            return SearchResultsWrapper(search_results)

        return self._execute_record_with_pool(
            "search_vectors", _search_vectors_operation
        )

    def search(
        self,
        collection_id: str = None,  # Can be positional or keyword
        query_vector: list[float] = None,  # Can be positional
        query_vectors: list[list[float]] = None,
        top_k: int = None,
        k: int = None,  # Backward compatibility alias for top_k
        collection_name: str = None,  # Backward compatibility alias
        metadata_filters: dict[str, Any] | None = None,
        include_vectors: bool = False,
        include_metadata: bool = True,
        search_hints: dict[str, Any] | None = None,
    ) -> SearchResult:
        """
        Alias for search_vectors() for backward compatibility and convenience

        This method provides the same functionality as search_vectors() but with
        a shorter, more intuitive name commonly expected by users.

        Args:
            collection_id: Target collection ID (can use collection_name instead)
            query_vector: Single query vector (convenience param)
            query_vectors: Multiple query vectors (batch search)
            top_k: Number of results to return per query
            k: Alias for top_k (backward compatibility)
            collection_name: Alias for collection_id (backward compatibility)
            metadata_filters: Metadata filter conditions
            include_vectors: Include vector data in results
            include_metadata: Include metadata in results
            search_hints: Optional search optimization hints

        Returns:
            SearchResult with found vectors
        """
        # Handle backward compatibility aliases
        if collection_name is not None and collection_id is None:
            collection_id = collection_name
        if collection_id is None:
            raise ValueError("Either collection_id or collection_name must be provided")
        if k is not None and top_k is None:
            top_k = k
        if top_k is None:
            top_k = 10  # Default value

        return self.search_vectors(
            collection_id=collection_id,
            query_vector=query_vector,
            query_vectors=query_vectors,
            top_k=top_k,
            metadata_filters=metadata_filters,
            include_vectors=include_vectors,
            include_metadata=include_metadata,
            search_hints=search_hints,
        )

    def get_vector(
        self,
        collection_id: str,
        vector_id: str,
        include_vector: bool = True,
        include_metadata: bool = True,
    ) -> dict[str, Any]:
        """Get a single record by ID via v2 ProximaRecordService.GetRecord."""

        def _get_record_operation(stub):
            request = v2_record_pb2.GetRecordRequest(
                collection_id=collection_id,
                id=vector_id,
                include_vector=include_vector,
            )
            response = stub.GetRecord(request, timeout=self.timeout)
            if not response.found or not response.HasField("record"):
                raise ProximaDBError(f"Vector {vector_id} not found")
            rec = response.record
            result: dict[str, Any] = {"id": rec.id}
            if include_vector and rec.vector:
                result["vector"] = list(rec.vector)
            if include_metadata and rec.props:
                result["metadata"] = {
                    k: self._v2_typed_value_to_python(v) for k, v in rec.props.items()
                }
            return VectorWrapper(result)

        return self._execute_collection_with_pool("get_vector", _get_record_operation)

    def update_vector(
        self,
        collection_id: str,
        vector_id: str,
        vector: list[float] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Update vector data and/or metadata"""
        # Treat vector update as an upsert over the v2 ProximaRecord contract.
        vector_data = {"id": vector_id}
        if vector is not None:
            vector_data["vector"] = vector
        if metadata is not None:
            vector_data["metadata"] = metadata

        # Use upsert functionality
        result = self.insert_vectors(
            collection_id=collection_id, vectors=[vector_data], upsert=True
        )

        return {
            "status": "updated" if result.success else "failed",
            "vector_id": vector_id,
            "success": result.success,
        }

    def delete_vector(self, collection_id: str, vector_id: str) -> dict[str, Any]:
        """Delete a vector through the v2 ProximaRecord gRPC surface."""

        def _delete_vector_operation(stub):
            record = v2_record_pb2.ProximaRecord(id=vector_id)
            request = v2_record_pb2.ProximaRecordBatch(
                collection_id=collection_id,
                write_mode=v2_record_pb2.DELETE,
                return_ids=True,
                return_errors=True,
            )
            request.records.append(record)
            response = stub.DeleteRecords(request, timeout=self.timeout)
            return DictWrapper(
                {
                    "status": "deleted" if response.failed_count == 0 else "failed",
                    "vector_id": vector_id,
                    "success": response.failed_count == 0,
                }
            )

        return self._execute_record_with_pool("delete_vector", _delete_vector_operation)

    def delete_vectors(
        self, collection_id: str, vector_ids: list[str]
    ) -> dict[str, Any]:
        """Delete multiple vectors through the v2 ProximaRecord gRPC surface."""

        def _delete_vectors_operation(stub):
            request = v2_record_pb2.ProximaRecordBatch(
                collection_id=collection_id,
                write_mode=v2_record_pb2.DELETE,
                return_ids=True,
                return_errors=True,
            )
            for vector_id in vector_ids:
                request.records.append(v2_record_pb2.ProximaRecord(id=vector_id))
            response = stub.DeleteRecords(request, timeout=self.timeout)

            return {
                "status": "completed",
                "deleted_count": int(response.success_count),
                "failed_count": int(response.failed_count),
                "total_requested": len(vector_ids),
            }

        return self._execute_record_with_pool(
            "delete_vectors", _delete_vectors_operation
        )

    def insert_vector(
        self,
        collection_id: str,
        vector_id: str,
        vector: list[float],
        metadata: dict[str, Any] | None = None,
        upsert: bool = False,
    ) -> VectorOperationResponse:
        """Insert a single vector - alias for batch insert with one vector

        Args:
            collection_id: Collection ID or name
            vector_id: Vector identifier
            vector: Vector data
            metadata: Optional metadata
            upsert: If True, update existing vector

        Returns:
            VectorOperationResponse
        """
        # Use the batch insert with a single vector
        vector_data = {"id": vector_id, "vector": vector}
        if metadata:
            vector_data["metadata"] = metadata

        return self.insert_vectors(
            collection_id=collection_id, vectors=[vector_data], upsert=upsert
        )

    # === GRAPH OPERATIONS (v1) ===

    def _convert_to_property_value(self, value: Any):
        """Convert Python value to a v2 GraphPropertyValue proto"""
        if v2_graph_pb2 is None:
            raise ProximaDBError(
                "Graph protos not available. Run: make -C clients/python gen-proto"
            )

        if isinstance(value, str):
            return v2_graph_pb2.GraphPropertyValue(string_value=value)
        elif isinstance(value, bool):
            return v2_graph_pb2.GraphPropertyValue(bool_value=value)
        elif isinstance(value, int):
            return v2_graph_pb2.GraphPropertyValue(int_value=value)
        elif isinstance(value, float):
            return v2_graph_pb2.GraphPropertyValue(double_value=value)
        elif isinstance(value, bytes):
            return v2_graph_pb2.GraphPropertyValue(bytes_value=value)
        elif isinstance(value, list):
            array_values = [self._convert_to_property_value(item) for item in value]
            return v2_graph_pb2.GraphPropertyValue(
                array_value=v2_graph_pb2.GraphPropertyArray(values=array_values)
            )
        elif isinstance(value, dict):
            map_fields = {
                k: self._convert_to_property_value(v) for k, v in value.items()
            }
            return v2_graph_pb2.GraphPropertyValue(
                map_value=v2_graph_pb2.GraphPropertyMap(fields=map_fields)
            )
        else:
            return v2_graph_pb2.GraphPropertyValue(string_value=str(value))

    def _convert_from_property_value(self, prop_value) -> Any:
        """Convert a v2 GraphPropertyValue proto to a Python value"""
        if prop_value.HasField("string_value"):
            return prop_value.string_value
        elif prop_value.HasField("int_value"):
            return prop_value.int_value
        elif prop_value.HasField("double_value"):
            return prop_value.double_value
        elif prop_value.HasField("bool_value"):
            return prop_value.bool_value
        elif prop_value.HasField("bytes_value"):
            return prop_value.bytes_value
        elif prop_value.HasField("array_value"):
            return [
                self._convert_from_property_value(item)
                for item in prop_value.array_value.values
            ]
        elif prop_value.HasField("map_value"):
            return {
                k: self._convert_from_property_value(v)
                for k, v in prop_value.map_value.fields.items()
            }
        else:
            return None

    def _convert_node_from_proto(self, node) -> dict[str, Any]:
        """Convert Node proto to dictionary"""
        from datetime import datetime

        return {
            "id": node.id,
            "labels": list(node.labels),
            "properties": {
                k: self._convert_from_property_value(v)
                for k, v in node.properties.items()
            },
            "created_at": (
                datetime.fromtimestamp(node.created_at_ms / 1000, tz=UTC).isoformat()
                if node.created_at_ms
                else None
            ),
            "updated_at": (
                datetime.fromtimestamp(node.updated_at_ms / 1000, tz=UTC).isoformat()
                if node.updated_at_ms
                else None
            ),
        }

    def _convert_edge_from_proto(self, edge) -> dict[str, Any]:
        """Convert Edge proto to dictionary"""
        from datetime import datetime

        return {
            "id": edge.id,
            "from_node_id": edge.from_node_id,
            "to_node_id": edge.to_node_id,
            "edge_type": edge.edge_type,
            "properties": {
                k: self._convert_from_property_value(v)
                for k, v in edge.properties.items()
            },
            "weight": edge.weight if edge.HasField("weight") else None,
            "created_at": (
                datetime.fromtimestamp(edge.created_at_ms / 1000, tz=UTC).isoformat()
                if edge.created_at_ms
                else None
            ),
            "updated_at": (
                datetime.fromtimestamp(edge.updated_at_ms / 1000, tz=UTC).isoformat()
                if edge.updated_at_ms
                else None
            ),
        }

    def _convert_path_from_proto(self, path) -> list[str]:
        """Convert GraphPath proto to list of node IDs"""
        if hasattr(path, "node_ids"):
            return list(path.node_ids)
        else:
            return []

    def create_node(
        self,
        node_id: str,
        labels: list[str],
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        graph_id: str = "default",
    ) -> dict[str, Any]:
        """Create a graph node via gRPC

        Args:
            node_id: Unique identifier for the node
            labels: List of labels for the node
            properties: Optional dictionary of node properties
            embedding: Optional embedding vector for the node
            graph_id: Graph collection ID (defaults to "default")

        Returns:
            Dictionary representation of the created node
        """
        if not GRPC_AVAILABLE:
            raise ProximaDBError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )
        if v2_graph_pb2_grpc is None or v2_graph_pb2 is None:
            raise ProximaDBError(
                "ProximaGraphService stubs not found. Run: make -C clients/python gen-proto"
            )

        def _op(channel):
            stub = _StubWithDefaultRpcKwargs(
                v2_graph_pb2_grpc.ProximaGraphServiceStub(channel), self._rpc_kwargs()
            )

            node_properties = {}
            if properties:
                for key, value in properties.items():
                    node_properties[key] = self._convert_to_property_value(value)

            node = v2_graph_pb2.GraphNode(
                id=node_id, labels=labels, properties=node_properties
            )

            request = v2_graph_pb2.CreateGraphNodeRequest(graph_id=graph_id, node=node)
            response = stub.CreateNode(request, timeout=self.timeout)
            return self._convert_node_from_proto(response.node)

        try:
            with GrpcChannelContext(self._connection_pool) as channel:
                return _op(channel)
        except grpc.RpcError as e:
            logger.error(f"gRPC create_node RPC error: {e.code()} - {e.details()}")
            raise ProximaDBError(f"create_node RPC failed: {e.details()}")
        except Exception as e:
            logger.error(f"gRPC create_node failed: {e}")
            raise ProximaDBError(f"create_node failed: {e}")

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
        """Create a graph edge via gRPC

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
        if not GRPC_AVAILABLE:
            raise ProximaDBError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )
        if v2_graph_pb2_grpc is None or v2_graph_pb2 is None:
            raise ProximaDBError(
                "ProximaGraphService stubs not found. Run: make -C clients/python gen-proto"
            )

        def _op(channel):
            stub = _StubWithDefaultRpcKwargs(
                v2_graph_pb2_grpc.ProximaGraphServiceStub(channel), self._rpc_kwargs()
            )

            edge_properties = {}
            if properties:
                for key, value in properties.items():
                    edge_properties[key] = self._convert_to_property_value(value)

            edge = v2_graph_pb2.GraphEdge(
                id=edge_id,
                from_node_id=from_node_id,
                to_node_id=to_node_id,
                edge_type=edge_type,
                properties=edge_properties,
            )

            if weight is not None:
                edge.weight = weight

            request = v2_graph_pb2.CreateGraphEdgeRequest(graph_id=graph_id, edge=edge)
            response = stub.CreateEdge(request, timeout=self.timeout)
            return self._convert_edge_from_proto(response.edge)

        try:
            with GrpcChannelContext(self._connection_pool) as channel:
                return _op(channel)
        except grpc.RpcError as e:
            logger.error(f"gRPC create_edge RPC error: {e.code()} - {e.details()}")
            raise ProximaDBError(f"create_edge RPC failed: {e.details()}")
        except Exception as e:
            logger.error(f"gRPC create_edge failed: {e}")
            raise ProximaDBError(f"create_edge failed: {e}")

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
        """Traverse graph from a starting node via gRPC

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
        if not GRPC_AVAILABLE:
            raise ProximaDBError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )
        if v2_graph_pb2_grpc is None or v2_graph_pb2 is None:
            raise ProximaDBError(
                "ProximaGraphService stubs not found. Run: make -C clients/python gen-proto"
            )

        def _op(channel):
            stub = _StubWithDefaultRpcKwargs(
                v2_graph_pb2_grpc.ProximaGraphServiceStub(channel), self._rpc_kwargs()
            )

            # Map algorithm string to enum
            algorithm_enum = v2_graph_pb2.GRAPH_TRAVERSAL_ALGORITHM_BFS
            if algorithm.upper() == "DFS":
                algorithm_enum = v2_graph_pb2.GRAPH_TRAVERSAL_ALGORITHM_DFS
            elif algorithm.upper() == "PARALLEL_BFS":
                algorithm_enum = v2_graph_pb2.GRAPH_TRAVERSAL_ALGORITHM_PARALLEL_BFS

            request = v2_graph_pb2.TraverseGraphRequest(
                graph_id=graph_id,
                start_node_id=start_node_id,
                max_depth=max_depth,
                edge_types=edge_types or [],
                node_labels=node_labels or [],
                algorithm=algorithm_enum,
            )

            if limit is not None:
                request.limit = limit

            response = stub.TraverseGraph(request, timeout=self.timeout)

            return {
                "nodes": [
                    self._convert_node_from_proto(node) for node in response.nodes
                ],
                "edges": [
                    self._convert_edge_from_proto(edge) for edge in response.edges
                ],
                "paths": [
                    self._convert_path_from_proto(path) for path in response.paths
                ],
                "stats": {
                    "nodes_visited": (
                        response.stats.nodes_visited
                        if hasattr(response, "stats")
                        else 0
                    ),
                    "edges_traversed": (
                        response.stats.edges_traversed
                        if hasattr(response, "stats")
                        else 0
                    ),
                    "max_depth_reached": (
                        response.stats.max_depth_reached
                        if hasattr(response, "stats")
                        else 0
                    ),
                    "execution_time_microseconds": (
                        response.stats.execution_time_microseconds
                        if hasattr(response, "stats")
                        else 0
                    ),
                },
            }

        try:
            with GrpcChannelContext(self._connection_pool) as channel:
                return _op(channel)
        except grpc.RpcError as e:
            logger.error(f"gRPC traverse_graph RPC error: {e.code()} - {e.details()}")
            raise ProximaDBError(f"traverse_graph RPC failed: {e.details()}")
        except Exception as e:
            logger.error(f"gRPC traverse_graph failed: {e}")
            raise ProximaDBError(f"traverse_graph failed: {e}")

    def query_nodes(
        self,
        labels: list[str] | None = None,
        properties: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        graph_id: str = "default",
    ) -> dict[str, Any]:
        """Query nodes by labels and properties via gRPC

        Args:
            labels: Optional list of labels to filter by
            properties: Optional dictionary of properties to filter by
            limit: Optional maximum number of results
            offset: Optional offset for pagination
            graph_id: Graph collection ID (defaults to "default")

        Returns:
            Dictionary with success status, nodes list, and total count
        """
        if not GRPC_AVAILABLE:
            raise ProximaDBError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )
        if v2_graph_pb2_grpc is None or v2_graph_pb2 is None:
            raise ProximaDBError(
                "ProximaGraphService stubs not found. Run: make -C clients/python gen-proto"
            )

        def _op(channel):
            stub = _StubWithDefaultRpcKwargs(
                v2_graph_pb2_grpc.ProximaGraphServiceStub(channel), self._rpc_kwargs()
            )

            filters = []
            if properties:
                for key, value in properties.items():
                    filters.append(
                        v2_graph_pb2.GraphPropertyFilter(
                            key=key,
                            operator=v2_graph_pb2.GRAPH_PROPERTY_FILTER_OPERATOR_EQUALS,
                            value=self._convert_to_property_value(value),
                        )
                    )

            request = v2_graph_pb2.QueryGraphNodesRequest(
                graph_id=graph_id, labels=labels or [], filters=filters
            )

            if limit is not None:
                request.limit = limit
            if offset is not None:
                request.offset = offset

            response = stub.QueryNodes(request, timeout=self.timeout)
            return {
                "success": True,
                "nodes": [
                    self._convert_node_from_proto(node) for node in response.nodes
                ],
                "total_count": len(response.nodes),
                "next_token": (
                    response.next_token if response.HasField("next_token") else None
                ),
            }

        try:
            with GrpcChannelContext(self._connection_pool) as channel:
                return _op(channel)
        except grpc.RpcError as e:
            logger.error(f"gRPC query_nodes RPC error: {e.code()} - {e.details()}")
            raise ProximaDBError(f"query_nodes RPC failed: {e.details()}")
        except Exception as e:
            logger.error(f"gRPC query_nodes failed: {e}")
            raise ProximaDBError(f"query_nodes failed: {e}")

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
        """Query edges by endpoints, type, and properties via gRPC."""
        if not GRPC_AVAILABLE:
            raise ProximaDBError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )
        if v2_graph_pb2_grpc is None or v2_graph_pb2 is None:
            raise ProximaDBError(
                "ProximaGraphService stubs not found. Run: make -C clients/python gen-proto"
            )

        def _op(channel):
            stub = _StubWithDefaultRpcKwargs(
                v2_graph_pb2_grpc.ProximaGraphServiceStub(channel), self._rpc_kwargs()
            )

            filters = []
            if properties:
                for key, value in properties.items():
                    filters.append(
                        v2_graph_pb2.GraphPropertyFilter(
                            key=key,
                            operator=v2_graph_pb2.GRAPH_PROPERTY_FILTER_OPERATOR_EQUALS,
                            value=self._convert_to_property_value(value),
                        )
                    )

            request = v2_graph_pb2.QueryGraphEdgesRequest(
                graph_id=graph_id,
                edge_types=[edge_type] if edge_type else [],
                filters=filters,
            )

            if from_node_id is not None:
                request.from_node_id = from_node_id
            if to_node_id is not None:
                request.to_node_id = to_node_id
            if limit is not None:
                request.limit = limit
            if offset is not None:
                request.offset = offset

            response = stub.QueryEdges(request, timeout=self.timeout)
            return {
                "success": True,
                "edges": [
                    self._convert_edge_from_proto(edge) for edge in response.edges
                ],
                "total_count": len(response.edges),
                "next_token": (
                    response.next_token if response.HasField("next_token") else None
                ),
            }

        try:
            with GrpcChannelContext(self._connection_pool) as channel:
                return _op(channel)
        except grpc.RpcError as e:
            logger.error(f"gRPC query_edges RPC error: {e.code()} - {e.details()}")
            raise ProximaDBError(f"query_edges RPC failed: {e.details()}")
        except Exception as e:
            logger.error(f"gRPC query_edges failed: {e}")
            raise ProximaDBError(f"query_edges failed: {e}")

    def get_node(
        self,
        node_id: str,
        graph_id: str = "default",
    ) -> dict[str, Any] | None:
        """Get a graph node by ID via gRPC."""
        if not GRPC_AVAILABLE:
            raise ProximaDBError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )
        if v2_graph_pb2_grpc is None or v2_graph_pb2 is None:
            raise ProximaDBError(
                "ProximaGraphService stubs not found. Run: make -C clients/python gen-proto"
            )

        def _op(channel):
            stub = _StubWithDefaultRpcKwargs(
                v2_graph_pb2_grpc.ProximaGraphServiceStub(channel), self._rpc_kwargs()
            )
            request = v2_graph_pb2.GetGraphNodeRequest(
                graph_id=graph_id, node_id=node_id
            )
            response = stub.GetNode(request, timeout=self.timeout)
            if not response.HasField("node"):
                return None
            return self._convert_node_from_proto(response.node)

        try:
            with GrpcChannelContext(self._connection_pool) as channel:
                return _op(channel)
        except grpc.RpcError as e:
            logger.error(f"gRPC get_node RPC error: {e.code()} - {e.details()}")
            raise ProximaDBError(f"get_node RPC failed: {e.details()}")
        except Exception as e:
            logger.error(f"gRPC get_node failed: {e}")
            raise ProximaDBError(f"get_node failed: {e}")

    def get_outgoing_edges(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        graph_id: str = "default",
    ) -> list[dict[str, Any]]:
        """Get outgoing graph edges for a node via gRPC."""
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
        """Get incoming graph edges for a node via gRPC."""
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
        """Delete a graph node by ID via gRPC."""
        if not GRPC_AVAILABLE:
            raise ProximaDBError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )
        if v2_graph_pb2_grpc is None or v2_graph_pb2 is None:
            raise ProximaDBError(
                "ProximaGraphService stubs not found. Run: make -C clients/python gen-proto"
            )

        def _op(channel):
            stub = _StubWithDefaultRpcKwargs(
                v2_graph_pb2_grpc.ProximaGraphServiceStub(channel), self._rpc_kwargs()
            )
            request = v2_graph_pb2.DeleteGraphNodeRequest(
                graph_id=graph_id, node_id=node_id
            )
            response = stub.DeleteNode(request, timeout=self.timeout)
            if response.HasField("node"):
                return self._convert_node_from_proto(response.node)
            return {"id": node_id, "deleted": response.deleted}

        try:
            with GrpcChannelContext(self._connection_pool) as channel:
                return _op(channel)
        except grpc.RpcError as e:
            logger.error(f"gRPC delete_node RPC error: {e.code()} - {e.details()}")
            raise ProximaDBError(f"delete_node RPC failed: {e.details()}")
        except Exception as e:
            logger.error(f"gRPC delete_node failed: {e}")
            raise ProximaDBError(f"delete_node failed: {e}")

    # ── Analytic / batch / constraint / streaming RPCs (TD-124) ─────────────

    def batch_create_nodes(
        self,
        nodes: list[dict[str, Any]],
        graph_id: str = "default",
    ) -> dict[str, Any]:
        """Batch-create graph nodes via gRPC.

        Args:
            nodes: List of node dicts, each with ``id``/``node_id``, ``labels``,
                and optional ``properties``.
            graph_id: Graph collection ID (defaults to "default").

        Returns:
            Dict with ``success``, ``created_count`` and the created ``nodes``.
        """
        if not GRPC_AVAILABLE:
            raise ProximaDBError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )
        if v2_graph_pb2_grpc is None or v2_graph_pb2 is None:
            raise ProximaDBError(
                "ProximaGraphService stubs not found. Run: make -C clients/python gen-proto"
            )

        def _op(channel):
            stub = _StubWithDefaultRpcKwargs(
                v2_graph_pb2_grpc.ProximaGraphServiceStub(channel), self._rpc_kwargs()
            )
            proto_nodes = []
            for node in nodes:
                node_properties = {}
                for key, value in (node.get("properties") or {}).items():
                    node_properties[key] = self._convert_to_property_value(value)
                proto_nodes.append(
                    v2_graph_pb2.GraphNode(
                        id=node.get("id") or node.get("node_id", ""),
                        labels=node.get("labels", []),
                        properties=node_properties,
                    )
                )
            request = v2_graph_pb2.BatchCreateGraphNodesRequest(
                graph_id=graph_id, nodes=proto_nodes
            )
            response = stub.BatchCreateNodes(request, timeout=self.timeout)
            return {
                "success": response.success,
                "created_count": response.created_count,
                "nodes": [self._convert_node_from_proto(n) for n in response.nodes],
            }

        try:
            with GrpcChannelContext(self._connection_pool) as channel:
                return _op(channel)
        except grpc.RpcError as e:
            logger.error(
                f"gRPC batch_create_nodes RPC error: {e.code()} - {e.details()}"
            )
            raise ProximaDBError(f"batch_create_nodes RPC failed: {e.details()}")
        except Exception as e:
            logger.error(f"gRPC batch_create_nodes failed: {e}")
            raise ProximaDBError(f"batch_create_nodes failed: {e}")

    def batch_create_edges(
        self,
        edges: list[dict[str, Any]],
        graph_id: str = "default",
    ) -> dict[str, Any]:
        """Batch-create graph edges via gRPC.

        Args:
            edges: List of edge dicts, each with ``id``/``edge_id``,
                ``from_node_id``, ``to_node_id``, ``edge_type``, optional
                ``properties`` and ``weight``.
            graph_id: Graph collection ID (defaults to "default").

        Returns:
            Dict with ``success``, ``created_count`` and the created ``edges``.
        """
        if not GRPC_AVAILABLE:
            raise ProximaDBError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )
        if v2_graph_pb2_grpc is None or v2_graph_pb2 is None:
            raise ProximaDBError(
                "ProximaGraphService stubs not found. Run: make -C clients/python gen-proto"
            )

        def _op(channel):
            stub = _StubWithDefaultRpcKwargs(
                v2_graph_pb2_grpc.ProximaGraphServiceStub(channel), self._rpc_kwargs()
            )
            proto_edges = []
            for edge in edges:
                edge_properties = {}
                for key, value in (edge.get("properties") or {}).items():
                    edge_properties[key] = self._convert_to_property_value(value)
                proto_edge = v2_graph_pb2.GraphEdge(
                    id=edge.get("id") or edge.get("edge_id", ""),
                    from_node_id=edge.get("from_node_id", ""),
                    to_node_id=edge.get("to_node_id", ""),
                    edge_type=edge.get("edge_type", ""),
                    properties=edge_properties,
                )
                if edge.get("weight") is not None:
                    proto_edge.weight = edge["weight"]
                proto_edges.append(proto_edge)
            request = v2_graph_pb2.BatchCreateGraphEdgesRequest(
                graph_id=graph_id, edges=proto_edges
            )
            response = stub.BatchCreateEdges(request, timeout=self.timeout)
            return {
                "success": response.success,
                "created_count": response.created_count,
                "edges": [self._convert_edge_from_proto(e) for e in response.edges],
            }

        try:
            with GrpcChannelContext(self._connection_pool) as channel:
                return _op(channel)
        except grpc.RpcError as e:
            logger.error(
                f"gRPC batch_create_edges RPC error: {e.code()} - {e.details()}"
            )
            raise ProximaDBError(f"batch_create_edges RPC failed: {e.details()}")
        except Exception as e:
            logger.error(f"gRPC batch_create_edges failed: {e}")
            raise ProximaDBError(f"batch_create_edges failed: {e}")

    def get_connected_components(
        self,
        graph_id: str = "default",
    ) -> list[list[str]]:
        """Return weakly-connected components as lists of node IDs via gRPC."""
        if not GRPC_AVAILABLE:
            raise ProximaDBError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )
        if v2_graph_pb2_grpc is None or v2_graph_pb2 is None:
            raise ProximaDBError(
                "ProximaGraphService stubs not found. Run: make -C clients/python gen-proto"
            )

        def _op(channel):
            stub = _StubWithDefaultRpcKwargs(
                v2_graph_pb2_grpc.ProximaGraphServiceStub(channel), self._rpc_kwargs()
            )
            request = v2_graph_pb2.GraphConnectedComponentsRequest(graph_id=graph_id)
            response = stub.GetConnectedComponents(request, timeout=self.timeout)
            return [list(component.node_ids) for component in response.components]

        try:
            with GrpcChannelContext(self._connection_pool) as channel:
                return _op(channel)
        except grpc.RpcError as e:
            logger.error(
                f"gRPC get_connected_components RPC error: {e.code()} - {e.details()}"
            )
            raise ProximaDBError(f"get_connected_components RPC failed: {e.details()}")
        except Exception as e:
            logger.error(f"gRPC get_connected_components failed: {e}")
            raise ProximaDBError(f"get_connected_components failed: {e}")

    def has_cycle(
        self,
        graph_id: str = "default",
    ) -> bool:
        """Return whether the graph contains a directed cycle via gRPC."""
        if not GRPC_AVAILABLE:
            raise ProximaDBError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )
        if v2_graph_pb2_grpc is None or v2_graph_pb2 is None:
            raise ProximaDBError(
                "ProximaGraphService stubs not found. Run: make -C clients/python gen-proto"
            )

        def _op(channel):
            stub = _StubWithDefaultRpcKwargs(
                v2_graph_pb2_grpc.ProximaGraphServiceStub(channel), self._rpc_kwargs()
            )
            request = v2_graph_pb2.GraphHasCycleRequest(graph_id=graph_id)
            response = stub.HasCycle(request, timeout=self.timeout)
            return response.has_cycle

        try:
            with GrpcChannelContext(self._connection_pool) as channel:
                return _op(channel)
        except grpc.RpcError as e:
            logger.error(f"gRPC has_cycle RPC error: {e.code()} - {e.details()}")
            raise ProximaDBError(f"has_cycle RPC failed: {e.details()}")
        except Exception as e:
            logger.error(f"gRPC has_cycle failed: {e}")
            raise ProximaDBError(f"has_cycle failed: {e}")

    def add_unique_constraint(
        self,
        label: str,
        property: str,
        graph_id: str = "default",
    ) -> dict[str, Any]:
        """Add a unique constraint on (label, property) via gRPC.

        Returns a dict with ``success`` and optional ``error_message``.
        """
        return self._unique_constraint("AddUniqueConstraint", label, property, graph_id)

    def remove_unique_constraint(
        self,
        label: str,
        property: str,
        graph_id: str = "default",
    ) -> dict[str, Any]:
        """Remove a unique constraint on (label, property) via gRPC.

        Returns a dict with ``success`` and optional ``error_message``.
        """
        return self._unique_constraint(
            "RemoveUniqueConstraint", label, property, graph_id
        )

    def _unique_constraint(
        self,
        rpc_name: str,
        label: str,
        property: str,
        graph_id: str,
    ) -> dict[str, Any]:
        if not GRPC_AVAILABLE:
            raise ProximaDBError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )
        if v2_graph_pb2_grpc is None or v2_graph_pb2 is None:
            raise ProximaDBError(
                "ProximaGraphService stubs not found. Run: make -C clients/python gen-proto"
            )

        def _op(channel):
            stub = _StubWithDefaultRpcKwargs(
                v2_graph_pb2_grpc.ProximaGraphServiceStub(channel), self._rpc_kwargs()
            )
            request = v2_graph_pb2.GraphUniqueConstraintRequest(
                graph_id=graph_id, label=label, property=property
            )
            response = getattr(stub, rpc_name)(request, timeout=self.timeout)
            result: dict[str, Any] = {"success": response.success}
            if response.HasField("error_message"):
                result["error_message"] = response.error_message
            return result

        try:
            with GrpcChannelContext(self._connection_pool) as channel:
                return _op(channel)
        except grpc.RpcError as e:
            logger.error(f"gRPC {rpc_name} RPC error: {e.code()} - {e.details()}")
            raise ProximaDBError(f"{rpc_name} RPC failed: {e.details()}")
        except Exception as e:
            logger.error(f"gRPC {rpc_name} failed: {e}")
            raise ProximaDBError(f"{rpc_name} failed: {e}")

    def stream_traverse(
        self,
        start_node_id: str,
        max_depth: int = 3,
        edge_types: list[str] | None = None,
        node_labels: list[str] | None = None,
        algorithm: str = "BFS",
        limit: int | None = None,
        graph_id: str = "default",
    ) -> list[dict[str, Any]]:
        """Server-streaming traversal via gRPC; returns the list of chunks.

        Each chunk is a dict with ``nodes``, ``edges``, ``paths`` and ``done``.
        """
        if not GRPC_AVAILABLE:
            raise ProximaDBError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )
        if v2_graph_pb2_grpc is None or v2_graph_pb2 is None:
            raise ProximaDBError(
                "ProximaGraphService stubs not found. Run: make -C clients/python gen-proto"
            )

        def _op(channel):
            stub = _StubWithDefaultRpcKwargs(
                v2_graph_pb2_grpc.ProximaGraphServiceStub(channel), self._rpc_kwargs()
            )

            algorithm_enum = v2_graph_pb2.GRAPH_TRAVERSAL_ALGORITHM_BFS
            if algorithm.upper() == "DFS":
                algorithm_enum = v2_graph_pb2.GRAPH_TRAVERSAL_ALGORITHM_DFS
            elif algorithm.upper() == "PARALLEL_BFS":
                algorithm_enum = v2_graph_pb2.GRAPH_TRAVERSAL_ALGORITHM_PARALLEL_BFS

            request = v2_graph_pb2.TraverseGraphRequest(
                graph_id=graph_id,
                start_node_id=start_node_id,
                max_depth=max_depth,
                edge_types=edge_types or [],
                node_labels=node_labels or [],
                algorithm=algorithm_enum,
            )
            if limit is not None:
                request.limit = limit

            chunks = []
            for chunk in stub.StreamTraverse(request, timeout=self.timeout):
                chunks.append(
                    {
                        "nodes": [
                            self._convert_node_from_proto(n) for n in chunk.nodes
                        ],
                        "edges": [
                            self._convert_edge_from_proto(e) for e in chunk.edges
                        ],
                        "paths": [
                            self._convert_path_from_proto(p) for p in chunk.paths
                        ],
                        "done": chunk.done,
                    }
                )
            return chunks

        try:
            with GrpcChannelContext(self._connection_pool) as channel:
                return _op(channel)
        except grpc.RpcError as e:
            logger.error(f"gRPC stream_traverse RPC error: {e.code()} - {e.details()}")
            raise ProximaDBError(f"stream_traverse RPC failed: {e.details()}")
        except Exception as e:
            logger.error(f"gRPC stream_traverse failed: {e}")
            raise ProximaDBError(f"stream_traverse failed: {e}")

    def execute_query(
        self,
        query: str,
        language: str = "CYPHER",
        graph_id: str = "default",
    ) -> list[dict[str, Any]]:
        """Execute a declarative graph query (supported openCypher subset).

        Args:
            query: e.g. "MATCH (n:Person) WHERE n.age = 30 RETURN n".
            language: "CYPHER" (default) or "NATIVE". "GREMLIN" is not backed.
            graph_id: Graph collection ID (defaults to "default").

        Returns:
            A list of result rows, each a dict of column name -> value.
        """
        if not GRPC_AVAILABLE:
            raise ProximaDBError(
                "gRPC not available. Install with: pip install grpcio grpcio-tools"
            )
        if v2_graph_pb2_grpc is None or v2_graph_pb2 is None:
            raise ProximaDBError(
                "ProximaGraphService stubs not found. Run: make -C clients/python gen-proto"
            )

        def _op(channel):
            stub = _StubWithDefaultRpcKwargs(
                v2_graph_pb2_grpc.ProximaGraphServiceStub(channel), self._rpc_kwargs()
            )
            language_enum = v2_graph_pb2.GRAPH_QUERY_LANGUAGE_CYPHER
            if language.upper() == "NATIVE":
                language_enum = v2_graph_pb2.GRAPH_QUERY_LANGUAGE_NATIVE
            elif language.upper() == "GREMLIN":
                language_enum = v2_graph_pb2.GRAPH_QUERY_LANGUAGE_GREMLIN

            request = v2_graph_pb2.ExecuteGraphQueryRequest(
                graph_id=graph_id, language=language_enum, query=query
            )
            response = stub.ExecuteQuery(request, timeout=self.timeout)
            rows = []
            for row in response.rows:
                columns = {
                    key: self._convert_from_property_value(value)
                    for key, value in row.columns.items()
                }
                rows.append(columns)
            return rows

        try:
            with GrpcChannelContext(self._connection_pool) as channel:
                return _op(channel)
        except grpc.RpcError as e:
            logger.error(f"gRPC execute_query RPC error: {e.code()} - {e.details()}")
            raise ProximaDBError(f"execute_query RPC failed: {e.details()}")
        except Exception as e:
            logger.error(f"gRPC execute_query failed: {e}")
            raise ProximaDBError(f"execute_query failed: {e}")


# Alias for consistency
ProximaDBClient = ProximaDBSyncGrpcClient
