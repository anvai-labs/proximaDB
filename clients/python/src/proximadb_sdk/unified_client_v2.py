"""
ProximaDB Unified Python Client (Refactored v2)

Unified client interface using Protocol Adapter Pattern.
Supports REST, gRPC, and embedded protocols with consistent API.

This refactored version eliminates protocol-specific branching by
delegating all operations to protocol adapters.

Copyright 2025 ProximaDB Contributors
Licensed under the Apache License, Version 2.0
"""

import logging
from typing import Any

import numpy as np

from ._document_ingest import (
    insert_records_locally,
    validate_ingest_records,
)
from ._graph_validation import (
    validate_create_edge_args,
    validate_create_node_args,
    validate_traverse_graph_args,
)
from .adapters import BaseProtocolAdapter, create_adapter
from .config import ClientConfig, Protocol, load_config
from .exceptions import ProximaDBError, ProximaDBNotSupportedError
from .models import (
    BatchResult,
    Collection,
    CollectionConfig,
    FilterDict,
    HealthStatus,
    MetadataDict,
    SearchResult,
    VectorOperationResponse,
    VectorRecord,
    normalize_collection_config_kwargs,
)
from .models_v2 import ProximaRecord

logger = logging.getLogger(__name__)


class ProximaDBClient:
    """
    Unified ProximaDB Python Client (v2 - Adapter Pattern)

    Supports REST, gRPC, and embedded protocols with automatic selection.
    Uses Protocol Adapter Pattern for clean, maintainable code.
    """

    def __init__(
        self,
        url: str | None = None,
        api_key: str | None = None,
        protocol: Protocol | str = Protocol.AUTO,
        config: ClientConfig | None = None,
        data_dir: str | None = None,
        pool_size: int = 10,
        timeout: float = 60.0,
        **kwargs,
    ):
        """
        Initialize ProximaDB client.

        Args:
            url: ProximaDB server URL (for REST/gRPC)
            api_key: API key for authentication
            protocol: Communication protocol (auto, grpc, rest, embedded)
            config: Client configuration object
            data_dir: Data directory for embedded mode
            pool_size: Connection pool size
            timeout: Request timeout in seconds
            **kwargs: Additional configuration parameters
        """
        if config is None:
            config = load_config(url=url, api_key=api_key, **kwargs)

        self.config = config
        self._protocol = Protocol(protocol) if isinstance(protocol, str) else protocol
        self._adapter: BaseProtocolAdapter | None = None
        self._document_repository = None
        self._url = url
        self._data_dir = data_dir
        self._timeout = timeout
        self._pool_size = pool_size
        self._kwargs = kwargs

        self._setup_adapter()

    def _setup_adapter(self):
        """Setup the appropriate protocol adapter."""
        protocol_name = (
            self._protocol.value.lower()
            if hasattr(self._protocol, "value")
            else str(self._protocol).lower()
        )

        if protocol_name == "embedded":
            self._adapter = create_adapter(
                "embedded",
                data_dir=self._data_dir or "/tmp/proximadb/data",
                **self._kwargs,
            )
            logger.info("Using embedded adapter (in-process)")

        elif protocol_name == "grpc":
            # Extract gRPC address
            grpc_url = self._get_grpc_url()
            self._adapter = create_adapter(
                "grpc",
                server_address=grpc_url,
                config=self.config,
                timeout=self._timeout,
                pool_size=self._pool_size,
                **self._kwargs,
            )
            logger.info(f"Using gRPC adapter: {grpc_url}")

        elif protocol_name == "rest":
            self._adapter = create_adapter(
                "rest",
                url=self._url or "http://localhost:5678",
                config=self.config,
                timeout=self._timeout,
                **self._kwargs,
            )
            logger.info(f"Using REST adapter: {self._url}")

        elif protocol_name == "auto":
            # Try gRPC first, fallback to REST
            try:
                grpc_url = self._get_grpc_url()
                self._adapter = create_adapter(
                    "grpc",
                    server_address=grpc_url,
                    config=self.config,
                    timeout=self._timeout,
                    pool_size=self._pool_size,
                    **self._kwargs,
                )
                self._protocol = Protocol.GRPC
                logger.info(f"Auto-selected gRPC adapter: {grpc_url}")
            except ImportError:
                self._adapter = create_adapter(
                    "rest",
                    url=self._url or "http://localhost:5678",
                    config=self.config,
                    timeout=self._timeout,
                    **self._kwargs,
                )
                self._protocol = Protocol.REST
                logger.info(f"Auto-selected REST adapter: {self._url}")
        else:
            raise ValueError(f"Unknown protocol: {protocol_name}")

    def _get_grpc_url(self) -> str:
        """Get gRPC server address from config or URL."""
        if hasattr(self.config, "get_protocol_url"):
            return self.config.get_protocol_url(Protocol.GRPC)

        # Extract from URL
        if self._url:
            # Convert http://host:port to host:grpc_port
            from urllib.parse import urlparse

            parsed = urlparse(self._url)
            host = parsed.hostname or "localhost"
            # gRPC typically on port + 1
            grpc_port = (parsed.port or 5678) + 1
            return f"{host}:{grpc_port}"

        return "localhost:5679"

    @property
    def active_protocol(self) -> Protocol:
        """Get the currently active protocol."""
        return self._protocol

    @property
    def adapter(self) -> BaseProtocolAdapter:
        """Get the underlying protocol adapter."""
        return self._adapter

    # ==========================================================================
    # Health & Server Operations
    # ==========================================================================

    def health(self) -> HealthStatus:
        """Check server health status."""
        return self._adapter.health()

    # ==========================================================================
    # Collection Operations
    # ==========================================================================

    def create_collection(
        self, name: str, config: CollectionConfig | None = None, **kwargs
    ) -> Collection:
        """Create a new vector collection."""
        # TD-SDKENG-1: never silently drop an unknown keyword; only the
        # transport-consumed pass-through travels past the config.
        config_kwargs, passthrough = normalize_collection_config_kwargs(kwargs)
        if config is None:
            config = CollectionConfig(name=name, **config_kwargs)
        elif config_kwargs:
            # Round-2 review: re-validate (model_copy skips validation and
            # does not translate field aliases like quantization=).
            merged = config.model_dump(exclude={"name"})
            merged.update(config_kwargs)
            config = CollectionConfig(name=name, **merged)
        return self._adapter.create_collection(name, config, **passthrough)

    def get_collection(self, collection_id: str) -> Collection | None:
        """Get collection metadata by ID or name."""
        return self._adapter.get_collection(collection_id)

    def list_collections(self) -> list[Collection]:
        """List all collections."""
        return self._adapter.list_collections()

    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection."""
        return self._adapter.delete_collection(collection_id)

    # ==========================================================================
    # Record Operations
    # ==========================================================================

    @staticmethod
    def _batch_to_vector_response(
        result: BatchResult, operation: str
    ) -> VectorOperationResponse:
        return VectorOperationResponse(
            success=result.success,
            operation=operation,
            metrics=result.metrics,
            error_message="; ".join(result.errors) if result.errors else None,
        )

    def insert_records(
        self,
        collection_id: str,
        records: list[ProximaRecord | dict[str, Any]],
        **kwargs,
    ) -> BatchResult:
        """Insert ProximaRecord-shaped payloads through the active adapter."""
        if records is None or len(records) == 0:
            raise ValueError("'records' must be provided")
        return self._adapter.insert_records(collection_id, records, **kwargs)

    def upsert_records(
        self,
        collection_id: str,
        records: list[ProximaRecord | dict[str, Any]],
        **kwargs,
    ) -> BatchResult:
        """Upsert ProximaRecord-shaped payloads through the active adapter."""
        if records is None or len(records) == 0:
            raise ValueError("'records' must be provided")
        return self._adapter.upsert_records(collection_id, records, **kwargs)

    # ==========================================================================
    # Vector Compatibility Aliases
    # ==========================================================================

    def insert_vectors(
        self,
        collection_id: str,
        vectors: list[list[float]] | list[VectorRecord] | np.ndarray | None = None,
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
        records: list[VectorRecord] | None = None,
        **kwargs,
    ) -> VectorOperationResponse:
        """Compatibility alias for record-native inserts.

        Supports record objects plus legacy vectors/ids/metadata inputs.
        """
        # Handle backward compatibility
        if vectors is not None:
            if hasattr(vectors, "tolist"):
                vectors = vectors.tolist()

            # Check if vectors is already a list of VectorRecord
            if (
                hasattr(vectors, "__len__")
                and len(vectors) > 0
                and hasattr(vectors[0], "vector")
                and hasattr(vectors[0], "id")
            ):
                records = vectors
            else:
                # Convert old API to VectorRecord objects
                records = []
                for i, vector in enumerate(vectors):
                    vec_list = (
                        vector
                        if isinstance(vector, list)
                        else (
                            vector.tolist()
                            if hasattr(vector, "tolist")
                            else list(vector)
                        )
                    )
                    record = VectorRecord(
                        id=ids[i] if ids and i < len(ids) else None,
                        vector=vec_list,
                        metadata=metadata[i] if metadata and i < len(metadata) else {},
                    )
                    records.append(record)

        if records is None or len(records) == 0:
            raise ValueError("Either 'records' or 'vectors' must be provided")

        return self._batch_to_vector_response(
            self.insert_records(collection_id, records, **kwargs), "INSERT"
        )

    def upsert_vectors(
        self, collection_id: str, records: list[VectorRecord]
    ) -> VectorOperationResponse:
        """Compatibility alias for record-native upserts."""
        return self._batch_to_vector_response(
            self.upsert_records(collection_id, records), "UPSERT"
        )

    def get_vectors(
        self,
        collection_id: str,
        vector_ids: list[str],
        include_vectors: bool = True,
        **kwargs,
    ) -> list[VectorRecord]:
        """Get vectors by IDs."""
        return self._adapter.get_vectors(
            collection_id, vector_ids, include_vectors, **kwargs
        )

    def get_vector(
        self,
        collection_id: str,
        vector_id: str,
        include_vector: bool = True,
        include_metadata: bool = True,
    ) -> VectorRecord | None:
        """Get a single vector by ID."""
        results = self.get_vectors(collection_id, [vector_id], include_vector)
        return results[0] if results else None

    def delete_vectors(
        self, collection_id: str, vector_ids: list[str]
    ) -> VectorOperationResponse:
        """Delete vectors by IDs."""
        return self._adapter.delete_vectors(collection_id, vector_ids)

    def delete_vector(
        self, collection_id: str, vector_id: str
    ) -> VectorOperationResponse:
        """Delete a single vector."""
        return self.delete_vectors(collection_id, [vector_id])

    def update_vector_metadata(
        self, collection_id: str, vector_id: str, metadata: MetadataDict, **kwargs
    ) -> VectorOperationResponse:
        """Update metadata for a specific vector."""
        return self._adapter.update_vector_metadata(
            collection_id, vector_id, metadata, **kwargs
        )

    def insert_vector(
        self,
        collection_id: str,
        vector_id: str,
        vector: list[float] | np.ndarray,
        metadata: dict[str, Any] | None = None,
        upsert: bool = False,
        **kwargs,
    ) -> VectorOperationResponse:
        """Insert a single vector."""
        record = VectorRecord(
            id=vector_id,
            vector=vector if isinstance(vector, list) else vector.tolist(),
            metadata=metadata or {},
        )

        if upsert:
            return self.upsert_vectors(collection_id, [record])
        else:
            return self.insert_vectors(collection_id, records=[record])

    # ==========================================================================
    # Search Operations
    # ==========================================================================

    def search(
        self,
        collection_id: str,
        vector: list[float] | np.ndarray,
        top_k: int = 10,
        metadata_filter: FilterDict | None = None,
        include_metadata: bool = True,
        include_vectors: bool = False,
        **kwargs,
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        if top_k <= 0:
            raise ProximaDBError(f"top_k must be positive, got {top_k}")

        query_vector = vector if isinstance(vector, list) else vector.tolist()

        return self._adapter.search(
            collection_id=collection_id,
            query_vector=query_vector,
            top_k=top_k,
            filter=metadata_filter,
            include_vectors=include_vectors,
            include_metadata=include_metadata,
            **kwargs,
        )

    def search_single(
        self,
        collection_id: str,
        vector: list[float] | np.ndarray,
        top_k: int = 10,
        metadata_filter: FilterDict | None = None,
        **kwargs,
    ) -> list[SearchResult]:
        """Alias for search() for backward compatibility."""
        return self.search(
            collection_id=collection_id,
            vector=vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
            **kwargs,
        )

    def search_batch(
        self,
        collection_id: str,
        vectors: list[list[float]] | np.ndarray,
        top_k: int = 10,
        metadata_filter: FilterDict | None = None,
        **kwargs,
    ) -> list[list[SearchResult]]:
        """Batch search for similar vectors."""
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()

        return self._adapter.batch_search(
            collection_id=collection_id,
            query_vectors=vectors,
            top_k=top_k,
            filter=metadata_filter,
            **kwargs,
        )

    # ==========================================================================
    # Legacy Compatibility Methods
    # ==========================================================================

    def insert(
        self,
        collection_id: str,
        vectors: list[list[float]] | np.ndarray,
        ids: list[str] | None = None,
        metadata: list[dict[str, Any]] | None = None,
    ) -> VectorOperationResponse:
        """Legacy insert method."""
        return self.insert_vectors(collection_id, vectors, ids, metadata)

    def upsert(
        self,
        collection_id: str,
        vectors: list[list[float]] | np.ndarray,
        ids: list[str],
        metadata: list[dict[str, Any]] | None = None,
    ) -> VectorOperationResponse:
        """Legacy upsert method."""
        if isinstance(vectors, np.ndarray):
            vectors = vectors.tolist()

        records = []
        for i, (vector, vector_id) in enumerate(zip(vectors, ids)):
            record = VectorRecord(
                vector=vector,
                id=vector_id,
                metadata=metadata[i] if metadata and i < len(metadata) else {},
            )
            records.append(record)

        return self.upsert_vectors(collection_id, records)

    def delete(self, collection_id: str, ids: list[str]) -> VectorOperationResponse:
        """Legacy delete method."""
        return self.delete_vectors(collection_id, ids)

    # ==========================================================================
    # Utility Methods
    # ==========================================================================

    def get_collection_stats(self, collection_id: str) -> dict[str, Any]:
        """Get collection statistics."""
        collection = self.get_collection(collection_id)
        if collection:
            return {
                "id": collection.id,
                "name": collection.config.name if collection.config else collection_id,
                "dimension": collection.config.dimension if collection.config else 0,
                "created_at": collection.created_at,
                "updated_at": collection.updated_at,
            }
        return {}

    def get_performance_info(self) -> dict[str, Any]:
        """Get performance information about the active protocol."""
        if self._protocol == Protocol.GRPC:
            return {
                "protocol": "gRPC",
                "advantages": [
                    "40% smaller payloads (binary protobuf)",
                    "HTTP/2 multiplexing",
                    "Better type safety",
                ],
            }
        elif self._protocol == Protocol.REST:
            return {
                "protocol": "REST",
                "advantages": [
                    "Universal compatibility",
                    "Easy debugging",
                    "Human-readable JSON",
                ],
            }
        else:
            return {
                "protocol": "Embedded",
                "advantages": [
                    "Zero network latency",
                    "Direct memory access",
                    "Lowest overhead",
                ],
            }

    # ==========================================================================
    # AQL/UQL Query Operations
    # ==========================================================================

    def execute_query(
        self,
        query: str,
        *,
        language: str = "uql",
        parameters: list[Any] | None = None,
        collection: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Execute AQL/UQL through the active adapter.

        REST adapters use the OpenAPI v2 `/api/v2/query` contract. Other
        adapters may add native support later; until then they fail explicitly.
        """
        if not hasattr(self._adapter, "execute_query"):
            raise ProximaDBError(
                f"{self._adapter.protocol_name} adapter does not support execute_query"
            )
        return self._adapter.execute_query(
            query,
            language=language,
            parameters=parameters,
            collection=collection,
            limit=limit,
        )

    def execute_uql(
        self,
        query: str,
        *,
        parameters: list[Any] | None = None,
        collection: str | None = None,
        limit: int | None = None,
    ) -> dict[str, Any]:
        """Execute a UQL query through the OpenAPI v2 REST query surface."""
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
        """Execute an AQL query through the OpenAPI v2 REST query surface."""
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

    def explain_query(
        self,
        query: str,
        *,
        language: str = "uql",
        collection: str | None = None,
    ) -> dict[str, Any]:
        """Explain an AQL/UQL query through the active adapter."""
        if not hasattr(self._adapter, "explain_query"):
            raise ProximaDBError(
                f"{self._adapter.protocol_name} adapter does not support explain_query"
            )
        return self._adapter.explain_query(
            query, language=language, collection=collection
        )

    # ==========================================================================
    # Graph Operations (TD-SDK-3 batch 1 — the graph-CRUD family, ported from
    # the v1 unified client onto the adapter protocol)
    #
    # Thin delegations: the facade validates protocol-independent inputs and
    # the adapter owns everything wire-shaped (``graph_id=None`` → the
    # server's "default" graph is normalized adapter-side). Adapter
    # NotImplementedError surfaces as ProximaDBNotSupportedError (a typed
    # capability signal — a ProximaDBError subclass — so callers see one
    # error vocabulary while compatibility wrappers can still detect the
    # fallback condition).
    # ==========================================================================

    def _adapter_call(self, method: str, *args, **kwargs) -> Any:
        """Delegate an op to the adapter, normalizing the error surface.

        Adapter ``NotImplementedError`` (transport lacks the operation)
        becomes :class:`ProximaDBNotSupportedError` — a typed capability
        signal, still a ``ProximaDBError`` for generic handlers, but
        recognizable by compatibility wrappers (``graph.py`` batch helpers)
        so they can fall back to per-item operations exactly as they did on
        the v1-era ``AttributeError`` instead of failing the whole batch.
        """
        try:
            return getattr(self._adapter, method)(*args, **kwargs)
        except NotImplementedError as e:
            raise ProximaDBNotSupportedError(
                f"{self._adapter.protocol_name} adapter does not support "
                f"{method}: {e}"
            ) from e

    def create_node(
        self,
        node_id: str,
        labels: list[str],
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        graph_id: str | None = None,
    ) -> dict[str, Any]:
        """Create a graph node.

        Example:
            >>> client.create_node(
            ...     node_id="person_123",
            ...     labels=["Person"],
            ...     properties={"name": "Alice", "age": 30},
            ... )
        """
        validate_create_node_args(node_id, labels, properties, embedding)

        return self._adapter_call(
            "create_node",
            node_id=node_id,
            labels=labels,
            properties=properties,
            embedding=embedding,
            graph_id=graph_id,
        )

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
        """Create a graph edge between two nodes.

        Example:
            >>> client.create_edge(
            ...     edge_id="edge_123",
            ...     from_node_id="person_123",
            ...     to_node_id="person_456",
            ...     edge_type="KNOWS",
            ...     properties={"since": 2020},
            ...     weight=1.0,
            ... )
        """
        validate_create_edge_args(
            edge_id, from_node_id, to_node_id, edge_type, properties, weight
        )

        return self._adapter_call(
            "create_edge",
            edge_id=edge_id,
            from_node_id=from_node_id,
            to_node_id=to_node_id,
            edge_type=edge_type,
            properties=properties,
            weight=weight,
            graph_id=graph_id,
        )

    def batch_create_nodes(
        self, nodes: list[dict[str, Any]], graph_id: str | None = None
    ) -> dict[str, Any]:
        """Create many nodes in one request (admissions per server engine)."""
        return self._adapter_call("batch_create_nodes", nodes=nodes, graph_id=graph_id)

    def batch_create_edges(
        self, edges: list[dict[str, Any]], graph_id: str | None = None
    ) -> dict[str, Any]:
        """Create many edges in one request with per-edge admission."""
        return self._adapter_call("batch_create_edges", edges=edges, graph_id=graph_id)

    def get_node(
        self, node_id: str, graph_id: str | None = None
    ) -> dict[str, Any] | None:
        """Get a graph node by ID (``None`` when absent)."""
        return self._adapter_call("get_node", node_id=node_id, graph_id=graph_id)

    def get_outgoing_edges(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        graph_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get a node's outgoing edges."""
        return self._adapter_call(
            "get_outgoing_edges",
            node_id=node_id,
            edge_types=edge_types,
            graph_id=graph_id,
        )

    def get_incoming_edges(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        graph_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Get a node's incoming edges."""
        return self._adapter_call(
            "get_incoming_edges",
            node_id=node_id,
            edge_types=edge_types,
            graph_id=graph_id,
        )

    def delete_node(self, node_id: str, graph_id: str | None = None) -> bool:
        """Delete a graph node by ID."""
        return bool(
            self._adapter_call("delete_node", node_id=node_id, graph_id=graph_id)
        )

    def query_nodes(
        self,
        labels: list[str] | None = None,
        properties: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        graph_id: str | None = None,
    ) -> dict[str, Any]:
        """Query nodes by labels and properties."""
        return self._adapter_call(
            "query_nodes",
            labels=labels,
            properties=properties,
            limit=limit,
            offset=offset,
            graph_id=graph_id,
        )

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
        """Traverse the graph from a starting node.

        Returns a dict with ``nodes``, ``edges``, ``paths`` and ``stats``.

        Example:
            >>> result = client.traverse_graph(
            ...     start_node_id="person_123",
            ...     max_depth=2,
            ...     edge_types=["KNOWS"],
            ...     algorithm="BFS",
            ...     limit=50,
            ... )
        """
        validate_traverse_graph_args(
            start_node_id, max_depth, edge_types, node_labels, algorithm, limit
        )

        return self._adapter_call(
            "traverse_graph",
            start_node_id=start_node_id,
            max_depth=max_depth,
            edge_types=edge_types,
            node_labels=node_labels,
            algorithm=algorithm,
            limit=limit,
            graph_id=graph_id,
        )

    def create_graph(
        self,
        graph_id: str,
        name: str | None = None,
        description: str | None = None,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a graph collection."""
        return self._adapter_call(
            "create_graph",
            graph_id=graph_id,
            name=name,
            description=description,
            schema=schema,
        )

    def delete_graph(self, graph_id: str) -> dict[str, Any]:
        """Delete a graph collection."""
        return self._adapter_call("delete_graph", graph_id=graph_id)

    def get_graph(self, graph_id: str) -> dict[str, Any]:
        """Get graph-collection metadata."""
        return self._adapter_call("get_graph", graph_id=graph_id)

    def list_graphs(self) -> dict[str, Any]:
        """List all graph collections."""
        return self._adapter_call("list_graphs")

    def get_graph_stats(self, graph_id: str) -> dict[str, Any]:
        """Get statistics (node/edge counts, …) for a graph collection."""
        return self._adapter_call("get_graph_stats", graph_id=graph_id)

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
        """Compute shortest path(s) with optional prefetch overrides.

        ``graph_id`` names the graph (TD-SDKDOC-1: dropping it silently ran
        every call on ``default``).
        """
        return self._adapter_call(
            "graph_shortest_path",
            start_node_id,
            target_node_id,
            max_depth,
            edge_types,
            algorithm,
            k,
            enable_prefetch,
            prefetch_budget,
            timeout=timeout,
            graph_id=graph_id,
        )

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
        """Tuned traversal with prefetch/timeout overrides (analytics surface).

        Unlike the v1 wrapper, ``graph_id`` is honored — the REST transport
        has accepted it since TD-SDKDOC-1; v1 dropped it, running every tuned
        traversal on ``default``.
        """
        return self._adapter_call(
            "graph_traverse",
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
            graph_id=graph_id,
        )

    # ==========================================================================
    # Document Collections Operations (TD-SDK-3 batch 2 — the document family,
    # ported from the v1 unified client onto the adapter protocol)
    #
    # Thin delegations: the facade validates protocol-independent inputs
    # (ingest records) and the adapter owns everything wire-shaped. Adapter
    # NotImplementedError surfaces as ProximaDBNotSupportedError. Unlike v1's
    # wrappers there is NO silent in-memory fallback on adapter errors — a
    # server-backed client surfaces failures; local mode is the embedded
    # adapter, or the caller-opted ``allow_local_fallback`` on ingest.
    #
    # REST error semantics (spec-driven ops, one vocabulary): transport
    # failures raise NetworkError, server rejections raise ServerError,
    # 404 maps to the not-found contract (get_document → None, deletes →
    # False). The previous rest_adapter code never issued a request at all
    # (it called a nonexistent protocol-client attribute and swallowed the
    # AttributeError into None/False/[]) — fixed in this batch.
    # ==========================================================================

    def _get_document_repository(self):
        """Lazily build the in-memory repository (caller-opted local mode)."""
        if self._document_repository is None:
            from .document import DocumentRepository

            self._document_repository = DocumentRepository(client=self)
        return self._document_repository

    def create_document_collection(
        self, name: str, config: dict[str, Any] | None = None, **kwargs
    ) -> dict[str, Any]:
        """Create a document collection."""
        return self._adapter_call("create_document_collection", name, config, **kwargs)

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

        Records are validated protocol-independently (each is a ``dict`` with
        at least ``id``, optionally ``text``/``metadata``/``vector``;
        ``IngestDocument`` records are normalized to plain dicts at the
        facade — adapters receive dicts), then the wire is the adapter's:
        REST runs the shared ``_document_ingest`` generated-op protocol (the
        same one v1's canonical path uses — per-call
        ``X-Embed-Source``/``X-Tenant-ID``/``X-Ingest-Mode`` headers, no
        accumulation), embedded uses the engine's insert path.

        Args:
            collection_id: Target collection id.
            records: Ergonomic record dicts (or ``IngestDocument``s).
            tenant_id: Optional tenant scope -> ``X-Tenant-ID``.
            ingest_mode: Optional billing/metering mode -> ``X-Ingest-Mode``.
            embed_source: Embedding source -> ``X-Embed-Source`` (default
                ``"native"``). Pass ``""`` or ``None`` to omit the header.
            allow_local_fallback: If ``True`` and the adapter lacks the
                ingest capability (typed ``ProximaDBNotSupportedError``, e.g.
                the gRPC transport), store the records in the in-memory
                ``DocumentRepository`` instead of raising. Default ``False`` —
                a server-backed client must surface failures. Transport and
                server errors (``NetworkError``/``ServerError``) always
                propagate.

        Returns:
            The parsed ``IngestDocumentsResponse`` as a dict; ``{"mode":
            "local", ...}`` on the opted-in fallback path.

        Raises:
            ProximaDBError: A record is malformed, or the adapter lacks the
                capability and ``allow_local_fallback`` is False.
            ProximaDBNotSupportedError: The adapter transport has no ingest
                path (without fallback).
            NetworkError: Transport-level failure talking to the server.
            ServerError: The server returned a documented error envelope or
                an unexpected status.
        """
        validate_ingest_records(records)
        # Facade contract: adapters receive plain dicts. ``IngestDocument``
        # records (accepted for signature parity with v1) are normalized
        # here — the embedded adapter (and any future one) must not have to
        # know the generated model.
        records = [rec if isinstance(rec, dict) else rec.to_dict() for rec in records]
        try:
            return self._adapter_call(
                "ingest_documents",
                collection_id,
                records,
                tenant_id=tenant_id,
                ingest_mode=ingest_mode,
                embed_source=embed_source,
                **kwargs,
            )
        except ProximaDBNotSupportedError:
            if not allow_local_fallback:
                raise
            return insert_records_locally(
                self._get_document_repository(), collection_id, records
            )

    def get_document(
        self,
        collection_name: str,
        doc_id: str,
        projection: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any] | None:
        """Get a document by ID."""
        return self._adapter_call(
            "get_document", collection_name, doc_id, projection, **kwargs
        )

    def query_documents(
        self,
        collection_name: str,
        filter: dict[str, Any] | None = None,
        projection: list[str] | None = None,
        limit: int = 100,
        **kwargs,
    ) -> dict[str, Any]:
        """Query documents."""
        return self._adapter_call(
            "query_documents",
            collection_name,
            filter=filter,
            projection=projection,
            limit=limit,
            **kwargs,
        )

    def update_document(
        self,
        collection_name: str,
        doc_id: str,
        updates: list[dict[str, Any]],
        **kwargs,
    ) -> dict[str, Any]:
        """Update a document."""
        return self._adapter_call(
            "update_document", collection_name, doc_id, updates, **kwargs
        )

    def delete_document(self, collection_name: str, doc_id: str, **kwargs) -> bool:
        """Delete a document."""
        return bool(
            self._adapter_call("delete_document", collection_name, doc_id, **kwargs)
        )

    def list_document_collections(self, **kwargs) -> list[dict[str, Any]]:
        """List document collections."""
        return self._adapter_call("list_document_collections", **kwargs)

    def delete_document_collection(self, collection_name: str, **kwargs) -> bool:
        """Delete a document collection."""
        return bool(
            self._adapter_call("delete_document_collection", collection_name, **kwargs)
        )

    # ==========================================================================
    # Lifecycle Methods
    # ==========================================================================

    def close(self):
        """Close the client and cleanup resources."""
        if self._adapter:
            self._adapter.close()
            self._adapter = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Destructor - cleanup resources."""
        try:
            self.close()
        except Exception:
            pass


# ==========================================================================
# Convenience Functions
# ==========================================================================


def connect(
    url: str | None = None,
    api_key: str | None = None,
    protocol: Protocol | str = Protocol.AUTO,
    **kwargs,
) -> ProximaDBClient:
    """Create a ProximaDB client with simplified parameters."""
    return ProximaDBClient(url=url, api_key=api_key, protocol=protocol, **kwargs)


def connect_grpc(
    url: str | None = None, api_key: str | None = None, **kwargs
) -> ProximaDBClient:
    """Create a ProximaDB client using gRPC protocol."""
    return ProximaDBClient(url=url, api_key=api_key, protocol=Protocol.GRPC, **kwargs)


def connect_rest(
    url: str | None = None, api_key: str | None = None, **kwargs
) -> ProximaDBClient:
    """Create a ProximaDB client using REST protocol."""
    return ProximaDBClient(url=url, api_key=api_key, protocol=Protocol.REST, **kwargs)


def connect_embedded(
    data_dir: str = "/tmp/proximadb/data", **kwargs
) -> ProximaDBClient:
    """Create a ProximaDB client using embedded mode."""
    return ProximaDBClient(protocol="embedded", data_dir=data_dir, **kwargs)
