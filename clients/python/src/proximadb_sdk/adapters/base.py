"""
ProximaDB Adapter Base Class

Abstract base class defining the interface for transport and embedded adapters.
Enables consistent API regardless of whether calls use REST, gRPC, or direct
in-process embedded bindings.

Copyright 2025 ProximaDB Contributors
Licensed under the Apache License, Version 2.0
"""

from abc import ABC, abstractmethod
from typing import Any

from ..models import (
    BatchResult,
    Collection,
    CollectionConfig,
    FilterDict,
    HealthStatus,
    MetadataDict,
    SearchResult,
    VectorArray,
    VectorOperationResponse,
    VectorRecord,
)
from ..models_v2 import ProximaRecord


class BaseProtocolAdapter(ABC):
    """Abstract base class for SDK adapters.

    Adapters encapsulate transport-specific or embedded binding logic, enabling
    the unified client to delegate operations without conditional branches.

    All methods return Pydantic models, regardless of the underlying
    native format (JSON for REST, protobuf for gRPC, PyO3 objects for embedded).
    """

    @property
    @abstractmethod
    def protocol_name(self) -> str:
        """Return the protocol name (e.g., 'rest', 'grpc', 'embedded')."""
        pass

    @property
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if the adapter is connected and operational."""
        pass

    # ==========================================================================
    # Health & Server Operations
    # ==========================================================================

    @abstractmethod
    def health(self) -> HealthStatus:
        """Check adapter health status."""
        pass

    # ==========================================================================
    # Collection Operations
    # ==========================================================================

    @abstractmethod
    def create_collection(
        self, name: str, config: CollectionConfig | None = None, **kwargs
    ) -> Collection:
        """Create a new vector collection."""
        pass

    @abstractmethod
    def get_collection(self, collection_id: str) -> Collection | None:
        """Get collection metadata by ID or name."""
        pass

    @abstractmethod
    def list_collections(self) -> list[Collection]:
        """List all collections."""
        pass

    @abstractmethod
    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection."""
        pass

    # ==========================================================================
    # Record Operations
    # ==========================================================================

    @abstractmethod
    def insert_records(
        self,
        collection_id: str,
        records: list[ProximaRecord] | list[dict[str, Any]],
        **kwargs,
    ) -> BatchResult:
        """Insert ProximaRecord-shaped payloads into a collection."""
        pass

    @abstractmethod
    def upsert_records(
        self,
        collection_id: str,
        records: list[ProximaRecord] | list[dict[str, Any]],
        **kwargs,
    ) -> BatchResult:
        """Upsert ProximaRecord-shaped payloads into a collection."""
        pass

    # ==========================================================================
    # Vector Compatibility Aliases
    # ==========================================================================

    @abstractmethod
    def insert_vectors(
        self,
        collection_id: str,
        vectors: list[VectorRecord] | list[dict[str, Any]],
        **kwargs,
    ) -> VectorOperationResponse:
        """Compatibility alias for record-native inserts."""
        pass

    @abstractmethod
    def upsert_vectors(
        self,
        collection_id: str,
        vectors: list[VectorRecord] | list[dict[str, Any]],
        **kwargs,
    ) -> VectorOperationResponse:
        """Compatibility alias for record-native upserts."""
        pass

    @abstractmethod
    def get_vectors(
        self,
        collection_id: str,
        vector_ids: list[str],
        include_vectors: bool = True,
        **kwargs,
    ) -> list[VectorRecord]:
        """Get vectors by IDs."""
        pass

    @abstractmethod
    def delete_vectors(
        self, collection_id: str, vector_ids: list[str], **kwargs
    ) -> VectorOperationResponse:
        """Delete vectors by IDs."""
        pass

    @abstractmethod
    def update_vector_metadata(
        self, collection_id: str, vector_id: str, metadata: MetadataDict, **kwargs
    ) -> VectorOperationResponse:
        """Update metadata for a specific vector."""
        pass

    # ==========================================================================
    # Search Operations
    # ==========================================================================

    @abstractmethod
    def search(
        self,
        collection_id: str,
        query_vector: VectorArray,
        top_k: int = 10,
        filter: FilterDict | None = None,
        include_vectors: bool = False,
        include_metadata: bool = True,
        **kwargs,
    ) -> list[SearchResult]:
        """Search for similar vectors."""
        pass

    @abstractmethod
    def batch_search(
        self,
        collection_id: str,
        query_vectors: list[VectorArray],
        top_k: int = 10,
        filter: FilterDict | None = None,
        include_vectors: bool = False,
        include_metadata: bool = True,
        **kwargs,
    ) -> list[list[SearchResult]]:
        """Batch search for similar vectors."""
        pass

    # ==========================================================================
    # Document Operations
    # ==========================================================================

    def create_document_collection(
        self, name: str, config: dict[str, Any] | None = None, **kwargs
    ) -> dict[str, Any]:
        """Create a document collection."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support document collections"
        )

    def insert_document(
        self,
        collection_name: str,
        document: dict[str, Any],
        id: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Insert a document into a document-collection (legacy document-collections surface).

        Retained for document-collection CRUD consumers (e.g. the agentic store);
        for new semantic-document writes prefer :meth:`ingest_documents` (ADR-041).
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support document inserts"
        )

    def ingest_documents(
        self,
        collection_id: str,
        records: list[dict[str, Any]],
        *,
        tenant_id: str | None = None,
        ingest_mode: str | None = None,
        embed_source: str = "native",
        **kwargs,
    ) -> dict[str, Any]:
        """Ingest documents for native server-side embedding (ADR-041 canonical surface).

        Each record is a ``{id, text, metadata?, vector?}`` dict; the server embeds
        ``text`` natively under ``embed_source="native"`` when no ``vector`` is given.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support document ingest"
        )

    def get_document(
        self,
        collection_name: str,
        doc_id: str,
        projection: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any] | None:
        """Get a document by ID."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support document reads"
        )

    def query_documents(
        self,
        collection_name: str,
        filter: dict[str, Any] | None = None,
        projection: list[str] | None = None,
        limit: int = 100,
        **kwargs,
    ) -> dict[str, Any]:
        """Query documents with filter."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support document queries"
        )

    def update_document(
        self, collection_name: str, doc_id: str, updates: list[dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        """Update a document."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support document updates"
        )

    def delete_document(self, collection_name: str, doc_id: str, **kwargs) -> bool:
        """Delete a document."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support document deletes"
        )

    def list_document_collections(self, **kwargs) -> list[dict[str, Any]]:
        """List all document collections."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support document collection listing"
        )

    def delete_document_collection(self, collection_name: str, **kwargs) -> bool:
        """Delete a document collection."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support document collection deletion"
        )

    # ==========================================================================
    # Hybrid Search Operations
    # ==========================================================================

    # ==========================================================================
    # Search/Scan/Query Extensions (TD-SDK-3 batch 4) — optional namespace,
    # same pattern: REST implements over the spec-driven ops / protocol
    # surfaces, embedded over the engine, gRPC stays on these defaults.
    # ==========================================================================

    def scan_page(
        self,
        collection_id: str,
        *,
        filter: list[dict[str, Any]] | dict[str, Any] | None = None,
        limit: int | None = None,
        cursor: str | None = None,
        include_vectors: bool = False,
        include_text: bool = False,
    ) -> tuple[list, str | None]:
        """Scan one page of records via the vector-free metadata-scan
        endpoint (REST-only surface). Returns ``(records, next_cursor)``."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support metadata scans"
        )

    def search_envelope(
        self,
        collection_id: str,
        vector: list[float],
        top_k: int = 10,
        include_vectors: bool = False,
        include_metadata: bool = True,
        **kwargs,
    ) -> dict[str, Any]:
        """Run the paged REST OpenAPI v2 search envelope."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support the paged search envelope"
        )

    def execute_sql(
        self,
        query: str,
        parameters: list[Any] | None = None,
        collection: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Execute a SQL query (the spec-driven sql op / engine SQL)."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support SQL execution"
        )

    def execute_unified_query(
        self,
        query: str,
        query_vector: list[float] | None = None,
        fusion_strategy: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Execute a federated multi-model (UQL) query — embedded native."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support unified queries"
        )

    def hybrid_search(
        self,
        collection: str,
        text_query: str,
        query_vector: list[float],
        fusion_strategy: str = "rrf",
        top_k: int = 10,
        **kwargs,
    ) -> dict[str, Any]:
        """Execute hybrid search combining BM25 and vector similarity."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support hybrid search"
        )

    # ==========================================================================
    # Time-Series Operations
    # ==========================================================================

    def create_timeseries_collection(
        self, name: str, config: dict[str, Any] | None = None, **kwargs
    ) -> dict[str, Any]:
        """Create a time-series collection."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support time-series collections"
        )

    def ingest_timeseries(
        self, collection_name: str, points: list[dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        """Ingest time-series data points."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support time-series ingest"
        )

    def query_timeseries(
        self,
        collection_name: str,
        start_time: str,
        end_time: str,
        aggregation: str = "avg",
        bucket_ms: int | None = None,
        tag_filters: dict[str, str] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Query time-series data with optional aggregation."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support time-series queries"
        )

    def list_timeseries_collections(self, **kwargs) -> list[dict[str, Any]]:
        """List all time-series collections."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support time-series collection listing"
        )

    def delete_timeseries_collection(self, collection_name: str, **kwargs) -> bool:
        """Delete a time-series collection."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support time-series collection deletion"
        )

    # ==========================================================================
    # Observability Operations (TD-SDK-3 batch 3 — logs, metrics, traces)
    #
    # Optional namespace, same pattern as documents/timeseries: adapters
    # implement what their transport supports (embedded implements the whole
    # family; REST covers namespace/logs/metrics where the spec defines
    # endpoints) and the rest fall through to these NotImplementedError
    # defaults — the honest capability signal, surfacing as the typed
    # ProximaDBNotSupportedError on the v2 facade. Canonical signatures
    # mirror the v1 unified client's public observability surface.
    # ==========================================================================

    def create_observability_namespace(
        self,
        name: str,
        retention_days: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Create a namespace for logs, metrics, and traces."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support observability namespaces"
        )

    def ingest_logs(self, namespace: str, logs: list[dict[str, Any]], **kwargs) -> int:
        """Ingest structured log events into an observability namespace."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support log ingest"
        )

    def query_logs(
        self,
        namespace: str,
        start_time_ns: int,
        end_time_ns: int,
        query: str | None = None,
        limit: int = 100,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Query log events from an observability namespace."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support log queries"
        )

    def ingest_metrics(
        self, namespace: str, samples: list[dict[str, Any]], **kwargs
    ) -> int:
        """Ingest metric samples into an observability namespace."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support metric ingest"
        )

    def aggregate_metrics(
        self,
        namespace: str,
        metric_name: str,
        aggregation: str = "avg",
        start_time: str | None = None,
        end_time: str | None = None,
        step_seconds: int = 60,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Aggregate metrics from an observability namespace."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support metric aggregation"
        )

    def ingest_traces(
        self, namespace: str, traces: list[dict[str, Any]], **kwargs
    ) -> int:
        """Ingest trace spans into an observability namespace."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support trace ingest"
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
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Query trace spans in an observability namespace."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support trace queries"
        )

    def get_trace(self, namespace: str, trace_id: str, **kwargs) -> dict[str, Any]:
        """Get all spans for a specific trace ID."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support trace reads"
        )

    # ==========================================================================
    # Graph Operations (TD-SDK-3 batch 1 — the graph-CRUD family)
    #
    # Optional namespace, same pattern as documents/timeseries above: adapters
    # implement what their transport supports and let the rest fall through to
    # these NotImplementedError defaults. Canonical signatures mirror the v1
    # unified client's public graph surface so v2 methods are drop-in; the
    # ``graph_id`` keyword (``None`` → the server's "default" graph) is the one
    # canonical spelling — adapters translate, never the facade.
    # ==========================================================================

    def create_node(
        self,
        node_id: str,
        labels: list[str],
        properties: dict[str, Any] | None = None,
        embedding: list[float] | None = None,
        graph_id: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support graph node creation"
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
        **kwargs,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support graph edge creation"
        )

    def batch_create_nodes(
        self, nodes: list[dict[str, Any]], graph_id: str | None = None, **kwargs
    ) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support graph node batches"
        )

    def batch_create_edges(
        self, edges: list[dict[str, Any]], graph_id: str | None = None, **kwargs
    ) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support graph edge batches"
        )

    def get_node(
        self, node_id: str, graph_id: str | None = None, **kwargs
    ) -> dict[str, Any] | None:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support graph node reads"
        )

    def get_outgoing_edges(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        graph_id: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support graph edge reads"
        )

    def get_incoming_edges(
        self,
        node_id: str,
        edge_types: list[str] | None = None,
        graph_id: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support graph edge reads"
        )

    def delete_node(self, node_id: str, graph_id: str | None = None, **kwargs) -> bool:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support graph node deletion"
        )

    def query_nodes(
        self,
        labels: list[str] | None = None,
        properties: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        graph_id: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support graph node queries"
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
        **kwargs,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support graph traversal"
        )

    def create_graph(
        self,
        graph_id: str,
        name: str | None = None,
        description: str | None = None,
        schema: dict[str, Any] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support graph collection creation"
        )

    def delete_graph(self, graph_id: str, **kwargs) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support graph collection deletion"
        )

    def get_graph(self, graph_id: str, **kwargs) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support graph collection reads"
        )

    def list_graphs(self, **kwargs) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support graph collection listing"
        )

    def get_graph_stats(self, graph_id: str, **kwargs) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support graph collection stats"
        )

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
        **kwargs,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support graph shortest path"
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
        **kwargs,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support tuned graph traversal"
        )

    # ==========================================================================
    # Lifecycle Methods
    # ==========================================================================

    def close(self) -> None:
        """Close any open connections. Override in subclasses if needed."""
        pass

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
