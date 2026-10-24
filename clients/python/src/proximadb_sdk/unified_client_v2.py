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

from .adapters import BaseProtocolAdapter, create_adapter
from .config import ClientConfig, Protocol, load_config
from .exceptions import ProximaDBError
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
