"""
ProximaDB Embedded Adapter

Wraps the PyO3 embedded bindings to implement the SDK adapter interface.
This adapter does not open REST, gRPC, Arrow Flight, or PostgreSQL wire ports;
it calls the in-process embedded database object directly and converts raw
PyO3 responses to standardized Pydantic models.

This adapter is the key to Task 2.2: Unified Embedded API - ensuring
embedded mode returns the same response types as REST/gRPC modes.

Copyright 2025 ProximaDB Contributors
Licensed under the Apache License, Version 2.0
"""

import logging
import time
from typing import Any

import numpy as np

from ..models import (
    BatchResult,
    Collection,
    CollectionConfig,
    CollectionStats,
    DistanceMetric,
    FilterDict,
    HealthStatus,
    MetadataDict,
    OperationMetrics,
    SearchResult,
    VectorArray,
    VectorOperationResponse,
    VectorRecord,
)
from ..models_v2 import ProximaRecord
from ..proto_conversion import ProtoConverter
from .base import BaseProtocolAdapter

logger = logging.getLogger(__name__)


class EmbeddedProtocolAdapter(BaseProtocolAdapter):
    """Embedded adapter implementing BaseProtocolAdapter.

    Wraps the PyO3 EmbeddedProximaDB bindings to provide a consistent
    interface that returns Pydantic models. This is critical for ensuring
    embedded mode has API parity with REST/gRPC modes.

    Embedded is not a network protocol. It shares the adapter interface so the
    unified client can route operations consistently. The hot path is direct
    in-process calls into the Rust service facade ONLY when the
    ``proximadb_embedded`` PyO3 wheel is importable; otherwise this adapter
    silently used to fall back to a managed ``proximadb-server`` subprocess
    reached over UDS+HTTP+JSON — a very different performance envelope (every
    call pays HTTP framing and JSON both ways). The fallback still exists for
    availability, but it now logs a WARNING and is visible via
    ``transport_kind`` so nobody benchmarks the subprocess believing it is
    in-process (which is exactly how a published comparison went wrong).

    Key transformations:
    - insert() returns int (count) -> VectorOperationResponse
    - search() returns list of tuples -> List[SearchResult]
    - create_collection() returns raw object -> Collection
    """

    def __init__(
        self,
        data_dir: str = "/tmp/proximadb/data",
        config: dict[str, Any] | None = None,
        embedded_db: Any | None = None,
        **kwargs,
    ):
        """Initialize embedded protocol adapter.

        Args:
            data_dir: Directory for persistent storage
            config: Optional configuration dictionary
            **kwargs: Additional configuration passed to embedded DB
        """
        self._data_dir = data_dir
        self._connected = False
        self._collections: dict[str, Collection] = {}
        self._db = embedded_db

        try:
            if self._db is None:
                try:
                    try:
                        from proximadb_embedded import ProximaDB
                    except ImportError:
                        from proximadb import ProximaDB

                    if hasattr(config, "model_dump"):
                        embedded_kwargs = config.model_dump(exclude_none=True)
                    elif isinstance(config, dict):
                        embedded_kwargs = dict(config)
                    else:
                        embedded_kwargs = {}
                    embedded_kwargs.update(kwargs)

                    data_dirs = embedded_kwargs.pop(
                        "data_dirs", None
                    ) or embedded_kwargs.pop("data_dir", data_dir)
                    metadata_dir = embedded_kwargs.pop("metadata_dir", None)
                    cache_size_mb = embedded_kwargs.pop("cache_size_mb", 512)
                    default_engine = embedded_kwargs.pop("default_engine", "sst")
                    enable_wal = embedded_kwargs.pop("enable_wal", True)
                    prune_mode = embedded_kwargs.pop("prune_mode", None)
                    mode = embedded_kwargs.pop("mode", "exclusive")
                    node_id = embedded_kwargs.pop("node_id", None)

                    self._db = ProximaDB(
                        data_dirs=data_dirs,
                        metadata_dir=metadata_dir,
                        cache_size_mb=cache_size_mb,
                        default_engine=default_engine,
                        enable_wal=enable_wal,
                        prune_mode=prune_mode,
                        mode=mode,
                        node_id=node_id,
                    )
                    self._transport_kind = "in-process"
                except ImportError:
                    logger.warning(
                        "proximadb_embedded (in-process PyO3) is not installed; "
                        "falling back to a managed proximadb-server SUBPROCESS "
                        "over UDS+HTTP. Every operation pays HTTP framing and "
                        "JSON serialization. For the in-process engine: "
                        "pip install proximadb_embedded  (see proximaDB#1675)."
                    )
                    self._transport_kind = "subprocess"
                    from ..embedded import EmbeddedConfig, EmbeddedProximaDB

                    if hasattr(config, "model_dump"):
                        embedded_config = EmbeddedConfig(
                            **config.model_dump(exclude_none=True)
                        )
                    elif isinstance(config, dict):
                        embedded_config = EmbeddedConfig(**config)
                    else:
                        embedded_config = EmbeddedConfig(data_dir=data_dir, **kwargs)
                    self._db = EmbeddedProximaDB(config=embedded_config)

            self._connected = self._db is not None
        except ImportError as e:
            logger.error(f"Embedded mode not available: {e}")
            raise ImportError(
                "Embedded mode requires the native ProximaDB embedded package "
                "(canonical package: `proximadb_embedded`; legacy local alias: `proximadb`). "
                "Install/build the embedded release first."
            ) from e

    @property
    def transport_kind(self) -> str:
        """``"in-process"`` (PyO3 wheel) or ``"subprocess"`` (UDS+HTTP fallback).

        ``protocol_name`` stays ``"embedded"`` on both branches because the
        unified client routes on it; this property exists so callers and
        benchmarks can tell WHICH embedded they actually got.
        """
        return getattr(self, "_transport_kind", "unknown")

    @property
    def protocol_name(self) -> str:
        """Return the adapter name used by SDK routing."""
        return "embedded"

    @property
    def is_connected(self) -> bool:
        """Check if the adapter is connected and operational."""
        return self._connected and self._db is not None

    @staticmethod
    def _build_collection_model(
        name: str,
        dimension: int,
        storage_engine: str = "sst",
        vector_count: int = 0,
    ) -> Collection:
        """Build a Collection model aligned with the SDK schema."""
        return Collection(
            id=name,
            config=CollectionConfig(
                name=name,
                dimension=dimension,
                distance_metric=DistanceMetric.COSINE,
                storage_engine=storage_engine,
            ),
            stats=CollectionStats(vector_count=vector_count),
        )

    # ==========================================================================
    # Health & Server Operations
    # ==========================================================================

    def health(self) -> HealthStatus:
        """Check embedded database health without network I/O."""
        if not self._connected or self._db is None:
            return HealthStatus(
                status="unhealthy",
                healthy=False,
                timestamp_ms=int(time.time() * 1000),
                services={"embedded": "not initialized"},
            )

        try:
            # Basic health check - list collections to verify DB is operational
            collections = (
                self._db.list_collections()
                if hasattr(self._db, "list_collections")
                else []
            )

            return HealthStatus(
                status="healthy",
                healthy=True,
                timestamp_ms=int(time.time() * 1000),
                services={
                    "embedded": "ok",
                    "collections_count": len(collections) if collections else 0,
                },
            )
        except Exception as e:
            return HealthStatus(
                status="unhealthy",
                healthy=False,
                timestamp_ms=int(time.time() * 1000),
                services={"embedded": str(e)},
            )

    # ==========================================================================
    # Collection Operations
    # ==========================================================================

    def create_collection(
        self, name: str, config: CollectionConfig | None = None, **kwargs
    ) -> Collection:
        """Create a new vector collection."""
        # Extract parameters from config or kwargs
        dimension = config.dimension if config else kwargs.get("dimension", 128)

        # Convert storage engine to string for embedded API
        storage_engine = "sst"
        if config and config.storage_engine:
            storage_engine = ProtoConverter.storage_engine_to_str(config.storage_engine)
        elif "storage_engine" in kwargs or "engine" in kwargs:
            engine = kwargs.get("storage_engine") or kwargs.get("engine")
            storage_engine = ProtoConverter.storage_engine_to_str(engine)

        # Create collection via embedded API
        try:
            try:
                self._db.create_collection(name, dimension, storage_engine)
            except TypeError:
                self._db.create_collection(
                    name=name,
                    dimension=dimension,
                    engine=storage_engine,
                )

            # Build Collection model
            collection = self._build_collection_model(
                name=name,
                dimension=dimension,
                storage_engine=storage_engine,
            )

            # Cache collection for later lookups
            self._collections[name] = collection

            return collection

        except Exception as e:
            logger.error(f"Failed to create collection: {e}")
            raise

    def get_collection(self, collection_id: str) -> Collection | None:
        """Get collection metadata by ID or name."""
        # Check cache first
        if collection_id in self._collections:
            return self._collections[collection_id]

        try:
            if hasattr(self._db, "get_collection"):
                result = self._db.get_collection(collection_id)
                if result:
                    collection = self._build_collection_model(
                        name=getattr(result, "name", collection_id),
                        dimension=getattr(result, "dimension", 0),
                        storage_engine=getattr(result, "engine", "sst"),
                        vector_count=getattr(result, "vector_count", 0) or 0,
                    )
                    self._collections[collection_id] = collection
                    return collection

            # Fallback: check if collection exists in list
            collections = self.list_collections()
            for c in collections:
                if c.id == collection_id or c.name == collection_id:
                    return c

            return None
        except Exception as e:
            logger.debug(f"Collection not found: {collection_id} - {e}")
            return None

    def list_collections(self) -> list[Collection]:
        """List all collections."""
        try:
            if hasattr(self._db, "list_collections"):
                results = self._db.list_collections()

                collections = []
                for item in results or []:
                    if isinstance(item, Collection):
                        collections.append(item)
                    elif isinstance(item, str):
                        # Some embedded APIs return just names
                        collection = self._build_collection_model(
                            name=item,
                            dimension=0,
                        )
                        collections.append(collection)
                    elif hasattr(item, "name"):
                        collection = self._build_collection_model(
                            name=getattr(item, "name", ""),
                            dimension=getattr(item, "dimension", 0),
                            storage_engine=getattr(item, "engine", "sst"),
                            vector_count=getattr(item, "vector_count", 0) or 0,
                        )
                        collections.append(collection)

                return collections

            return list(self._collections.values())
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection."""
        try:
            if hasattr(self._db, "delete_collection"):
                self._db.delete_collection(collection_id)

            # Remove from cache
            self._collections.pop(collection_id, None)
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    def _normalize_vector_records(
        self, vectors: list[VectorRecord] | list[dict[str, Any]]
    ) -> tuple[list[str], list[list[float]], list[dict[str, Any]] | None]:
        ids: list[str] = []
        vector_values: list[list[float]] = []
        metadata_values: list[dict[str, Any]] = []
        include_metadata = False

        for index, vector in enumerate(vectors):
            if isinstance(vector, dict):
                payload = dict(vector)
            elif hasattr(vector, "model_dump"):
                payload = vector.model_dump(exclude_none=True)
            else:
                payload = ProtoConverter.vector_record_to_dict(vector)

            ids.append(str(payload.get("id") or f"vec_{index}"))
            vector_values.append(list(payload.get("vector") or []))

            metadata = dict(payload.get("metadata") or {})
            if metadata:
                include_metadata = True
            metadata_values.append(metadata)

        return ids, vector_values, (metadata_values if include_metadata else None)

    @staticmethod
    def _build_vector_operation_response(
        operation: str,
        total_count: int,
        start_time: float,
        result: Any,
        error: Exception | None = None,
    ) -> VectorOperationResponse:
        duration_ms = (time.time() - start_time) * 1000
        if error is not None:
            return VectorOperationResponse(
                success=False,
                operation=operation,
                error_message=str(error),
                metrics=OperationMetrics(
                    successful_count=0,
                    failed_count=total_count,
                    duration_ms=duration_ms,
                    total_count=total_count,
                ),
            )

        successful_count = result if isinstance(result, int) else total_count
        failed_count = max(total_count - successful_count, 0)
        return VectorOperationResponse(
            success=True,
            operation=operation,
            metrics=OperationMetrics(
                successful_count=successful_count,
                failed_count=failed_count,
                duration_ms=duration_ms,
                total_count=total_count,
            ),
        )

    def _execute_numpy_vector_batch(
        self,
        collection_id: str,
        ids: list[str],
        vectors: VectorArray | list[list[float]],
        metadata_values: list[dict[str, Any]] | None,
        operation: str,
        *,
        upsert: bool,
    ) -> VectorOperationResponse:
        start_time = time.time()
        vector_array = np.asarray(vectors, dtype=np.float32)
        if vector_array.ndim != 2:
            raise ValueError(
                f"Expected 2D vector array for embedded {operation.lower()}, "
                f"got shape {vector_array.shape}"
            )

        try:
            if upsert and hasattr(self._db, "upsert_numpy"):
                result = self._db.upsert_numpy(
                    collection_id,
                    ids,
                    vector_array,
                    metadata_values,
                )
            elif upsert and hasattr(self._db, "upsert"):
                result = self._db.upsert(
                    collection_id,
                    ids,
                    vector_array.tolist(),
                    metadata_values,
                )
            elif hasattr(self._db, "insert_numpy"):
                result = self._db.insert_numpy(
                    collection_id,
                    ids,
                    vector_array,
                    metadata_values,
                )
            else:
                result = self._db.insert(
                    collection_id,
                    ids,
                    vector_array.tolist(),
                    metadata_values,
                )

            return self._build_vector_operation_response(
                operation,
                len(ids),
                start_time,
                result,
            )
        except Exception as e:
            return self._build_vector_operation_response(
                operation,
                len(ids),
                start_time,
                None,
                error=e,
            )

    def insert_numpy(
        self,
        collection_id: str,
        ids: list[str],
        vectors: VectorArray | list[list[float]],
        metadata_values: list[dict[str, Any]] | None = None,
    ) -> VectorOperationResponse:
        return self._execute_numpy_vector_batch(
            collection_id,
            ids,
            vectors,
            metadata_values,
            "INSERT",
            upsert=False,
        )

    def upsert_numpy(
        self,
        collection_id: str,
        ids: list[str],
        vectors: VectorArray | list[list[float]],
        metadata_values: list[dict[str, Any]] | None = None,
    ) -> VectorOperationResponse:
        return self._execute_numpy_vector_batch(
            collection_id,
            ids,
            vectors,
            metadata_values,
            "UPSERT",
            upsert=True,
        )

    # ==========================================================================
    # Record Operations
    # ==========================================================================

    @staticmethod
    def _batch_result_from_vector_response(
        response: VectorOperationResponse, total_count: int
    ) -> BatchResult:
        return BatchResult(
            total=total_count,
            success=response.metrics.successful_count,
            failed=response.metrics.failed_count,
            errors=[response.error_message] if response.error_message else [],
            metrics=response.metrics,
        )

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
        records: list[ProximaRecord] | list[dict[str, Any]],
        **kwargs,
    ) -> BatchResult:
        """Insert ProximaRecord-shaped payloads into a collection.

        Prefer a native embedded record helper when the PyO3 package exposes it.
        Current vector-only embedded builds still route through the shared numpy
        write path as a temporary compatibility alias.
        """
        if self._db is not None and hasattr(self._db, "insert_records"):
            start_time = time.time()
            try:
                result = self._db.insert_records(collection_id, records, **kwargs)
                successful = result if isinstance(result, int) else len(records)
                failed = max(len(records) - successful, 0)
                return BatchResult(
                    total=len(records),
                    success=successful,
                    failed=failed,
                    metrics=OperationMetrics(
                        total_processed=len(records),
                        successful_count=successful,
                        failed_count=failed,
                        processing_time_us=int((time.time() - start_time) * 1_000_000),
                    ),
                )
            except Exception as exc:
                return BatchResult(
                    total=len(records),
                    success=0,
                    failed=len(records),
                    errors=[str(exc)],
                    metrics=OperationMetrics(
                        total_processed=len(records),
                        successful_count=0,
                        failed_count=len(records),
                        processing_time_us=int((time.time() - start_time) * 1_000_000),
                    ),
                )

        ids, vector_values, metadata_values = self._normalize_vector_records(records)
        response = self.insert_numpy(collection_id, ids, vector_values, metadata_values)
        return self._batch_result_from_vector_response(response, len(records))

    def upsert_records(
        self,
        collection_id: str,
        records: list[ProximaRecord] | list[dict[str, Any]],
        **kwargs,
    ) -> BatchResult:
        """Upsert ProximaRecord-shaped payloads into a collection."""
        if self._db is not None and hasattr(self._db, "upsert_records"):
            start_time = time.time()
            try:
                result = self._db.upsert_records(collection_id, records, **kwargs)
                successful = result[0] if isinstance(result, tuple) else len(records)
                failed = max(len(records) - successful, 0)
                return BatchResult(
                    total=len(records),
                    success=successful,
                    failed=failed,
                    metrics=OperationMetrics(
                        total_processed=len(records),
                        successful_count=successful,
                        failed_count=failed,
                        processing_time_us=int((time.time() - start_time) * 1_000_000),
                    ),
                )
            except Exception as exc:
                return BatchResult(
                    total=len(records),
                    success=0,
                    failed=len(records),
                    errors=[str(exc)],
                    metrics=OperationMetrics(
                        total_processed=len(records),
                        successful_count=0,
                        failed_count=len(records),
                        processing_time_us=int((time.time() - start_time) * 1_000_000),
                    ),
                )

        ids, vector_values, metadata_values = self._normalize_vector_records(records)
        response = self.upsert_numpy(collection_id, ids, vector_values, metadata_values)
        return self._batch_result_from_vector_response(response, len(records))

    # ==========================================================================
    # Vector Compatibility Aliases
    # ==========================================================================

    def insert_vectors(
        self,
        collection_id: str,
        vectors: list[VectorRecord] | list[dict[str, Any]],
        **kwargs,
    ) -> VectorOperationResponse:
        """Compatibility alias for record-native inserts."""
        return self._batch_to_vector_response(
            self.insert_records(collection_id, vectors, **kwargs), "INSERT"
        )

    def upsert_vectors(
        self,
        collection_id: str,
        vectors: list[VectorRecord] | list[dict[str, Any]],
        **kwargs,
    ) -> VectorOperationResponse:
        """Compatibility alias for record-native upserts."""
        return self._batch_to_vector_response(
            self.upsert_records(collection_id, vectors, **kwargs), "UPSERT"
        )

    def get_vectors(
        self,
        collection_id: str,
        vector_ids: list[str],
        include_vectors: bool = True,
        **kwargs,
    ) -> list[VectorRecord]:
        """Get vectors by IDs."""
        try:
            if hasattr(self._db, "get_vectors"):
                results = self._db.get_vectors(collection_id, vector_ids)
            elif hasattr(self._db, "get_vector"):
                results = [
                    self._db.get_vector(collection_id, vector_id)
                    for vector_id in vector_ids
                ]
            else:
                logger.warning("get_vectors not implemented in embedded API")
                return []

            # Convert to VectorRecord list
            records = []
            for r in results or []:
                if isinstance(r, VectorRecord):
                    records.append(r)
                elif isinstance(r, dict):
                    records.append(VectorRecord(**r))
                elif hasattr(r, "id"):
                    records.append(
                        VectorRecord(
                            id=getattr(r, "id", ""),
                            vector=(
                                list(getattr(r, "vector", []))
                                if include_vectors
                                else None
                            ),
                            metadata=dict(getattr(r, "metadata", {})),
                        )
                    )

            return records

        except Exception as e:
            logger.error(f"Failed to get vectors: {e}")
            return []

    def delete_vectors(
        self, collection_id: str, vector_ids: list[str], **kwargs
    ) -> VectorOperationResponse:
        """Delete vectors by IDs."""
        start_time = time.time()

        try:
            if hasattr(self._db, "delete_vectors"):
                result = self._db.delete_vectors(collection_id, vector_ids)
            elif hasattr(self._db, "delete_vector") and len(vector_ids) == 1:
                result = (
                    1 if self._db.delete_vector(collection_id, vector_ids[0]) else 0
                )
            else:
                return VectorOperationResponse(
                    success=False,
                    operation="DELETE",
                    error_message="delete_vectors not implemented in embedded API",
                )

            duration_ms = (time.time() - start_time) * 1000

            if isinstance(result, int):
                return VectorOperationResponse(
                    success=True,
                    operation="DELETE",
                    metrics=OperationMetrics(
                        successful_count=result,
                        failed_count=len(vector_ids) - result,
                        duration_ms=duration_ms,
                        total_count=len(vector_ids),
                    ),
                )

            return VectorOperationResponse(
                success=True,
                operation="DELETE",
                metrics=OperationMetrics(
                    successful_count=len(vector_ids),
                    failed_count=0,
                    duration_ms=duration_ms,
                    total_count=len(vector_ids),
                ),
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return VectorOperationResponse(
                success=False,
                operation="DELETE",
                error_message=str(e),
                metrics=OperationMetrics(
                    successful_count=0,
                    failed_count=len(vector_ids),
                    duration_ms=duration_ms,
                    total_count=len(vector_ids),
                ),
            )

    def update_vector_metadata(
        self, collection_id: str, vector_id: str, metadata: MetadataDict, **kwargs
    ) -> VectorOperationResponse:
        """Update metadata for a specific vector."""
        start_time = time.time()

        try:
            if hasattr(self._db, "update_metadata"):
                result = self._db.update_metadata(collection_id, vector_id, metadata)
            else:
                # Fallback: get, update, upsert
                vectors = self.get_vectors(collection_id, [vector_id])
                if vectors:
                    v = vectors[0]
                    updated_meta = (
                        {**v.metadata, **metadata} if v.metadata else metadata
                    )
                    return self.upsert_vectors(
                        collection_id,
                        [
                            VectorRecord(
                                id=vector_id, vector=v.vector, metadata=updated_meta
                            )
                        ],
                    )
                return VectorOperationResponse(
                    success=False,
                    operation="UPDATE",
                    error_message=f"Vector {vector_id} not found",
                )

            duration_ms = (time.time() - start_time) * 1000

            return VectorOperationResponse(
                success=True,
                operation="UPDATE",
                metrics=OperationMetrics(
                    successful_count=1,
                    failed_count=0,
                    duration_ms=duration_ms,
                    total_count=1,
                ),
            )

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            return VectorOperationResponse(
                success=False,
                operation="UPDATE",
                error_message=str(e),
                metrics=OperationMetrics(
                    successful_count=0,
                    failed_count=1,
                    duration_ms=duration_ms,
                    total_count=1,
                ),
            )

    # ==========================================================================
    # Search Operations
    # ==========================================================================

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
        """Search for similar vectors.

        The embedded API returns PyO3 objects or dictionaries from direct
        service calls. This method converts them to SearchResult objects.
        """
        try:
            filter_expr = None
            if isinstance(filter, dict) and filter:
                filter_expr = " AND ".join(
                    f"{key} = '{value}'" for key, value in filter.items()
                )

            if isinstance(query_vector, np.ndarray):
                query_array = np.asarray(query_vector, dtype=np.float32)
                query_list = None
            else:
                query_list = list(query_vector)
                query_array = np.asarray(query_list, dtype=np.float32)

            if hasattr(self._db, "search_numpy"):
                results = self._db.search_numpy(
                    collection_id,
                    query_array,
                    top_k=top_k,
                    filter=filter_expr,
                )
            else:
                results = self._db.search(
                    collection_id,
                    query_list if query_list is not None else query_array.tolist(),
                    top_k=top_k,
                    filter=filter_expr,
                )

            return self._to_search_results(results, include_vectors, include_metadata)

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []

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
        # Normalize query vectors
        normalized_queries = []
        for qv in query_vectors:
            if hasattr(qv, "tolist"):
                normalized_queries.append(qv.tolist())
            else:
                normalized_queries.append(list(qv))

        try:
            if hasattr(self._db, "batch_search"):
                results = self._db.batch_search(
                    collection_id,
                    normalized_queries,
                    k=top_k,
                    filter=filter,
                    include_vectors=include_vectors,
                    include_metadata=include_metadata,
                )

                # Convert batch results
                batch_results = []
                for query_results in results or []:
                    batch_results.append(
                        self._to_search_results(
                            query_results, include_vectors, include_metadata
                        )
                    )
                return batch_results

            else:
                # Fallback: execute individual searches
                batch_results = []
                for qv in normalized_queries:
                    r = self.search(
                        collection_id,
                        qv,
                        top_k,
                        filter,
                        include_vectors,
                        include_metadata,
                        **kwargs,
                    )
                    batch_results.append(r)
                return batch_results

        except Exception as e:
            logger.error(f"Batch search failed: {e}")
            return [[] for _ in query_vectors]

    def _to_search_results(
        self, results: Any, include_vectors: bool, include_metadata: bool
    ) -> list[SearchResult]:
        """Convert embedded search results to SearchResult list.

        Handles various result formats:
        - List of tuples (id, score, metadata, vector)
        - List of dicts
        - List of objects with attributes
        """
        if results is None:
            return []

        search_results = []
        for r in results:
            try:
                if isinstance(r, SearchResult):
                    search_results.append(r)
                elif isinstance(r, tuple):
                    # Common format: (id, score, metadata, vector) or (id, score)
                    result_id = r[0] if len(r) > 0 else ""
                    score = r[1] if len(r) > 1 else 0.0
                    metadata = r[2] if len(r) > 2 and include_metadata else None
                    vector = r[3] if len(r) > 3 and include_vectors else None

                    search_results.append(
                        SearchResult(
                            id=str(result_id),
                            score=float(score),
                            vector=list(vector) if vector else None,
                            metadata=dict(metadata) if metadata else None,
                        )
                    )
                elif isinstance(r, dict):
                    search_results.append(
                        SearchResult(
                            id=r.get("id", r.get("vector_id", "")),
                            score=r.get("score", r.get("distance", 0.0)),
                            vector=r.get("vector") if include_vectors else None,
                            metadata=r.get("metadata") if include_metadata else None,
                        )
                    )
                elif hasattr(r, "id"):
                    vector = None
                    if include_vectors and hasattr(r, "vector"):
                        vector = list(r.vector) if r.vector else None

                    metadata = None
                    if include_metadata and hasattr(r, "metadata"):
                        metadata = dict(r.metadata) if r.metadata else {}

                    search_results.append(
                        SearchResult(
                            id=getattr(r, "id", ""),
                            score=getattr(r, "score", getattr(r, "distance", 0.0)),
                            vector=vector,
                            metadata=metadata,
                        )
                    )
            except Exception as e:
                logger.warning(f"Failed to convert search result: {e}")

        return search_results

    # ==========================================================================
    # Graph Operations
    # ==========================================================================

    @staticmethod
    def _graph_id_from_args(
        graph: str | None,
        kwargs: dict[str, Any],
        default: str = "default",
    ) -> str:
        return graph or kwargs.pop("graph_id", None) or default

    @staticmethod
    def _normalize_graph_node(node: Any) -> dict[str, Any] | None:
        if node is None:
            return None
        if isinstance(node, dict):
            return {
                "id": node.get("id"),
                "labels": list(node.get("labels", []) or []),
                "properties": dict(node.get("properties", {}) or {}),
            }
        return {
            "id": getattr(node, "id", None),
            "labels": list(getattr(node, "labels", []) or []),
            "properties": dict(getattr(node, "properties", {}) or {}),
        }

    @staticmethod
    def _normalize_graph_edge(edge: Any) -> dict[str, Any]:
        if isinstance(edge, dict):
            return {
                "id": edge.get("id"),
                "from_node_id": edge.get("from_node_id") or edge.get("from_node"),
                "to_node_id": edge.get("to_node_id") or edge.get("to_node"),
                "edge_type": edge.get("edge_type"),
                "weight": edge.get("weight"),
                "properties": dict(edge.get("properties", {}) or {}),
            }
        return {
            "id": getattr(edge, "id", None),
            "from_node_id": getattr(edge, "from_node_id", None)
            or getattr(edge, "from_node", None),
            "to_node_id": getattr(edge, "to_node_id", None)
            or getattr(edge, "to_node", None),
            "edge_type": getattr(edge, "edge_type", None),
            "weight": getattr(edge, "weight", None),
            "properties": dict(getattr(edge, "properties", {}) or {}),
        }

    def create_graph(self, graph_id: str, **kwargs) -> dict[str, Any]:
        """Create a graph collection via embedded API."""
        engine = kwargs.get("engine")
        try:
            if hasattr(self._db, "create_graph"):
                try:
                    self._db.create_graph(graph_id, engine)
                except TypeError:
                    self._db.create_graph(graph_id=graph_id, engine=engine)
                return {"success": True, "graph_id": graph_id}
            raise NotImplementedError("create_graph not implemented in embedded API")
        except Exception as e:
            logger.error(f"Failed to create graph: {e}")
            raise

    def delete_graph(self, graph_id: str, **kwargs) -> dict[str, Any]:
        """Delete a graph collection via embedded API."""
        try:
            if hasattr(self._db, "delete_graph"):
                self._db.delete_graph(graph_id)
                return {"success": True, "graph_id": graph_id}
            raise NotImplementedError("delete_graph not implemented in embedded API")
        except Exception as e:
            logger.error(f"Failed to delete graph: {e}")
            raise

    def query_nodes(
        self,
        graph: str | None = None,
        labels: list[str] | None = None,
        properties: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Query graph nodes via embedded API."""
        try:
            graph_id = self._graph_id_from_args(graph, kwargs)
            if hasattr(self._db, "query_nodes"):
                results = self._db.query_nodes(
                    graph_id=graph_id,
                    labels=labels,
                    properties=properties,
                    limit=limit,
                    offset=offset,
                )
                nodes = [self._normalize_graph_node(node) for node in (results or [])]
                return {"nodes": nodes, "total_count": len(nodes), "has_more": False}
            raise NotImplementedError("query_nodes not implemented in embedded API")
        except Exception as e:
            logger.error(f"Failed to query nodes: {e}")
            raise

    def traverse_graph(
        self,
        start_node_id: str,
        max_depth: int = 3,
        edge_types: list[str] | None = None,
        limit: int | None = None,
        graph: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Traverse graph via embedded API."""
        try:
            graph_id = self._graph_id_from_args(graph, kwargs)
            if hasattr(self._db, "traverse_graph"):
                result = self._db.traverse_graph(
                    graph_id=graph_id,
                    start_node_id=start_node_id,
                    max_depth=max_depth,
                    edge_types=edge_types,
                    limit=limit,
                )
                return (
                    result if isinstance(result, dict) else {"nodes": [], "edges": []}
                )
            raise NotImplementedError("traverse_graph not implemented in embedded API")
        except Exception as e:
            logger.error(f"Failed to traverse graph: {e}")
            raise

    def create_node(
        self,
        node_id: str,
        labels: list[str],
        properties: dict[str, Any],
        graph: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Create a graph node via embedded API."""
        try:
            graph_id = self._graph_id_from_args(graph, kwargs)
            if hasattr(self._db, "create_node"):
                result = self._db.create_node(
                    graph_id=graph_id,
                    node_id=node_id,
                    labels=labels,
                    properties=properties or {},
                )
                return {"success": True, "node_id": node_id, "result": result}
            else:
                raise NotImplementedError("create_node not implemented in embedded API")
        except Exception as e:
            logger.error(f"Failed to create node: {e}")
            raise

    def create_edge(
        self,
        edge_id: str,
        edge_type: str,
        from_node: str | None = None,
        to_node: str | None = None,
        properties: dict[str, Any] | None = None,
        graph: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Create a graph edge via embedded API."""
        try:
            graph_id = self._graph_id_from_args(graph, kwargs)
            from_node_id = from_node or kwargs.pop("from_node_id", None)
            to_node_id = to_node or kwargs.pop("to_node_id", None)
            if hasattr(self._db, "create_edge"):
                result = self._db.create_edge(
                    graph_id=graph_id,
                    id=edge_id,
                    from_node_id=from_node_id,
                    to_node_id=to_node_id,
                    edge_type=edge_type,
                    weight=kwargs.get("weight"),
                    properties=properties or {},
                )
                return {"success": True, "edge_id": edge_id, "result": result}
            else:
                raise NotImplementedError("create_edge not implemented in embedded API")
        except Exception as e:
            logger.error(f"Failed to create edge: {e}")
            raise

    def get_node(
        self,
        node_id: str,
        graph: str | None = None,
        **kwargs,
    ) -> dict[str, Any] | None:
        """Get a graph node by ID via embedded API."""
        try:
            graph_id = self._graph_id_from_args(graph, kwargs)
            if hasattr(self._db, "get_node"):
                node = self._db.get_node(graph_id=graph_id, node_id=node_id)
                return self._normalize_graph_node(node)
            raise NotImplementedError("get_node not implemented in embedded API")
        except Exception as e:
            logger.error(f"Failed to get node: {e}")
            raise

    def get_outgoing_edges(
        self,
        node_id: str,
        graph: str | None = None,
        edge_types: list[str] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Get outgoing graph edges via embedded API."""
        try:
            graph_id = self._graph_id_from_args(graph, kwargs)
            if hasattr(self._db, "get_outgoing_edges"):
                edges = self._db.get_outgoing_edges(
                    graph_id=graph_id,
                    node_id=node_id,
                    edge_types=edge_types,
                )
                return [self._normalize_graph_edge(edge) for edge in (edges or [])]
            raise NotImplementedError(
                "get_outgoing_edges not implemented in embedded API"
            )
        except Exception as e:
            logger.error(f"Failed to get outgoing edges: {e}")
            raise

    def get_incoming_edges(
        self,
        node_id: str,
        graph: str | None = None,
        edge_types: list[str] | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Get incoming graph edges via embedded API."""
        try:
            graph_id = self._graph_id_from_args(graph, kwargs)
            if hasattr(self._db, "get_incoming_edges"):
                edges = self._db.get_incoming_edges(
                    graph_id=graph_id,
                    node_id=node_id,
                    edge_types=edge_types,
                )
                return [self._normalize_graph_edge(edge) for edge in (edges or [])]
            raise NotImplementedError(
                "get_incoming_edges not implemented in embedded API"
            )
        except Exception as e:
            logger.error(f"Failed to get incoming edges: {e}")
            raise

    def delete_node(
        self,
        node_id: str,
        graph: str | None = None,
        **kwargs,
    ) -> bool:
        """Delete a graph node via embedded API."""
        try:
            graph_id = self._graph_id_from_args(graph, kwargs)
            if hasattr(self._db, "delete_node"):
                return bool(self._db.delete_node(graph_id=graph_id, node_id=node_id))
            raise NotImplementedError("delete_node not implemented in embedded API")
        except Exception as e:
            logger.error(f"Failed to delete node: {e}")
            raise

    def get_graph_stats(self, graph_id: str, **kwargs) -> dict[str, Any]:
        """Get graph statistics via embedded API."""
        try:
            if hasattr(self._db, "graph_stats"):
                stats = self._db.graph_stats(graph_id)
                return {
                    "total_nodes": getattr(stats, "total_nodes", 0),
                    "total_edges": getattr(stats, "total_edges", 0),
                }
            raise NotImplementedError("graph_stats not implemented in embedded API")
        except Exception as e:
            logger.error(f"Failed to get graph stats: {e}")
            raise

    # NOTE (canonical-contract divergences, TD-SDK-3 batch 1 review): the
    # pre-existing graph methods below predate the base-class canonical
    # signatures and are kept kwargs-compatible rather than reordered — live
    # callers pass the legacy spellings positionally/kwarg-wise
    # (agentic_io.py ``from_node=``, embedded cov tests pass ``edge_type``
    # as the 2nd positional). Call them with KEYWORDS per the canonical
    # names (``from_node_id``/``to_node_id`` are popped from kwargs). Known
    # engine limitations, documented rather than faked: the engine's
    # ``traverse_graph`` has no ``algorithm``/``node_labels`` params (a DFS
    # request runs BFS), ``create_graph`` ignores ``name``/``description``/
    # ``schema``, and ``create_node`` ignores ``embedding``. Canonicalizing
    # these belongs to the v1→v2 swap batch (TD-SDK-3 step 3) together with
    # the agentic_io migration.
    def _engine_graph_classes(self) -> tuple[type, type] | None:
        """Resolve ``(GraphNode, GraphEdge)`` from the engine package.

        Resolution order mirrors the adapter's own engine import:

        1. ``proximadb_embedded`` — the CANONICAL wheel module (the adapter
           imports it first at connect time) and the one that re-exports
           GraphNode/GraphEdge in its ``__init__``.
        2. ``proximadb`` — the legacy top-level engine module (also shipped
           as a compat shim by the wheel, but WITHOUT the graph classes; it
           resolves here only for the raw legacy layout).
        3. The db object's own module — for pure-Python engines that co-locate
           the classes with the db class. This is the ONLY path that could
           work for PyO3 classes derived from ``type(db).__module__``, and it
           never fires there: a ``#[pyclass]`` without ``module = "..."``
           reports ``__module__ == "builtins"``, which is skipped.

        ``None`` → the batch methods degrade to the base class's
        NotImplementedError (graph.py then falls back to per-item creates).
        """
        import importlib
        import sys

        candidates = ["proximadb_embedded", "proximadb"]
        db_module = type(self._db).__module__
        if db_module and db_module != "builtins" and db_module not in candidates:
            candidates.append(db_module)
        for name in candidates:
            try:
                module = sys.modules.get(name) or importlib.import_module(name)
            except Exception:
                continue
            node_cls = getattr(module, "GraphNode", None)
            edge_cls = getattr(module, "GraphEdge", None)
            if node_cls is not None and edge_cls is not None:
                return node_cls, edge_cls
        return None

    def batch_create_nodes(
        self, nodes: list[dict[str, Any]], graph_id: str | None = None, **kwargs
    ) -> dict[str, Any]:
        """Batch-create nodes via the embedded engine's ``create_nodes``.

        Returns the same ``{success, created_count, nodes}`` shape as the
        REST/gRPC batch endpoints (a bare count would break facade parity).
        """
        try:
            gid = graph_id or kwargs.pop("graph_id", None) or "default"
            if hasattr(self._db, "create_nodes"):
                classes = self._engine_graph_classes()
                if classes is not None:
                    node_cls, _ = classes
                    items = [
                        node_cls(
                            n.get("id") or n.get("node_id"),
                            n.get("labels"),
                            n.get("properties"),
                        )
                        for n in nodes
                    ]
                    created = self._db.create_nodes(gid, items)
                    return {
                        "success": True,
                        "created_count": int(created),
                        "nodes": nodes,
                    }
            raise NotImplementedError(
                "batch_create_nodes not implemented in embedded API"
            )
        except Exception as e:
            logger.error(f"Failed to batch create nodes: {e}")
            raise

    def batch_create_edges(
        self, edges: list[dict[str, Any]], graph_id: str | None = None, **kwargs
    ) -> dict[str, Any]:
        """Batch-create edges via the embedded engine's ``create_edges``.

        Returns the same ``{success, created_count, edges}`` shape as the
        REST/gRPC batch endpoints.
        """
        try:
            gid = graph_id or kwargs.pop("graph_id", None) or "default"
            if hasattr(self._db, "create_edges"):
                classes = self._engine_graph_classes()
                if classes is not None:
                    _, edge_cls = classes
                    items = [
                        edge_cls(
                            e.get("from_node_id") or e.get("source"),
                            e.get("to_node_id") or e.get("target"),
                            e.get("edge_type") or e.get("type") or "RELATED",
                            e.get("id") or e.get("edge_id"),
                            e.get("weight"),
                            e.get("properties"),
                        )
                        for e in edges
                    ]
                    created = self._db.create_edges(gid, items)
                    return {
                        "success": True,
                        "created_count": int(created),
                        "edges": edges,
                    }
            raise NotImplementedError(
                "batch_create_edges not implemented in embedded API"
            )
        except Exception as e:
            logger.error(f"Failed to batch create edges: {e}")
            raise

    def execute_graph_query(self, graph: str, query: str, **kwargs) -> dict[str, Any]:
        """Execute a graph query via embedded API."""
        try:
            if hasattr(self._db, "execute_graph_query"):
                result = self._db.execute_graph_query(graph=graph, query=query)
                return {"results": result, "query": query}
            else:
                # Fall back to multi-modal query execution
                if hasattr(self._db, "execute_multi_modal_query"):
                    from ..models import MultiModalQuery, QueryComponent

                    component = QueryComponent(
                        type="graph",
                        collection=graph,
                        query=query,
                    )
                    mm_query = MultiModalQuery(components=[component])
                    result = self._db.execute_multi_modal_query(mm_query)
                    return {"results": result}
                else:
                    raise NotImplementedError(
                        "Graph query not implemented in embedded API"
                    )
        except Exception as e:
            logger.error(f"Failed to execute graph query: {e}")
            raise

    # ==========================================================================
    # Document Operations
    # ==========================================================================

    def create_document_collection(
        self, name: str, config: dict[str, Any] | None = None, **kwargs
    ) -> dict[str, Any]:
        """Create a document collection via embedded API."""
        try:
            if hasattr(self._db, "create_document_collection"):
                indexed_paths = None
                if config:
                    indexed_paths = config.get("indexed_paths") or config.get("indexes")
                result = self._db.create_document_collection(name, indexed_paths)
                return {"success": True, "collection_id": name, "result": result}
            else:
                # Fall back to creating a vector collection with document metadata
                return self._create_document_collection_as_vector(name, config)
        except Exception as e:
            logger.error(f"Failed to create document collection: {e}")
            raise

    def _create_document_collection_as_vector(
        self, name: str, config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create a document collection using vector storage as fallback."""
        # Create a vector collection with a special tag for documents
        dimension = config.get("dimension", 768) if config else 768
        collection = self.create_collection(name, config={"dimension": dimension})
        return {
            "success": True,
            "collection_id": name,
            "implementation": "vector_fallback",
            "collection": {
                "id": collection.id,
                "name": collection.name,
                "dimension": collection.dimension,
            },
        }

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
        """Ingest documents via the embedded API (native, else vector-fallback per record)."""
        results: list[dict[str, Any]] = []
        for rec in records:
            rid = rec.get("id")
            # Store the record's metadata AS the document — matches the legacy
            # insert_document shape so document reads (get_document/query_documents)
            # stay consistent. (embed_source/text aren't used on the embedded
            # vector-fallback path.)
            document = dict(rec.get("metadata") or {})
            document.setdefault("id", rid)
            try:
                if hasattr(self._db, "insert_document"):
                    doc_id, version = self._db.insert_document(
                        collection_id, document, rid
                    )
                else:
                    r = self._insert_document_as_vector(collection_id, document, rid)
                    doc_id, version = r["id"], r.get("version", 1)
                results.append({"id": doc_id, "version": version})
            except Exception as e:
                logger.error(f"Failed to ingest document {rid}: {e}")
                raise
        return {"mode": "embedded", "records": results}

    def insert_document(
        self,
        collection_name: str,
        document: dict[str, Any],
        id: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Insert a document via embedded API."""
        try:
            if hasattr(self._db, "insert_document"):
                doc_id, version = self._db.insert_document(
                    collection_name,
                    document,
                    id,
                )
                return {"id": doc_id, "success": True, "version": version}
            else:
                # Fall back to vector storage
                return self._insert_document_as_vector(collection_name, document, id)
        except Exception as e:
            logger.error(f"Failed to insert document: {e}")
            raise

    def _insert_document_as_vector(
        self, collection_name: str, document: dict[str, Any], id: str | None = None
    ) -> dict[str, Any]:
        """Insert a document using vector storage as fallback."""
        import json

        # Create a vector record from the document
        # Use a dummy vector for now (could be improved with embedding)
        doc_id = (
            id
            or document.get("id")
            or f"doc_{hash(json.dumps(document, sort_keys=True))}"
        )

        # Store document content in the source field
        vector_record = VectorRecord(
            id=doc_id,
            vector=[0.0] * 768,  # Dummy vector
            source=json.dumps(document),
            metadata={
                "document_type": "document",
                "collection": collection_name,
                **document.get("metadata", {}),
            },
        )

        result = self.insert_vectors(collection_name, [vector_record])
        return {
            "id": doc_id,
            "success": result.success,
            "version": 1,
            "implementation": "vector_fallback",
        }

    def get_document(
        self,
        collection_name: str,
        doc_id: str,
        projection: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any] | None:
        """Get a document by ID via embedded API."""
        try:
            if hasattr(self._db, "get_document"):
                result = self._db.get_document(collection_name, doc_id)
                return result
            else:
                # Fall back to vector storage
                return self._get_document_as_vector(collection_name, doc_id)
        except Exception as e:
            logger.debug(f"Document not found: {doc_id} - {e}")
            return None

    def _get_document_as_vector(
        self, collection_name: str, doc_id: str
    ) -> dict[str, Any] | None:
        """Get a document using vector storage as fallback."""
        import json

        vectors = self.get_vectors(collection_name, [doc_id], include_vectors=False)
        if not vectors:
            return None

        v = vectors[0]
        if v.source:
            try:
                document = json.loads(v.source)
                return {"id": doc_id, "document": document, "metadata": v.metadata}
            except json.JSONDecodeError:
                pass

        return {"id": doc_id, "document": {"source": v.source}, "metadata": v.metadata}

    def query_documents(
        self,
        collection_name: str,
        filter: dict[str, Any] | None = None,
        projection: list[str] | None = None,
        limit: int = 100,
        **kwargs,
    ) -> dict[str, Any]:
        """Query documents with filter via embedded API."""
        try:
            if hasattr(self._db, "query_documents"):
                filter_expr = None
                if isinstance(filter, dict) and filter:
                    filter_expr = " AND ".join(
                        f"{key} = '{value}'" for key, value in filter.items()
                    )
                result = self._db.query_documents(collection_name, filter_expr, limit)
                documents = [
                    {"id": doc_id, "document": document}
                    for doc_id, document in (result or [])
                ]
                return {"documents": documents, "count": len(documents)}
            else:
                # Fall back to vector search with metadata filter
                return self._query_documents_as_vector(collection_name, filter, limit)
        except Exception as e:
            logger.error(f"Failed to query documents: {e}")
            raise

    def _query_documents_as_vector(
        self, collection_name: str, filter: dict[str, Any] | None, limit: int
    ) -> dict[str, Any]:
        """Query documents using vector storage as fallback."""
        # For now, return all vectors (could be improved with filtering)
        # This is a simplified fallback implementation
        return {"documents": [], "count": 0, "implementation": "vector_fallback"}

    def update_document(
        self, collection_name: str, doc_id: str, updates: list[dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        """Update a document via embedded API."""
        try:
            if hasattr(self._db, "update_document"):
                update_map: dict[str, Any] = {}
                for update in updates:
                    path = update.get("path")
                    if path:
                        update_map[path] = update.get("value")
                self._db.update_document(collection_name, doc_id, update_map)
                return {"success": True}
            else:
                # Fall back to vector storage
                return self._update_document_as_vector(collection_name, doc_id, updates)
        except Exception as e:
            logger.error(f"Failed to update document: {e}")
            raise

    def _update_document_as_vector(
        self, collection_name: str, doc_id: str, updates: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Update a document using vector storage as fallback."""
        # Get existing document
        existing = self.get_document(collection_name, doc_id)
        if not existing:
            return {"success": False, "error": "Document not found"}

        # Apply updates
        document = existing.get("document", {})
        for update in updates:
            path = update.get("path", "")
            value = update.get("value")
            operation = update.get("operation", "SET")

            if operation == "SET" and path:
                # Simple dot notation support
                parts = path.replace("$.", "").split(".")
                target = document
                for part in parts[:-1]:
                    if part not in target:
                        target[part] = {}
                    target = target[part]
                target[parts[-1]] = value

        # Re-insert the updated document
        result = self.insert_document(collection_name, document, doc_id)
        return {"success": True, "new_version": 1, "implementation": "vector_fallback"}

    def delete_document(self, collection_name: str, doc_id: str, **kwargs) -> bool:
        """Delete a document via embedded API."""
        try:
            if hasattr(self._db, "delete_document"):
                return bool(self._db.delete_document(collection_name, doc_id))
            else:
                # Fall back to vector storage
                result = self.delete_vectors(collection_name, [doc_id])
                return result.success
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False

    def list_document_collections(self, **kwargs) -> list[dict[str, Any]]:
        """List all document collections via embedded API."""
        try:
            if hasattr(self._db, "list_document_collections"):
                result = self._db.list_document_collections()
                return result if isinstance(result, list) else []
            else:
                # Fall back to listing vector collections
                collections = self.list_collections()
                return [
                    {
                        "name": c.name,
                        "id": c.id,
                        "dimension": c.dimension,
                    }
                    for c in collections
                ]
        except Exception as e:
            logger.error(f"Failed to list document collections: {e}")
            return []

    def delete_document_collection(self, collection_name: str, **kwargs) -> bool:
        """Delete a document collection via embedded API."""
        try:
            if hasattr(self._db, "delete_document_collection"):
                return bool(self._db.delete_document_collection(collection_name))
            else:
                # Fall back to deleting vector collection
                return self.delete_collection(collection_name)
        except Exception as e:
            logger.error(f"Failed to delete document collection: {e}")
            return False

    # ==========================================================================
    # Hybrid Search Operations
    # ==========================================================================

    def hybrid_search(
        self,
        collection: str,
        text_query: str,
        query_vector: list[float],
        fusion_strategy: str = "rrf",
        top_k: int = 10,
        **kwargs,
    ) -> dict[str, Any]:
        """Execute hybrid search via embedded API."""
        try:
            if hasattr(self._db, "hybrid_search"):
                result = self._db.hybrid_search(
                    collection=collection,
                    text_query=text_query,
                    query_vector=query_vector,
                    fusion_strategy=fusion_strategy,
                    top_k=top_k,
                )
                return result
            else:
                # Fall back to vector search only
                return self._hybrid_search_as_vector(collection, query_vector, top_k)
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise

    def _hybrid_search_as_vector(
        self, collection: str, query_vector: list[float], top_k: int
    ) -> dict[str, Any]:
        """Fallback hybrid search using vector search only."""
        results = self.search(
            collection_id=collection,
            query_vector=query_vector,
            top_k=top_k,
        )

        return {
            "results": [
                {
                    "id": r.id,
                    "score": r.score,
                    "metadata": r.metadata,
                    "implementation": "vector_fallback",
                }
                for r in results
            ],
            "fusion_strategy": "vector_only",
            "total_time_ms": 0,
        }

    # ==========================================================================
    # Time-Series Operations
    # ==========================================================================

    def create_timeseries_collection(
        self, name: str, config: dict[str, Any] | None = None, **kwargs
    ) -> dict[str, Any]:
        """Create a time-series collection via embedded API."""
        try:
            if hasattr(self._db, "create_timeseries_collection"):
                result = self._db.create_timeseries_collection(
                    name=name, config=config or {}
                )
                return {"success": True, "collection_id": name, "result": result}
            else:
                # Fall back to creating a vector collection
                return self._create_timeseries_collection_as_vector(name, config)
        except Exception as e:
            logger.error(f"Failed to create time-series collection: {e}")
            raise

    def _create_timeseries_collection_as_vector(
        self, name: str, config: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create a time-series collection using vector storage as fallback."""
        dimension = config.get("dimension", 128) if config else 128
        collection = self.create_collection(name, config={"dimension": dimension})
        return {
            "success": True,
            "collection_id": name,
            "implementation": "vector_fallback",
            "collection": {
                "id": collection.id,
                "name": collection.name,
                "dimension": collection.dimension,
            },
        }

    def ingest_timeseries(
        self, collection_name: str, points: list[dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        """Ingest time-series data points via embedded API."""
        try:
            if hasattr(self._db, "ingest_timeseries"):
                result = self._db.ingest_timeseries(
                    collection=collection_name,
                    points=points,
                )
                return result
            else:
                # Fall back to vector storage
                return self._ingest_timeseries_as_vector(collection_name, points)
        except Exception as e:
            logger.error(f"Failed to ingest time-series data: {e}")
            raise

    def _ingest_timeseries_as_vector(
        self, collection_name: str, points: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Ingest time-series data using vector storage as fallback."""
        import json
        from datetime import datetime

        vectors = []
        for point in points:
            timestamp = point.get("timestamp", datetime.utcnow().isoformat())
            values = point.get("values", {})
            tags = point.get("tags", {})

            # Create a summary vector (hash of timestamp + values)
            vector_input = f"{timestamp}:{json.dumps(values, sort_keys=True)}"
            vector_hash = hash(vector_input) % 1000000 / 1000000.0
            dummy_vector = [vector_hash] * 128

            vector_record = VectorRecord(
                id=f"ts_{timestamp}_{hash(json.dumps(point, sort_keys=True))}",
                vector=dummy_vector,
                source=json.dumps(point),
                metadata={
                    "timestamp": timestamp,
                    "tags": tags,
                    "metric_names": list(values.keys()) if values else [],
                },
            )
            vectors.append(vector_record)

        result = self.insert_vectors(collection_name, vectors)
        return {
            "ingested_count": result.metrics.successful_count,
            "failed_count": result.metrics.failed_count,
            "implementation": "vector_fallback",
        }

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
        """Query time-series data with optional aggregation via embedded API."""
        try:
            if hasattr(self._db, "query_timeseries"):
                result = self._db.query_timeseries(
                    collection=collection_name,
                    start_time=start_time,
                    end_time=end_time,
                    aggregation=aggregation,
                    bucket_ms=bucket_ms,
                    tag_filters=tag_filters,
                )
                return result
            else:
                # Fall back to vector storage
                return self._query_timeseries_as_vector(
                    collection_name, start_time, end_time, tag_filters
                )
        except Exception as e:
            logger.error(f"Failed to query time-series data: {e}")
            raise

    def _query_timeseries_as_vector(
        self,
        collection_name: str,
        start_time: str,
        end_time: str,
        tag_filters: dict[str, str] | None,
    ) -> dict[str, Any]:
        """Query time-series data using vector storage as fallback."""
        import json

        # Get all vectors in the collection and filter by timestamp range
        all_vectors = self.get_vectors(collection_name, [], include_vectors=False)

        filtered_points = []
        for v in all_vectors:
            metadata = v.metadata or {}
            timestamp = metadata.get("timestamp", "")

            # Check time range
            if start_time and timestamp < start_time:
                continue
            if end_time and timestamp > end_time:
                continue

            # Check tag filters
            if tag_filters:
                tags = metadata.get("tags", {})
                match = True
                for key, value in tag_filters.items():
                    if tags.get(key) != value:
                        match = False
                        break
                if not match:
                    continue

            # Parse the point data
            point_data = {}
            if v.source:
                try:
                    point_data = json.loads(v.source)
                except json.JSONDecodeError:
                    point_data = {"raw": v.source}

            filtered_points.append(
                {
                    "timestamp": timestamp,
                    "values": point_data.get("values", {}),
                    "tags": metadata.get("tags", {}),
                }
            )

        return {
            "raw_points": filtered_points,
            "total_points": len(filtered_points),
            "implementation": "vector_fallback",
        }

    def list_timeseries_collections(self, **kwargs) -> list[dict[str, Any]]:
        """List all time-series collections via embedded API."""
        try:
            if hasattr(self._db, "list_timeseries_collections"):
                result = self._db.list_timeseries_collections()
                return result if isinstance(result, list) else []
            else:
                # Fall back to listing vector collections
                collections = self.list_collections()
                return [
                    {
                        "name": c.name,
                        "id": c.id,
                        "dimension": c.dimension,
                    }
                    for c in collections
                ]
        except Exception as e:
            logger.error(f"Failed to list time-series collections: {e}")
            return []

    def delete_timeseries_collection(self, collection_name: str, **kwargs) -> bool:
        """Delete a time-series collection via embedded API."""
        try:
            if hasattr(self._db, "delete_timeseries_collection"):
                result = self._db.delete_timeseries_collection(
                    collection=collection_name
                )
                return (
                    result.get("success", False) if isinstance(result, dict) else result
                )
            else:
                # Fall back to deleting vector collection
                return self.delete_collection(collection_name)
        except Exception as e:
            logger.error(f"Failed to delete time-series collection: {e}")
            return False

    def execute_sql(
        self,
        query: str,
        parameters: list[Any] | None = None,
        collection: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Execute SQL via the embedded API."""
        try:
            if hasattr(self._db, "execute_sql"):
                return self._db.execute_sql(query, parameters, collection)
            raise NotImplementedError("execute_sql not implemented in embedded API")
        except Exception as e:
            logger.error(f"Failed to execute SQL: {e}")
            raise

    def execute_unified_query(
        self,
        query: str,
        query_vector: list[float] | None = None,
        fusion_strategy: str | None = None,
        **kwargs,
    ) -> list[dict[str, Any]]:
        """Execute a multi-model query via the embedded API."""
        try:
            if hasattr(self._db, "execute_unified_query"):
                results = self._db.execute_unified_query(
                    query, query_vector, fusion_strategy
                )
                return list(results or [])
            raise NotImplementedError(
                "execute_unified_query not implemented in embedded API"
            )
        except Exception as e:
            logger.error(f"Failed to execute unified query: {e}")
            raise

    def create_observability_namespace(
        self, name: str, retention_days: int | None = None, **kwargs
    ) -> dict[str, Any]:
        try:
            if hasattr(self._db, "create_observability_namespace"):
                self._db.create_observability_namespace(name, retention_days)
                return {"success": True, "namespace": name}
            raise NotImplementedError(
                "create_observability_namespace not implemented in embedded API"
            )
        except Exception as e:
            logger.error(f"Failed to create observability namespace: {e}")
            raise

    def ingest_logs(self, namespace: str, logs: list[dict[str, Any]], **kwargs) -> int:
        if hasattr(self._db, "ingest_logs"):
            return int(self._db.ingest_logs(namespace, logs))
        raise NotImplementedError("ingest_logs not implemented in embedded API")

    def query_logs(
        self,
        namespace: str,
        start_time_ns: int,
        end_time_ns: int,
        query: str | None = None,
        limit: int = 100,
        **kwargs,
    ) -> list[dict[str, Any]]:
        if hasattr(self._db, "query_logs"):
            return list(
                self._db.query_logs(
                    namespace,
                    start_time_ns,
                    end_time_ns,
                    query,
                    limit,
                )
                or []
            )
        raise NotImplementedError("query_logs not implemented in embedded API")

    def ingest_metrics(
        self, namespace: str, samples: list[dict[str, Any]], **kwargs
    ) -> int:
        if hasattr(self._db, "ingest_metrics"):
            return int(self._db.ingest_metrics(namespace, samples))
        raise NotImplementedError("ingest_metrics not implemented in embedded API")

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
        if hasattr(self._db, "aggregate_metrics"):
            return list(
                self._db.aggregate_metrics(
                    namespace,
                    metric_name,
                    aggregation,
                    start_time,
                    end_time,
                    step_seconds,
                )
                or []
            )
        raise NotImplementedError("aggregate_metrics not implemented in embedded API")

    def ingest_traces(
        self, namespace: str, traces: list[dict[str, Any]], **kwargs
    ) -> int:
        if hasattr(self._db, "ingest_traces"):
            return int(self._db.ingest_traces(namespace, traces))
        raise NotImplementedError("ingest_traces not implemented in embedded API")

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
        if hasattr(self._db, "query_traces"):
            return list(
                self._db.query_traces(
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
        raise NotImplementedError("query_traces not implemented in embedded API")

    def get_trace(self, namespace: str, trace_id: str, **kwargs) -> dict[str, Any]:
        if hasattr(self._db, "get_trace"):
            return dict(self._db.get_trace(namespace, trace_id) or {})
        raise NotImplementedError("get_trace not implemented in embedded API")

    # ==========================================================================
    # Lifecycle Methods
    # ==========================================================================

    def close(self) -> None:
        """Close the embedded database."""
        if self._db is not None:
            try:
                if hasattr(self._db, "close"):
                    self._db.close()
                elif hasattr(self._db, "shutdown"):
                    self._db.shutdown()
            except Exception as e:
                logger.warning(f"Error closing embedded database: {e}")
            finally:
                self._db = None
                self._connected = False
                self._collections.clear()
