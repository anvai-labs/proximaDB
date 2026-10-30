"""
ProximaDB gRPC Protocol Adapter

Wraps the gRPC protocol client to implement the BaseProtocolAdapter interface.
Converts gRPC/protobuf responses to standardized Pydantic models.

Copyright 2025 ProximaDB Contributors
Licensed under the Apache License, Version 2.0
"""

import logging
from typing import Any

from ..models import (
    BatchResult,
    Collection,
    CollectionConfig,
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


class GrpcProtocolAdapter(BaseProtocolAdapter):
    """gRPC protocol adapter implementing BaseProtocolAdapter.

    Wraps the existing ProximaDBSyncGrpcClient to provide a consistent
    interface that returns Pydantic models.
    """

    def __init__(
        self,
        server_address: str = "localhost:5678",
        timeout: float = 60.0,
        pool_size: int = 5,
        max_message_size: int = 64 * 1024 * 1024,
        **kwargs,
    ):
        """Initialize gRPC protocol adapter.

        Args:
            server_address: gRPC server address (host:port). Default is unified port 5678
                           which handles both REST and gRPC via TCP multiplexing.
            timeout: Request timeout in seconds
            pool_size: Number of gRPC channels in connection pool
            max_message_size: Maximum message size in bytes
            **kwargs: Additional configuration passed to underlying client
        """
        from ..protocols.grpc_sync import ProximaDBSyncGrpcClient

        config = kwargs.pop("config", None)
        api_key = kwargs.pop("api_key", None)
        auth = kwargs.pop("auth", None)
        kwargs.pop("url", None)
        kwargs.pop("base_url", None)

        if config is not None and server_address == "localhost:5678":
            config_url = getattr(config, "url", None) or getattr(
                config, "base_url", None
            )
            if config_url:
                server_address = (
                    str(config_url).replace("http://", "").replace("https://", "")
                )

        metadata = (
            config.get_grpc_metadata() if hasattr(config, "get_grpc_metadata") else None
        )
        if not metadata and api_key:
            metadata = [("authorization", f"Bearer {api_key}")]
        if not metadata and auth:
            metadata = [("authorization", f"Bearer {auth}")]
        use_tls = bool(str(server_address).startswith("https://")) or bool(
            getattr(getattr(config, "tls", None), "verify", False)
            and str(getattr(config, "url", "")).startswith("https://")
        )

        # Create the underlying gRPC client
        self._client = ProximaDBSyncGrpcClient(
            server_address=server_address,
            timeout=timeout,
            pool_size=pool_size,
            max_message_size=max_message_size,
            metadata=metadata,
            use_tls=use_tls,
        )
        self._server_address = server_address
        self._connected = True

    @property
    def protocol_name(self) -> str:
        """Return the protocol name."""
        return "grpc"

    @property
    def is_connected(self) -> bool:
        """Check if the adapter is connected and operational."""
        return self._connected

    # ==========================================================================
    # Health & Server Operations
    # ==========================================================================

    def health(self) -> HealthStatus:
        """Check server health status."""
        try:
            result = self._client.health_check()

            # Convert HealthCheckResponse to HealthStatus
            if hasattr(result, "healthy"):
                is_healthy = bool(result.healthy)
                return HealthStatus(
                    status="healthy" if is_healthy else "running",
                    version=getattr(result, "version", "0.0.0") or "0.0.0",
                    uptime_seconds=getattr(result, "uptime_seconds", 0) or 0,
                    timestamp_ms=max(0, int(getattr(result, "latency_ms", 0) or 0)),
                    services={"grpc": "ok" if is_healthy else "unavailable"},
                )

            return HealthStatus(
                status="running",
                version="0.0.0",
                uptime_seconds=0,
                timestamp_ms=0,
                services={"grpc": "unknown"},
            )
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthStatus(
                status="running",
                version="0.0.0",
                uptime_seconds=0,
                timestamp_ms=0,
                services={"grpc": "unavailable"},
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
        distance_metric = ProtoConverter.distance_metric_to_int(
            config.distance_metric if config else kwargs.get("distance_metric")
        )
        storage_engine = ProtoConverter.storage_engine_to_int(
            config.storage_engine if config else kwargs.get("storage_engine")
        )
        # Map EmbeddingPrecision (str-Enum) → proto int discriminant.
        # Mirrors proto: Unspecified=0, Fp32=1, Fp16=2, Bf16=3, Int8=4, Uint8=5.
        precision_int: int | None = None
        precision_raw = (
            config.canonical_embedding_precision
            if config is not None
            else kwargs.get("canonical_embedding_precision")
        )
        if precision_raw is not None:
            precision_label = getattr(precision_raw, "value", precision_raw)
            precision_int = {
                "fp32": 1,
                "fp16": 2,
                "bf16": 3,
                "int8": 4,
                "uint8": 5,
            }.get(str(precision_label).lower())

        result = self._client.create_collection(
            name=name,
            dimension=dimension,
            distance_metric=distance_metric,
            storage_engine=storage_engine,
            canonical_embedding_precision=precision_int,
            **{
                k: v
                for k, v in kwargs.items()
                if k
                not in [
                    "dimension",
                    "distance_metric",
                    "storage_engine",
                    "canonical_embedding_precision",
                ]
            },
        )

        # Convert CollectionWrapper or proto to Collection
        return self._to_collection(result, name, dimension)

    def get_collection(self, collection_id: str) -> Collection | None:
        """Get collection metadata by ID or name."""
        try:
            result = self._client.get_collection(collection_id)

            if result is None:
                return None

            return self._to_collection(result, collection_id)
        except Exception as e:
            logger.debug(f"Collection not found: {collection_id} - {e}")
            return None

    def list_collections(self) -> list[Collection]:
        """List all collections."""
        results = self._client.list_collections()

        collections = []
        for item in results or []:
            try:
                collections.append(self._to_collection(item))
            except Exception as e:
                logger.warning(f"Failed to convert collection: {e}")

        return collections

    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection."""
        try:
            result = self._client.delete_collection(collection_id)

            if hasattr(result, "success"):
                return result.success
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

    def _to_collection(
        self, result: Any, fallback_name: str = "", fallback_dimension: int = 0
    ) -> Collection:
        """Convert various result types to Collection model."""
        if isinstance(result, Collection):
            return result

        if isinstance(result, dict):
            # Collection requires `config`; supply a minimal one when the
            # raw dict only carries id/name/dimension (the path taken when
            # gRPC returns the new typed Collection wrapper).
            if "config" not in result:
                cfg_payload = {
                    "name": result.get("name", fallback_name or ""),
                    "dimension": result.get("dimension", fallback_dimension or 0),
                }
                result = {**result, "config": cfg_payload}
            return Collection(**result)

        # Handle CollectionWrapper or protobuf objects
        name = getattr(result, "name", fallback_name)
        dimension = getattr(result, "dimension", fallback_dimension)
        coll_id = getattr(result, "id", name)

        return Collection(
            id=coll_id,
            config=CollectionConfig(name=name or coll_id, dimension=dimension or 0),
        )

    # ==========================================================================
    # Record Operations
    # ==========================================================================

    @staticmethod
    def _record_payloads(records: list[ProximaRecord] | list[dict[str, Any]]):
        payloads = []
        for record in records:
            if isinstance(record, dict):
                payloads.append(record)
            elif hasattr(record, "model_dump"):
                payloads.append(record.model_dump(exclude_none=True))
            else:
                payloads.append(ProtoConverter.vector_record_to_dict(record))
        return payloads

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
        """Insert ProximaRecord-shaped payloads into a collection."""
        payloads = self._record_payloads(records)
        if hasattr(self._client, "insert_records"):
            result = self._client.insert_records(
                collection_id=collection_id, records=payloads, **kwargs
            )
        else:
            result = self._client.insert_vectors(
                collection_id=collection_id, vectors=payloads, **kwargs
            )
        response = self._to_vector_operation_response(result, "INSERT", len(records))
        return BatchResult(
            total=len(records),
            success=response.metrics.successful_count,
            failed=response.metrics.failed_count,
            errors=[response.error_message] if response.error_message else [],
            metrics=response.metrics,
        )

    def upsert_records(
        self,
        collection_id: str,
        records: list[ProximaRecord] | list[dict[str, Any]],
        **kwargs,
    ) -> BatchResult:
        """Upsert ProximaRecord-shaped payloads into a collection."""
        payloads = self._record_payloads(records)
        if hasattr(self._client, "upsert_records"):
            result = self._client.upsert_records(
                collection_id=collection_id, records=payloads, **kwargs
            )
        else:
            result = self._client.insert_vectors(
                collection_id=collection_id,
                vectors=payloads,
                upsert=True,
                **kwargs,
            )
        response = self._to_vector_operation_response(result, "UPSERT", len(records))
        return BatchResult(
            total=len(records),
            success=response.metrics.successful_count,
            failed=response.metrics.failed_count,
            errors=[response.error_message] if response.error_message else [],
            metrics=response.metrics,
        )

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
        if hasattr(self._client, "get_vectors"):
            results = self._client.get_vectors(
                collection_id, vector_ids, include_vectors=include_vectors, **kwargs
            )
        else:
            # Fallback: not implemented in gRPC client
            logger.warning("get_vectors not implemented in gRPC client")
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
                        vector=list(getattr(r, "vector", [])),
                        metadata=dict(getattr(r, "metadata", {})),
                    )
                )

        return records

    def delete_vectors(
        self, collection_id: str, vector_ids: list[str], **kwargs
    ) -> VectorOperationResponse:
        """Delete vectors by IDs."""
        if hasattr(self._client, "delete_vectors"):
            result = self._client.delete_vectors(collection_id, vector_ids, **kwargs)
        else:
            # Fallback: not implemented
            return VectorOperationResponse(
                success=False,
                operation="DELETE",
                error_message="delete_vectors not implemented in gRPC client",
            )

        return self._to_vector_operation_response(result, "DELETE", len(vector_ids))

    def update_vector_metadata(
        self, collection_id: str, vector_id: str, metadata: MetadataDict, **kwargs
    ) -> VectorOperationResponse:
        """Update metadata for a specific vector."""
        if hasattr(self._client, "update_vector_metadata"):
            result = self._client.update_vector_metadata(
                collection_id, vector_id, metadata, **kwargs
            )
            return self._to_vector_operation_response(result, "UPDATE", 1)

        # Fallback: not implemented
        return VectorOperationResponse(
            success=False,
            operation="UPDATE",
            error_message="update_vector_metadata not implemented in gRPC client",
        )

    def _to_vector_operation_response(
        self, result: Any, operation: str, total_count: int
    ) -> VectorOperationResponse:
        """Convert various result types to VectorOperationResponse."""
        if isinstance(result, VectorOperationResponse):
            return result

        if isinstance(result, dict):
            record_errors = result.get("errors") or []
            failed = result.get("failed_count", 0)
            error_message = result.get("error_message")
            if not error_message and (record_errors or failed):
                # TD-SDKERR-1: per-record failures (RECORD_INSERT_FAILED) used
                # to vanish here — the batch came back success=0 failed=1
                # errors=[] with no exception. Surface them in error_message
                # so BatchResult carries them.
                parts = [
                    str(e.get("error_message") or e.get("error_code") or e)
                    for e in record_errors
                ] or [f"{failed} record(s) failed with no per-record detail"]
                error_message = "; ".join(parts)
            return VectorOperationResponse(
                success=result.get("success", True),
                operation=operation,
                metrics=OperationMetrics(
                    # client_v1's gRPC path spells it "success_count" (TD-SDKERR-1
                    # review: reading only "successful_count" reported
                    # successful=total alongside failed=N).
                    successful_count=result.get(
                        "successful_count", result.get("success_count", total_count)
                    ),
                    failed_count=failed,
                    total_count=total_count,
                ),
                error_message=error_message,
            )

        # Handle wrapper objects
        success = getattr(result, "success", True)
        metrics = getattr(result, "metrics", None)

        if metrics:
            record_errors = list(getattr(result, "errors", None) or [])
            failed = getattr(metrics, "failed_count", 0)
            error_message = getattr(result, "error_message", None)
            if not error_message and (record_errors or failed):
                parts = [
                    str(
                        getattr(e, "error_message", None)
                        or getattr(e, "error_code", None)
                        or e
                    )
                    for e in record_errors
                ] or [f"{failed} record(s) failed with no per-record detail"]
                error_message = "; ".join(parts)
            return VectorOperationResponse(
                success=success,
                operation=operation,
                metrics=OperationMetrics(
                    successful_count=getattr(metrics, "successful_count", total_count),
                    failed_count=failed,
                    duration_ms=getattr(metrics, "duration_ms", 0),
                    total_count=total_count,
                ),
                error_message=error_message,
            )

        return VectorOperationResponse(
            success=success,
            operation=operation,
            metrics=OperationMetrics(
                successful_count=total_count if success else 0,
                failed_count=0 if success else total_count,
                total_count=total_count,
            ),
            error_message=getattr(result, "error_message", None),
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
        """Search for similar vectors."""
        # Normalize query vector
        if hasattr(query_vector, "tolist"):
            query_vector = query_vector.tolist()

        results = self._client.search_vectors(
            collection_id=collection_id,
            query_vector=list(query_vector),
            top_k=top_k,
            metadata_filters=filter,
            include_vectors=include_vectors,
            include_metadata=include_metadata,
            **kwargs,
        )

        return self._to_search_results(results, include_vectors, include_metadata)

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

        # Use search_vectors with multiple query vectors
        results = self._client.search_vectors(
            collection_id=collection_id,
            query_vectors=normalized_queries,
            top_k=top_k,
            metadata_filters=filter,
            include_vectors=include_vectors,
            include_metadata=include_metadata,
            **kwargs,
        )

        # Handle single query result vs batch results
        if results and not isinstance(results[0], list):
            # Single query result - wrap in list for batch format
            return [self._to_search_results(results, include_vectors, include_metadata)]

        # Multiple query results
        batch_results = []
        for query_results in results or []:
            batch_results.append(
                self._to_search_results(
                    query_results, include_vectors, include_metadata
                )
            )

        return batch_results

    def _to_search_results(
        self, results: Any, include_vectors: bool, include_metadata: bool
    ) -> list[SearchResult]:
        """Convert various result types to SearchResult list."""
        if results is None:
            return []

        search_results = []
        for r in results:
            if isinstance(r, SearchResult):
                search_results.append(r)
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

        return search_results

    # ==========================================================================
    # Lifecycle Methods
    # ==========================================================================

    def close(self) -> None:
        """Close the gRPC client connection pool."""
        if hasattr(self._client, "close"):
            self._client.close()
        self._connected = False
