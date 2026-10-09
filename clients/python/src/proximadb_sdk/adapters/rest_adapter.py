"""
ProximaDB REST Protocol Adapter

Wraps the REST protocol client to implement the BaseProtocolAdapter interface.
Converts REST responses to standardized Pydantic models.

Copyright 2025 ProximaDB Contributors
Licensed under the Apache License, Version 2.0
"""

import logging
import time
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


class RestProtocolAdapter(BaseProtocolAdapter):
    """REST protocol adapter implementing BaseProtocolAdapter.

    Wraps the existing ProximaDBClient (REST) to provide a consistent
    interface that returns Pydantic models.
    """

    def __init__(
        self,
        url: str = "http://localhost:5678",
        api_key: str | None = None,
        timeout: float = 30.0,
        **kwargs,
    ):
        """Initialize REST protocol adapter.

        Args:
            url: ProximaDB REST server URL
            api_key: Optional API key for authentication
            timeout: Request timeout in seconds
            **kwargs: Additional configuration passed to underlying client
        """
        from ..config import load_config
        from ..protocols.rest_sync import ProximaDBClient

        # Reuse the caller's resolved config when available so auth/custom headers
        # are not dropped by the unified client adapter boundary.
        config = kwargs.pop("config", None) or load_config(
            url=url, api_key=api_key, timeout=timeout, **kwargs
        )

        # Create the underlying REST client
        self._client = ProximaDBClient(config=config)
        self._url = url
        self._connected = True

    @property
    def protocol_name(self) -> str:
        """Return the protocol name."""
        return "rest"

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
            result = self._client.health()
            # Result is already a HealthStatus from the underlying client
            if isinstance(result, HealthStatus):
                return result
            # If it's a dict, convert to HealthStatus
            return HealthStatus(**result) if isinstance(result, dict) else result
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return HealthStatus(
                status="running",
                version="0.0.0",
                uptime_seconds=0,
                timestamp_ms=int(time.time() * 1000),
                services={"rest": "unavailable"},
            )

    # ==========================================================================
    # Collection Operations
    # ==========================================================================

    def create_collection(
        self, name: str, config: CollectionConfig | None = None, **kwargs
    ) -> Collection:
        """Create a new vector collection."""
        result = self._client.create_collection(name=name, config=config, **kwargs)

        # The REST client already returns a Collection model
        if isinstance(result, Collection):
            return result

        # Convert dict to Collection if needed
        if isinstance(result, dict):
            return Collection(**result)

        # Handle wrapper objects
        if hasattr(result, "name") and hasattr(result, "id"):
            return Collection(
                id=getattr(result, "id", ""),
                name=getattr(result, "name", name),
                dimension=getattr(
                    result, "dimension", config.dimension if config else 0
                ),
            )

        return result

    def get_collection(self, collection_id: str) -> Collection | None:
        """Get collection metadata by ID or name."""
        try:
            result = self._client.get_collection(collection_id)

            if result is None:
                return None

            if isinstance(result, Collection):
                return result

            if isinstance(result, dict):
                return Collection(**result)

            # Handle wrapper objects
            if hasattr(result, "name"):
                return Collection(
                    id=getattr(result, "id", collection_id),
                    name=getattr(result, "name", ""),
                    dimension=getattr(result, "dimension", 0),
                )

            return result
        except Exception as e:
            logger.debug(f"Collection not found: {collection_id} - {e}")
            return None

    def list_collections(self) -> list[Collection]:
        """List all collections."""
        try:
            results = self._client.list_collections()
        except Exception as e:
            logger.error(f"Failed to list collections: {e}")
            return []

        collections = []
        for item in results:
            if isinstance(item, Collection):
                collections.append(item)
            elif isinstance(item, dict):
                collections.append(Collection(**item))
            elif hasattr(item, "name"):
                collections.append(
                    Collection(
                        id=getattr(item, "id", ""),
                        name=getattr(item, "name", ""),
                        dimension=getattr(item, "dimension", 0),
                    )
                )

        return collections

    def delete_collection(self, collection_id: str) -> bool:
        """Delete a collection."""
        try:
            result = self._client.delete_collection(collection_id)
            if isinstance(result, bool):
                return result
            if hasattr(result, "success"):
                return result.success
            return True
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            return False

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
    def _to_batch_result(result: Any, total_count: int) -> BatchResult:
        if isinstance(result, BatchResult):
            return result
        if isinstance(result, VectorOperationResponse):
            metrics = result.metrics or OperationMetrics()
            return BatchResult(
                total=total_count,
                success=metrics.successful_count,
                failed=metrics.failed_count,
                errors=[result.error_message] if result.error_message else [],
                metrics=metrics,
            )
        if isinstance(result, dict):
            success = result.get(
                "success",
                result.get(
                    "successful_count", result.get("inserted_count", total_count)
                ),
            )
            failed = result.get("failed", result.get("failed_count", 0))
            return BatchResult(
                total=result.get("total", success + failed),
                success=success,
                failed=failed,
                errors=result.get("errors", []),
                metrics=OperationMetrics(
                    total_processed=result.get("total", success + failed),
                    successful_count=success,
                    failed_count=failed,
                ),
            )
        successful = getattr(result, "success", total_count)
        if isinstance(successful, bool):
            successful = total_count if successful else 0
        failed = getattr(result, "failed", 0)
        return BatchResult(
            total=successful + failed,
            success=successful,
            failed=failed,
            metrics=OperationMetrics(
                total_processed=successful + failed,
                successful_count=successful,
                failed_count=failed,
            ),
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
        """Insert ProximaRecord-shaped payloads into a collection."""
        result = self._client.insert_records(
            collection_id, self._record_payloads(records), **kwargs
        )
        return self._to_batch_result(result, len(records))

    def upsert_records(
        self,
        collection_id: str,
        records: list[ProximaRecord] | list[dict[str, Any]],
        **kwargs,
    ) -> BatchResult:
        """Upsert ProximaRecord-shaped payloads into a collection."""
        if hasattr(self._client, "upsert_records"):
            result = self._client.upsert_records(
                collection_id, self._record_payloads(records), **kwargs
            )
        else:
            result = self._client.insert_records(
                collection_id, self._record_payloads(records), upsert=True, **kwargs
            )
        return self._to_batch_result(result, len(records))

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
            # Fallback: fetch one by one if batch get not available
            results = []
            for vid in vector_ids:
                try:
                    v = self._client.get_vector(collection_id, vid)
                    if v:
                        results.append(v)
                except Exception:
                    pass

        # Convert to VectorRecord list
        records = []
        for r in results:
            if isinstance(r, VectorRecord):
                records.append(r)
            elif isinstance(r, dict):
                records.append(VectorRecord(**r))
            elif hasattr(r, "id"):
                records.append(
                    VectorRecord(
                        id=getattr(r, "id", ""),
                        vector=list(getattr(r, "vector", [])),
                        metadata=getattr(r, "metadata", {}),
                    )
                )

        return records

    def delete_vectors(
        self, collection_id: str, vector_ids: list[str], **kwargs
    ) -> VectorOperationResponse:
        """Delete vectors by IDs."""
        result = self._client.delete_vectors(collection_id, vector_ids, **kwargs)

        if isinstance(result, VectorOperationResponse):
            return result

        return VectorOperationResponse(
            success=(
                getattr(result, "success", True) if hasattr(result, "success") else True
            ),
            operation="DELETE",
            metrics=OperationMetrics(
                successful_count=len(vector_ids),
                failed_count=0,
                total_count=len(vector_ids),
            ),
        )

    def update_vector_metadata(
        self, collection_id: str, vector_id: str, metadata: MetadataDict, **kwargs
    ) -> VectorOperationResponse:
        """Update metadata for a specific vector."""
        if hasattr(self._client, "update_vector_metadata"):
            result = self._client.update_vector_metadata(
                collection_id, vector_id, metadata, **kwargs
            )
        elif hasattr(self._client, "update_metadata"):
            result = self._client.update_metadata(
                collection_id, vector_id, metadata, **kwargs
            )
        else:
            # Fallback: fetch, update, upsert
            vectors = self.get_vectors(collection_id, [vector_id])
            if vectors:
                v = vectors[0]
                updated_meta = {**v.metadata, **metadata} if v.metadata else metadata
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

        if isinstance(result, VectorOperationResponse):
            return result

        return VectorOperationResponse(
            success=True,
            operation="UPDATE",
            metrics=OperationMetrics(successful_count=1, total_count=1),
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

        # Delegate to the REST transport (protocols.rest_sync.ProximaDBClient).
        # Kwarg names must match the transport's `search` signature EXACTLY
        # (`vector` / `metadata_filter`, not `query_vector` / `metadata_filters`);
        # a mismatch raises TypeError, which unified_client.search_single catches
        # and silently converts into a local-only fallback (TD-SDK-1 S2b / ADR-068).
        results = self._client.search(
            collection_id=collection_id,
            vector=query_vector,
            top_k=top_k,
            metadata_filter=filter,
            include_vectors=include_vectors,
            include_metadata=include_metadata,
            **kwargs,
        )

        # Convert to SearchResult list
        search_results = []
        for r in results or []:
            if isinstance(r, SearchResult):
                search_results.append(r)
            elif isinstance(r, dict):
                search_results.append(
                    SearchResult(
                        id=r.get("id", r.get("vector_id", "")),
                        score=r.get("score", r.get("distance", 0.0)),
                        vector=r.get("vector", []) if include_vectors else None,
                        metadata=r.get("metadata", {}) if include_metadata else None,
                    )
                )
            elif hasattr(r, "id"):
                search_results.append(
                    SearchResult(
                        id=getattr(r, "id", ""),
                        score=getattr(r, "score", getattr(r, "distance", 0.0)),
                        vector=(
                            list(getattr(r, "vector", [])) if include_vectors else None
                        ),
                        metadata=(
                            getattr(r, "metadata", {}) if include_metadata else None
                        ),
                    )
                )

        return search_results

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
        """Execute AQL/UQL through the OpenAPI v2 REST query surface."""
        return self._client.execute_query(
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

    def explain_query(
        self,
        query: str,
        *,
        language: str = "uql",
        collection: str | None = None,
    ) -> dict[str, Any]:
        """Explain AQL/UQL through the OpenAPI v2 REST query surface."""
        return self._client.explain_query(
            query, language=language, collection=collection
        )

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

        if hasattr(self._client, "batch_search"):
            results = self._client.batch_search(
                collection_id=collection_id,
                query_vectors=normalized_queries,
                top_k=top_k,
                metadata_filters=filter,
                include_vectors=include_vectors,
                include_metadata=include_metadata,
                **kwargs,
            )
        else:
            # Fallback: execute individual searches
            results = []
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
                results.append(r)
            return results

        # Convert results
        batch_results = []
        for query_results in results or []:
            search_results = []
            for r in query_results or []:
                if isinstance(r, SearchResult):
                    search_results.append(r)
                elif isinstance(r, dict):
                    search_results.append(
                        SearchResult(
                            id=r.get("id", ""),
                            score=r.get("score", 0.0),
                            vector=r.get("vector") if include_vectors else None,
                            metadata=r.get("metadata") if include_metadata else None,
                        )
                    )
            batch_results.append(search_results)

        return batch_results

    # ==========================================================================
    # Document Operations
    # ==========================================================================

    def create_document_collection(
        self, name: str, config: dict[str, Any] | None = None, **kwargs
    ) -> dict[str, Any]:
        """Create a document collection via REST."""
        try:

            response = self._client._session.post(
                f"{self._url}/api/v2/document-collections",
                json={"name": name, **(config or {})},
                timeout=self._client._timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to create document collection: {e}")
            raise

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
        """Ingest documents via the canonical generated client (ADR-041).

        Targets the **collections** vector-store document surface with server-side
        embedding. For the separate JSON **document-collections** surface (no
        embedding) see :meth:`insert_document`.
        """
        return self._client.ingest_documents(
            collection_id,
            records,
            tenant_id=tenant_id,
            ingest_mode=ingest_mode,
            embed_source=embed_source,
            **kwargs,
        )

    def insert_document(
        self,
        collection_name: str,
        document: dict[str, Any],
        id: str | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Insert a JSON document into a document-collection (spec-driven, ADR-041).

        Targets the **document-collections** surface
        (`POST /api/v2/document-collections/{c}/documents`) via the GENERATED client
        op — pure JSON storage, no server-side embedding. For semantic content with
        embedding use :meth:`ingest_documents` (the collections vector store).
        """
        from .._generated.rest.api.documents.insert_document import sync_detailed
        from .._generated.rest.client import Client as GeneratedClient
        from .._generated.rest.models.doc_insert_request import DocInsertRequest

        # Wave-9 typed the previously free-form request body
        # (InsertDocumentBody) into DocInsertRequest {id?, document}.
        body = DocInsertRequest.from_dict({"id": id, "document": document})
        client = GeneratedClient(base_url=self._url, raise_on_unexpected_status=False)
        response = sync_detailed(collection=collection_name, client=client, body=body)
        if response.status_code in (200, 201, 202):
            return response.parsed.to_dict() if response.parsed is not None else {}
        raise RuntimeError(
            f"insert_document (document-collections) failed {response.status_code}: "
            f"{response.content!r}"
        )

    def get_document(
        self,
        collection_name: str,
        doc_id: str,
        projection: list[str] | None = None,
        **kwargs,
    ) -> dict[str, Any] | None:
        """Get a document by ID via REST."""
        try:

            params = {}
            if projection:
                params["projection"] = ",".join(projection)
            response = self._client._session.get(
                f"{self._url}/api/v2/document-collections/{collection_name}/documents/{doc_id}",
                params=params,
                timeout=self._client._timeout,
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.debug(f"Document not found: {doc_id} - {e}")
            return None

    def query_documents(
        self,
        collection_name: str,
        filter: dict[str, Any] | None = None,
        projection: list[str] | None = None,
        limit: int = 100,
        **kwargs,
    ) -> dict[str, Any]:
        """Query documents with filter via REST."""
        try:

            body = {}
            if filter:
                body["filter"] = filter
            if projection:
                body["projection"] = projection
            body["limit"] = limit
            response = self._client._session.post(
                f"{self._url}/api/v2/document-collections/{collection_name}/documents",
                json=body,
                timeout=self._client._timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to query documents: {e}")
            raise

    def update_document(
        self, collection_name: str, doc_id: str, updates: list[dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        """Update a document via REST."""
        try:

            response = self._client._session.put(
                f"{self._url}/api/v2/document-collections/{collection_name}/documents/{doc_id}",
                json={"updates": updates},
                timeout=self._client._timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to update document: {e}")
            raise

    def delete_document(self, collection_name: str, doc_id: str, **kwargs) -> bool:
        """Delete a document via REST."""
        try:

            response = self._client._session.delete(
                f"{self._url}/api/v2/document-collections/{collection_name}/documents/{doc_id}",
                timeout=self._client._timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("deleted", False)
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False

    def list_document_collections(self, **kwargs) -> list[dict[str, Any]]:
        """List all document collections via REST."""
        try:

            response = self._client._session.get(
                f"{self._url}/api/v2/document-collections",
                timeout=self._client._timeout,
            )
            response.raise_for_status()
            return response.json().get("collections", [])
        except Exception as e:
            logger.error(f"Failed to list document collections: {e}")
            return []

    def delete_document_collection(self, collection_name: str, **kwargs) -> bool:
        """Delete a document collection via REST."""
        try:

            response = self._client._session.delete(
                f"{self._url}/api/v2/document-collections/{collection_name}",
                timeout=self._client._timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("success", False)
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
        """Execute hybrid search via REST."""
        try:

            response = self._client._session.post(
                f"{self._url}/api/v2/hybrid/search",
                json={
                    "collection": collection,
                    "text_query": text_query,
                    "query_vector": query_vector,
                    "fusion_strategy": fusion_strategy,
                    "top_k": top_k,
                },
                timeout=self._client._timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")
            raise

    # ==========================================================================
    # Time-Series Operations
    # ==========================================================================

    def create_timeseries_collection(
        self, name: str, config: dict[str, Any] | None = None, **kwargs
    ) -> dict[str, Any]:
        """Create a time-series collection via REST."""
        try:

            response = self._client._session.post(
                f"{self._url}/api/v2/timeseries/collections",
                json={"name": name, **(config or {})},
                timeout=self._client._timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to create time-series collection: {e}")
            raise

    def ingest_timeseries(
        self, collection_name: str, points: list[dict[str, Any]], **kwargs
    ) -> dict[str, Any]:
        """Ingest time-series data points via REST."""
        try:

            response = self._client._session.post(
                f"{self._url}/api/v2/timeseries/collections/{collection_name}/ingest",
                json={"points": points},
                timeout=self._client._timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to ingest time-series data: {e}")
            raise

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
        """Query time-series data with optional aggregation via REST."""
        try:

            body = {
                "start_time": start_time,
                "end_time": end_time,
                "aggregation": aggregation,
            }
            if bucket_ms:
                body["bucket_ms"] = bucket_ms
            if tag_filters:
                body["tag_filters"] = tag_filters
            response = self._client._session.post(
                f"{self._url}/api/v2/timeseries/collections/{collection_name}/query",
                json=body,
                timeout=self._client._timeout,
            )
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to query time-series data: {e}")
            raise

    def list_timeseries_collections(self, **kwargs) -> list[dict[str, Any]]:
        """List all time-series collections via REST."""
        try:

            response = self._client._session.get(
                f"{self._url}/api/v2/timeseries/collections",
                timeout=self._client._timeout,
            )
            response.raise_for_status()
            return response.json().get("collections", [])
        except Exception as e:
            logger.error(f"Failed to list time-series collections: {e}")
            return []

    def delete_timeseries_collection(self, collection_name: str, **kwargs) -> bool:
        """Delete a time-series collection via REST."""
        try:

            response = self._client._session.delete(
                f"{self._url}/api/v2/timeseries/collections/{collection_name}",
                timeout=self._client._timeout,
            )
            response.raise_for_status()
            result = response.json()
            return result.get("success", False)
        except Exception as e:
            logger.error(f"Failed to delete time-series collection: {e}")
            return False

    # ==========================================================================
    # Lifecycle Methods
    # ==========================================================================

    def close(self) -> None:
        """Close the REST client connection."""
        if hasattr(self._client, "close"):
            self._client.close()
        self._connected = False
